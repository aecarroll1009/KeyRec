"""model.py  --  File 3 of the acoustic-keystroke reproduction pipeline.

Two classifiers over a single-channel 64x64 log-mel image, exposed through one
interface:

    build_model(kind, num_classes) -> nn.Module        # input (B, 1, 64, 64)

    kind="cnn"      small 3-block conv net (build/test this FIRST)
    kind="coatnet"  compact CoAtNet: conv stem + MBConv(+SE) + transformer

Design constraints (from the project spec):
  * torch is imported LAZILY -- `import model` is cheap and works even with no
    torch installed; the import only happens the first time you build a model.
    The nn.Module subclasses are therefore defined inside a cached factory
    (you cannot subclass nn.Module before torch exists).
  * The CoAtNet convolutional trunk must reduce a 64x64 input to an 8x8 grid.
    That is asserted in forward() so a wrong input size fails LOUDLY rather
    than silently reshaping garbage into the transformer.
"""

GRID = 8            # CoAtNet conv trunk must produce an 8x8 token grid (64 -> 8)
DROPOUT = 0.3
EMBED_DIM = 128     # transformer / final feature width
N_HEADS = 4
N_TRANSFORMER = 2
MBCONV_EXPAND = 4


# ----------------------------------------------------------------------------
# Lazy torch import + cached class definitions
# ----------------------------------------------------------------------------
_CLASSES = None


def _require_torch():
    """Import torch on demand. Kept out of module top-level so that merely
    importing this file (e.g. for its constants) never forces a torch install."""
    import torch
    from torch import nn
    return torch, nn


def _get_classes():
    """Define and cache the nn.Module subclasses on first use."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    torch, nn = _require_torch()

    # ---- small CNN ------------------------------------------------------
    class SmallCNN(nn.Module):
        """3 x (conv-bn-relu-pool), 1->32->64->128, global avg pool, dropout, linear.

        Each block halves the spatial size: 64 -> 32 -> 16 -> 8. The final
        global average pool makes the head independent of the exact grid, so
        this model is the robust baseline to bring up first.
        """

        def __init__(self, num_classes):
            super().__init__()

            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )

            self.features = nn.Sequential(
                block(1, 32),
                block(32, 64),
                block(64, 128),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)   # global average pool -> (B,128,1,1)
            self.drop = nn.Dropout(DROPOUT)
            self.head = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)           # (B, 128)
            x = self.drop(x)
            return self.head(x)

    # ---- CoAtNet building blocks ---------------------------------------
    class SqueezeExcite(nn.Module):
        """Channel attention: squeeze to a per-channel scalar, gate with sigmoid."""

        def __init__(self, channels, reduction=4):
            super().__init__()
            hidden = max(1, channels // reduction)
            self.fc1 = nn.Conv2d(channels, hidden, 1)
            self.fc2 = nn.Conv2d(hidden, channels, 1)

        def forward(self, x):
            s = x.mean(dim=(2, 3), keepdim=True)  # global squeeze
            s = torch.relu(self.fc1(s))
            s = torch.sigmoid(self.fc2(s))
            return x * s

    class MBConv(nn.Module):
        """Inverted-residual block: 1x1 expand -> depthwise 3x3 -> SE -> 1x1 project.

        A residual connection is used only when it is valid (stride 1 and equal
        in/out channels); the two blocks here both downsample, so they do not.
        """

        def __init__(self, cin, cout, stride, expand=MBCONV_EXPAND):
            super().__init__()
            mid = cin * expand
            self.use_res = (stride == 1 and cin == cout)

            self.expand = nn.Sequential(
                nn.Conv2d(cin, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.GELU(),
            )
            self.dw = nn.Sequential(
                nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.GELU(),
            )
            self.se = SqueezeExcite(mid)
            self.project = nn.Sequential(
                nn.Conv2d(mid, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
            )

        def forward(self, x):
            out = self.expand(x)
            out = self.dw(out)
            out = self.se(out)
            out = self.project(out)
            if self.use_res:
                out = out + x
            return out

    class TransformerBlock(nn.Module):
        """Pre-norm self-attention + MLP, standard residual transformer block."""

        def __init__(self, dim, n_heads, mlp_ratio=2):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Linear(dim * mlp_ratio, dim),
            )

        def forward(self, x):
            h = self.norm1(x)
            attn, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn
            x = x + self.mlp(self.norm2(x))
            return x

    class CoAtNet(nn.Module):
        """Conv stem -> 32, two MBConv(+SE) blocks 64->128, then transformer.

        Spatial trace on a 64x64 input:
            stem  (stride 2): 64 -> 32,  channels 1  -> 32
            MBConv(stride 2): 32 -> 16,  channels 32 -> 64
            MBConv(stride 2): 16 -> 8,   channels 64 -> 128   (8x8 token grid)
        The 8x8 grid is flattened to 64 tokens of width 128, given a learned
        positional embedding, processed by two transformer blocks, mean-pooled,
        and sent through a linear head.
        """

        def __init__(self, num_classes):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
            )
            self.mbconv1 = MBConv(32, 64, stride=2)
            self.mbconv2 = MBConv(64, 128, stride=2)

            self.pos_embed = nn.Parameter(torch.zeros(1, GRID * GRID, EMBED_DIM))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            self.blocks = nn.ModuleList(
                [TransformerBlock(EMBED_DIM, N_HEADS) for _ in range(N_TRANSFORMER)]
            )
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.mbconv1(x)
            x = self.mbconv2(x)

            b, c, h, w = x.shape
            # Fail loudly if the conv trunk did not land on the expected grid --
            # a wrong input size would otherwise be silently reshaped into the
            # transformer and mis-add the positional embedding.
            assert h == GRID and w == GRID, (
                f"CoAtNet expects a {GRID}x{GRID} grid before the transformer "
                f"(got {h}x{w}); input must be (B, 1, 64, 64)"
            )
            assert c == EMBED_DIM, f"expected {EMBED_DIM} channels, got {c}"

            tokens = x.flatten(2).transpose(1, 2)     # (B, 64, 128)
            tokens = tokens + self.pos_embed
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.norm(tokens)
            pooled = tokens.mean(dim=1)               # (B, 128)
            return self.head(pooled)

    _CLASSES = {"cnn": SmallCNN, "coatnet": CoAtNet}
    return _CLASSES


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------
def build_model(kind, num_classes):
    """Construct a classifier. kind in {"cnn", "coatnet"}; input is (B,1,64,64)."""
    classes = _get_classes()
    key = kind.lower()
    if key not in classes:
        raise ValueError(f"unknown model kind {kind!r}; choose from {sorted(classes)}")
    return classes[key](num_classes)


def count_parameters(model):
    """Number of trainable parameters -- handy for a quick sanity print."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke check when run directly.
    torch, _ = _require_torch()
    for kind in ("cnn", "coatnet"):
        m = build_model(kind, num_classes=37)
        y = m(torch.zeros(2, 1, 64, 64))
        print(f"{kind:8s} out={tuple(y.shape)} params={count_parameters(m):,}")
