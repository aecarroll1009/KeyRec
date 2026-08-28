"""File 3 of the acoustic-keystroke reproduction pipeline.

Defines two classifiers over a single-channel 64x64 log-mel image.
build_model(kind, num_classes) returns an nn.Module taking input (B, 1, 64, 64).
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
    """Import torch on demand.

    Merely importing this module does not require torch.

    Returns:
        A (torch, nn) tuple.
    """
    import torch
    from torch import nn
    return torch, nn


def _make_small_cnn(nn):
    """Build the SmallCNN class.

    Args:
        nn: The torch.nn module.

    Returns:
        The SmallCNN nn.Module subclass.
    """

    class SmallCNN(nn.Module):
        """3 x (conv-bn-relu-pool), 1->32->64->128, global avg pool, dropout, linear.

        Each block halves the spatial size: 64 -> 32 -> 16 -> 8. Global
        average pooling makes the head independent of the exact grid.
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

    return SmallCNN


def _make_squeeze_excite(nn, torch):
    """Build the SqueezeExcite class.

    Args:
        nn: The torch.nn module.
        torch: The torch package.

    Returns:
        The SqueezeExcite nn.Module subclass.
    """

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

    return SqueezeExcite


def _make_mbconv(nn, squeeze_excite):
    """Build the MBConv class.

    Args:
        nn: The torch.nn module.
        squeeze_excite: The SqueezeExcite class, as returned by
            _make_squeeze_excite().

    Returns:
        The MBConv nn.Module subclass.
    """

    class MBConv(nn.Module):
        """Inverted-residual block: 1x1 expand -> depthwise 3x3 -> SE -> 1x1 project.

        A residual connection is used only when stride is 1 and the in/out
        channels match. Both CoAtNet blocks downsample, so neither uses one.
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
            self.se = squeeze_excite(mid)
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

    return MBConv


def _make_transformer_block(nn):
    """Build the TransformerBlock class.

    Args:
        nn: The torch.nn module.

    Returns:
        The TransformerBlock nn.Module subclass.
    """

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

    return TransformerBlock


def _make_coatnet(nn, torch, mbconv, transformer_block):
    """Build the CoAtNet class.

    Args:
        nn: The torch.nn module.
        torch: The torch package.
        mbconv: The MBConv class, as returned by _make_mbconv().
        transformer_block: The TransformerBlock class, as returned by
            _make_transformer_block().

    Returns:
        The CoAtNet nn.Module subclass.
    """

    class CoAtNet(nn.Module):
        """Conv stem -> 32, two MBConv(+SE) blocks 64->128, then transformer.

        Spatial trace on a 64x64 input:
            stem  (stride 2): 64 -> 32,  channels 1  -> 32
            MBConv(stride 2): 32 -> 16,  channels 32 -> 64
            MBConv(stride 2): 16 -> 8,   channels 64 -> 128   (8x8 token grid)
        The 8x8 grid becomes 64 tokens of width 128, plus a learned
        positional embedding. Two transformer blocks process them, then
        mean-pooling and a linear head classify.
        """

        def __init__(self, num_classes):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
            )
            self.mbconv1 = mbconv(32, 64, stride=2)
            self.mbconv2 = mbconv(64, 128, stride=2)

            self.pos_embed = nn.Parameter(torch.zeros(1, GRID * GRID, EMBED_DIM))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            self.blocks = nn.ModuleList(
                [transformer_block(EMBED_DIM, N_HEADS) for _ in range(N_TRANSFORMER)]
            )
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.mbconv1(x)
            x = self.mbconv2(x)

            b, c, h, w = x.shape
            # A wrong input size would otherwise be silently reshaped into
            # the transformer and misapply the positional embedding.
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

    return CoAtNet


def _get_classes():
    """Build and cache the model classes on first use.

    Returns:
        Dict mapping model kind ("cnn", "coatnet") to its nn.Module subclass.
    """
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    torch, nn = _require_torch()

    small_cnn = _make_small_cnn(nn)
    squeeze_excite = _make_squeeze_excite(nn, torch)
    mbconv = _make_mbconv(nn, squeeze_excite)
    transformer_block = _make_transformer_block(nn)
    coatnet = _make_coatnet(nn, torch, mbconv, transformer_block)

    _CLASSES = {"cnn": small_cnn, "coatnet": coatnet}
    return _CLASSES


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------
def build_model(kind, num_classes):
    """Construct a classifier.

    Args:
        kind: Model kind, "cnn" or "coatnet".
        num_classes: Number of output classes.

    Returns:
        An nn.Module accepting input of shape (B, 1, 64, 64).
    """
    classes = _get_classes()
    key = kind.lower()
    if key not in classes:
        raise ValueError(f"unknown model kind {kind!r}; choose from {sorted(classes)}")
    return classes[key](num_classes)


def count_parameters(model):
    """Count trainable parameters.

    Args:
        model: An nn.Module.

    Returns:
        The number of parameters with requires_grad set.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke check when run directly.
    torch, _ = _require_torch()
    for kind in ("cnn", "coatnet"):
        m = build_model(kind, num_classes=37)
        y = m(torch.zeros(2, 1, 64, 64))
        print(f"{kind:8s} out={tuple(y.shape)} params={count_parameters(m):,}")
