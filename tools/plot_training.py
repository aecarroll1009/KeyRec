"""plot_training.py  --  training-curve comparison for the checkpointed models.

Reads the per-epoch `history` block that train.py stores in each checkpoint and
draws validation accuracy against epoch for every model given.

The curve is drawn twice per model: the raw per-epoch trace at low opacity and a
rolling mean on top. That is deliberate, not decoration -- the raw trace is how
you SEE that the headline "best val_acc" is the maximum of a noisy series rather
than a level the model reached and held. The gap between each peak marker and
its own rolling mean is the selection optimism in that number.

Usage:
    python tools/plot_training.py --ckpt day_N/checkpoint.pt:SmallCNN \
                            --ckpt day_N/checkpoint_coatnet.pt:CoAtNet \
                            --out day_N/training_curves.png
"""

import argparse
import os
import sys

import numpy as np

# Categorical slots 1 and 2 of the validated default palette, used unmodified
# and in fixed order. Blue/orange is the canonical colour-vision-deficiency-safe
# pair; series identity is also carried by a direct label, never colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e0dfda"


def rolling_mean(y, window):
    """Centred rolling mean, shrinking the window at the edges (no padding
    artefacts, no phantom flat run at either end)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return y
    out = np.empty(n)
    half = max(1, window // 2)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def load_history(path):
    import torch
    ck = torch.load(path, weights_only=False, map_location="cpu")
    if "history" not in ck:
        raise SystemExit(
            f"{path} has no 'history' block -- it was trained before train.py "
            "recorded per-epoch history. Re-run training to regenerate it."
        )
    h = ck["history"]
    return {
        "epoch": np.asarray(h["epoch"]),
        "val_acc": np.asarray(h["val_acc"], dtype=np.float64),
        "loss": np.asarray(h["loss"], dtype=np.float64),
        "best": float(ck["val_acc"]),
        "best_epoch": int(ck["epoch"]),
        "n_classes": len(ck["labels"]),
        "kind": ck["model_kind"],
    }


def plot(series, out_path, smooth=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    n_classes = series[0][1]["n_classes"]
    chance = 1.0 / n_classes
    ax.axhline(chance, color=TEXT_SECONDARY, lw=1.0, ls=(0, (4, 4)), alpha=0.7, zorder=1)
    ax.annotate(f"chance  {chance:.1%}", xy=(0.004, chance), xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=9, color=TEXT_SECONDARY)

    for i, (name, h) in enumerate(series):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(h["epoch"], h["val_acc"], color=c, lw=0.7, alpha=0.22, zorder=2)
        ax.plot(h["epoch"], rolling_mean(h["val_acc"], smooth), color=c, lw=2.0,
                zorder=3, label=f"{name} (best {h['best']:.1%})")
        # Peak marker: where the saved checkpoint actually came from.
        ax.plot([h["best_epoch"]], [h["best"]], marker="o", ms=8, color=c,
                mec=SURFACE, mew=2.0, zorder=5)
        # Direct label at the right edge -- identity without reading the legend.
        ax.annotate(f" {name}", xy=(h["epoch"][-1], rolling_mean(h["val_acc"], smooth)[-1]),
                    va="center", ha="left", fontsize=10, color=c, fontweight="bold",
                    annotation_clip=False)

    ax.set_xlabel("epoch", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("validation accuracy", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(f"Validation accuracy vs epoch  ({n_classes} classes)",
                 fontsize=13, color=TEXT_PRIMARY, pad=14, loc="left")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, max(h["epoch"][-1] for _, h in series) * 1.10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    # Lifted off the baseline so the box never sits on the chance line.
    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.09),
                    frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)

    fig.text(0.008, 0.012,
             "faint line = per-epoch value, solid = rolling mean, dot = saved checkpoint.  "
             "The dot sits above the mean because it is the max of a noisy series.",
             fontsize=8, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _split_spec(spec):
    """Split 'path' or 'path:Display Name' into (path, name).

    Splits on the LAST colon, but only when it is past index 1 -- a colon at
    index 1 is a Windows drive letter (C:\\...), not a name separator.
    """
    idx = spec.rfind(":")
    if idx > 1:
        return spec[:idx], spec[idx + 1:]
    return spec, ""


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot validation accuracy vs epoch from checkpoints.")
    p.add_argument("--ckpt", action="append", required=True,
                   help="checkpoint path, optionally 'path:Display Name'. Repeatable.")
    p.add_argument("--out", default="training_curves.png")
    p.add_argument("--smooth", type=int, default=25, help="rolling-mean window (epochs)")
    args = p.parse_args(argv)

    series = []
    for spec in args.ckpt:
        path, name = _split_spec(spec)
        if not os.path.isfile(path):
            print(f"error: checkpoint not found: {path}", file=sys.stderr)
            return 2
        h = load_history(path)
        series.append((name or h["kind"], h))

    out = plot(series, args.out, smooth=args.smooth)
    for name, h in series:
        tail = h["val_acc"][-100:]
        print(f"{name:>10}: best {h['best']:.4f} @ epoch {h['best_epoch']}   "
              f"last-100 mean {tail.mean():.4f}   "
              f"selection gap {h['best'] - tail.mean():+.4f}")
    print(f"plot -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
