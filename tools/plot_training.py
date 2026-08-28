"""plot_training.py  --  training-curve comparison for the checkpointed models.

Reads the per-epoch `history` block train.py stores in each checkpoint and
draws validation accuracy against epoch for every model given. Plots both
the raw per-epoch trace and a rolling mean, so the gap between them shows
the selection optimism in the reported best val_acc.

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
    """Compute a centred rolling mean, shrinking the window at the edges.

    Shrinking rather than padding avoids introducing edge artefacts or a
    phantom flat run at either end of the series.

    Args:
        y: 1-D sequence of values.
        window: Full window width, in samples.

    Returns:
        Array of the same length as y.
    """
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
    """Load the per-epoch training history from a checkpoint.

    Args:
        path: Path to a checkpoint saved by train.py.

    Returns:
        Dict with epoch, val_acc, and loss arrays, plus the best val_acc,
        the epoch it was reached at, the number of classes, and the model
        kind.

    Raises:
        SystemExit: If the checkpoint has no 'history' block.
    """
    import torch
    ck = torch.load(path, weights_only=False, map_location="cpu")
    if "history" not in ck:
        raise SystemExit(f"{path} has no 'history' block; re-run training to regenerate it.")
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


def _draw_chance_line(ax, n_classes):
    """Draw the dashed chance-accuracy reference line and its label.

    Args:
        ax: Matplotlib axes to draw on.
        n_classes: Number of classes, used to compute chance accuracy.
    """
    chance = 1.0 / n_classes
    ax.axhline(chance, color=TEXT_SECONDARY, lw=1.0, ls=(0, (4, 4)), alpha=0.7, zorder=1)
    ax.annotate(f"chance  {chance:.1%}", xy=(0.004, chance), xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=9, color=TEXT_SECONDARY)


def _draw_series(ax, series, smooth):
    """Draw the raw trace, rolling mean, peak marker, and label for each series.

    Args:
        ax: Matplotlib axes to draw on.
        series: List of (name, history) pairs, where history is a dict as
            returned by load_history().
        smooth: Rolling-mean window, in epochs.
    """
    for i, (name, h) in enumerate(series):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        mean = rolling_mean(h["val_acc"], smooth)
        ax.plot(h["epoch"], h["val_acc"], color=c, lw=0.7, alpha=0.22, zorder=2)
        ax.plot(h["epoch"], mean, color=c, lw=2.0, zorder=3,
                label=f"{name} (best {h['best']:.1%})")
        # Peak marker: where the saved checkpoint actually came from.
        ax.plot([h["best_epoch"]], [h["best"]], marker="o", ms=8, color=c,
                mec=SURFACE, mew=2.0, zorder=5)
        # Direct label at the right edge -- identity without reading the legend.
        ax.annotate(f" {name}", xy=(h["epoch"][-1], mean[-1]),
                    va="center", ha="left", fontsize=10, color=c, fontweight="bold",
                    annotation_clip=False)


def _style_axes(ax, series, n_classes):
    """Apply axis labels, title, limits, and gridline/spine styling.

    Args:
        ax: Matplotlib axes to style.
        series: List of (name, history) pairs, used to compute the x-limit.
        n_classes: Number of classes, shown in the title.
    """
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


def _add_legend_and_footnote(fig, ax):
    """Add the legend and the explanatory footnote below the axes.

    Args:
        fig: Matplotlib figure to add the footnote to.
        ax: Matplotlib axes to add the legend to.
    """
    # Lifted off the baseline so the legend box never sits on the chance line.
    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.09),
                    frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)

    fig.text(0.008, 0.012,
             "faint line = per-epoch value, solid = rolling mean, dot = saved checkpoint.  "
             "The dot sits above the mean because it is the max of a noisy series.",
             fontsize=8, color=TEXT_SECONDARY)


def plot(series, out_path, smooth=25):
    """Draw validation accuracy vs epoch for one or more checkpoints.

    Args:
        series: List of (name, history) pairs, where history is a dict as
            returned by load_history().
        out_path: Path to write the PNG to.
        smooth: Rolling-mean window, in epochs.

    Returns:
        The output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    n_classes = series[0][1]["n_classes"]
    _draw_chance_line(ax, n_classes)
    _draw_series(ax, series, smooth)
    _style_axes(ax, series, n_classes)
    _add_legend_and_footnote(fig, ax)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _split_spec(spec):
    """Split 'path' or 'path:Display Name' into (path, name).

    Splits on the last colon, but only past index 1. A colon at index 1 is
    a Windows drive letter, not a name separator.

    Args:
        spec: The --ckpt argument value.

    Returns:
        (path, name) tuple; name is "" when no display name was given.
    """
    idx = spec.rfind(":")
    if idx > 1:
        return spec[:idx], spec[idx + 1:]
    return spec, ""


def main(argv=None):
    """Parse arguments, plot the requested checkpoints, and print a summary.

    Args:
        argv: Argument list to parse; defaults to sys.argv when None.

    Returns:
        Process exit code (0 on success, 2 if a checkpoint is missing).
    """
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
