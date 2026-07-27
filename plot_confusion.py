"""plot_confusion.py  --  labelled confusion-matrix figure for a checkpoint.

evaluate.py already writes a confusion PNG, but by design it uses only numpy +
zlib (the spec forbids a plotting dependency on File 5's core path), so that
image is a bare unlabelled heatmap -- fine as a sanity check that the diagonal
is bright, useless as a figure because you cannot tell WHICH keys are confused.

This is the presentation version: same matrix, with key labels on both axes,
counts in the cells, and the off-diagonal errors called out.

Usage:
    python plot_confusion.py --data dataset.npz --ckpt checkpoint.pt \
                             --title SmallCNN --out day_1/confusion_cnn.png
"""

import argparse
import os
import sys

import numpy as np

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e0dfda"


def plot_confusion(cm, labels, title, out_path, accuracy=None, note=None):
    """Row-normalised heatmap with labelled axes and per-cell counts.

    Row-normalised (each row sums to 1) so every key is judged on ITS OWN
    recall regardless of support -- with equal support per class a raw-count
    map would look the same, but the moment the pool grows unevenly across
    sessions raw counts would make the better-represented keys look brighter
    for no reason.

    Sequential single hue, light -> dark, because the value being encoded is a
    magnitude (0..1). Not a rainbow: a multi-hue ramp would imply categories
    that aren't there.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.asarray(cm, dtype=np.float64)
    n = len(labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sums, 1e-12))

    # Single-hue sequential ramp (surface -> blue), matching the categorical
    # slot-1 blue used in the training-curve figure.
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#c7ddf5", "#2a78d6", "#12395f"])

    size = max(7.5, 0.34 * n + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.92), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("predicted key", fontsize=10, color=TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("true key", fontsize=10, color=TEXT_SECONDARY, labelpad=8)

    head = title if accuracy is None else f"{title}   accuracy {accuracy:.1%}"
    ax.set_title(head, fontsize=13, color=TEXT_PRIMARY, pad=14, loc="left")

    # Counts only where something happened; a number in all 729 cells would be
    # noise. Off-diagonal errors get a ring so they are findable at a glance.
    for i in range(n):
        for j in range(n):
            c = int(round(cm[i, j]))
            if c == 0:
                continue
            ink = "#ffffff" if norm[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, str(c), ha="center", va="center", fontsize=7, color=ink)
            if i != j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#eb6834", lw=1.4, zorder=3))

    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=TEXT_SECONDARY)
    for s in ax.spines.values():
        s.set_color(GRID)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of that key's true samples", fontsize=9, color=TEXT_SECONDARY)
    cb.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cb.outline.set_edgecolor(GRID)

    if note:
        fig.text(0.008, 0.008, note, fontsize=8, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None):
    import evaluate as E

    p = argparse.ArgumentParser(description="Labelled confusion matrix for a checkpoint.")
    p.add_argument("--data", default="dataset.npz")
    p.add_argument("--ckpt", default="checkpoint.pt")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default="confusion_labelled.png")
    args = p.parse_args(argv)

    for path, what in ((args.data, "dataset"), (args.ckpt, "checkpoint")):
        if not os.path.isfile(path):
            print(f"error: {what} not found: {path}", file=sys.stderr)
            return 2

    # cm_path=None: the unlabelled PNG is not wanted here, only the numbers.
    res = E.evaluate(args.data, args.ckpt, cm_path=None, device="cpu", verbose=False)
    cm, labels = res["confusion"], res["labels"]
    acc = res.get("accuracy")

    adj = res.get("adjacency") or {}
    note = None
    if adj.get("n_scored"):
        note = (f"{adj['n_errors']} errors; {adj['n_adjacent']} on a physically adjacent key "
                f"({adj['frac_adjacent']:.0%} vs {adj['baseline']:.0%} chance). "
                "Orange rings mark off-diagonal errors.")

    out = plot_confusion(cm, labels, args.title or res.get("model_kind", "model"),
                         args.out, accuracy=acc, note=note)
    print(f"accuracy {acc:.4f}  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
