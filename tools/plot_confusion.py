"""plot_confusion.py  --  labelled confusion-matrix figure for a checkpoint.

evaluate.py already writes an unlabelled confusion PNG, using only numpy and
zlib. This script draws the same matrix with key labels, cell counts, and
off-diagonal errors called out.

Usage:
    python tools/plot_confusion.py --data day_N/dataset.npz --ckpt day_N/checkpoint.pt \
                             --title SmallCNN --out day_N/confusion_cnn.png
"""

import argparse
import os
import sys

import numpy as np

# --- repo layout bootstrap --------------------------------------------------
# Pipeline modules live in training/ and the tools in tools/, so a module in one
# cannot import one from the other by name alone. Put both directories on
# sys.path so every script keeps working when run directly from any cwd.
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in (_os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "tools")):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# ----------------------------------------------------------------------------


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e0dfda"


def _normalize_rows(cm):
    """Row-normalize a confusion matrix so each row sums to 1.

    Row normalization judges each key on its own recall, regardless of
    support. A raw-count map would instead make better-represented keys
    look brighter once the pool grows unevenly across sessions.

    Args:
        cm: 2-D confusion matrix, rows are true class, columns are predicted.

    Returns:
        The row-normalized matrix, same shape as cm.
    """
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.maximum(row_sums, 1e-12))


def _make_colormap():
    """Build the sequential heatmap colormap.

    Sequential single hue, light to dark, since the encoded value is a
    magnitude in [0, 1]. Matches the categorical slot-1 blue used in the
    training-curve figure.

    Returns:
        A LinearSegmentedColormap running from the surface colour to a
        dark blue.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "seq_blue", [SURFACE, "#c7ddf5", "#2a78d6", "#12395f"])


def _draw_heatmap(ax, norm, labels):
    """Draw the heatmap image and the axis tick labels.

    Args:
        ax: Matplotlib axes to draw on.
        norm: Row-normalized confusion matrix.
        labels: Class labels, in matrix order.

    Returns:
        The image artist from ax.imshow(), for building a colorbar.
    """
    n = len(labels)
    im = ax.imshow(norm, cmap=_make_colormap(), vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("predicted key", fontsize=10, color=TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("true key", fontsize=10, color=TEXT_SECONDARY, labelpad=8)
    return im


def _annotate_cells(ax, cm, norm):
    """Write per-cell counts and ring off-diagonal errors.

    Counts are drawn only where a cell is nonzero. Off-diagonal errors get
    a ring so they stand out.

    Args:
        ax: Matplotlib axes to draw on.
        cm: Raw confusion matrix (counts).
        norm: Row-normalized confusion matrix, used to pick legible text
            colour against the cell's fill.
    """
    from matplotlib.patches import Rectangle

    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            c = int(round(cm[i, j]))
            if c == 0:
                continue
            ink = "#ffffff" if norm[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, str(c), ha="center", va="center", fontsize=7, color=ink)
            if i != j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#eb6834", lw=1.4, zorder=3))


def _style_grid(ax, n):
    """Add the cell-separator gridlines and style ticks and spines.

    Args:
        ax: Matplotlib axes to style.
        n: Number of classes; the grid is n by n.
    """
    # 2px surface-colour gap between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=TEXT_SECONDARY)
    for s in ax.spines.values():
        s.set_color(GRID)


def _add_colorbar(fig, ax, im):
    """Add the colorbar and its label.

    Args:
        fig: Matplotlib figure to attach the colorbar to.
        ax: Matplotlib axes the heatmap was drawn on.
        im: Image artist returned by ax.imshow().
    """
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of that key's true samples", fontsize=9, color=TEXT_SECONDARY)
    cb.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cb.outline.set_edgecolor(GRID)


def plot_confusion(cm, labels, title, out_path, accuracy=None, note=None):
    """Draw a row-normalized confusion-matrix heatmap with labelled axes.

    Args:
        cm: 2-D confusion matrix, rows are true class, columns are predicted.
        labels: Class labels, in matrix order.
        title: Figure title; accuracy is appended to it when given.
        out_path: Path to write the PNG to.
        accuracy: Overall accuracy, shown in the title when given.
        note: Optional footnote text.

    Returns:
        The output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    n = len(labels)
    norm = _normalize_rows(cm)

    size = max(7.5, 0.34 * n + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.92), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    im = _draw_heatmap(ax, norm, labels)

    head = title if accuracy is None else f"{title}   accuracy {accuracy:.1%}"
    ax.set_title(head, fontsize=13, color=TEXT_PRIMARY, pad=14, loc="left")

    _annotate_cells(ax, cm, norm)
    _style_grid(ax, n)
    _add_colorbar(fig, ax, im)

    if note:
        fig.text(0.008, 0.008, note, fontsize=8, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None):
    """Parse arguments, evaluate a checkpoint, and write the labelled figure.

    Args:
        argv: Argument list to parse; defaults to sys.argv when None.

    Returns:
        Process exit code (0 on success, 2 if an input file is missing).
    """
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
