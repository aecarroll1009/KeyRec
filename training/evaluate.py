"""evaluate.py -- File 5 of the acoustic-keystroke reproduction pipeline.

Loads a checkpoint from train.py and re-evaluates it on its own stored held-out split.
Reports accuracy, per-key precision/recall/F1, a confusion-matrix PNG, and an
adjacent-key clustering check. Refuses to run if the dataset no longer matches the hash
the checkpoint was trained against.

Usage:
    python training/evaluate.py --data day_N/dataset.npz --ckpt day_N/checkpoint.pt
    python training/evaluate.py --data day_N/dataset.npz --ckpt day_N/checkpoint.pt --cm confusion.png
"""

import argparse
import os
import struct
import sys
import zlib

import numpy as np

import features as ft
from model import build_model
from train import dataset_hash, resolve_device


class DatasetContractError(RuntimeError):
    """Raised when the dataset no longer matches the checkpoint it was trained against.

    Its own type lets the CLI print the `REFUSED:` banner for exactly this failure.
    Subclasses RuntimeError, so existing `except RuntimeError` callers still work.
    """


# ----------------------------------------------------------------------------
# Physical keyboard layout  --  for the adjacent-key clustering check.
# Coordinates are in key-widths; rows are horizontally staggered like a real
# QWERTY board so that diagonal neighbours land at a realistic distance.
# ----------------------------------------------------------------------------
def _keyboard_coords():
    """Approximate physical (x, y) position of each key, in key-widths.

    Returns:
        Dict mapping key label to an (x, y) tuple.
    """
    coords = {}
    # number row (label '0' sits at the far right, as on the keyboard)
    for i, ch in enumerate("1234567890"):
        coords[ch] = (float(i), 0.0)
    for i, ch in enumerate("qwertyuiop"):
        coords[ch] = (0.5 + i, 1.0)
    for i, ch in enumerate("asdfghjkl"):
        coords[ch] = (0.75 + i, 2.0)
    for i, ch in enumerate("zxcvbnm"):
        coords[ch] = (1.25 + i, 3.0)
    coords["space"] = (4.5, 4.0)
    return coords


# Immediate horizontal neighbours sit 1.0 apart; vertical/diagonal neighbours
# land around 1.0-1.3; the next-key-over is 2.0. 1.4 captures the former only.
ADJ_THRESHOLD = 1.4


# ----------------------------------------------------------------------------
# Metrics  (pure numpy -- no sklearn)
# ----------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes):
    """Build a confusion matrix from true and predicted labels.

    Args:
        y_true: True class indices, int array (N,).
        y_pred: Predicted class indices, int array (N,).
        n_classes: Total number of classes.

    Returns:
        int64 array of shape (n_classes, n_classes) where entry [t, p] is
        the number of samples with true class t predicted as class p.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    """Compute precision, recall, F1, and support for each class.

    Guards against 0/0 by returning 0.0 instead of NaN.

    Args:
        cm: Confusion matrix, shape (n_classes, n_classes), rows = true
            class, columns = predicted class.

    Returns:
        Tuple (precision, recall, f1, support), each a float64 (or int64
        for support) array of length n_classes.
    """
    tp = np.diag(cm).astype(np.float64)
    pred_tot = cm.sum(axis=0).astype(np.float64)   # column sums: predicted-as-c
    true_tot = cm.sum(axis=1).astype(np.float64)   # row sums:    actually-c (support)

    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    recall = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, true_tot.astype(np.int64)


def macro_f1_present(f1, support):
    """Average F1 over classes that actually appear in the held-out set.

    Absent classes are excluded. Including them would drag the score down for
    reasons unrelated to model quality.

    Args:
        f1: Per-class F1 scores, float array (n_classes,).
        support: Per-class true-sample counts, int array (n_classes,).

    Returns:
        The mean F1 over classes with support > 0, or 0.0 if none are
        present.
    """
    present = support > 0
    if not np.any(present):
        return 0.0
    return float(f1[present].mean())


# ----------------------------------------------------------------------------
# Adjacent-key clustering check
# ----------------------------------------------------------------------------
def _keyboard_distance(coords, labels, a, b):
    """Euclidean distance in key-widths between two labels' key positions.

    Args:
        coords: Dict from _keyboard_coords, mapping key label to (x, y).
        labels: Ordered list of class names, indexed by class id.
        a: Class index of the first key.
        b: Class index of the second key.

    Returns:
        Distance in key-widths, or None if either label has no known
        keyboard position.
    """
    ca, cb = coords.get(labels[a]), coords.get(labels[b])
    if ca is None or cb is None:
        return None
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def _adjacent_error_fraction(y_true, y_pred, labels, coords, threshold):
    """Score prediction errors by whether they land on an adjacent key.

    Args:
        y_true: True class indices, int array (N,).
        y_pred: Predicted class indices, int array (N,).
        labels: Ordered list of class names, indexed by class id.
        coords: Dict from _keyboard_coords.
        threshold: Maximum key-width distance counted as "adjacent".

    Returns:
        Tuple (n_errors, n_scored, n_adjacent, frac_adjacent). n_scored excludes
        errors with no known keyboard position. frac_adjacent is n_adjacent /
        n_scored, or 0.0 if n_scored is 0.
    """
    err = np.flatnonzero(y_true != y_pred)
    n_adj = n_scored = 0
    for i in err:
        d = _keyboard_distance(coords, labels, int(y_true[i]), int(y_pred[i]))
        if d is None:
            continue
        n_scored += 1
        if d <= threshold:
            n_adj += 1
    frac_adjacent = (n_adj / n_scored) if n_scored else 0.0
    return len(err), n_scored, n_adj, frac_adjacent


def _adjacent_pair_baseline(y_true, y_pred, labels, coords, threshold):
    """Compute the chance rate of adjacency among the keys present in this eval.

    Args:
        y_true: True class indices, int array (N,).
        y_pred: Predicted class indices, int array (N,).
        labels: Ordered list of class names, indexed by class id.
        coords: Dict from _keyboard_coords.
        threshold: Maximum key-width distance counted as "adjacent".

    Returns:
        Fraction of adjacent pairs among the distinct labels present in
        y_true/y_pred that have a known keyboard position. 0.0 if fewer than
        two such labels are present.
    """
    present = [c for c in np.unique(np.concatenate([y_true, y_pred]))
               if labels[int(c)] in coords]
    pairs = adj_pairs = 0
    for a in present:
        for b in present:
            if a == b:
                continue
            d = _keyboard_distance(coords, labels, int(a), int(b))
            if d is None:
                continue
            pairs += 1
            if d <= threshold:
                adj_pairs += 1
    return (adj_pairs / pairs) if pairs else 0.0


def adjacency_analysis(y_true, y_pred, labels, threshold=ADJ_THRESHOLD):
    """Check whether prediction errors cluster on physically nearby keys.

    A frac_adjacent well above baseline matches the paper's finding.

    Args:
        y_true: True class indices, int array (N,).
        y_pred: Predicted class indices, int array (N,).
        labels: Ordered list of class names, indexed by class id.
        threshold: Maximum key-width distance counted as "adjacent".

    Returns:
        Dict with n_errors, n_scored, n_adjacent, frac_adjacent (the
        fraction of scored errors on an adjacent key), baseline (the chance
        rate of adjacency among all distinct pairs of present keys), and
        threshold.
    """
    coords = _keyboard_coords()
    n_errors, n_scored, n_adjacent, frac_adjacent = _adjacent_error_fraction(
        y_true, y_pred, labels, coords, threshold)
    baseline = _adjacent_pair_baseline(y_true, y_pred, labels, coords, threshold)

    return {
        "n_errors": int(n_errors),
        "n_scored": int(n_scored),
        "n_adjacent": int(n_adjacent),
        "frac_adjacent": float(frac_adjacent),
        "baseline": float(baseline),
        "threshold": float(threshold),
    }


# ----------------------------------------------------------------------------
# Confusion-matrix PNG  (row-normalised heatmap, stdlib zlib encoder)
# ----------------------------------------------------------------------------
def _colormap(v):
    """Map a value in [0, 1] to an RGB heatmap color.

    Uses a black -> red -> yellow ramp.

    Args:
        v: Value to map; clamped to [0, 1].

    Returns:
        (r, g, b) tuple of ints in [0, 255].
    """
    anchors = np.array([[0, 0, 0], [200, 50, 30], [255, 255, 180]], dtype=np.float64)
    pos = np.array([0.0, 0.5, 1.0])
    v = float(np.clip(v, 0.0, 1.0))
    r = np.interp(v, pos, anchors[:, 0])
    g = np.interp(v, pos, anchors[:, 1])
    b = np.interp(v, pos, anchors[:, 2])
    return int(round(r)), int(round(g)), int(round(b))


def _write_png(path, rgb):
    """Write an RGB image array to disk as a PNG.

    Uses only the stdlib zlib and struct modules. No imaging library is required.

    Args:
        path: Output file path.
        rgb: uint8 array of shape (H, W, 3).
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = bytearray()
    for row in range(h):
        raw.append(0)
        raw.extend(rgb[row].tobytes())

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 (RGB)
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def save_confusion_png(cm, path, cell=12):
    """Render a row-normalized confusion matrix as a PNG heatmap.

    Each row is normalized by its support, keeping the diagonal comparable
    across classes even on an imbalanced held-out set. Rows with zero support
    stay black.

    Args:
        cm: Confusion matrix, shape (n_classes, n_classes).
        path: Output PNG path.
        cell: Pixel size of each square cell in the heatmap.

    Returns:
        The path the PNG was written to.
    """
    n = cm.shape[0]
    row_tot = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_tot, out=np.zeros(cm.shape, dtype=np.float64),
                     where=row_tot > 0)

    img = np.zeros((n * cell, n * cell, 3), dtype=np.uint8)
    for t in range(n):
        for p in range(n):
            r, g, b = _colormap(norm[t, p])
            img[t * cell:(t + 1) * cell, p * cell:(p + 1) * cell] = (r, g, b)
    _write_png(path, img)
    return path


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict(X, model_kind, state_dict, num_classes, device="cpu", batch_size=64):
    """Run a checkpointed model over a batch of images and return predicted labels.

    Args:
        X: Images, float array (N, 64, 64).
        model_kind: Architecture name understood by build_model.
        state_dict: Trained model weights.
        num_classes: Number of output classes.
        device: torch device to run inference on.
        batch_size: Number of images per forward pass.

    Returns:
        int64 array (N,) of argmax class predictions.
    """
    import torch

    model = build_model(model_kind, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)
            xb = torch.from_numpy(xb).unsqueeze(1).to(device)   # (b,1,64,64)
            preds[start:start + batch_size] = model(xb).argmax(dim=1).cpu().numpy()
    return preds


# ----------------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------------
def _load_checkpoint(ckpt_path, device):
    """Load a checkpoint dict written by train.py.

    Args:
        ckpt_path: Path to a .pt checkpoint file.
        device: Device string passed to torch.load as map_location.

    Returns:
        The checkpoint dict.
    """
    import torch

    return torch.load(ckpt_path, weights_only=False, map_location=device)


def _check_dataset_contract(ckpt, X, Y, labels):
    """Verify the dataset still matches the one the checkpoint was trained on.

    Args:
        ckpt: Checkpoint dict from _load_checkpoint.
        X: Dataset images, float32 (N, 64, 64).
        Y: Dataset labels, int64 (N,).
        labels: Ordered list of class names for this dataset.

    Returns:
        The checkpoint's val_index, as an int64 array of row numbers into
        X and Y.

    Raises:
        DatasetContractError: If the hash, sample count, label set/order, or
            val_index range no longer matches the checkpoint.
    """
    cur_hash = dataset_hash(X, Y)
    if cur_hash != ckpt["data_hash"]:
        raise DatasetContractError(
            "dataset hash mismatch -- the stored val_index no longer refers to "
            "the same samples the model was trained on.\n"
            f"  checkpoint data_hash: {ckpt['data_hash']}\n"
            f"  current   data_hash: {cur_hash}\n"
            "Refusing to evaluate: reusing val_index here would report a fake "
            "accuracy on a contaminated split. Retrain on the current pool "
            "(train.py) so the checkpoint and dataset agree, then evaluate."
        )
    if ckpt.get("n_samples") not in (None, len(X)):
        raise DatasetContractError(
            f"n_samples mismatch: checkpoint={ckpt['n_samples']} dataset={len(X)}"
        )

    # Labels should match too; the checkpoint's ordering is authoritative for
    # the model's output units.
    ckpt_labels = list(ckpt["labels"])
    if ckpt_labels != list(labels):
        raise DatasetContractError(
            "label set/order differs between checkpoint and dataset; "
            f"checkpoint={ckpt_labels} dataset={list(labels)}"
        )

    val_idx = np.asarray(ckpt["val_index"], dtype=np.int64)
    if val_idx.size and (val_idx.min() < 0 or val_idx.max() >= len(X)):
        raise DatasetContractError("stored val_index is out of range for this dataset")
    return val_idx


def _score_predictions(Yv, y_pred, labels):
    """Compute the confusion matrix and every metric derived from it.

    Args:
        Yv: True labels for the held-out split, int64 (N,).
        y_pred: Predicted labels for the same split, int64 (N,).
        labels: Ordered list of class names.

    Returns:
        Dict with confusion, precision, recall, f1, support, accuracy,
        macro_f1, and adjacency (the dict returned by adjacency_analysis).
    """
    n_classes = len(labels)
    cm = confusion_matrix(Yv, y_pred, n_classes)
    precision, recall, f1, support = per_class_prf(cm)
    accuracy = float((y_pred == Yv).mean()) if len(Yv) else 0.0
    macro_f1 = macro_f1_present(f1, support)
    adjacency = adjacency_analysis(Yv, y_pred, labels)

    return {
        "confusion": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "adjacency": adjacency,
    }


def evaluate(data_path, ckpt_path, cm_path="confusion.png", device="cpu", verbose=True):
    """Evaluate a checkpoint on its own stored held-out split.

    Verifies the dataset still matches the checkpoint's hash before running
    inference. Computes accuracy, per-key precision/recall/F1, and the
    adjacent-key clustering check.

    Args:
        data_path: Path to a dataset .npz written by features.py.
        ckpt_path: Path to a checkpoint .pt written by train.py.
        cm_path: Output path for a confusion-matrix PNG, or a falsy value to
            skip writing one.
        device: torch device string to run inference on.
        verbose: If True, print a formatted report to stdout.

    Returns:
        Results dict with accuracy, macro_f1, n_val, labels, confusion,
        precision, recall, f1, support, adjacency, cm_path, model_kind,
        ckpt_epoch, and ckpt_val_acc.

    Raises:
        DatasetContractError: If the dataset no longer matches the checkpoint.
    """
    ckpt = _load_checkpoint(ckpt_path, device)
    X, Y, labels = ft.load_dataset(data_path)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.int64)

    val_idx = _check_dataset_contract(ckpt, X, Y, labels)
    Xv, Yv = X[val_idx], Y[val_idx]
    y_pred = predict(Xv, ckpt["model_kind"], ckpt["state_dict"],
                     num_classes=len(labels), device=device)

    scored = _score_predictions(Yv, y_pred, labels)
    cm_written = save_confusion_png(scored["confusion"], cm_path) if cm_path else None

    results = {
        "accuracy": scored["accuracy"],
        "macro_f1": scored["macro_f1"],
        "n_val": int(len(Yv)),
        "labels": labels,
        "confusion": scored["confusion"],
        "precision": scored["precision"],
        "recall": scored["recall"],
        "f1": scored["f1"],
        "support": scored["support"],
        "adjacency": scored["adjacency"],
        "cm_path": cm_written,
        "model_kind": ckpt["model_kind"],
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_acc": ckpt.get("val_acc"),
    }
    if verbose:
        _print_report(results)
    return results


def _print_summary_header(r):
    """Print the model/accuracy summary lines of an evaluation report.

    Args:
        r: Results dict as returned by evaluate().
    """
    print(f"model={r['model_kind']}  held-out samples={r['n_val']}  "
          f"(checkpoint best val_acc={r['ckpt_val_acc']})")
    print(f"accuracy = {r['accuracy']:.4f}   macro-F1 (present classes) = {r['macro_f1']:.4f}")


def _print_per_key_table(r):
    """Print the per-key precision/recall/F1/support table of an evaluation report.

    Args:
        r: Results dict as returned by evaluate().
    """
    print()
    print(f"{'key':>5}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'support':>7}")
    present = r["support"] > 0
    for i, lab in enumerate(r["labels"]):
        if not present[i]:
            continue
        print(f"{lab:>5}  {r['precision'][i]:6.3f}  {r['recall'][i]:6.3f}  "
              f"{r['f1'][i]:6.3f}  {int(r['support'][i]):7d}")


def _print_adjacency_report(r):
    """Print the adjacent-key clustering check section of an evaluation report.

    Args:
        r: Results dict as returned by evaluate().
    """
    a = r["adjacency"]
    print()
    print("adjacent-key clustering check (are wrong guesses on physically near keys?):")
    print(f"  errors={a['n_errors']}  scored={a['n_scored']}  "
          f"on-adjacent-key={a['n_adjacent']}")
    print(f"  frac on adjacent key = {a['frac_adjacent']:.3f}   "
          f"chance baseline = {a['baseline']:.3f}   (threshold={a['threshold']} key-widths)")
    verdict = "above" if a["frac_adjacent"] > a["baseline"] else "not above"
    print(f"  -> errors are {verdict} chance for landing on neighbouring keys "
          "(paper: expect above).")
    if r["cm_path"]:
        print(f"\nconfusion matrix PNG -> {r['cm_path']}")


def _print_report(r):
    """Print a formatted evaluation report to stdout.

    Args:
        r: Results dict as returned by evaluate().
    """
    _print_summary_header(r)
    _print_per_key_table(r)
    _print_adjacency_report(r)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    """Build the command-line argument parser for the evaluation CLI.

    Returns:
        Configured argparse.ArgumentParser.
    """
    p = argparse.ArgumentParser(description="Evaluate an acoustic-keystroke checkpoint on its stored held-out split.")
    p.add_argument("--data", default="dataset.npz", help="dataset .npz (must be the same pool the checkpoint was trained on)")
    p.add_argument("--ckpt", default="checkpoint.pt", help="checkpoint .pt from train.py")
    p.add_argument("--cm", default="confusion.png", help="confusion-matrix PNG output path ('' to skip)")
    # cpu is the default here, unlike train.py. CPU and CUDA inference differ
    # by about 1e-4 in the logits, enough to flip an argmax on a near-tied
    # sample. An `auto` default would make accuracy depend on whether a GPU
    # happened to be visible.
    p.add_argument("--device", default="cpu",
                   help="cpu (default, for a stable published number), auto, cuda, or cuda:N")
    return p


def main(argv=None):
    """Run the evaluation CLI.

    Args:
        argv: Argument list to parse; defaults to sys.argv when None.

    Returns:
        Process exit code: 0 on success, 1 if the dataset contract guard
        refused to evaluate, 2 on any other error (missing file, bad
        device, corrupt checkpoint, etc.).
    """
    args = build_parser().parse_args(argv)
    for path, what in ((args.data, "dataset"), (args.ckpt, "checkpoint")):
        if not os.path.isfile(path):
            print(f"error: {what} not found: {path}", file=sys.stderr)
            return 2
    # Resolved outside the try block below. A device problem is not a
    # hash-guard failure, so it must not print under the `REFUSED:` banner.
    try:
        device = resolve_device(args.device, verbose=False)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        evaluate(args.data, args.ckpt, cm_path=(args.cm or None), device=device)
    except DatasetContractError as e:
        # Exit code 1 plus `REFUSED:` means this dataset is not the one the
        # checkpoint was trained on.
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        # Everything else (OOM, bad device ordinal, corrupt checkpoint) is a
        # plain failure, not the hash guard firing. Do not print it as REFUSED.
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
