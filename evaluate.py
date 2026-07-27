"""evaluate.py  --  File 5 of the acoustic-keystroke reproduction pipeline.

Loads a checkpoint written by train.py (File 4), re-evaluates the model on the
SAME held-out split that was stored in the checkpoint, and reports:

  * overall top-1 accuracy,
  * macro-F1 over the classes actually PRESENT in the held-out set,
  * per-key precision / recall / F1,
  * a confusion-matrix PNG (written with a tiny stdlib encoder -- no matplotlib
    required, matching the pure-numpy ethos of Files 1-2),
  * an adjacent-key clustering check: when the model is wrong, does it tend to
    guess a PHYSICALLY neighbouring key? The paper found that it does, so this
    is the sanity signal that the model learned something acoustically real
    (nearby keys sound alike) rather than fitting noise.

CRITICAL guard (the whole reason train.py stores a hash):
    The stored `val_index` is a set of ROW NUMBERS into the dataset array. If
    the dataset .npz was rebuilt since training -- more clips recorded, or even
    just re-scanned in a different order -- those row numbers now point at
    different samples, and evaluating on them would silently report a fake
    accuracy on a train/val-contaminated split. So before trusting val_index we
    recompute `dataset_hash(X, Y)` and REFUSE to proceed unless it matches the
    hash the checkpoint was trained against.

Usage:
    python evaluate.py --data dataset.npz --ckpt checkpoint.pt
    python evaluate.py --data dataset.npz --ckpt checkpoint.pt --cm confusion.png
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
    """The dataset no longer matches the checkpoint it was trained against.

    Its own type so the CLI can print the REFUSED banner for exactly this and
    nothing else. Everything else that can raise RuntimeError in here -- a CUDA
    OOM, a bad device ordinal, a corrupt checkpoint, a state_dict shape
    mismatch -- would otherwise be reported as "your dataset changed", sending
    you to rebuild a pool that was never the problem.

    Subclasses RuntimeError so existing `except RuntimeError` callers and the
    File 5 guard tests keep working unchanged.
    """


# ----------------------------------------------------------------------------
# Physical keyboard layout  --  for the adjacent-key clustering check.
# Coordinates are in key-widths; rows are horizontally staggered like a real
# QWERTY board so that diagonal neighbours land at a realistic distance.
# ----------------------------------------------------------------------------
def _keyboard_coords():
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
    """C[t, p] = # of samples whose true class is t and predicted class is p."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    """Precision, recall, F1 and support per class from a confusion matrix.

    Divisions guard against 0/0 (a class never predicted, or absent from the
    held-out set) by returning 0.0 there rather than a NaN.
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
    """Macro-F1 averaged over PRESENT classes only (support > 0).

    Averaging over absent classes would drag the score toward 0 for reasons
    that have nothing to do with model quality on a small held-out set.
    """
    present = support > 0
    if not np.any(present):
        return 0.0
    return float(f1[present].mean())


# ----------------------------------------------------------------------------
# Adjacent-key clustering check
# ----------------------------------------------------------------------------
def adjacency_analysis(y_true, y_pred, labels, threshold=ADJ_THRESHOLD):
    """Do the model's mistakes land on physically nearby keys?

    Returns a dict with the fraction of ERRORS that fall on an adjacent key,
    alongside the chance baseline (the fraction of all distinct key pairs that
    are adjacent, restricted to the keys present here). A frac_adjacent well
    above baseline reproduces the paper's finding that confusions are dominated
    by neighbouring keys.
    """
    coords = _keyboard_coords()

    def dist(a, b):
        ca, cb = coords.get(labels[a]), coords.get(labels[b])
        if ca is None or cb is None:
            return None
        return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))

    err = np.flatnonzero(y_true != y_pred)
    n_adj = n_scored = 0
    for i in err:
        d = dist(int(y_true[i]), int(y_pred[i]))
        if d is None:
            continue
        n_scored += 1
        if d <= threshold:
            n_adj += 1
    frac_adjacent = (n_adj / n_scored) if n_scored else 0.0

    # Chance baseline: among the present labels that have coordinates, what
    # fraction of ordered distinct pairs are adjacent?
    present = [c for c in np.unique(np.concatenate([y_true, y_pred]))
               if labels[int(c)] in coords]
    pairs = adj_pairs = 0
    for a in present:
        for b in present:
            if a == b:
                continue
            d = dist(int(a), int(b))
            if d is None:
                continue
            pairs += 1
            if d <= threshold:
                adj_pairs += 1
    baseline = (adj_pairs / pairs) if pairs else 0.0

    return {
        "n_errors": int(len(err)),
        "n_scored": int(n_scored),
        "n_adjacent": int(n_adj),
        "frac_adjacent": float(frac_adjacent),
        "baseline": float(baseline),
        "threshold": float(threshold),
    }


# ----------------------------------------------------------------------------
# Confusion-matrix PNG  (row-normalised heatmap, stdlib zlib encoder)
# ----------------------------------------------------------------------------
def _colormap(v):
    """Map v in [0,1] to an RGB uint8 triple via a simple black->red->yellow
    'hot' ramp -- enough to make the diagonal pop without a plotting library."""
    anchors = np.array([[0, 0, 0], [200, 50, 30], [255, 255, 180]], dtype=np.float64)
    pos = np.array([0.0, 0.5, 1.0])
    v = float(np.clip(v, 0.0, 1.0))
    r = np.interp(v, pos, anchors[:, 0])
    g = np.interp(v, pos, anchors[:, 1])
    b = np.interp(v, pos, anchors[:, 2])
    return int(round(r)), int(round(g)), int(round(b))


def _write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a PNG using only stdlib zlib/struct."""
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
    """Render a row-normalised confusion matrix to `path` as a PNG heatmap.

    Rows are normalised by their support (true-class count) so the diagonal is
    comparable across classes even when the held-out set is imbalanced; empty
    rows stay black.
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
    """Run the checkpointed model over X (N,64,64) and return argmax labels."""
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
def evaluate(data_path, ckpt_path, cm_path="confusion.png", device="cpu", verbose=True):
    """Load checkpoint + dataset, verify the hash contract, evaluate on the
    stored held-out split, and return a results dict.

    Raises RuntimeError if the dataset no longer matches the checkpoint's hash
    (the anti-fake-accuracy guard) or if the stored split is inconsistent.
    """
    import torch

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    X, Y, labels = ft.load_dataset(data_path)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.int64)

    # ---- the critical guard: dataset must be byte-identical to training time
    cur_hash = dataset_hash(X, Y)
    if cur_hash != ckpt["data_hash"]:
        raise DatasetContractError(
            "dataset hash mismatch -- the stored val_index no longer refers to "
            "the same samples the model was trained on.\n"
            f"  checkpoint data_hash: {ckpt['data_hash']}\n"
            f"  current   data_hash: {cur_hash}\n"
            "Refusing to evaluate: reusing val_index here would report a FAKE "
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

    Xv, Yv = X[val_idx], Y[val_idx]
    y_pred = predict(Xv, ckpt["model_kind"], ckpt["state_dict"],
                     num_classes=len(labels), device=device)

    n_classes = len(labels)
    cm = confusion_matrix(Yv, y_pred, n_classes)
    precision, recall, f1, support = per_class_prf(cm)
    accuracy = float((y_pred == Yv).mean()) if len(Yv) else 0.0
    macro_f1 = macro_f1_present(f1, support)
    adj = adjacency_analysis(Yv, y_pred, labels)

    cm_written = None
    if cm_path:
        cm_written = save_confusion_png(cm, cm_path)

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_val": int(len(Yv)),
        "labels": labels,
        "confusion": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "adjacency": adj,
        "cm_path": cm_written,
        "model_kind": ckpt["model_kind"],
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_acc": ckpt.get("val_acc"),
    }
    if verbose:
        _print_report(results)
    return results


def _print_report(r):
    print(f"model={r['model_kind']}  held-out samples={r['n_val']}  "
          f"(checkpoint best val_acc={r['ckpt_val_acc']})")
    print(f"accuracy = {r['accuracy']:.4f}   macro-F1 (present classes) = {r['macro_f1']:.4f}")
    print()
    print(f"{'key':>5}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'support':>7}")
    present = r["support"] > 0
    for i, lab in enumerate(r["labels"]):
        if not present[i]:
            continue
        print(f"{lab:>5}  {r['precision'][i]:6.3f}  {r['recall'][i]:6.3f}  "
              f"{r['f1'][i]:6.3f}  {int(r['support'][i]):7d}")
    a = r["adjacency"]
    print()
    print("adjacent-key clustering check (are wrong guesses on physically near keys?):")
    print(f"  errors={a['n_errors']}  scored={a['n_scored']}  "
          f"on-adjacent-key={a['n_adjacent']}")
    print(f"  frac on adjacent key = {a['frac_adjacent']:.3f}   "
          f"chance baseline = {a['baseline']:.3f}   (threshold={a['threshold']} key-widths)")
    verdict = "ABOVE" if a["frac_adjacent"] > a["baseline"] else "not above"
    print(f"  -> errors are {verdict} chance for landing on neighbouring keys "
          "(paper: expect above).")
    if r["cm_path"]:
        print(f"\nconfusion matrix PNG -> {r['cm_path']}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Evaluate an acoustic-keystroke checkpoint on its stored held-out split.")
    p.add_argument("--data", default="dataset.npz", help="dataset .npz (must be the SAME pool the checkpoint was trained on)")
    p.add_argument("--ckpt", default="checkpoint.pt", help="checkpoint .pt from train.py")
    p.add_argument("--cm", default="confusion.png", help="confusion-matrix PNG output path ('' to skip)")
    # Defaults to cpu ON PURPOSE, unlike train.py. CPU and CUDA inference differ
    # by ~1e-4 in the logits, which is enough to flip an argmax on a near-tied
    # sample -- so an `auto` default would make the reported accuracy depend on
    # whether a GPU happened to be visible. Evaluation is a few hundred forward
    # passes; the speed is worth nothing and the stability is worth a lot.
    p.add_argument("--device", default="cpu",
                   help="cpu (default, for a stable published number), auto, cuda, or cuda:N")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for path, what in ((args.data, "dataset"), (args.ckpt, "checkpoint")):
        if not os.path.isfile(path):
            print(f"error: {what} not found: {path}", file=sys.stderr)
            return 2
    # Resolved OUTSIDE the try below: a device problem is not a hash-guard
    # failure, and must not print under the REFUSED banner that means "this
    # dataset no longer matches the checkpoint".
    try:
        device = resolve_device(args.device, verbose=False)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        evaluate(args.data, args.ckpt, cm_path=(args.cm or None), device=device)
    except DatasetContractError as e:
        # Exit 1 + REFUSED means one thing only: this dataset is not the one the
        # checkpoint was trained on.
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        # Everything else -- OOM, bad ordinal, corrupt checkpoint -- is a plain
        # failure and must not be dressed up as the hash guard firing.
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
