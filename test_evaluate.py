"""Tiny synthetic-data test for evaluate.py (File 5).

Proves the evaluation path works end-to-end before we trust it on real clips:
  * the pure-numpy metrics (confusion matrix, per-class P/R/F1, macro-F1 over
    PRESENT classes) match hand-computed values on a tiny known example,
  * macro-F1 ignores absent classes rather than being dragged toward 0,
  * save_confusion_png writes a byte-valid PNG (correct signature + IHDR dims)
    with no matplotlib installed,
  * the adjacent-key clustering check scores physically-near confusions as
    adjacent and reports a sensible chance baseline,
  * a full train->evaluate round-trip on separable synthetic data reproduces
    the checkpoint's own val accuracy on the SAME stored split,
  * THE CRITICAL GUARD: if the dataset is mutated after training, evaluate()
    REFUSES (raises) instead of silently reporting a fake accuracy.

Run:  python test_evaluate.py
"""

import os
import struct
import tempfile

import numpy as np

import evaluate as E
import train as T
from test_train import make_separable_dataset


def test_metrics_match_hand_computed():
    # 3 classes, 6 samples. Confusion we construct by hand:
    #   true=[0,0,1,1,2,2], pred=[0,1,1,1,2,0]
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)
    cm = E.confusion_matrix(y_true, y_pred, 3)
    # rows=true, cols=pred
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]], cm.tolist()

    prec, rec, f1, sup = E.per_class_prf(cm)
    # class 0: tp=1, pred_col=2 -> prec .5 ; support row=2 -> rec .5 ; f1 .5
    assert np.isclose(prec[0], 0.5) and np.isclose(rec[0], 0.5) and np.isclose(f1[0], 0.5)
    # class 1: tp=2, pred_col=3 -> prec 2/3 ; support=2 -> rec 1.0
    assert np.isclose(prec[1], 2 / 3) and np.isclose(rec[1], 1.0)
    assert np.isclose(f1[1], 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))
    # class 2: tp=1, pred_col=1 -> prec 1.0 ; support=2 -> rec .5
    assert np.isclose(prec[2], 1.0) and np.isclose(rec[2], 0.5)
    assert sup.tolist() == [2, 2, 2]

    macro = E.macro_f1_present(f1, sup)
    assert np.isclose(macro, np.mean([f1[0], f1[1], f1[2]]))
    print("metrics: confusion matrix + per-class P/R/F1 + macro-F1 match hand values OK")


def test_macro_f1_ignores_absent_classes():
    # 4-class label space but only classes 0 and 1 appear in the held-out set.
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 1], dtype=np.int64)   # perfect
    cm = E.confusion_matrix(y_true, y_pred, 4)
    _, _, f1, sup = E.per_class_prf(cm)
    macro = E.macro_f1_present(f1, sup)
    assert np.isclose(macro, 1.0), f"absent classes dragged macro-F1 to {macro}"
    assert sup.tolist() == [2, 2, 0, 0]
    print("macro-F1: absent classes excluded (perfect present classes -> 1.0) OK")


def test_confusion_png_is_valid():
    cm = np.array([[5, 1, 0], [0, 4, 2], [1, 0, 6]], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cm.png")
        E.save_confusion_png(cm, path, cell=8)
        assert os.path.isfile(path)
        with open(path, "rb") as f:
            data = f.read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG signature"
        # First chunk after the signature must be IHDR; parse its width/height.
        assert data[12:16] == b"IHDR", "IHDR chunk missing"
        w, h = struct.unpack(">II", data[16:24])
        assert (w, h) == (3 * 8, 3 * 8), f"unexpected PNG dims {(w, h)}"
        # IEND is the final chunk: length(0) + b"IEND" + 4-byte CRC.
        assert data[-12:-8] == struct.pack(">I", 0) and data[-8:-4] == b"IEND", "IEND chunk missing"
    print("confusion PNG: valid signature + IHDR dims + IEND, no matplotlib OK")


def test_adjacency_scores_near_keys():
    labels = [str(d) for d in range(10)] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["space"]
    idx = {lab: i for i, lab in enumerate(labels)}
    # 'a' mistaken as 's' (right neighbour) and 'q' as 'w' (right neighbour):
    # both adjacent. 'a' mistaken as 'p' (far away): not adjacent.
    y_true = np.array([idx["a"], idx["q"], idx["a"]], dtype=np.int64)
    y_pred = np.array([idx["s"], idx["w"], idx["p"]], dtype=np.int64)
    a = E.adjacency_analysis(y_true, y_pred, labels)
    assert a["n_errors"] == 3 and a["n_scored"] == 3
    assert a["n_adjacent"] == 2, a
    assert np.isclose(a["frac_adjacent"], 2 / 3)
    # Baseline is computed over only the PRESENT keys (5 top-left keys here, so
    # it's high); on a real 37-key eval it is far below frac_adjacent.
    assert 0.0 < a["baseline"] < 1.0, f"chance baseline looks wrong: {a['baseline']}"
    print(f"adjacency: near-key confusions scored adjacent (2/3), "
          f"baseline={a['baseline']:.3f} OK")


def test_roundtrip_reproduces_stored_val_acc_and_guard():
    import torch
    X, Y, labels = make_separable_dataset(3)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "dataset.npz")
        ckpt_path = os.path.join(tmp, "checkpoint.pt")
        cm_path = os.path.join(tmp, "confusion.png")

        # Save the dataset EXACTLY as train/evaluate load it, then train on it.
        np.savez(data_path, X=X, Y=Y, labels=np.array(labels))
        best_acc, _ = T.train_model(
            X, Y, labels, kind="cnn", epochs=40, batch_size=16, lr=5e-4,
            val_frac=0.25, seed=0, out_path=ckpt_path, verbose=False,
        )

        res = E.evaluate(data_path, ckpt_path, cm_path=cm_path, verbose=False)
        # Re-evaluating the best checkpoint on the SAME stored split reproduces
        # its recorded accuracy (best checkpoint == final here for separable data).
        assert np.isclose(res["accuracy"], best_acc, atol=1e-6), \
            f"eval acc {res['accuracy']} != stored best {best_acc}"
        assert res["cm_path"] == cm_path and os.path.isfile(cm_path)
        assert res["n_val"] > 0
        print(f"round-trip: eval reproduced stored val_acc={res['accuracy']:.3f} "
              f"on the same split OK")

        # ---- THE GUARD, case 1: mutate a pixel, evaluate must REFUSE ----
        Xbad = X.copy()
        Xbad[0, 0, 0] += 0.5           # one pixel -> different hash
        np.savez(data_path, X=Xbad, Y=Y, labels=np.array(labels))
        try:
            E.evaluate(data_path, ckpt_path, cm_path=None, verbose=False)
        except RuntimeError as e:
            assert "hash" in str(e).lower()
            print("guard: mutated dataset -> evaluate() REFUSED (hash mismatch) OK")
        else:
            raise AssertionError("evaluate() did NOT refuse a mutated dataset -- fake-accuracy guard broken")

        # ---- THE GUARD, case 2: SAME content, rows REORDERED, must REFUSE ----
        # This is the guard's real purpose: a rebuilt pool with identical images
        # but a different row order silently invalidates the stored val_index.
        # The bytes differ, so the hash differs, so evaluate must refuse -- even
        # though every image and label is still present.
        perm = np.random.default_rng(7).permutation(len(Y))
        assert not np.array_equal(perm, np.arange(len(Y))), "permutation was identity"
        np.savez(data_path, X=X[perm], Y=Y[perm], labels=np.array(labels))
        try:
            E.evaluate(data_path, ckpt_path, cm_path=None, verbose=False)
        except RuntimeError as e:
            assert "hash" in str(e).lower()
            print("guard: reordered-but-same-content dataset -> evaluate() REFUSED OK")
        else:
            raise AssertionError("evaluate() did NOT refuse a reordered dataset -- val_index would mis-index")


def test_refused_banner_means_only_the_hash_guard():
    """`REFUSED:` + exit 1 must mean one thing: this dataset is not the one the
    checkpoint was trained on.

    Every other RuntimeError the CLI can hit -- a bad device ordinal, an OOM, a
    corrupt checkpoint -- has to report as a plain error, or the user goes off
    rebuilding a pool that was never the problem.
    """
    import subprocess
    import sys as _sys

    # The guard's own errors are the dedicated type ...
    assert issubclass(E.DatasetContractError, RuntimeError), \
        "DatasetContractError must subclass RuntimeError (existing callers rely on it)"

    # ... and a device failure must NOT come back under the REFUSED banner.
    # A nonexistent ordinal is the cheapest way to force a non-guard failure.
    r = subprocess.run(
        [_sys.executable, "evaluate.py", "--data", "nonexistent_pool.npz",
         "--ckpt", "nonexistent_ckpt.pt", "--device", "cuda:99"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    combined = r.stdout + r.stderr
    assert "REFUSED" not in combined, \
        f"a non-hash failure printed under the REFUSED banner:\n{combined}"
    print("guard: REFUSED banner is reserved for the hash contract, not device/IO errors OK")


def main():
    test_metrics_match_hand_computed()
    test_macro_f1_ignores_absent_classes()
    test_confusion_png_is_valid()
    test_adjacency_scores_near_keys()
    test_roundtrip_reproduces_stored_val_acc_and_guard()
    test_refused_banner_means_only_the_hash_guard()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
