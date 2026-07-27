"""Tiny synthetic-data test for train.py.

Proves the core path works before we move on to File 5 (evaluate.py):
  * dataset_hash is deterministic and changes when X or Y changes (this is the
    contract evaluate.py relies on to reject a stale val_index),
  * stratified_split puts every class in BOTH train and val, with no overlap,
  * augment_batch preserves shape, actually changes the data, and its time-shift
    is zero-filled (a contiguous zero edge, not a circular wrap),
  * a short training run on separable synthetic data LEARNS (val accuracy well
    above chance) and saves a checkpoint containing all required keys,
  * the saved checkpoint's data_hash matches the dataset it was trained on.

Run:  python test_train.py
"""

import os
import tempfile

import numpy as np

import train as T

N_CLASSES = 3
PER_CLASS = 12


def make_separable_dataset(seed=0):
    """Each class gets a bright horizontal band in a distinct frequency region,
    on a low-noise floor. Bands are on the FREQUENCY axis, so the time-shift
    augmentation (which moves columns) leaves the class cue intact -- the model
    should learn this quickly."""
    rng = np.random.default_rng(seed)
    imgs, ys = [], []
    band_h = 64 // N_CLASSES
    for cls in range(N_CLASSES):
        r0 = cls * band_h
        for _ in range(PER_CLASS):
            img = rng.uniform(0.0, 0.15, size=(64, 64)).astype(np.float32)
            img[r0:r0 + band_h, :] += 0.8
            imgs.append(np.clip(img, 0.0, 1.0))
            ys.append(cls)
    X = np.stack(imgs).astype(np.float32)
    Y = np.array(ys, dtype=np.int64)
    labels = [f"c{c}" for c in range(N_CLASSES)]
    return X, Y, labels


def test_dataset_hash():
    X, Y, _ = make_separable_dataset(0)
    h1 = T.dataset_hash(X, Y)
    h2 = T.dataset_hash(X.copy(), Y.copy())
    assert h1 == h2, "dataset_hash is not deterministic"

    Xp = X.copy(); Xp[0, 0, 0] += 0.01
    assert T.dataset_hash(Xp, Y) != h1, "hash did not change when X changed"
    Yp = Y.copy(); Yp[0] = (Yp[0] + 1) % N_CLASSES
    assert T.dataset_hash(X, Yp) != h1, "hash did not change when Y changed"
    print("dataset_hash: deterministic + sensitive to X and Y OK")


def test_stratified_split_covers_all_classes():
    X, Y, _ = make_separable_dataset(1)
    tr, va = T.stratified_split(Y, val_frac=0.25, seed=1)
    assert len(set(tr) & set(va)) == 0, "train/val overlap"
    assert len(tr) + len(va) == len(Y), "split does not cover all samples"
    assert set(Y[tr].tolist()) == set(range(N_CLASSES)), "a class missing from train"
    assert set(Y[va].tolist()) == set(range(N_CLASSES)), "a class missing from val"
    print(f"stratified_split: all {N_CLASSES} classes in both sides, no overlap OK "
          f"(train={len(tr)}, val={len(va)})")


def test_augment_batch_shape_and_zero_fill():
    X, Y, _ = make_separable_dataset(2)
    rng = np.random.default_rng(0)
    batch = X[:4].copy()
    out = T.augment_batch(batch, rng)
    assert out.shape == batch.shape, "augment changed batch shape"
    assert out.dtype == np.float32
    assert np.any(out != batch), "augment_batch changed nothing"

    # Directly exercise the spectrogram time-shift and confirm a zero-filled
    # contiguous edge (not a circular wrap) for a forced nonzero shift.
    img = np.ones((64, 64), dtype=np.float32)
    for _ in range(50):
        shifted = T._time_shift_spec(img, rng, max_frac=0.4)
        zero_cols = np.flatnonzero(np.all(shifted == 0.0, axis=0))
        if len(zero_cols) > 0:
            break
    assert len(zero_cols) > 0, "expected a zero-filled column region from a nonzero shift"
    at_start = np.array_equal(zero_cols, np.arange(len(zero_cols)))
    at_end = np.array_equal(zero_cols, np.arange(64 - len(zero_cols), 64))
    assert at_start or at_end, "zero-filled columns are not a contiguous edge (looks circular)"
    print("augment_batch: shape preserved, time-shift zero-filled (not circular) OK")


def test_augment_batch_torch_matches_numpy_semantics():
    """The on-device augmentation is a separate implementation, so it needs the
    same guarantees proven separately -- especially zero-fill-not-circular,
    which is bug #6 from the spec. Runs on CPU tensors so it tests everywhere."""
    import torch

    X, _, _ = make_separable_dataset(4)
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)

    batch = torch.from_numpy(X[:8].copy())
    out = T.augment_batch_torch(batch, gen)
    assert out.shape == batch.shape, "torch augment changed batch shape"
    assert out.dtype == torch.float32
    assert torch.any(out != batch), "torch augment changed nothing"
    assert torch.all(torch.isfinite(out)), "torch augment produced non-finite values"

    # Zero-filled contiguous edge, never a circular wrap.
    ones = torch.ones((16, 64, 64), dtype=torch.float32)
    shifted = T.augment_batch_torch(ones, gen, frac=0.0, n_masks=0)
    found = 0
    for i in range(shifted.shape[0]):
        zero_cols = torch.nonzero(torch.all(shifted[i] == 0.0, dim=0)).flatten()
        if len(zero_cols) == 0:
            continue                      # this image drew shift == 0
        found += 1
        k = len(zero_cols)
        at_start = torch.equal(zero_cols, torch.arange(k))
        at_end = torch.equal(zero_cols, torch.arange(64 - k, 64))
        assert at_start or at_end, "torch time-shift zeros are not a contiguous edge (circular?)"
        assert k <= int(round(0.4 * 64)), "torch time-shift exceeded max_frac"
    assert found > 0, "no image in the batch drew a nonzero shift"

    # Mask fill value. Pinned with max_frac=0.0 so there is no shift: the mean
    # is then exactly the input's own mean, and every cell the augmentation
    # touched must equal it. An all-ones probe would pass for any fill <= 1
    # (including 0.0, or the wrong image's mean), so use varied per-image data.
    probe = torch.rand((6, 64, 64), generator=gen, dtype=torch.float32)
    probe[1] *= 0.2                       # give the images clearly different means
    probe[2] = probe[2] * 0.5 + 0.5
    filled = T.augment_batch_torch(probe, gen, max_frac=0.0)
    changed = filled != probe
    assert bool(changed.any()), "no cell was masked"
    expected = probe.mean(dim=(1, 2), keepdim=True).expand_as(probe)
    assert torch.allclose(filled[changed], expected[changed], atol=1e-6), \
        "masked cells are not filled with their own image's mean"
    assert torch.equal(filled[~changed], probe[~changed]), "unmasked cells were altered"
    print("augment_batch_torch: shape/zero-fill/mask-fill match the numpy path OK")


def test_resolve_device_never_falls_back_silently():
    """auto may resolve to cpu, but an EXPLICIT cuda request on a box without
    working CUDA must raise -- silently training on the CPU for 1100 epochs is
    the failure this guard exists to prevent."""
    import torch

    assert T.resolve_device("cpu", verbose=False) == "cpu"
    auto = T.resolve_device("auto", verbose=False)
    assert auto in ("cpu", "cuda"), f"auto resolved to something odd: {auto!r}"

    cuda_ready = torch.version.cuda is not None and torch.cuda.is_available()
    assert (auto == "cuda") == cuda_ready, "auto disagrees with torch.cuda.is_available()"

    # Typos must be rejected here, not several hundred lines of setup later.
    for bad in ("gpu", "cuda0", "cuda:", "cuda:x", ""):
        try:
            T.resolve_device(bad, verbose=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"resolve_device accepted junk spec {bad!r}")

    if cuda_ready:
        assert T.resolve_device("cuda", verbose=False) == "cuda"
        assert T.resolve_device("cuda:0", verbose=False) == "cuda:0"
        # An out-of-range ordinal must be caught here. Left to torch it
        # surfaces later, from whatever first touches the device.
        n_gpu = torch.cuda.device_count()
        try:
            T.resolve_device(f"cuda:{n_gpu + 5}", verbose=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("resolve_device accepted an out-of-range CUDA ordinal")
        print(f"resolve_device: auto -> cuda, ordinals validated (device_count={n_gpu}), "
              "junk specs rejected OK")
    else:
        try:
            T.resolve_device("cuda", verbose=False)
        except RuntimeError:
            print("resolve_device: auto -> cpu, explicit cuda raises (no silent fallback) OK")
        else:
            raise AssertionError("explicit --device cuda silently fell back to CPU")


def test_same_seed_reproduces_on_cpu():
    """--seed must pin the WHOLE run, not just the split and the augmentation.

    Without torch.manual_seed the weight init and dropout masks come from
    torch's unseeded global RNG, so two runs at the same seed can land on very
    different accuracies -- and no quoted number is reproducible.
    """
    import torch

    X, Y, labels = make_separable_dataset(5)
    runs = []
    for _ in range(2):
        acc, ckpt = T.train_model(X, Y, labels, kind="cnn", epochs=6, batch_size=16,
                                  seed=1234, device="cpu", out_path=None, verbose=False)
        flat = torch.cat([v.flatten() for v in ckpt["state_dict"].values()])
        runs.append((acc, flat))

    assert runs[0][0] == runs[1][0], (
        f"same seed gave different val_acc: {runs[0][0]} vs {runs[1][0]}")
    assert torch.equal(runs[0][1], runs[1][1]), "same seed gave different weights"

    # And a different seed must actually change something, or the above is vacuous.
    _, other = T.train_model(X, Y, labels, kind="cnn", epochs=6, batch_size=16,
                             seed=99, device="cpu", out_path=None, verbose=False)
    other_flat = torch.cat([v.flatten() for v in other["state_dict"].values()])
    assert not torch.equal(runs[0][1], other_flat), "different seeds gave identical weights"
    print("seeding: same seed -> bit-identical CPU run, different seed -> different run OK")


def test_deterministic_flag_makes_cuda_repeatable():
    """--deterministic is the only way to get a bit-repeatable GPU run.

    By default cuDNN uses nondeterministic backward kernels and the autotuner
    may pick different algorithms per run, so two same-seed CUDA runs drift.
    """
    import torch
    if not torch.cuda.is_available():
        print("deterministic: skipped (no CUDA device)")
        return

    X, Y, labels = make_separable_dataset(7)
    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

    def run(det):
        _, ck = T.train_model(X, Y, labels, kind="cnn", epochs=8, batch_size=16,
                              seed=7, device="cuda", out_path=None, verbose=False,
                              deterministic=det)
        return torch.cat([v.flatten() for v in ck["state_dict"].values()]), ck

    a, ck_a = run(True)
    b, _ = run(True)
    assert torch.equal(a, b), (
        "--deterministic did not give a bit-identical CUDA run "
        f"(max|dw|={float((a - b).abs().max()):.3e})")
    assert ck_a["run_env"]["cudnn_deterministic"] is True
    assert ck_a["run_env"]["cudnn_benchmark"] is False, \
        "deterministic must also disable the cudnn autotuner"

    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved
    print("deterministic: same-seed CUDA runs are bit-identical, recorded in run_env OK")


def test_tf32_default_leaves_torch_settings_alone():
    """The default must not silently drop matmul mantissa bits.

    torch ships matmul TF32 OFF; a default of tf32=True would flip it on
    process-wide and quietly reduce precision under a reproduction baseline.
    """
    import torch
    if not torch.cuda.is_available():
        print("tf32: skipped (no CUDA device)")
        return

    X, Y, labels = make_separable_dataset(6)
    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    T.train_model(X, Y, labels, kind="cnn", epochs=2, batch_size=16, seed=0,
                  device="cuda", out_path=None, verbose=False)          # tf32=None
    after = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    assert before == after, f"default run changed TF32 settings: {before} -> {after}"

    # Explicit requests must still take effect, and be recorded truthfully.
    _, ck = T.train_model(X, Y, labels, kind="cnn", epochs=2, batch_size=16, seed=0,
                          device="cuda", out_path=None, verbose=False, tf32=False)
    assert ck["run_env"]["tf32_matmul"] is False, "explicit --no-tf32 not applied/recorded"
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before
    print("tf32: default leaves torch settings untouched, explicit setting honoured OK")


def test_training_learns_and_saves_checkpoint():
    import torch
    X, Y, labels = make_separable_dataset(3)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "checkpoint.pt")
        best_acc, ckpt = T.train_model(
            X, Y, labels, kind="cnn", epochs=40, batch_size=16, lr=5e-4,
            val_frac=0.25, seed=0, out_path=out, verbose=False,
        )
        assert best_acc > 0.6, f"model did not learn separable data (val_acc={best_acc})"

        # Required checkpoint keys (the File 5 contract).
        for key in ("model_kind", "state_dict", "labels", "val_index", "data_hash", "n_samples"):
            assert key in ckpt, f"checkpoint missing required key {key!r}"
        assert ckpt["model_kind"] == "cnn"
        assert ckpt["labels"] == labels
        assert ckpt["n_samples"] == len(Y)
        assert ckpt["data_hash"] == T.dataset_hash(X, Y), "checkpoint hash != dataset hash"

        assert os.path.isfile(out), "checkpoint file was not written"
        reloaded = torch.load(out, weights_only=False)
        assert reloaded["data_hash"] == ckpt["data_hash"], "saved hash mismatch"
        assert set(reloaded["val_index"].tolist()) == set(ckpt["val_index"].tolist())
        print(f"training: learned separable data (val_acc={best_acc:.3f}), "
              f"checkpoint has all required keys + matching hash OK")


def main():
    test_dataset_hash()
    test_stratified_split_covers_all_classes()
    test_augment_batch_shape_and_zero_fill()
    test_augment_batch_torch_matches_numpy_semantics()
    test_resolve_device_never_falls_back_silently()
    test_same_seed_reproduces_on_cpu()
    test_deterministic_flag_makes_cuda_repeatable()
    test_tf32_default_leaves_torch_settings_alone()
    test_training_learns_and_saves_checkpoint()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
