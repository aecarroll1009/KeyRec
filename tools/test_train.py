"""Synthetic-data tests for train.py.

Covers hashing, splitting, augmentation, device resolution, determinism, and
a short training run that saves a checkpoint.

Run:  python test_train.py
"""

import os
import tempfile

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

import train as T

N_CLASSES = 3
PER_CLASS = 12


def make_separable_dataset(seed=0):
    """Build a synthetic dataset where each class occupies a distinct frequency band.

    The bands sit on the frequency axis. Time-shift augmentation only moves
    columns and leaves them intact.

    Args:
        seed: Seed for the dataset's random generator.

    Returns:
        An (X, Y, labels) tuple: images of shape (N, 64, 64), integer class
        labels, and the class name for each label index.
    """
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


def _flatten_state_dict(ckpt):
    """Flatten a checkpoint's state_dict into one 1-D tensor for weight comparison.

    Args:
        ckpt: A checkpoint dict as returned by train_model, containing state_dict.

    Returns:
        A 1-D torch tensor concatenating every parameter tensor's values.
    """
    import torch
    return torch.cat([v.flatten() for v in ckpt["state_dict"].values()])


def test_dataset_hash():
    """Verify dataset_hash is deterministic and changes whenever X or Y changes."""
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
    """Verify stratified_split puts every class in both train and val with no overlap."""
    X, Y, _ = make_separable_dataset(1)
    tr, va = T.stratified_split(Y, val_frac=0.25, seed=1)
    assert len(set(tr) & set(va)) == 0, "train/val overlap"
    assert len(tr) + len(va) == len(Y), "split does not cover all samples"
    assert set(Y[tr].tolist()) == set(range(N_CLASSES)), "a class missing from train"
    assert set(Y[va].tolist()) == set(range(N_CLASSES)), "a class missing from val"
    print(f"stratified_split: all {N_CLASSES} classes in both sides, no overlap OK "
          f"(train={len(tr)}, val={len(va)})")


def test_augment_batch_shape_and_zero_fill():
    """Verify augment_batch preserves shape, changes the data, and zero-fills its time-shift."""
    X, Y, _ = make_separable_dataset(2)
    rng = np.random.default_rng(0)
    batch = X[:4].copy()
    out = T.augment_batch(batch, rng)
    assert out.shape == batch.shape, "augment changed batch shape"
    assert out.dtype == np.float32
    assert np.any(out != batch), "augment_batch changed nothing"

    # Exercise the spectrogram time-shift directly and confirm a zero-filled
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
    """Verify the torch augmentation matches the numpy path's shape, zero-fill, and mask-fill behavior.

    Runs on CPU tensors.
    """
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
            continue  # this image drew shift == 0
        found += 1
        k = len(zero_cols)
        at_start = torch.equal(zero_cols, torch.arange(k))
        at_end = torch.equal(zero_cols, torch.arange(64 - k, 64))
        assert at_start or at_end, "torch time-shift zeros are not a contiguous edge (circular?)"
        assert k <= int(round(0.4 * 64)), "torch time-shift exceeded max_frac"
    assert found > 0, "no image in the batch drew a nonzero shift"

    # Mask fill value: pinned with max_frac=0.0 so there is no shift, so the
    # mean is exactly the input's own mean and every masked cell must equal
    # it. An all-ones probe would pass for any fill <= 1 (including 0.0, or
    # the wrong image's mean), so use varied per-image data instead.
    probe = torch.rand((6, 64, 64), generator=gen, dtype=torch.float32)
    probe[1] *= 0.2  # give the images clearly different means
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
    """Verify resolve_device never silently substitutes a different device than what was requested.

    An explicit cuda request must raise if CUDA is unavailable, not fall back to cpu.
    """
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
        # An out-of-range ordinal must be caught here, rather than surfacing
        # later from whatever first touches the device.
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
    """Verify --seed pins the entire run, including weight init and dropout.

    Not just the split and augmentation.
    """
    import torch

    X, Y, labels = make_separable_dataset(5)
    runs = []
    for _ in range(2):
        acc, ckpt = T.train_model(X, Y, labels, kind="cnn", epochs=6, batch_size=16,
                                  seed=1234, device="cpu", out_path=None, verbose=False)
        runs.append((acc, _flatten_state_dict(ckpt)))

    assert runs[0][0] == runs[1][0], (
        f"same seed gave different val_acc: {runs[0][0]} vs {runs[1][0]}")
    assert torch.equal(runs[0][1], runs[1][1]), "same seed gave different weights"

    # A different seed must actually change something, or the check above is vacuous.
    _, other = T.train_model(X, Y, labels, kind="cnn", epochs=6, batch_size=16,
                             seed=99, device="cpu", out_path=None, verbose=False)
    other_flat = _flatten_state_dict(other)
    assert not torch.equal(runs[0][1], other_flat), "different seeds gave identical weights"
    print("seeding: same seed -> bit-identical CPU run, different seed -> different run OK")


def test_deterministic_flag_makes_cuda_repeatable():
    """Verify --deterministic makes two same-seed CUDA runs bit-identical.

    Without it, cuDNN's autotuner can pick a different kernel each run.
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
        return _flatten_state_dict(ck), ck

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
    """Verify training does not enable TF32 unless explicitly requested.

    Also verifies the setting is recorded truthfully in run_env.
    """
    import torch
    if not torch.cuda.is_available():
        print("tf32: skipped (no CUDA device)")
        return

    X, Y, labels = make_separable_dataset(6)
    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    T.train_model(X, Y, labels, kind="cnn", epochs=2, batch_size=16, seed=0,
                  device="cuda", out_path=None, verbose=False)  # tf32=None
    after = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    assert before == after, f"default run changed TF32 settings: {before} -> {after}"

    # Explicit requests must still take effect, and be recorded truthfully.
    _, ck = T.train_model(X, Y, labels, kind="cnn", epochs=2, batch_size=16, seed=0,
                          device="cuda", out_path=None, verbose=False, tf32=False)
    assert ck["run_env"]["tf32_matmul"] is False, "explicit --no-tf32 not applied/recorded"
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before
    print("tf32: default leaves torch settings untouched, explicit setting honoured OK")


def test_training_learns_and_saves_checkpoint():
    """Verify a short training run learns separable data and saves a checkpoint with the required keys."""
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
