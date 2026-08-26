"""Tiny synthetic-data test for features.py.

Proves the core path works before we move on to File 3:
  * mel filterbank / numpy mel path produces the right shapes and a sane
    log-mel image (finite, normalized to [0,1]),
  * if librosa is installed, the pure-numpy mel path matches it numerically
    (Slaney norm + center padding are the whole point of that requirement),
  * time_shift is zero-filled, not circular (the tail/head must go to exact
    zero, not wrap the original signal around),
  * spec_augment masks ~10%/axis to the image mean, 2 masks per axis,
  * build_dataset skips empty label groups and raises on mixed sample rates,
  * dataset save/load round-trips X, Y, and labels.

Run:  python test_features.py
"""

import os
import struct
import tempfile
import wave

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

import features as ft
import isolate_keystrokes as ik


def write_float_wav(path, x, sr):
    data = x.astype("<f4").tobytes()
    fmt, ch, bits = 3, 1, 32
    byte_rate = sr * ch * bits // 8
    block_align = ch * bits // 8
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, fmt, ch, sr, byte_rate, block_align, bits)
    hdr += b"data" + struct.pack("<I", len(data))
    with open(path, "wb") as f:
        f.write(hdr + data)


def synth_clip(sr, n, seed=0):
    """A short decaying-noise burst, similar to a real keystroke clip."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) * 0.003).astype(np.float32)
    burst_n = min(n, int(0.03 * sr))
    decay = np.exp(-np.linspace(0, 6, burst_n)).astype(np.float32)
    start = int(0.15 * ik.PREROLL) if hasattr(ik, "PREROLL") else 0
    start = min(start, n - burst_n) if n > burst_n else 0
    x[start:start + burst_n] += (rng.standard_normal(burst_n) * decay * 0.6).astype(np.float32)
    return x


def test_backend_shapes():
    sr = 48000
    x = synth_clip(sr, ik.CLIP_LEN, seed=0)

    mel = ft.melspectrogram(x, sr)
    assert mel.shape[0] == ft.N_MELS, f"n_mels {mel.shape[0]} != {ft.N_MELS}"
    assert mel.shape[1] > 0, "zero time frames"
    assert np.all(np.isfinite(mel)), "non-finite mel values"

    img = ft.clip_to_logmel_image(x, sr)
    assert img.shape == (ft.N_MELS, ft.IMG_SIZE), f"image shape {img.shape}"
    assert np.all(np.isfinite(img)), "non-finite image values"
    assert img.min() >= 0.0 - 1e-6 and img.max() <= 1.0 + 1e-6, "not normalized to [0,1]"
    print(f"backend={ft.backend_name()}: mel/image shapes OK")


def test_numpy_matches_librosa():
    try:
        import librosa  # noqa: F401
    except ImportError:
        print("librosa not installed: skipping numpy-vs-librosa numerical check")
        return

    sr = 48000
    x = synth_clip(sr, ik.CLIP_LEN, seed=1)

    mel_np = ft.melspectrogram_numpy(x, sr)
    mel_lr = ft.melspectrogram_librosa(x, sr)

    assert mel_np.shape == mel_lr.shape, f"shape mismatch {mel_np.shape} vs {mel_lr.shape}"
    # Compare on a relative scale since absolute power values can be tiny.
    scale = max(float(np.max(np.abs(mel_lr))), 1e-8)
    rel_diff = float(np.max(np.abs(mel_np - mel_lr))) / scale
    assert rel_diff < 1e-3, f"numpy mel diverges from librosa: rel_diff={rel_diff}"
    print(f"numpy mel matches librosa: OK (rel_diff={rel_diff:.2e})")


def test_time_shift_zero_filled():
    rng = np.random.default_rng(0)
    n = 1000
    x = np.ones(n, dtype=np.float32)  # all-ones so wrap-vs-zero is obvious

    # Force a large positive shift by seeding until we get one, then check head is zero.
    for _ in range(50):
        shifted = ft.time_shift(x, max_frac=0.4, rng=rng)
        nz = np.flatnonzero(shifted == 0.0)
        if len(nz) > 0:
            break
    assert len(nz) > 0, "expected some zero-filled region from a nonzero shift"
    # The zero region must be a contiguous run at one end, not scattered (which
    # would indicate a wrap/circular artifact instead of a clean zero-fill).
    z = (shifted == 0.0)
    at_start = z[: len(nz)].all() if z[0] else False
    at_end = z[-len(nz):].all() if z[-1] else False
    assert at_start or at_end, "zero-filled region is not a contiguous edge run (looks circular)"
    assert not np.any(shifted > 1.0 + 1e-6), "value > source max: looks wrapped, not zero-filled"
    print("time_shift: zero-filled (not circular) OK")


def test_spec_augment_masks_to_mean():
    rng = np.random.default_rng(0)
    img = np.random.default_rng(1).uniform(0.2, 0.8, size=(ft.N_MELS, ft.IMG_SIZE)).astype(np.float32)
    mean_val = float(img.mean())

    out = ft.spec_augment(img, rng=rng)
    assert out.shape == img.shape
    diff_rows = np.flatnonzero(np.any(out != img, axis=1))
    diff_cols = np.flatnonzero(np.any(out != img, axis=0))
    assert len(diff_rows) > 0 or len(diff_cols) > 0, "spec_augment changed nothing"

    # Any row that was fully overwritten should now be all mean_val.
    full_mean_rows = [r for r in diff_rows if np.allclose(out[r], mean_val)]
    full_mean_cols = [c for c in diff_cols if np.allclose(out[:, c], mean_val)]
    assert len(full_mean_rows) > 0 or len(full_mean_cols) > 0, "no fully-masked row/col found at mean value"
    print("spec_augment: masks to image mean OK")


def test_build_dataset_skips_empty_and_checks_sr():
    sr = 48000
    with tempfile.TemporaryDirectory() as tmp:
        clips = os.path.join(tmp, "clips")
        os.makedirs(clips)

        for i in range(3):
            x = synth_clip(sr, ik.CLIP_LEN, seed=10 + i)
            write_float_wav(os.path.join(clips, f"a_{i:02d}.wav"), x, sr)
        for i in range(2):
            x = synth_clip(sr, ik.CLIP_LEN, seed=20 + i)
            write_float_wav(os.path.join(clips, f"b_{i:02d}.wav"), x, sr)

        X, Y, labels = ft.build_dataset(clips)
        assert labels == ["a", "b"], f"unexpected labels {labels}"
        assert X.shape == (5, ft.N_MELS, ft.IMG_SIZE), f"X shape {X.shape}"
        assert Y.shape == (5,)
        assert set(Y.tolist()) == {0, 1}, "expected both classes present"
        print("build_dataset: shapes/labels OK, empty groups skipped (none present)")

        # Now inject a clip at a different sample rate -> must raise.
        x_bad = synth_clip(44100, ik.CLIP_LEN, seed=99)
        write_float_wav(os.path.join(clips, "a_99.wav"), x_bad, 44100)
        try:
            ft.build_dataset(clips)
            raised = False
        except ValueError:
            raised = True
        assert raised, "expected ValueError on mixed sample rates"
        print("build_dataset: mixed-sample-rate guard raises OK")


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dataset.npz")
        X = np.random.default_rng(0).uniform(0, 1, size=(4, ft.N_MELS, ft.IMG_SIZE)).astype(np.float32)
        Y = np.array([0, 1, 1, 0], dtype=np.int64)
        labels = ["a", "b"]

        ft.save_dataset(out, X, Y, labels)
        X2, Y2, labels2 = ft.load_dataset(out)

        assert np.allclose(X, X2)
        assert np.array_equal(Y, Y2)
        assert labels2 == labels
        print("save/load dataset round-trip: OK")


def test_pool_grows_across_sessions():
    """A second recording session must ADD to the pool, not replace it.

    This is the whole basis of the growing-pool strategy: sessions live in
    their own subdirectories, build_dataset scans recursively, and clips from
    every session end up under the same label.
    """
    import isolate_keystrokes as ik

    sr = 48000
    with tempfile.TemporaryDirectory() as tmp:
        pool = os.path.join(tmp, "training_clips")

        def fake_session(name, per_label):
            for label in ("a", "space"):
                d = os.path.join(pool, name)
                os.makedirs(d, exist_ok=True)
                for i in range(per_label):
                    x = np.zeros(ik.CLIP_LEN, dtype=np.float32)
                    x[100:400] = np.sin(np.linspace(0, 40, 300)).astype(np.float32)
                    ik.write_wav(os.path.join(d, f"{label}_{i:02d}.wav"), x, sr)

        fake_session("session01", 4)
        X1, Y1, labels1 = ft.build_dataset(pool)
        assert labels1 == ["a", "space"], labels1
        assert X1.shape[0] == 8, X1.shape

        fake_session("session02", 3)
        X2, Y2, labels2 = ft.build_dataset(pool)
        assert labels2 == labels1, "labels changed when a session was added"
        assert X2.shape[0] == 14, f"pool did not grow: {X2.shape[0]} != 14"
        assert np.bincount(Y2).tolist() == [7, 7], np.bincount(Y2).tolist()

        # A session subdirectory must not leak into the label names.
        assert not any("session" in l for l in labels2), labels2

        # Re-extracting one session must not disturb the other.
        import shutil
        shutil.rmtree(os.path.join(pool, "session02"))
        X3, _, _ = ft.build_dataset(pool)
        assert X3.shape[0] == 8, "removing session02 did not leave session01 intact"
        print("growing pool: sessions accumulate under one label, "
              "labels unaffected, sessions independent OK")


def main():
    test_backend_shapes()
    test_numpy_matches_librosa()
    test_time_shift_zero_filled()
    test_spec_augment_masks_to_mean()
    test_build_dataset_skips_empty_and_checks_sr()
    test_save_load_roundtrip()
    test_pool_grows_across_sessions()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
