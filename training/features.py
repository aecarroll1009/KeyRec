"""features.py  --  File 2 of the acoustic-keystroke reproduction pipeline.

Reads clips/<label>_NN.wav (written by isolate_keystrokes.py) and turns each
clip into a 64x64 log-mel spectrogram image. Saves X (N,64,64) float32 in
[0,1], Y (N,) int labels, and the label name list to a single .npz.

Design constraints (from the project spec):
  * numpy required; librosa OPTIONAL. The mel backend (librosa vs. a pure-numpy
    fallback) is picked ONCE at import time, not per clip.
  * The numpy fallback MUST numerically match librosa's mel filterbank:
      - Slaney filterbank normalization: scale each triangle by 2/(hz_hi-hz_lo)
      - center=True framing: reflect-pad n_fft/2 samples on each side before
        framing (this is what librosa does by default).
  * BUG GUARD: mixing sample rates across clips would poison the mel frequency
    axis, so building a dataset asserts a single, consistent sample rate.
  * BUG GUARD: label folders/groups with zero clips are skipped, not treated
    as an error.
  * Augmentation (both zero-filled, not circular):
      - time_shift: shift the waveform +/-40% of its length, zero-filling the
        exposed region (a circular roll would wrap the attack transient
        around to the other end and corrupt the label's signature).
      - spec_augment: ~10% of the frequency axis and ~10% of the time axis
        masked to the image's own mean value, 2 masks per axis.
"""

import argparse
import glob
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


# ----------------------------------------------------------------------------
# Constants (paper defaults, with the ThinkPad hop-length note from the spec)
# ----------------------------------------------------------------------------
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 500          # 225 in the paper (MacBook); wider here for the
                           # ThinkPad's longer keystroke transients
IMG_SIZE = 64             # output image is N_MELS x IMG_SIZE (time frames)
TOP_DB = 80.0             # power_to_db floor below the max
FMIN = 0.0

TIME_SHIFT_FRAC = 0.4     # +/- 40% of clip length
SPECAUG_MASKS_PER_AXIS = 2
SPECAUG_FRAC = 0.10       # ~10% of the axis length per mask


# ----------------------------------------------------------------------------
# Backend selection -- decided once at import time
# ----------------------------------------------------------------------------
try:
    import librosa
    _BACKEND = "librosa"
except ImportError:
    librosa = None
    _BACKEND = "numpy"


def backend_name():
    return _BACKEND


# ----------------------------------------------------------------------------
# Pure-numpy mel path (must match librosa numerically)
# ----------------------------------------------------------------------------
# librosa's default (htk=False) mel scale is Slaney's: linear below 1 kHz,
# logarithmic above. This is a different curve from the HTK 2595*log10(1+f/700)
# formula -- using HTK here would make the numpy path diverge from librosa
# even with a correctly Slaney-normalized filterbank.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    mel = hz / _F_SP
    log_region = hz >= _MIN_LOG_HZ
    mel = np.where(log_region, _MIN_LOG_MEL + np.log(np.maximum(hz, 1e-12) / _MIN_LOG_HZ) / _LOGSTEP, mel)
    return mel


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    hz = mel * _F_SP
    log_region = mel >= _MIN_LOG_MEL
    hz = np.where(log_region, _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel - _MIN_LOG_MEL)), hz)
    return hz


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Slaney-normalized triangular mel filterbank, matching librosa's default.

    Each triangle is scaled by 2/(hz_hi - hz_lo) so that equal-energy input
    produces (approximately) equal-energy output regardless of band width.
    """
    if fmax is None:
        fmax = sr / 2.0

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, center, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (fft_freqs - lo) / max(center - lo, 1e-12)
        falling = (hi - fft_freqs) / max(hi - center, 1e-12)
        tri = np.maximum(0.0, np.minimum(rising, falling))

        enorm = 2.0 / max(hi - lo, 1e-12)  # Slaney normalization
        fb[m] = tri * enorm

    return fb.astype(np.float32)


def _frame_signal_centered(x, n_fft, hop_length):
    """Reflect-pad n_fft/2 each side, then frame -- matches librosa center=True."""
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((0, n_fft), dtype=xp.dtype)
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return xp[idx]


def _periodic_hann(n_fft):
    """Periodic ("fftbins=True") Hann window -- matches scipy/librosa's default,
    NOT np.hanning (which is symmetric, denominator n_fft-1 instead of n_fft)."""
    n = np.arange(n_fft)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)


def melspectrogram_numpy(x, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """Power mel-spectrogram, shape (n_mels, n_frames). Pure-numpy fallback."""
    frames = _frame_signal_centered(x.astype(np.float64), n_fft, hop_length)
    if frames.shape[0] == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)

    window = _periodic_hann(n_fft)
    spec = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float64)  # (n_frames, n_bins)

    fb = mel_filterbank(sr, n_fft, n_mels).astype(np.float64)  # (n_mels, n_bins)
    mel = fb @ power.T  # (n_mels, n_frames)
    return mel.astype(np.float32)


def melspectrogram_librosa(x, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    # pad_mode is pinned to "reflect" explicitly: librosa's own default changed
    # across versions ("reflect" historically, "constant"/zero in 0.11+), and
    # the numpy fallback below always reflect-pads, so leaving this on the
    # library default would silently desync the two paths on a librosa upgrade.
    return librosa.feature.melspectrogram(
        y=x.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=FMIN, power=2.0, center=True, htk=False, norm="slaney",
        pad_mode="reflect",
    ).astype(np.float32)


def melspectrogram(x, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    if _BACKEND == "librosa":
        return melspectrogram_librosa(x, sr, n_fft, hop_length, n_mels)
    return melspectrogram_numpy(x, sr, n_fft, hop_length, n_mels)


def power_to_db(mel, top_db=TOP_DB):
    """librosa.power_to_db(ref=max) reimplemented: 10*log10(mel/max), floored."""
    mel = np.maximum(mel, 1e-12)
    ref = float(np.max(mel)) if mel.size else 1e-12
    ref = max(ref, 1e-12)
    db = 10.0 * np.log10(mel / ref)
    return np.maximum(db, -top_db)


def resize_time_axis(img, target_frames):
    """Resample the time (column) axis to exactly `target_frames` via linear interp."""
    n_mels, n_frames = img.shape
    if n_frames == target_frames:
        return img
    if n_frames == 0:
        return np.zeros((n_mels, target_frames), dtype=img.dtype)
    src = np.linspace(0.0, 1.0, n_frames)
    dst = np.linspace(0.0, 1.0, target_frames)
    out = np.empty((n_mels, target_frames), dtype=np.float32)
    for i in range(n_mels):
        out[i] = np.interp(dst, src, img[i])
    return out


def normalize_01(img):
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-9:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def clip_to_logmel_image(x, sr):
    """Full per-clip pipeline: waveform -> 64x64 normalized log-mel image."""
    mel = melspectrogram(x, sr)
    db = power_to_db(mel)
    img = resize_time_axis(db, IMG_SIZE)
    return normalize_01(img)


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def time_shift(x, max_frac=TIME_SHIFT_FRAC, rng=None):
    """Zero-filled time shift of a 1-D waveform by up to +/-max_frac of its length.

    A positive shift delays the signal (zeros at the start); negative shifts it
    earlier (zeros at the end). Never wraps -- a circular roll would smear the
    attack transient onto the opposite end and corrupt the label's signature.
    """
    rng = rng or np.random.default_rng()
    n = len(x)
    max_shift = int(round(max_frac * n))
    if max_shift == 0:
        return x.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))

    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[: n - shift]
    elif shift < 0:
        out[: n + shift] = x[-shift:]
    else:
        out[:] = x
    return out


def spec_augment(img, n_masks=SPECAUG_MASKS_PER_AXIS, frac=SPECAUG_FRAC, rng=None):
    """Mask ~frac of the freq axis and ~frac of the time axis, n_masks times each,
    filling masked regions with the image's own mean value."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    n_mels, n_frames = out.shape
    mean_val = float(out.mean())

    freq_w = max(1, int(round(frac * n_mels)))
    for _ in range(n_masks):
        f0 = int(rng.integers(0, max(1, n_mels - freq_w + 1)))
        out[f0:f0 + freq_w, :] = mean_val

    time_w = max(1, int(round(frac * n_frames)))
    for _ in range(n_masks):
        t0 = int(rng.integers(0, max(1, n_frames - time_w + 1)))
        out[:, t0:t0 + time_w] = mean_val

    return out


# ----------------------------------------------------------------------------
# Dataset building
# ----------------------------------------------------------------------------
def _read_clip_wav(path):
    """Read a clip WAV via isolate_keystrokes' own parser (handles float too)."""
    import isolate_keystrokes as ik
    return ik.read_wav(path)


def build_dataset(clips_dir):
    """Scan clips_dir for <label>_NN.wav, at any depth. Return (X, Y, labels).

    X: float32 (N, 64, 64) in [0,1]. Y: int64 (N,) indices into `labels`.
    Empty label groups are skipped. Asserts a single consistent sample rate
    across every clip (mixing sample rates poisons the mel frequency axis).

    The scan is RECURSIVE so the pool can grow by session:

        training_clips/session01/a_00.wav
        training_clips/session02/a_00.wav

    Both are label 'a' and both land in the pool -- the label comes from the
    file name, the directory only records provenance. This is what lets a new
    recording session ADD to the pool instead of replacing it, and it keeps a
    flat clips_dir/a_00.wav layout working unchanged.
    """
    paths = sorted(glob.glob(os.path.join(clips_dir, "**", "*.wav"), recursive=True))

    groups = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label = base.rsplit("_", 1)[0] if "_" in base else base
        groups.setdefault(label, []).append(p)

    labels = sorted(l for l, files in groups.items() if files)
    if not labels:
        raise ValueError(f"no clips found in {clips_dir}")

    images, targets = [], []
    common_sr = None
    for label_idx, label in enumerate(labels):
        for p in groups[label]:
            x, sr = _read_clip_wav(p)
            if common_sr is None:
                common_sr = sr
            elif sr != common_sr:
                raise ValueError(
                    f"sample rate mismatch: {p} is {sr} Hz, expected {common_sr} Hz "
                    "(mixing sample rates would poison the mel frequency axis)"
                )
            images.append(clip_to_logmel_image(x, sr))
            targets.append(label_idx)

    X = np.stack(images).astype(np.float32)
    Y = np.array(targets, dtype=np.int64)
    return X, Y, labels


def save_dataset(out_path, X, Y, labels):
    np.savez(out_path, X=X, Y=Y, labels=np.array(labels))


def load_dataset(path):
    data = np.load(path, allow_pickle=False)
    return data["X"], data["Y"], [str(l) for l in data["labels"]]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Build a mel-spectrogram dataset from clips/<label>_NN.wav.")
    p.add_argument("--clips-dir", default="training_clips",
                   help="input directory of clips, scanned recursively (default: training_clips)")
    p.add_argument("--out", default="dataset.npz", help="output .npz path (default: dataset.npz)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"mel backend: {_BACKEND}")

    if not os.path.isdir(args.clips_dir):
        print(f"error: clips dir not found: {args.clips_dir}", file=sys.stderr)
        return 2

    X, Y, labels = build_dataset(args.clips_dir)
    save_dataset(args.out, X, Y, labels)

    print(f"built dataset: {X.shape[0]} clips, {len(labels)} classes -> {args.out}")
    for i, label in enumerate(labels):
        print(f"  {label:>6}: {int(np.sum(Y == i)):3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
