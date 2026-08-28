"""features.py -- File 2 of the acoustic-keystroke reproduction pipeline.

Reads clips/<label>_NN.wav files and turns each clip into a 64x64 log-mel spectrogram
image. Saves the images, integer labels, and label names to a single .npz. Works with
or without librosa, using a pure-numpy fallback that matches librosa's own output.
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
    """Return which mel-spectrogram backend is active.

    Returns:
        "librosa" if librosa is installed, otherwise "numpy".
    """
    return _BACKEND


# ----------------------------------------------------------------------------
# Pure-numpy mel path (must match librosa numerically)
# ----------------------------------------------------------------------------
# librosa's default (htk=False) mel scale is Slaney's, linear below 1 kHz and
# logarithmic above. This differs from the HTK formula. Using HTK here would
# make the numpy path diverge from librosa.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(hz):
    """Convert frequency in Hz to the Slaney mel scale.

    Args:
        hz: Frequency or array of frequencies in Hz.

    Returns:
        float64 array of mel values, same shape as hz.
    """
    hz = np.asarray(hz, dtype=np.float64)
    mel = hz / _F_SP
    log_region = hz >= _MIN_LOG_HZ
    mel = np.where(log_region, _MIN_LOG_MEL + np.log(np.maximum(hz, 1e-12) / _MIN_LOG_HZ) / _LOGSTEP, mel)
    return mel


def _mel_to_hz(mel):
    """Convert Slaney mel values back to frequency in Hz.

    Args:
        mel: Mel value or array of mel values.

    Returns:
        float64 array of frequencies in Hz, same shape as mel.
    """
    mel = np.asarray(mel, dtype=np.float64)
    hz = mel * _F_SP
    log_region = mel >= _MIN_LOG_MEL
    hz = np.where(log_region, _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel - _MIN_LOG_MEL)), hz)
    return hz


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Build a Slaney-normalized triangular mel filterbank, matching librosa's default.

    Each triangle is scaled by 2/(hz_hi - hz_lo). This keeps equal-energy input
    producing roughly equal-energy output across band widths.

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT size used to derive the frequency bins.
        n_mels: Number of mel bands.
        fmin: Lowest frequency in Hz.
        fmax: Highest frequency in Hz; defaults to sr / 2.

    Returns:
        float32 array of shape (n_mels, n_fft // 2 + 1).
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
    """Split a signal into overlapping frames using librosa's center=True convention.

    Reflect-pads n_fft // 2 samples on each side before framing.

    Args:
        x: 1-D waveform.
        n_fft: Frame length in samples.
        hop_length: Hop size in samples between frame starts.

    Returns:
        Array of shape (n_frames, n_fft); empty along axis 0 if the padded
        signal is shorter than one frame.
    """
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((0, n_fft), dtype=xp.dtype)
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return xp[idx]


def _periodic_hann(n_fft):
    """Build a periodic Hann window matching scipy/librosa's default.

    This is the periodic ("fftbins=True") window, not np.hanning's symmetric one.
    The denominator here is n_fft, not n_fft - 1.

    Args:
        n_fft: Window length in samples.

    Returns:
        float64 array of shape (n_fft,).
    """
    n = np.arange(n_fft)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)


def melspectrogram_numpy(x, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """Compute a power mel-spectrogram without librosa.

    Args:
        x: 1-D waveform.
        sr: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop size in samples between frames.
        n_mels: Number of mel bands.

    Returns:
        float32 array of shape (n_mels, n_frames).
    """
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
    """Compute a power mel-spectrogram using librosa.

    Args:
        x: 1-D waveform.
        sr: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop size in samples between frames.
        n_mels: Number of mel bands.

    Returns:
        float32 array of shape (n_mels, n_frames).
    """
    # pad_mode is pinned to "reflect" explicitly. librosa's own default has
    # changed across versions, but the numpy fallback above always reflect-pads.
    return librosa.feature.melspectrogram(
        y=x.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=FMIN, power=2.0, center=True, htk=False, norm="slaney",
        pad_mode="reflect",
    ).astype(np.float32)


def melspectrogram(x, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """Compute a power mel-spectrogram using the active backend.

    Args:
        x: 1-D waveform.
        sr: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop size in samples between frames.
        n_mels: Number of mel bands.

    Returns:
        float32 array of shape (n_mels, n_frames), from
        melspectrogram_librosa if librosa is installed, otherwise from
        melspectrogram_numpy.
    """
    if _BACKEND == "librosa":
        return melspectrogram_librosa(x, sr, n_fft, hop_length, n_mels)
    return melspectrogram_numpy(x, sr, n_fft, hop_length, n_mels)


def power_to_db(mel, top_db=TOP_DB):
    """Convert a power mel-spectrogram to decibels relative to its own max.

    Equivalent to librosa.power_to_db(mel, ref=np.max, top_db=top_db).

    Args:
        mel: Power mel-spectrogram.
        top_db: Dynamic range floor below the max, in dB.

    Returns:
        Array the same shape as mel, in dB, floored at -top_db.
    """
    mel = np.maximum(mel, 1e-12)
    ref = float(np.max(mel)) if mel.size else 1e-12
    ref = max(ref, 1e-12)
    db = 10.0 * np.log10(mel / ref)
    return np.maximum(db, -top_db)


def resize_time_axis(img, target_frames):
    """Resample an image's time axis to a fixed number of frames.

    Args:
        img: 2-D array (n_mels, n_frames).
        target_frames: Desired number of columns after resampling.

    Returns:
        Array of shape (n_mels, target_frames), linearly interpolated along
        the time axis. An all-zero array if img has zero frames.
    """
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
    """Rescale an array to [0, 1] by its own min and max.

    Args:
        img: Input array.

    Returns:
        float32 array the same shape as img, scaled to [0, 1]. All zeros if
        the input's range is smaller than 1e-9 (a near-constant image).
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-9:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def clip_to_logmel_image(x, sr):
    """Convert a waveform to a normalized log-mel spectrogram image.

    Runs the full per-clip pipeline: mel-spectrogram, dB, resize, and rescale.

    Args:
        x: 1-D waveform.
        sr: Sample rate in Hz.

    Returns:
        float32 array of shape (N_MELS, IMG_SIZE) in [0, 1].
    """
    mel = melspectrogram(x, sr)
    db = power_to_db(mel)
    img = resize_time_axis(db, IMG_SIZE)
    return normalize_01(img)


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def time_shift(x, max_frac=TIME_SHIFT_FRAC, rng=None):
    """Shift a waveform in time by a random amount, zero-filling the gap.

    The shift never wraps around. A circular roll would smear the keystroke's
    transient onto the opposite end.

    Args:
        x: 1-D waveform.
        max_frac: Maximum shift magnitude, as a fraction of len(x).
        rng: numpy Generator to draw the shift from; a fresh default_rng()
            is used if None.

    Returns:
        Shifted waveform, same shape and dtype as x.
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
    """Apply SpecAugment-style frequency and time masking to a spectrogram image.

    Args:
        img: 2-D array (n_mels, n_frames).
        n_masks: Number of masks applied along each axis.
        frac: Width of each mask, as a fraction of that axis's length.
        rng: numpy Generator to draw mask positions from; a fresh
            default_rng() is used if None.

    Returns:
        A copy of img with the masked regions set to img's own mean value.
    """
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
    """Read a clip WAV file.

    Delegates to isolate_keystrokes.read_wav, which also handles float-format WAVs.

    Args:
        path: Path to the WAV file.

    Returns:
        Tuple (x, sr): waveform and sample rate.
    """
    import isolate_keystrokes as ik
    return ik.read_wav(path)


def _group_clip_paths_by_label(clips_dir):
    """Discover clip WAV files under clips_dir and group them by label.

    Args:
        clips_dir: Root directory to scan, recursively, for *.wav files.

    Returns:
        Dict mapping label name to a sorted list of file paths for clips
        whose file name is `<label>_NN.wav`.
    """
    paths = sorted(glob.glob(os.path.join(clips_dir, "**", "*.wav"), recursive=True))

    groups = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label = base.rsplit("_", 1)[0] if "_" in base else base
        groups.setdefault(label, []).append(p)
    return groups


def _featurize_clip_group(paths, common_sr):
    """Load and featurize every clip in one label's group.

    Args:
        paths: File paths for one label's clips.
        common_sr: Sample rate established so far by the caller, or None if
            no clip has been read yet.

    Returns:
        Tuple (images, common_sr): a list of 64x64 log-mel images, and the
        (possibly newly established) common sample rate.

    Raises:
        ValueError: If a clip's sample rate differs from common_sr.
    """
    images = []
    for p in paths:
        x, sr = _read_clip_wav(p)
        if common_sr is None:
            common_sr = sr
        elif sr != common_sr:
            raise ValueError(
                f"sample rate mismatch: {p} is {sr} Hz, expected {common_sr} Hz "
                "(mixing sample rates would poison the mel frequency axis)"
            )
        images.append(clip_to_logmel_image(x, sr))
    return images, common_sr


def build_dataset(clips_dir):
    """Scan clips_dir for <label>_NN.wav files, at any depth, and featurize them.

    The scan is recursive. A new recording session's clips join the existing
    pool instead of replacing it.

    Args:
        clips_dir: Root directory to scan for clip WAV files.

    Returns:
        Tuple (X, Y, labels): X is float32 (N, 64, 64) in [0, 1], Y is
        int64 (N,) indices into labels, and labels is the sorted list of
        label names present. Label groups with zero clips are skipped.

    Raises:
        ValueError: If clips_dir has no clips at all, or if clips have
            inconsistent sample rates (mixing sample rates would poison the
            mel frequency axis).
    """
    groups = _group_clip_paths_by_label(clips_dir)
    labels = sorted(l for l, files in groups.items() if files)
    if not labels:
        raise ValueError(f"no clips found in {clips_dir}")

    images, targets = [], []
    common_sr = None
    for label_idx, label in enumerate(labels):
        group_images, common_sr = _featurize_clip_group(groups[label], common_sr)
        images.extend(group_images)
        targets.extend([label_idx] * len(group_images))

    X = np.stack(images).astype(np.float32)
    Y = np.array(targets, dtype=np.int64)
    return X, Y, labels


def save_dataset(out_path, X, Y, labels):
    """Save a featurized dataset to a single .npz file.

    Args:
        out_path: Output .npz path.
        X: Images, float32 (N, 64, 64).
        Y: Labels, int64 (N,).
        labels: Ordered list of class names.
    """
    np.savez(out_path, X=X, Y=Y, labels=np.array(labels))


def load_dataset(path):
    """Load a dataset saved by save_dataset.

    Args:
        path: Path to a dataset .npz file.

    Returns:
        Tuple (X, Y, labels): X is float32 (N, 64, 64), Y is int64 (N,),
        and labels is a list of str class names.
    """
    data = np.load(path, allow_pickle=False)
    return data["X"], data["Y"], [str(l) for l in data["labels"]]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    """Build the command-line argument parser for the dataset-building CLI.

    Returns:
        Configured argparse.ArgumentParser.
    """
    p = argparse.ArgumentParser(description="Build a mel-spectrogram dataset from clips/<label>_NN.wav.")
    p.add_argument("--clips-dir", default="training_clips",
                   help="input directory of clips, scanned recursively (default: training_clips)")
    p.add_argument("--out", default="dataset.npz", help="output .npz path (default: dataset.npz)")
    return p


def main(argv=None):
    """Run the dataset-building CLI.

    Args:
        argv: Argument list to parse; defaults to sys.argv when None.

    Returns:
        Process exit code: 0 on success, 2 if clips_dir doesn't exist.
    """
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
