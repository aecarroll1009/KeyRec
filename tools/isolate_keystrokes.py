"""isolate_keystrokes.py -- File 1 of the acoustic-keystroke reproduction pipeline.

Reads <raw-dir>/<label>.wav (one key pressed ~25 times), detects each individual
keystroke with a robust short-time RMS energy envelope, and writes fixed-length
14400-sample clips to <clips-dir>/[<session>/]<label>_NN.wav.

Extraction workflow
--------------------
Every stage is subdivided by session, and the names must match across all three:
    unconverted_raw/day_N/  ->  converted_wavs/day_N/  ->  training_clips/day_N/

1. Record one .m4a per key, ~25 presses, ~1 s apart, into unconverted_raw/day_N/.
   Crop the record/stop finger-taps off each file first.
2. ./tools/convert_recordings.ps1 -Session day_N -> converted_wavs/day_N/<label>.wav
   at 48 kHz. One sample rate for the whole pool; features.py asserts it.
3. Dry-run before writing:

     python tools/isolate_keystrokes.py --auto --dry-run \
            --raw-dir converted_wavs/day_N --expected 25

4. Read the report, then drop --dry-run and add --session day_N to write.
   --raw-dir is not recursive by design: one run extracts one session, so two
   sessions can never be silently merged into one set of clips.

Extraction rules
-----------------
* Never reuse one key's parameters for another. The k that yields exactly 25
  has ranged from 4.5 to 26 across a single 27-file session. A global
  threshold over-detects the loud keys and under-detects the quiet ones.
  --auto searches every file independently; keep it that way.
* Judge a file on coverage, not just the count. Coverage is the fraction of a
  press's energy its clip captures. 25 detections with several clips under
  LOW_COVERAGE is worse than 24 clean ones: a low-coverage clip is mostly
  room noise filed under a real key, and it poisons that class.
* A narrow k window means fragile, not fine. `kwin` is how many grid steps
  yield exactly `expected`. kwin=1 means one noisy press tips the file to
  one fewer or one more detection -- treat that as a recording problem, not
  a tuning success; the file needs re-recording.
* If no k works, the recording is bad. Do not widen the grid to force it.
  Real causes include a contaminating noise burst mid-take, and presses
  tapering to half their initial loudness by the end of the take.
* SNR consistency across keys is a research concern, not just a data one. A
  key recorded at 6x SNR when the rest are at 20x will be the model's worst
  class, and that reads as an acoustic finding when it is really a recording
  artifact. Re-record the outliers.
* Growing the pool requires --session. write_clips(replace=True) deletes
  every clip for a label in its target directory, so extracting a new
  session into an existing session's directory destroys the old clips.
  Sessions in their own subdirectories accumulate, because features.py
  scans recursively.
* Per-key settings land in <clips-dir>/[<session>/]isolation_params.json.
  That file is the extraction record for the write-up -- keep it.

Design constraints
-------------------
* stdlib and numpy only on the core path, so it runs with no third-party
  installs. matplotlib is imported lazily and only when --plot is passed.
* stdlib `wave` cannot read 32-bit float WAV (which phones and ffmpeg emit),
  so this module parses the RIFF header directly and handles 8/16/24/32-bit
  PCM and 32/64-bit float.
* Detection is robust: threshold = median + k*MAD of the energy envelope,
  with rising-edge onsets and a debounce (min-gap) so the decaying tail is
  not re-detected.
* The prominence floor references a high percentile of the envelope rather
  than its max, so one loud bump or thump cannot dominate the max and
  suppress every genuine, quieter keystroke.
"""

import argparse
import json
import os
import re
import struct
import sys
import wave
from collections import namedtuple

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CLIP_LEN = 14400            # fixed output clip length, samples (~300 ms @ 48 kHz)
PREROLL = int(0.15 * CLIP_LEN)   # samples kept before the onset so the attack transient is not clipped
FRAME_MS = 10.0             # RMS analysis window
HOP_MS = 5.0                # RMS analysis hop
DEFAULT_K = 5.0             # threshold = median + k * MAD
DEFAULT_MIN_GAP_MS = 350.0  # debounce between successive onsets
DEFAULT_EXPECTED = 25       # presses per key (for the OK/CHECK report)
MIN_PROM_FRAC = 0.2         # a detection's peak must exceed this fraction of the
                            # 99th-percentile envelope level, not of the max
PEAK_PERCENTILE = 99.0      # "typical strong keystroke" reference level
LOW_COVERAGE = 0.75         # auto mode flags clips capturing less than this
                            # fraction of their press's energy


# ----------------------------------------------------------------------------
# WAV reading -- custom RIFF parser (stdlib `wave` cannot read float WAV)
# ----------------------------------------------------------------------------
def _parse_riff_chunks(buf, path):
    """Walk a RIFF/WAVE chunk list and pull out the format and data chunks.

    Args:
        buf: Raw bytes of the WAV file.
        path: Source path, used only for error messages.

    Returns:
        Tuple (audio_format, num_channels, sample_rate, bits, data_bytes).

    Raises:
        ValueError: If the file has no fmt chunk or no data chunk.
    """
    audio_format = num_channels = sample_rate = bits = None
    data_bytes = None

    pos = 12
    n = len(buf)
    while pos + 8 <= n:
        chunk_id = buf[pos:pos + 4]
        (chunk_size,) = struct.unpack_from("<I", buf, pos + 4)
        # Some encoders write a placeholder or bogus data-chunk size (e.g.
        # 0xFFFFFFFF); clamp to the bytes actually present in the buffer.
        chunk_size = min(chunk_size, n - (pos + 8))
        payload = buf[pos + 8: pos + 8 + chunk_size]

        if chunk_id == b"fmt ":
            (audio_format, num_channels, sample_rate,
             _byte_rate, _block_align, bits) = struct.unpack_from("<HHIIHH", payload, 0)
            # WAVE_FORMAT_EXTENSIBLE: the real format code is the first two
            # bytes of the SubFormat GUID, at offset 24 in the fmt payload.
            if audio_format == 0xFFFE and len(payload) >= 26:
                (audio_format,) = struct.unpack_from("<H", payload, 24)
        elif chunk_id == b"data":
            data_bytes = payload

        pos += 8 + chunk_size + (chunk_size & 1)  # chunks are word-aligned

    if audio_format is None:
        raise ValueError(f"{path}: no fmt chunk")
    if data_bytes is None:
        raise ValueError(f"{path}: no data chunk")

    return audio_format, num_channels, sample_rate, bits, data_bytes


def read_wav(path):
    """Load a WAV file as mono float32 samples.

    Handles PCM 8/16/24/32-bit and IEEE float 32/64-bit, mono or
    multi-channel (channels are averaged to mono), including
    WAVE_FORMAT_EXTENSIBLE.

    Args:
        path: Path to the WAV file.

    Returns:
        Tuple (samples, sample_rate): float32 samples in [-1, 1], and the
        sample rate in Hz.
    """
    with open(path, "rb") as f:
        buf = f.read()

    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    audio_format, num_channels, sample_rate, bits, data_bytes = _parse_riff_chunks(buf, path)
    samples = _decode_samples(data_bytes, audio_format, bits, path)

    if num_channels > 1:
        usable = (len(samples) // num_channels) * num_channels
        samples = samples[:usable].reshape(-1, num_channels).mean(axis=1)

    return samples.astype(np.float32, copy=False), int(sample_rate)


def _decode_samples(data_bytes, audio_format, bits, path):
    """Decode raw PCM/float bytes to float32 samples in [-1, 1].

    Args:
        data_bytes: Raw bytes from the WAV data chunk.
        audio_format: WAV format code (1 = PCM, 3 = IEEE float).
        bits: Bits per sample.
        path: Source path, used only for error messages.

    Returns:
        1-D float32 array of decoded samples.

    Raises:
        ValueError: If the format/bit-depth combination is unsupported.
    """
    if audio_format == 1:  # PCM (integer)
        if bits == 8:  # 8-bit PCM is unsigned, centred at 128
            x = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32)
            return (x - 128.0) / 128.0
        if bits == 16:
            x = np.frombuffer(data_bytes, dtype="<i2").astype(np.float32)
            return x / 32768.0
        if bits == 24:
            raw = np.frombuffer(data_bytes, dtype=np.uint8)
            raw = raw[: (len(raw) // 3) * 3].reshape(-1, 3).astype(np.int32)
            val = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
            val = np.where(val & 0x800000, val - 0x1000000, val)  # sign-extend
            return val.astype(np.float32) / float(2 ** 23)
        if bits == 32:
            x = np.frombuffer(data_bytes, dtype="<i4").astype(np.float64)
            return (x / float(2 ** 31)).astype(np.float32)
        raise ValueError(f"{path}: unsupported PCM bit depth {bits}")

    if audio_format == 3:  # IEEE float
        if bits == 32:
            return np.frombuffer(data_bytes, dtype="<f4").astype(np.float32)
        if bits == 64:
            return np.frombuffer(data_bytes, dtype="<f8").astype(np.float32)
        raise ValueError(f"{path}: unsupported float bit depth {bits}")

    raise ValueError(f"{path}: unsupported WAV audio_format {audio_format}")


def write_wav(path, x, sample_rate):
    """Write mono float samples as a 16-bit PCM WAV file.

    Args:
        path: Output file path.
        x: Mono samples in [-1, 1].
        sample_rate: Sample rate, Hz.
    """
    x16 = np.clip(x, -1.0, 1.0)
    x16 = np.round(x16 * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(x16.tobytes())


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def rms_envelope(x, frame, hop):
    """Compute a short-time RMS energy envelope.

    Args:
        x: Mono audio samples.
        frame: Analysis window length, samples.
        hop: Analysis hop length, samples.

    Returns:
        Tuple (env, hop): the envelope array and the hop size passed in.
    """
    if len(x) < frame:
        return np.zeros(0, dtype=np.float32), hop
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    env = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return env.astype(np.float32), hop


def detect_onsets(env, hop, sample_rate, k, min_gap_ms):
    """Detect keystroke onsets from an energy envelope.

    Threshold uses robust statistics: median + k*MAD. The prominence floor
    is a high percentile of the envelope, not its max, so one loud transient
    cannot raise the bar above the genuine keystrokes.

    Args:
        env: RMS energy envelope.
        hop: Envelope hop size, samples.
        sample_rate: Sample rate, Hz.
        k: Threshold multiplier: threshold = median + k*MAD.
        min_gap_ms: Minimum gap between onsets, ms (debounce).

    Returns:
        Tuple (onsets, threshold): onset sample positions and the threshold used.
    """
    if len(env) == 0:
        return [], 0.0

    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    if mad <= 1e-12:                     # near-silent / degenerate: fall back to std
        mad = float(np.std(env)) or 1e-9
    threshold = med + k * mad

    ref_peak = float(np.percentile(env, PEAK_PERCENTILE))   # robust "strong keystroke" level
    min_peak = MIN_PROM_FRAC * ref_peak

    min_gap_frames = max(1, int(round((min_gap_ms / 1000.0) * sample_rate / hop)))

    onsets = []
    i = 0
    n = len(env)
    while i < n:
        if env[i] > threshold:
            region_peak = float(env[i:i + min_gap_frames].max())
            if region_peak >= min_peak:
                onsets.append(i * hop)          # onset position in samples
                i += min_gap_frames             # debounce: skip past the decay tail
                continue
        i += 1

    return onsets, threshold


def extract_clip(x, onset_sample):
    """Extract a fixed-length clip around a detected onset.

    The clip starts PREROLL samples before the onset and is zero-padded at
    the tail if it would run past the end of `x`.

    Args:
        x: Full audio signal the onset was detected in.
        onset_sample: Onset position, samples.

    Returns:
        A CLIP_LEN-sample clip.
    """
    start = onset_sample - PREROLL
    start = max(0, min(start, max(0, len(x) - CLIP_LEN)))
    clip = x[start:start + CLIP_LEN]
    if len(clip) < CLIP_LEN:
        clip = np.concatenate([clip, np.zeros(CLIP_LEN - len(clip), dtype=x.dtype)])
    return clip


# ----------------------------------------------------------------------------
# Analysis / writing -- kept separate
# ----------------------------------------------------------------------------
# Detection is pure and writes nothing, so auto mode can sweep the whole
# parameter grid for a file without leaving a single clip on disk. Clips are
# written once, after the settings for that file are settled.
class Analysis:
    """Everything one detection pass produced, with nothing written to disk."""

    __slots__ = ("x", "sr", "env", "hop", "onsets", "threshold", "k", "min_gap_ms")

    def __init__(self, x, sr, env, hop, onsets, threshold, k, min_gap_ms):
        self.x, self.sr = x, sr
        self.env, self.hop = env, hop
        self.onsets, self.threshold = onsets, threshold
        self.k, self.min_gap_ms = k, min_gap_ms

    @property
    def n(self):
        return len(self.onsets)

    @property
    def duration_s(self):
        return len(self.x) / float(self.sr)

    def gaps_ms(self):
        """Compute inter-onset intervals.

        Returns:
            Gaps between consecutive onsets, in ms. Useful for spotting a
            double-strike (a gap far below the rest) or a missed press (one
            roughly doubled).
        """
        if len(self.onsets) < 2:
            return np.zeros(0)
        return np.diff(np.asarray(self.onsets, dtype=np.float64)) / self.sr * 1000.0

    def coverage(self, look_ms=800.0):
        """Compute the fraction of each press's energy its clip captures.

        Low coverage has two causes. A press can ring longer than the fixed
        clip window. Or the onset fired early on room noise, pushing the
        real strike toward the end of the clip or past it -- a misaligned
        detection that silently poisons a class if left in the training set.

        Energy is noise-subtracted, using the 20th percentile of the
        envelope as the room floor.

        Args:
            look_ms: How far past the clip window to look when computing
                each press's total energy, ms.

        Returns:
            Coverage fraction per onset, in [0, 1].
        """
        n = len(self.onsets)
        if n == 0 or len(self.env) == 0:
            return np.zeros(0)

        floor = float(np.percentile(self.env, 20.0))
        power = np.maximum(self.env.astype(np.float64) - floor, 0.0) ** 2
        look = max(1, int(round(look_ms / 1000.0 * self.sr / self.hop)))

        out = np.empty(n, dtype=np.float64)
        for i, onset in enumerate(self.onsets):
            start = max(0, (onset - PREROLL) // self.hop)          # what the clip covers
            stop = start + max(1, CLIP_LEN // self.hop)
            ref_stop = max(stop, start + look)                     # clip vs a longer look-ahead
            total = float(power[start:ref_stop].sum())
            out[i] = float(power[start:stop].sum()) / total if total > 0 else 1.0
        return out


def analyze(x, sr, k, min_gap_ms):
    """Run the detector over already-loaded audio.

    Args:
        x: Mono audio samples.
        sr: Sample rate, Hz.
        k: Threshold multiplier passed to detect_onsets.
        min_gap_ms: Debounce gap passed to detect_onsets, ms.

    Returns:
        An Analysis with the detection results. Writes nothing to disk.
    """
    frame = max(1, int(round(FRAME_MS / 1000.0 * sr)))
    hop = max(1, int(round(HOP_MS / 1000.0 * sr)))
    env, hop = rms_envelope(x, frame, hop)
    onsets, threshold = detect_onsets(env, hop, sr, k, min_gap_ms)
    return Analysis(x, sr, env, hop, onsets, threshold, k, min_gap_ms)


def analyze_file(wav_path, k, min_gap_ms):
    """Load a WAV file and run the detector over it.

    Args:
        wav_path: Path to the WAV file.
        k: Threshold multiplier passed to detect_onsets.
        min_gap_ms: Debounce gap passed to detect_onsets, ms.

    Returns:
        An Analysis with the detection results. Writes nothing to disk.
    """
    x, sr = read_wav(wav_path)
    return analyze(x, sr, k, min_gap_ms)


def existing_clips(label, clips_dir):
    """List clips already on disk for one label.

    Matches only the exact `<label>_NN.wav` pattern this module writes, so a
    label like 'a' cannot match an unrelated file such as 'space_01.wav'.

    Args:
        label: Key label to match.
        clips_dir: Directory to search.

    Returns:
        Sorted list of matching clip paths. Empty if clips_dir does not exist.
    """
    if not os.path.isdir(clips_dir):
        return []
    pat = re.compile(r"^" + re.escape(label) + r"_\d+\.wav$")
    return sorted(os.path.join(clips_dir, f)
                  for f in os.listdir(clips_dir) if pat.match(f))


def write_clips(analysis, label, clips_dir, replace=True):
    """Write one clip per detected onset.

    Args:
        analysis: Analysis to write clips from.
        label: Key label used as the clip filename prefix.
        clips_dir: Directory to write clips into (created if missing).
        replace: If True, delete this label's existing clips first. Without
            it, a re-run that detects fewer presses than before leaves the
            old run's extra clips behind, duplicating data in the pool.

    Returns:
        Tuple (paths written, number of existing clips removed).
    """
    os.makedirs(clips_dir, exist_ok=True)
    removed = 0
    if replace:
        for p in existing_clips(label, clips_dir):
            os.remove(p)
            removed += 1

    written = []
    for i, onset in enumerate(analysis.onsets):
        out = os.path.join(clips_dir, f"{label}_{i:02d}.wav")
        write_wav(out, extract_clip(analysis.x, onset), analysis.sr)
        written.append(out)
    return written, removed


# ----------------------------------------------------------------------------
# Per-file driver (batch mode)
# ----------------------------------------------------------------------------
def process_file(wav_path, label, clips_dir, k, min_gap_ms, expected, do_plot):
    """Detect and write clips for one raw WAV file, using fixed settings.

    Args:
        wav_path: Path to the raw per-key recording.
        label: Key label.
        clips_dir: Output directory for clips.
        k: Threshold multiplier: threshold = median + k*MAD.
        min_gap_ms: Debounce gap between onsets, ms.
        expected: Expected press count, for the OK/CHECK report.
        do_plot: Whether to save a waveform+envelope PNG.

    Returns:
        Number of onsets detected (and clips written).
    """
    a = analyze_file(wav_path, k, min_gap_ms)
    write_clips(a, label, clips_dir, replace=True)

    status = "OK   " if a.n == expected else "CHECK"
    print(f"[{status}] {label:>6}: detected {a.n:3d} / expected {expected:3d}  "
          f"(sr={a.sr}, thr={a.threshold:.5f})")

    if do_plot:
        _plot(label, a.x, a.env, a.hop, a.sr, a.onsets, a.threshold, clips_dir)

    return a.n


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _plot(label, x, env, hop, sr, onsets, threshold, clips_dir):
    """Save a waveform + envelope PNG for one file, if matplotlib is available.

    Args:
        label: Key label, used in the title and output filename.
        x: Mono audio samples.
        env: RMS energy envelope.
        hop: Envelope hop size, samples.
        sr: Sample rate, Hz.
        onsets: Detected onset sample positions.
        threshold: Detection threshold used.
        clips_dir: Directory to save the PNG into.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (--plot skipped: matplotlib not installed)")
        return
    t_env = np.arange(len(env)) * hop / sr
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(np.arange(len(x)) / sr, x, lw=0.4)
    ax[0].set_ylabel("waveform")
    ax[0].set_title(f"{label}: {len(onsets)} detected")
    ax[1].plot(t_env, env, lw=0.6, label="RMS env")
    ax[1].axhline(threshold, color="r", ls="--", lw=0.8, label="threshold")
    for onset in onsets:
        ax[1].axvline(onset / sr, color="g", lw=0.6, alpha=0.6)
    ax[1].set_ylabel("energy")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="upper right", fontsize=8)
    out = os.path.join(clips_dir, f"{label}_envelope.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  plot -> {out}")


# ----------------------------------------------------------------------------
# Auto mode -- tune each file independently to the expected press count
# ----------------------------------------------------------------------------
PARAMS_FILE = "isolation_params.json"

# Every recording differs (mic distance, room noise, how hard the key was hit),
# so a single global threshold does not fit the set -- measured k values range
# from ~4.5 to ~23 across 27 files. Auto mode searches each file on its own.
AUTO_K_GRID = np.round(np.arange(1.0, 30.01, 0.25), 2)
AUTO_GAP_GRID = (350.0, 450.0, 550.0, 700.0, 850.0, 1000.0)


def _contiguous_runs(values, step=0.26):
    """Split a sorted list into runs of adjacent grid values.

    Args:
        values: Sorted list of numeric grid values.
        step: Maximum gap between consecutive values to treat as adjacent.

    Returns:
        List of runs, each a list of consecutive adjacent values.
    """
    runs, cur = [], [values[0]]
    for a, b in zip(values, values[1:]):
        if abs(b - a) <= step:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


def tune_file(env, hop, sr, expected, k_grid=AUTO_K_GRID, gap_grid=AUTO_GAP_GRID):
    """Search the parameter grid for settings giving exactly `expected` onsets.

    Prefers the smallest working min-gap, since a larger gap can hide a
    genuine double press. Among k values that hit the target, picks the
    middle of the widest contiguous run, since an edge value sits one noisy
    press away from a different count.

    Args:
        env: RMS energy envelope.
        hop: Envelope hop size, samples.
        sr: Sample rate, Hz.
        expected: Target onset count.
        k_grid: Candidate k values to search.
        gap_grid: Candidate min-gap values to search, ms.

    Returns:
        Tuple (k, min_gap_ms, onsets, threshold, k_window_width), or None if
        no combination on the grid produces exactly `expected` onsets.
    """
    for gap in gap_grid:
        hits = [float(k) for k in k_grid
                if len(detect_onsets(env, hop, sr, float(k), gap)[0]) == expected]
        if not hits:
            continue
        widest = max(_contiguous_runs(hits), key=len)
        k = widest[len(widest) // 2]
        onsets, threshold = detect_onsets(env, hop, sr, k, gap)
        return k, gap, onsets, threshold, len(widest)
    return None


TunedFile = namedtuple("TunedFile", "analysis kwin cov n_low note row")


def _load_and_tune(wav_path, expected):
    """Load a raw WAV file and search for settings hitting `expected` onsets.

    Args:
        wav_path: Path to the raw per-key recording.
        expected: Target press count.

    Returns:
        Tuple (x, sr, env, hop, tuned), where `tuned` is the tune_file()
        result and is None if no grid setting produces exactly `expected`
        detections.
    """
    x, sr = read_wav(wav_path)
    frame = max(1, int(round(FRAME_MS / 1000.0 * sr)))
    hop = max(1, int(round(HOP_MS / 1000.0 * sr)))
    env, hop = rms_envelope(x, frame, hop)
    tuned = tune_file(env, hop, sr, expected)
    return x, sr, env, hop, tuned


def _closest_achievable_count(env, hop, sr, expected):
    """Find the press count nearest to `expected` reachable by any k in the grid.

    Args:
        env: RMS energy envelope for the file.
        hop: Envelope hop size, samples.
        sr: Sample rate, Hz.
        expected: Target press count.

    Returns:
        The achievable count closest to `expected`.
    """
    counts = sorted({len(detect_onsets(env, hop, sr, float(k), 350.0)[0])
                      for k in AUTO_K_GRID})
    return min(counts, key=lambda c: abs(c - expected))


def _build_tuned_result(label, x, sr, env, hop, tuned):
    """Assemble the Analysis and review note for one successfully tuned file.

    Args:
        label: Key label being processed.
        x: Loaded audio samples for the file.
        sr: Sample rate, Hz.
        env: RMS energy envelope for the file.
        hop: Envelope hop size, samples.
        tuned: Result tuple from tune_file().

    Returns:
        A TunedFile with the Analysis, k-window width, coverage array,
        low-coverage count, review note ("OK" or "CHECK: ..."), and the
        formatted report row for this file.
    """
    k, gap, onsets, threshold, kwin = tuned
    a = Analysis(x, sr, env, hop, onsets, threshold, k, gap)
    cov = a.coverage()
    n_low = int((cov < LOW_COVERAGE).sum())

    note = "OK"
    if n_low or kwin < 3:
        bits = []
        if n_low:
            bits.append(f"{n_low} clip(s) under {LOW_COVERAGE:.0%} coverage")
        if kwin < 3:
            bits.append(f"narrow k window ({kwin} step{'s' if kwin != 1 else ''})")
        note = "CHECK: " + ", ".join(bits)

    row = (f"{label:>6} {k:7g} {gap:6.0f} {kwin:5d} {len(onsets):4d} "
           f"{np.median(cov):7.0%} {cov.min():7.0%} {n_low:4d}  {note}")
    return TunedFile(a, kwin, cov, n_low, note, row)


def _params_record(analysis, kwin, cov, n_low, expected, wav_path):
    """Build the JSON-serializable settings record for one tuned file.

    Args:
        analysis: Analysis for the file's chosen settings.
        kwin: Width, in grid steps, of the k window that reached `expected`.
        cov: Per-onset coverage fractions for the file.
        n_low: Number of clips below LOW_COVERAGE.
        expected: Target press count for this session.
        wav_path: Path to the source raw recording.

    Returns:
        A dict suitable for writing into isolation_params.json.
    """
    return {
        "threshold_k": analysis.k,
        "min_gap_ms": analysis.min_gap_ms,
        "expected": int(expected),
        "detected": analysis.n,
        "k_window_steps": kwin,
        "sample_rate": analysis.sr,
        "source": os.path.basename(wav_path),
        "coverage_median": round(float(np.median(cov)), 4),
        "coverage_min": round(float(cov.min()), 4),
        "clips_below_coverage": n_low,
    }


def _report_auto_summary(clips_dir, dry_run, ok, flagged, failed, saved):
    """Print the end-of-run summary for auto mode and save per-key parameters.

    Args:
        clips_dir: Directory clips were written into (already session-qualified).
        dry_run: Whether this was a dry run (nothing written).
        ok: List of (label, clips_written, clips_removed) for succeeded files.
        flagged: List of (label, note) for files that need review.
        failed: List of (label, reason) for files that could not be tuned.
        saved: Per-key settings dict to write to isolation_params.json.

    Returns:
        Exit code: 1 if any file failed, 0 otherwise.
    """
    print()
    if not dry_run and ok:
        path = _save_params(clips_dir, saved)
        total = sum(n for _, n, _ in ok)
        replaced = sum(r for _, _, r in ok)
        print(f"wrote {total} clips for {len(ok)} key(s) to {clips_dir}/"
              + (f" (replaced {replaced} existing)" if replaced else ""))
        print(f"per-key parameters recorded in {path}")
    if flagged:
        print(f"\nflagged for review ({len(flagged)}):")
        for label, note in flagged:
            print(f"  {label:>6}: {note}")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for label, why in failed:
            print(f"  {label:>6}: {why}")
    return 1 if failed else 0


def auto_session(labels, raw_dir, clips_dir, expected, dry_run=False, session=None):
    """Tune every raw file to `expected` presses independently, then write clips.

    Each file's detector settings are searched from scratch, since one
    global setting does not fit every recording equally well. Pass
    `session` to add this pool to a session's clips instead of replacing
    them.

    Args:
        labels: Key labels to process, each backed by <raw_dir>/<label>.wav.
        raw_dir: Directory containing the raw per-key recordings.
        clips_dir: Base output directory for clips.
        expected: Target press count per key.
        dry_run: If True, report tuning results for every file but write nothing.
        session: Subdirectory of clips_dir to write into, so a new recording
            session adds to the pool instead of replacing an earlier one.

    Returns:
        Exit code: 1 if any file failed to tune, 0 otherwise.
    """
    if session:
        clips_dir = os.path.join(clips_dir, session)
    saved = _load_params(clips_dir)
    ok, failed, flagged = [], [], []

    hdr = (f"{'key':>6} {'k':>7} {'gap':>6} {'kwin':>5} {'n':>4} "
           f"{'cov med':>8} {'cov min':>8} {'low':>4}  result")
    print(f"\nauto-tuning {len(labels)} file(s) to exactly {expected} presses each"
          + ("  [DRY RUN -- nothing will be written]" if dry_run else ""))
    print(hdr)
    print("-" * len(hdr))

    for label in labels:
        wav_path = os.path.join(raw_dir, f"{label}.wav")
        if not os.path.isfile(wav_path):
            print(f"{label:>6}  {'MISSING':>50}")
            failed.append((label, "file not found"))
            continue

        x, sr, env, hop, tuned = _load_and_tune(wav_path, expected)
        if tuned is None:
            near = _closest_achievable_count(env, hop, sr, expected)
            print(f"{label:>6} {'-':>7} {'-':>6} {'-':>5} {'-':>4} "
                  f"{'-':>8} {'-':>8} {'-':>4}  FAILED: never {expected} "
                  f"(closest {near})")
            failed.append((label, f"no setting yields {expected}; closest {near}"))
            continue

        tf = _build_tuned_result(label, x, sr, env, hop, tuned)
        print(tf.row)
        if tf.note != "OK":
            flagged.append((label, tf.note))

        if not dry_run:
            written, removed = write_clips(tf.analysis, label, clips_dir, replace=True)
            saved[label] = _params_record(tf.analysis, tf.kwin, tf.cov, tf.n_low,
                                           expected, wav_path)
            ok.append((label, len(written), removed))

    return _report_auto_summary(clips_dir, dry_run, ok, flagged, failed, saved)


def _load_params(clips_dir):
    """Load previously saved per-key settings, if any.

    Args:
        clips_dir: Directory containing isolation_params.json.

    Returns:
        The saved settings dict, or {} if none exists or it cannot be read.
    """
    path = os.path.join(clips_dir, PARAMS_FILE)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            print(f"  (could not read {path}; starting fresh)")
    return {}


def _save_params(clips_dir, params):
    """Write per-key settings to isolation_params.json.

    Args:
        clips_dir: Directory to write into (created if missing).
        params: Settings dict to serialize.

    Returns:
        Path to the written file.
    """
    os.makedirs(clips_dir, exist_ok=True)
    path = os.path.join(clips_dir, PARAMS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    """Build the command-line argument parser.

    Returns:
        A configured argparse.ArgumentParser.
    """
    p = argparse.ArgumentParser(description="Isolate individual keystrokes from raw/<label>.wav recordings.")
    p.add_argument("--raw-dir", default="raw", help="input directory of <label>.wav files (default: raw)")
    p.add_argument("--clips-dir", default="training_clips",
                   help="output directory for clips (default: training_clips)")
    p.add_argument("--only", default=None, help="process only this label (e.g. --only a)")
    p.add_argument("--threshold-k", type=float, default=DEFAULT_K, help="threshold = median + k*MAD (default: %(default)s)")
    p.add_argument("--min-gap-ms", type=float, default=DEFAULT_MIN_GAP_MS, help="debounce between onsets, ms (default: %(default)s)")
    p.add_argument("--expected", type=int, default=DEFAULT_EXPECTED, help="expected presses per key for OK/CHECK (default: %(default)s)")
    p.add_argument("--plot", action="store_true", help="save a waveform+envelope PNG per file (needs matplotlib)")
    p.add_argument("--auto", action="store_true",
                   help="tune each file independently until it yields exactly --expected "
                        "detections, then write its clips. Per-key settings are recorded in "
                        "clips/" + PARAMS_FILE + ". Ignores --threshold-k/--min-gap-ms.")
    p.add_argument("--dry-run", action="store_true",
                   help="with --auto: report the tuning result for every file but write nothing")
    p.add_argument("--session", default=None,
                   help="with --auto: write into <clips-dir>/<session>/ so a new recording "
                        "session adds to the pool. features.py scans recursively, so every "
                        "session is included; without this a re-extract replaces the old clips.")
    return p


def _print_no_labels_diagnostic(raw_dir):
    """Report that a raw directory has no top-level .wav files.

    --raw-dir is not recursive, so a directory of session subfolders looks
    empty. If raw_dir contains session subfolders, suggests pointing at one.

    Args:
        raw_dir: Directory that was scanned.
    """
    subs = sorted(d for d in os.listdir(raw_dir)
                  if os.path.isdir(os.path.join(raw_dir, d))) \
        if os.path.isdir(raw_dir) else []
    print(f"no .wav files directly in {raw_dir}", file=sys.stderr)
    if subs:
        print(f"  it contains session subfolder(s): {', '.join(subs)}", file=sys.stderr)
        print(f"  point --raw-dir at one, e.g.:\n"
              f"    --raw-dir {os.path.join(raw_dir, subs[-1])} "
              f"--session {subs[-1]}", file=sys.stderr)


def _run_manual_mode(labels, raw_dir, clips_dir, threshold_k, min_gap_ms, expected, do_plot):
    """Process every label with fixed detector settings (non-auto mode).

    Args:
        labels: Key labels to process.
        raw_dir: Directory containing the raw per-key recordings.
        clips_dir: Output directory for clips.
        threshold_k: Fixed k for threshold = median + k*MAD.
        min_gap_ms: Fixed debounce between onsets, ms.
        expected: Expected press count, for the OK/CHECK report.
        do_plot: Whether to save a waveform+envelope PNG per file.

    Returns:
        Total number of clips written across all labels.
    """
    total = 0
    for label in labels:
        wav_path = os.path.join(raw_dir, f"{label}.wav")
        if not os.path.isfile(wav_path):
            print(f"[MISS ] {label:>6}: {wav_path} not found")
            continue
        total += process_file(wav_path, label, clips_dir,
                              threshold_k, min_gap_ms, expected, do_plot)
    return total


def main(argv=None):
    """Parse CLI arguments and run either auto or manual extraction mode.

    Args:
        argv: Argument list to parse, or None to use sys.argv.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    if args.only is not None:
        labels = [args.only]
    else:
        if not os.path.isdir(args.raw_dir):
            print(f"error: raw dir not found: {args.raw_dir}", file=sys.stderr)
            return 2
        labels = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(args.raw_dir)
            if f.lower().endswith(".wav")
        )

    if not labels:
        _print_no_labels_diagnostic(args.raw_dir)
        return 1

    if args.auto:
        return auto_session(labels, args.raw_dir, args.clips_dir,
                            args.expected, dry_run=args.dry_run, session=args.session)

    total = _run_manual_mode(labels, args.raw_dir, args.clips_dir,
                             args.threshold_k, args.min_gap_ms, args.expected, args.plot)
    print(f"done: {total} clips written to {args.clips_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
