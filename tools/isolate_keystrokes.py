"""isolate_keystrokes.py  --  File 1 of the acoustic-keystroke reproduction pipeline.

Reads <raw-dir>/<label>.wav (one key pressed ~25 times), detects each individual
keystroke with a robust short-time RMS energy envelope, and writes fixed-length
14400-sample clips to <clips-dir>/[<session>/]<label>_NN.wav.

===========================================================================
HOW TO EXTRACT CLIPS -- follow this, do not improvise
===========================================================================
Established 2026-07-26 on the first full 27-key set. Read this before running
anything; the failure modes below were all hit for real.

  Every stage is subdivided by session and the names must match across all three:
      unconverted_raw/day_N/  ->  converted_wavs/day_N/  ->  training_clips/day_N/

  1. Record one .m4a per key, ~25 presses, ~1 s apart, into unconverted_raw/day_N/.
     Crop the record/stop finger-taps off each file yourself first.
  2. ./tools/convert_recordings.ps1 -Session day_N   -> converted_wavs/day_N/<label>.wav
     @ 48 kHz. One sample rate for the whole pool; features.py asserts it.
  3. ALWAYS dry-run before writing:

       python tools/isolate_keystrokes.py --auto --dry-run \
              --raw-dir converted_wavs/day_N --expected 25

  4. Read the report, then drop --dry-run and add --session day_N to write.
     --raw-dir is NOT recursive on purpose: one run extracts one session, so
     two sessions can never be silently merged into one set of clips.

RULES, each learned the hard way:

  * NEVER reuse one key's parameters for another. The k that yields exactly 25
    ranged from 4.5 to 26 across 27 files recorded in ONE sitting. A global
    threshold silently over-detects the loud keys and under-detects the quiet
    ones. --auto searches every file independently; keep it that way.
  * Judge a file on COVERAGE, not just the count. Coverage is the fraction of a
    press's energy its clip captures. 25 detections with several clips under
    LOW_COVERAGE is WORSE than 24 clean ones: a low-coverage clip is mostly
    room noise filed under a real key, and it poisons that class.
  * A narrow k window means fragile, not fine. `kwin` is how many grid steps
    yield exactly `expected`. kwin=1 means one noisy press tips the file to 24
    or 26 -- treat it as a recording problem, not a tuning success. Every file
    that showed kwin=1 (e, v, p) turned out to have a genuine defect and needed
    re-recording; after re-recording, all had kwin >= 30.
  * If no k works, the recording is bad. Do not widen the grid to force it.
    Real causes seen: a contaminating noise burst mid-take, and presses tapering
    to half their initial loudness by the end of the take.
  * SNR consistency across keys is a RESEARCH issue, not just a data issue. A
    key recorded at 6x SNR when the rest are at 20x will be the model's worst
    class, and that reads as an acoustic finding when it is really a recording
    artifact. Re-record the outliers.
  * Growing the pool: use --session. write_clips(replace=True) deletes every
    clip for a label in its target directory, so extracting a new session into
    an existing session's directory DESTROYS the old clips. Sessions in their
    own subdirectories accumulate, because features.py scans recursively.
  * Per-key settings land in <clips-dir>/[<session>/]isolation_params.json.
    That file is the extraction record for the write-up -- keep it.

Design constraints (from the project spec):
  * stdlib + numpy ONLY on the core path -- must run with no third-party installs.
    matplotlib is imported lazily and only when --plot is passed.
  * stdlib `wave` chokes on 32-bit float WAV (which phones / ffmpeg emit), so we
    parse the RIFF header ourselves and handle 8/16/24/32-bit PCM and 32/64-bit float.
  * robust detection: threshold = median + k*MAD of the energy envelope, rising-edge
    onsets with a debounce (min-gap) so the decaying tail is not re-detected.
  * BUG GUARD: the prominence floor references a high PERCENTILE of the envelope, not
    its max -- one loud bump/couch-thump would otherwise dominate max and suppress
    every genuine (quieter) keystroke.
"""

import argparse
import json
import os
import re
import struct
import sys
import wave

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CLIP_LEN = 14400            # fixed output clip length, samples (~300 ms @ 48 kHz)
PREROLL = int(0.15 * CLIP_LEN)   # samples kept before the onset so the attack transient is not clipped
FRAME_MS = 10.0             # RMS analysis window
HOP_MS = 5.0                # RMS analysis hop
DEFAULT_K = 5.0             # threshold = median + K * MAD
DEFAULT_MIN_GAP_MS = 350.0  # debounce between successive onsets
DEFAULT_EXPECTED = 25       # presses per key (for the OK/CHECK report)
MIN_PROM_FRAC = 0.2         # a detection's peak must exceed this fraction of the
                            # 99th-percentile envelope level (NOT of the max)
PEAK_PERCENTILE = 99.0      # "typical strong keystroke" reference level
LOW_COVERAGE = 0.75         # auto mode flags clips capturing less than this
                            # fraction of their press's energy


# ----------------------------------------------------------------------------
# WAV reading -- custom RIFF parser (stdlib `wave` cannot read float WAV)
# ----------------------------------------------------------------------------
def read_wav(path):
    """Return (mono float32 samples in [-1, 1], sample_rate).

    Handles PCM 8/16/24/32-bit and IEEE float 32/64-bit, mono or multi-channel
    (channels are averaged to mono), including WAVE_FORMAT_EXTENSIBLE.
    """
    with open(path, "rb") as f:
        buf = f.read()

    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    audio_format = num_channels = sample_rate = bits = None
    data_bytes = None

    # Walk the chunk list: each chunk is id(4) + size(4) + payload, padded to even.
    pos = 12
    n = len(buf)
    while pos + 8 <= n:
        chunk_id = buf[pos:pos + 4]
        (chunk_size,) = struct.unpack_from("<I", buf, pos + 4)
        # Clamp to the bytes actually present: some streamers/ffmpeg pipes write a
        # placeholder or bogus data-chunk size (e.g. 0xFFFFFFFF) that overruns the
        # buffer; without this the chunk walk would jump past the file or a bad
        # `data` size would drag in garbage. Reading only what's here is safe.
        chunk_size = min(chunk_size, n - (pos + 8))
        payload = buf[pos + 8: pos + 8 + chunk_size]

        if chunk_id == b"fmt ":
            (audio_format, num_channels, sample_rate,
             _byte_rate, _block_align, bits) = struct.unpack_from("<HHIIHH", payload, 0)
            # WAVE_FORMAT_EXTENSIBLE (0xFFFE): the real format is the first 2 bytes
            # of the SubFormat GUID, at offset 24 inside the fmt payload.
            if audio_format == 0xFFFE and len(payload) >= 26:
                (audio_format,) = struct.unpack_from("<H", payload, 24)
        elif chunk_id == b"data":
            data_bytes = payload

        pos += 8 + chunk_size + (chunk_size & 1)  # chunks are word-aligned

    if audio_format is None:
        raise ValueError(f"{path}: no fmt chunk")
    if data_bytes is None:
        raise ValueError(f"{path}: no data chunk")

    samples = _decode_samples(data_bytes, audio_format, bits, path)

    if num_channels > 1:
        usable = (len(samples) // num_channels) * num_channels
        samples = samples[:usable].reshape(-1, num_channels).mean(axis=1)

    return samples.astype(np.float32, copy=False), int(sample_rate)


def _decode_samples(data_bytes, audio_format, bits, path):
    """Decode raw PCM/float bytes to float32 in [-1, 1]."""
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
    """Write mono float samples as 16-bit PCM (stdlib `wave` writes this fine)."""
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
    """Short-time RMS energy envelope. Returns (env, hop)."""
    if len(x) < frame:
        return np.zeros(0, dtype=np.float32), hop
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    env = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return env.astype(np.float32), hop


def detect_onsets(env, hop, sample_rate, k, min_gap_ms):
    """Return (onset_samples list, threshold) from the energy envelope.

    Threshold uses robust stats (median + k*MAD). The prominence floor references
    a high percentile of the envelope -- NOT its max -- so a single loud transient
    cannot raise the bar above the genuine keystrokes.
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
    """Fixed-length clip starting PREROLL before the onset, zero-padded at the tail."""
    start = onset_sample - PREROLL
    start = max(0, min(start, max(0, len(x) - CLIP_LEN)))
    clip = x[start:start + CLIP_LEN]
    if len(clip) < CLIP_LEN:
        clip = np.concatenate([clip, np.zeros(CLIP_LEN - len(clip), dtype=x.dtype)])
    return clip


# ----------------------------------------------------------------------------
# Analysis / writing -- kept SEPARATE on purpose
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
        """Inter-onset intervals in ms -- the fastest way to spot a double-strike
        (a gap far below the rest) or a missed press (one roughly doubled)."""
        if len(self.onsets) < 2:
            return np.zeros(0)
        return np.diff(np.asarray(self.onsets, dtype=np.float64)) / self.sr * 1000.0

    def coverage(self, look_ms=800.0):
        """Fraction of each press's energy that its fixed-length clip captures.

        Clips are a fixed CLIP_LEN window (PREROLL before the onset), so this
        answers the two questions that window can get wrong:
          * a press that rings longer than the window -> coverage a bit under 1;
          * an onset that fired EARLY, on room noise just before the real
            strike -> coverage far under 1, because the keystroke itself lands
            at the very end of the clip or past it. That is a misaligned
            detection, not a length problem, and it silently poisons a class.

        Energy is noise-subtracted (the 20th percentile of the envelope is
        taken as the room floor) so a quiet key is not reported as truncated
        just because the room hiss continues after it.
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
    """Run the detector over already-loaded audio. Writes nothing."""
    frame = max(1, int(round(FRAME_MS / 1000.0 * sr)))
    hop = max(1, int(round(HOP_MS / 1000.0 * sr)))
    env, hop = rms_envelope(x, frame, hop)
    onsets, threshold = detect_onsets(env, hop, sr, k, min_gap_ms)
    return Analysis(x, sr, env, hop, onsets, threshold, k, min_gap_ms)


def analyze_file(wav_path, k, min_gap_ms):
    """Load a WAV and run the detector over it. Writes nothing."""
    x, sr = read_wav(wav_path)
    return analyze(x, sr, k, min_gap_ms)


def existing_clips(label, clips_dir):
    """Paths of clips already on disk for `label`, sorted.

    Matches only the exact `<label>_NN.wav` pattern this file writes, so a label
    like 'a' can never sweep up 'space_01.wav' or an unrelated file.
    """
    if not os.path.isdir(clips_dir):
        return []
    pat = re.compile(r"^" + re.escape(label) + r"_\d+\.wav$")
    return sorted(os.path.join(clips_dir, f)
                  for f in os.listdir(clips_dir) if pat.match(f))


def write_clips(analysis, label, clips_dir, replace=True):
    """Write one clip per detected onset. Returns the list of paths written.

    `replace` deletes this label's existing clips first. Without it a re-run
    that detects FEWER presses would leave the tail of the previous run behind
    (25 clips, re-run finds 22 -> _22.._24 survive as orphans from different
    parameters), quietly poisoning the pool with duplicates.
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
    """Detect and write clips for one raw WAV. Returns detected count."""
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
    """Split a sorted list into runs of adjacent grid values."""
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
    """Find detector settings giving exactly `expected` onsets for one file.

    Returns (k, min_gap_ms, onsets, threshold, k_window_width) or None if no
    setting on the grid produces the expected count.

    Two deliberate choices:
      * the smallest min-gap that works is preferred -- debounce is a blunt
        instrument, and a larger one can hide a genuine double press;
      * among the k values that hit the target, the MIDDLE of the widest
        contiguous run is chosen, not the first. An edge value sits one noisy
        press away from tipping to 24 or 26; the centre of a wide plateau is
        the stable choice, and the run's width is itself a confidence measure.
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


def auto_session(labels, raw_dir, clips_dir, expected, dry_run=False, session=None):
    """Tune every file to `expected` presses, then write its clips.

    Each file is searched independently -- no parameter is carried over from
    the previous one, because doing so is exactly how a key that needed k=23
    ends up extracted at k=8.

    `session` writes into clips_dir/<session>/ instead of clips_dir directly.
    That is what makes the pool GROW: re-extracting a label replaces only that
    session's clips, never another session's, and features.py scans clips_dir
    recursively so every session ends up in the pool. Extracting a new session
    into the same directory as an old one would delete the old clips instead.
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

        x, sr = read_wav(wav_path)
        frame = max(1, int(round(FRAME_MS / 1000.0 * sr)))
        hop = max(1, int(round(HOP_MS / 1000.0 * sr)))
        env, hop = rms_envelope(x, frame, hop)

        tuned = tune_file(env, hop, sr, expected)
        if tuned is None:
            counts = sorted({len(detect_onsets(env, hop, sr, float(k), 350.0)[0])
                             for k in AUTO_K_GRID})
            near = min(counts, key=lambda c: abs(c - expected))
            print(f"{label:>6} {'-':>7} {'-':>6} {'-':>5} {'-':>4} "
                  f"{'-':>8} {'-':>8} {'-':>4}  FAILED: never {expected} "
                  f"(closest {near})")
            failed.append((label, f"no setting yields {expected}; closest {near}"))
            continue

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
            flagged.append((label, note))

        print(f"{label:>6} {k:7g} {gap:6.0f} {kwin:5d} {len(onsets):4d} "
              f"{np.median(cov):7.0%} {cov.min():7.0%} {n_low:4d}  {note}")

        if not dry_run:
            written, removed = write_clips(a, label, clips_dir, replace=True)
            saved[label] = {
                "threshold_k": k,
                "min_gap_ms": gap,
                "expected": int(expected),
                "detected": len(onsets),
                "k_window_steps": kwin,
                "sample_rate": sr,
                "source": os.path.basename(wav_path),
                "coverage_median": round(float(np.median(cov)), 4),
                "coverage_min": round(float(cov.min()), 4),
                "clips_below_coverage": n_low,
            }
            ok.append((label, len(written), removed))

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


def _load_params(clips_dir):
    path = os.path.join(clips_dir, PARAMS_FILE)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            print(f"  (could not read {path}; starting fresh)")
    return {}


def _save_params(clips_dir, params):
    os.makedirs(clips_dir, exist_ok=True)
    path = os.path.join(clips_dir, PARAMS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
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
                        "session ADDS to the pool. features.py scans recursively, so every "
                        "session is included; without this a re-extract replaces the old clips.")
    return p


def main(argv=None):
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
        # Recordings live in per-session subfolders (converted_wavs/day_1/...),
        # and --raw-dir is deliberately NOT recursive: each run extracts exactly
        # one session. Point at the session rather than silently merging them.
        subs = sorted(d for d in os.listdir(args.raw_dir)
                      if os.path.isdir(os.path.join(args.raw_dir, d))) \
            if os.path.isdir(args.raw_dir) else []
        print(f"no .wav files directly in {args.raw_dir}", file=sys.stderr)
        if subs:
            print(f"  it contains session subfolder(s): {', '.join(subs)}", file=sys.stderr)
            print(f"  point --raw-dir at one, e.g.:\n"
                  f"    --raw-dir {os.path.join(args.raw_dir, subs[-1])} "
                  f"--session {subs[-1]}", file=sys.stderr)
        return 1

    if args.auto:
        return auto_session(labels, args.raw_dir, args.clips_dir,
                            args.expected, dry_run=args.dry_run, session=args.session)

    total = 0
    for label in labels:
        wav_path = os.path.join(args.raw_dir, f"{label}.wav")
        if not os.path.isfile(wav_path):
            print(f"[MISS ] {label:>6}: {wav_path} not found")
            continue
        total += process_file(wav_path, label, args.clips_dir,
                              args.threshold_k, args.min_gap_ms, args.expected, args.plot)

    print(f"done: {total} clips written to {args.clips_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
