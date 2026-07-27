"""Tiny synthetic-data test for isolate_keystrokes.py.

Proves the core path works before we move on to File 2:
  * the custom RIFF parser reads a 32-bit FLOAT WAV (stdlib `wave` cannot),
  * detection finds exactly the injected number of keystrokes,
  * a robustness case: one loud "couch bump" does NOT suppress genuine keystrokes
    (the percentile prominence floor, not max),
  * clips are written at exactly CLIP_LEN samples,
  * the 16-bit PCM path also parses/detects.

Run:  python test_isolate_keystrokes.py
"""

import os
import struct
import tempfile
import wave

import numpy as np

import isolate_keystrokes as ik


def synth_keystrokes(sr, n, gap_s=1.0, burst_ms=30.0, amp=0.6, seed=0, bumps=()):
    """Build a signal: n decaying-noise bursts spaced gap_s apart over quiet noise.

    `bumps` is a list of (time_s, amplitude) extra transients (e.g. a couch thump)
    used to test that a loud outlier does not break detection.
    """
    rng = np.random.default_rng(seed)
    lead = 0.3
    total = int((lead + n * gap_s + 0.5) * sr)
    x = (rng.standard_normal(total) * 0.004).astype(np.float32)  # quiet background

    burst_n = int(burst_ms / 1000.0 * sr)
    decay = np.exp(-np.linspace(0, 6, burst_n)).astype(np.float32)

    def add_burst(center_s, a):
        start = int(center_s * sr)
        b = (rng.standard_normal(burst_n) * decay * a).astype(np.float32)
        end = min(start + burst_n, len(x))
        x[start:end] += b[: end - start]

    for i in range(n):
        add_burst(lead + i * gap_s, amp)
    for t, a in bumps:
        add_burst(t, a)
    return x


def write_float_wav(path, x, sr):
    """Write a 32-bit IEEE-float WAV by hand (stdlib `wave` cannot)."""
    data = x.astype("<f4").tobytes()
    fmt, ch, bits = 3, 1, 32
    byte_rate = sr * ch * bits // 8
    block_align = ch * bits // 8
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, fmt, ch, sr, byte_rate, block_align, bits)
    hdr += b"data" + struct.pack("<I", len(data))
    with open(path, "wb") as f:
        f.write(hdr + data)


def write_pcm16_wav(path, x, sr):
    x16 = np.round(np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x16.tobytes())


def main():
    sr = 48000
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        clips = os.path.join(tmp, "clips")
        os.makedirs(raw)

        # --- Case 1: 32-bit float WAV, 25 keystrokes + one loud couch bump ------
        x = synth_keystrokes(sr, 25, bumps=[(0.15, 3.0)])  # bump ~5x louder, off-grid
        write_float_wav(os.path.join(raw, "a.wav"), x, sr)

        # sanity: our parser round-trips the float data
        got, got_sr = ik.read_wav(os.path.join(raw, "a.wav"))
        assert got_sr == sr, f"sr mismatch: {got_sr}"
        assert len(got) == len(x), f"length mismatch: {len(got)} vs {len(x)}"
        assert np.max(np.abs(got - x)) < 1e-6, "float WAV decode differs from source"
        print("float WAV parse: OK")

        n_a = ik.process_file(os.path.join(raw, "a.wav"), "a", clips,
                              ik.DEFAULT_K, ik.DEFAULT_MIN_GAP_MS, 25, do_plot=False)
        # 25 real keystrokes; the single bump may add at most 1 detection.
        assert 25 <= n_a <= 26, f"expected 25(-26 w/ bump) detections, got {n_a}"
        print(f"detection w/ loud bump present: OK ({n_a} found, bump did not suppress)")

        # clips exist and are exactly CLIP_LEN samples
        clip_files = sorted(f for f in os.listdir(clips) if f.startswith("a_"))
        assert len(clip_files) == n_a, f"clip count {len(clip_files)} != detections {n_a}"
        c, c_sr = ik.read_wav(os.path.join(clips, clip_files[0]))
        assert len(c) == ik.CLIP_LEN, f"clip length {len(c)} != {ik.CLIP_LEN}"
        assert c_sr == sr
        print(f"clip write: OK ({len(clip_files)} clips of {ik.CLIP_LEN} samples)")

        # --- Case 2: 16-bit PCM WAV, 10 keystrokes ----------------------------
        x2 = synth_keystrokes(sr, 10, seed=1)
        write_pcm16_wav(os.path.join(raw, "b.wav"), x2, sr)
        n_b = ik.process_file(os.path.join(raw, "b.wav"), "b", clips,
                              ik.DEFAULT_K, ik.DEFAULT_MIN_GAP_MS, 10, do_plot=False)
        assert n_b == 10, f"expected 10 detections, got {n_b}"
        print("16-bit PCM parse + detection: OK (10 found)")

        # --- Case 3: lying data-chunk size must not overrun the buffer --------
        # Some streamers/ffmpeg pipes write a placeholder data size like
        # 0xFFFFFFFF. The parser must clamp to the bytes actually present and
        # still decode the real audio rather than jumping past the file.
        x3 = synth_keystrokes(sr, 5, seed=2)
        data = x3.astype("<f4").tobytes()
        byte_rate = sr * 1 * 32 // 8
        hdr = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"       # bogus RIFF size too
        hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, sr, byte_rate, 4, 32)
        hdr += b"data" + struct.pack("<I", 0xFFFFFFFF)                # bogus data size
        bad_path = os.path.join(raw, "c.wav")
        with open(bad_path, "wb") as f:
            f.write(hdr + data)
        got3, got3_sr = ik.read_wav(bad_path)
        assert got3_sr == sr
        assert len(got3) == len(x3), f"clamp read wrong length: {len(got3)} vs {len(x3)}"
        assert np.max(np.abs(got3 - x3)) < 1e-6, "clamped decode differs from source"
        print("lying data-chunk size clamp: OK (decoded real audio, no overrun)")

    test_analysis_writes_nothing()
    test_accept_replaces_instead_of_accumulating()
    test_existing_clips_does_not_match_other_labels()
    test_coverage_flags_a_misaligned_onset()

    print("\nALL TESTS PASSED")


def test_analysis_writes_nothing():
    """Review mode re-runs detection on every parameter tweak. If analysis
    touched the disk, tuning a key would litter it with exactly the redundant
    clips the mode exists to avoid."""
    sr = 48000
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        clips = os.path.join(tmp, "clips")
        os.makedirs(raw)
        path = os.path.join(raw, "a.wav")
        write_float_wav(path, synth_keystrokes(sr, 12, seed=3), sr)

        for k in (3.0, 5.0, 7.0, 9.0):
            a = ik.analyze_file(path, k, ik.DEFAULT_MIN_GAP_MS)
            assert a.n >= 0 and a.sr == sr
        assert not os.path.exists(clips), "analyze created the clips dir"
        assert sorted(os.listdir(raw)) == ["a.wav"], "analyze wrote into the raw dir"
        print("analyze: repeated detection passes wrote nothing to disk OK")


def test_accept_replaces_instead_of_accumulating():
    """Re-accepting a key with stricter parameters must leave ONLY the new
    clips. Without the replace step the tail of the previous, larger run
    survives as orphans produced under different parameters."""
    sr = 48000
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        clips = os.path.join(tmp, "clips")
        os.makedirs(raw)
        path = os.path.join(raw, "a.wav")
        write_float_wav(path, synth_keystrokes(sr, 20, seed=4), sr)

        loose = ik.analyze_file(path, 3.0, ik.DEFAULT_MIN_GAP_MS)
        written, removed = ik.write_clips(loose, "a", clips)
        assert removed == 0 and len(written) == loose.n
        first = len(ik.existing_clips("a", clips))

        # A pass that detects strictly fewer presses.
        strict = ik.analyze_file(path, 3.0, ik.DEFAULT_MIN_GAP_MS)
        strict.onsets = strict.onsets[: max(1, loose.n - 5)]
        written2, removed2 = ik.write_clips(strict, "a", clips)
        assert removed2 == first, f"replace removed {removed2}, expected {first}"

        on_disk = ik.existing_clips("a", clips)
        assert len(on_disk) == len(written2), (
            f"{len(on_disk)} clips on disk after re-accept, expected {len(written2)} "
            "-- orphans from the previous parameters survived")
        assert sorted(on_disk) == sorted(written2)
        print(f"accept: re-accepting replaced {removed2} clips, left exactly "
              f"{len(on_disk)} (no orphans) OK")


def test_coverage_flags_a_misaligned_onset():
    """Coverage must be ~1 for a well-centred press and clearly low for one
    whose onset fired early -- that is the case that silently poisons a class,
    because the clip is mostly room noise with the keystroke at its tail."""
    sr = 48000
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        path = os.path.join(raw, "a.wav")
        write_float_wav(path, synth_keystrokes(sr, 8, seed=5), sr)

        a = ik.analyze_file(path, ik.DEFAULT_K, ik.DEFAULT_MIN_GAP_MS)
        good = a.coverage()
        assert len(good) == a.n
        assert good.min() > 0.9, f"well-aligned presses scored low: {good.min():.2f}"

        # Shift every onset EARLIER by ~250 ms: the clip now starts in silence
        # and the keystroke lands at its very end.
        early = ik.analyze_file(path, ik.DEFAULT_K, ik.DEFAULT_MIN_GAP_MS)
        early.onsets = [max(0, o - int(0.25 * sr)) for o in early.onsets]
        shifted = early.coverage()
        assert shifted.max() < good.min(), (
            f"misaligned onsets ({shifted.max():.2f}) did not score below "
            f"aligned ones ({good.min():.2f})")
        assert int((shifted < ik.LOW_COVERAGE).sum()) > 0, \
            "no misaligned clip was flagged as low coverage"
        print(f"coverage: aligned presses >={good.min():.0%}, misaligned flagged "
              f"(worst {shifted.min():.0%}) OK")


def test_existing_clips_does_not_match_other_labels():
    """The replace step deletes files. It must match only <label>_NN.wav, or
    accepting one key could delete another key's clips."""
    with tempfile.TemporaryDirectory() as tmp:
        clips = os.path.join(tmp, "clips")
        os.makedirs(clips)
        for name in ("a_00.wav", "a_01.wav", "space_00.wav", "ab_00.wav",
                     "a_notanumber.wav", "a.wav"):
            with open(os.path.join(clips, name), "wb") as f:
                f.write(b"")

        found = [os.path.basename(p) for p in ik.existing_clips("a", clips)]
        assert found == ["a_00.wav", "a_01.wav"], f"matched the wrong files: {found}"
        assert [os.path.basename(p) for p in ik.existing_clips("space", clips)] == ["space_00.wav"]
        print("existing_clips: matches only <label>_NN.wav, not other labels OK")


if __name__ == "__main__":
    main()
