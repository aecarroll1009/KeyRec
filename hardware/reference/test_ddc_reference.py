"""Tests for the DDC reference model.

This model is the reference the DDC RTL gets checked against, so it is the one
piece of the hardware flow that has nothing above it to catch its mistakes. A
wrong reference model does not fail -- it certifies a broken chip. These tests are
therefore aimed at the errors that would still LOOK right:

  * a mirrored spectrum (conjugate backwards) -- the magnitude spectrogram is
    unchanged, so every downstream plot looks fine,
  * K applied instead of 1/K -- a 4.3 dB level shift that reads as "gain",
  * the CORDIC silently failing to converge near pi/2,
  * truncation that rounds toward zero instead of toward -inf, which matches
    C but not Verilog's >>>,
  * the two mixer architectures quietly computing different functions, which
    would make the whole synthesis comparison meaningless.

Several tests deliberately construct the BUG and assert the check catches it
(`_mix_mirrored`, `test_..._would_be_caught`). A test that cannot fail is not
evidence, and sign-convention tests are unusually prone to being written in a
way that passes either way.

Run:  python hardware/reference/test_ddc_reference.py
"""

import json
import math
import os
import shutil
import tempfile

import numpy as np

# --- repo layout bootstrap --------------------------------------------------
# The reference model sits in hardware/reference/ next to this test; add its directory
# so the test runs from the repo root (the project-wide convention) as well as
# from here.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
# ----------------------------------------------------------------------------

import ddc_reference as G
from ddc_reference import DDCConfig, DDC, MIX_SEPARATE, MIX_FUSED, TRUNC, ROUND


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _peak_bin_hz(y, fs):
    """Frequency of the largest FFT bin, signed (negative for the lower half)."""
    spec = np.abs(np.fft.fft(y * np.hanning(len(y))))
    k = int(np.argmax(spec))
    if k > len(y) // 2:
        k -= len(y)
    return k * fs / len(y)


def _mix_mirrored(ddc, xi, xq, cos, sin):
    """The QUIET sign bug: correct mix, then Q negated.

    Worth being precise about, because the two plausible sign errors have very
    different signatures and only one of them is dangerous:

      * multiply by the NCO instead of its conjugate -> the tone moves to
        2*f_lo + delta and the lowpass deletes it. The output is empty. Nobody
        ships this; it fails on the first plot.

      * negate Q (a flipped subtraction in the Q equation) -> the output is
        conj(x * exp(-j*theta)), so the spectrum is MIRRORED about DC. Full
        amplitude, correct bandwidth, plausible-looking magnitude spectrogram,
        and every feature sitting on the wrong side of the carrier. This one
        ships.

    So this helper implements the second, and `test_sign_convention_is_not_
    mirrored` has to be able to catch it. The first case is covered separately
    by `test_wrong_rotation_direction_in_fused_would_upconvert`.
    """
    c = ddc.cfg
    pi_ = np.asarray(xi, np.int64) * cos + np.asarray(xq, np.int64) * sin
    pq_ = np.asarray(xi, np.int64) * sin - np.asarray(xq, np.int64) * cos
    sh = c.data_bits - 1
    return G.sat(G.shr(pi_, sh), c.data_bits), G.sat(G.shr(pq_, sh), c.data_bits)


# --------------------------------------------------------------------------
# CORDIC
# --------------------------------------------------------------------------


def test_cordic_gain_matches_the_published_constant():
    k = G.cordic_gain(40)
    assert abs(k - 1.6467602581210656) < 1e-12, f"K = {k!r}"
    # K grows monotonically with iterations and is already converged by 16.
    assert abs(G.cordic_gain(16) - k) < 1e-9
    assert G.cordic_gain(4) < G.cordic_gain(8) < k
    print(f"cordic gain: K = {k:.13f} OK")


def test_convergence_limit_covers_the_quadrant_residual():
    lim = G.cordic_convergence_limit(16)
    assert abs(lim - 1.7432561028942164) < 1e-12, lim
    # It keeps creeping up toward ~1.74328 as iterations are added, but the
    # 16-iteration value is the one this design actually gets.
    assert G.cordic_convergence_limit(64) > lim
    assert lim > math.pi / 2, "range reduction to [0, pi/2) would not converge"
    # The margin is only ~0.17 rad; if someone shortens the CORDIC it vanishes.
    assert G.cordic_convergence_limit(2) < math.pi / 2, (
        "a 2-iteration CORDIC should NOT cover pi/2 -- if it does, the limit "
        "calculation is wrong and the guard in DDCConfig is vacuous"
    )
    print(f"convergence limit: {lim:.6f} rad > pi/2 OK")


def test_cordic_reproduces_sin_cos_across_the_full_circle():
    """Sweep every quadrant, not just the first -- the fixups are per-quadrant."""
    cfg = DDCConfig()
    ddc = DDC(cfg)
    n = 4096
    phase = np.linspace(0, (1 << cfg.phase_bits) - 1, n).astype(np.int64)
    cos, sin = ddc.nco(phase)
    fs = G.full_scale(cfg.data_bits)
    theta = phase / (1 << cfg.phase_bits) * 2 * np.pi
    ec = np.abs(cos / fs - np.cos(theta)).max()
    es = np.abs(sin / fs - np.sin(theta)).max()
    assert ec < 2e-3, f"cos error {ec}"
    assert es < 2e-3, f"sin error {es}"
    # And all four quadrants were actually exercised.
    q, _ = G.quadrant_split(ddc.angle_word(phase), cfg.ang_bits)
    assert set(np.unique(q).tolist()) == {0, 1, 2, 3}, "sweep missed a quadrant"
    print(f"nco vs math: max cos err {ec:.2e}, sin err {es:.2e} OK")


def test_cordic_converges_at_the_range_reduction_boundary():
    """The worst case is a residual just under pi/2, the top of the input range."""
    cfg = DDCConfig()
    tbl = G.atan_table(cfg.n_iter, cfg.ang_bits)
    quarter = 1 << (cfg.ang_bits - 2)
    z0 = np.array([0, 1, quarter // 2, quarter - 2, quarter - 1], np.int64)
    x0 = np.full(z0.shape, int(round(G.full_scale(cfg.cordic_bits) / cfg.k_gain)), np.int64)
    x, y, z = G.cordic_rotate(x0, np.zeros_like(z0), z0, tbl, cfg.cordic_bits)
    # The residual cannot go below the LAST rotation step -- that is the finest
    # correction the CORDIC has left. Asserting a tighter bound than the
    # algorithm can reach would just be a test tuned to today's numbers, so the
    # criterion is the step size itself.
    last_step = int(tbl[-1])
    assert np.abs(z).max() <= last_step, (
        f"residual angle {np.abs(z).max()} LSB exceeds the final rotation step "
        f"{last_step} LSB -- not converged"
    )
    # Unit amplitude preserved (the 1/K seed did its job) at every angle.
    mag = np.hypot(x, y) / G.full_scale(cfg.cordic_bits)
    assert np.abs(mag - 1.0).max() < 3e-3, f"amplitude drift {np.abs(mag - 1.0).max()}"
    print(f"boundary convergence: |z| <= {np.abs(z).max()} of {last_step} LSB step, "
          f"|mag-1| < 3e-3 OK")


def test_nco_amplitude_is_unit_because_of_the_inv_k_seed():
    """Drop the 1/K seed and the amplitude should come out K, not 1.

    This is the K-scaling test with teeth: it asserts the correct value AND that
    the wrong seed produces a detectably different one.
    """
    cfg = DDCConfig()
    ddc = DDC(cfg)
    phase = np.linspace(0, (1 << cfg.phase_bits) - 1, 512).astype(np.int64)
    cos, sin = ddc.nco(phase)
    mag = np.hypot(cos, sin) / G.full_scale(cfg.data_bits)
    assert abs(mag.mean() - 1.0) < 2e-3, f"seeded NCO amplitude {mag.mean()}"

    # Unseeded, i.e. x0 not divided by K: amplitude comes out K instead of 1.
    # Seeded at HALF scale deliberately -- at full scale the K growth saturates
    # the CORDIC word and the measurement would read the clip level (~1.11)
    # rather than K. That saturation is real, and is exactly why the fused mixer
    # needs its extra bit; here it just has to be kept out of the way.
    tbl = ddc.atan
    q, rem = G.quadrant_split(ddc.angle_word(phase), cfg.ang_bits)
    half = G.full_scale(cfg.cordic_bits) // 2
    x0 = np.full(rem.shape, half, np.int64)
    x, y, _ = G.cordic_rotate(x0, np.zeros_like(rem), rem, tbl, cfg.cordic_bits)
    assert G.would_saturate(np.hypot(x, y).astype(np.int64), cfg.cordic_bits) == 0
    bad = np.hypot(x, y) / half
    assert abs(bad.mean() - cfg.k_gain) < 5e-3, (
        f"unseeded amplitude {bad.mean()} should be K={cfg.k_gain}"
    )
    print(f"inv-K seed: amplitude 1.000 seeded vs {bad.mean():.4f} unseeded OK")


def test_atan_table_exhaustion_is_rejected():
    """Iterations past what ang_bits can resolve are dead hardware, not accuracy."""
    try:
        DDCConfig(ang_bits=12, phase_trunc_bits=12, n_iter=24)
        raised = False
    except ValueError as e:
        raised = "resolve" in str(e)
    assert raised, "a CORDIC longer than its angle table allows was accepted"
    print("atan table exhaustion rejected OK")


# --------------------------------------------------------------------------
# fixed-point primitives
# --------------------------------------------------------------------------


def test_trunc_shift_matches_verilog_not_c():
    """Verilog's >>> floors; C's / truncates toward zero. They differ on negatives."""
    x = np.array([-9, -1, 1, 9], np.int64)
    got = G.shr(x, 1, TRUNC)
    assert got.tolist() == [-5, -1, 0, 4], got.tolist()
    assert (x // 2).tolist() == got.tolist(), "should match floor division"
    assert (x.astype(float) / 2).astype(np.int64).tolist() != got.tolist(), (
        "should NOT match round-toward-zero -- that would be the C semantics"
    )
    rnd = G.shr(x, 1, ROUND)
    assert rnd.tolist() == [-4, 0, 1, 5], rnd.tolist()
    print("shift semantics: trunc floors like >>>, round differs OK")


def test_saturation_clips_and_is_counted():
    v = np.array([-40000, -100, 100, 40000], np.int64)
    assert G.sat(v, 16).tolist() == [-32768, -100, 100, 32767]
    assert G.would_saturate(v, 16) == 2
    assert G.would_saturate(v, 32) == 0
    print("saturation clips and is counted OK")


def test_int64_overflow_is_caught_not_silent():
    try:
        G.assert_fits(np.array([1 << 62], np.int64), "probe")
        raised = False
    except OverflowError:
        raised = True
    assert raised, "an intermediate past int64 exactness went unreported"
    print("int64 overflow guard fires OK")


# --------------------------------------------------------------------------
# the sign convention -- the one this file exists for
# --------------------------------------------------------------------------


def test_sign_convention_is_not_mirrored():
    """A tone at f_lo + delta must land at +delta, not -delta.

    If the mixer multiplies by the NCO instead of its conjugate, the output
    spectrum is mirrored. Magnitude spectrograms are unaffected, so nothing
    downstream complains -- this test is the only thing standing between that
    bug and silicon.
    """
    cfg = DDCConfig()
    ddc = DDC(cfg)
    n = 8192
    delta = 40_000.0

    for sign in (+1.0, -1.0):
        want = sign * delta
        xi, xq = G.tone(n, cfg.fs_in, cfg.f_lo_actual + want, -6.0, cfg.data_bits)
        r = ddc.run(xi, xq)
        skip = 0  # fir_decimate is valid-only: no fill transient to trim
        y = (r["out_i"] + 1j * r["out_q"])[skip:]
        got = _peak_bin_hz(y, cfg.fs_out)
        assert abs(got - want) < cfg.fs_out / 100, (
            f"tone at f_lo{want:+.0f} Hz landed at {got:+.0f} Hz -- "
            f"spectrum is mirrored (conjugate backwards)"
        )
    print(f"sign convention: +/-{delta / 1e3:.0f} kHz land on the correct sides OK")


def test_a_mirrored_mixer_would_be_caught():
    """Prove the check above discriminates, by feeding it the actual bug.

    The mirrored output must have the SAME amplitude as the correct one -- that
    is what makes the bug survive review -- and land on the opposite side.
    """
    cfg = DDCConfig()
    ddc = DDC(cfg)
    n = 8192
    delta = 40_000.0
    xi, xq = G.tone(n, cfg.fs_in, cfg.f_lo_actual + delta, -6.0, cfg.data_bits)

    ph = ddc.phase(n)
    cos, sin = ddc.nco(ph)
    mi, mq = _mix_mirrored(ddc, xi, xq, cos, sin)
    yi, yq, _ = G.fir_decimate(
        mi, mq, ddc.coef, cfg.decim, cfg.coef_bits, cfg.acc_bits, cfg.out_bits, cfg.shift_mode
    )
    skip = 0  # fir_decimate is valid-only: no fill transient to trim
    bad = (yi + 1j * yq)[skip:]
    got = _peak_bin_hz(bad, cfg.fs_out)
    assert abs(got + delta) < cfg.fs_out / 100, (
        f"the deliberately-mirrored mixer landed at {got:+.0f} Hz; expected "
        f"{-delta:+.0f} Hz. The sign test cannot discriminate."
    )
    # Same amplitude as the correct path: the bug is invisible in magnitude.
    good = (ddc.run(xi, xq)["out_i"] + 1j * ddc.run(xi, xq)["out_q"])[skip:]
    ratio = np.abs(bad).mean() / np.abs(good).mean()
    assert 0.98 < ratio < 1.02, (
        f"mirrored output amplitude ratio {ratio:.3f} -- if it differed this "
        f"much the bug would be obvious, and the test would be too easy"
    )
    print(
        f"mirrored mixer lands at {got / 1e3:+.1f} kHz at {ratio:.3f}x amplitude "
        f"-- silent bug, test discriminates OK"
    )


def test_dc_lands_at_dc():
    """A tone exactly at the LO must come out at DC with a steady phase."""
    cfg = DDCConfig()
    ddc = DDC(cfg)
    n = 4096
    xi, xq = G.tone(n, cfg.fs_in, cfg.f_lo_actual, -6.0, cfg.data_bits)
    r = ddc.run(xi, xq)
    skip = 0  # fir_decimate is valid-only: no fill transient to trim
    y = (r["out_i"] + 1j * r["out_q"])[skip:]
    got = _peak_bin_hz(y, cfg.fs_out)
    assert abs(got) < cfg.fs_out / 200, f"LO tone landed at {got:+.1f} Hz, not DC"
    # Constant phase: the residual frequency is zero, not merely small in |X|.
    ph = np.unwrap(np.angle(y))
    drift = abs(ph[-1] - ph[0]) / (2 * np.pi) * cfg.fs_out / len(y)
    assert drift < 1.0, f"residual frequency {drift:.3f} Hz -- LO is not exact"
    print(f"LO tone -> DC, residual {drift:.4f} Hz OK")


# --------------------------------------------------------------------------
# K scaling
# --------------------------------------------------------------------------


def test_k_folding_lands_both_architectures_on_the_same_scale():
    """Separate removes K at the seed, fused in the coefficients -- same output."""
    sep = DDC(DDCConfig(mix_arch=MIX_SEPARATE))
    fus = DDC(DDCConfig(mix_arch=MIX_FUSED))
    assert not sep.fold_inv_k and fus.fold_inv_k

    dc_sep = sep.coef.sum() / G.full_scale(sep.cfg.coef_bits)
    dc_fus = fus.coef.sum() / G.full_scale(fus.cfg.coef_bits)
    assert abs(dc_sep - 1.0) < 1e-4, f"separate FIR DC gain {dc_sep}"
    assert abs(dc_fus - 1.0 / fus.cfg.k_gain) < 1e-4, f"fused FIR DC gain {dc_fus}"
    # The ratio IS K -- this is the assertion that catches K vs 1/K inverted.
    assert abs(dc_sep / dc_fus - fus.cfg.k_gain) < 1e-3, (
        f"coefficient ratio {dc_sep / dc_fus} should equal K={fus.cfg.k_gain}"
    )
    print(f"K folding: DC gains {dc_sep:.4f} / {dc_fus:.4f}, ratio = K OK")


def test_k_folded_the_wrong_way_would_be_4_3_db_hot():
    """Confirm the magnitude of the bug the previous test guards against."""
    cfg = DDCConfig(mix_arch=MIX_FUSED)
    h = G.firwin_lowpass(cfg.n_taps, cfg.fir_cutoff / (cfg.fs_in / 2))
    right = G.fir_taps_quantized(h, cfg.coef_bits, True, cfg.k_gain)
    wrong = G.fir_taps_quantized(h, cfg.coef_bits, False, cfg.k_gain)
    db = 20 * math.log10(wrong.sum() / right.sum())
    assert 4.2 < db < 4.4, f"expected ~4.34 dB error, got {db}"
    print(f"K folded the wrong way = {db:+.2f} dB level error OK")


def test_fused_mixer_carries_one_extra_bit():
    """The fused K growth needs headroom; that cost must be visible in the config."""
    assert DDCConfig(mix_arch=MIX_SEPARATE).mix_bits == 16
    assert DDCConfig(mix_arch=MIX_FUSED).mix_bits == 17
    # And a fused config with no CORDIC guard bits is rejected outright.
    try:
        DDCConfig(mix_arch=MIX_FUSED, cordic_bits=16, data_bits=16)
        raised = False
    except ValueError as e:
        raised = "headroom" in str(e)
    assert raised, "fused mixer accepted with no headroom for K"
    print("fused mixer: 17-bit output, zero-headroom config rejected OK")


# --------------------------------------------------------------------------
# the two architectures must compute the same function
# --------------------------------------------------------------------------


def test_separate_and_fused_agree():
    """The synthesis comparison is meaningless unless both do the same thing.

    Not bit-exact -- they quantize in different places -- but they must agree to
    well within a quantization step of each other, and both must track the ideal
    model closely. A structural error in either (wrong quadrant fixup, wrong
    rotation direction) blows this up immediately.
    """
    n = 8192
    cs = DDCConfig(mix_arch=MIX_SEPARATE)
    cf = DDCConfig(mix_arch=MIX_FUSED)
    ds, df = DDC(cs), DDC(cf)
    xi, xq = G.two_tone(n, cs)

    skip = 0  # fir_decimate is valid-only: no fill transient to trim
    fs_ = G.full_scale(cs.out_bits)
    ys = (ds.run(xi, xq)["out_i"] + 1j * ds.run(xi, xq)["out_q"]) / fs_
    yf = (df.run(xi, xq)["out_i"] + 1j * df.run(xi, xq)["out_q"]) / fs_
    ref = G.ddc_ideal(xi, xq, cs, ds.h_float)

    snr_s = G.snr_db(ref, ys, skip)
    snr_f = G.snr_db(ref, yf, skip)
    snr_sf = G.snr_db(ys, yf, skip)
    assert snr_s > 60, f"separate SNR only {snr_s:.1f} dB"
    assert snr_f > 60, f"fused SNR only {snr_f:.1f} dB"
    assert snr_sf > 55, f"the two architectures disagree at {snr_sf:.1f} dB"
    print(f"separate {snr_s:.1f} dB / fused {snr_f:.1f} dB / mutual {snr_sf:.1f} dB OK")


def test_wrong_rotation_direction_in_fused_is_caught():
    """Seeding z0=+rem instead of -rem must be detectable.

    The obvious guess -- "it up-converts, so the lowpass deletes it" -- is
    WRONG, and worth writing down because believing it would leave this test
    toothless. The quadrant pre-rotation is still applied in the correct
    (negative) direction; only the residual is inverted. So the output is the
    correctly down-converted signal multiplied by exp(+j*2*rem), and rem is a
    sawtooth ramp across each quadrant. That is phase distortion, not
    translation: the energy stays in band and the amplitude only drops ~30%,
    which no power check would flag.

    What collapses is fidelity. The correct path tracks the ideal model at
    ~79 dB; the sawtooth phase error drops that to single digits. SNR against
    the ideal model is the discriminator here, not signal power.
    """
    cfg = DDCConfig(mix_arch=MIX_FUSED)
    ddc = DDC(cfg)
    n = 8192
    delta = 40_000.0
    xi, xq = G.tone(n, cfg.fs_in, cfg.f_lo_actual + delta, -6.0, cfg.data_bits)

    ph = ddc.phase(n)
    q, rem = G.quadrant_split(ddc.angle_word(ph), cfg.ang_bits)
    ri, rq = G.prerotate_conj(q, xi, xq)
    g = cfg.cordic_bits - cfg.data_bits - 1
    # +rem: the bug.
    x, y, _ = G.cordic_rotate(
        ri << np.int64(g), rq << np.int64(g), rem, ddc.atan, cfg.cordic_bits, cfg.shift_mode
    )
    mi = G.sat(G.shr(x, g), cfg.mix_bits)
    mq = G.sat(G.shr(y, g), cfg.mix_bits)
    yi, yq, _ = G.fir_decimate(
        mi, mq, ddc.coef, cfg.decim, cfg.coef_bits, cfg.acc_bits, cfg.out_bits, cfg.shift_mode
    )
    fs_ = G.full_scale(cfg.out_bits)
    r = ddc.run(xi, xq)
    good = (r["out_i"] + 1j * r["out_q"]) / fs_
    bad = (yi + 1j * yq) / fs_
    ref = G.ddc_ideal(xi, xq, cfg, ddc.h_float)

    snr_good = G.snr_db(ref, good)
    snr_bad = G.snr_db(ref, bad)
    assert snr_good > 60, f"the correct fused path only scored {snr_good:.1f} dB"
    assert snr_bad < 20, (
        f"the inverted rotation scored {snr_bad:.1f} dB against the ideal model; "
        f"if a sign error can score that well the SNR check is not discriminating"
    )
    # And confirm the trap: power alone does NOT separate them.
    ratio = np.abs(good).mean() / np.abs(bad).mean()
    assert ratio < 2.0, (
        f"amplitude ratio {ratio:.2f} -- if power alone separated these, the "
        f"docstring's warning is wrong and worth revisiting"
    )
    print(
        f"inverted rotation: SNR {snr_bad:.1f} dB vs {snr_good:.1f} dB correct, "
        f"but only {ratio:.2f}x in amplitude -- caught by SNR, not power OK"
    )


# --------------------------------------------------------------------------
# filter and decimation
# --------------------------------------------------------------------------


def test_fir_has_unit_dc_gain_and_linear_phase():
    h = G.firwin_lowpass(63, 0.1)
    assert abs(h.sum() - 1.0) < 1e-12, h.sum()
    assert np.allclose(h, h[::-1]), "taps are not symmetric -- not linear phase"
    try:
        G.firwin_lowpass(64, 0.1)
        raised = False
    except ValueError:
        raised = True
    assert raised, "an even tap count was accepted"
    print("fir: unit DC gain, symmetric, even taps rejected OK")


def test_fir_rejects_out_of_band():
    """A tone past the cutoff must be attenuated, or decimation aliases it in."""
    cfg = DDCConfig()
    ddc = DDC(cfg)
    n = 8192
    skip = 0  # fir_decimate is valid-only: no fill transient to trim

    in_band = G.tone(n, cfg.fs_in, cfg.f_lo_actual + 20_000.0, -6.0, cfg.data_bits)
    # Past the cutoff AND past the decimated Nyquist, so it would fold back.
    out_band = G.tone(n, cfg.fs_in, cfg.f_lo_actual + 200_000.0, -6.0, cfg.data_bits)

    a = ddc.run(*in_band)
    b = ddc.run(*out_band)
    pa = np.abs(a["out_i"][skip:] + 1j * a["out_q"][skip:]).mean()
    pb = np.abs(b["out_i"][skip:] + 1j * b["out_q"][skip:]).mean()
    rej = 20 * math.log10(pa / max(pb, 1e-9))
    assert rej > 40, f"stopband rejection only {rej:.1f} dB -- aliases will get in"
    print(f"stopband rejection {rej:.1f} dB OK")


def test_decimation_takes_the_right_phase():
    """Off-by-one in the decimation offset is a delay, and delays hide.

    Checked against an independently computed convolution rather than against
    the model's own path.
    """
    cfg = DDCConfig(decim=4, n_taps=15)
    coef = np.arange(1, 16, dtype=np.int64)
    x = np.arange(100, dtype=np.int64)
    yi, _, _ = G.fir_decimate(x, np.zeros_like(x), coef, 4, 16, 40, 16, TRUNC)
    want = [
        G.shr(np.int64(sum(int(coef[k]) * int(x[m * 4 + 14 - k]) for k in range(15))), 15)
        for m in range((100 - 14) // 4 + 1)
    ]
    got = yi.tolist()[: len(want)]
    assert got == [int(w) for w in want], f"decimation phase wrong:\n{got}\n{want}"
    print(f"decimation phase matches direct convolution ({len(want)} samples) OK")


def test_aliasing_cutoff_is_rejected():
    try:
        DDCConfig(fir_cutoff=200_000.0, decim=8, fs_in=2_400_000.0)
        raised = False
    except ValueError as e:
        raised = "alias" in str(e)
    assert raised, "a cutoff above the decimated Nyquist was accepted"
    print("aliasing cutoff rejected OK")


# --------------------------------------------------------------------------
# quality of the whole thing
# --------------------------------------------------------------------------


def test_fixed_point_tracks_the_ideal_model():
    m = G.report(DDCConfig(), n=8192)
    assert m["ddc_snr_db"] > 70, f"DDC SNR {m['ddc_snr_db']:.1f} dB"
    assert m["ddc_sfdr_db"] > 50, f"DDC SFDR {m['ddc_sfdr_db']:.1f} dB"
    assert m["nco_snr_db"] > 85, f"NCO SNR {m['nco_snr_db']:.1f} dB"
    assert m["n_saturated"] == 0, f"{m['n_saturated']} samples saturated at -6 dBFS"
    assert m["ddc_enob"] > 11, f"ENOB {m['ddc_enob']:.2f}"
    print(
        f"quality: SNR {m['ddc_snr_db']:.1f} dB, SFDR {m['ddc_sfdr_db']:.1f} dB, "
        f"ENOB {m['ddc_enob']:.2f} OK"
    )


def test_narrower_datapath_is_measurably_worse():
    """Sanity: the SNR metric must respond to width, or it is measuring nothing."""
    wide = G.report(DDCConfig(), n=4096)["ddc_snr_db"]
    narrow = G.report(DDCConfig(data_bits=10, cordic_bits=14, out_bits=10), n=4096)[
        "ddc_snr_db"
    ]
    assert narrow < wide - 20, (
        f"a 10-bit datapath scored {narrow:.1f} dB against 16-bit's {wide:.1f} dB; "
        f"the metric is not tracking quantization"
    )
    print(f"width sensitivity: 16-bit {wide:.1f} dB vs 10-bit {narrow:.1f} dB OK")


def test_lo_quantization_is_reported_honestly():
    """A frequency the accumulator cannot hit must be reported, not rounded away."""
    # A 12-bit accumulator, with the angle path narrowed to match and the CORDIC
    # shortened to what a 12-bit angle table can actually resolve.
    cfg = DDCConfig(
        phase_bits=12, phase_trunc_bits=12, ang_bits=12, n_iter=9, f_lo=300_123.0
    )
    assert cfg.f_lo_actual != cfg.f_lo
    assert abs(cfg.f_lo_actual - cfg.f_lo) > 100, (
        "a 12-bit accumulator cannot place this LO to within 100 Hz; the model "
        "should be showing that error, not hiding it"
    )
    fine = DDCConfig(phase_bits=32, f_lo=300_123.0)
    assert abs(fine.f_lo_actual - fine.f_lo) < 1e-3
    print(
        f"LO quantization: 12-bit err {cfg.f_lo_actual - cfg.f_lo:+.1f} Hz, "
        f"32-bit err {fine.f_lo_actual - fine.f_lo:+.2e} Hz OK"
    )


# --------------------------------------------------------------------------
# test-vector emission
# --------------------------------------------------------------------------


def test_m_and_n_are_independent_knobs():
    """M sets frequency resolution, N sets spectral purity. Not the same number."""
    c = DDCConfig()
    assert c.phase_bits == 32 and c.phase_trunc_bits == 14

    # M controls how exactly an LO can be placed...
    coarse = DDCConfig(phase_bits=16, phase_trunc_bits=14, ang_bits=14, n_iter=13,
                       f_lo=300_123.0)
    fine = DDCConfig(phase_bits=32, f_lo=300_123.0)
    assert abs(coarse.f_lo_actual - 300_123.0) > abs(fine.f_lo_actual - 300_123.0)

    # ...while N controls the spur bound, independently of M.
    assert DDCConfig(phase_trunc_bits=10).phase_trunc_sfdr_bound_db < (
        DDCConfig(phase_trunc_bits=16).phase_trunc_sfdr_bound_db
    )
    assert abs(c.phase_trunc_sfdr_bound_db - 84.28) < 0.01
    # N cannot exceed M, and ang_bits cannot be narrower than N.
    for kw in (dict(phase_bits=12, phase_trunc_bits=14),
               dict(phase_trunc_bits=14, ang_bits=12)):
        try:
            DDCConfig(**kw)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"invalid width combination accepted: {kw}"
    print("M and N are independent, both validated OK")


def test_phase_truncation_only_bites_when_the_fcw_exercises_it():
    """The binary-fraction trap, asserted rather than trusted.

    The default LO discards only zero bits, so N costs nothing there. An LO that
    discards nonzero bits pays the ~6.02*N spur penalty. If these two ever
    measure the same, the truncation is not being modelled.
    """
    cfg = DDCConfig()
    ddc = DDC(cfg)
    n = 8192
    assert cfg.phase_trunc_residue == 0, (
        "the default LO is expected to be a binary fraction of fs; if this "
        "changed, the trap this test documents no longer applies here"
    )

    fs_ = G.full_scale(cfg.data_bits)
    benign = ddc.nco(ddc.phase(n))
    hard_inc = (cfg.phase_inc + 0x0002AAAB) & ((1 << cfg.phase_bits) - 1)
    assert hard_inc & ((1 << (cfg.phase_bits - cfg.phase_trunc_bits)) - 1) != 0
    hard = ddc.nco(ddc.phase(n, inc=hard_inc))

    s_benign = G.sfdr_db((benign[0] + 1j * benign[1]) / fs_)
    s_hard = G.sfdr_db((hard[0] + 1j * hard[1]) / fs_)
    assert s_benign > s_hard + 5, (
        f"benign LO {s_benign:.1f} dB vs truncating LO {s_hard:.1f} dB -- phase "
        f"truncation is not being modelled, or the FCWs do not differ below N"
    )
    # And the truncating case should sit near the 6.02*N bound, not far past it.
    assert s_hard < cfg.phase_trunc_sfdr_bound_db + 3, (
        f"{s_hard:.1f} dB beats the {cfg.phase_trunc_sfdr_bound_db:.1f} dBc bound "
        f"for N={cfg.phase_trunc_bits}; the truncation is being skipped"
    )
    print(
        f"phase truncation: {s_benign:.1f} dB benign vs {s_hard:.1f} dB truncating "
        f"(bound {cfg.phase_trunc_sfdr_bound_db:.1f}) OK"
    )


def test_widening_n_improves_spectral_purity():
    """N is the knob that moves NCO purity -- confirm it actually does."""
    n = 8192
    got = {}
    for N in (10, 14, 18):
        cfg = DDCConfig(phase_trunc_bits=N, ang_bits=max(N, 18))
        ddc = DDC(cfg)
        inc = (cfg.phase_inc + 0x0002AAAB) & ((1 << cfg.phase_bits) - 1)
        c_, s_ = ddc.nco(ddc.phase(n, inc=inc))
        got[N] = G.sfdr_db((c_ + 1j * s_) / G.full_scale(cfg.data_bits))
    assert got[10] < got[14] < got[18], f"SFDR did not improve with N: {got}"
    assert got[14] - got[10] > 15, (
        f"4 more phase bits bought only {got[14] - got[10]:.1f} dB; the classic "
        f"bound predicts ~24"
    )
    print("N sweep: " + ", ".join(f"N={k} -> {v:.1f} dB" for k, v in got.items()) + " OK")


def test_angle_word_zero_pads_rather_than_requantising():
    """N-bit truncation then zero-pad: the low ang_bits-N bits must be zero."""
    cfg = DDCConfig()
    ddc = DDC(cfg)
    pad = cfg.ang_bits - cfg.phase_trunc_bits
    w = ddc.angle_word(ddc.phase(512, inc=(cfg.phase_inc + 12345)))
    assert pad > 0
    assert np.all((w & ((1 << pad) - 1)) == 0), "zero-padding is not zero"
    assert int(w.max()) < (1 << cfg.ang_bits)
    # Distinct truncated phases must stay distinct -- padding adds no collisions.
    assert len(np.unique(w)) == len(np.unique(G.shr(
        ddc.phase(512, inc=(cfg.phase_inc + 12345)), cfg.phase_bits - cfg.phase_trunc_bits
    )))
    print(f"angle word: N={cfg.phase_trunc_bits} truncated, zero-padded by {pad} OK")


def test_vectors_round_trip():
    """What the testbench reads back must be exactly what the model produced."""
    d = tempfile.mkdtemp(prefix="ddcvec_")
    try:
        cfg = DDCConfig()
        ddc = DDC(cfg)
        man = G.emit_vectors(ddc, d, n=1024)

        for name in man["files"]:
            assert os.path.isfile(os.path.join(d, name)), f"missing {name}"

        xi, xq = G.two_tone(1024, cfg)
        r = ddc.run(xi, xq)

        def read(name, bits):
            with open(os.path.join(d, name)) as f:
                vals = [int(line, 16) for line in f if line.strip()]
            return np.array(
                [v - (1 << bits) if v >= (1 << (bits - 1)) else v for v in vals], np.int64
            )

        for name, arr, bits in (
            ("stim_i.hex", r["stim_i"], cfg.data_bits),
            ("nco_cos.hex", r["cos"], cfg.data_bits),
            ("mix_q.hex", r["mix_q"], cfg.mix_bits),
            ("out_i.hex", r["out_i"], cfg.out_bits),
            ("fir_coef.hex", ddc.coef, cfg.coef_bits),
        ):
            back = read(name, bits)
            assert np.array_equal(back, arr), f"{name} did not round-trip"

        assert man["derived"]["n_output_samples"] == len(r["out_i"])
        assert man["derived"]["phase_inc"] == cfg.phase_inc
        json.load(open(os.path.join(d, "manifest.json")))

        svh = open(os.path.join(d, "ddc_params.svh")).read()
        assert f"PHASE_INC = {cfg.phase_bits}'h{cfg.phase_inc:08x}" in svh
        assert f"localparam int N_ITER      = {cfg.n_iter};" in svh
        assert "`endif" in svh
        print(f"vectors round-trip ({len(man['files'])} files) OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_negative_values_encode_as_twos_complement():
    assert G._hex_lines(np.array([-1], np.int64), 16) == ["ffff"]
    assert G._hex_lines(np.array([-32768], np.int64), 16) == ["8000"]
    assert G._hex_lines(np.array([32767], np.int64), 16) == ["7fff"]
    assert G._hex_lines(np.array([-1], np.int64), 17) == ["1ffff"]
    print("two's-complement hex encoding OK")


def main():
    test_cordic_gain_matches_the_published_constant()
    test_convergence_limit_covers_the_quadrant_residual()
    test_cordic_reproduces_sin_cos_across_the_full_circle()
    test_cordic_converges_at_the_range_reduction_boundary()
    test_nco_amplitude_is_unit_because_of_the_inv_k_seed()
    test_atan_table_exhaustion_is_rejected()

    test_trunc_shift_matches_verilog_not_c()
    test_saturation_clips_and_is_counted()
    test_int64_overflow_is_caught_not_silent()

    test_sign_convention_is_not_mirrored()
    test_a_mirrored_mixer_would_be_caught()
    test_dc_lands_at_dc()

    test_k_folding_lands_both_architectures_on_the_same_scale()
    test_k_folded_the_wrong_way_would_be_4_3_db_hot()
    test_fused_mixer_carries_one_extra_bit()

    test_separate_and_fused_agree()
    test_wrong_rotation_direction_in_fused_is_caught()

    test_fir_has_unit_dc_gain_and_linear_phase()
    test_fir_rejects_out_of_band()
    test_decimation_takes_the_right_phase()
    test_aliasing_cutoff_is_rejected()

    test_fixed_point_tracks_the_ideal_model()
    test_narrower_datapath_is_measurably_worse()
    test_lo_quantization_is_reported_honestly()
    test_m_and_n_are_independent_knobs()
    test_phase_truncation_only_bites_when_the_fcw_exercises_it()
    test_widening_n_improves_spectral_purity()
    test_angle_word_zero_pads_rather_than_requantising()

    test_vectors_round_trip()
    test_negative_values_encode_as_twos_complement()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
