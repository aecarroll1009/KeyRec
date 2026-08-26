"""
Numpy reference model of the EM channel's DDC front-end (the "floor" blocks).

    EM samples --> [ Mixer ] --> [ Decimating FIR ] --> baseband IQ
                       ^
                   [  NCO  ]  phase accumulator -> rotation-mode CORDIC

This is the reference the SystemVerilog in `hardware/` is checked against.
Per `hardware/DDC_FRONTEND_SCOPE.md`: the reference model comes first because it
catches sign-convention and K-scaling bugs in minutes, where finding the same
bug in RTL simulation costs a day.


WHY THERE ARE TWO MODELS IN HERE
--------------------------------
`ddc_ideal()`  -- float64, exact. Says what the answer SHOULD be.
`DDC.run()`    -- bit-exact fixed point. Says what the RTL MUST produce.

They serve different jobs and you need both:

  * RTL is checked against the FIXED-POINT model, bit for bit. Any mismatch is
    an RTL bug, full stop. There is no "close enough" tolerance to hide behind.
  * The fixed-point model is checked against the IDEAL model in SNR/SFDR. That
    tells you whether the chosen widths are good enough, which is an
    architecture question, not a correctness question.

A float-only reference model would let a truncation-vs-rounding disagreement, a
saturation that never fires in the model but does in RTL, or a coefficient
scaled by K instead of 1/K all pass silently. Those are exactly the bugs this
file exists to catch.


THE TWO SIGN CONVENTIONS THAT WILL BITE
---------------------------------------
1. Down-conversion multiplies by exp(-j*theta), i.e. by the CONJUGATE of the
   NCO output. There are two ways to get this wrong and they fail very
   differently:

     * multiplying by the NCO rather than its conjugate UP-converts: the tone
       moves to 2*f_lo + delta and the decimating lowpass deletes it. Loud,
       obvious, caught on the first plot.

     * negating Q -- one flipped subtraction in the Q equation -- yields
       conj(x * exp(-j*theta)), which MIRRORS the spectrum about DC. Same
       amplitude, same bandwidth, plausible magnitude spectrogram, and every
       feature on the wrong side of the carrier. This is the one that ships,
       and it stays invisible right up until you care which sideband something
       came from.

   `test_ddc_reference.py::test_sign_convention_is_not_mirrored` pins both down,
   and `test_a_mirrored_mixer_would_be_caught` builds the quiet bug on purpose
   to prove that test can actually see it.

2. CORDIC rotation mode drives z to zero, so to rotate a vector BY -theta you
   seed z0 = -theta, not +theta. In the fused mixer that is the difference
   between down-converting and up-converting.


K-SCALING: WHERE THE 1.6467 GOES, AND WHY IT DIFFERS PER ARCHITECTURE
---------------------------------------------------------------------
The rotation-mode CORDIC has gain K = prod sqrt(1 + 2^-2i) ~= 1.6467. It has to
be removed somewhere, and the two mixer architectures in the comparison study
remove it in DIFFERENT places:

  MIX_SEPARATE : the CORDIC only makes sin/cos. Seed it with x0 = 1/K and the
                 output comes out unit-amplitude for free -- a constant folded
                 into a ROM/immediate, costing nothing. The FIR coefficients are
                 then plain.

  MIX_FUSED    : the SIGNAL itself is the rotated vector, so it comes out scaled
                 by K. There is no free seed to hide it in, so the FIR
                 coefficients carry the 1/K instead (this is the "gain K~1.647
                 folded" note in the datapath diagram).

Both paths therefore land at the SAME output scale, which is the only reason
their areas and SNRs are comparable at all. If you ever change one, change the
other: a fused build with plain coefficients is 4.3 dB hot and will look like it
has better SNR than it does. `DDC` sets this automatically from `mix_arch`;
`fir_taps_quantized()` takes `fold_inv_k` explicitly so the choice stays visible.


THE NCO'S TWO WIDTHS: M AND N
-----------------------------
M (`phase_bits`) is the accumulator width and sets FREQUENCY RESOLUTION:
fs/2^M. At M=32 and 2.4 MS/s that is 0.56 mHz, so any LO you ask for is placed
essentially exactly.

N (`phase_trunc_bits`) is how many of those bits reach the angle path, and sets
SPECTRAL PURITY. Truncating M down to N throws information away every sample,
and the error is periodic, so it appears as discrete spurs rather than noise.
The classic worst-case bound is ~6.02*N dBc -- 84 dB at N=14 -- and nothing
downstream buys past it: not CORDIC iterations, not datapath width.

These are independent knobs and it is worth keeping them that way. `ang_bits`
is a third, separate number: the CORDIC's internal angle register, which is
kept wider than N (the truncated phase is zero-padded into it) so the rotation
converges on the truncated angle rather than quantizing it a second time. Using
N as the z width too would cap useful iterations at 13 for N=14, because past
that the atan table entries round to zero and the stages do nothing.

MEASURING N HONESTLY -- THE BINARY-FRACTION TRAP
------------------------------------------------
Phase truncation only produces error when the FCW has nonzero bits below the
truncation point. If the LO is a binary fraction of the sample rate, those bits
are always zero and there is NO truncation error at all, at any N.

The default config is exactly that case: 300 kHz at 2.4 MS/s gives PHASE_INC =
0x20000000, whose low 18 bits are zero. Measured there, the NCO scores 93 dB
SFDR and N=14 looks free. Measured 98 Hz away, it scores 82 dB -- right at the
6.02*N bound. The second number is the real one.

So `report()` measures both and labels which case the configured LO is, and
`cfg.phase_trunc_residue` is the flag: zero means the measurement is flattering
and must not be quoted as the NCO's spectral purity.


CONVERGENCE / RANGE REDUCTION
-----------------------------
A rotation-mode CORDIC only converges for |z| <= sum(atan(2^-i)) ~= 1.7433 rad,
which is just over pi/2 and well under pi. So the full circle is handled by
splitting the phase word into a 2-bit quadrant and a residual in [0, pi/2), then
fixing up the quadrant with negates and swaps -- muxes in RTL, no arithmetic.
`quadrant_split()` / `apply_quadrant_sincos()` / `prerotate_conj()`.

Note the residual can reach pi/2 = 1.5708, which is inside the 1.7433 bound but
not by a lot. That margin is real and is what `test_cordic_converges_at_the_
range_reduction_boundary` guards.


USAGE
-----
    python hardware/reference/ddc_reference.py --report
    python hardware/reference/ddc_reference.py --report --mix-arch fused
    python hardware/reference/ddc_reference.py --compare-arch
    python hardware/reference/ddc_reference.py --emit-vectors build/ddc_vectors

`--emit-vectors` writes flat hex files plus a `ddc_params.svh`, so the testbench
reads the same numbers this model produced rather than re-deriving them (a
testbench that re-derives the reference is testing the testbench).

Write them into `build/` -- they are regenerable from one command, and the
repo's rule is to commit only what cannot be recreated. Regenerate them whenever
the config changes; `manifest.json` records the config each set was made under,
so a stale set is identifiable rather than merely wrong.

Depends on numpy only -- deliberately. No scipy, so this can run wherever the
simulator runs. The window-method FIR design is ten lines and is inlined below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict, replace

import numpy as np

# Full circle in CORDIC angle units is 2**ang_bits, so pi/2 is 2**(ang_bits-2).
# Keeping the circle a power of two is what makes the quadrant split a bit slice.

MIX_SEPARATE = "separate"
MIX_FUSED = "fused"

TRUNC = "trunc"
ROUND = "round"


# --------------------------------------------------------------------------
# fixed-point primitives
#
# Everything is a plain signed integer. A DATA_BITS word represents a value in
# [-1, 1) with the binary point just below the sign bit, so full scale is
# 2**(bits-1). Python/numpy int64 holds all intermediates exactly; assert_fits()
# is the guard that we never silently exceed it.
# --------------------------------------------------------------------------


def full_scale(bits: int) -> int:
    """Integer value representing 1.0 in a `bits`-wide signed word."""
    return 1 << (bits - 1)


def sat(x, bits: int):
    """Saturate to a `bits`-wide two's-complement range.

    Saturation, not wrapping. A DDC that wraps on overflow turns a loud burst
    into full-scale noise across the whole band, which is far worse than
    clipping it -- and keystroke EM capture is exactly the bursty case.
    """
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def would_saturate(x, bits: int) -> int:
    """How many elements sat() would clip. Reported, never silently ignored."""
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    a = np.asarray(x, dtype=np.int64)
    return int(np.count_nonzero((a < lo) | (a > hi)))


def assert_fits(x, name: str, bits: int = 62) -> None:
    """Guard against silent int64 wrap in an intermediate."""
    a = np.asarray(x, dtype=np.int64)
    if a.size and int(np.max(np.abs(a))) >= (1 << bits):
        raise OverflowError(
            f"{name} exceeded {bits} bits -- int64 intermediates are no longer "
            f"exact; reduce widths or split the accumulation"
        )


def shr(x, s: int, mode: str = TRUNC):
    """Arithmetic right shift by `s`, matching what the RTL does.

    TRUNC is Verilog's `>>>` on a signed value: floor, biased toward -inf.
    ROUND is round-half-up, one adder more expensive.

    Which one you pick matters. Truncation in the CORDIC iteration is a DC
    offset that accumulates over iterations; most published CORDIC RTL truncates
    and eats it. The model defaults to TRUNC so it matches the cheap RTL, and
    `--shift-mode round` lets you measure what the adders would buy.
    """
    a = np.asarray(x, dtype=np.int64)
    if s <= 0:
        return a
    if mode == ROUND:
        return (a + (np.int64(1) << np.int64(s - 1))) >> np.int64(s)
    if mode == TRUNC:
        return a >> np.int64(s)
    raise ValueError(f"unknown shift mode {mode!r}; expected {TRUNC!r} or {ROUND!r}")


# --------------------------------------------------------------------------
# CORDIC
# --------------------------------------------------------------------------


def cordic_gain(n_iter: int) -> float:
    """K = prod_{i<n} sqrt(1 + 2^-2i). Converges to ~1.6467602581210656."""
    k = 1.0
    for i in range(n_iter):
        k *= math.sqrt(1.0 + 2.0 ** (-2 * i))
    return k


def cordic_convergence_limit(n_iter: int) -> float:
    """sum atan(2^-i) -- the largest |z0| the rotation mode can drive to zero."""
    return float(sum(math.atan(2.0 ** -i) for i in range(n_iter)))


def atan_table(n_iter: int, ang_bits: int) -> np.ndarray:
    """atan(2^-i) in angle LSBs, where a full circle is 2**ang_bits.

    This table is quantized, so the CORDIC cannot resolve angle better than one
    LSB no matter how many iterations you run. Past roughly ang_bits iterations
    the table entries collapse to 1 and then 0, and extra stages buy nothing --
    `DDC.__post_init__` checks for that rather than letting you pay for dead
    hardware.
    """
    scale = (1 << ang_bits) / (2.0 * math.pi)
    return np.array(
        [int(round(math.atan(2.0**-i) * scale)) for i in range(n_iter)],
        dtype=np.int64,
    )


def cordic_rotate(x, y, z, table: np.ndarray, width: int, shift_mode: str = TRUNC):
    """Bit-exact rotation-mode CORDIC, vectorised over a whole sample array.

    Rotates (x, y) by the angle z and scales it by K. Drives z toward zero, so
    a POSITIVE z0 rotates counter-clockwise; to rotate by -theta pass z0=-theta.

    The loop body is deliberately written as the RTL stage it maps to: a compare
    for the direction, two shifts, two add/subs, one angle add/sub. Nothing here
    should be tempting to "optimise" -- the point is that it mirrors hardware.
    """
    n_iter = len(table)
    x = sat(x, width)
    y = sat(y, width)
    z = np.asarray(z, dtype=np.int64)
    for i in range(n_iter):
        d = np.where(z < 0, np.int64(-1), np.int64(1))
        xs = shr(x, i, shift_mode)
        ys = shr(y, i, shift_mode)
        xn = sat(x - d * ys, width)
        yn = sat(y + d * xs, width)
        z = z - d * table[i]
        x, y = xn, yn
    return x, y, z


def quadrant_split(phase, ang_bits: int):
    """Split an unsigned phase word into (quadrant 0..3, residual in [0, pi/2)).

    In RTL this is a bit slice, not arithmetic: the top two bits are the
    quadrant and the rest is the residual, because a full circle is 2**ang_bits.
    """
    p = np.asarray(phase, dtype=np.int64) & ((1 << ang_bits) - 1)
    return p >> np.int64(ang_bits - 2), p & ((1 << (ang_bits - 2)) - 1)


def apply_quadrant_sincos(q, c, s):
    """Lift (cos, sin) of the residual up to the full circle.

    cos(q*pi/2 + r) and sin(q*pi/2 + r) are just the residual's pair with signs
    swapped around -- muxes and negates, no multiplier.
    """
    q = np.asarray(q, dtype=np.int64)
    conds = [q == 0, q == 1, q == 2, q == 3]
    cos = np.select(conds, [c, -s, -c, s])
    sin = np.select(conds, [s, c, -s, -c])
    return cos.astype(np.int64), sin.astype(np.int64)


def prerotate_conj(q, xi, xq):
    """Multiply (xi + j*xq) by exp(-j*q*pi/2).

    The fused mixer's range reduction. The CORDIC can only take the residual
    angle, so the quadrant part of the rotation is applied to the input vector
    first -- and because it is a multiple of 90 degrees it is again only swaps
    and negates.
    """
    q = np.asarray(q, dtype=np.int64)
    conds = [q == 0, q == 1, q == 2, q == 3]
    i = np.select(conds, [xi, xq, -xi, -xq])
    Q = np.select(conds, [xq, -xi, -xq, xi])
    return i.astype(np.int64), Q.astype(np.int64)


# --------------------------------------------------------------------------
# FIR design and quantization
# --------------------------------------------------------------------------


def firwin_lowpass(n_taps: int, cutoff_norm: float) -> np.ndarray:
    """Window-method lowpass, unit DC gain. cutoff_norm is fc/(fs/2).

    Blackman window: ~-74 dB sidelobes, which is the right neighbourhood for a
    16-bit datapath (a -74 dB alias sits near the quantization floor, so neither
    dominates). Inlined instead of calling scipy.signal.firwin so this model has
    no dependency beyond numpy.
    """
    if n_taps % 2 == 0:
        raise ValueError("use an odd tap count so the filter is exactly linear phase")
    if not 0.0 < cutoff_norm < 1.0:
        raise ValueError(f"cutoff_norm must be in (0, 1), got {cutoff_norm}")
    n = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = cutoff_norm * np.sinc(cutoff_norm * n)
    m = np.arange(n_taps)
    w = (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * m / (n_taps - 1))
        + 0.08 * np.cos(4.0 * np.pi * m / (n_taps - 1))
    )
    h = h * w
    return h / h.sum()


def fir_taps_quantized(
    h: np.ndarray, coef_bits: int, fold_inv_k: bool, k_gain: float
) -> np.ndarray:
    """Quantize taps to `coef_bits`, optionally folding in the 1/K correction.

    Two details that matter more than they look:

    * The target DC gain is 1/K when folding, 1 when not (see the K-scaling note
      in the module docstring).
    * After rounding, the tap sum drifts off that target by a few LSBs. Left
      alone that is a small but real DC gain error that shows up as a broadband
      level offset. The fix -- push the residual onto the largest tap, where it
      is the smallest relative perturbation -- is the standard one, and is done
      here so the model and the RTL's coefficient ROM agree exactly.
    """
    target_dc = (1.0 / k_gain) if fold_inv_k else 1.0
    scale = full_scale(coef_bits)
    q = np.round(h * target_dc * scale).astype(np.int64)
    if would_saturate(q, coef_bits):
        raise ValueError(
            f"coefficients overflow {coef_bits} bits; widen coef_bits or lower the gain"
        )
    want = int(round(target_dc * scale))
    q[int(np.argmax(np.abs(q)))] += want - int(q.sum())
    return sat(q, coef_bits)


def fir_decimate(xi, xq, coef, decim, coef_bits, acc_bits, out_bits, shift_mode):
    """Decimating FIR on a complex stream. Returns (i, q, n_saturated).

    Written as convolution then subsample. That is numerically identical to the
    polyphase form the RTL will use -- polyphase is an implementation saving (it
    never computes the samples it throws away), not a different answer. The
    model stays in the obvious form so that when RTL and model disagree, the
    polyphase commutation is a suspect rather than an assumption.

    VALID convolution only: every returned sample is a complete dot product over
    real input samples. Full-mode convolution would also emit the n_taps-1
    samples where the filter is running off the end of the buffer, and those are
    not outputs the hardware ever produces -- an RTL comparison would mismatch
    on every one of them. They are not merely a transient to be trimmed later;
    they do not exist. (Their amplitude collapses toward zero, so their phase is
    meaningless, which is how this was found.)

    The same argument applies at the start, and valid mode handles it: there is
    no fill transient in the returned data at all, so callers need no skip.
    """
    xi = np.asarray(xi, dtype=np.int64)
    xq = np.asarray(xq, dtype=np.int64)
    n_taps = len(coef)
    if xi.size < n_taps:
        raise ValueError(f"need at least {n_taps} samples to fill the filter")

    acc_i = np.convolve(xi, coef, mode="valid")[::decim]
    acc_q = np.convolve(xq, coef, mode="valid")[::decim]
    assert_fits(acc_i, "FIR accumulator")
    assert_fits(acc_q, "FIR accumulator")

    n_sat = would_saturate(acc_i, acc_bits) + would_saturate(acc_q, acc_bits)
    acc_i = sat(acc_i, acc_bits)
    acc_q = sat(acc_q, acc_bits)

    sh = coef_bits - 1
    yi = shr(acc_i, sh, shift_mode)
    yq = shr(acc_q, sh, shift_mode)
    n_sat += would_saturate(yi, out_bits) + would_saturate(yq, out_bits)
    return sat(yi, out_bits), sat(yq, out_bits), n_sat


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DDCConfig:
    """Every number the RTL needs to be parameterised with.

    Defaults target an SDR EM capture of keyboard-controller emissions: a
    2.4 MS/s complex capture (HackRF's practical floor), a carrier 300 kHz off
    tune, decimated by 8 to 300 kS/s with a 100 kHz passband. Those are starting
    points to be re-measured against a real capture, not measured facts -- the
    front-end's job is to survive being re-tuned, so nothing downstream should
    assume them.
    """

    fs_in: float = 2_400_000.0
    f_lo: float = 300_000.0
    decim: int = 8
    fir_cutoff: float = 100_000.0
    n_taps: int = 63

    phase_bits: int = 32  # M: accumulator width -> frequency resolution
    phase_trunc_bits: int = 14  # N: phase bits that reach the angle path
    ang_bits: int = 18  # CORDIC internal angle width (>= N, zero-padded)
    n_iter: int = 16
    cordic_bits: int = 20  # datapath width inside the CORDIC (data + guard)
    data_bits: int = 16
    coef_bits: int = 16
    acc_bits: int = 40
    out_bits: int = 16

    mix_arch: str = MIX_SEPARATE
    shift_mode: str = TRUNC

    def __post_init__(self):
        if self.mix_arch not in (MIX_SEPARATE, MIX_FUSED):
            raise ValueError(f"mix_arch must be {MIX_SEPARATE!r} or {MIX_FUSED!r}")
        if self.shift_mode not in (TRUNC, ROUND):
            raise ValueError(f"shift_mode must be {TRUNC!r} or {ROUND!r}")
        if self.ang_bits > self.phase_bits:
            raise ValueError("ang_bits cannot exceed phase_bits")
        if self.phase_trunc_bits > self.phase_bits:
            raise ValueError("phase_trunc_bits (N) cannot exceed phase_bits (M)")
        if self.phase_trunc_bits < 3:
            raise ValueError("phase_trunc_bits below 3 cannot even index a quadrant")
        if self.ang_bits < self.phase_trunc_bits:
            raise ValueError(
                "ang_bits below phase_trunc_bits would truncate the phase a second "
                "time inside the CORDIC; widen ang_bits or lower N"
            )
        if self.cordic_bits < self.data_bits:
            raise ValueError("cordic_bits below data_bits throws away input precision")
        if self.mix_arch == MIX_FUSED and self.cordic_bits <= self.data_bits:
            raise ValueError(
                "the fused mixer puts the SIGNAL through the CORDIC, so cordic_bits "
                "must exceed data_bits to leave headroom for the K ~= 1.647 growth"
            )
        if self.fir_cutoff >= self.fs_in / (2 * self.decim):
            raise ValueError(
                f"cutoff {self.fir_cutoff:g} Hz is at or above the decimated Nyquist "
                f"{self.fs_in / (2 * self.decim):g} Hz -- the output would alias"
            )
        # Beyond ~ang_bits iterations the atan table entries quantize to zero and
        # the extra stages are pure area for no accuracy.
        tbl = atan_table(self.n_iter, self.ang_bits)
        if int(tbl[-1]) == 0:
            raise ValueError(
                f"n_iter={self.n_iter} exceeds what ang_bits={self.ang_bits} can "
                f"resolve: the last atan entries are 0, so those stages do nothing"
            )
        lim = cordic_convergence_limit(self.n_iter)
        if lim < math.pi / 2:
            raise ValueError(
                f"n_iter={self.n_iter} gives convergence limit {lim:.4f} rad, below "
                f"the pi/2 the quadrant range reduction needs"
            )

    @property
    def fs_out(self) -> float:
        return self.fs_in / self.decim

    @property
    def phase_inc(self) -> int:
        return int(round(self.f_lo / self.fs_in * (1 << self.phase_bits))) & (
            (1 << self.phase_bits) - 1
        )

    @property
    def f_lo_actual(self) -> float:
        """The LO the hardware actually makes -- phase_inc is an integer.

        Always report this, never the requested f_lo. With phase_bits=32 the
        error is sub-milli-Hz and irrelevant; the habit matters because with a
        narrower accumulator it stops being irrelevant, and a reference model that
        quietly used the ideal frequency would then disagree with the RTL for a
        reason nobody could find.
        """
        return self.phase_inc / (1 << self.phase_bits) * self.fs_in

    @property
    def k_gain(self) -> float:
        return cordic_gain(self.n_iter)

    @property
    def phase_trunc_residue(self) -> int:
        """FCW bits that fall off the bottom when the phase is truncated to N.

        Zero means this LO produces NO phase-truncation error at all, because
        the accumulator lands on an exact multiple of the N-bit angle step and
        the discarded bits are always zero.

        That is a trap, not a win. It is the case for every LO that is a binary
        fraction of the sample rate -- including the 300 kHz / 2.4 MS/s default,
        where PHASE_INC = 0x20000000 and the low 18 bits are zero. Measure the
        NCO there and N looks free at any width. Tune 1 kHz away and the
        truncation spurs appear at their real level. `report()` therefore
        measures both, and says which case the configured LO is.
        """
        return self.phase_inc & ((1 << (self.phase_bits - self.phase_trunc_bits)) - 1)

    @property
    def phase_trunc_sfdr_bound_db(self) -> float:
        """Classic worst-case phase-truncation spur bound, ~6.02*N dBc.

        A ceiling on what the NCO can do, set purely by N. No amount of CORDIC
        iterations or datapath width buys past it.
        """
        return 6.02 * self.phase_trunc_bits

    @property
    def mix_bits(self) -> int:
        """Width of the mixer output word.

        The fused mixer needs ONE MORE BIT than the separate one, and this is a
        real cost of the architecture rather than a modelling detail.

        Separate: the mixer output is x * exp(-j*theta), magnitude unchanged, so
        it fits in the same width as the input.

        Fused: the mixer output is K * x * exp(-j*theta), and K ~= 1.6467 > 1.
        A full-scale input therefore overflows a same-width output. The choices
        are (a) carry one extra bit here, (b) back the input off by 20*log10(K)
        = 4.34 dB, or (c) spend a multiplier undoing K -- which would give back
        exactly the multiplier the fused architecture was supposed to save.

        This model takes (a), because it is the cheapest and because burying the
        cost in an input back-off would silently hand the fused design 4.3 dB
        less dynamic range while the SNR comparison still looked fair. Carry the
        bit into the synthesis study: "fused removes a complex multiplier but
        widens the mixer output and the FIR input by one bit" is the honest
        claim, and only the netlist can say which side wins.

        The extra bit is headroom ABOVE full scale -- the LSB weight is
        unchanged -- so the FIR's output shift does not move.
        """
        return self.data_bits + (1 if self.mix_arch == MIX_FUSED else 0)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class DDC:
    """Bit-exact fixed-point DDC. `run()` returns every intermediate.

    Intermediates are returned, not just the output, because that is what makes
    the model useful for debugging RTL: when the final IQ mismatches you want to
    know whether the NCO, the mixer, or the FIR diverged, and comparing one
    stage at a time finds that in one simulation instead of three.
    """

    def __init__(self, cfg: DDCConfig | None = None):
        self.cfg = cfg or DDCConfig()
        c = self.cfg
        self.atan = atan_table(c.n_iter, c.ang_bits)
        self.h_float = firwin_lowpass(c.n_taps, c.fir_cutoff / (c.fs_in / 2))
        # See the module docstring: separate/fused put the 1/K in different places.
        self.fold_inv_k = c.mix_arch == MIX_FUSED
        self.coef = fir_taps_quantized(
            self.h_float, c.coef_bits, self.fold_inv_k, c.k_gain
        )

    # -- phase -----------------------------------------------------------

    def angle_word(self, phase: np.ndarray) -> np.ndarray:
        """M-bit accumulator output -> the angle the CORDIC actually receives.

        Two steps, and keeping them distinct is the whole point of having both
        M and N:

          1. TRUNCATE the M-bit accumulator to N bits. This throws information
             away and is what creates phase-truncation spurs. It is the step
             that N controls.
          2. ZERO-PAD back up to ang_bits, the CORDIC's internal angle width.
             This adds no information and no error -- it just gives the rotation
             room to converge on the truncated angle rather than quantizing it a
             second time.

        Collapsing these into one number (using N as the CORDIC's z width too)
        would conflate a spectral-purity choice with an arithmetic-precision
        one, and would cap the useful iteration count at 13 for N=14 -- past
        that the atan table entries round to zero and the stages do nothing.

        Both the NCO and the fused mixer call this, so they cannot drift apart.
        """
        c = self.cfg
        truncated = shr(phase, c.phase_bits - c.phase_trunc_bits)
        return truncated << np.int64(c.ang_bits - c.phase_trunc_bits)

    def phase(self, n: int, phase0: int = 0, inc: int | None = None) -> np.ndarray:
        """Phase accumulator output: (phase0 + n*inc) mod 2**phase_bits.

        `inc` overrides the configured FCW, which report() uses to measure the
        NCO at an LO that actually exercises phase truncation.
        """
        c = self.cfg
        mask = (1 << c.phase_bits) - 1
        step = c.phase_inc if inc is None else (int(inc) & mask)
        return (phase0 + np.arange(n, dtype=np.int64) * step) & mask

    def nco(self, phase: np.ndarray):
        """Phase word -> (cos, sin) at data_bits, unit amplitude.

        The 1/K seed is what makes the output unit amplitude without a
        multiplier. Note the deliberate saturation: cos(0) wants to be exactly
        +1.0, which a signed word cannot represent, so it saturates to
        full_scale-1. That one-LSB asymmetry is a genuine property of the
        hardware and the model reproduces it rather than rounding it away.
        """
        c = self.cfg
        q, rem = quadrant_split(self.angle_word(phase), c.ang_bits)
        x0 = np.full(rem.shape, int(round(full_scale(c.cordic_bits) / c.k_gain)), np.int64)
        y0 = np.zeros(rem.shape, np.int64)
        x, y, _ = cordic_rotate(x0, y0, rem, self.atan, c.cordic_bits, c.shift_mode)
        cos, sin = apply_quadrant_sincos(q, x, y)
        sh = c.cordic_bits - c.data_bits
        return sat(shr(cos, sh, c.shift_mode), c.data_bits), sat(
            shr(sin, sh, c.shift_mode), c.data_bits
        )

    # -- mixers ----------------------------------------------------------

    def mix_separate(self, xi, xq, cos, sin):
        """(xi + j*xq) * (cos - j*sin), four real multiplies.

        The conjugate is the down-conversion. With a real input (xq == 0) two of
        the four multiplies drop out, which is worth remembering when comparing
        gate counts against the fused path.
        """
        c = self.cfg
        xi = np.asarray(xi, np.int64)
        xq = np.asarray(xq, np.int64)
        pi = xi * cos + xq * sin
        pq = xq * cos - xi * sin
        assert_fits(pi, "mixer product")
        assert_fits(pq, "mixer product")
        sh = c.data_bits - 1
        return sat(shr(pi, sh, c.shift_mode), c.data_bits), sat(
            shr(pq, sh, c.shift_mode), c.data_bits
        )

    def mix_fused(self, xi, xq, phase):
        """Rotate the input vector by -theta directly: the rotation IS the mix.

        No complex multiplier at all. The cost is that the CORDIC is now in the
        signal path rather than off to the side generating a carrier, so its
        width is set by the signal's dynamic range, and its output carries the K
        gain (removed downstream in the FIR coefficients).
        """
        c = self.cfg
        q, rem = quadrant_split(self.angle_word(phase), c.ang_bits)
        ri, rq = prerotate_conj(q, np.asarray(xi, np.int64), np.asarray(xq, np.int64))
        # Shift up by one LESS than the available guard, so the K growth has a
        # bit to grow into. Using the full guard would put a full-scale input at
        # the top of the CORDIC word and K would saturate it on iteration one.
        g = c.cordic_bits - c.data_bits - 1
        n_pre = would_saturate(ri << np.int64(g), c.cordic_bits)
        x, y, _ = cordic_rotate(
            ri << np.int64(g), rq << np.int64(g), -rem, self.atan,
            c.cordic_bits, c.shift_mode,
        )
        self._fused_sat = n_pre
        return sat(shr(x, g, c.shift_mode), c.mix_bits), sat(
            shr(y, g, c.shift_mode), c.mix_bits
        )

    # -- top level -------------------------------------------------------

    def run(self, xi, xq=None, phase0: int = 0) -> dict:
        """Full DDC. Real input is accepted as xq=None (the common EM case)."""
        c = self.cfg
        xi = sat(np.asarray(xi, np.int64), c.data_bits)
        xq = np.zeros_like(xi) if xq is None else sat(np.asarray(xq, np.int64), c.data_bits)
        if xi.shape != xq.shape:
            raise ValueError(f"I/Q length mismatch: {xi.shape} vs {xq.shape}")

        ph = self.phase(len(xi), phase0)
        cos, sin = self.nco(ph)
        self._fused_sat = 0
        if c.mix_arch == MIX_SEPARATE:
            mi, mq = self.mix_separate(xi, xq, cos, sin)
        else:
            mi, mq = self.mix_fused(xi, xq, ph)
        n_mix_sat = self._fused_sat + would_saturate(mi, c.mix_bits) + would_saturate(
            mq, c.mix_bits
        )
        yi, yq, n_sat = fir_decimate(
            mi, mq, self.coef, c.decim, c.coef_bits, c.acc_bits, c.out_bits, c.shift_mode
        )
        n_sat += n_mix_sat
        return {
            "phase": ph, "cos": cos, "sin": sin,
            "mix_i": mi, "mix_q": mq,
            "out_i": yi, "out_q": yq,
            "stim_i": xi, "stim_q": xq,
            "n_saturated": n_sat,
        }


# --------------------------------------------------------------------------
# the ideal (float) model
# --------------------------------------------------------------------------


def ddc_ideal(xi, xq, cfg: DDCConfig, h: np.ndarray, phase0: int = 0):
    """Float64 DDC: what the answer should be, with no quantization anywhere.

    Input is integers at cfg.data_bits; output is NORMALISED so that 1.0 means
    full scale. The fixed-point model's output is normalised the same way, so
    the two are directly comparable -- an un-normalised reference is a 90 dB
    error that looks like a catastrophic datapath bug.

    Uses cfg.f_lo_actual rather than cfg.f_lo, so a frequency the accumulator
    cannot represent is not scored as a quantization error of the datapath.
    Uses the FLOAT taps, so the metric isolates datapath quantization from
    filter shape -- otherwise coefficient rounding would be double-counted.
    """
    x = (np.asarray(xi, float) + 1j * np.asarray(xq, float)) / full_scale(cfg.data_bits)
    n = np.arange(len(x))
    ph = 2 * np.pi * cfg.f_lo_actual / cfg.fs_in * n + 2 * np.pi * phase0 / (
        1 << cfg.phase_bits
    )
    # Valid-only and the same decimation phase as fir_decimate -- the two models
    # have to be sample-aligned or the SNR comparison is measuring a delay.
    y = np.convolve(x * np.exp(-1j * ph), h, mode="valid")[:: cfg.decim]
    return y


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def snr_db(ref: np.ndarray, test: np.ndarray, skip: int = 0) -> float:
    """SNR of `test` against `ref`, both complex, in dB.

    `skip` drops leading samples so the filter's fill transient -- which is a
    real difference but not an error -- does not dominate the number.
    """
    r, t = ref[skip:], test[skip:]
    p_sig = float(np.mean(np.abs(r) ** 2))
    p_err = float(np.mean(np.abs(t - r) ** 2))
    if p_err == 0.0:
        return float("inf")
    return 10.0 * math.log10(p_sig / p_err)


def sfdr_db(y: np.ndarray, skip: int = 0) -> float:
    """Spurious-free dynamic range of a SINGLE-TONE output, in dB.

    Carrier bin vs the largest other bin. This is the number that actually
    catches an NCO problem: phase-truncation spurs and atan-table quantization
    show up here long before they move an SNR figure.

    Single tone is not a suggestion. Feed this a two-tone stimulus and the
    "largest spur" it finds is the second tone, so it reports ~0 dB and tells
    you nothing. `report()` therefore measures SNR on the two-tone case (which
    needs the second tone, to catch a mirrored spectrum) and SFDR on a separate
    single-tone run.
    """
    y = y[skip:]
    if len(y) < 64:
        return float("nan")

    # Four-term Blackman-Harris, not Hanning. The tone will not land exactly on
    # an FFT bin -- the output length is set by the decimation, not chosen to
    # make the tone coherent -- so an off-bin tone leaks into its neighbours and
    # the "worst spur" found is that leakage rather than anything the hardware
    # did. Hanning's -31 dB first sidelobe puts that floor around 40 dB, which
    # is well above the ~58 dB spurs actually being looked for: the measurement
    # would report the window.
    #
    # Blackman-Harris has -92 dB sidelobes, comfortably below the real spurs, at
    # the cost of a wider main lobe -- hence the 8-bin guard below, which is what
    # that main lobe spans.
    n = len(y)
    m_ = np.arange(n)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    w = (
        a[0]
        - a[1] * np.cos(2 * np.pi * m_ / (n - 1))
        + a[2] * np.cos(4 * np.pi * m_ / (n - 1))
        - a[3] * np.cos(6 * np.pi * m_ / (n - 1))
    )
    spec = np.abs(np.fft.fft(y * w))
    peak = int(np.argmax(spec))
    guard = 8
    masked = spec.copy()
    for d in range(-guard, guard + 1):
        masked[(peak + d) % n] = 0.0
    spur = float(np.max(masked))
    if spur == 0.0:
        return float("inf")
    return 20.0 * math.log10(spec[peak] / spur)


def effective_bits(snr: float) -> float:
    """SNR -> ENOB, via the usual 6.02N + 1.76 sine-wave relation."""
    return (snr - 1.76) / 6.02


# --------------------------------------------------------------------------
# stimulus
# --------------------------------------------------------------------------


def tone(n: int, fs: float, f: float, amp_dbfs: float, bits: int, phase: float = 0.0):
    """Complex tone at `f` Hz, quantized to `bits`. Returns (i, q) integers."""
    a = full_scale(bits) * (10.0 ** (amp_dbfs / 20.0))
    t = np.arange(n)
    z = a * np.exp(1j * (2 * np.pi * f / fs * t + phase))
    return (
        sat(np.round(z.real).astype(np.int64), bits),
        sat(np.round(z.imag).astype(np.int64), bits),
    )


def two_tone(n: int, cfg: DDCConfig, offsets=(25_000.0, -60_000.0), amp_dbfs=-6.0):
    """In-band tone plus a second one, both offset from the LO.

    The negative offset is the point: it is the test that fails loudly if the
    conjugate is backwards, because a mirrored spectrum swaps the two tones and
    nothing else about the output looks wrong.
    """
    zi = np.zeros(n, np.int64)
    zq = np.zeros(n, np.int64)
    per = amp_dbfs - 20 * math.log10(len(offsets))
    for k, off in enumerate(offsets):
        i, q = tone(n, cfg.fs_in, cfg.f_lo_actual + off, per, cfg.data_bits, 0.3 * k)
        zi, zq = zi + i, zq + q
    return sat(zi, cfg.data_bits), sat(zq, cfg.data_bits)


# --------------------------------------------------------------------------
# test-vector emission
# --------------------------------------------------------------------------


def _hex_lines(a: np.ndarray, bits: int) -> list[str]:
    """Two's-complement hex, one value per line, for $readmemh."""
    mask = (1 << bits) - 1
    nib = (bits + 3) // 4
    return [format(int(v) & mask, f"0{nib}x") for v in np.asarray(a).ravel()]


def emit_vectors(ddc: DDC, out_dir: str, n: int = 4096) -> dict:
    """Write stimulus, per-stage expected values, and the RTL parameter header.

    The testbench reads these rather than recomputing the reference. A testbench
    that recomputes is only testing that two of your own implementations agree,
    which is exactly the failure mode where both share the same sign error.
    """
    c = ddc.cfg
    os.makedirs(out_dir, exist_ok=True)
    xi, xq = two_tone(n, c)
    r = ddc.run(xi, xq)

    files = {
        "stim_i.hex": (r["stim_i"], c.data_bits),
        "stim_q.hex": (r["stim_q"], c.data_bits),
        "nco_cos.hex": (r["cos"], c.data_bits),
        "nco_sin.hex": (r["sin"], c.data_bits),
        "mix_i.hex": (r["mix_i"], c.data_bits),
        "mix_q.hex": (r["mix_q"], c.data_bits),
        "out_i.hex": (r["out_i"], c.out_bits),
        "out_q.hex": (r["out_q"], c.out_bits),
        "fir_coef.hex": (ddc.coef, c.coef_bits),
    }
    for name, (arr, bits) in files.items():
        with open(os.path.join(out_dir, name), "w", newline="\n") as f:
            f.write("\n".join(_hex_lines(arr, bits)) + "\n")

    svh = os.path.join(out_dir, "ddc_params.svh")
    with open(svh, "w", newline="\n") as f:
        f.write(_params_svh(ddc, n, len(r["out_i"])))

    manifest = {
        "config": asdict(c),
        "derived": {
            "fs_out": c.fs_out,
            "phase_inc": c.phase_inc,
            "f_lo_actual": c.f_lo_actual,
            "k_gain": c.k_gain,
            "inv_k_folded_into_coefficients": ddc.fold_inv_k,
            "n_input_samples": n,
            "n_output_samples": len(r["out_i"]),
            "n_saturated": r["n_saturated"],
        },
        "files": sorted(files) + ["ddc_params.svh"],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def _params_svh(ddc: DDC, n_in: int, n_out: int) -> str:
    c = ddc.cfg
    return f"""// Generated by hardware/reference/ddc_reference.py -- do not edit by hand.
// Regenerate:  python hardware/reference/ddc_reference.py --emit-vectors <dir>
`ifndef DDC_PARAMS_SVH
`define DDC_PARAMS_SVH

// M: phase accumulator width. Sets frequency resolution, fs/2^M =
// {c.fs_in / (1 << c.phase_bits):.6f} Hz.
localparam int PHASE_BITS  = {c.phase_bits};
// N: phase bits that reach the angle path. Sets spectral purity -- the
// worst-case phase-truncation spur bound is ~6.02*N = {c.phase_trunc_sfdr_bound_db:.1f} dBc,
// and no CORDIC width or iteration count buys past it.
localparam int PHASE_TRUNC_BITS = {c.phase_trunc_bits};
// CORDIC internal angle width. >= N, with the truncated phase zero-padded into
// it. Kept wider than N so the rotation converges on the truncated angle
// instead of quantizing it a second time; at N={c.phase_trunc_bits} an N-wide z register
// would cap useful iterations at 13.
localparam int ANG_BITS    = {c.ang_bits};
localparam int N_ITER      = {c.n_iter};
localparam int CORDIC_BITS = {c.cordic_bits};
localparam int DATA_BITS   = {c.data_bits};
localparam int COEF_BITS   = {c.coef_bits};
localparam int MIX_BITS    = {c.mix_bits};   // {"data_bits + 1: headroom for the fused mixer's K growth"
                                if c.mix_arch == MIX_FUSED else
                                "= data_bits: the separate mixer preserves magnitude"}
localparam int ACC_BITS    = {c.acc_bits};
localparam int OUT_BITS    = {c.out_bits};
localparam int N_TAPS      = {c.n_taps};
localparam int DECIM       = {c.decim};

// Phase accumulator increment for f_lo = {c.f_lo:.0f} Hz at fs = {c.fs_in:.0f} Hz.
// The LO the hardware actually produces is {c.f_lo_actual:.6f} Hz.
// Bits falling below the N-bit truncation point: {c.phase_trunc_residue}.
// {"NOTE: zero -- this LO is a binary fraction of fs, so it exercises NO phase" if c.phase_trunc_residue == 0 else "This LO exercises phase truncation, so the spurs are real here."}
// {"truncation at all. Do not characterise the NCO at this LO alone." if c.phase_trunc_residue == 0 else ""}
localparam logic [PHASE_BITS-1:0] PHASE_INC = {c.phase_bits}'h{c.phase_inc:0{(c.phase_bits + 3) // 4}x};

// CORDIC gain K = {c.k_gain:.10f} for N_ITER={c.n_iter}.
// 1/K is {"folded into FIR_COEF below (fused mixer scales the signal by K)"
         if ddc.fold_inv_k else
         "applied at the NCO seed X0 (separate mixer: sin/cos are unit amplitude)"}.
localparam logic signed [CORDIC_BITS-1:0] CORDIC_X0 =
    {c.cordic_bits}'sd{int(round(full_scale(c.cordic_bits) / c.k_gain))};

localparam int SHIFT_ROUNDS = {1 if c.shift_mode == ROUND else 0};  // 0 = truncate

localparam int N_STIM = {n_in};
localparam int N_OUT  = {n_out};

`endif
"""


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def report(cfg: DDCConfig, n: int = 8192) -> dict:
    ddc = DDC(cfg)
    xi, xq = two_tone(n, cfg)
    r = ddc.run(xi, xq)
    ref = ddc_ideal(xi, xq, cfg, ddc.h_float)

    fixed = (r["out_i"] + 1j * r["out_q"]).astype(complex) / full_scale(cfg.out_bits)

    # SFDR needs a single tone -- see sfdr_db(). Separate run, offset well off
    # both DC and the band edge so neither the filter skirt nor a DC term is
    # mistaken for a spur.
    si, sq = tone(n, cfg.fs_in, cfg.f_lo_actual + 37_000.0, -6.0, cfg.data_bits)
    rs = ddc.run(si, sq)
    single = (rs["out_i"] + 1j * rs["out_q"]).astype(complex) / full_scale(cfg.out_bits)
    # The fused path's K lives in the coefficients, so both land on the same
    # scale as the ideal model, which has no K at all. Nothing to undo here --
    # if this ever needs a fudge factor, the K accounting has drifted.
    #
    # No skip: fir_decimate returns valid-convolution output only, so there is
    # no fill transient to trim on either end.
    skip = 0

    # NCO measured on its own: the carrier's own purity, independent of the
    # signal path, which is where phase-truncation spurs actually live.
    ph = ddc.phase(n)
    cos, sin = ddc.nco(ph)
    nco_c = (cos + 1j * sin).astype(complex) / full_scale(cfg.data_bits)
    ideal_nco = np.exp(1j * 2 * np.pi * cfg.f_lo_actual / cfg.fs_in * np.arange(n))

    # And again at an LO chosen to exercise phase truncation. The configured LO
    # may well be a binary fraction of fs (the default is), in which case the
    # discarded bits are always zero and N looks free at any width. Setting a
    # spread of low FCW bits is what makes the N-bit truncation actually bite --
    # this is the number that reflects what a real tuning will do.
    hard_inc = (cfg.phase_inc + 0x0002AAAB) & ((1 << cfg.phase_bits) - 1)
    hard_cos, hard_sin = ddc.nco(ddc.phase(n, inc=hard_inc))
    hard_c = (hard_cos + 1j * hard_sin).astype(complex) / full_scale(cfg.data_bits)

    out = {
        "fs_out": cfg.fs_out,
        "f_lo_actual": cfg.f_lo_actual,
        "f_lo_error_hz": cfg.f_lo_actual - cfg.f_lo,
        "k_gain": cfg.k_gain,
        "inv_k_in_coefficients": ddc.fold_inv_k,
        "coef_dc_gain": float(ddc.coef.sum()) / full_scale(cfg.coef_bits),
        "nco_snr_db": snr_db(ideal_nco, nco_c),
        "nco_sfdr_db": sfdr_db(nco_c),
        "nco_sfdr_hard_lo_db": sfdr_db(hard_c),
        "hard_lo_hz": hard_inc / (1 << cfg.phase_bits) * cfg.fs_in,
        "phase_trunc_residue": cfg.phase_trunc_residue,
        "phase_trunc_bound_db": cfg.phase_trunc_sfdr_bound_db,
        "ddc_snr_db": snr_db(ref, fixed, skip),
        "ddc_enob": effective_bits(snr_db(ref, fixed, skip)),
        "ddc_sfdr_db": sfdr_db(single, skip),
        "n_saturated": r["n_saturated"] + rs["n_saturated"],
        "convergence_limit_rad": cordic_convergence_limit(cfg.n_iter),
    }
    return out


def _print_report(cfg: DDCConfig, m: dict) -> None:
    print(f"  mixer architecture     {cfg.mix_arch}")
    print(f"  shift mode             {cfg.shift_mode}")
    print(f"  fs in / out            {cfg.fs_in / 1e6:.3f} MS/s -> {m['fs_out'] / 1e3:.1f} kS/s  (/{cfg.decim})")
    print(f"  LO requested / actual  {cfg.f_lo / 1e3:.3f} kHz / {m['f_lo_actual'] / 1e3:.6f} kHz  (err {m['f_lo_error_hz']:+.6f} Hz)")
    print(f"  phase accum (M)        {cfg.phase_bits} bits  ->  resolution "
          f"{cfg.fs_in / (1 << cfg.phase_bits):.6f} Hz")
    print(f"  phase to angle (N)     {cfg.phase_trunc_bits} bits  ->  angle LSB "
          f"{2 * math.pi / (1 << cfg.phase_trunc_bits):.3e} rad, "
          f"spur bound ~{m['phase_trunc_bound_db']:.1f} dBc")
    print(f"  CORDIC angle width     {cfg.ang_bits} bits internal "
          f"(N zero-padded by {cfg.ang_bits - cfg.phase_trunc_bits})")
    print(f"  CORDIC K               {m['k_gain']:.10f}  ({cfg.n_iter} iterations)")
    print(f"  1/K removed at         {'FIR coefficients' if m['inv_k_in_coefficients'] else 'NCO seed'}")
    print(f"  FIR DC gain            {m['coef_dc_gain']:.6f}  (target {1 / m['k_gain']:.6f})"
          if m["inv_k_in_coefficients"] else
          f"  FIR DC gain            {m['coef_dc_gain']:.6f}  (target 1.000000)")
    print(f"  convergence limit      {m['convergence_limit_rad']:.4f} rad  (need >= {math.pi / 2:.4f})")
    print()
    print(f"  NCO   SNR {m['nco_snr_db']:7.2f} dB    SFDR {m['nco_sfdr_db']:7.2f} dB "
          f"(at the configured LO)")
    if m["phase_trunc_residue"] == 0:
        print(f"        ^ this LO is a binary fraction of fs, so NO phase truncation")
        print(f"          occurs and N={cfg.phase_trunc_bits} costs nothing here. Do not")
        print(f"          quote this figure as the NCO's spectral purity.")
    print(f"  NCO   SFDR {m['nco_sfdr_hard_lo_db']:6.2f} dB at {m['hard_lo_hz'] / 1e3:.3f} kHz "
          f"-- an LO that DOES exercise the N-bit truncation")
    print(f"  DDC   SNR {m['ddc_snr_db']:7.2f} dB    SFDR {m['ddc_sfdr_db']:7.2f} dB    ENOB {m['ddc_enob']:.2f} bits")
    if m["n_saturated"]:
        print(f"  !! {m['n_saturated']} samples saturated -- back the input off or widen the datapath")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Numpy reference model of the DDC front-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--report", action="store_true", help="print SNR/SFDR for the configuration")
    p.add_argument("--compare-arch", action="store_true",
                   help="run separate and fused side by side (the synthesis study's premise)")
    p.add_argument("--emit-vectors", metavar="DIR", help="write RTL test vectors to DIR")
    p.add_argument("--n", type=int, default=8192, help="stimulus length in samples")
    p.add_argument("--mix-arch", choices=(MIX_SEPARATE, MIX_FUSED), default=MIX_SEPARATE)
    p.add_argument("--shift-mode", choices=(TRUNC, ROUND), default=TRUNC)
    p.add_argument("--n-iter", type=int, default=DDCConfig.n_iter)
    p.add_argument("--data-bits", type=int, default=DDCConfig.data_bits)
    p.add_argument("--cordic-bits", type=int, default=DDCConfig.cordic_bits)
    p.add_argument("--fs-in", type=float, default=DDCConfig.fs_in)
    p.add_argument("--f-lo", type=float, default=DDCConfig.f_lo)
    p.add_argument("--decim", type=int, default=DDCConfig.decim)
    a = p.parse_args(argv)

    try:
        cfg = DDCConfig(
            fs_in=a.fs_in, f_lo=a.f_lo, decim=a.decim,
            mix_arch=a.mix_arch, shift_mode=a.shift_mode, n_iter=a.n_iter,
            data_bits=a.data_bits, cordic_bits=a.cordic_bits,
        )
    except (ValueError, OverflowError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    did = False
    if a.compare_arch:
        did = True
        print("DDC reference model -- architecture comparison\n")
        results = {}
        for arch in (MIX_SEPARATE, MIX_FUSED):
            c = replace(cfg, mix_arch=arch)
            results[arch] = report(c, a.n)
            print(f"[{arch}]")
            _print_report(c, results[arch])
            print()
        d = results[MIX_FUSED]["ddc_snr_db"] - results[MIX_SEPARATE]["ddc_snr_db"]
        print(f"  fused - separate: {d:+.2f} dB SNR")
        print("  The synthesis comparison is only meaningful at matched SNR. If this")
        print("  gap is large, equalise it (usually via cordic_bits) BEFORE comparing")
        print("  area -- otherwise you are comparing a cheap design to an accurate one.")
    elif a.report:
        did = True
        print("DDC reference model\n")
        _print_report(cfg, report(cfg, a.n))

    if a.emit_vectors:
        did = True
        man = emit_vectors(DDC(cfg), a.emit_vectors, a.n)
        d = man["derived"]
        print(f"\nwrote {len(man['files'])} files to {a.emit_vectors}")
        print(f"  {d['n_input_samples']} input samples -> {d['n_output_samples']} output samples")
        print(f"  phase_inc = 0x{d['phase_inc']:08x}, K = {d['k_gain']:.10f}")

    if not did:
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
