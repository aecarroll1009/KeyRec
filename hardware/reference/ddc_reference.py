"""Numpy reference model of the EM channel's DDC front-end.

    EM samples --> [ Mixer ] --> [ Decimating FIR ] --> baseband IQ
                       ^
                   [  NCO  ]  phase accumulator -> rotation-mode CORDIC

`ddc_ideal()` computes the exact float64 answer. `DDC.run()` computes the
bit-exact fixed-point answer the RTL must match. Run with --report,
--compare-arch, or --emit-vectors.
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

    Clips rather than wraps, so an overflow does not corrupt the whole band.

    Args:
        x: Values to saturate.
        bits: Target word width.

    Returns:
        `x` clipped to the representable range of a `bits`-wide signed word.
    """
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def would_saturate(x, bits: int) -> int:
    """Count how many elements sat() would clip."""
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
    """Right-shift `x` by `s`, matching the RTL's shift behavior.

    TRUNC floors, like Verilog's `>>>`. ROUND is round-half-up.

    Args:
        x: Values to shift.
        s: Shift amount; values with s <= 0 are returned unchanged.
        mode: TRUNC or ROUND.

    Returns:
        `x` shifted right by `s`.
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
    """The largest |z0| the rotation mode can drive to zero: sum atan(2^-i)."""
    return float(sum(math.atan(2.0 ** -i) for i in range(n_iter)))


def atan_table(n_iter: int, ang_bits: int) -> np.ndarray:
    """Compute atan(2^-i) in angle LSBs for each CORDIC iteration.

    Entries round to zero past about ang_bits iterations, so extra stages
    beyond that add no accuracy.

    Args:
        n_iter: Number of CORDIC iterations to generate entries for.
        ang_bits: Angle word width; a full circle is 2**ang_bits.

    Returns:
        One table entry per iteration, in angle LSBs.
    """
    scale = (1 << ang_bits) / (2.0 * math.pi)
    return np.array(
        [int(round(math.atan(2.0**-i) * scale)) for i in range(n_iter)],
        dtype=np.int64,
    )


def cordic_rotate(x, y, z, table: np.ndarray, width: int, shift_mode: str = TRUNC):
    """Bit-exact rotation-mode CORDIC, vectorised over a whole sample array.

    Rotates (x, y) by angle z, scaling the result by K. Pass z0=-theta to
    rotate by -theta.

    Args:
        x, y: Initial vector components, at `width` bits.
        z: Initial angle, in the same LSB units as `table`.
        table: Per-iteration atan values from `atan_table()`.
        width: Datapath width for x and y.
        shift_mode: TRUNC or ROUND, for the per-iteration shifts.

    Returns:
        The rotated (x, y) and the residual angle z after all iterations.
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

    In RTL this is a bit slice, not arithmetic. The top two bits give the
    quadrant, and the rest gives the residual.

    Args:
        phase: Angle-domain phase values, at ang_bits width.
        ang_bits: Angle word width.

    Returns:
        A (quadrant, residual) tuple.
    """
    p = np.asarray(phase, dtype=np.int64) & ((1 << ang_bits) - 1)
    return p >> np.int64(ang_bits - 2), p & ((1 << (ang_bits - 2)) - 1)


def apply_quadrant_sincos(q, c, s):
    """Lift the residual's (cos, sin) pair to the full circle.

    Uses sign swaps only, no multiplier.

    Args:
        q: Quadrant, 0..3.
        c, s: (cos, sin) of the residual angle.

    Returns:
        A (cos, sin) tuple lifted to the full circle.
    """
    q = np.asarray(q, dtype=np.int64)
    conds = [q == 0, q == 1, q == 2, q == 3]
    cos = np.select(conds, [c, -s, -c, s])
    sin = np.select(conds, [s, c, -s, -c])
    return cos.astype(np.int64), sin.astype(np.int64)


def prerotate_conj(q, xi, xq):
    """Multiply (xi + j*xq) by exp(-j*q*pi/2).

    Applies the quadrant part of the rotation before the CORDIC handles the
    residual. Being a multiple of 90 degrees, this needs only sign swaps.

    Args:
        q: Quadrant, 0..3.
        xi, xq: Input I/Q samples.

    Returns:
        The (i, q) input rotated by -q*pi/2.
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
    """Design a window-method lowpass FIR with unit DC gain.

    Uses a Blackman window, whose -74 dB sidelobes suit a 16-bit datapath.
    Implemented directly rather than via scipy, so this module depends only
    on numpy.

    Args:
        n_taps: Number of taps; must be odd, for exact linear phase.
        cutoff_norm: Cutoff frequency, normalised as fc/(fs/2), in (0, 1).

    Returns:
        The filter taps, with unit DC gain.
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

    Any rounding error is absorbed into the largest tap, so the quantized
    coefficient sum matches the target DC gain exactly.

    Args:
        h: Float-precision filter taps, unit DC gain.
        coef_bits: Target coefficient word width.
        fold_inv_k: Whether to fold the 1/K correction into the DC gain.
        k_gain: The CORDIC gain K, used when fold_inv_k is set.

    Returns:
        The quantized, saturated taps at `coef_bits`.
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
    """Run a decimating FIR on a complex stream.

    Uses valid-only convolution, so every returned sample is a complete
    filter output with no partial edge samples.

    Args:
        xi, xq: Input I/Q samples.
        coef: Quantized filter taps.
        decim: Decimation factor.
        coef_bits: Coefficient word width.
        acc_bits: Accumulator width.
        out_bits: Output word width.
        shift_mode: TRUNC or ROUND, for the output shift.

    Returns:
        An (i, q, n_saturated) tuple: decimated output at `out_bits`, and
        the count of saturated samples across both stages.
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


def _validate_modes(cfg: "DDCConfig") -> None:
    """Validate that mix_arch and shift_mode name a recognised mode.

    Args:
        cfg: The config being validated.

    Raises:
        ValueError: If mix_arch or shift_mode is not one of the defined
            constants.
    """
    if cfg.mix_arch not in (MIX_SEPARATE, MIX_FUSED):
        raise ValueError(f"mix_arch must be {MIX_SEPARATE!r} or {MIX_FUSED!r}")
    if cfg.shift_mode not in (TRUNC, ROUND):
        raise ValueError(f"shift_mode must be {TRUNC!r} or {ROUND!r}")


def _validate_widths(cfg: "DDCConfig") -> None:
    """Validate the phase, angle, and datapath bit-width relationships.

    Args:
        cfg: The config being validated.

    Raises:
        ValueError: If any width relationship the NCO/CORDIC/mixer datapath
            depends on is violated.
    """
    if cfg.ang_bits > cfg.phase_bits:
        raise ValueError("ang_bits cannot exceed phase_bits")
    if cfg.phase_trunc_bits > cfg.phase_bits:
        raise ValueError("phase_trunc_bits (N) cannot exceed phase_bits (M)")
    if cfg.phase_trunc_bits < 3:
        raise ValueError("phase_trunc_bits below 3 cannot even index a quadrant")
    if cfg.ang_bits < cfg.phase_trunc_bits:
        raise ValueError(
            "ang_bits below phase_trunc_bits would truncate the phase a second "
            "time inside the CORDIC; widen ang_bits or lower N"
        )
    if cfg.cordic_bits < cfg.data_bits:
        raise ValueError("cordic_bits below data_bits throws away input precision")
    if cfg.mix_arch == MIX_FUSED and cfg.cordic_bits <= cfg.data_bits:
        raise ValueError(
            "the fused mixer puts the signal through the CORDIC, so cordic_bits "
            "must exceed data_bits to leave headroom for the K ~= 1.647 growth"
        )


def _validate_no_aliasing(cfg: "DDCConfig") -> None:
    """Validate that the FIR cutoff sits below the decimated Nyquist frequency.

    Args:
        cfg: The config being validated.

    Raises:
        ValueError: If fir_cutoff is at or above fs_in / (2 * decim).
    """
    if cfg.fir_cutoff >= cfg.fs_in / (2 * cfg.decim):
        raise ValueError(
            f"cutoff {cfg.fir_cutoff:g} Hz is at or above the decimated Nyquist "
            f"{cfg.fs_in / (2 * cfg.decim):g} Hz -- the output would alias"
        )


def _validate_cordic_iterations(cfg: "DDCConfig") -> None:
    """Validate n_iter is neither wasted past ang_bits nor short of convergence.

    Args:
        cfg: The config being validated.

    Raises:
        ValueError: If n_iter exceeds what ang_bits can resolve, or is too
            small for the convergence limit to cover the quadrant residual.
    """
    tbl = atan_table(cfg.n_iter, cfg.ang_bits)
    if int(tbl[-1]) == 0:
        raise ValueError(
            f"n_iter={cfg.n_iter} exceeds what ang_bits={cfg.ang_bits} can "
            f"resolve: the last atan entries are 0, so those stages do nothing"
        )
    lim = cordic_convergence_limit(cfg.n_iter)
    if lim < math.pi / 2:
        raise ValueError(
            f"n_iter={cfg.n_iter} gives convergence limit {lim:.4f} rad, below "
            f"the pi/2 the quadrant range reduction needs"
        )


@dataclass(frozen=True)
class DDCConfig:
    """Every value the RTL needs, as a frozen dataclass.

    Defaults target a 2.4 MS/s EM capture at a 300 kHz carrier, decimated by
    8 to a 300 kHz passband.
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
        _validate_modes(self)
        _validate_widths(self)
        _validate_no_aliasing(self)
        _validate_cordic_iterations(self)

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
        """The LO frequency the hardware actually produces.

        Use this instead of the requested f_lo when comparing against RTL.

        Returns:
            The actual LO frequency in Hz, given the quantized phase_inc.
        """
        return self.phase_inc / (1 << self.phase_bits) * self.fs_in

    @property
    def k_gain(self) -> float:
        return cordic_gain(self.n_iter)

    @property
    def phase_trunc_residue(self) -> int:
        """FCW bits discarded when truncating the phase to N bits.

        Zero means this LO exercises no phase-truncation error, so its
        measured NCO purity should not be quoted as representative.

        Returns:
            The discarded FCW bits; zero if this LO exercises no truncation.
        """
        return self.phase_inc & ((1 << (self.phase_bits - self.phase_trunc_bits)) - 1)

    @property
    def phase_trunc_sfdr_bound_db(self) -> float:
        """Worst-case phase-truncation spur bound, ~6.02*N dBc.

        Depends only on N, not on CORDIC iterations or datapath width.

        Returns:
            The bound in dBc.
        """
        return 6.02 * self.phase_trunc_bits

    @property
    def mix_bits(self) -> int:
        """Width of the mixer output word.

        The fused mixer's output carries the CORDIC gain K > 1, so it needs
        one more bit than the separate mixer's to avoid overflow.

        Returns:
            The mixer output width in bits: data_bits, plus one for the
            fused architecture.
        """
        return self.data_bits + (1 if self.mix_arch == MIX_FUSED else 0)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class DDC:
    """Bit-exact fixed-point DDC model.

    `run()` returns every intermediate stage's output, so an RTL mismatch
    can be localized to one stage.
    """

    def __init__(self, cfg: DDCConfig | None = None):
        self.cfg = cfg or DDCConfig()
        c = self.cfg
        self.atan = atan_table(c.n_iter, c.ang_bits)
        self.h_float = firwin_lowpass(c.n_taps, c.fir_cutoff / (c.fs_in / 2))
        self.fold_inv_k = c.mix_arch == MIX_FUSED
        self.coef = fir_taps_quantized(
            self.h_float, c.coef_bits, self.fold_inv_k, c.k_gain
        )

    # -- phase -----------------------------------------------------------

    def angle_word(self, phase: np.ndarray) -> np.ndarray:
        """Convert the M-bit accumulator phase to the CORDIC's angle input.

        Truncates to N bits, then zero-pads to ang_bits so the CORDIC
        converges without re-quantizing. Shared by the NCO and the fused
        mixer.

        Args:
            phase: M-bit phase accumulator values.

        Returns:
            The ang_bits-wide angle word the CORDIC receives.
        """
        c = self.cfg
        truncated = shr(phase, c.phase_bits - c.phase_trunc_bits)
        return truncated << np.int64(c.ang_bits - c.phase_trunc_bits)

    def phase(self, n: int, phase0: int = 0, inc: int | None = None) -> np.ndarray:
        """Compute phase accumulator output: (phase0 + n*inc) mod 2**phase_bits.

        Args:
            n: Number of samples to generate.
            phase0: Initial phase.
            inc: FCW to use instead of the configured one; `report()` uses
                this to measure the NCO at an LO that exercises phase
                truncation.

        Returns:
            The phase accumulator sequence, one value per sample.
        """
        c = self.cfg
        mask = (1 << c.phase_bits) - 1
        step = c.phase_inc if inc is None else (int(inc) & mask)
        return (phase0 + np.arange(n, dtype=np.int64) * step) & mask

    def nco(self, phase: np.ndarray):
        """Convert a phase word to (cos, sin) at data_bits, unit amplitude.

        Saturates at full_scale-1 rather than +1.0, since a signed word
        cannot represent exactly 1.0.

        Args:
            phase: Angle-domain phase values, at the accumulator's M-bit
                width.

        Returns:
            A (cos, sin) tuple, each at data_bits.
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
        """Mix (xi + j*xq) with (cos - j*sin), four real multiplies.

        Multiplies by the NCO's conjugate to down-convert. With a real
        input, two of the four multiplies are zero.

        Args:
            xi, xq: Input I/Q samples, at data_bits.
            cos, sin: NCO output, at data_bits.

        Returns:
            A (i, q) tuple of mixed output, each at data_bits.
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
        """Rotate the input vector by -theta directly: the rotation is the mix.

        No complex multiplier at all. The CORDIC output carries the gain K,
        removed later in the FIR coefficients.

        Args:
            xi, xq: Input I/Q samples, at data_bits.
            phase: Accumulator phase for each sample.

        Returns:
            A (i, q) tuple of mixed output, each at mix_bits.
        """
        c = self.cfg
        q, rem = quadrant_split(self.angle_word(phase), c.ang_bits)
        ri, rq = prerotate_conj(q, np.asarray(xi, np.int64), np.asarray(xq, np.int64))
        # Leaves one guard bit for the K growth, so a full-scale input does
        # not saturate on the first iteration.
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
        """Run the full DDC over a stimulus.

        Args:
            xi: Input I samples, at data_bits.
            xq: Input Q samples, at data_bits. None for a real input (the
                common EM case), in which case Q is treated as zero.
            phase0: Initial NCO phase.

        Returns:
            A dict of every stage's output: phase, cos, sin, mix_i, mix_q,
            out_i, out_q, stim_i, stim_q, and n_saturated.
        """
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
    """Compute the float64 DDC output: the intended answer, with no quantization.

    Normalized so full scale is 1.0, matching the fixed-point model's scale.
    Uses the actual quantized LO frequency and the float filter taps, so
    the result isolates datapath error from LO placement and filter design.

    Args:
        xi, xq: Input I/Q samples, integers at cfg.data_bits.
        cfg: The DDC configuration.
        h: Float-precision FIR taps.
        phase0: Initial NCO phase.

    Returns:
        The normalised complex baseband output, decimated and valid-only.
    """
    x = (np.asarray(xi, float) + 1j * np.asarray(xq, float)) / full_scale(cfg.data_bits)
    n = np.arange(len(x))
    ph = 2 * np.pi * cfg.f_lo_actual / cfg.fs_in * n + 2 * np.pi * phase0 / (
        1 << cfg.phase_bits
    )
    # Same valid-only convolution and decimation phase as fir_decimate, so
    # the two models stay sample-aligned.
    y = np.convolve(x * np.exp(-1j * ph), h, mode="valid")[:: cfg.decim]
    return y


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def snr_db(ref: np.ndarray, test: np.ndarray, skip: int = 0) -> float:
    """Compute the SNR of `test` against `ref`, both complex, in dB.

    Args:
        ref: Reference signal.
        test: Signal under test.
        skip: Leading samples to drop before scoring.

    Returns:
        SNR in dB, or inf if the two signals are identical.
    """
    r, t = ref[skip:], test[skip:]
    p_sig = float(np.mean(np.abs(r) ** 2))
    p_err = float(np.mean(np.abs(t - r) ** 2))
    if p_err == 0.0:
        return float("inf")
    return 10.0 * math.log10(p_sig / p_err)


def sfdr_db(y: np.ndarray, skip: int = 0) -> float:
    """Compute the spurious-free dynamic range of a single-tone output, in dB.

    Compares the carrier bin against the largest other bin. Requires a
    single-tone input. On a two-tone stimulus the largest other bin is just
    the second tone.

    Args:
        y: Complex output signal, single-tone.
        skip: Leading samples to drop.

    Returns:
        SFDR in dB, nan if too few samples remain, inf if no spur is found.
    """
    y = y[skip:]
    if len(y) < 64:
        return float("nan")

    # Blackman-Harris window: -92 dB sidelobes keep off-bin leakage below the
    # spurs being measured. The 8-bin guard matches its wider main lobe.
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
    """Generate a complex tone at `f` Hz, quantized to `bits`.

    Args:
        n: Number of samples.
        fs: Sample rate in Hz.
        f: Tone frequency in Hz.
        amp_dbfs: Amplitude relative to full scale, in dB.
        bits: Output word width.
        phase: Starting phase in radians.

    Returns:
        An (i, q) tuple of integer samples at `bits`.
    """
    a = full_scale(bits) * (10.0 ** (amp_dbfs / 20.0))
    t = np.arange(n)
    z = a * np.exp(1j * (2 * np.pi * f / fs * t + phase))
    return (
        sat(np.round(z.real).astype(np.int64), bits),
        sat(np.round(z.imag).astype(np.int64), bits),
    )


def two_tone(n: int, cfg: DDCConfig, offsets=(25_000.0, -60_000.0), amp_dbfs=-6.0):
    """Generate an in-band tone plus a second one, both offset from the LO.

    Includes one negative-offset tone, so a mirrored spectrum swaps the two
    tones instead of passing silently.

    Args:
        n: Number of samples.
        cfg: The DDC configuration, for fs_in, f_lo_actual, and data_bits.
        offsets: Frequency offsets from the LO, in Hz.
        amp_dbfs: Combined amplitude relative to full scale, in dB.

    Returns:
        An (i, q) tuple of integer samples at cfg.data_bits.
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
    """Format values as two's-complement hex, one per line, for $readmemh.

    Args:
        a: Values to format.
        bits: Word width.

    Returns:
        One hex string per value, zero-padded to the word width.
    """
    mask = (1 << bits) - 1
    nib = (bits + 3) // 4
    return [format(int(v) & mask, f"0{nib}x") for v in np.asarray(a).ravel()]


def _write_hex_files(out_dir: str, files: dict) -> None:
    """Write each array in `files` to its own $readmemh-style hex file.

    Args:
        out_dir: Directory to write into.
        files: Maps file name to an (array, bits) pair.
    """
    for name, (arr, bits) in files.items():
        with open(os.path.join(out_dir, name), "w", newline="\n") as f:
            f.write("\n".join(_hex_lines(arr, bits)) + "\n")


def emit_vectors(ddc: DDC, out_dir: str, n: int = 4096) -> dict:
    """Write stimulus, per-stage expected values, and the RTL parameter header.

    The testbench reads these values rather than recomputing them, so a
    sign error shared by both implementations cannot hide.

    Args:
        ddc: The DDC model to generate vectors from.
        out_dir: Directory to write the hex files, params header, and
            manifest into.
        n: Stimulus length in samples.

    Returns:
        The manifest dict that was also written to manifest.json.
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
    _write_hex_files(out_dir, files)

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
    """Render the RTL parameter header for the given DDC configuration.

    Args:
        ddc: The DDC model the parameters are drawn from.
        n_in: Stimulus length in samples, for the N_STIM localparam.
        n_out: Output length in samples, for the N_OUT localparam.

    Returns:
        The contents of ddc_params.svh as a string.
    """
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
// {"Note: zero -- this LO is a binary fraction of fs, so it exercises no phase" if c.phase_trunc_residue == 0 else "This LO exercises phase truncation, so the spurs are real here."}
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


def _measure_nco_purity(ddc: DDC, cfg: DDCConfig, n: int) -> dict:
    """Measure NCO spectral purity at the configured LO and at a harder one.

    Args:
        ddc: The DDC model to measure.
        cfg: The DDC configuration.
        n: Number of NCO samples to generate.

    Returns:
        A dict with nco_snr_db, nco_sfdr_db, nco_sfdr_hard_lo_db, and
        hard_lo_hz.
    """
    # Purity at the configured LO, independent of the signal path.
    ph = ddc.phase(n)
    cos, sin = ddc.nco(ph)
    nco_c = (cos + 1j * sin).astype(complex) / full_scale(cfg.data_bits)
    ideal_nco = np.exp(1j * 2 * np.pi * cfg.f_lo_actual / cfg.fs_in * np.arange(n))

    # And at an LO whose FCW has nonzero low bits, so truncation actually
    # occurs.
    hard_inc = (cfg.phase_inc + 0x0002AAAB) & ((1 << cfg.phase_bits) - 1)
    hard_cos, hard_sin = ddc.nco(ddc.phase(n, inc=hard_inc))
    hard_c = (hard_cos + 1j * hard_sin).astype(complex) / full_scale(cfg.data_bits)

    return {
        "nco_snr_db": snr_db(ideal_nco, nco_c),
        "nco_sfdr_db": sfdr_db(nco_c),
        "nco_sfdr_hard_lo_db": sfdr_db(hard_c),
        "hard_lo_hz": hard_inc / (1 << cfg.phase_bits) * cfg.fs_in,
    }


def _measure_ddc_quality(ddc: DDC, cfg: DDCConfig, n: int) -> dict:
    """Measure the fixed-point DDC's SNR/SFDR against the ideal model.

    Args:
        ddc: The DDC model to measure.
        cfg: The DDC configuration.
        n: Stimulus length in samples.

    Returns:
        A dict with ddc_snr_db, ddc_enob, ddc_sfdr_db, and n_saturated.
    """
    xi, xq = two_tone(n, cfg)
    r = ddc.run(xi, xq)
    ref = ddc_ideal(xi, xq, cfg, ddc.h_float)
    fixed = (r["out_i"] + 1j * r["out_q"]).astype(complex) / full_scale(cfg.out_bits)

    # SFDR needs a single tone, measured separately at an offset away from
    # DC and the band edge.
    si, sq = tone(n, cfg.fs_in, cfg.f_lo_actual + 37_000.0, -6.0, cfg.data_bits)
    rs = ddc.run(si, sq)
    single = (rs["out_i"] + 1j * rs["out_q"]).astype(complex) / full_scale(cfg.out_bits)

    skip = 0
    snr = snr_db(ref, fixed, skip)
    return {
        "ddc_snr_db": snr,
        "ddc_enob": effective_bits(snr),
        "ddc_sfdr_db": sfdr_db(single, skip),
        "n_saturated": r["n_saturated"] + rs["n_saturated"],
    }


def report(cfg: DDCConfig, n: int = 8192) -> dict:
    """Measure and collect the DDC's key SNR/SFDR figures for one config.

    Args:
        cfg: The DDC configuration to measure.
        n: Stimulus length in samples.

    Returns:
        A dict of the figures printed by `_print_report()`.
    """
    ddc = DDC(cfg)
    out = {
        "fs_out": cfg.fs_out,
        "f_lo_actual": cfg.f_lo_actual,
        "f_lo_error_hz": cfg.f_lo_actual - cfg.f_lo,
        "k_gain": cfg.k_gain,
        "inv_k_in_coefficients": ddc.fold_inv_k,
        "coef_dc_gain": float(ddc.coef.sum()) / full_scale(cfg.coef_bits),
        "phase_trunc_residue": cfg.phase_trunc_residue,
        "phase_trunc_bound_db": cfg.phase_trunc_sfdr_bound_db,
        "convergence_limit_rad": cordic_convergence_limit(cfg.n_iter),
    }
    out.update(_measure_nco_purity(ddc, cfg, n))
    out.update(_measure_ddc_quality(ddc, cfg, n))
    return out


def _print_report(cfg: DDCConfig, m: dict) -> None:
    """Print the figures from `report()` in human-readable form.

    Args:
        cfg: The DDC configuration the figures were measured under.
        m: The dict returned by `report(cfg, ...)`.
    """
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
        print(f"        ^ this LO is a binary fraction of fs, so no phase truncation")
        print(f"          occurs and N={cfg.phase_trunc_bits} costs nothing here. Do not")
        print(f"          quote this figure as the NCO's spectral purity.")
    print(f"  NCO   SFDR {m['nco_sfdr_hard_lo_db']:6.2f} dB at {m['hard_lo_hz'] / 1e3:.3f} kHz "
          f"-- an LO that does exercise the N-bit truncation")
    print(f"  DDC   SNR {m['ddc_snr_db']:7.2f} dB    SFDR {m['ddc_sfdr_db']:7.2f} dB    ENOB {m['ddc_enob']:.2f} bits")
    if m["n_saturated"]:
        print(f"  !! {m['n_saturated']} samples saturated -- back the input off or widen the datapath")


def _run_compare_arch(cfg: DDCConfig, n: int) -> None:
    """Print the separate and fused architectures side by side.

    Args:
        cfg: Base configuration; mix_arch is overridden per architecture.
        n: Stimulus length in samples.
    """
    print("DDC reference model -- architecture comparison\n")
    results = {}
    for arch in (MIX_SEPARATE, MIX_FUSED):
        c = replace(cfg, mix_arch=arch)
        results[arch] = report(c, n)
        print(f"[{arch}]")
        _print_report(c, results[arch])
        print()
    d = results[MIX_FUSED]["ddc_snr_db"] - results[MIX_SEPARATE]["ddc_snr_db"]
    print(f"  fused - separate: {d:+.2f} dB SNR")
    print("  The synthesis comparison is only meaningful at matched SNR. If this")
    print("  gap is large, equalise it (usually via cordic_bits) before comparing")
    print("  area -- otherwise you are comparing a cheap design to an accurate one.")


def _run_emit_vectors(cfg: DDCConfig, out_dir: str, n: int) -> None:
    """Write RTL test vectors and print a summary of what was written.

    Args:
        cfg: Configuration to generate vectors for.
        out_dir: Destination directory.
        n: Stimulus length in samples.
    """
    man = emit_vectors(DDC(cfg), out_dir, n)
    d = man["derived"]
    print(f"\nwrote {len(man['files'])} files to {out_dir}")
    print(f"  {d['n_input_samples']} input samples -> {d['n_output_samples']} output samples")
    print(f"  phase_inc = 0x{d['phase_inc']:08x}, K = {d['k_gain']:.10f}")


def main(argv=None) -> int:
    """Run the CLI: parse arguments and dispatch to report/compare/emit.

    Args:
        argv: Argument list to parse; defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
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
        _run_compare_arch(cfg, a.n)
    elif a.report:
        did = True
        print("DDC reference model\n")
        _print_report(cfg, report(cfg, a.n))

    if a.emit_vectors:
        did = True
        _run_emit_vectors(cfg, a.emit_vectors, a.n)

    if not did:
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
