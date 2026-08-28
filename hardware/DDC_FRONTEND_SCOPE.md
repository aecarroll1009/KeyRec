# hardware/

SystemVerilog for the EM channel's DDC front-end — the "streaming log-mel spectral
front-end" die described in `docs/PROJECT_SCOPE.md`, with the block diagram in
`docs/cordic_ddc_nco_mixer_datapath.png`.

## Build order

Floor (build now) — the green blocks in the datapath diagram:

- `nco/` — phase accumulator → rotation-mode CORDIC producing sin/cos
- `mixer/` — complex multiply, down-converting to baseband IQ
- `decim/` — decimating FIR/CIC, with the CORDIC gain K ≈ 1.6467 folded into the filter
  coefficients rather than spent on a separate scaling stage

Stretch (in development) — input mux (EM vs acoustic) → folded FFT (one reused butterfly)
→ magnitude (vectoring CORDIC) → mel filterbank → log (hyperbolic CORDIC) → log-mel
frames out.

## Planned comparison study

Two implementations of the same function, synthesised and compared:

1. **Separate** — CORDIC NCO generating sin/cos, feeding a discrete complex multiplier
   (four real multiplies).
2. **Fused** — the input sample loaded as the CORDIC's initial vector and rotated by −θ
   directly, so the rotation is the mix and the complex multiplier disappears entirely.

For the comparison to mean anything, both must be synthesised at matched throughput,
matched output SNR/SFDR, identical bit widths, and the same tool/target/constraints, with
Fmax reported alongside area, since a smaller design that misses timing is not a smaller
design. The conclusion likely differs between FPGA, where multipliers are free hard DSP
blocks, and ASIC, where multipliers are real area and power.

## Verification

The numpy reference model is `reference/ddc_reference.py`, with 30 tests in
`reference/test_ddc_reference.py` (`python hardware/reference/test_ddc_reference.py`).

It is two models in one file. `ddc_ideal()` is float64, exact: what the answer should be.
`DDC.run()` is bit-exact fixed point: what the RTL must produce. RTL is checked against
the fixed-point model bit for bit, with no tolerance to hide in. The fixed-point model is
checked against the ideal one in SNR/SFDR, which is a question of whether the chosen
widths are good enough, not a correctness question.

`--emit-vectors build/ddc_vectors` writes `$readmemh`-ready hex for the stimulus and
every intermediate stage (NCO, mixer, output), plus a generated `ddc_params.svh` so the
RTL is parameterised from the same numbers. The testbench reads these values rather than
recomputing the reference, so a sign error shared by both implementations cannot hide.

Current numbers at the default 16-bit config (`--report`):

| | SNR | SFDR | ENOB |
|---|---|---|---|
| NCO alone | 91.6 dB | 93.3 dB | — |
| DDC, separate mixer | 76.4 dB | 80.8 dB | 12.39 |
| DDC, fused mixer | 79.2 dB | 82.8 dB | 12.86 |

### Design facts from the reference model

The fused mixer needs one more bit than the separate one. Its output is K·x·e^(−jθ) with
K > 1, so a full-scale input overflows a same-width output. The options are to carry an
extra bit, back the input off by 4.34 dB, or spend a multiplier undoing K — the last of
which hands back exactly the multiplier the fused architecture was supposed to save. This
belongs in the comparison study: "fused removes a complex multiplier but widens the mixer
output and the FIR input by one bit" is the honest claim, settled only by the netlist.

The 1/K correction goes in a different place per architecture. Separate folds it into the
CORDIC seed (x0 = 1/K), free, with plain FIR coefficients. Fused has no free seed — the
signal itself carries the K — so the 1/K lives in the FIR coefficients instead. Both then
land on the same output scale, which is why their SNRs are comparable at all; getting it
backwards leaves the design 4.34 dB hot while still looking correct.

Only fully-loaded filter outputs are real outputs. Hardware never produces the `n_taps-1`
tail samples where a filter runs off the end of its input buffer, so the model uses
valid-only convolution to match.

### NCO widths: M = 32, N = 14

Three separate numbers, and conflating them loses information:

- **M = 32** (`phase_bits`) — accumulator width. Sets frequency resolution, fs/2^M =
  0.56 mHz at 2.4 MS/s. Any LO is placed essentially exactly.
- **N = 14** (`phase_trunc_bits`) — phase bits reaching the angle path. Sets spectral
  purity. Truncating M→N discards information every sample, and the error is periodic, so
  it shows up as discrete spurs: worst-case bound ~6.02·N = 84 dBc. Nothing downstream
  buys past it.
- **`ang_bits` = 18** — the CORDIC's internal angle register, wider than N with the
  truncated phase zero-padded into it, so the rotation converges on the truncated angle
  instead of quantising it a second time. Using N as the z width too would cap useful
  iterations at 13, since past that the atan entries round to zero.

Measured NCO SFDR vs N, at an LO that exercises truncation:

| N | measured | 6.02·N predicts |
|---|---|---|
| 10 | 59.8 dB | 60.2 |
| 14 | 82.3 dB | 84.3 |
| 18 | 93.5 dB | 108.4 |

N=18 falls short of its bound because 16-bit amplitude quantisation dominates by then. N=14
is well matched to a 16-bit datapath; widening N alone would not buy much without
widening the datapath too.

The binary-fraction trap: phase truncation only errs when the FCW has nonzero bits below
the truncation point. The default 300 kHz / 2.4 MS/s LO gives `PHASE_INC = 0x20000000`,
whose low 18 bits are zero, so it exercises no truncation and scores 93 dB — making N=14
look free at any width. Tuned 98 Hz away, it scores 82 dB, the representative number.
`cfg.phase_trunc_residue == 0` flags the flattering case, and `--report` prints both.

### Sign conventions

Two ways to get the mix backwards, with very different consequences:

- Multiplying by the NCO instead of its conjugate up-converts — the tone lands at
  2·f_lo + delta and the lowpass deletes it. Loud and obvious.
- Negating Q yields conj(x·e^(−jθ)): the spectrum is mirrored about DC at full amplitude
  and correct bandwidth. The magnitude spectrogram looks correct, and it stays invisible
  until something downstream cares which sideband a feature came from.
  `test_sign_convention_is_not_mirrored` checks for this.

Later hardware check, unchanged: compare the custom CORDIC DDC against an SDR's own
internal DDC output, to within quantisation tolerance.
