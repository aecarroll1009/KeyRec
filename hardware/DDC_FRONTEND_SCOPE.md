# hardware/

SystemVerilog for the EM channel's DDC front-end — the "streaming log-mel
spectral front-end" die described in `docs/PROJECT_SCOPE.md`, with the block
diagram in `docs/cordic_ddc_nco_mixer_datapath.png`.

## Build order

**Floor (build now)** — the green blocks in the datapath diagram:

- `nco/` — phase accumulator → rotation-mode CORDIC producing sin/cos
- `mixer/` — complex multiply, down-converting to baseband IQ
- `decim/` — decimating FIR/CIC, with the CORDIC gain K ≈ 1.6467 folded into
  the filter coefficients rather than spent on a separate scaling stage

**Stretch (in development)** — input mux (EM vs acoustic) → folded FFT (one
reused butterfly) → magnitude (vectoring CORDIC) → mel filterbank → log
(hyperbolic CORDIC) → log-mel frames out.

## Planned comparison study

Two implementations of the same function, synthesised and compared:

1. **separate** — CORDIC NCO generating sin/cos, feeding a discrete complex
   multiplier (4 real multiplies)
2. **fused** — the input sample loaded as the CORDIC's initial vector and
   rotated by −θ directly, so the rotation *is* the mix and the complex
   multiplier disappears entirely

For the comparison to mean anything, both must be synthesised at **matched
throughput, matched output SNR/SFDR, identical bit widths, and the same tool /
target / constraints** — and Fmax reported alongside area, since a smaller
design that misses timing is not a smaller design. Expect the conclusion to
differ between FPGA (multipliers are free hard DSP blocks) and ASIC
(multipliers are real area and power).

## Verification

A numpy golden model of the DDC comes first — it catches sign-convention and
K-scaling bugs fast. Checking the custom CORDIC DDC against an SDR's own
internal DDC output, to within quantisation tolerance, is the later hardware
check.
