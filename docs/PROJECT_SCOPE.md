# Project Scope

Alex is building a keystroke-inference ML project in three phases.

1. **Reproduce.** Recreate a 2023 academic paper on acoustic side-channel keyboard
   attacks — mel-spectrogram features into a CNN/CoAtNet-style classifier, ~95% reported
   accuracy from keystroke audio including audio captured over a call mic — as closely as
   possible before diverging, so any later improvement is measurable against a known
   baseline.

2. **Generalize.** Extend from one typist/one keyboard to multiple typists on one
   keyboard, then to different keyboard families (mechanical, ThinkPad-style chiclet, Mac
   chiclet/scissor). Cross-keyboard/cross-typist generalization was the weak point of the
   original paper.

3. **EM+acoustic sensor fusion**, Alex's own extension, not in the paper. Add a second,
   independent side channel — EM/Van Eck emissions from the keyboard controller, captured
   via SDR — as its own classifier, then a fusion/arbitration layer that reconciles the
   acoustic model's and EM model's per-keystroke predictions. Disagreement between the
   two channels is informative rather than noise, since they have different, largely
   uncorrelated failure modes: acoustic degrades with ambient noise, distance, and
   chiclet-style keys; EM degrades with shielding, distance, and controller design. This
   is multimodal late fusion, the same family as camera+LIDAR fusion in robotics, and
   EM+acoustic fusion specifically for keystroke inference does not appear to be a heavily
   published combination.

**Load-bearing framing.** A DDC+FFT front-end alone only supports the claim that Alex can
tape out real DSP silicon — the attack classifies identically whether that DSP runs
on-chip or in numpy. To make the silicon necessary, the narrative anchors to a
self-contained, real-time, untethered side-channel appliance that captures EM+acoustic
and featurizes on-chip, so classification happens live at the edge with no PC in the
loop.

**CORDIC facts:** rotation mode does NCO/mixer/down-conversion; vectoring mode does
rect-to-polar (magnitude/phase), and the EM feature path may want both. Gain factor
K≈1.647 needs correction. Verification approach: use an SDR's own internal DDC output as
a reference to check the custom CORDIC DDC to within quantization tolerance.

**Front-end build order**, from `cordic_ddc_nco_mixer_datapath.png` in this folder: floor
(build now) is NCO (phase accumulator → CORDIC in rotation mode, sin/cos) → mixer
(complex multiply against EM input) → decimating FIR/CIC, where the K≈1.647 gain
correction is folded in. Stretch (in development) is input mux (EM-path output vs.
baseband acoustic input) → FFT (folded, single reused butterfly) → magnitude
(vectoring-mode CORDIC) → mel filterbank → log (hyperbolic-mode CORDIC) → log-mel frames
out to the ML recognizer.

Current ML progress and the full silicon build-order/verification detail live in
`CLAUDE.md` and `hardware/DDC_FRONTEND_SCOPE.md`, which are kept current; this file is
the stable project overview.
