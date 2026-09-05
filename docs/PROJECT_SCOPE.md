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

Current ML progress lives in `CLAUDE.md`, which is kept current; this file is the stable
project overview.
