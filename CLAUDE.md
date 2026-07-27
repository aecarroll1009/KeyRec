# ML Keystroke Recognizer — Project Scope

Alex's sole flagship project: reproduce a 2023 acoustic side-channel keyboard-inference
paper, extend it with an EM/Van Eck side channel, fuse the two, and back the EM channel
with a custom silicon tapeout. Spans acquisition → DSP → ML → custom silicon → published
research.

## Phases

1. **Reproduce first.** Recreate the 2023 paper (mel-spectrogram features into a
   CNN/CoAtNet-style classifier, ~95% reported accuracy from keystroke audio, including
   audio captured over a call mic) as closely as possible before diverging, so any later
   improvement is measurable against a known baseline. Concrete five-file build spec for
   this phase: see `instructions from work.txt` in this folder (the original spec written
   at Alex's GD internship) and the file-by-file progress notes below.

2. **Generalize.** Extend from one typist/one keyboard to multiple typists on one
   keyboard, then to different keyboard families (mechanical, ThinkPad-style chiclet, Mac
   chiclet/scissor, etc.) — cross-keyboard/cross-typist generalization was the weak point
   of the original paper.

3. **EM+acoustic sensor fusion (Alex's own idea, not in the paper).** Add a second,
   independent side channel — EM/Van Eck emissions from the keyboard controller, captured
   via SDR (a HamGeek XC7Z010 board, firmware-hacked AD9363→AD9361 behavior, ~$65) — as
   its own classifier, then a fusion/arbitration layer that reconciles the acoustic
   model's and EM model's per-keystroke predictions. Disagreement between the two
   channels is treated as informative rather than noise, since they have different,
   largely uncorrelated failure modes (acoustic degrades with ambient noise/distance/
   chiclet-style keys; EM degrades with shielding/distance/controller design). This is a
   multimodal late-fusion architecture (same family as camera+LIDAR fusion in robotics);
   EM+acoustic fusion for keystroke inference doesn't appear to be a heavily published
   combination — a potential genuine differentiator, possibly workshop-paper-worthy with
   a rigorous generalization study.

## Silicon scope (decided 2026-07-17)

**One tapeout: a CORDIC + FFT front-end chip.** Originally explored as a separate
"modem/CORDIC tapeout" resume project — that standalone project is dropped, and the
hardware is folded directly into this project instead, because the load-bearing test
(the silicon must sit on the critical path, not be bolted on) only passes once the
CORDIC's output is literally the EM channel's DDC front-end.

**"Streaming log-mel spectral front-end"** — one die combining:
- **CORDIC DDC**: rotation-mode NCO + mixer → decimating FIR, down-converts captured
  EM/Van Eck emissions to baseband IQ.
- **Folded FFT → mel-filterbank → log** feature engine (the log stage can reuse a CORDIC
  in hyperbolic mode).

One streaming datapath: raw EM samples in → log-mel spectrogram frames out. Acoustic
audio is already baseband, so it bypasses the DDC via a mux and enters directly at the
FFT stage — the two channels share the back half of the chip in hardware, not just in
concept.

Keep the FFT small/folded (modest point size, single reused butterfly,
memory-scheduled) — the combined front-end is too big for Tiny Tapeout (the iterative
CORDIC alone would fit) and needs a real MPW shuttle. **Verify which free student MPW
shuttle is actually open before planning** — that landscape shifted a lot in 2025.

CORDIC technical facts: rotation mode = NCO/mixer/down-conversion; vectoring mode =
rect-to-polar (magnitude/phase) — the EM feature path may want both. Gain-factor
K≈1.647 correction needed. Design-depth talking points: iterative-vs-pipelined
area/throughput tradeoff, bit-width/precision management, angle range reduction.
Verification idea if real RF hardware is added: use an SDR's own internal DDC output as
a golden reference to check the custom CORDIC DDC to within quantization tolerance.

**CNN inference accelerator (second chip) — benched, not active.** Considered as a
possible companion tapeout (small MAC/systolic array streaming the SmallCNN) but parked
to keep scope focused on the front-end chip. Clean future phase-2 if revived — its input
would be exactly this chip's spectrogram-frame output. If revived: separate die from the
front-end (risk/schedule isolation on scarce student shuttle slots), designed as one
integrated RTL/FPGA system but fabricated as two dies, joined by a valid/ready
AXI-Stream-like frame interface.

**Load-bearing framing:** a DDC+FFT front-end alone only proves "I can tape out real DSP
silicon" — the attack classifies identically whether that DSP runs on-chip or in numpy.
To make the silicon *necessary*, anchor the narrative to a self-contained, real-time,
untethered side-channel appliance that captures EM+acoustic and featurizes on-chip so
classification happens live at the edge, no PC in the loop.

## Priority & career framing

This is Alex's primary, most-invested project among his RF/security hobby work — ranked
above his TEMPEST-video (HDMI/laptop-screen Van Eck) and RollJam projects, which he
considers secondary/fun side things. Deliberate career-strategy bridge: pairs
DSP/signal-processing/silicon-design skill with hardware side-channel security, directly
relevant to his goal of an Apple Hardware role (Secure Enclave/anti-tamper silicon teams
study exactly this class of leakage).

## Publication goal

Genuine independent/student research, not just a portfolio repo. Plan: (1) arXiv
preprint once the reproduction + fusion work is solid enough to write up, for a citable
public timestamp regardless of venue outcome; (2) target USENIX WOOT (Workshop on
Offensive Technologies) as the realistic first peer-reviewed venue, since it's built for
exactly this kind of practical-attack work — IEEE HOST and EuroS&PW (the original
acoustic paper's own venue) are backup targets. Also exploring a UW affiliation via the
Allen School's Security and Privacy Research Lab (seclab.cs.washington.edu —
Roesner/Kohno/Tyagi as of 2026) for a faculty advisor, independent-study credit, and —
critically — IRB approval, needed before collecting audio/EM data from any typist other
than Alex for the generalization phase.

## Practical implementation notes

Needs a synchronized capture pipeline (audio interface + SDR EM capture, with a
calibrated fixed-delay offset since EM propagates ~instantly vs. sound at ~343 m/s) and
ground-truth keylogging on the test machine to label both streams per keystroke.

## Work/home air-gap

Alex is a UW Seattle student interning at General Dynamics in San Jose (summer 2026). GD
policy bars transferring digital files between work and personal machines, so any spec
that originates at work moves via non-digital channels (printed then scanned) rather
than email/cloud — this is why `instructions from work.txt` exists as a written spec
rather than original code, and why any future work-side updates will likely arrive the
same way.

## Git

Alex runs all `git commit`/`git push` himself — prepare and hand off commands/diffs
rather than committing or pushing on his behalf, unless he explicitly says otherwise for
a specific commit.

## Phase-1 build progress

Five-file pipeline (isolate_keystrokes.py → features.py → model.py → train.py →
evaluate.py). **All five files built and passing as of 2026-07-05**, survived an Opus
critic audit (2026-07-06) with two hardening fixes applied (WAV chunk-size clamping,
hash-guard reorder test). GPU migration done 2026-07-26 (torch 2.12.1+cu130, RTX 3070 Ti
Laptop, sm_86); a second Opus critic audit that day found 7 real defects in the CUDA work,
all fixed. `day_1/` holds that session's roadmap, dataset snapshot, checkpoints and plots.

**First full run, 2026-07-26.** 27 classes (a–z + space), 25 presses each, 675 clips.
Both architectures trained; full detail and figures in `day_1/README.md`.

| | SmallCNN (96,379 p) | CoAtNet (385,499 p) |
|---|---|---|
| best val accuracy | 92.6% | **98.5%** |
| errors (of 135) | 10 | 2 |
| on an adjacent key | 9/10 (90%) | 2/2 (100%), vs 16.2% chance |

The paper's signature result — errors concentrating on physically adjacent keys far
above chance — reproduced by both. **CoAtNet is the paper's own architecture**, so its
98.5% is the figure comparable to the paper's ~95%; SmallCNN is the in-house baseline.
CoAtNet won decisively, which was *not* predicted (4× params on 540 training samples was
expected to overfit) — it hit ~90% within 100 epochs, a level the CNN needed ~900 to reach.

⚠️ **Both figures are optimistic and must not be quoted as held-out results.** Each is the
maximum of 1100 noisy measurements on the same 135-sample val set, and `evaluate.py` then
re-scores that same split — the number reported is the criterion the checkpoint was
selected by. Measured: seed-to-seed spread ±1.3 pp (4 CNN seeds, 0.911–0.941), and the
best-epoch-vs-last-100-mean gap is +3.6 pp (CNN) / +2.7 pp (CoAtNet). Realistic held-out:
**CNN ≈ 89–90%, CoAtNet ≈ 95–96%.** Per-key P/R is noisier still — 5 val samples per class,
so recall moves in 20% steps. Fixing this is what `corpus/` is for — score it once, never
select on it. **Not yet wired into evaluate.py.**

## Clip extraction — the established procedure

**Read the module docstring at the top of `isolate_keystrokes.py` before extracting
anything.** It documents the full workflow and the failure modes, all of which were hit
for real on 2026-07-26. The three that cause silent damage:

1. **Never reuse one key's parameters for another.** The `k` yielding exactly 25 ranged
   from **4.5 to 26** across 27 files recorded in one sitting. `--auto` searches each file
   independently — keep it that way.
2. **Judge on coverage, not just count.** 25 detections with several low-coverage clips is
   worse than 24 clean ones; a low-coverage clip is mostly room noise filed under a key.
   A narrow `kwin` (grid steps yielding the target) means fragile — every `kwin=1` file
   turned out to have a genuine recording defect.
3. **Growing the pool needs `--session`.** `write_clips(replace=True)` deletes every clip
   for a label in its target directory, so extracting a new session into an existing one
   DESTROYS the old clips. Sessions get their own subdirectory; `features.py` scans
   recursively so they accumulate.

**Layout — all three stages are subdivided by session, and the names must match:**

```
unconverted_raw/day_1/*.m4a  ->  converted_wavs/day_1/*.wav  ->  training_clips/day_1/*.wav
corpus/          held-out test material, kept strictly out of training
day_1/           per-session snapshot: dataset.npz, checkpoints, figures, README
```

A new session is `day_2` etc. throughout. `features.py` scans `training_clips`
**recursively**, so every session joins the pool automatically. `--raw-dir` is
deliberately NOT recursive — each extraction run handles exactly one session.

```powershell
./convert_recordings.ps1 -Session day_2
& $py isolate_keystrokes.py --auto --dry-run --raw-dir converted_wavs/day_2 --expected 25
& $py isolate_keystrokes.py --auto --raw-dir converted_wavs/day_2 --expected 25 --session day_2
& $py features.py --out day_2/dataset.npz      # picks up day_1 AND day_2
```

Per-key settings are recorded in each session's `isolation_params.json` — that is the
extraction record for the write-up.
