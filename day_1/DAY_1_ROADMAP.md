# Day 1 roadmap — first real-data run

> Historical snapshot of the plan this session followed. Commands here predate the
> `training/` + `tools/` reorganization and the `--review` mode that `--auto`
> replaced; see `CLAUDE.md` for the current workflow.

Full end-to-end test of the 5-file acoustic-keystroke pipeline on real recordings.

**Shell setup (PowerShell).** Python isn't on PATH, so set a shortcut once per terminal:
```powershell
$py = "C:\Program Files\Python312\python.exe"
cd "C:\Users\PC\Documents\ML Keystroke Recognizer"
```

**Core strategy: pilot before you commit.** Don't record all 27 keys and *then* discover your
mic placement or detection threshold is wrong — you'd re-record everything. Record 3 keys first,
push them through Files 1–2, lock your parameters, *then* batch-record the rest.

---

## Phase 0 — Pre-flight (5 min)

- [ ] Re-confirm the unit tests still pass:
```powershell
foreach ($t in "test_isolate_keystrokes","test_features","test_model","test_train","test_evaluate") { & $py "$t.py" | Select-Object -Last 1 }
```
- [ ] Install matplotlib so `--plot` works (optional but strongly recommended for tuning):
```powershell
& $py -m pip install matplotlib
```
- [ ] Confirm you have **ffmpeg** — iPhone Voice Memos are `.m4a`, and the pipeline expects WAV.
  `ffmpeg -version`. If missing: `winget install ffmpeg`.

---

## Phase 1 — Record the 3-key pilot (10 min)

Setup that must stay **identical** for all 27 keys later: same room (quiet), same iPhone position
relative to the ThinkPad, same recording app/settings.

- [ ] On the ThinkPad, record iPhone audio of ~25 presses each of **just `a`, `s`, and `space`** —
  one continuous recording per key, ~1 s gaps between presses, that key only.
- [ ] Transfer the 3 `.m4a` files to the PC and drop them in **`unconverted_raw\`**, naming each file
  = the label (`a.m4a`, `s.m4a`, `space.m4a`). The space key's file must be literally `space.m4a`.
- [ ] **Crop each recording yourself** — cut the record/stop finger-taps off the front and back of
  every `.m4a` before converting. Nothing in the pipeline trims for you; an uncropped tap will be
  detected as a keystroke and poison that key's clips.
- [ ] Convert them with the project's converter script (format conversion + 48 kHz resample only,
  no trimming → `converted_wavs\`):
```powershell
./convert_recordings.ps1
```
**One sample rate for all 27 files** — the script pins 48 kHz because File 2 asserts a single rate
across the pool and will (correctly) reject a mix of 44.1k/48k. Output lands in `converted_wavs\` as
`a.wav`, `s.wav`, `space.wav`.

---

## Phase 2 — File 1 on the pilot: tune the detector (15 min)

- [ ] Run isolation over `converted_wavs\` with a plot and the expected count:
```powershell
& $py isolate_keystrokes.py --raw-dir converted_wavs --plot --expected 25
```
- [ ] Read the per-file line: `[OK   ]` means detected == 25; `[CHECK]` means it's off.
- [ ] If any key mis-counts, tune and re-run just that key:
  - **Over-counting** (detects 40 from 25): raise `--threshold-k` (try 6–8) or raise
    `--min-gap-ms` (double-strikes/reverb — try 400–500).
  - **Under-counting** (detects 12): lower `--threshold-k` (try 3–4).
```powershell
& $py isolate_keystrokes.py --raw-dir converted_wavs --only a --threshold-k 6 --min-gap-ms 400 --expected 25 --plot
```
  - Open the PNG it writes and eyeball the RMS envelope vs. the detected onset marks — that tells
    you *why* it's mis-firing.
- [ ] **Write down the `--threshold-k` / `--min-gap-ms` that gives OK on all three.** These are your
  locked params. Clips land in `clips\<label>_NN.wav`.

> ✅ Good: three `[OK   ]` lines, 25 clips each.
> 🚩 Red flag: every file wildly off in the same direction → mic level/placement issue, fix the
> recording setup now, not the threshold.

---

## Phase 3 — File 2 on the pilot: prove features build (5 min)

- [ ] Build a dataset from just the 3 keys:
```powershell
& $py features.py --out dataset_pilot.npz
```
- [ ] Confirm it prints the mel backend (should say `librosa`) and doesn't trip the
  single-sample-rate assert. Quick shape check:
```powershell
& $py -c "import features as ft; X,Y,l=ft.load_dataset('dataset_pilot.npz'); print(X.shape, X.min(), X.max(), l)"
```
> ✅ Good: `X` is `(~75, 64, 64)`, values in `[0,1]`, labels `['a','s','space']`. This confirms the
> whole raw→clips→dataset chain works on your real audio. **Stop and fix here if anything's wrong —
> before recording 24 more keys.**

---

## Phase 4 — Batch-record all 27 keys (60–90 min)

**No digits.** Scope is letters + space only — `0`–`9` are out of scope, don't record them.

- [ ] With params and setup locked, record the remaining 24 keys (`b-r`, `t-z` — everything except
  a/s/space you already have): 25 presses each, same conditions.
- [ ] Drop all the new `.m4a` files in `unconverted_raw\` (named `<label>.m4a`) and run
  `./convert_recordings.ps1` again to convert them into `converted_wavs\<label>.wav` at 48 kHz (same
  as the pilot). Classes total: `a`–`z`, `space` = **27**.

---

## Phase 5 — File 1 full run (10 min)

**Use review mode.** Every recording is a bit different, and one global `--threshold-k` will not fit
all 27. Review mode steps through the files one at a time, lets you tune each one live, and **writes
nothing until you accept** — so you never generate clips at the wrong settings and have to clean them
up afterwards:
```powershell
& $py isolate_keystrokes.py --review --raw-dir converted_wavs
```

For each file it prints the RMS envelope as an ASCII plot with the threshold line and the detected
onsets marked, the detected-vs-expected count, the min/median/max inter-onset gaps, and a suggestion
when the count is off. Then:

| command | effect |
|---|---|
| `k <val>` | set `threshold-k` — higher = fewer detections |
| `g <val>` | set `min-gap-ms` — higher = more debounce, kills double-strikes |
| `e <val>` | set the expected count for this file |
| `+` / `-` | nudge `threshold-k` by ±0.5 |
| `a` or Enter | **accept**: write this key's clips at the current settings, go to next file |
| `s` | skip, write nothing |
| `b` | back to the previous file |
| `l` | list onset times, gaps and per-clip coverage (spot double-strikes / missed presses / misaligned onsets) |
| `p` | save and open the full matplotlib PNG |
| `r` | reset to the session's starting parameters |
| `q` | quit — nothing further is written |

Three things it does for you:
- **Accepting replaces that key's clips wholesale.** Re-accept `a` at a stricter `k` and the previous
  run's extra clips are deleted, not left behind as orphans recorded under different parameters.
- **Parameters carry forward** to the next file (the recording setup is constant, so the last key's
  settings are the best starting guess).
- **Accepted parameters are saved** to `clips\isolation_params.json` per key, and reloaded when you
  revisit that key or re-run the tool. That file is also your record of how each key was extracted —
  worth keeping for the write-up.

> Real example from the `a` recording: default `k=5` gave 33 detections for 25 presses; `k 8` gave
> exactly 25. `space` at the same `k=8` also gave 25. So expect per-key values in that range, not the
> default.

**Clip coverage — watch this, not just the count.** Clips are a fixed 14400 samples (45 ms before the
onset, 255 ms after); nothing measures per-keystroke duration, and it shouldn't, because `features.py`
stretches every clip to 64 time frames — per-keystroke windows would normalise away duration
differences that may themselves be discriminative. What *can* go wrong is an onset firing early on
room noise, leaving the keystroke at the very tail of its clip. Review mode reports the fraction of
each press's energy its clip actually captures, and flags anything under 75%:

```
  clip coverage: median 95%   worst 20%   <-- 4 below 75%, check with 'l'
```

A right count with several low-coverage clips is worse than a slightly wrong count with clean ones —
those clips are mostly room noise filed under a real key. `l` shows which ones.

> Measured on the real recordings: `space`, `m`, `z` capture 98%+ of their energy and are fine.
> `a` and `q` sit at ~88% because their key-*release* click lands past the 300 ms window — accepted,
> and left that way deliberately to stay faithful to the paper's 14400 samples.
>
> ⚠️ **`e` has a genuine problem.** It is clean for its first 10 presses (94–100% coverage, ~1400 ms
> gaps), then from t≈14.8s produces a burst of extra detections with 20–43% coverage and 350–535 ms
> gaps. No setting fixes it: `k 8`→28 detections, `k 12`→27, `min-gap 600`→26, and 3 bad clips
> survive every time. That's a contaminating sound in the recording itself, not a tuning problem.
> **Re-record `e`, or crop t≈14.8–17.5s out of the `.m4a` before converting.**

- [ ] Work through all 27, accepting each only when the count looks right.
- [ ] A few keys off by ±1–2 is fine; wildly-off keys aren't — use `l` to see whether it's a
  double-strike (one very short gap) or a missed press (one roughly doubled gap).

<details>
<summary>Batch mode (unchanged, if you'd rather do it non-interactively)</summary>

```powershell
& $py isolate_keystrokes.py --raw-dir converted_wavs --threshold-k <locked> --min-gap-ms <locked> --expected 25
```
Scan for `[CHECK]` lines and fix outliers with `--only <key>`. This writes immediately, which is
exactly the clean-up problem review mode exists to avoid.
</details>

---

## Phase 6 — File 2 full: build the pool (2 min)

- [ ] Build the real dataset (this `.npz` **is** "the pool"):
```powershell
& $py features.py --out dataset.npz
& $py -c "import features as ft,numpy as np; X,Y,l=ft.load_dataset('dataset.npz'); print(X.shape, len(l), np.bincount(Y).min(),'min per class')"
```
> ✅ Good: ~`(675, 64, 64)`, 27 labels, ≥ ~20 per class.

---

## Phase 7 — File 4: train (short smoke, then full)

- [ ] **Smoke run first** — 30 epochs to confirm it learns above chance (1/27 ≈ 3.7%) and doesn't
  crash on real data:
```powershell
& $py train.py --data dataset.npz --model cnn --epochs 30 --out ckpt_smoke.pt
```
  Look at the printed `val_acc` climbing above ~0.05–0.10. If it's stuck at chance, something's
  wrong with the data, not the run length.
- [ ] **Full run** — 1100 epochs. Defaults to the GPU now (`--device auto`):
```powershell
& $py train.py --data dataset.npz --model cnn --out checkpoint.pt
```
  (Start with `cnn`; once that works, optionally train `--model coatnet --out checkpoint_coatnet.pt`
  and compare.)

> ⏱️ **Measured** on this box (RTX 3070 Ti Laptop, 675 samples × 1100 epochs, best-of-3 with a
> warmup run per config, 2026-07-26):
>
> | config | cnn | coatnet |
> |---|---|---|
> | `--device cpu` | 12.3 min | 20.3 min |
> | `--device cuda` (default aug) | **2.1 min** | **5.4 min** |
> | `--device cuda --aug gpu` | 2.2 min | 5.3 min |
> | `--device cuda --aug gpu --deterministic` | 2.0 min | 5.3 min |
>
> So: GPU is ~6× on cnn, ~3.8× on coatnet, and the full schedule is a coffee break either way — no
> need to background it.
>
> Two things that table settles:
> - **`--aug gpu` buys nothing measurable at this pool size** (112 vs 118 ms/epoch — if anything
>   marginally *slower* on cnn). Augmentation was never the bottleneck; at batch 16 the run is
>   dominated by kernel-launch overhead. It exists for a much larger pool later. **Use the default.**
> - **`--deterministic` is free** here (111.7 vs 112.0 ms/epoch), so there is no reason not to use it
>   for anything you intend to publish.
>
> ⚠️ **What `--seed` does and does not guarantee.** It pins the split, the batch order, the numpy
> augmentation, the weight init and the dropout masks. So:
> - **Same seed, same device, same aug, on CPU → bit-identical runs.** Verified by a test.
> - **On CUDA, NOT by default** — cuDNN uses nondeterministic backward kernels and the autotuner can
>   pick different algorithms between runs (observed drift over 8 epochs: `max|Δw| ≈ 2.8e-05`).
>   **`--deterministic` fixes this**, also verified by a test, and costs nothing (above).
> - **A CPU run and a CUDA run at the same seed are NOT the same run on different hardware.** They
>   share the split, batch order, augmented pixels and initial weights — but dropout masks come from
>   each device's own RNG, so they diverge for reasons beyond float arithmetic. Don't quote them as
>   equivalent.
>
> **For the published baseline: `--device cuda --deterministic`.** Repeatable, and 2 minutes.
>
> **TF32** is left at torch's own defaults (matmul off) unless you pass `--tf32` / `--no-tf32` —
> silently dropping mantissa bits under a reproduction baseline is the kind of unrecorded difference
> this project can't afford.
>
> **`evaluate.py` defaults to `--device cpu`, deliberately unlike `train.py`.** CPU and CUDA
> inference differ by ~1e-4 in the logits, enough to flip an argmax on a near-tied sample, so an
> `auto` default would make your reported accuracy depend on whether a GPU was visible. Evaluation is
> a few hundred forward passes; the speed is worth nothing, the stability is worth a lot.
>
> Every checkpoint records a `run_env` block (device, torch/CUDA version, GPU name, aug path, the
> *effective* TF32 / cudnn.benchmark / cudnn.deterministic state, seed) so any number you quote later
> is traceable to the machine and precision that produced it.

---

## Phase 8 — File 5: evaluate (5 min)

- [ ] Evaluate on the checkpoint's stored held-out split:
```powershell
& $py evaluate.py --data dataset.npz --ckpt checkpoint.pt --cm confusion.png
```
- [ ] Read: overall **accuracy**, **macro-F1 (present classes)**, the per-key P/R/F1 table, and the
  **adjacent-key clustering** line — you want `frac on adjacent key` meaningfully **above** the
  chance baseline (the paper's signature that the model learned real acoustics). Open
  `confusion.png` — a bright diagonal is what you want.
- [ ] **Deliberately test the safety guard** (this is the pipeline's most important guarantee):
  change the dataset's *contents* so its hash changes, then confirm evaluate *refuses* instead of
  reporting a fake number against a stale `val_index`.
  > ⚠️ A plain `features.py` rebuild from the *same* clips will **not** trigger it — `build_dataset`
  > is deterministic (sorted globs, deterministic mel), so it produces a byte-identical `X,Y` and the
  > **same** hash. To actually change the hash you must change the clip set. Easiest safe way:
  > temporarily move one clip out, rebuild to a throwaway file, and evaluate that:
```powershell
mkdir _held; Move-Item clips\a_01.wav _held\      # remove one clip → different pool
& $py features.py --out dataset_mutated.npz
& $py evaluate.py --data dataset_mutated.npz --ckpt checkpoint.pt
Move-Item _held\a_01.wav clips\ ; Remove-Item _held, dataset_mutated.npz  # restore
```
  You should see `REFUSED: dataset hash mismatch …` on the mutated pool. That's correct behavior: the
  stored `val_index` only matches the exact dataset the checkpoint was trained on. Your real
  `dataset.npz` / `checkpoint.pt` pair is untouched by this test.

---

## Phase 9 — Iterate / grow the pool

- Low per-key accuracy → record more presses for the worst keys (from the P/R table), drop the new
  `.m4a` in `unconverted_raw\`, re-run `./convert_recordings.ps1`, then **re-run Files 1→2→4→5 on the
  whole pool** (never train on new clips alone — catastrophic forgetting).
- Compare `cnn` vs `coatnet` accuracy once both are trained.

---

**Realistic timeline:** Phases 0–3 (~35 min) → recording Phase 4 (~1–1.5 hr, the bulk) →
Phases 5–6 (~15 min) → training Phase 7 (background, possibly hours on CPU) → Phase 8 (~5 min).
Plan to **kick off the full train in the background and evaluate later** rather than waiting on it.
