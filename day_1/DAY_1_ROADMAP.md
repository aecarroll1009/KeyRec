# Day 1 roadmap — first real-data run

Historical record of the plan this session followed. Commands here predate the
`training/` + `tools/` reorganization and the `--review` mode that `--auto` replaced —
see `CLAUDE.md` for the current workflow.

Full end-to-end test of the 5-file acoustic-keystroke pipeline on real recordings.
Strategy: pilot three keys through the whole pipeline first to lock recording setup and
detector parameters, before batch-recording the remaining 24.

## Phase 0 — pre-flight

Confirmed the unit test suites passed, installed matplotlib for `--plot`, confirmed
ffmpeg was available for the `.m4a` → WAV conversion.

## Phase 1 — record the 3-key pilot

Recorded ~25 presses each of `a`, `s`, and `space` on the ThinkPad via iPhone, cropping
the record/stop finger-taps off each file before conversion, and converted to 48 kHz WAV
with the project's converter script.

## Phase 2 — tune the detector on the pilot

Ran isolation over the pilot with `--plot` and `--expected 25`, adjusting `--threshold-k`
and `--min-gap-ms` per file until each hit exactly 25 detections. Locked parameters:
default `k=5` gave 33 detections for `a`; `k=8` gave exactly 25, and worked for `space`
too.

## Phase 3 — prove features build

Built a dataset from the 3 pilot keys and confirmed the mel backend, single-sample-rate
assertion, and output shape all checked out before committing to the full recording
session.

## Phase 4 — batch-record all 27 keys

Recorded the remaining 24 letters with the locked setup, 25 presses each. Scope is
letters and space only — digits are out of scope. Total: 27 classes.

## Phase 5 — full isolation run

Extraction is per-file: one global `--threshold-k` does not fit all 27 recordings. The
`k` giving exactly 25 detections ranged from 4.5 to 26 across files recorded in a single
sitting. Clip coverage — the fraction of a press's energy its clip actually captures —
mattered more than raw count: several files hit the target detection count with one or
more clips under 75% coverage, which meant a misaligned onset rather than a correct
detection.

One recording (`e`) had a genuine defect: clean for its first 10 presses, then a burst of
extra low-coverage detections from roughly the 15-second mark on, caused by a
contaminating sound in the take itself. No threshold setting fixed it; it needed
re-recording.

## Phase 6 — build the pool

Built the real dataset from all 27 keys: (675, 64, 64), 27 labels, ≥20 samples per class.

## Phase 7 — train

Smoke-tested at 30 epochs to confirm the model learned above chance before committing to
the full 1100-epoch run. Trained both SmallCNN and CoAtNet on GPU.

Timing measured on this box (RTX 3070 Ti Laptop, 675 samples × 1100 epochs, best-of-3
with a warmup run per config):

| config | cnn | coatnet |
|---|---|---|
| `--device cpu` | 12.3 min | 20.3 min |
| `--device cuda` | 2.1 min | 5.4 min |
| `--device cuda --deterministic` | 2.0 min | 5.3 min |

GPU is roughly 6× on cnn, 3.8× on coatnet. `--deterministic` cost nothing measurable, so
there was no reason not to use it for the published baseline
(`--device cuda --deterministic`). `--aug gpu` also bought nothing measurable at this
pool size and was skipped in favor of the default CPU augmentation path.

Seed/determinism/TF32 guarantees and `evaluate.py`'s CPU-default rationale are documented
directly in `train.py` and `evaluate.py` now, rather than restated here.

## Phase 8 — evaluate

Evaluated on the checkpoint's stored held-out split: overall accuracy, macro-F1, per-key
precision/recall/F1, and the adjacent-key clustering check. Confirmed the dataset-hash
guard refuses to evaluate against a mutated dataset rather than reporting a stale number.

## Phase 9 — iterate / grow the pool

Plan for later sessions: record more presses for the worst-performing keys, re-run
extraction and feature-building on the whole pool, and retrain on the full accumulated
pool rather than the new clips alone, to avoid catastrophic forgetting.
