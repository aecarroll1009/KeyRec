# Day 1 — first full 27-key run (2026-07-26)

Snapshot of the first complete pass through the Phase-1 pipeline: 27 classes
(`a`–`z` + space), 25 presses each, 675 clips, both architectures trained.

## Results

| | SmallCNN | CoAtNet |
|---|---|---|
| parameters | 96,379 | 385,499 |
| **best val accuracy** | 92.6% | **98.5%** |
| macro-F1 | 0.9256 | 0.9850 |
| last-100-epoch mean | 89.0% | 95.9% |
| selection gap (best − mean) | +3.6 pp | +2.7 pp |
| errors (of 135) | 10 | **2** |
| errors on an adjacent key | 9 / 10 (90%) | 2 / 2 (100%) |
| chance baseline for adjacency | 16.2% | 16.2% |
| wall time (1100 epochs, RTX 3070 Ti) | 2m10s | ~5m |

Both reproduce the paper's signature result — **errors concentrate on physically
adjacent keys far above chance**. CoAtNet's only two mistakes were `c`→`x` and
`k`→`m`, both neighbours.

CoAtNet is the architecture the paper itself used, so its 98.5% is the number
directly comparable to the paper's ~95%; SmallCNN is the simpler in-house baseline.

## ⚠️ These numbers are optimistic — do not quote them as held-out results

`train.py` saves the checkpoint whenever validation accuracy improves, so the
headline is **the maximum of 1100 noisy measurements on the same 135-sample val
set**, and `evaluate.py` then re-scores that same split. The reported number is
the criterion the checkpoint was selected by.

Two measurements of the size of the problem:

* **Seed spread** — 4 seeds of SmallCNN, identical data: 0.911 / 0.926 / 0.933 /
  0.941. Mean 92.8%, σ = 1.3 pp. So the headline is ~93% ± 1, not 92.6%.
* **Selection gap** — best epoch vs. the last-100-epoch mean: +3.6 pp (CNN),
  +2.7 pp (CoAtNet). Visible in `training_curves.png` as the dot sitting above
  its own curve.

Realistic held-out estimates: **SmallCNN ≈ 89–90%, CoAtNet ≈ 95–96%.**

Per-key precision/recall from this run is noisier still — only **5 val samples
per class**, so recall moves in 20% steps. Do not read much into any single key.

The fix is `corpus/` (in the project root, currently empty): clips from a session
that never touched training or model selection, scored once. Not yet wired into
`evaluate.py`.

## Contents

| file | what it is |
|---|---|
| `dataset.npz` | the pool this run trained on — (675, 64, 64), 27 classes, 25/class |
| `checkpoint.pt` | SmallCNN weights + `val_index` + dataset hash + `run_env` + per-epoch history |
| `checkpoint_coatnet.pt` | CoAtNet, same structure |
| `training_curves.png` | validation accuracy vs epoch, both models |
| `confusion_cnn.png` | labelled confusion matrix, SmallCNN |
| `confusion_coatnet.png` | labelled confusion matrix, CoAtNet |
| `confusion_*_raw.png` | the unlabelled PNGs `evaluate.py` writes itself (numpy+zlib, no matplotlib per spec) |
| `test day 1.md` | the roadmap this session followed |

`training_clips/` and `corpus/` deliberately stay in the project root — they are
live and growing, not part of this snapshot.

## Reproducing this run

The clips are in `training_clips/session01/`, with per-key detector settings in
that folder's `isolation_params.json`. From the project root:

```powershell
$py = "C:\Program Files\Python312\python.exe"
& $py features.py --out day_1/dataset.npz
& $py train.py --data day_1/dataset.npz --model cnn     --deterministic --out day_1/checkpoint.pt
& $py train.py --data day_1/dataset.npz --model coatnet --deterministic --out day_1/checkpoint_coatnet.pt
& $py plot_training.py --ckpt "day_1/checkpoint.pt:SmallCNN" --ckpt "day_1/checkpoint_coatnet.pt:CoAtNet" --out day_1/training_curves.png
```

`--deterministic` makes CUDA runs bit-reproducible and costs nothing measurable
on this workload; the CNN was re-run mid-session and reproduced bit-identically.

## What this run established

* Per-key detector tuning is mandatory — the `threshold-k` giving exactly 25
  detections ranged from **4.5 to 26** across files recorded in one sitting.
* Six recordings had to be re-made (`a`, `e`, `p`, `q`, `v`, `w`); `p` twice,
  because the first re-record was noisier than the original.
* CoAtNet beats SmallCNN decisively at this pool size, which was **not** the
  expected outcome — 4× the parameters on 540 training samples was predicted to
  overfit. It reached ~90% within 100 epochs, a level the CNN needed ~900 to
  match, and its curve is markedly less noisy. Its training loss reached ~0
  (it fits the training set perfectly) while still generalising, so the extra
  capacity is not being spent on memorisation in a way that hurts.
