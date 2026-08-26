"""train.py  --  File 4 of the acoustic-keystroke reproduction pipeline.

Trains one of the File-3 classifiers on the File-2 dataset (.npz of 64x64
log-mel images) and saves the best checkpoint.

Hyperparameters (paper defaults, with the collapse fix from the spec):
  * optimizer: Adam  -- the Adaptive Moment Estimation algorithm, the standard
    deep-learning weight updater; not a person and not model-specific.
  * max learning rate: 5e-4. NOT 1e-3 -- the paper's run collapsed to random
    accuracy at 1e-3 after ~1100 epochs; 5e-4 fixes it.
  * linear LR anneal from the max down toward ~0 over all epochs.
  * 1100 epochs, batch 16, cross-entropy loss.
  * gradient clipping at norm 1.0 as collapse insurance.

Data handling constraints (from the spec):
  * per-CLASS stratified split -- a plain random split on 37 classes x ~25
    samples can zero out whole classes from train or val.
  * on-the-fly augmentation EACH epoch: zero-padded (not circular) spectrogram
    time-shift + SpecAugment. Fresh randomness every epoch; the stored val set
    is never augmented.
  * the checkpoint stores model_kind, state_dict, labels, val_index, a content
    hash of (X, Y), and n_samples. File 5 (evaluate.py) refuses to reuse
    val_index unless the hash still matches -- a silently rebuilt dataset would
    otherwise mis-index the held-out split and report fake accuracy.

Growing pool: keep accumulating recordings over time and ALWAYS retrain on the
whole accumulated pile. `dataset.npz` IS the pool -- rebuild it from every clip
in clips/ (features.py already scans them all) before each retrain. Training on
only the newest clips causes catastrophic forgetting.

GPU: `--device auto` (the CLI default) uses CUDA when a CUDA-enabled torch and a
visible GPU are both present, else CPU. An EXPLICIT `--device cuda` that cannot
be honoured is a hard error, never a silent CPU fallback -- a 1100-epoch run
that quietly lands on the CPU just looks like a hung job.

Two augmentation paths, because this is a reproduction baseline and results have
to stay comparable across machines:
  * default (`--aug cpu`): the original numpy augmentation, unchanged.
  * `--aug gpu`: a vectorised torch port of the same augmentation, run on-device
    with the pool resident in GPU memory. Same distribution, same semantics, but
    a different RNG stream -- so it is NOT step-for-step comparable to a cpu-aug
    run (their batch orders diverge from epoch 1 on). Use it for the many
    retrains of a growing pool, not for the baseline number.

Reproducibility, precisely (see train_model's docstring for the full statement):
same seed + same device + same aug repeats bit-for-bit on CPU; on CUDA only with
`--deterministic`. A CPU run and a CUDA run at the same seed are NOT the same run
on different hardware -- dropout masks come from each device's own RNG.
"""

import argparse
import hashlib
import os
import re
import sys

import numpy as np

import features as ft
from model import build_model, count_parameters

MAX_LR = 5e-4
EPOCHS = 1100
BATCH_SIZE = 16
VAL_FRAC = 0.2
GRAD_CLIP_NORM = 1.0
MIN_LR_FACTOR = 0.0       # linear anneal endpoint (fraction of MAX_LR at last epoch)


# ----------------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------------
def device_index(spec):
    """Return the GPU ordinal named by a device spec, or None for cpu/bare cuda.

    'cuda:1' -> 1, 'cuda' -> None (torch's current device), 'cpu' -> None.
    """
    spec = str(spec).strip().lower()
    if spec.startswith("cuda:"):
        return int(spec.split(":", 1)[1])
    return None


def resolve_device(spec="auto", verbose=True):
    """Resolve a device spec to a concrete torch device string, failing loudly.

    'auto' picks cuda when it is actually usable, else cpu. Anything else must
    be exactly 'cpu', 'cuda', or 'cuda:N' with N a real GPU on this box. Every
    rejection raises RuntimeError with a specific message, because the ways
    this goes wrong (CPU-only wheel / no GPU visible / bad ordinal / typo) need
    different fixes -- and because the alternative, silently training 1100
    epochs on the CPU, is indistinguishable from a hung job.
    """
    import torch

    # None (the library default) means auto; an empty string is a typo and
    # falls through to the validation below rather than being coerced to auto.
    spec = "auto" if spec is None else str(spec).strip().lower()
    cuda_built = torch.version.cuda is not None
    cuda_ready = cuda_built and torch.cuda.is_available()

    if spec == "auto":
        spec = "cuda" if cuda_ready else "cpu"
        if verbose and spec == "cpu":
            why = ("torch is the CPU-only build" if not cuda_built
                   else "no CUDA device is visible")
            print(f"device: auto -> cpu ({why})")

    if spec == "cpu":
        return spec

    # Reject typos ('gpu', 'cuda0', 'cuda:') here rather than letting them sail
    # through to a raw torch traceback several hundred lines of setup later.
    if spec != "cuda" and not re.fullmatch(r"cuda:\d+", spec):
        raise RuntimeError(
            f"unrecognised device {spec!r}; expected 'auto', 'cpu', 'cuda', or 'cuda:N'"
        )

    if not cuda_built:
        raise RuntimeError(
            f"--device {spec} requested but torch {torch.__version__} is the "
            "CPU-only build (no CUDA runtime compiled in). Install a CUDA "
            "wheel, e.g.:\n"
            "  python -m pip install --index-url "
            "https://download.pytorch.org/whl/cu130 torch==2.12.1+cu130"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {spec} requested but no CUDA device is visible to torch "
            f"(torch {torch.__version__}, CUDA {torch.version.cuda}). Check "
            "`nvidia-smi` and that the driver supports this CUDA version."
        )

    # Validate the ordinal HERE. Left to torch it surfaces much later, as an
    # "invalid device ordinal" traceback from whatever happened to touch the
    # device first -- in evaluate.py that was torch.load, inside the try block
    # that prints the dataset-hash REFUSED banner.
    idx = device_index(spec)
    n_gpu = torch.cuda.device_count()
    if idx is not None and idx >= n_gpu:
        raise RuntimeError(
            f"--device {spec} requested but this machine has {n_gpu} CUDA "
            f"device(s) (valid: cuda:0..cuda:{n_gpu - 1})"
        )

    if verbose:
        # Report the ordinal that was ASKED FOR, not current_device() -- naming
        # GPU 0 while running on GPU 1 is worse than printing nothing.
        shown = idx if idx is not None else torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(shown)
        print(f"device: {spec} -> [{shown}] {torch.cuda.get_device_name(shown)} "
              f"({total_b / 1024**3:.1f} GiB total, {free_b / 1024**3:.1f} GiB free), "
              f"torch {torch.__version__} / CUDA {torch.version.cuda}")
    return spec


# ----------------------------------------------------------------------------
# Dataset content hash -- the checkpoint <-> dataset contract (File 5 checks it)
# ----------------------------------------------------------------------------
def dataset_hash(X, Y):
    """SHA-256 over the raw bytes of X (float32) and Y (int64).

    Any change to the images or labels -- including a rebuild that reorders
    samples -- changes this digest, which is exactly what lets evaluate.py
    detect that a stored val_index no longer refers to the same rows.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(Y, dtype=np.int64).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Per-class stratified split
# ----------------------------------------------------------------------------
def stratified_split(Y, val_frac=VAL_FRAC, seed=0):
    """Return (train_index, val_index) with every class represented in both.

    For each class the samples are shuffled and a val_frac slice goes to val;
    a class with a single sample stays entirely in train (cannot be split).
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y)
    train_idx, val_idx = [], []
    for cls in np.unique(Y):
        idx = np.flatnonzero(Y == cls)
        rng.shuffle(idx)
        if len(idx) < 2:
            train_idx.extend(idx.tolist())      # too few to hold any out
            continue
        n_val = int(round(val_frac * len(idx)))
        n_val = min(max(n_val, 1), len(idx) - 1)  # at least 1 each side
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.sort(np.array(train_idx, dtype=np.int64)), np.sort(np.array(val_idx, dtype=np.int64))


# ----------------------------------------------------------------------------
# On-the-fly augmentation (image space)
# ----------------------------------------------------------------------------
def _time_shift_spec(img, rng, max_frac=ft.TIME_SHIFT_FRAC):
    """Shift a spectrogram image along the TIME axis (columns), zero-filling the
    exposed edge. Never wraps -- a circular roll would smear the keystroke's
    onset onto the opposite edge and corrupt the label's signature."""
    n_mels, n_frames = img.shape
    max_shift = int(round(max_frac * n_frames))
    if max_shift == 0:
        return img.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    out = np.zeros_like(img)
    if shift > 0:
        out[:, shift:] = img[:, : n_frames - shift]
    elif shift < 0:
        out[:, : n_frames + shift] = img[:, -shift:]
    else:
        out[:] = img
    return out


def augment_batch(batch, rng):
    """Apply a fresh zero-padded time-shift + SpecAugment to each image in a
    (B, H, W) batch. Returns a new float32 array; the input is untouched."""
    out = np.empty_like(batch, dtype=np.float32)
    for i in range(batch.shape[0]):
        img = _time_shift_spec(batch[i], rng)
        img = ft.spec_augment(img, rng=rng)
        out[i] = img
    return out


def augment_batch_torch(batch, gen, max_frac=ft.TIME_SHIFT_FRAC,
                        n_masks=ft.SPECAUG_MASKS_PER_AXIS, frac=ft.SPECAUG_FRAC):
    """Vectorised on-device port of `augment_batch` for a (B, H, W) tensor.

    Same semantics as the numpy path, no Python-level per-image loop:
      * time shift: one independent zero-filled column shift per image, drawn
        uniformly from [-max_shift, +max_shift]. Built with a gather over
        clamped source indices, then zeroed where the source fell off the edge
        -- so it never wraps, same as the numpy version.
      * SpecAugment: masks are computed on the SHIFTED image's mean and every
        mask writes that same constant, so applying their union in one write is
        equivalent to the numpy path's sequential (possibly overlapping) writes.

    Draws from `gen`, a torch Generator, so this does not consume the numpy RNG
    stream -- see the module docstring on why that makes it non-comparable to a
    CPU run.
    """
    import torch

    B, H, W = batch.shape
    dev = batch.device
    cols = torch.arange(W, device=dev).unsqueeze(0)          # (1, W)
    rows = torch.arange(H, device=dev).unsqueeze(0)          # (1, H)

    # --- zero-filled time shift ---
    max_shift = int(round(max_frac * W))
    if max_shift > 0:
        shift = torch.randint(-max_shift, max_shift + 1, (B, 1),
                              generator=gen, device=dev)
        src = cols - shift                                   # (B, W)
        valid = (src >= 0) & (src < W)
        src = src.clamp(0, W - 1)
        out = torch.gather(batch, 2, src.unsqueeze(1).expand(B, H, W))
        out = out * valid.unsqueeze(1)                       # zero fill, no wrap
    else:
        out = batch.clone()

    # --- SpecAugment (mean of the shifted image, as in the numpy path) ---
    mean_val = out.mean(dim=(1, 2), keepdim=True)            # (B, 1, 1)
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=dev)

    freq_w = max(1, int(round(frac * H)))
    for _ in range(n_masks):
        f0 = torch.randint(0, max(1, H - freq_w + 1), (B, 1), generator=gen, device=dev)
        band = (rows >= f0) & (rows < f0 + freq_w)           # (B, H)
        mask |= band.unsqueeze(2)

    time_w = max(1, int(round(frac * W)))
    for _ in range(n_masks):
        t0 = torch.randint(0, max(1, W - time_w + 1), (B, 1), generator=gen, device=dev)
        band = (cols >= t0) & (cols < t0 + time_w)           # (B, W)
        mask |= band.unsqueeze(1)

    return torch.where(mask, mean_val.expand_as(out), out)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _accuracy(torch, model, X_sub, Y_sub, device, batch_size=64):
    """Plain top-1 accuracy over a subset, no augmentation, eval mode."""
    if len(X_sub) == 0:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(X_sub), batch_size):
            xb = X_sub[start:start + batch_size]
            xb = torch.from_numpy(xb).unsqueeze(1).to(device)  # (b,1,64,64)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            correct += int((pred == Y_sub[start:start + batch_size]).sum())
    return correct / len(X_sub)


def _accuracy_dev(torch, model, Xv, Yv, batch_size=64):
    """Top-1 accuracy over a val set already resident on the target device.

    Same maths as `_accuracy`, but the val images are uploaded once before the
    epoch loop instead of once per epoch -- at 1100 epochs that host-to-device
    copy is pure overhead, and the comparison stays on-device so there is no
    per-batch sync back to the host either.
    """
    n = Xv.shape[0]
    if n == 0:
        return 0.0
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=Xv.device)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = Xv[start:start + batch_size].unsqueeze(1)     # (b,1,64,64)
            pred = model(xb).argmax(dim=1)
            correct += (pred == Yv[start:start + batch_size]).sum()
    return int(correct.item()) / n


def train_model(X, Y, labels, kind="cnn", epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=MAX_LR, val_frac=VAL_FRAC, seed=0, device="cpu",
                out_path="checkpoint.pt", min_lr_factor=MIN_LR_FACTOR, verbose=True,
                aug="cpu", tf32=None, deterministic=False):
    """Train `kind` on (X, Y) and save the best-val checkpoint to out_path.

    `device` is taken as already-resolved (the CLI runs it through
    resolve_device first); the default stays "cpu" so library callers and the
    tests keep the original behaviour.

    `aug` selects the augmentation path -- "cpu" (numpy) or "gpu" (vectorised
    on-device, faster, different RNG stream). "gpu" on a cpu device falls back
    to "cpu" with a warning.

    `tf32` is tri-state: None (default) leaves torch's own TF32 settings alone,
    True/False force them on/off. It is NOT defaulted to True -- torch ships
    matmul TF32 off, and silently dropping mantissa bits under a reproduction
    baseline is exactly the kind of unrecorded difference this project cannot
    afford.

    `seed` seeds BOTH the numpy RNG (split + cpu augmentation) and torch's RNG
    (weight init + dropout). What that does and does not buy you:
      * same seed, same device, same aug -> bit-identical runs on CPU. On CUDA
        NOT by default: cuDNN picks nondeterministic backward kernels, and
        cudnn.benchmark (enabled below) can select different algorithms between
        runs. Pass deterministic=True to trade speed for exact repeatability.
      * a CPU run and a CUDA run at the same seed share the split, the batch
        order, the augmented pixels and the INITIAL WEIGHTS -- but not the
        dropout masks, which are drawn from each device's own RNG. So the two
        diverge for reasons beyond conv float arithmetic. Do not quote them as
        the same run on different hardware.

    Returns (best_val_acc, checkpoint_dict).
    """
    import torch
    import torch.nn.functional as Fnn

    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.int64)
    n_samples = X.shape[0]
    data_h = dataset_hash(X, Y)

    on_cuda = str(device).startswith("cuda")
    gpu_aug = (aug == "gpu") and on_cuda
    if aug == "gpu" and not on_cuda and verbose:
        print("warning: --aug gpu ignored on a cpu device; using the numpy path")

    if on_cuda:
        if deterministic:
            # cuDNN's default backward kernels are nondeterministic, and the
            # autotuner can pick different algorithms between runs -- so BOTH
            # have to go for a GPU run to repeat itself bit-for-bit. Costs
            # speed, which is the whole trade.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Fixed 64x64 input and a single model shape, so cudnn's autotuner
            # pays for itself many times over across 1100 epochs.
            torch.backends.cudnn.benchmark = True
        # TF32 trades mantissa bits for tensor-core throughput on Ampere. Only
        # touched when explicitly asked; None leaves torch's defaults intact.
        if tf32 is not None:
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)

    train_idx, val_idx = stratified_split(Y, val_frac, seed)
    rng = np.random.default_rng(seed)
    # Seed torch too. Without this the split and the augmentation are
    # reproducible but the weight init and dropout masks are not, so two runs
    # at the same --seed can land on very different accuracies -- which makes
    # any single quoted number unreproducible.
    torch.manual_seed(seed)
    if on_cuda:
        torch.cuda.manual_seed_all(seed)

    model = build_model(kind, num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Stage the val set on-device once; it is never augmented and never changes.
    Xv = torch.from_numpy(X[val_idx]).to(device)
    Yv = torch.from_numpy(Y[val_idx]).to(device)

    # Whole-pool residency for the gpu-aug path: (675, 64, 64) float32 is ~11 MB,
    # so the entire dataset lives on the GPU and no batch is ever uploaded.
    X_dev = Y_dev = gen = None
    if gpu_aug:
        X_dev = torch.from_numpy(X).to(device)
        Y_dev = torch.from_numpy(Y).to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    # Linear anneal, set explicitly at the START of each epoch: factor 1.0 at
    # epoch 0 decaying to lr*(1/epochs)*... toward min_lr_factor at the end.
    # Dividing by `epochs` (not epochs-1) keeps the last epoch's LR small but
    # strictly positive -- no wasted zero-LR epoch and never negative.
    def _lr_at(epoch):
        factor = 1.0 - (1.0 - min_lr_factor) * (epoch / epochs)
        return lr * max(factor, min_lr_factor)

    if verbose:
        print(f"model={kind}  params={count_parameters(model):,}  "
              f"samples={n_samples}  train={len(train_idx)}  val={len(val_idx)}  "
              f"classes={len(labels)}  device={device}  "
              f"aug={'gpu' if gpu_aug else 'cpu'}  seed={seed}"
              + (f"  tf32(cudnn/matmul)="
                 f"{'on' if torch.backends.cudnn.allow_tf32 else 'off'}/"
                 f"{'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}"
                 f"  deterministic={'on' if deterministic else 'off'}"
                 if on_cuda else ""))

    # Run provenance, built once. Records the EFFECTIVE backend state read back
    # from torch rather than what was requested, and names the GPU actually
    # selected by `device` rather than current_device() -- the whole point of
    # this block is that a number quoted in a write-up stays traceable to the
    # hardware and precision that produced it.
    _gpu_idx = device_index(device)
    if _gpu_idx is None and on_cuda:
        _gpu_idx = torch.cuda.current_device()
    run_env = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(_gpu_idx) if on_cuda else None,
        "aug": "gpu" if gpu_aug else "cpu",
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if on_cuda else None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if on_cuda else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if on_cuda else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if on_cuda else None,
        "seed": int(seed),
    }

    best_acc = -1.0
    best_ckpt = None
    # Per-epoch trace. Kept because the single "best val_acc" number hides how
    # it was obtained -- it is the MAXIMUM of `epochs` noisy measurements on the
    # same small val set, so the gap between this curve's tail and its peak is
    # the selection optimism in the headline figure.
    history = {"epoch": [], "loss": [], "val_acc": [], "lr": []}

    for epoch in range(epochs):
        cur_lr = _lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        model.train()
        # Both paths draw the permutation from `rng`, but only the cpu path
        # also draws its augmentation from it -- so the two streams desync after
        # epoch 0 and the batch orders diverge from epoch 1 on. Do NOT read this
        # as "same batches, different augmentation": it holds for epoch 0 only.
        # Left as-is deliberately; giving augmentation its own generator would
        # make the claim true but change every CPU result produced so far.
        perm = rng.permutation(train_idx)
        for start in range(0, len(perm), batch_size):
            bidx = perm[start:start + batch_size]
            if gpu_aug:
                bt = torch.from_numpy(bidx).to(device)
                xb = augment_batch_torch(X_dev[bt], gen).unsqueeze(1)  # (b,1,64,64)
                yb = Y_dev[bt]
            else:
                xb = augment_batch(X[bidx], rng)                    # fresh aug each epoch
                xb = torch.from_numpy(xb).unsqueeze(1).to(device)   # (b,1,64,64)
                yb = torch.from_numpy(Y[bidx]).to(device)

            opt.zero_grad()
            loss = Fnn.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

        val_acc = _accuracy_dev(torch, model, Xv, Yv)
        history["epoch"].append(int(epoch))
        history["loss"].append(float(loss.detach()))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(cur_lr))

        if val_acc >= best_acc:
            best_acc = val_acc
            best_ckpt = {
                "model_kind": kind,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "labels": list(labels),
                "val_index": np.array(val_idx, dtype=np.int64),
                "train_index": np.array(train_idx, dtype=np.int64),
                "data_hash": data_h,
                "n_samples": int(n_samples),
                "epoch": epoch,
                "val_acc": float(val_acc),
                # Run provenance -- which machine/precision/aug path produced
                # this number. Extra keys, ignored by evaluate.py, but a run
                # that ends up in a write-up should be self-describing.
                "run_env": run_env,
            }
            if out_path:
                torch.save(best_ckpt, out_path)

        if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}/{epochs}  loss={float(loss.detach()):.4f}  "
                  f"val_acc={val_acc:.3f}  best={best_acc:.3f}  lr={cur_lr:.2e}")

    # The best checkpoint was written mid-run, so its history was truncated at
    # whatever epoch happened to be the peak. Attach the FULL trace and re-save
    # so the file describes the whole run, not just the part before the peak.
    if best_ckpt is not None:
        best_ckpt["history"] = history
        if out_path:
            torch.save(best_ckpt, out_path)

    if verbose:
        tail = history["val_acc"][-100:]
        print(f"best val_acc={best_acc:.3f}  saved -> {out_path}")
        if tail:
            print(f"  (last 100 epochs: mean {float(np.mean(tail)):.3f}, "
                  f"max {float(np.max(tail)):.3f} -- the gap to `best` is selection optimism)")
    return best_acc, best_ckpt


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Train the acoustic-keystroke classifier.")
    p.add_argument("--data", default="dataset.npz", help="input dataset .npz (the pool; default: dataset.npz)")
    p.add_argument("--model", default="cnn", choices=["cnn", "coatnet"], help="model kind")
    p.add_argument("--out", default="checkpoint.pt", help="output checkpoint path")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=MAX_LR)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto",
                   help="auto (default: cuda if usable, else cpu), cpu, cuda, or cuda:N. "
                        "An explicit cuda that cannot be honoured is an error, not a fallback.")
    p.add_argument("--aug", default="cpu", choices=["cpu", "gpu"],
                   help="augmentation path: cpu = numpy, comparable to a CPU run at the "
                        "same seed (default); gpu = on-device, faster, different RNG stream")
    # Tri-state on purpose: the default must not touch torch's own TF32
    # settings. Forcing it on would silently drop mantissa bits under a
    # reproduction baseline; forcing it off would misreport the machine's
    # actual behaviour. Both directions stay available, neither is assumed.
    p.add_argument("--tf32", dest="tf32", action="store_const", const=True, default=None,
                   help="force TF32 on for cuda runs (faster, fewer mantissa bits)")
    p.add_argument("--no-tf32", dest="tf32", action="store_const", const=False,
                   help="force TF32 off for cuda runs (closer to CPU float behaviour)")
    p.add_argument("--deterministic", action="store_true",
                   help="make cuda runs bit-reproducible run-to-run (cudnn.deterministic on, "
                        "autotuner off). Measured cost on this pool: none. Use it for any "
                        "number you intend to publish.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.data):
        print(f"error: dataset not found: {args.data}", file=sys.stderr)
        return 2

    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    X, Y, labels = ft.load_dataset(args.data)
    print(f"loaded pool: {X.shape[0]} clips, {len(labels)} classes from {args.data}")
    train_model(X, Y, labels, kind=args.model, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac,
                seed=args.seed, device=device, out_path=args.out,
                aug=args.aug, tf32=args.tf32, deterministic=args.deterministic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
