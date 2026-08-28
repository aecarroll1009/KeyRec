"""train.py -- file 4 of the acoustic-keystroke reproduction pipeline.

Trains one of the file-3 classifiers on the file-2 dataset (.npz of 64x64
log-mel images) and saves the best-val checkpoint.

See each function's docstring for hyperparameters, device resolution, and
reproducibility details.
"""

import argparse
import hashlib
import os
import re
import sys

import numpy as np

import features as ft
from model import build_model, count_parameters

MAX_LR = 5e-4             # 1e-3 collapses training to random accuracy over 1100 epochs
EPOCHS = 1100
BATCH_SIZE = 16
VAL_FRAC = 0.2
GRAD_CLIP_NORM = 1.0
MIN_LR_FACTOR = 0.0       # linear anneal endpoint (fraction of MAX_LR at last epoch)


# ----------------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------------
def device_index(spec):
    """Return the GPU ordinal named by a device spec.

    Args:
        spec: a device spec such as 'cpu', 'cuda', or 'cuda:N'.

    Returns:
        The integer N for 'cuda:N'; None for 'cuda' (torch's current
        device) or 'cpu'.
    """
    spec = str(spec).strip().lower()
    if spec.startswith("cuda:"):
        return int(spec.split(":", 1)[1])
    return None


def _resolve_auto_device(cuda_ready, cuda_built, verbose):
    """Pick cuda or cpu for an 'auto' device spec.

    Args:
        cuda_ready: True if a CUDA device is visible and usable.
        cuda_built: True if this torch build includes CUDA support.
        verbose: print the reason when auto resolves to cpu.

    Returns:
        'cuda' or 'cpu'.
    """
    spec = "cuda" if cuda_ready else "cpu"
    if verbose and spec == "cpu":
        why = ("torch is the CPU-only build" if not cuda_built
               else "no CUDA device is visible")
        print(f"device: auto -> cpu ({why})")
    return spec


def _validate_cuda_spec(spec, cuda_built, torch):
    """Validate an explicit cuda/cuda:N device spec.

    Raises RuntimeError on a bad spec, a CPU-only torch build, no visible
    CUDA device, or a bad ordinal. A silent CPU fallback would look like a
    hung job.

    Args:
        spec: normalized spec string, already known not to be 'auto' or
            'cpu'.
        cuda_built: True if this torch build includes CUDA support.
        torch: the imported torch module.

    Raises:
        RuntimeError: one specific reason per failure case.
    """
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

    # Validating the ordinal here means a bad --device value fails
    # immediately with a specific message, rather than surfacing much later
    # as a torch "invalid device ordinal" traceback wherever the device is
    # first touched -- in evaluate.py that is inside the try block that
    # prints the dataset-hash "REFUSED" banner.
    idx = device_index(spec)
    n_gpu = torch.cuda.device_count()
    if idx is not None and idx >= n_gpu:
        raise RuntimeError(
            f"--device {spec} requested but this machine has {n_gpu} CUDA "
            f"device(s) (valid: cuda:0..cuda:{n_gpu - 1})"
        )


def _print_cuda_device(spec, torch):
    """Print the resolved GPU's name, memory, and torch/CUDA versions.

    Args:
        spec: the validated cuda/cuda:N spec.
        torch: the imported torch module.
    """
    # Name the ordinal that was asked for, not current_device() -- naming
    # GPU 0 while running on GPU 1 is worse than printing nothing.
    idx = device_index(spec)
    shown = idx if idx is not None else torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(shown)
    print(f"device: {spec} -> [{shown}] {torch.cuda.get_device_name(shown)} "
          f"({total_b / 1024**3:.1f} GiB total, {free_b / 1024**3:.1f} GiB free), "
          f"torch {torch.__version__} / CUDA {torch.version.cuda}")


def resolve_device(spec="auto", verbose=True):
    """Resolve a device spec to a concrete torch device string, failing loudly.

    'auto' picks cuda when it is actually usable, else cpu. Anything else
    must be exactly 'cpu', 'cuda', or 'cuda:N' with N a real GPU on this
    box.

    Args:
        spec: 'auto', 'cpu', 'cuda', or 'cuda:N'. None is treated as
            'auto'.
        verbose: print the resolved device (and, for an explicit cuda
            spec, its name and free/total memory).

    Returns:
        The resolved device string ('cpu', 'cuda', or 'cuda:N').

    Raises:
        RuntimeError: spec is malformed, or names a CUDA device that is
            not actually available.
    """
    import torch

    # None (the library default) means auto; an empty string is a typo and
    # falls through to the validation below rather than being coerced to
    # auto.
    spec = "auto" if spec is None else str(spec).strip().lower()
    cuda_built = torch.version.cuda is not None
    cuda_ready = cuda_built and torch.cuda.is_available()

    if spec == "auto":
        spec = _resolve_auto_device(cuda_ready, cuda_built, verbose)

    if spec == "cpu":
        return spec

    _validate_cuda_spec(spec, cuda_built, torch)
    if verbose:
        _print_cuda_device(spec, torch)
    return spec


# ----------------------------------------------------------------------------
# Dataset content hash -- the checkpoint <-> dataset contract (File 5 checks it)
# ----------------------------------------------------------------------------
def dataset_hash(X, Y):
    """Compute a SHA-256 digest over the raw bytes of X (float32) and Y (int64).

    Changes if the images, labels, or row order change. evaluate.py uses
    this to detect a stale val_index.

    Args:
        X: array of images.
        Y: array of integer labels.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(Y, dtype=np.int64).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Per-class stratified split
# ----------------------------------------------------------------------------
def stratified_split(Y, val_frac=VAL_FRAC, seed=0):
    """Split indices into train/val with every class represented in both.

    Each class is shuffled and a val_frac slice goes to val. A class with
    only one sample stays entirely in train.

    Args:
        Y: array of integer labels.
        val_frac: target fraction of each class held out for validation.
        seed: seed for the shuffle.

    Returns:
        (train_index, val_index), each a sorted int64 array.
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
    """Shift a spectrogram image along the time axis, zero-filling the edge.

    Never wraps. A circular roll would corrupt the keystroke's onset.

    Args:
        img: (n_mels, n_frames) spectrogram image.
        rng: numpy Generator to draw the shift from.
        max_frac: maximum shift as a fraction of n_frames.

    Returns:
        A new shifted image. Input unchanged.
    """
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
    """Apply a fresh zero-padded time-shift and SpecAugment to each image.

    Args:
        batch: (B, H, W) array of spectrogram images.
        rng: numpy Generator supplying all randomness.

    Returns:
        A new (B, H, W) float32 array. Input unchanged.
    """
    out = np.empty_like(batch, dtype=np.float32)
    for i in range(batch.shape[0]):
        img = _time_shift_spec(batch[i], rng)
        img = ft.spec_augment(img, rng=rng)
        out[i] = img
    return out


def augment_batch_torch(batch, gen, max_frac=ft.TIME_SHIFT_FRAC,
                        n_masks=ft.SPECAUG_MASKS_PER_AXIS, frac=ft.SPECAUG_FRAC):
    """Apply the vectorised on-device port of augment_batch to a (B, H, W) tensor.

    Same semantics as the numpy path: zero-filled time shift, then
    SpecAugment masking. Uses its own torch Generator, so its RNG stream
    does not match the numpy path's.

    Args:
        batch: (B, H, W) tensor of spectrogram images.
        gen: torch Generator supplying all randomness.
        max_frac: maximum time shift as a fraction of W.
        n_masks: number of frequency masks and number of time masks.
        frac: mask width as a fraction of H (frequency) or W (time).

    Returns:
        A new (B, H, W) tensor. Input unchanged.
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
    """Compute top-1 accuracy over a subset, no augmentation, eval mode.

    Args:
        torch: the imported torch module.
        model: the model to evaluate.
        X_sub: (N, H, W) numpy array of images, resident on the host.
        Y_sub: (N,) numpy array of integer labels.
        device: torch device to run inference on.
        batch_size: inference batch size.

    Returns:
        Fraction of X_sub classified correctly. 0.0 if X_sub is empty.
    """
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
    """Compute top-1 accuracy over a val set already resident on-device.

    Same computation as `_accuracy`. Xv/Yv are uploaded once before the
    epoch loop, and the comparison stays on-device.

    Args:
        torch: the imported torch module.
        model: the model to evaluate.
        Xv: (N, H, W) tensor of images, already on the target device.
        Yv: (N,) tensor of integer labels, already on the target device.
        batch_size: inference batch size.

    Returns:
        Fraction of Xv classified correctly. 0.0 if Xv is empty.
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


def _configure_cuda_backend(deterministic, tf32, torch):
    """Configure cudnn determinism/autotuning and TF32 for a cuda run.

    Deterministic mode disables cudnn's autotuner for bit-reproducible
    runs, at a speed cost.

    Args:
        deterministic: use deterministic cudnn kernels, trading speed for
            reproducibility.
        tf32: True/False to force TF32 on/off, or None to leave torch's
            defaults untouched.
        torch: the imported torch module.
    """
    if deterministic:
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


def _seed_torch(seed, on_cuda, torch):
    """Seed torch's RNG so weight init and dropout are reproducible.

    Without this, two runs at the same seed can still land on different
    accuracies.

    Args:
        seed: seed value, shared with the numpy RNG used elsewhere.
        on_cuda: also seed the CUDA RNG state.
        torch: the imported torch module.
    """
    torch.manual_seed(seed)
    if on_cuda:
        torch.cuda.manual_seed_all(seed)


def _stage_val_set(X, Y, val_idx, device, torch):
    """Upload the validation subset to the target device once.

    Avoids a re-upload every epoch.

    Args:
        X: (N, H, W) array of images.
        Y: (N,) array of integer labels.
        val_idx: indices of the validation rows.
        device: target torch device.
        torch: the imported torch module.

    Returns:
        (Xv, Yv) tensors resident on device.
    """
    Xv = torch.from_numpy(X[val_idx]).to(device)
    Yv = torch.from_numpy(Y[val_idx]).to(device)
    return Xv, Yv


def _stage_gpu_aug_pool(X, Y, device, seed, torch):
    """Upload the whole training pool to device for the gpu augmentation path.

    The pool is about 11 MB, small enough to fit on device whole.

    Args:
        X: (N, H, W) array of images.
        Y: (N,) array of integer labels.
        device: target torch device.
        seed: seed for the dedicated torch Generator that drives
            on-device augmentation.
        torch: the imported torch module.

    Returns:
        (X_dev, Y_dev, gen): the pool on device, and a torch Generator
        seeded independently of the numpy RNG.
    """
    X_dev = torch.from_numpy(X).to(device)
    Y_dev = torch.from_numpy(Y).to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return X_dev, Y_dev, gen


def _build_run_env(device, on_cuda, gpu_aug, seed, torch):
    """Build the run-provenance dict stored in the checkpoint.

    Records the actual backend state read back from torch, not just what
    was requested.

    Args:
        device: the resolved torch device string.
        on_cuda: whether device is a cuda device.
        gpu_aug: whether the gpu augmentation path is in use.
        seed: the seed used for this run.
        torch: the imported torch module.

    Returns:
        dict of provenance fields, stored under the checkpoint's "run_env"
        key.
    """
    gpu_idx = device_index(device)
    if gpu_idx is None and on_cuda:
        gpu_idx = torch.cuda.current_device()
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(gpu_idx) if on_cuda else None,
        "aug": "gpu" if gpu_aug else "cpu",
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if on_cuda else None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if on_cuda else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if on_cuda else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if on_cuda else None,
        "seed": int(seed),
    }


def _print_train_header(kind, model, n_samples, train_idx, val_idx, labels,
                        device, gpu_aug, seed, deterministic, on_cuda, torch):
    """Print the one-line run summary shown at the start of a verbose run."""
    print(f"model={kind}  params={count_parameters(model):,}  "
          f"samples={n_samples}  train={len(train_idx)}  val={len(val_idx)}  "
          f"classes={len(labels)}  device={device}  "
          f"aug={'gpu' if gpu_aug else 'cpu'}  seed={seed}"
          + (f"  tf32(cudnn/matmul)="
             f"{'on' if torch.backends.cudnn.allow_tf32 else 'off'}/"
             f"{'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}"
             f"  deterministic={'on' if deterministic else 'off'}"
             if on_cuda else ""))


def _run_train_epoch(model, opt, X, Y, train_idx, batch_size, rng, device,
                     gpu_aug, X_dev, Y_dev, gen, torch, Fnn):
    """Run one training epoch over a shuffled permutation of train_idx.

    Only epoch 0 uses the same batch order between the cpu and gpu
    augmentation paths. Later epochs diverge.

    Args:
        model: the model being trained, already in train mode.
        opt: the optimizer.
        X: (N, H, W) array of images (used on the cpu augmentation path).
        Y: (N,) array of integer labels (used on the cpu augmentation
            path).
        train_idx: indices eligible for this epoch.
        batch_size: training batch size.
        rng: numpy Generator for the batch permutation, and for cpu
            augmentation when gpu_aug is False.
        device: target torch device.
        gpu_aug: use the on-device augmentation path.
        X_dev: training pool resident on device (gpu_aug only).
        Y_dev: training labels resident on device (gpu_aug only).
        gen: torch Generator driving on-device augmentation (gpu_aug
            only).
        torch: the imported torch module.
        Fnn: torch.nn.functional.

    Returns:
        The final batch's loss tensor, for logging.
    """
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
    return loss


def _record_epoch(history, epoch, loss, val_acc, cur_lr):
    """Append one epoch's metrics to the running training history.

    Args:
        history: dict of per-epoch lists, updated in place.
        epoch: epoch index.
        loss: final training-batch loss tensor for this epoch.
        val_acc: validation accuracy for this epoch.
        cur_lr: learning rate used this epoch.
    """
    history["epoch"].append(int(epoch))
    history["loss"].append(float(loss.detach()))
    history["val_acc"].append(float(val_acc))
    history["lr"].append(float(cur_lr))


def _build_checkpoint(kind, model, labels, val_idx, train_idx, data_h,
                      n_samples, epoch, val_acc, run_env):
    """Assemble the checkpoint dict for the current best-val epoch.

    Args:
        kind: model architecture name.
        model: the model to snapshot.
        labels: class names, indexed by label id.
        val_idx: validation row indices.
        train_idx: training row indices.
        data_h: dataset_hash(X, Y) for the training pool.
        n_samples: total row count in the training pool.
        epoch: epoch this checkpoint was taken at.
        val_acc: validation accuracy at this epoch.
        run_env: provenance dict from _build_run_env.

    Returns:
        dict with the model state and enough metadata for evaluate.py to
        verify it against the dataset that produced it (see dataset_hash).
    """
    return {
        "model_kind": kind,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "labels": list(labels),
        "val_index": np.array(val_idx, dtype=np.int64),
        "train_index": np.array(train_idx, dtype=np.int64),
        "data_hash": data_h,
        "n_samples": int(n_samples),
        "epoch": epoch,
        "val_acc": float(val_acc),
        # Run provenance -- which machine/precision/aug path produced this
        # number. Extra keys, ignored by evaluate.py, but a run that ends
        # up in a write-up should be self-describing.
        "run_env": run_env,
    }


def _maybe_print_progress(epoch, epochs, loss, val_acc, best_acc, cur_lr, verbose):
    """Print a progress line roughly 20 times over the run, plus the last epoch."""
    if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
        print(f"  epoch {epoch:4d}/{epochs}  loss={float(loss.detach()):.4f}  "
              f"val_acc={val_acc:.3f}  best={best_acc:.3f}  lr={cur_lr:.2e}")


def _print_train_summary(history, best_acc, out_path):
    """Print the final best-accuracy line and the last-100-epoch optimism gap."""
    tail = history["val_acc"][-100:]
    print(f"best val_acc={best_acc:.3f}  saved -> {out_path}")
    if tail:
        print(f"  (last 100 epochs: mean {float(np.mean(tail)):.3f}, "
              f"max {float(np.max(tail)):.3f} -- the gap to `best` is selection optimism)")


def train_model(X, Y, labels, kind="cnn", epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=MAX_LR, val_frac=VAL_FRAC, seed=0, device="cpu",
                out_path="checkpoint.pt", min_lr_factor=MIN_LR_FACTOR, verbose=True,
                aug="cpu", tf32=None, deterministic=False):
    """Train `kind` on (X, Y) and save the best-val checkpoint to out_path.

    Device is assumed already resolved. Defaults to cpu for compatibility
    with existing callers.

    `aug` selects "cpu" (numpy) or "gpu" (on-device, different RNG stream)
    augmentation. "gpu" falls back to "cpu" on a cpu device.

    `seed` seeds numpy (split, cpu augmentation) and torch (weight init,
    dropout). Same seed and device reproduce bit-for-bit on CPU, and on
    CUDA only with `deterministic=True`.

    Args:
        X: (N, H, W) array of spectrogram images.
        Y: (N,) array of integer labels.
        labels: class names, indexed by label id.
        kind: model architecture, passed to build_model.
        epochs: number of training epochs.
        batch_size: training batch size.
        lr: initial (maximum) learning rate.
        val_frac: fraction of each class held out for validation.
        seed: seeds the numpy and torch RNGs.
        device: an already-resolved torch device string.
        out_path: checkpoint save path. Falsy skips saving.
        min_lr_factor: fraction of lr the linear anneal decays toward.
        verbose: print progress and a final summary.
        aug: "cpu" or "gpu" augmentation path.
        tf32: True/False to force TF32 on/off on cuda, or None (default)
            to leave torch's own setting untouched.
        deterministic: make a cuda run bit-reproducible run-to-run, at a
            speed cost.

    Returns:
        (best_val_acc, checkpoint_dict).
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
        _configure_cuda_backend(deterministic, tf32, torch)

    train_idx, val_idx = stratified_split(Y, val_frac, seed)
    rng = np.random.default_rng(seed)
    _seed_torch(seed, on_cuda, torch)

    model = build_model(kind, num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xv, Yv = _stage_val_set(X, Y, val_idx, device, torch)
    X_dev = Y_dev = gen = None
    if gpu_aug:
        X_dev, Y_dev, gen = _stage_gpu_aug_pool(X, Y, device, seed, torch)

    # Linear anneal, set explicitly at the start of each epoch: factor 1.0
    # at epoch 0 decaying toward min_lr_factor at the end. Dividing by
    # `epochs` (not epochs - 1) keeps the last epoch's LR small but
    # strictly positive -- no wasted zero-LR epoch and never negative.
    def _lr_at(epoch):
        factor = 1.0 - (1.0 - min_lr_factor) * (epoch / epochs)
        return lr * max(factor, min_lr_factor)

    if verbose:
        _print_train_header(kind, model, n_samples, train_idx, val_idx, labels,
                            device, gpu_aug, seed, deterministic, on_cuda, torch)

    # Run provenance, built once, so a number quoted in a write-up stays
    # traceable to the hardware and precision that produced it.
    run_env = _build_run_env(device, on_cuda, gpu_aug, seed, torch)

    best_acc = -1.0
    best_ckpt = None
    # Per-epoch trace. Kept because the single "best val_acc" number hides
    # how it was obtained: it is the maximum of `epochs` noisy measurements
    # on the same small val set, so the gap between this curve's tail and
    # its peak is the selection optimism in the headline figure.
    history = {"epoch": [], "loss": [], "val_acc": [], "lr": []}

    for epoch in range(epochs):
        cur_lr = _lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        model.train()
        loss = _run_train_epoch(model, opt, X, Y, train_idx, batch_size, rng,
                                device, gpu_aug, X_dev, Y_dev, gen, torch, Fnn)

        val_acc = _accuracy_dev(torch, model, Xv, Yv)
        _record_epoch(history, epoch, loss, val_acc, cur_lr)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_ckpt = _build_checkpoint(kind, model, labels, val_idx, train_idx,
                                          data_h, n_samples, epoch, val_acc, run_env)
            if out_path:
                torch.save(best_ckpt, out_path)

        _maybe_print_progress(epoch, epochs, loss, val_acc, best_acc, cur_lr, verbose)

    # The best checkpoint was written mid-run, so its history was truncated
    # at whatever epoch happened to be the peak. Attach the full trace and
    # re-save so the file describes the whole run, not just the part
    # before the peak.
    if best_ckpt is not None:
        best_ckpt["history"] = history
        if out_path:
            torch.save(best_ckpt, out_path)

    if verbose:
        _print_train_summary(history, best_acc, out_path)
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
                   help="make cuda runs bit-reproducible run-to-run (cudnn.deterministic "
                        "on, autotuner off), at some speed cost. Use it for any number "
                        "you intend to publish.")
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
