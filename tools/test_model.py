"""Synthetic-data tests for model.py.

Covers lazy torch import, CNN and CoAtNet forward/backward passes, the
CoAtNet grid-size guard, and rejection of unknown model kinds.

Run:  python test_model.py
"""

import sys

import numpy as np

# --- repo layout bootstrap --------------------------------------------------
# Pipeline modules live in training/ and the tools in tools/, so a module in one
# cannot import one from the other by name alone. Put both directories on
# sys.path so every script keeps working when run directly from any cwd.
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in (_os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "tools")):
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# ----------------------------------------------------------------------------

import model as M

NUM_CLASSES = 37


def _batch(torch, n=4, size=64, seed=0):
    """Build a synthetic (n, 1, size, size) input batch of standard-normal noise.

    Args:
        torch: The torch module, passed in so callers control when it loads.
        n: Batch size.
        size: Height and width of each single-channel image.
        seed: Seed for the batch's random generator.

    Returns:
        A float32 torch tensor of shape (n, 1, size, size).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1, size, size)).astype(np.float32)
    return torch.from_numpy(x)


def _eval_forward(kind, seed):
    """Build a model of the given kind and run one no-grad forward pass.

    Args:
        kind: Model kind passed to build_model, e.g. "cnn" or "coatnet".
        seed: Seed for the synthetic input batch.

    Returns:
        A (model, output_tensor) tuple.
    """
    import torch
    net = M.build_model(kind, NUM_CLASSES)
    net.eval()
    with torch.no_grad():
        y = net(_batch(torch, n=4, seed=seed))
    return net, y


def _train_backward(kind, seed):
    """Build a model of the given kind and run one forward+backward training step.

    Args:
        kind: Model kind passed to build_model, e.g. "cnn" or "coatnet".
        seed: Seed for the synthetic input batch and target labels.

    Returns:
        The gradient tensor for each trainable parameter.
    """
    import torch
    net = M.build_model(kind, NUM_CLASSES)
    net.train()
    x = _batch(torch, n=4, seed=seed)
    target = torch.randint(0, NUM_CLASSES, (4,))
    loss = torch.nn.functional.cross_entropy(net(x), target)
    loss.backward()
    return [p.grad for p in net.parameters() if p.requires_grad]


def test_lazy_torch_import():
    """Verify that importing model.py alone does not import torch.

    Must run first, before any other test imports torch.
    """
    assert "torch" not in sys.modules, "model.py imported torch eagerly (should be lazy)"
    print("lazy import: `import model` did not pull in torch OK")


def test_cnn_shapes_and_finite():
    """Verify build_model("cnn", ...) maps a batch to finite (B, num_classes) logits."""
    import torch
    net, y = _eval_forward("cnn", seed=0)
    assert tuple(y.shape) == (4, NUM_CLASSES), f"cnn out shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "cnn produced non-finite logits"
    print(f"cnn: out shape {tuple(y.shape)}, finite OK  ({M.count_parameters(net):,} params)")


def test_cnn_backward():
    """Verify a CNN forward+backward pass produces finite, nonzero gradients."""
    grads = _train_backward("cnn", seed=1)
    assert all(g is not None for g in grads), "some parameters got no gradient"
    total = sum(float(g.abs().sum()) for g in grads)
    assert np.isfinite(total) and total > 0, f"gradients absent or non-finite (sum={total})"
    print("cnn: forward+backward produces finite nonzero gradients OK")


def test_coatnet_shapes_and_finite():
    """Verify build_model("coatnet", ...) maps a batch to finite (B, num_classes) logits."""
    import torch
    net, y = _eval_forward("coatnet", seed=2)
    assert tuple(y.shape) == (4, NUM_CLASSES), f"coatnet out shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "coatnet produced non-finite logits"
    print(f"coatnet: out shape {tuple(y.shape)}, finite OK  ({M.count_parameters(net):,} params)")


def test_coatnet_backward():
    """Verify a CoAtNet forward+backward pass produces finite, nonzero gradients."""
    grads = _train_backward("coatnet", seed=3)
    assert all(g is not None for g in grads), "some parameters got no gradient"
    total = sum(float(g.abs().sum()) for g in grads)
    assert np.isfinite(total) and total > 0, f"gradients absent or non-finite (sum={total})"
    print("coatnet: forward+backward produces finite nonzero gradients OK")


def test_coatnet_grid_assert_fires():
    """Verify the CoAtNet's 8x8-grid guard raises on an input size that doesn't reduce to 8x8.

    Uses a 32x32 input, which reduces to a 4x4 grid instead.
    """
    import torch
    net = M.build_model("coatnet", NUM_CLASSES)
    net.eval()
    bad = torch.zeros(2, 1, 32, 32)
    try:
        with torch.no_grad():
            net(bad)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "coatnet accepted a wrong input size (8x8 grid guard did not fire)"
    print("coatnet: 8x8-grid guard fires on wrong input size OK")


def test_unknown_kind_rejected():
    """Verify build_model raises ValueError for a model kind it does not recognize."""
    try:
        M.build_model("resnet", NUM_CLASSES)
        raised = False
    except ValueError:
        raised = True
    assert raised, "unknown model kind was not rejected"
    print("build_model: unknown kind rejected OK")


def main():
    test_lazy_torch_import()
    test_cnn_shapes_and_finite()
    test_cnn_backward()
    test_coatnet_shapes_and_finite()
    test_coatnet_backward()
    test_coatnet_grid_assert_fires()
    test_unknown_kind_rejected()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
