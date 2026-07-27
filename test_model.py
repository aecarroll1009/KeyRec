"""Tiny synthetic-data test for model.py.

Proves the core path works before we move on to File 4 (train.py):
  * torch is imported LAZILY -- `import model` alone must not pull in torch,
  * build_model("cnn", ...) builds and maps (B,1,64,64) -> (B, num_classes)
    with finite outputs (bring the CNN up FIRST, per the spec),
  * a forward+backward pass produces gradients (the net is trainable),
  * build_model("coatnet", ...) does the same, and its conv trunk lands on the
    required 8x8 grid,
  * the 8x8-grid assertion fires LOUDLY on a wrong input size instead of
    silently reshaping garbage,
  * an unknown kind is rejected.

Run:  python test_model.py
"""

import sys

import numpy as np

import model as M

NUM_CLASSES = 37


def test_lazy_torch_import():
    # model.py must not import torch at module-import time -- importing it above
    # should have left torch absent from sys.modules. Run this check FIRST,
    # before any other test pulls torch in locally.
    assert "torch" not in sys.modules, "model.py imported torch eagerly (should be lazy)"
    print("lazy import: `import model` did not pull in torch OK")


def _batch(torch, n=4, size=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1, size, size)).astype(np.float32)
    return torch.from_numpy(x)


def test_cnn_shapes_and_finite():
    import torch
    net = M.build_model("cnn", NUM_CLASSES)
    net.eval()
    with torch.no_grad():
        y = net(_batch(torch, n=4))
    assert tuple(y.shape) == (4, NUM_CLASSES), f"cnn out shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "cnn produced non-finite logits"
    print(f"cnn: out shape {tuple(y.shape)}, finite OK  ({M.count_parameters(net):,} params)")


def test_cnn_backward():
    import torch
    net = M.build_model("cnn", NUM_CLASSES)
    net.train()
    x = _batch(torch, n=4, seed=1)
    target = torch.randint(0, NUM_CLASSES, (4,))
    loss = torch.nn.functional.cross_entropy(net(x), target)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "some parameters got no gradient"
    total = sum(float(g.abs().sum()) for g in grads)
    assert np.isfinite(total) and total > 0, f"gradients absent or non-finite (sum={total})"
    print("cnn: forward+backward produces finite nonzero gradients OK")


def test_coatnet_shapes_and_finite():
    import torch
    net = M.build_model("coatnet", NUM_CLASSES)
    net.eval()
    with torch.no_grad():
        y = net(_batch(torch, n=4, seed=2))
    assert tuple(y.shape) == (4, NUM_CLASSES), f"coatnet out shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "coatnet produced non-finite logits"
    print(f"coatnet: out shape {tuple(y.shape)}, finite OK  ({M.count_parameters(net):,} params)")


def test_coatnet_backward():
    import torch
    net = M.build_model("coatnet", NUM_CLASSES)
    net.train()
    x = _batch(torch, n=4, seed=3)
    target = torch.randint(0, NUM_CLASSES, (4,))
    loss = torch.nn.functional.cross_entropy(net(x), target)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "some parameters got no gradient"
    total = sum(float(g.abs().sum()) for g in grads)
    assert np.isfinite(total) and total > 0, f"gradients absent or non-finite (sum={total})"
    print("coatnet: forward+backward produces finite nonzero gradients OK")


def test_coatnet_grid_assert_fires():
    import torch
    net = M.build_model("coatnet", NUM_CLASSES)
    net.eval()
    # A 32x32 input reduces to a 4x4 grid, not 8x8 -> the guard must fire.
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
