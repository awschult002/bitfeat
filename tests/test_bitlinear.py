"""Ternarization in {-1,0,1} and STE gradient smoke tests."""

from __future__ import annotations

import torch

from bitfeat.layers.bitlinear import BitLinear, ternarize
from bitfeat.layers.bitconv import BitConv2d


def test_ternarize_values_absmean():
    w = torch.randn(16, 8)
    w_q, gamma = ternarize(w, method="absmean")
    uniq = set(w_q.detach().unique().tolist())
    assert uniq <= {-1.0, 0.0, 1.0}, uniq
    assert gamma.ndim == 0
    assert float(gamma) > 0


def test_ternarize_values_absmedian():
    w = torch.randn(16, 8)
    w_q, gamma = ternarize(w, method="absmedian")
    uniq = set(w_q.detach().unique().tolist())
    assert uniq <= {-1.0, 0.0, 1.0}, uniq
    assert float(gamma) > 0


def test_bitlinear_ste_gradients():
    layer = BitLinear(8, 4, bias=True)
    x = torch.randn(2, 8, requires_grad=True)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert torch.isfinite(layer.weight.grad).all()
    # Bias and input should also receive grads.
    assert layer.bias.grad is not None
    assert x.grad is not None


def test_bitconv_forward_and_grad():
    conv = BitConv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    y = conv(x)
    assert y.shape == (1, 8, 32, 32)
    y.mean().backward()
    assert conv.weight.grad is not None
    assert torch.isfinite(conv.weight.grad).all()
