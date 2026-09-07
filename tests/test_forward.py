"""BitFeatNet forward smoke on a random 480x480 tensor (CPU)."""

from __future__ import annotations

import torch

from bitfeat.models.bitfeat import BitFeatNet


def test_bitfeat_forward_480():
    # Use a slim config so the 480^2 CPU test stays reasonable in CI.
    model = BitFeatNet(
        channels=(32, 64, 64, 64),
        blocks_per_stage=(1, 1, 1, 1),
        desc_dim=64,
        top_k=128,
    )
    model.eval()
    x = torch.randn(1, 3, 480, 480)
    with torch.no_grad():
        out = model(x, detect=True, top_k=128)
    assert out["scores"].shape == (1, 1, 480, 480)
    assert out["descriptors"].shape == (1, 64, 480, 480)
    assert out["keypoints"].shape == (1, 128, 2)
    assert out["keypoint_scores"].shape == (1, 128)
    assert out["keypoint_descriptors"].shape == (1, 128, 64)
    # Descriptors should be roughly unit-norm.
    norms = out["descriptors"].norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_bitfeat_train_step_smoke():
    from bitfeat.losses.matching import MatchingLoss

    model = BitFeatNet(
        channels=(16, 32, 32),
        blocks_per_stage=(1, 1, 1),
        desc_dim=32,
    )
    criterion = MatchingLoss()
    x0 = torch.randn(1, 3, 64, 64)
    x1 = torch.randn(1, 3, 64, 64)
    out0 = model(x0)
    out1 = model(x1)
    losses = criterion(out0, out1, {})
    losses["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
