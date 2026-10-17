"""Mixed-precision forward/backward (CUDA only) and autocast-safety of ternarization."""

from __future__ import annotations

import pytest
import torch

from bitfeat.layers.bitlinear import ternarize
from bitfeat.losses.matching import MatchingLoss
from bitfeat.models.bitfeat import BitFeatNet
from bitfeat.train import _float_outputs, _make_grad_scaler

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _batch(b, h, w, device):
    from bitfeat.data.pairs import SyntheticPairDataset

    ds = SyntheticPairDataset(length=b, image_size=(h, w))
    items = [ds[i] for i in range(b)]
    return {k: torch.stack([it[k] for it in items]).to(device) for k in items[0]}


@cuda_only
def test_ternarize_stays_fp32_under_autocast():
    w = torch.randn(32, 16, 3, 3, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        w_q, gamma = ternarize(w)
    assert w_q.dtype == torch.float32 and gamma.dtype == torch.float32
    ref_q, ref_g = ternarize(w)
    assert torch.equal(w_q, ref_q) and torch.equal(gamma, ref_g)


@cuda_only
@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_amp_forward_backward_cuda(grad_checkpoint):
    torch.manual_seed(0)
    model = BitFeatNet(
        channels=(16, 32, 32), blocks_per_stage=(1, 1, 1), desc_dim=32, grad_checkpoint=grad_checkpoint
    ).cuda()
    model.train()
    crit = MatchingLoss(num_samples=64)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = _make_grad_scaler(True, init_scale=2.0**10)
    batch = _batch(2, 64, 64, "cuda")
    applied = 0

    for _ in range(5):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out0 = model(batch["image0"])
            out1 = model(batch["image1"])
        assert out0["features"][-1].dtype == torch.float16  # autocast actually active
        losses = crit(_float_outputs(out0), _float_outputs(out1), batch)
        assert losses["loss"].dtype == torch.float32
        assert torch.isfinite(losses["loss"])
        optim.zero_grad(set_to_none=True)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optim)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(g.dtype == torch.float32 for g in grads)  # FP32 master weights
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        scale = scaler.get_scale()
        scaler.step(optim)  # GradScaler skips the step on overflow
        scaler.update()
        applied += int(finite)
        if not finite:
            assert scaler.get_scale() < scale
    assert applied >= 1
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters())


def test_grad_checkpoint_matches_plain_backward():
    """Gradient checkpointing must not change gradients (CPU, FP32)."""
    torch.manual_seed(0)
    a = BitFeatNet(channels=(8, 16), blocks_per_stage=(1, 1), desc_dim=16)
    b = BitFeatNet(channels=(8, 16), blocks_per_stage=(1, 1), desc_dim=16, grad_checkpoint=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 32, 32)
    for m in (a, b):
        m.train()
        out = m(x)
        (out["descriptors"].sum() + out["scores"].sum()).backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5, rtol=1e-4), n
