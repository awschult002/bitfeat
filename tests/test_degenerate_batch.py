"""A pair with no valid correspondences must not crash training (no-grad loss)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

import bitfeat.train as train_mod
from bitfeat.data.pairs import SyntheticPairDataset
from bitfeat.losses.matching import MatchingLoss
from bitfeat.models.bitfeat import BitFeatNet
from bitfeat.train import load_config, train_loop


def _degenerate(item):
    """Depth present (has_depth=True) but zero everywhere -> every warped sample invalid."""
    item = dict(item)
    item["depth0"] = torch.zeros_like(item["depth0"])
    item["depth1"] = torch.zeros_like(item["depth1"])
    item["has_depth"] = torch.tensor(True)
    return item


class _MixedPairs(Dataset):
    """Synthetic pairs where the indices in ``bad`` are degenerate."""

    def __init__(self, n, image_size, bad):
        self.base = SyntheticPairDataset(length=n, image_size=image_size)
        self.bad = set(bad)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        return _degenerate(item) if i in self.bad else item


def test_matching_loss_has_no_grad_on_degenerate_pair():
    """Documents the root cause: both geometric terms fall back to constant zeros."""
    model = BitFeatNet(channels=(8, 16), blocks_per_stage=(1, 1), desc_dim=16)
    item = _degenerate(SyntheticPairDataset(length=1, image_size=(32, 32))[0])
    batch = {k: v.unsqueeze(0) for k, v in item.items()}
    losses = MatchingLoss(num_samples=32)(model(batch["image0"]), model(batch["image1"]), batch)
    assert float(losses["loss"]) == 0.0
    assert not losses["loss"].requires_grad


def _cfg(out_dir, steps):
    cfg = load_config(None)
    cfg["model"].update(channels=[8, 16], blocks_per_stage=[1, 1], desc_dim=16, top_k=32)
    cfg["train"].update(
        batch_size=1, image_size=[32, 32], num_samples=32, log_every=1, max_steps=steps,
        output_dir=str(out_dir), checkpoint_every=0, seed=0, amp=False,
    )
    cfg["data"].update(root="")
    return cfg


def test_train_loop_skips_degenerate_batches(tmp_path, monkeypatch, capsys):
    n, bad = 6, {1, 4}
    monkeypatch.setattr(train_mod, "build_dataset", lambda cfg: _MixedPairs(n, (32, 32), bad))
    res = train_loop(_cfg(tmp_path / "run", steps=n), device="cpu")
    assert res["step"] == n  # finished, did not crash
    assert res["degenerate_skipped"] == len(bad)
    err = capsys.readouterr().err
    assert err.count("degenerate batch") == len(bad)
    ckpt = torch.load(res["final_checkpoint"], map_location="cpu", weights_only=True)
    assert ckpt["extra"]["degenerate_skipped"] == len(bad)


def test_all_degenerate_batches_leave_weights_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "build_dataset", lambda cfg: _MixedPairs(3, (32, 32), {0, 1, 2}))
    res = train_loop(_cfg(tmp_path / "run", steps=3), device="cpu")
    # train_loop seeds before building the model, so the same seed gives the same
    # init; no optimizer step ran, so the parameters must be unchanged.
    train_mod._seed_everything(0)
    ref = train_mod.build_model(_cfg(tmp_path / "x", steps=3))
    for (k, a), (_, b) in zip(res["model"].named_parameters(), ref.named_parameters()):
        assert torch.equal(a, b), k
    assert res["degenerate_skipped"] == 3
