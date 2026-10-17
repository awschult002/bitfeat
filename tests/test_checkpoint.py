"""Checkpoint save / resume round-trip on CPU with a tiny model + synthetic pairs."""

from __future__ import annotations

import copy

import pytest
import torch

from bitfeat.checkpoint import (
    FINAL_NAME,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
)
from bitfeat.train import build_model, load_config, train_loop


def _tiny_cfg(out_dir, **train_overrides):
    cfg = load_config(None)
    cfg["model"].update(channels=[8, 16], blocks_per_stage=[1, 1], desc_dim=16, top_k=32)
    cfg["train"].update(
        batch_size=2,
        image_size=[32, 32],
        num_samples=32,
        log_every=1,
        output_dir=str(out_dir),
        checkpoint_every=2,
        keep_last=2,
        seed=123,
        amp=False,
    )
    cfg["train"].update(train_overrides)
    # Synthetic pairs, 5 batches per epoch, so resume crosses an epoch boundary.
    cfg["data"].update(root="", synthetic_length=10)
    return cfg


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_resume_matches_uninterrupted_run(tmp_path):
    steps = 8
    ref = train_loop(_tiny_cfg(tmp_path / "ref", max_steps=steps), device="cpu")
    assert ref["step"] == steps
    ref_state = _state(ref["model"])

    # Same run, interrupted after 3 steps (not a multiple of checkpoint_every)...
    part_dir = tmp_path / "part"
    first = train_loop(_tiny_cfg(part_dir, max_steps=3), device="cpu")
    assert first["step"] == 3
    # ...then continued to 8 from the newest checkpoint in the run dir.
    resumed = train_loop(_tiny_cfg(part_dir, max_steps=steps), device="cpu", resume="auto")
    assert resumed["step"] == steps

    res_state = _state(resumed["model"])
    assert ref_state.keys() == res_state.keys()
    for k in ref_state:
        assert torch.equal(ref_state[k], res_state[k]), f"mismatch in {k}"


def test_checkpoint_retention_and_contents(tmp_path):
    out = tmp_path / "run"
    train_loop(_tiny_cfg(out, max_steps=9), device="cpu")
    # checkpoint_every=2 → steps 2,4,6,8; keep_last=2 keeps 6 and 8, plus final.pt.
    assert [s for s, _ in list_checkpoints(out)] == [6, 8]
    assert (out / FINAL_NAME).is_file()
    assert latest_checkpoint(out) == out / FINAL_NAME
    assert (out / "config.yaml").is_file() and (out / "metrics.jsonl").is_file()

    cfg = _tiny_cfg(out, max_steps=9)
    model = build_model(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = load_checkpoint(out / FINAL_NAME, model=model, optimizer=optim)
    assert ckpt["step"] == 9
    assert ckpt["config"]["model"] == cfg["model"]
    assert {"python", "numpy", "torch"} <= set(ckpt["rng"])
    assert len(optim.state) > 0  # Adam moments restored
    # bitfeat.extract loads with weights_only=True; checkpoints must stay compatible.
    raw = torch.load(out / FINAL_NAME, map_location="cpu", weights_only=True)
    assert raw["step"] == 9 and "model" in raw


def test_resume_rejects_model_mismatch(tmp_path):
    out = tmp_path / "run"
    train_loop(_tiny_cfg(out, max_steps=2), device="cpu")
    cfg = _tiny_cfg(out, max_steps=4)
    cfg["model"]["desc_dim"] = 8
    with pytest.raises(ValueError, match="model config differs"):
        train_loop(cfg, device="cpu", resume="auto")


def test_resume_explicit_path_and_missing(tmp_path):
    out = tmp_path / "run"
    train_loop(_tiny_cfg(out, max_steps=4), device="cpu")
    ckpt2 = list_checkpoints(out)[0][1]  # step 2
    res = train_loop(_tiny_cfg(tmp_path / "other", max_steps=5), device="cpu", resume=str(ckpt2))
    assert res["step"] == 5
    with pytest.raises(FileNotFoundError):
        train_loop(_tiny_cfg(out, max_steps=5), device="cpu", resume=str(tmp_path / "nope.pt"))
