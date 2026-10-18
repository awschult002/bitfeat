"""Train BitFeat: ``python -m bitfeat.train --config configs/default.yaml``.

Smoke (no data)::

    python -m bitfeat.train --smoke

Real MegaDepth (box layout)::

    python -m bitfeat.train --config configs/default.yaml \\
      --data-root /workspace/datasets/megadepth \\
      --index-dir index/train_local --device cuda

Mixed precision + checkpoints (e.g. a 4 GB laptop GPU)::

    python -m bitfeat.train --config configs/default.yaml \\
      --data-root ~/data/megadepth --device cuda --amp --grad-checkpoint \\
      --output-dir runs/bitfeat_main --max-steps 100000

    # after an interruption, continue from the newest checkpoint in --output-dir
    python -m bitfeat.train ... --output-dir runs/bitfeat_main --resume auto
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from bitfeat.checkpoint import (
    FINAL_NAME,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from bitfeat.data.pairs import SyntheticPairDataset
from bitfeat.losses.matching import MatchingLoss
from bitfeat.models.bitfeat import BitFeatNet


def load_config(path: str | None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "model": {
            "channels": [64, 128, 256, 256],
            "blocks_per_stage": [2, 2, 2, 2],
            "desc_dim": 128,
            "method": "absmean",
            "top_k": 1024,
        },
        "train": {
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "batch_size": 2,
            "num_workers": 0,
            "max_steps": 100000,
            "log_every": 10,
            "image_size": [480, 480],
            "desc_weight": 1.0,
            "reproj_weight": 0.1,
            "desc_mode": "dual_softmax",
            "num_samples": 512,
            # Mixed precision (CUDA only). fp16 needs GradScaler; bf16 does not
            # (Turing GPUs such as the T2000 have no bf16, so use float16 there).
            "amp": False,
            "amp_dtype": "float16",
            "grad_checkpoint": False,     # recompute full-res stem + stage 0 in backward
            "grad_clip": 0.0,             # max grad norm; 0 disables clipping
            "max_nonfinite_steps": 20,    # abort after this many consecutive NaN/inf losses
            # Off by default: on a 4 GB T2000 it nearly doubled peak memory
            # (cuDNN workspaces) without a measurable speedup.
            "cudnn_benchmark": False,
            "seed": 0,
            # Checkpointing
            "output_dir": "runs/bitfeat",
            "checkpoint_every": 1000,
            "keep_last": 3,
        },
        "data": {
            "root": "/workspace/datasets/megadepth",
            "split": "train",
            "index_dir": "index/train_local",
            "scene_info_dir": "",
            "min_overlap": 0.1,
            "max_overlap": 0.7,
            "require_depth": False,
        },
    }
    if path is None:
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    for k, v in user.items():
        if isinstance(v, dict) and k in defaults:
            defaults[k].update(v)
        else:
            defaults[k] = v
    return defaults


def build_model(cfg: Dict[str, Any]) -> BitFeatNet:
    m = cfg["model"]
    return BitFeatNet(
        channels=tuple(m["channels"]),
        blocks_per_stage=tuple(m["blocks_per_stage"]),
        desc_dim=int(m["desc_dim"]),
        method=str(m["method"]),
        top_k=int(m["top_k"]),
        grad_checkpoint=bool(cfg.get("train", {}).get("grad_checkpoint", False)),
    )


def build_criterion(tcfg: Dict[str, Any]) -> MatchingLoss:
    return MatchingLoss(
        desc_weight=tcfg.get("desc_weight", 1.0),
        reproj_weight=tcfg.get("reproj_weight", 0.1),
        desc_mode=str(tcfg.get("desc_mode", "dual_softmax")),
        num_samples=int(tcfg.get("num_samples", 512)),
    )


def _batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def smoke_train(cfg: Dict[str, Any], steps: int = 3, device: str = "cpu") -> None:
    tcfg = cfg["train"]
    h, w = tcfg.get("image_size", [64, 64])
    ds = SyntheticPairDataset(length=max(steps * tcfg["batch_size"], 4), image_size=(h, w))
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0)
    model = build_model(cfg).to(device)
    criterion = build_criterion(tcfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = _batch_to_device(batch, device)
            out0 = model(batch["image0"])
            out1 = model(batch["image1"])
            losses = criterion(out0, out1, batch)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optim.step()
            print(
                f"[smoke] step={step} loss={losses['loss'].item():.4f} "
                f"desc={losses['loss_desc'].item():.4f} "
                f"reproj={losses['loss_reproj'].item():.4f}"
            )
            step += 1
    print("[smoke] ok")


def build_dataset(cfg: Dict[str, Any]):
    """ImagePairDataset when ``data.root`` has usable pairs, else synthetic pairs."""
    tcfg = cfg["train"]
    dcfg = cfg.get("data", {})
    data_root = dcfg.get("root", "") or ""
    h, w = tcfg.get("image_size", [480, 480])
    use_real = False

    if data_root and Path(data_root).is_dir():
        from bitfeat.data.pairs import ImagePairDataset

        index_dir = dcfg.get("index_dir") or None
        scene_info = dcfg.get("scene_info_dir") or None
        if scene_info == "":
            scene_info = None
        if index_dir == "":
            index_dir = None
        ds = ImagePairDataset(
            root=data_root,
            split=dcfg.get("split", "train"),
            image_size=(h, w),
            index_dir=index_dir,
            scene_info_dir=scene_info,
            min_overlap=float(dcfg.get("min_overlap", 0.1)),
            max_overlap=float(dcfg.get("max_overlap", 0.7)),
            require_depth=bool(dcfg.get("require_depth", False)),
        )
        if len(ds) > 0:
            use_real = True
            n_depth = sum(1 for s in ds.samples if s.get("depth0") is not None)
            print(
                f"Loaded ImagePairDataset: {len(ds)} pairs from {data_root} "
                f"(index={index_dir or 'auto'}; {n_depth} with depth paths)"
            )
        else:
            print(
                f"ImagePairDataset at {data_root!r} has 0 pairs "
                "(need index/train_local npz + images). Falling back to synthetic.",
                file=sys.stderr,
            )

    if not use_real:
        print(
            "No usable MegaDepth pairs — using synthetic pairs.\n"
            "See scripts/prepare_megadepth1500.sh",
            file=sys.stderr,
        )
        ds = SyntheticPairDataset(length=int(dcfg.get("synthetic_length", 1024)), image_size=(h, w))
    return ds


class ResumableRandomSampler(Sampler[int]):
    """Shuffled order that is a pure function of (seed, epoch), starting at ``start``.

    Resuming at step ``s`` rebuilds the same permutation for epoch
    ``s // steps_per_epoch`` and skips the already-consumed prefix without
    loading those samples.
    """

    def __init__(self, num_samples: int, seed: int, epoch: int = 0, start: int = 0) -> None:
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start = int(start)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        perm = torch.randperm(self.num_samples, generator=g).tolist()
        return iter(perm[self.start :])

    def __len__(self) -> int:
        return max(self.num_samples - self.start, 0)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _make_grad_scaler(enabled: bool, init_scale: float = 2.0**16):
    try:
        return torch.amp.GradScaler("cuda", init_scale=init_scale, enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)


_AMP_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _float_outputs(out: Dict[str, Any]) -> Dict[str, Any]:
    """Upcast the maps the loss reads so every loss term is computed in FP32.

    BCE is not autocast-safe, the score clamp (1 - 1e-4) is not representable in
    FP16, and dual-softmax log-probs need FP32 range.
    """
    res = dict(out)
    for k in ("scores", "descriptors"):
        v = res.get(k)
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
            res[k] = v.float()
    return res


def _first_nonfinite(*outs: Dict[str, Any]) -> Optional[str]:
    """Name of the first loss input map containing NaN/inf, else None."""
    for i, out in enumerate(outs):
        for k in ("scores", "descriptors"):
            v = out.get(k)
            if isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all()):
                return f"{k}{i}"
    return None


def train_loop(cfg: Dict[str, Any], device: str, resume: Optional[str] = None) -> Dict[str, Any]:
    """Train until ``train.max_steps`` optimizer steps have completed.

    ``resume`` is a checkpoint path, ``"auto"`` (newest in ``train.output_dir``)
    or None. Returns ``{"step", "model", "output_dir", "final_checkpoint"}``.
    """
    tcfg = cfg["train"]
    out_dir = Path(str(tcfg.get("output_dir", "runs/bitfeat"))).expanduser()
    max_steps = int(tcfg.get("max_steps", 1000))
    log_every = max(int(tcfg.get("log_every", 10)), 1)
    ckpt_every = int(tcfg.get("checkpoint_every", 1000))
    keep_last = int(tcfg.get("keep_last", 3))
    batch_size = int(tcfg["batch_size"])
    seed = int(tcfg.get("seed", 0))
    grad_clip = float(tcfg.get("grad_clip", 0.0) or 0.0)
    max_nonfinite = int(tcfg.get("max_nonfinite_steps", 20))
    device_type = torch.device(device).type

    amp_dtype = _AMP_DTYPES.get(str(tcfg.get("amp_dtype", "float16")).lower())
    if amp_dtype is None:
        raise ValueError(f"train.amp_dtype must be float16 or bfloat16, got {tcfg.get('amp_dtype')!r}")
    amp = bool(tcfg.get("amp", False))
    if amp and device_type != "cuda":
        print(f"AMP requested but device is {device!r}; training in FP32.", file=sys.stderr)
        amp = False
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bfloat16 not supported on this GPU; using float16 + GradScaler.", file=sys.stderr)
        amp_dtype = torch.float16
    use_scaler = amp and amp_dtype == torch.float16
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", False))

    _seed_everything(seed)
    ds = build_dataset(cfg)
    steps_per_epoch = len(ds) // batch_size
    if steps_per_epoch < 1:
        raise ValueError(f"dataset has {len(ds)} samples, fewer than batch_size={batch_size}")

    model = build_model(cfg).to(device)
    criterion = build_criterion(tcfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    scaler = _make_grad_scaler(use_scaler)

    step = 0
    resume_extra: Dict[str, Any] = {}
    resume_path: Optional[Path] = None
    if resume:
        if str(resume) == "auto":
            resume_path = latest_checkpoint(out_dir)
            if resume_path is None:
                print(f"--resume auto: no checkpoint in {out_dir}; starting fresh.")
        else:
            resume_path = Path(resume).expanduser()
            if not resume_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {resume_path}")
    if resume_path is not None:

        def _check(raw: Dict[str, Any]) -> None:
            old_model = (raw.get("config") or {}).get("model")
            if old_model and old_model != cfg["model"]:
                raise ValueError(
                    f"model config differs from checkpoint {resume_path}: "
                    f"{old_model} != {cfg['model']}"
                )

        ckpt = load_checkpoint(
            resume_path, model=model, optimizer=optim, scaler=scaler, map_location=device, validate=_check
        )
        step = int(ckpt["step"])
        resume_extra = ckpt.get("extra") or {}
        old_cfg = ckpt.get("config") or {}
        old_bs = (old_cfg.get("train") or {}).get("batch_size")
        if old_bs is not None and int(old_bs) != batch_size:
            print(
                f"WARNING: batch_size {batch_size} != checkpoint batch_size {old_bs}; "
                "data order will not match an uninterrupted run.",
                file=sys.stderr,
            )
        print(f"Resumed from {resume_path} at step {step}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    metrics_f = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")

    print(
        f"Training: steps {step}->{max_steps} | batch={batch_size} "
        f"image_size={tcfg.get('image_size')} | amp={'off' if not amp else str(amp_dtype).replace('torch.', '')} "
        f"grad_checkpoint={bool(tcfg.get('grad_checkpoint', False))} | out={out_dir}"
    )

    def _save(path: Path) -> Path:
        return save_checkpoint(
            path, model=model, optimizer=optim, scaler=scaler, step=step, config=cfg,
            extra={
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "degenerate_skipped": degenerate_skipped,
            },
        )

    # Stop cleanly on SIGINT / SIGTERM: finish the current step, checkpoint, exit.
    stop = {"sig": None}

    def _on_signal(signum, _frame):
        if stop["sig"] is not None:  # second signal: give up immediately
            raise KeyboardInterrupt
        stop["sig"] = signum
        print(f"\nReceived signal {signum}; checkpointing after this step...", file=sys.stderr, flush=True)

    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[sig] = signal.signal(sig, _on_signal)
        except ValueError:  # not in main thread (e.g. some test runners)
            pass

    autocast_ctx = (
        (lambda: torch.autocast(device_type=device_type, dtype=amp_dtype)) if amp else contextlib.nullcontext
    )
    model.train()
    nonfinite_run = 0
    amp_skipped = 0
    degenerate_skipped = int(resume_extra.get("degenerate_skipped", 0))
    t_last, step_last = time.perf_counter(), step
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    final_path: Optional[Path] = None
    try:
        while step < max_steps and stop["sig"] is None:
            epoch, offset = divmod(step, steps_per_epoch)
            sampler = ResumableRandomSampler(len(ds), seed, epoch=epoch, start=offset * batch_size)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=int(tcfg.get("num_workers", 0)),
                drop_last=True,
                pin_memory=(device_type == "cuda"),
                # Private generator: creating the iterator must not consume the
                # global torch RNG, or a resumed run would diverge.
                generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch),
            )
            for batch in loader:
                if step >= max_steps or stop["sig"] is not None:
                    break
                batch = _batch_to_device(batch, device)
                with autocast_ctx():
                    out0 = model(batch["image0"])
                    out1 = model(batch["image1"])
                f0, f1 = _float_outputs(out0), _float_outputs(out1)
                del out0, out1
                optim.zero_grad(set_to_none=True)
                # Check the maps before the loss: BCE device-asserts on NaN scores,
                # which would kill the process instead of skipping the step.
                bad = _first_nonfinite(f0, f1)
                if bad is None:
                    losses = criterion(f0, f1, batch)
                    loss = losses["loss"]
                    if not torch.isfinite(loss):
                        bad = "loss"
                else:
                    nan = torch.full((), float("nan"), device=device)
                    losses = {"loss": nan, "loss_desc": nan, "loss_reproj": nan}
                    loss = nan
                del f0, f1
                if bad is not None:
                    nonfinite_run += 1
                    print(f"step={step} non-finite {bad}; skipping update", file=sys.stderr)
                    if nonfinite_run >= max_nonfinite:
                        raise RuntimeError(f"{nonfinite_run} consecutive non-finite steps at step {step} ({bad})")
                elif not loss.requires_grad:
                    # Degenerate pair: no valid depth-warped correspondences, so
                    # every loss term fell back to a constant zero with no graph
                    # (DescriptorLoss / ReprojectionLoss early returns). Nothing to
                    # learn from; skip the update but still consume the step.
                    nonfinite_run = 0
                    degenerate_skipped += 1
                    print(
                        f"step={step} degenerate batch (no valid correspondences, loss has no grad); "
                        f"skipping update [{degenerate_skipped} so far]",
                        file=sys.stderr,
                    )
                else:
                    nonfinite_run = 0
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scale_before = scaler.get_scale() if use_scaler else None
                    scaler.step(optim)  # skipped internally if fp16 grads overflowed
                    scaler.update()
                    if use_scaler and scaler.get_scale() < scale_before:
                        amp_skipped += 1

                if step % log_every == 0:
                    now = time.perf_counter()
                    sps = (step - step_last) / (now - t_last) if step > step_last else float("nan")
                    t_last, step_last = now, step
                    rec = {
                        "step": step,
                        "loss": float(loss.item()),
                        "desc": float(losses["loss_desc"].item()),
                        "reproj": float(losses["loss_reproj"].item()),
                        "steps_per_s": None if math.isnan(sps) else round(sps, 3),
                    }
                    msg = (
                        f"step={step} loss={rec['loss']:.4f} desc={rec['desc']:.4f} "
                        f"reproj={rec['reproj']:.4f}"
                    )
                    if not math.isnan(sps):
                        msg += f" it/s={sps:.2f}"
                    if use_scaler:
                        rec["amp_scale"] = scaler.get_scale()
                        rec["amp_skipped"] = amp_skipped
                        msg += f" scale={rec['amp_scale']:.0f} skipped={amp_skipped}"
                    if degenerate_skipped:
                        rec["degenerate_skipped"] = degenerate_skipped
                        msg += f" degenerate={degenerate_skipped}"
                    if device_type == "cuda":
                        rec["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
                        msg += f" mem={rec['peak_mem_mib']}MiB"
                    print(msg, flush=True)
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()

                step += 1
                if ckpt_every > 0 and step % ckpt_every == 0 and step < max_steps:
                    _save(checkpoint_path(out_dir, step))
                    prune_checkpoints(out_dir, keep_last)

        if step >= max_steps:
            final_path = _save(out_dir / FINAL_NAME)
            print(f"Done: {step} steps; final checkpoint {final_path}")
        else:
            p = _save(checkpoint_path(out_dir, step))
            prune_checkpoints(out_dir, keep_last)
            print(f"Stopped at step {step}; checkpoint {p} (continue with --resume auto)")
    finally:
        metrics_f.close()
        for sig, h in old_handlers.items():
            signal.signal(sig, h)

    if stop["sig"] is not None and step < max_steps:
        # Preserve conventional exit status for the signal (130 for SIGINT, 143 for SIGTERM).
        raise SystemExit(128 + int(stop["sig"]))
    return {
        "step": step,
        "model": model,
        "output_dir": out_dir,
        "final_checkpoint": final_path,
        "degenerate_skipped": degenerate_skipped,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BitFeat (1.58-bit local features)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--smoke", action="store_true", help="Run a few synthetic steps and exit")
    p.add_argument("--smoke-steps", type=int, default=3)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data.root (e.g. /workspace/datasets/megadepth)",
    )
    p.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Scene-info npz dir relative to data-root or absolute "
        "(default: index/train_local)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override train.image_size",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    p.add_argument("--max-steps", type=int, default=None, help="Override train.max_steps")
    p.add_argument("--log-every", type=int, default=None, help="Override train.log_every")
    p.add_argument("--output-dir", type=str, default=None, help="Run dir for checkpoints/metrics (train.output_dir)")
    p.add_argument("--checkpoint-every", type=int, default=None, help="Override train.checkpoint_every (0 = only final)")
    p.add_argument("--keep-last", type=int, default=None, help="Periodic checkpoints to keep (train.keep_last)")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from, or 'auto' for the newest one in --output-dir",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mixed precision via torch.autocast + GradScaler (CUDA only; train.amp)",
    )
    p.add_argument("--amp-dtype", type=str, default=None, choices=["float16", "bfloat16"])
    p.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Gradient checkpointing for the full-res stem + first stage (train.grad_checkpoint)",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    t = cfg.setdefault("train", {})
    d = cfg.setdefault("data", {})
    if args.image_size is not None:
        t["image_size"] = list(args.image_size)
    if args.data_root is not None:
        d["root"] = args.data_root
    if args.index_dir is not None:
        d["index_dir"] = args.index_dir
    for arg_name, key in (
        ("batch_size", "batch_size"),
        ("max_steps", "max_steps"),
        ("log_every", "log_every"),
        ("output_dir", "output_dir"),
        ("checkpoint_every", "checkpoint_every"),
        ("keep_last", "keep_last"),
        ("amp", "amp"),
        ("amp_dtype", "amp_dtype"),
        ("grad_checkpoint", "grad_checkpoint"),
        ("seed", "seed"),
    ):
        v = getattr(args, arg_name, None)
        if v is not None:
            t[key] = v
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg_path = args.config if Path(args.config).is_file() else None
    if cfg_path is None and args.config and not args.smoke:
        print(f"Config not found ({args.config}); using built-in defaults.", file=sys.stderr)
    cfg = load_config(cfg_path if cfg_path and Path(cfg_path).is_file() else None)
    cfg = apply_cli_overrides(cfg, args)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        if args.image_size is None and cfg["train"].get("image_size") == [480, 480]:
            cfg["train"]["image_size"] = [64, 64]
            cfg["train"]["batch_size"] = min(cfg["train"]["batch_size"], 2)
            cfg["model"]["channels"] = [32, 64, 64, 64]
            cfg["model"]["blocks_per_stage"] = [1, 1, 1, 1]
        smoke_train(cfg, steps=args.smoke_steps, device=device)
    else:
        train_loop(cfg, device=device, resume=args.resume)


if __name__ == "__main__":
    main()
