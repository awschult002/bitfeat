"""Train BitFeat: ``python -m bitfeat.train --config configs/default.yaml``.

Use ``--smoke`` for a few synthetic CPU/GPU steps without MegaDepth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

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
        },
        "data": {"root": "", "split": "train"},
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
    )


def smoke_train(cfg: Dict[str, Any], steps: int = 3, device: str = "cpu") -> None:
    """Run a few dummy steps on synthetic pairs (no GPU/data required)."""
    tcfg = cfg["train"]
    h, w = tcfg.get("image_size", [64, 64])
    # Smaller spatial size for fast CPU smoke if still 480 — keep configurable.
    ds = SyntheticPairDataset(length=max(steps * tcfg["batch_size"], 4), image_size=(h, w))
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0)
    model = build_model(cfg).to(device)
    criterion = MatchingLoss(
        desc_weight=tcfg.get("desc_weight", 1.0),
        reproj_weight=tcfg.get("reproj_weight", 0.1),
    )
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
            img0 = batch["image0"].to(device)
            img1 = batch["image1"].to(device)
            out0 = model(img0)
            out1 = model(img1)
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


def train_loop(cfg: Dict[str, Any], device: str) -> None:
    """Full training entry (MegaDepth path still a TODO)."""
    tcfg = cfg["train"]
    data_root = cfg.get("data", {}).get("root", "")
    if not data_root:
        print(
            "No data.root configured — falling back to synthetic pairs.\n"
            "Set data.root to a MegaDepth directory for real training.",
            file=sys.stderr,
        )
        h, w = tcfg.get("image_size", [480, 480])
        ds = SyntheticPairDataset(length=1024, image_size=(h, w))
    else:
        from bitfeat.data.pairs import ImagePairDataset

        ds = ImagePairDataset(root=data_root, split=cfg["data"].get("split", "train"))
        if len(ds) == 0:
            raise RuntimeError(
                f"ImagePairDataset at {data_root!r} is empty — "
                "implement MegaDepth loading in bitfeat/data/pairs.py"
            )

    loader = DataLoader(
        ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 0)),
        drop_last=True,
    )
    model = build_model(cfg).to(device)
    criterion = MatchingLoss(
        desc_weight=tcfg.get("desc_weight", 1.0),
        reproj_weight=tcfg.get("reproj_weight", 0.1),
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    model.train()
    max_steps = int(tcfg.get("max_steps", 1000))
    log_every = int(tcfg.get("log_every", 10))
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            img0 = batch["image0"].to(device)
            img1 = batch["image1"].to(device)
            out0 = model(img0)
            out1 = model(img1)
            losses = criterion(out0, out1, batch)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optim.step()
            if step % log_every == 0:
                print(
                    f"step={step} loss={losses['loss'].item():.4f} "
                    f"desc={losses['loss_desc'].item():.4f} "
                    f"reproj={losses['loss_reproj'].item():.4f}"
                )
            step += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BitFeat (1.58-bit local features)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--smoke", action="store_true", help="Run a few synthetic steps and exit")
    p.add_argument("--smoke-steps", type=int, default=3)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override train.image_size (useful for fast smoke)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg_path = args.config if Path(args.config).is_file() else None
    if cfg_path is None and args.config and not args.smoke:
        # Allow missing config when smoking from an unset cwd.
        if Path(args.config).exists() is False:
            print(f"Config not found ({args.config}); using built-in defaults.", file=sys.stderr)
    cfg = load_config(cfg_path if cfg_path and Path(cfg_path).is_file() else None)

    if args.image_size is not None:
        cfg["train"]["image_size"] = list(args.image_size)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        # Default to a smaller canvas for CPU smoke if still at 480.
        if args.image_size is None and cfg["train"].get("image_size") == [480, 480]:
            cfg["train"]["image_size"] = [64, 64]
            cfg["train"]["batch_size"] = min(cfg["train"]["batch_size"], 2)
            # Narrower channels for speed while keeping architecture shape.
            cfg["model"]["channels"] = [32, 64, 64, 64]
            cfg["model"]["blocks_per_stage"] = [1, 1, 1, 1]
        smoke_train(cfg, steps=args.smoke_steps, device=device)
    else:
        train_loop(cfg, device=device)


if __name__ == "__main__":
    main()
