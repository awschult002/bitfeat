"""Extract keypoints + descriptors: ``python -m bitfeat.extract --image path.jpg``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bitfeat.models.bitfeat import BitFeatNet


def load_image(path: str, size: Optional[tuple[int, int]] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, 3
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W


def build_model(checkpoint: Optional[str], device: str, top_k: int) -> BitFeatNet:
    model = BitFeatNet(top_k=top_k)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract BitFeat keypoints and descriptors")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="Output .npz path")
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    return p.parse_args(argv)


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint, device, top_k=args.top_k)
    size = tuple(args.resize) if args.resize else None
    x = load_image(args.image, size=size).to(device)
    out = model(x, detect=True, top_k=args.top_k)

    result = {
        "keypoints": out["keypoints"][0].cpu().numpy(),
        "scores": out["keypoint_scores"][0].cpu().numpy(),
        "descriptors": out["keypoint_descriptors"][0].cpu().numpy(),
        "image_shape": list(x.shape[-2:]),
    }
    out_path = args.out or str(Path(args.image).with_suffix(".bitfeat.npz"))
    np.savez_compressed(out_path, **result)
    meta = {
        "out": out_path,
        "num_keypoints": int(result["keypoints"].shape[0]),
        "desc_dim": int(result["descriptors"].shape[-1]),
        "image_shape": result["image_shape"],
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
