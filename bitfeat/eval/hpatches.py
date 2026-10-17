"""HPatches Mean Matching Accuracy (MMA) stub / CLI.

Expected data::

    /workspace/datasets/hpatches/hpatches-sequences-release/
      i_*/  v_*/   # illumination / viewpoint sequences
        1.ppm ... 6.ppm
        H_1_2 ... H_1_6

Usage::

    python -m bitfeat.eval.hpatches \\
      --hpatches-root /workspace/datasets/hpatches/hpatches-sequences-release \\
      --checkpoint path/to/bitfeat.pt \\
      --thresholds 1 3 5

Without a checkpoint, runs a dry structure check and prints sequence counts.
Full MMA wires BitFeatNet extract → mutual NN match → homography warp error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def list_sequences(root: Path) -> List[Path]:
    seqs = sorted([p for p in root.iterdir() if p.is_dir() and (p.name.startswith("i_") or p.name.startswith("v_"))])
    return seqs


def load_homography(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=np.float64).reshape(3, 3)


def mma_for_pair(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches01: np.ndarray,
    H_0to1: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[float, float]:
    """Compute MMA at pixel thresholds.

    Args:
        kpts0/kpts1: (N,2) / (M,2)
        matches01: (K,2) index pairs into kpts0/kpts1
        H_0to1: 3x3 homography mapping image0 → image1
        thresholds: pixel radii
    """
    if matches01.size == 0:
        return {float(t): 0.0 for t in thresholds}
    pts0 = kpts0[matches01[:, 0]]
    pts1 = kpts1[matches01[:, 1]]
    ones = np.ones((pts0.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([pts0.astype(np.float64), ones], axis=1).T
    proj = H_0to1 @ homog
    proj = (proj[:2] / (proj[2:3] + 1e-8)).T
    err = np.linalg.norm(proj - pts1.astype(np.float64), axis=1)
    return {float(t): float((err <= t).mean()) for t in thresholds}


def mutual_nn_matches(desc0: np.ndarray, desc1: np.ndarray) -> np.ndarray:
    """Brute-force mutual nearest neighbors on L2-normalized descriptors."""
    sim = desc0 @ desc1.T
    nn01 = sim.argmax(axis=1)
    nn10 = sim.argmax(axis=0)
    mutual = []
    for i, j in enumerate(nn01):
        if nn10[j] == i:
            mutual.append((i, j))
    if not mutual:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(mutual, dtype=np.int64)


def dry_check(root: Path) -> None:
    seqs = list_sequences(root)
    print(f"HPatches root: {root}")
    print(f"Sequences: {len(seqs)} (i_={sum(s.name.startswith('i_') for s in seqs)}, "
          f"v_={sum(s.name.startswith('v_') for s in seqs)})")
    if not seqs:
        print("No sequences found — unpack hpatches-sequences-release here.", file=sys.stderr)
        return
    sample = seqs[0]
    imgs = sorted(sample.glob("*.ppm")) + sorted(sample.glob("*.png")) + sorted(sample.glob("*.jpg"))
    Hs = sorted(sample.glob("H_*"))
    print(f"Sample {sample.name}: {len(imgs)} images, {len(Hs)} homographies")


def run_mma(
    root: Path,
    checkpoint: Path | None,
    thresholds: Sequence[float],
    device: str,
    top_k: int,
) -> None:
    dry_check(root)
    if checkpoint is None:
        print("No --checkpoint given; skipping feature extraction / MMA.")
        print("TODO: load BitFeatNet, extract per image, mutual NN, average MMA.")
        return

    import torch
    from PIL import Image

    from bitfeat.models.bitfeat import BitFeatNet

    ckpt = torch.load(str(checkpoint), map_location=device)
    model = BitFeatNet()
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    seqs = list_sequences(root)
    acc: Dict[float, List[float]] = {float(t): [] for t in thresholds}

    def extract(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t, detect=True, top_k=top_k)
        kpts = out["keypoints"][0].detach().cpu().numpy()
        desc = out["keypoint_descriptors"][0].detach().cpu().numpy()
        return kpts, desc

    for seq in seqs:
        imgs = []
        for i in range(1, 7):
            for ext in (".ppm", ".png", ".jpg"):
                p = seq / f"{i}{ext}"
                if p.is_file():
                    imgs.append(p)
                    break
        if len(imgs) < 2:
            continue
        try:
            k0, d0 = extract(imgs[0])
        except Exception as e:
            print(f"skip {seq.name}: {e}", file=sys.stderr)
            continue
        for j in range(2, len(imgs) + 1):
            H_path = seq / f"H_1_{j}"
            if not H_path.is_file():
                continue
            H = load_homography(H_path)
            k1, d1 = extract(imgs[j - 1])
            matches = mutual_nn_matches(d0, d1)
            scores = mma_for_pair(k0, k1, matches, H, thresholds)
            for t, v in scores.items():
                acc[t].append(v)

    print("MMA (mean over pairs):")
    for t in thresholds:
        vals = acc[float(t)]
        mean = float(np.mean(vals)) if vals else float("nan")
        print(f"  @{t:.0f}px: {mean:.4f}  (n={len(vals)})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPatches MMA for BitFeat")
    p.add_argument(
        "--hpatches-root",
        type=str,
        default="/workspace/datasets/hpatches/hpatches-sequences-release",
    )
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--top-k", type=int, default=1024)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.hpatches_root)
    if not root.is_dir():
        print(f"HPatches root not found: {root}", file=sys.stderr)
        sys.exit(1)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_mma(root, ckpt, args.thresholds, device, args.top_k)


if __name__ == "__main__":
    main()
