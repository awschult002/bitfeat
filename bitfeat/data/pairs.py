"""Image-pair datasets: MegaDepth/DISK-style stub + synthetic smoke generator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    """Stub for MegaDepth / DISK-style image pairs.

    TODO:
      - Load scene folders with depth maps and camera poses.
      - Sample overlapping pairs via covisibility / overlap heuristics.
      - Return RGB tensors, depths, intrinsics K, relative pose.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (480, 480),
    ) -> None:
        self.root = root
        self.split = split
        self.image_size = image_size
        # No real files in the scaffold commit.
        self.samples: list = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError(
            "Wire MegaDepth paths in ImagePairDataset; use SyntheticPairDataset for smoke tests."
        )


class SyntheticPairDataset(Dataset):
    """Random RGB pairs with identity pose for smoke training / CI."""

    def __init__(
        self,
        length: int = 32,
        image_size: Tuple[int, int] = (480, 480),
        seed: int = 0,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(self.seed + index)
        h, w = self.image_size
        img0 = torch.rand(3, h, w, generator=g)
        # Mild photometric noise as a "second view".
        noise = 0.05 * torch.randn(3, h, w, generator=g)
        img1 = (img0 + noise).clamp(0.0, 1.0)
        pose = torch.eye(4)
        K = torch.tensor(
            [[w * 0.9, 0.0, w / 2.0], [0.0, h * 0.9, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        return {
            "image0": img0,
            "image1": img1,
            "pose": pose,
            "K0": K,
            "K1": K.clone(),
            "depth0": torch.ones(1, h, w),
            "depth1": torch.ones(1, h, w),
            # matches omitted until MegaDepth wiring (None breaks default_collate)
        }
