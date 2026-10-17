"""FP32 detection and descriptor heads (DKD-style soft detection stub)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreHead(nn.Module):
    """Dense keypoint score map (FP32).

    Simple 1x1 → ReLU → 1x1 → SoftPlus-ish positive scores. A DKD-style soft
    detection path can replace the NMS stub later.
    """

    def __init__(self, in_channels: int, mid_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Return score map of shape (B, 1, H, W), values in (0, 1) via sigmoid."""
        return torch.sigmoid(self.net(feat))


class DescriptorHead(nn.Module):
    """Dense L2-normalized descriptor map (FP32)."""

    def __init__(
        self,
        in_channels: int,
        desc_dim: int = 128,
        mid_channels: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, desc_dim, 1),
        )
        self.desc_dim = desc_dim

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Return descriptors of shape (B, D, H, W), L2-normalized per pixel."""
        desc = self.net(feat)
        return F.normalize(desc, p=2, dim=1)


def soft_nms_keypoints(
    scores: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    top_k: int = 1024,
    nms_radius: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DKD-style stub: max-pool NMS + top-k selection.

    Args:
        scores: (B, 1, H, W)
        descriptors: (B, D, H, W)
        top_k: Max keypoints per image.
        nms_radius: Local suppression radius (odd kernel = 2*r+1).

    Returns:
        keypoints: (B, K, 2) in (x, y) pixel coords (padded with -1 if fewer).
        kpt_scores: (B, K)
        kpt_descs: (B, K, D)
    """
    b, _, h, w = scores.shape
    d = descriptors.shape[1]
    ksize = 2 * nms_radius + 1
    pooled = F.max_pool2d(scores, kernel_size=ksize, stride=1, padding=nms_radius)
    keep = (scores == pooled).float() * scores  # (B, 1, H, W)

    flat = keep.view(b, -1)
    k = min(top_k, flat.shape[1])
    vals, idx = torch.topk(flat, k=k, dim=1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    keypoints = torch.stack([xs, ys], dim=-1)  # (B, K, 2)

    # Gather descriptors.
    # idx: (B, K) into HxW; descriptors: (B, D, H, W)
    desc_flat = descriptors.view(b, d, -1)  # (B, D, HW)
    idx_exp = idx.unsqueeze(1).expand(-1, d, -1)  # (B, D, K)
    kpt_descs = desc_flat.gather(2, idx_exp).transpose(1, 2)  # (B, K, D)
    kpt_descs = F.normalize(kpt_descs, p=2, dim=-1)

    return keypoints, vals, kpt_descs
