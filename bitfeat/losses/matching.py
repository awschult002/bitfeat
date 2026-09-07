"""Placeholder reprojection / descriptor losses (MegaDepth-ready stubs).

TODO (real training):
  - Warp keypoints with ground-truth depth + relative pose (MegaDepth / DISK).
  - Hard-negative or dual-softmax descriptor loss (SuperGlue / LoFTR style).
  - Peakiness / DKD detection loss on score maps.
  - Optional reliability / uncertainty weighting.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorLoss(nn.Module):
    """Circle / InfoNCE-style stub on dense descriptors.

    Expects optional correspondence coords; without them falls back to a
    trivial self-similarity regularizer so smoke training still runs.
    """

    def __init__(self, temperature: float = 0.1, pos_margin: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        matches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            desc0 / desc1: (B, D, H, W) L2-normalized dense descriptors.
            matches: Optional (B, N, 4) with [x0, y0, x1, y1] correspondences.
                     TODO: sample and apply dual-softmax / circle loss.
        """
        # Smoke-friendly placeholder: encourage unit-norm (already true) and
        # mild cross-view agreement via global average descriptors.
        g0 = F.normalize(desc0.mean(dim=(-2, -1)), dim=-1)  # (B, D)
        g1 = F.normalize(desc1.mean(dim=(-2, -1)), dim=-1)
        # Cosine distance to identity pairing within the batch.
        sim = (g0 * g1).sum(dim=-1)  # (B,)
        loss = (1.0 - sim).mean()

        if matches is not None:
            # TODO: gather descriptors at match locations and run real matching loss.
            _ = matches  # silence unused until MegaDepth wiring
        return loss


class ReprojectionLoss(nn.Module):
    """Geometric reprojection stub for score / keypoint supervision.

    TODO: project depth points with GT pose; penalize score away from
    reprojected locations and encourage peaks at inliers.
    """

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        scores0: torch.Tensor,
        scores1: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        depth0: Optional[torch.Tensor] = None,
        depth1: Optional[torch.Tensor] = None,
        K0: Optional[torch.Tensor] = None,
        K1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            scores0 / scores1: (B, 1, H, W)
            pose, depth*, K*: MegaDepth geometry (unused in stub).
        """
        # Placeholder: peakiness — encourage sparse high-confidence scores.
        # (Similar spirit to SuperPoint / DKD peakiness terms.)
        def peakiness(s: torch.Tensor) -> torch.Tensor:
            # Mean of scores should stay moderate; spatial entropy soft-push.
            flat = s.flatten(1)
            p = flat / (flat.sum(dim=1, keepdim=True) + 1e-6)
            entropy = -(p * (p + 1e-6).log()).sum(dim=1)
            # Lower entropy → peakier; we minimize a mild anti-uniform term.
            return entropy.mean()

        loss = 0.5 * (peakiness(scores0) + peakiness(scores1))
        if pose is not None:
            # TODO: implement warp + Gaussian heatmap supervision.
            _ = (depth0, depth1, K0, K1, pose)
        return loss


class MatchingLoss(nn.Module):
    """Combined descriptor + reprojection loss wrapper."""

    def __init__(
        self,
        desc_weight: float = 1.0,
        reproj_weight: float = 0.1,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.desc_loss = DescriptorLoss(temperature=temperature)
        self.reproj_loss = ReprojectionLoss()
        self.desc_weight = desc_weight
        self.reproj_weight = reproj_weight

    def forward(
        self,
        out0: Dict[str, torch.Tensor],
        out1: Dict[str, torch.Tensor],
        batch: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = batch or {}
        d = self.desc_loss(
            out0["descriptors"],
            out1["descriptors"],
            matches=batch.get("matches"),
        )
        r = self.reproj_loss(
            out0["scores"],
            out1["scores"],
            pose=batch.get("pose"),
            depth0=batch.get("depth0"),
            depth1=batch.get("depth1"),
            K0=batch.get("K0"),
            K1=batch.get("K1"),
        )
        total = self.desc_weight * d + self.reproj_weight * r
        return {"loss": total, "loss_desc": d.detach(), "loss_reproj": r.detach()}
