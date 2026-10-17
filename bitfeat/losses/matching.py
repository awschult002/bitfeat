"""MegaDepth matching losses: warp, score CE, dual-softmax/circle."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp_keypoints(
    kpts0: torch.Tensor,
    depth0: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp (x,y) keypoints from view0 to view1 via depth + pose.

    Args:
        kpts0: (B, N, 2) pixel coords in image0 (x, y)
        depth0: (B, 1, H, W)
        K0, K1: (B, 3, 3)
        T_0to1: (B, 4, 4)

    Returns:
        kpts1: (B, N, 2) projected coords
        valid: (B, N) bool mask
    """
    b, n, _ = kpts0.shape
    _, _, h, w = depth0.shape
    device = kpts0.device

    x = kpts0[..., 0]
    y = kpts0[..., 1]
    grid = torch.stack(
        [
            2.0 * x / max(w - 1, 1) - 1.0,
            2.0 * y / max(h - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(b, n, 1, 2)
    d = F.grid_sample(depth0, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
    d = d.view(b, n)

    ones = torch.ones(b, n, device=device, dtype=kpts0.dtype)
    pix = torch.stack([x, y, ones], dim=-1)
    K0_inv = torch.linalg.inv(K0)
    rays = torch.matmul(K0_inv.unsqueeze(1), pix.unsqueeze(-1)).squeeze(-1)
    X0 = rays * d.unsqueeze(-1)

    R = T_0to1[:, :3, :3]
    t = T_0to1[:, :3, 3]
    X1 = torch.matmul(R.unsqueeze(1), X0.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1)
    z1 = X1[..., 2]
    proj = torch.matmul(K1.unsqueeze(1), X1.unsqueeze(-1)).squeeze(-1)
    eps = 1e-6
    x1 = proj[..., 0] / (proj[..., 2] + eps)
    y1 = proj[..., 1] / (proj[..., 2] + eps)
    kpts1 = torch.stack([x1, y1], dim=-1)

    valid = (
        (d > 0)
        & (z1 > eps)
        & (x1 >= 0)
        & (y1 >= 0)
        & (x1 <= w - 1)
        & (y1 <= h - 1)
    )
    return kpts1, valid


def sample_correspondences(
    depth0: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
    num_samples: int = 512,
    border: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random pixels in view0 with valid depth and warp to view1."""
    b, _, h, w = depth0.shape
    device = depth0.device
    ys = torch.randint(border, max(h - border, border + 1), (b, num_samples), device=device)
    xs = torch.randint(border, max(w - border, border + 1), (b, num_samples), device=device)
    pts0 = torch.stack([xs.float(), ys.float()], dim=-1)
    pts1, valid = warp_keypoints(pts0, depth0, K0, K1, T_0to1)
    return pts0, pts1, valid


def gather_descriptors(desc: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinear sample descriptors at pts -> (B, N, D)."""
    b, d, h, w = desc.shape
    n = pts.shape[1]
    grid = torch.stack(
        [
            2.0 * pts[..., 0] / max(w - 1, 1) - 1.0,
            2.0 * pts[..., 1] / max(h - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(b, n, 1, 2)
    samp = F.grid_sample(desc, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return samp.view(b, d, n).permute(0, 2, 1).contiguous()


def gaussian_heatmap(
    pts: torch.Tensor,
    valid: torch.Tensor,
    hw: Tuple[int, int],
    sigma: float = 1.5,
) -> torch.Tensor:
    """Build soft GT score maps from projected points -> (B, 1, H, W)."""
    b, n, _ = pts.shape
    h, w = hw
    device = pts.device
    yy = torch.arange(h, device=device, dtype=pts.dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=pts.dtype).view(1, 1, 1, w)
    heat = torch.zeros(b, 1, h, w, device=device, dtype=pts.dtype)
    for i in range(n):
        px = pts[:, i, 0].view(b, 1, 1, 1)
        py = pts[:, i, 1].view(b, 1, 1, 1)
        m = valid[:, i].view(b, 1, 1, 1).float()
        g = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma ** 2))
        heat = torch.max(heat, g * m)
    return heat.clamp(0.0, 1.0)


class DescriptorLoss(nn.Module):
    """Dual-softmax (LoFTR-style) or circle loss on sampled correspondences."""

    def __init__(
        self,
        temperature: float = 0.1,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        mode: str = "dual_softmax",
        num_samples: int = 512,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.mode = mode
        self.num_samples = num_samples

    def dual_softmax(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        pts0: torch.Tensor,
        pts1: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        d0 = F.normalize(gather_descriptors(desc0, pts0), dim=-1)
        d1 = F.normalize(gather_descriptors(desc1, pts1), dim=-1)
        sim = torch.matmul(d0, d1.transpose(-1, -2)) / self.temperature
        log_p = F.log_softmax(sim, dim=-1) + F.log_softmax(sim, dim=-2)
        b, n, _ = sim.shape
        eye = torch.arange(n, device=sim.device)
        diag = log_p[:, eye, eye]
        w = valid.float()
        return -(diag * w).sum() / (w.sum() + 1e-6)

    def circle_loss(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        pts0: torch.Tensor,
        pts1: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        d0 = F.normalize(gather_descriptors(desc0, pts0), dim=-1)
        d1 = F.normalize(gather_descriptors(desc1, pts1), dim=-1)
        sim = torch.matmul(d0, d1.transpose(-1, -2))
        b, n, _ = sim.shape
        eye = torch.eye(n, device=sim.device, dtype=torch.bool).unsqueeze(0)
        pos = sim.masked_select(eye.expand_as(sim)).view(b, n)
        neg = sim.masked_fill(eye, -2.0)
        gamma = 80.0
        pos_loss = torch.logsumexp(-gamma * (pos - self.pos_margin) * valid.float(), dim=-1)
        neg_loss = torch.logsumexp(
            gamma * (neg + self.neg_margin).clamp_min(0) * valid.unsqueeze(-1).float(), dim=-1
        )
        w = (valid.sum(dim=-1) > 0).float()
        return ((pos_loss + neg_loss) * w).sum() / (w.sum() + 1e-6)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        matches: Optional[torch.Tensor] = None,
        depth0: Optional[torch.Tensor] = None,
        K0: Optional[torch.Tensor] = None,
        K1: Optional[torch.Tensor] = None,
        T_0to1: Optional[torch.Tensor] = None,
        has_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = desc0.device
        if matches is not None:
            pts0 = matches[..., :2]
            pts1 = matches[..., 2:]
            valid = torch.ones(pts0.shape[:2], dtype=torch.bool, device=device)
        elif depth0 is not None and K0 is not None and K1 is not None and T_0to1 is not None:
            pts0, pts1, valid = sample_correspondences(
                depth0, K0, K1, T_0to1, num_samples=self.num_samples
            )
            if has_depth is not None:
                valid = valid & has_depth.view(-1, 1).to(device=device)
        else:
            g0 = F.normalize(desc0.mean(dim=(-2, -1)), dim=-1)
            g1 = F.normalize(desc1.mean(dim=(-2, -1)), dim=-1)
            return (1.0 - (g0 * g1).sum(dim=-1)).mean()

        if valid.sum() < 2:
            return desc0.new_zeros(())
        if self.mode == "circle":
            return self.circle_loss(desc0, desc1, pts0, pts1, valid)
        return self.dual_softmax(desc0, desc1, pts0, pts1, valid)


class ReprojectionLoss(nn.Module):
    """Score-map supervision via warped correspondences (peak CE)."""

    def __init__(self, sigma: float = 1.5, num_samples: int = 256) -> None:
        super().__init__()
        self.sigma = sigma
        self.num_samples = num_samples

    def forward(
        self,
        scores0: torch.Tensor,
        scores1: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        depth0: Optional[torch.Tensor] = None,
        depth1: Optional[torch.Tensor] = None,
        K0: Optional[torch.Tensor] = None,
        K1: Optional[torch.Tensor] = None,
        T_0to1: Optional[torch.Tensor] = None,
        has_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = T_0to1 if T_0to1 is not None else pose
        if depth0 is None or K0 is None or K1 is None or T is None:
            return self._peakiness(scores0) * 0.5 + self._peakiness(scores1) * 0.5

        pts0, pts1, valid = sample_correspondences(
            depth0, K0, K1, T, num_samples=self.num_samples
        )
        if has_depth is not None:
            valid = valid & has_depth.view(-1, 1).to(device=valid.device)
        if valid.sum() < 1:
            return scores0.new_zeros(())

        h, w = scores0.shape[-2:]
        heat0 = gaussian_heatmap(pts0, valid, (h, w), sigma=self.sigma)
        heat1 = gaussian_heatmap(pts1, valid, (h, w), sigma=self.sigma)
        s0 = scores0.clamp(1e-4, 1.0 - 1e-4)
        s1 = scores1.clamp(1e-4, 1.0 - 1e-4)
        return 0.5 * (F.binary_cross_entropy(s0, heat0) + F.binary_cross_entropy(s1, heat1))

    @staticmethod
    def _peakiness(s: torch.Tensor) -> torch.Tensor:
        flat = s.flatten(1)
        p = flat / (flat.sum(dim=1, keepdim=True) + 1e-6)
        entropy = -(p * (p + 1e-6).log()).sum(dim=1)
        return entropy.mean()


class MatchingLoss(nn.Module):
    """Combined descriptor + reprojection loss with graceful geometry skips."""

    def __init__(
        self,
        desc_weight: float = 1.0,
        reproj_weight: float = 0.1,
        temperature: float = 0.1,
        desc_mode: str = "dual_softmax",
        num_samples: int = 512,
    ) -> None:
        super().__init__()
        self.desc_loss = DescriptorLoss(
            temperature=temperature, mode=desc_mode, num_samples=num_samples
        )
        self.reproj_loss = ReprojectionLoss(num_samples=min(256, num_samples))
        self.desc_weight = desc_weight
        self.reproj_weight = reproj_weight

    def forward(
        self,
        out0: Dict[str, torch.Tensor],
        out1: Dict[str, torch.Tensor],
        batch: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = batch or {}
        device = out0["descriptors"].device

        def _to(key: str):
            v = batch.get(key)
            if isinstance(v, torch.Tensor):
                return v.to(device)
            return v

        depth0 = _to("depth0")
        K0, K1 = _to("K0"), _to("K1")
        T = _to("T_0to1")
        if T is None:
            T = _to("pose")
        has_depth = _to("has_depth")
        matches = _to("matches")

        skip_geom = False
        if matches is None:
            if has_depth is not None and isinstance(has_depth, torch.Tensor):
                if not bool(has_depth.any()):
                    skip_geom = True
            elif depth0 is None or (isinstance(depth0, torch.Tensor) and not bool((depth0 > 0).any())):
                skip_geom = True

        if skip_geom:
            d = self.desc_loss(out0["descriptors"], out1["descriptors"])
            r = out0["scores"].new_zeros(())
        else:
            d = self.desc_loss(
                out0["descriptors"],
                out1["descriptors"],
                matches=matches,
                depth0=depth0,
                K0=K0,
                K1=K1,
                T_0to1=T,
                has_depth=has_depth,
            )
            r = self.reproj_loss(
                out0["scores"],
                out1["scores"],
                pose=T,
                depth0=depth0,
                depth1=_to("depth1"),
                K0=K0,
                K1=K1,
                T_0to1=T,
                has_depth=has_depth,
            )

        total = self.desc_weight * d + self.reproj_weight * r
        return {"loss": total, "loss_desc": d.detach(), "loss_reproj": r.detach()}
