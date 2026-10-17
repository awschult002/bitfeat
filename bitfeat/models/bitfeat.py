"""BitFeatNet: backbone + FP32 heads → scores, descriptors, optional keypoints."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from bitfeat.models.backbone import BitFeatBackbone
from bitfeat.models.heads import DescriptorHead, ScoreHead, soft_nms_keypoints


def _upsample_dense(scores_lr: torch.Tensor, desc_lr: torch.Tensor, h: int, w: int):
    scores = F.interpolate(scores_lr, size=(h, w), mode="bilinear", align_corners=False)
    descriptors = F.interpolate(desc_lr, size=(h, w), mode="bilinear", align_corners=False)
    return scores, F.normalize(descriptors, p=2, dim=1)


class BitFeatNet(nn.Module):
    """End-to-end 1.58-bit local feature network.

    Forward returns a dict with:
        scores: (B, 1, H, W) at input resolution
        descriptors: (B, D, H, W) at input resolution (bilinear upsample)
        features: list of backbone maps (optional, training)
        keypoints / keypoint_scores / keypoint_descriptors when ``detect=True``
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256, 256),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        desc_dim: int = 128,
        method: str = "absmean",
        top_k: int = 1024,
        nms_radius: int = 4,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = BitFeatBackbone(
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            method=method,
            grad_checkpoint=grad_checkpoint,
        )
        c_last = channels[-1]
        self.score_head = ScoreHead(c_last)
        self.desc_head = DescriptorHead(c_last, desc_dim=desc_dim)
        self.desc_dim = desc_dim
        self.grad_checkpoint = bool(grad_checkpoint)
        self.top_k = top_k
        self.nms_radius = nms_radius

    def forward(
        self,
        x: torch.Tensor,
        *,
        detect: bool = False,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            x: (B, 3, H, W) RGB in roughly [0, 1].
            detect: If True, also run soft-NMS keypoint extraction.
            top_k: Override default top-k keypoints.
        """
        feats = self.backbone(x)
        deep = feats[-1]
        scores_lr = self.score_head(deep)
        desc_lr = self.desc_head(deep)

        # Upsample dense maps to input resolution for loss / extraction convenience.
        h, w = x.shape[-2:]
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            # The full-res (B, D, H, W) upsample + L2-normalize intermediates are
            # the largest activations at 480x480; recompute them in backward.
            scores, descriptors = _grad_ckpt(
                _upsample_dense, scores_lr, desc_lr, h, w, use_reentrant=False
            )
        else:
            scores, descriptors = _upsample_dense(scores_lr, desc_lr, h, w)

        out: Dict[str, Any] = {
            "scores": scores,
            "descriptors": descriptors,
            "scores_lr": scores_lr,
            "descriptors_lr": desc_lr,
            "features": feats,
        }

        if detect:
            k = top_k if top_k is not None else self.top_k
            kpts, kpt_scores, kpt_descs = soft_nms_keypoints(
                scores, descriptors, top_k=k, nms_radius=self.nms_radius
            )
            out["keypoints"] = kpts
            out["keypoint_scores"] = kpt_scores
            out["keypoint_descriptors"] = kpt_descs

        return out
