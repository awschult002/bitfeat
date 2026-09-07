"""BitFeatBackbone: deeper+wider CNN with BitConv stages and an FP32 stem.

Channel progression ~2x a tiny baseline (e.g. ALIKE-tiny-ish 32→64→128→128
becomes 64→128→256→256) so ternary capacity can approach FP quality at small
scales (BitNet b1.58 Reloaded / When are 1.58 bits enough?).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitfeat.layers.bitconv import BitConv2d


class BitResidualBlock(nn.Module):
    """Two BitConv layers with residual connection and ReLU."""

    def __init__(self, channels: int, method: str = "absmean") -> None:
        super().__init__()
        self.conv1 = BitConv2d(channels, channels, 3, padding=1, bias=False, method=method)  # type: ignore[arg-type]
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = BitConv2d(channels, channels, 3, padding=1, bias=False, method=method)  # type: ignore[arg-type]
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class BitDownBlock(nn.Module):
    """Stride-2 BitConv projection + residual blocks at the new resolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int = 2,
        method: str = "absmean",
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            BitConv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False, method=method),  # type: ignore[arg-type]
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[BitResidualBlock(out_ch, method=method) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class BitFeatBackbone(nn.Module):
    """Deeper+wider BitConv backbone.

    Default stages: stem 3→64 (FP32), then BitConv stages
    64 → 128 → 256 → 256 with 2 residual blocks each (4 stages / 8 BitConv
    residual blocks + downs).
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 256, 256),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        method: str = "absmean",
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length")

        c0 = channels[0]
        # FP32 stem — keep early filters full-precision for stability.
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        stages: List[nn.Module] = []
        # First stage stays at full resolution (no downsample).
        stages.append(
            nn.Sequential(
                *[BitResidualBlock(c0, method=method) for _ in range(blocks_per_stage[0])]
            )
        )
        in_ch = c0
        for out_ch, n_blocks in zip(channels[1:], blocks_per_stage[1:]):
            stages.append(BitDownBlock(in_ch, out_ch, num_blocks=n_blocks, method=method))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        self.out_channels = list(channels)
        self.out_strides = [1] + [2 ** i for i in range(1, len(channels))]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale feature maps (one per stage)."""
        feats: List[torch.Tensor] = []
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats

    def forward_last(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only the deepest feature map."""
        return self.forward(x)[-1]
