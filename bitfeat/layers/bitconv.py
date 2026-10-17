"""BitConv2d: Conv2d analogue with BitNet b1.58 ternary weights."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitfeat.layers.bitlinear import ternarize

IntPair = Union[int, Tuple[int, int]]


def _pair(v: IntPair) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    return v


class BitConv2d(nn.Module):
    """2D convolution with ternary weights and FP32 shadow parameters.

    Uses the same absmean / absmedian ternarization as :class:`BitLinear`.
    For 1x1 convolutions this is equivalent to a channel-wise BitLinear.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntPair,
        stride: IntPair = 1,
        padding: IntPair = 0,
        dilation: IntPair = 1,
        groups: int = 1,
        bias: bool = True,
        method: Literal["absmean", "absmedian"] = "absmean",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.method = method
        self.eps = eps

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, gamma = ternarize(self.weight, method=self.method, eps=self.eps)
        return F.conv2d(
            x,
            gamma * w_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}, "
            f"bias={self.bias is not None}, method={self.method}"
        )
