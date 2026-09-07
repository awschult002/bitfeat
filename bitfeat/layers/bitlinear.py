"""BitLinear: BitNet b1.58 ternary linear layer with STE.

Classic absmean (optional absmedian) ternarization:
    W_q = round(clip(W / (gamma + eps), -1, 1))
    y   = x @ (gamma * W_q)^T + b

Shadow weights stay in FP32; only the forward uses ternary values.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternarize(
    weight: torch.Tensor,
    *,
    method: Literal["absmean", "absmedian"] = "absmean",
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ternarize weights to {-1, 0, +1} and return (W_q, gamma).

    gamma is the per-tensor scale (mean or median of |W|).
    """
    abs_w = weight.abs()
    if method == "absmedian":
        # Flatten for a single median scale (BitNet-style).
        gamma = abs_w.flatten().median().clamp_min(eps)
    else:
        gamma = abs_w.mean().clamp_min(eps)

    # STE: round(clip(W/gamma, -1, 1)) — identity gradient through the quantizer.
    w_scaled = (weight / (gamma + eps)).clamp(-1.0, 1.0)
    w_q = (w_scaled.round() - w_scaled).detach() + w_scaled
    return w_q, gamma


class BitLinear(nn.Module):
    """Linear layer with ternary weights and FP32 shadow parameters.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias (kept FP32).
        method: ``absmean`` or ``absmedian`` scale for ternarization.
        eps: Numerical stabilizer for gamma.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        method: Literal["absmean", "absmedian"] = "absmean",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Small init so early |W|/gamma is near unit scale.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, gamma = ternarize(self.weight, method=self.method, eps=self.eps)
        # Reconstruct effective weight: gamma * W_q (values in {-gamma, 0, +gamma}).
        return F.linear(x, gamma * w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, method={self.method}"
        )
