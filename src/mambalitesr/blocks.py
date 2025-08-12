from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    Mamba = None  # type: ignore
    _HAS_MAMBA = False


class DepthwiseConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LowRankPointwise(nn.Module):
    """
    Low-rank 1x1 convolution: C -> r -> C, parameterized by two small matrices.
    This is an efficient channel-mixing approximation.
    """

    def __init__(self, channels: int, rank: int):
        super().__init__()
        rank = max(1, min(rank, channels))
        self.reduce = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.reduce.weight, a=math.sqrt(5))
        nn.init.zeros_(self.expand.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.reduce(x))


class VisionMambaMixer(nn.Module):
    """
    A vision-friendly mixer that uses Mamba if available; otherwise uses
    depthwise conv + gated ffn + low-rank pointwise mixing.
    """

    def __init__(self, channels: int, low_rank: int = 4):
        super().__init__()
        self.channels = channels
        self.low_rank = max(1, low_rank)

        if _HAS_MAMBA:
            # Tokenize spatial dims to sequence, run Mamba, then reshape back
            # Use a small Mamba config suitable for C=channels
            self.norm = nn.LayerNorm(channels)
            self.mamba = Mamba(d_model=channels, d_state=16, d_conv=3, expand=2)
            self.out = nn.Linear(channels, channels)
        else:
            self.pw_in = nn.Conv2d(channels, 2 * channels, kernel_size=1)
            # Apply depthwise conv on C channels after gating
            self.dw = DepthwiseConv2d(channels, kernel_size=3)
            self.act = nn.SiLU()
            self.gate = nn.Sigmoid()
            self.low_rank_pw = LowRankPointwise(channels, rank=self.low_rank)
            self.pw_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MAMBA:
            b, c, h, w = x.shape
            y = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            y = self.norm(y)
            y = self.mamba(y)
            y = self.out(y)
            y = y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return x + y
        else:
            y = self.pw_in(x)
            y1, y2 = y.chunk(2, dim=1)
            y = self.act(y1) * self.gate(y2)  # shape: [B, C, H, W]
            y = self.dw(y)
            y = y + self.low_rank_pw(y)
            y = self.pw_out(y)
            return x + y


class ResidualMixedMambaBlock(nn.Module):
    """Residual block that stacks multiple vision mixers and uses a residual skip."""

    def __init__(self, channels: int, mixers_per_block: int = 2, low_rank: int = 4):
        super().__init__()
        self.mixers = nn.Sequential(*[VisionMambaMixer(channels, low_rank=low_rank) for _ in range(mixers_per_block)])
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.mixers(x)
