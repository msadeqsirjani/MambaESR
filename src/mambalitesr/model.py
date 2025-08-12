from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResidualMixedMambaBlock


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_channels: int, scale: int):
        super().__init__()
        assert scale in (2, 3, 4), "Scale must be 2, 3, or 4"
        layers = []
        if scale in (2, 3):
            layers += [
                nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
            ]
        elif scale == 4:
            # Two x2 steps for stability
            for _ in range(2):
                layers += [
                    nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                ]
        self.upsampler = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(x)


class MambaLiteSR(nn.Module):
    """
    Student model: 4 Residual Mixed Mamba Blocks (RMMB), each with 2 mixers (total 8 mixers).
    Embedding dimension default 32. Reconstruction via pixel shuffle upsampler.
    """

    def __init__(self, scale: int = 4, embed_dim: int = 32, num_rmmb: int = 4, mixers_per_block: int = 2, low_rank: int = 4):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim

        # Shallow feature extraction
        self.head = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)

        # Body with RMMBs
        self.body = nn.ModuleList(
            [ResidualMixedMambaBlock(embed_dim, mixers_per_block=mixers_per_block, low_rank=low_rank) for _ in range(num_rmmb)]
        )
        self.body_fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Upsample and reconstruct
        self.upsampler = PixelShuffleUpsampler(embed_dim, scale=scale)
        self.tail = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = feat
        for block in self.body:
            res = block(res)
        res = self.body_fuse(res)
        feat = feat + res
        up = self.upsampler(feat)
        out = self.tail(up)
        return out

    @torch.inference_mode()
    def forward_chop(self, x: torch.Tensor, shave: int = 10, min_size: int = 160000) -> torch.Tensor:
        # Optional: implement tiled forward for large images
        return self.forward(x)
