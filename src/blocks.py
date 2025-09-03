import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")


class LowRankProjection(nn.Module):
    """Low-rank projection layer for parameter efficiency"""

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x):
        return self.V(self.U(x))


class ResidualMixedMambaBlock(nn.Module):
    def __init__(self, embed_dim, mixers_per_block=2, low_rank=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.mixers = nn.ModuleList()

        # Create mixers with official Mamba implementation
        for _ in range(mixers_per_block):
            if MAMBA_AVAILABLE:
                mixer = Mamba(
                    d_model=embed_dim,
                    d_state=low_rank,  # Low-rank state size
                    d_conv=4,
                    expand=2,
                )
            else:
                # Fallback to simple linear layer if mamba_ssm not available
                mixer = nn.Linear(embed_dim, embed_dim)
            self.mixers.append(mixer)

        # Low-rank projection instead of standard linear
        self.proj = LowRankProjection(embed_dim, embed_dim, low_rank)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        residual = x

        # Flatten spatial dimensions for Mamba (which expects 1D sequences)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Apply Mamba mixers
        for mixer in self.mixers:
            if MAMBA_AVAILABLE:
                x = mixer(x)
            else:
                # Fallback: simple linear transformation
                x = mixer(x)

        # Apply low-rank projection
        x = self.proj(x)

        # Reshape back to image
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Add residual and normalize
        x = x + residual
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, embed_dim, scale=4):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim * (scale**2), 3, padding=1)
        self.upsample = nn.PixelShuffle(scale)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return self.activation(x)
