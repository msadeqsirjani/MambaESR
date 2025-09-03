import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Custom Selective State Space Model similar to Mamba
    Implements selective scan mechanism for efficient sequence modeling
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            dt_rank = math.ceil(self.d_model / 16)
        self.dt_rank = dt_rank
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # x_proj takes in `x` and outputs the input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, device=None, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=None, dtype=torch.float32))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x):
        """
        x : (batch, length, dim)
        Returns: same shape as x
        """
        batch, seqlen, dim = x.shape
        
        # Input projections
        xz = self.in_proj(x)  # (batch, seqlen, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (batch, seqlen, d_inner) each
        
        # Apply convolution
        x = x.transpose(-1, -2)  # (batch, d_inner, seqlen)
        x = self.conv1d(x)[..., :seqlen]
        x = x.transpose(-1, -2)  # (batch, seqlen, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM step
        y = self.selective_scan(x)
        
        # Gating mechanism
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        return output
    
    def selective_scan(self, x):
        """
        Selective scan mechanism - core of Mamba
        """
        batch, seqlen, d_inner = x.shape
        
        # Get input-dependent parameters
        x_dbl = self.x_proj(x)  # (batch, seqlen, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project dt
        dt = self.dt_proj(dt)  # (batch, seqlen, d_inner)
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize A and B using zero-order hold - memory-efficient version
        dt = F.softplus(dt)  # (batch, seqlen, d_inner)
        
        # Process in chunks to reduce memory usage
        chunk_size = min(seqlen, 64)  # Process sequences in smaller chunks
        dA_chunks = []
        dB_chunks = []
        
        for i in range(0, seqlen, chunk_size):
            end_i = min(i + chunk_size, seqlen)
            dt_chunk = dt[:, i:end_i]  # (batch, chunk_size, d_inner)
            B_chunk = B[:, i:end_i]    # (batch, chunk_size, d_state)
            
            dA_chunk = torch.exp(torch.einsum("bld,dn->bldn", dt_chunk, A))  # (batch, chunk_size, d_inner, d_state)
            dB_chunk = torch.einsum("bld,bln->bldn", dt_chunk, B_chunk)      # (batch, chunk_size, d_inner, d_state)
            
            dA_chunks.append(dA_chunk)
            dB_chunks.append(dB_chunk)
        
        dA = torch.cat(dA_chunks, dim=1)  # (batch, seqlen, d_inner, d_state)
        dB = torch.cat(dB_chunks, dim=1)  # (batch, seqlen, d_inner, d_state)
        
        # Selective scan - simplified version
        h = torch.zeros(batch, d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys = []
        
        for i in range(seqlen):
            # x[:, i:i+1] has shape (batch, 1, d_inner), we need (batch, d_inner, 1)
            x_i = x[:, i].unsqueeze(-1)  # (batch, d_inner, 1)
            h = dA[:, i] * h + dB[:, i] * x_i  # (batch, d_inner, d_state)
            # h: (batch, d_inner, d_state), C: (batch, d_state)  
            # We need to sum over state dimension
            y = (h * C[:, i].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            y = y + self.D.unsqueeze(0) * x[:, i]
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, seqlen, d_inner)
        return y


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

        # Create mixers with our custom SelectiveSSM
        for _ in range(mixers_per_block):
            mixer = SelectiveSSM(
                d_model=embed_dim,
                d_state=low_rank,  # Low-rank state size
                d_conv=4,
                expand=2,
            )
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
