import torch
import torch.nn as nn
from .blocks import ResidualMixedMambaBlock, PixelShuffleUpsampler

class MambaLiteSR(nn.Module):
    def __init__(self, scale=4, embed_dim=32, num_rmmb=4, mixers_per_block=2, low_rank=4):
        super().__init__()
        self.scale = scale
        
        # Shallow feature extraction
        self.head = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)
        
        # Body with Residual Mixed Mamba Blocks
        self.body = nn.ModuleList([
            ResidualMixedMambaBlock(
                embed_dim, 
                mixers_per_block=mixers_per_block,
                low_rank=low_rank
            ) for _ in range(num_rmmb)
        ])
        
        # Feature fusion
        self.body_fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsampler = PixelShuffleUpsampler(embed_dim, scale=scale)
        
        # Reconstruction
        self.tail = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)
    
    def forward(self, x, return_features=False):
        features = []
        feat = self.head(x)
        features.append(feat)  # Capture shallow features
        
        res = feat
        for block in self.body:
            res = block(res)
            features.append(res)  # Capture block outputs
        
        res = self.body_fuse(res)
        feat = feat + res
        features.append(feat)  # Capture fused features
        
        up = self.upsampler(feat)
        out = self.tail(up)
        
        if return_features:
            return out, features
        return out