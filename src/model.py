import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .blocks import ResidualMixedMambaBlock, PixelShuffleUpsampler


class MambaESR(nn.Module):
    def __init__(
        self,
        scale=4,
        embed_dim=32,
        num_rmmb=4,
        mixers_per_block=2,
        low_rank=4,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.scale = scale
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Shallow feature extraction
        self.head = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)

        # Body with Residual Mixed Mamba Blocks
        self.body = nn.ModuleList(
            [
                ResidualMixedMambaBlock(
                    embed_dim, mixers_per_block=mixers_per_block, low_rank=low_rank
                )
                for _ in range(num_rmmb)
            ]
        )

        # Feature fusion with residual connection
        self.body_fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Upsampling
        self.upsampler = PixelShuffleUpsampler(embed_dim, scale=scale)

        # Reconstruction
        self.tail = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x, return_features=False):
        features = []
        feat = self.head(x)
        features.append(feat)  # Capture shallow features

        res = feat
        for block in self.body:
            if self.use_gradient_checkpointing and self.training:
                res = checkpoint.checkpoint(block, res, use_reentrant=False)
            else:
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

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
