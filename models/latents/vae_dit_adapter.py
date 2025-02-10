import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    A simple residual block with GroupNorm + SiLU + 3x3 conv, repeated twice.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class VaeDitAdapter(nn.Module):
    """
    A small CNN-based adapter that maps DiT latents -> VAE latents.
    You can expand or shrink the hidden_dim and num_blocks to your needs.
    """
    def __init__(self, in_channels=4, hidden_dim=32, num_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

        # A small stack of residual blocks
        self.resblocks = nn.Sequential(*[BasicBlock(hidden_dim) for _ in range(num_blocks)])

        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, z_diff):
        """
        z_diff: (B,4,64,64) latents from the diffusion model
        return: (B,4,64,64) adjusted latents that better match the VAE's distribution
        """
        x = self.conv_in(z_diff)
        x = self.resblocks(x)
        x = self.conv_out(x)
        return x
