import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention on [B, C, H, W].
    We flatten spatial dims (H*W) as "tokens," do multi-head attention,
    then reshape back. Residual connection is included.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert (channels % num_heads) == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads

        self.qkv_proj = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Project Q,K,V
        qkv = self.qkv_proj(x)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3*self.num_heads, self.head_dim, H*W)
        q, k, v = torch.split(qkv, self.num_heads, dim=1)  # each [B, heads, head_dim, HW]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhej->bhij', q, k) * scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)        # [B, heads, head_dim, HW]

        # Reshape / merge heads
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)
        return x + out  # residual

class DoubleConv(nn.Module):
    """
    (Conv -> GroupNorm -> SiLU) x2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=16, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=16, num_channels=out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        return x

class DownBlock(nn.Module):
    """
    Downsample by 2x (avg_pool) + double conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    """
    Upsample by 2x (nearest), concat skip, double conv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class VaeDitAdapterUNet(nn.Module):
    """
    A single 4->4 UNet, deeper (4 down/up levels),
    with multiple attention blocks at the lower resolutions
    and multi-scale outputs for deep supervision.

    - base_ch: controls width (#channels)
    - attn_heads: number of heads in attention blocks
    """
    def __init__(self, base_ch=128, attn_heads=4):
        super().__init__()

        ###########################
        #        Encoder
        ###########################
        # Level 0 (no downsample)
        self.inc = DoubleConv(4, base_ch)

        # Level 1
        self.down1 = DownBlock(base_ch, base_ch * 2)

        # Level 2
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)

        # Level 3
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)

        # Insert attention at level 3 if you want:
        self.attn3 = SelfAttentionBlock(base_ch * 8, num_heads=attn_heads)

        # Level 4 (bottleneck)
        self.down4 = DownBlock(base_ch * 8, base_ch * 8)
        self.attn4 = SelfAttentionBlock(base_ch * 8, num_heads=attn_heads)

        ###########################
        #        Decoder
        ###########################
        self.up4 = UpBlock((base_ch * 8) + (base_ch * 8), base_ch * 8)
        self.up3 = UpBlock((base_ch * 8) + (base_ch * 4), base_ch * 4)
        self.up2 = UpBlock((base_ch * 4) + (base_ch * 2), base_ch * 2)
        self.up1 = UpBlock((base_ch * 2) + base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, 4, kernel_size=1)

        ###########################
        #   Multi-Scale Heads
        ###########################
        # Predict 4-ch output at 1/8, 1/4, 1/2 resolution
        self.pred_1_8_head = nn.Conv2d(base_ch * 8, 4, kernel_size=1)
        self.pred_1_4_head = nn.Conv2d(base_ch * 4, 4, kernel_size=1)
        self.pred_1_2_head = nn.Conv2d(base_ch * 2, 4, kernel_size=1)


    def forward(self, x):
        """
        Returns a tuple of multi-scale outputs:
         - out_full (H x W)
         - out_1_2  (H/2 x W/2)
         - out_1_4  (H/4 x W/4)
         - out_1_8  (H/8 x W/8)

        so you can apply deep supervision (multi-scale loss).
        """
        # ============ Encoder ============
        x0 = self.inc(x)      # Level 0: B, base_ch,     H,   W
        x1 = self.down1(x0)   # Level 1: B, base_ch*2,   H/2, W/2
        x2 = self.down2(x1)   # Level 2: B, base_ch*4,   H/4, W/4
        x3 = self.down3(x2)   # Level 3: B, base_ch*8,   H/8, W/8

        # Attention at 1/8 resolution
        x3 = self.attn3(x3)

        x4 = self.down4(x3)   # Level 4: B, base_ch*8,   H/16, W/16

        # Attention at bottleneck
        x4 = self.attn4(x4)

        # ============ Decoder ============
        up4 = self.up4(x4, x3)   # => [B, 8base_ch, H/8, W/8]
        up3 = self.up3(up4, x2)  # => [B, 4base_ch, H/4, W/4]
        up2 = self.up2(up3, x1)  # => [B, 2base_ch, H/2, W/2]
        up1 = self.up1(up2, x0)  # => [B, base_ch,  H,   W]

        out_full = self.outc(up1)  # => [B,4,H,W]

        # Multi-scale outputs
        out_1_8 = self.pred_1_8_head(up4)  # [B,4,H/8,W/8]
        out_1_4 = self.pred_1_4_head(up3)  # [B,4,H/4,W/4]
        out_1_2 = self.pred_1_2_head(up2)  # [B,4,H/2,W/2]

        return out_full, out_1_2, out_1_4, out_1_8