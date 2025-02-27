import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
#                           Basic Building Blocks
##############################################################################

class ResBlock(nn.Module):
    """
    Simple residual block:
      in -> Conv -> GN -> SiLU -> Conv -> GN -> SiLU -> + skip
    If in_channels != out_channels, use a 1x1 to match dims.
    """
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch

        self.skip_conv = None
        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(16, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(16, out_ch)

    def forward(self, x):
        identity = x
        if self.skip_conv is not None:
            identity = self.skip_conv(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.silu(out)

        return out + identity


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention on [B, C, H, W].
    Flatten H*W as tokens, do attention, then reshape back.
    """
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads

        self.qkv_proj = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Project Q,K,V
        qkv = self.qkv_proj(x)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3*self.num_heads, self.head_dim, H*W)
        q, k, v = torch.split(qkv, self.num_heads, dim=1)  # each [B, heads, head_dim, HW]

        # Scaled dot-prod attn
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdk,bhdj->bhkj', q, k) * scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhkj,bhdj->bhdk', attn, v)  # [B, heads, head_dim, HW]
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)

        return x + out  # residual


class SkipGating(nn.Module):
    """
    Gated skip:
      - optional self-attention on skip feature
      - gating mask from skip (1x1 conv -> sigmoid)
      - final skip = skip_feat * gate
    """
    def __init__(self, channels, attn_heads=2):
        super().__init__()
        self.skip_attn = SelfAttentionBlock(channels, num_heads=attn_heads)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, skip_feat):
        # (1) local attn
        skip_feat = self.skip_attn(skip_feat)

        # (2) gating
        gate = torch.sigmoid(self.gate_conv(skip_feat))
        return skip_feat * gate


##############################################################################
#                           Down and Up Blocks
##############################################################################

class DownBlock(nn.Module):
    """
    2x downsample + resblock + optional attention
    """
    def __init__(self, in_ch, out_ch, attn_heads=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.res = ResBlock(in_ch, out_ch)
        if attn_heads > 0:
            self.attn = SelfAttentionBlock(out_ch, num_heads=attn_heads)
        else:
            self.attn = None

    def forward(self, x):
        x = self.pool(x)
        x = self.res(x)
        if self.attn:
            x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling + (Gated) skip + ResBlock + optional attention
    Then produce a 4-channel output (for multi-scale supervision).
    """
    def __init__(self, in_ch, out_ch, skip_ch, attn_heads=0, skip_attn_heads=2):
        """
        in_ch: channels of the incoming feature to be upsampled
        skip_ch: channels of the skip feature
        out_ch: channels after merging
        """
        super().__init__()
        # Upsample
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # Gated skip
        self.skip_gate = SkipGating(skip_ch, attn_heads=skip_attn_heads)

        # 1x1 to unify skip & up-ch before the main ResBlock
        self.merge_conv = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1)

        self.res = ResBlock(out_ch, out_ch)

        if attn_heads > 0:
            self.attn = SelfAttentionBlock(out_ch, num_heads=attn_heads)
        else:
            self.attn = None

        # final 1x1 conv to produce 4 channels at this resolution
        self.out_conv = nn.Conv2d(out_ch, 4, kernel_size=1)

    def forward(self, x, skip):
        # 1) upsample
        x = self.up(x)

        # 2) skip gating
        skip_gated = self.skip_gate(skip)

        # 3) merge
        x = torch.cat([x, skip_gated], dim=1)
        x = self.merge_conv(x)

        # 4) res block + optional attention
        x = self.res(x)
        if self.attn:
            x = self.attn(x)

        # 5) produce 4-ch output at this scale
        out_scale = self.out_conv(x)

        return x, out_scale


##############################################################################
#                           Definitive Adapter UNet
##############################################################################

class VaeDitAdapterUNet(nn.Module):
    """
    A large-capacity UNet for 4->4 latent transformation:
      - 4-level encoder, each with DownBlock (res + attention).
      - 4-level decoder, each with UpBlock (gated skip + res + attention).
      - Returns 4 outputs at different scales:
         out_1_8, out_1_4, out_1_2, out_full
      - High capacity: Residual blocks, multi-res self-attention, gating.
    """
    def __init__(self, base_ch=128, attn_heads=2):
        """
        base_ch: #channels at level-0, grows by factor of 2 each down
        attn_heads: number of heads in SelfAttention blocks
        """
        super().__init__()

        # ---------------- Encoder ----------------
        # Level 0 (no downsample, just res + attention)
        self.enc0_res = ResBlock(4, base_ch)
        self.enc0_attn = SelfAttentionBlock(base_ch, attn_heads)

        # Level 1 => 1/2
        self.enc1 = DownBlock(base_ch, base_ch*2, attn_heads=attn_heads)

        # Level 2 => 1/4
        self.enc2 = DownBlock(base_ch*2, base_ch*4, attn_heads=attn_heads)

        # Level 3 => 1/8
        self.enc3 = DownBlock(base_ch*4, base_ch*8, attn_heads=attn_heads)

        # Bottleneck => 1/16
        self.enc4 = DownBlock(base_ch*8, base_ch*8, attn_heads=attn_heads)

        # ---------------- Decoder ----------------
        # up4 => from 1/16 -> 1/8, skip e3
        self.up4 = UpBlock(
            in_ch=base_ch*8,
            out_ch=base_ch*8,
            skip_ch=base_ch*8,
            attn_heads=attn_heads,
            skip_attn_heads=attn_heads
        )

        # up3 => from 1/8 -> 1/4, skip e2
        self.up3 = UpBlock(
            in_ch=base_ch*8,
            out_ch=base_ch*4,
            skip_ch=base_ch*4,
            attn_heads=attn_heads,
            skip_attn_heads=attn_heads
        )

        # up2 => from 1/4 -> 1/2, skip e1
        self.up2 = UpBlock(
            in_ch=base_ch*4,
            out_ch=base_ch*2,
            skip_ch=base_ch*2,
            attn_heads=attn_heads,
            skip_attn_heads=attn_heads
        )

        # up1 => from 1/2 -> 1×, skip e0
        self.up1 = UpBlock(
            in_ch=base_ch*2,
            out_ch=base_ch,
            skip_ch=base_ch,
            attn_heads=attn_heads,
            skip_attn_heads=attn_heads
        )

    def forward(self, x):
        """
        x: [B, 4, H, W] latents.
        Returns 4 outputs for multi-scale loss:
          out_1_8, out_1_4, out_1_2, out_full
        """

        # ----------------- Encoder Pass -----------------
        # Level 0
        e0 = self.enc0_res(x)      # [B, base_ch, H, W]
        e0 = self.enc0_attn(e0)    # attn at 1×

        # Level 1 => 1/2
        e1 = self.enc1(e0)         # [B, 2base_ch, H/2, W/2]

        # Level 2 => 1/4
        e2 = self.enc2(e1)         # [B, 4base_ch, H/4, W/4]

        # Level 3 => 1/8
        e3 = self.enc3(e2)         # [B, 8base_ch, H/8, W/8]

        # Bottleneck => 1/16
        e4 = self.enc4(e3)         # [B, 8base_ch, H/16, W/16]

        # ----------------- Decoder Pass -----------------
        # up4 => 1/16 -> 1/8
        u4, out_1_8 = self.up4(e4, e3)  # skip from e3

        # up3 => 1/8 -> 1/4
        u3, out_1_4 = self.up3(u4, e2)  # skip from e2

        # up2 => 1/4 -> 1/2
        u2, out_1_2 = self.up2(u3, e1)  # skip from e1

        # up1 => 1/2 -> 1x
        u1, out_full = self.up1(u2, e0)  # skip from e0

        return out_full, out_1_2, out_1_4, out_1_8
