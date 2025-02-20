import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    This follows the implementation from e.g. Denoising Diffusion Probabilistic Models.
    :param timesteps: a 1-D Tensor of timesteps
    :param embedding_dim: the dimension of the output
    :return: Tensor of shape [batch_size, embedding_dim]
    """
    # Make sure timesteps is a float tensor.
    half_dim = embedding_dim // 2
    timesteps = timesteps.float()
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    # Outer product => shape [batch_size, half_dim]
    args = timesteps[:, None] * freqs[None, :]
    # Embed with sin and cos
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if embedding_dim % 2 == 1:  # zero-pad
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


class ResidualBlock(nn.Module):
    """
    A simple residual block that incorporates time-based conditioning.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        """
        x: [batch, in_channels, H, W]
        t_emb: [batch, time_emb_dim]
        """
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        time_emb = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.res_conv(x)


class DownBlock(nn.Module):
    """
    A down-sampling block that:
    1) Applies a ResidualBlock
    2) Returns (downsampled_x, skip_x)
       where skip_x is the output of the ResidualBlock,
       and downsampled_x is skip_x passed through stride-2 conv.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        # Residual block
        x_res = self.res_block(x, t_emb)   # shape: [B, out_channels, H, W]
        # Downsample
        x_down = self.downsample(x_res)    # shape: [B, out_channels, H/2, W/2]
        return x_down, x_res


class UpBlock(nn.Module):
    """
    An up-sampling block that:
    1) Upsamples (ConvTranspose2d) from in_channels to out_channels
    2) Concatenates the skip connection (skip_channels) along the channel dimension
    3) Applies a ResidualBlock on the concatenated features
    """
    def __init__(self, in_channels, out_channels, skip_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )
        self.res_block = ResidualBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim
        )

    def forward(self, x, skip, t_emb):
        """
        :param x: [batch, in_channels, H, W]
        :param skip: [batch, skip_channels, H*2, W*2]
        :param t_emb: time embedding
        :return: [batch, out_channels, H*2, W*2]
        """
        x_up = self.upsample(x)         # => [B, out_channels, H*2, W*2]
        x_cat = torch.cat([x_up, skip], dim=1)  # => [B, out_channels+skip_channels, H*2, W*2]
        return self.res_block(x_cat, t_emb)


class MMToyUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, base_channels=64, tabular_features=174):
        """
        A simple UNet for noise prediction.
        :param in_channels:  number of channels in the input image (3 for RGB).
        :param out_channels: number of channels in the output image (3 for noise prediction in RGB).
        :param time_emb_dim: size of the time embedding vector.
        :param base_channels: base number of channels; the network grows in multiples of this.
        """
        super().__init__()

        self.image_size = tuple(int(x) for x in "3,64,64".split(','))
        self.tabular_size = 174

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_features, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # -------------------------
        # Down blocks
        # -------------------------
        # After down1: shape => base_channels=64, spatial => 32x32
        self.down1 = DownBlock(in_channels, base_channels, time_emb_dim)
        # After down2: shape => 128, spatial => 16x16
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        # After down3: shape => 256, spatial => 8x8
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        # Bottleneck
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # -------------------------
        # Up blocks
        # -------------------------
        # up3: in=256, skip=256, out=128 => output: [B,128,16,16]
        self.up3 = UpBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            skip_channels=base_channels * 4,
            time_emb_dim=time_emb_dim
        )
        # up2: in=128, skip=128, out=64 => output: [B,64,32,32]
        self.up2 = UpBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            skip_channels=base_channels * 2,
            time_emb_dim=time_emb_dim
        )
        # up1: in=64, skip=64, out=64 => output: [B,64,64,64]
        self.up1 = UpBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            skip_channels=base_channels,
            time_emb_dim=time_emb_dim
        )

        # Final 1×1 to map base_channels => out_channels (e.g. 3)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        # For the tabular path, we add a linear layer to predict noise in the same shape as the input
        self.final_fc_tab = nn.Linear(time_emb_dim, tabular_features)

    def forward(self, x_img, x_tab, t):
        """
        Forward pass of the UNet.
        :param x: [batch_size, 3, 64, 64] input images
        :param t: [batch_size] scaled timesteps
        :return: [batch_size, 3, 64, 64] predicted noise
        """

        if x_img.dtype != torch.float32:
            x_img = x_img.float()
        if x_tab.dtype != torch.float32:
            x_tab = x_tab.float()
        if t.dtype != torch.float32:
            t = t.float()

        # 1) Time embedding
        t_emb_in = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb_in)

        # 2) Compute tabular embedding
        tab_emb = self.tabular_mlp(x_tab)  # => [batch_size, time_emb_dim]
        combined_emb = t_emb + tab_emb

        # 3) Downsample path (image branch, conditioned by combined_emb)
        x_d1, skip1 = self.down1(x_img, combined_emb)
        x_d2, skip2 = self.down2(x_d1, combined_emb)
        x_d3, skip3 = self.down3(x_d2, combined_emb)

        # 4) Bottleneck (image branch)
        x_m = self.mid(x_d3, combined_emb)

        # 5) Upsample path (image branch)
        x_u3 = self.up3(x_m, skip3, combined_emb)
        x_u2 = self.up2(x_u3, skip2, combined_emb)
        x_u1 = self.up1(x_u2, skip1, combined_emb)

        # 6) Final 1×1 conv for image noise => [B, out_channels, 64, 64]
        noise_img = self.final_conv(x_u1)

        # 7) Predict tabular noise
        noise_tab = self.final_fc_tab(combined_emb)

        return noise_img, noise_tab


if __name__ == "__main__":

    batch_size = 2
    n_features = 10

    model = MMToyUnet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        base_channels=64,
        tabular_features=n_features
    )

    x_img = torch.randn(batch_size, 3, 64, 64)
    x_tab = torch.randn(batch_size, n_features)
    t = torch.randint(0, 1000, (batch_size,))  # just an example of some timesteps

    noise_img, noise_tab = model(x_img, x_tab, t)

    print("noise_img shape:", noise_img.shape)  # [batch_size, 3, 64, 64]
    print("noise_tab shape:", noise_tab.shape)  # [batch_size, n_features]
