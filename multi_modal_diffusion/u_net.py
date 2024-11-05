import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import torch as th
from abc import abstractmethod

from arch_utils import (conv_nd, avg_pool_nd, normalization, zero_module, count_flops_attn, checkpoint)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, image, tabular, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class InitialBlock(nn.Module):
    def __init__(
            self,
            image_in_channels,
            tabular_in_features,
            image_out_channels,
            tabular_out_features,
            kernel_size=3,
            stride=1,
            padding = "same",
            dilation = 1
    ):
        super().__init__()

        # Image processing layer: 2D convolution for images
        self.image_conv = conv_nd(2, image_in_channels, image_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        # Tabular data processing layer: fully connected layer
        self.tabular_fc = conv_nd(0, tabular_in_features, tabular_out_features)

    def forward(self, image, tabular):
        # Process image data
        image_out = self.image_conv(image)

        # Process tabular data
        tabular_out = self.tabular_fc(tabular)

        return image_out, tabular_out


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied after upsampling.
    :param dims: determines if the signal is 1D (for tabular data) or 2D (for images).
    :param out_channels: optional, the number of output channels. If not specified, defaults to the input channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        # Define the stride based on data type
        if dims == 1:
            # For tabular data: we'll increase the feature size with a linear layer
            self.linear_upsample = conv_nd(0, channels, self.out_channels)
        elif dims == 2:
            # For image data: standard spatial upsampling
            self.stride = 2
            if use_conv:
                self.conv = conv_nd(2, self.channels, self.out_channels, kernel_size=3, padding=1)
        else:
            raise ValueError("Unsupported dimensions for Upsample. Use dims=1 for tabular or dims=2 for images.")

    def forward(self, x):
        if self.dims == 1:
            # Tabular data upsampling with linear layer
            x = self.linear_upsample(x)
        elif self.dims == 2:
            # Image data upsampling with interpolation
            x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
            if self.use_conv:
                x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: number of channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied after downsampling.
    :param dims: determines if the signal is 1D (for tabular data) or 2D (for images).
    :param out_channels: optional, the number of output channels. If not specified, defaults to the input channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if dims == 1:
            # For tabular data, downsampling with a linear layer
            self.linear_downsample = conv_nd(0, channels, self.out_channels)
        elif dims == 2:
            # For image data, standard spatial downsampling
            self.stride = 2
            if use_conv:
                self.op = self.conv = conv_nd(2, self.channels, self.out_channels, kernel_size=3, padding=1)
            else:
                self.op = avg_pool_nd(dims, kernel_size=self.stride, stride=self.stride)
        else:
            raise ValueError("Unsupported dimensions for Downsample. Use dims=1 for tabular or dims=2 for images.")

    def forward(self, x):
        if self.dims == 1:
            # Tabular data downsampling
            x = self.linear_downsample(x)
        elif self.dims == 2:
            # Image data downsampling
            x = self.op(x)
        return x


class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order, adapted to handle both image and tabular data.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: Input tensor shaped [N, 3 * H * C, S], where S is the flattened spatial dimension for images or the feature dimension for tabular data.
        :return: Output tensor shaped [N, H * C, S] after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        # Split Q, K, V from concatenated input
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        # Calculate attention weights and apply them to values
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        # Reshape back to the expected output shape
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class SingleModalAtten(nn.Module):
    """
    An attention block that allows spatial (for images) or feature (for tabular data) positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels

        # Set the number of attention heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)

        # Define qkv and projection layers; they will adapt based on input shape
        self.qkv = conv_nd(1, channels, channels * 3, kernel_size=1)  # Conv1d for flexible shape handling
        self.attention = SingleModalQKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, kernel_size=1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        """
        Forward method for applying attention.

        :param x: For images, [batch, channels, height, width]; for tabular, [batch, channels, features]
        :return: x with attention applied.
        """
        b, c, *spatial = x.shape

        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)



class ResBlock(TimestepBlock):
    """
    A residual block adapted for image and tabular data that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of output channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_scale_shift_norm: if True, applies scale and shift conditioning with timestep embedding.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param image_attention: if True, use attention in the image model.
    :param tabular_attention: if True, use attention in the tabular model.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
        use_conv=False,
        image_attention=False,
        tabular_attention=False,
        num_heads=4,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # Image processing layers
        self.image_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(2, channels, self.out_channels, kernel_size=3)
        )

        # Tabular processing layers
        self.tabular_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(1, channels, self.out_channels, kernel_size=3)
        )

        # Upsampling and Downsampling
        self.updown = up or down
        if up:
            self.img_upd = Upsample(channels, use_conv, dims=2)
            self.img_upd_orig = Upsample(channels, use_conv, dims=2)  # For the original image input
            self.tab_upd = Upsample(channels, use_conv, dims=1)
            self.tab_upd_orig = Upsample(channels, use_conv, dims=1)  # For the original tabular input
        elif down:
            self.img_upd = Downsample(channels, use_conv, dims=2)
            self.img_upd_orig = Downsample(channels, use_conv, dims=2)  # For the original image input
            self.tab_upd = Upsample(channels, use_conv, dims=1)
            self.tab_upd_orig = Upsample(channels, use_conv, dims=1)  # For the original tabular input
        else:
            self.img_upd = self.img_upd_orig = self.tab_upd = self.tab_upd_orig = nn.Identity()

        # Embedding for timestep conditioning
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )

        # Output layers for image and tabular data
        self.image_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(2, channels, self.out_channels, kernel_size=1))
        )

        self.tabular_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(1, channels, self.out_channels, kernel_size=1))
        )

        # Skip connections
        if self.out_channels == channels:
            self.image_skip_connection = nn.Identity()
            self.tabular_skip_connection = nn.Identity()
        elif use_conv:
            self.image_skip_connection = conv_nd(2, channels, self.out_channels, kernel_size=3)
            self.tabular_skip_connection = conv_nd(1, channels, self.out_channels, kernel_size=3)
        else:
            self.image_skip_connection = conv_nd(2, channels, self.out_channels, kernel_size=1)
            self.tabular_skip_connection = conv_nd(1, channels, self.out_channels, kernel_size=1)

        # Attention blocks (optional)
        self.image_attention = image_attention
        self.tabular_attention = tabular_attention

        if self.image_attention:
            self.image_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, num_head_channels=-1,
                use_checkpoint=use_checkpoint
            )
        if self.tabular_attention:
            self.tabular_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, num_head_channels=-1,
                use_checkpoint=use_checkpoint
            )

    def forward(self, image, tabular, emb):
        """
        Apply the block to image and tabular data, conditioned on a timestep embedding.
        """
        return checkpoint(self._forward, (image, tabular, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, image, tabular, emb):
        """
        Image shape: [batch, channels, height, width]
        Tabular shape: [batch, features]
        """

        b, c, h, w = image.shape  # For image
        _, _, f = tabular.shape  # For tabular data, where f is the number of features

        if self.updown:
            # Processed features
            img_h = self.image_in_layers(image)
            img_h = self.img_upd(img_h)

            tab_h = self.tabular_in_layers(tabular)
            tab_h = self.tab_upd(tab_h)

            # Original inputs transformed to match processed features
            image = self.img_up_orig(image)
            tabular = self.tab_up_orig(tabular)
        else:
            img_h = self.image_in_layers(image)
            tab_h = self.tabular_in_layers(tabular)

        # Process timestep embedding
        emb_out = self.emb_layers(emb).type(image.dtype)

        if self.use_scale_shift_norm:
            # Scale and shift for images
            img_out_norm, img_out_rest = self.image_out_layers[0], self.image_out_layers[1:]
            img_emb_out = emb_out[:, None, :, None, None]  # Reshape to broadcast across height and width
            scale, shift = th.chunk(img_emb_out, 2, dim=2)
            img_h = img_out_norm(img_h) * (1 + scale) + shift
            img_h = img_out_rest(img_h)

            # Scale and shift for tabular data
            tab_out_norm, tab_out_rest = self.tabular_out_layers[0], self.tabular_out_layers[1:]
            tab_emb_out = emb_out[..., None]  # Reshape to [batch, channels, 1] to broadcast across features
            scale, shift = th.chunk(tab_emb_out, 2, dim=1)
            tab_h = tab_out_norm(tab_h) * (1 + scale) + shift
            tab_h = tab_out_rest(tab_h)

        else:
            # If no scale-shift normalization, directly add the embedding
            img_emb_out = emb_out[:, None, :, None, None]  # Broadcast across spatial dimensions for images
            img_h = img_h + img_emb_out
            img_h = self.image_out_layers(img_h)

            tab_emb_out = emb_out[..., None]  # Broadcast across features for tabular data
            tab_h = tab_h + tab_emb_out
            tab_h = self.tabular_out_layers(tab_h)

        # Add skip connections
        image_out = self.image_skip_connection(image) + img_h
        tabular_out = self.tabular_skip_connection(tabular) + tab_h

        # Apply attention if enabled
        if self.image_attention:
            image_out = rearrange(image_out, "b c h w -> b (h w) c")
            image_out = self.spatial_attention_block(image_out)
            image_out = rearrange(image_out, "b (h w) c -> b c h w", h=h, w=w)

        if self.tabular_attention:
            tabular_out = self.tabular_attention_block(tabular_out)

        return image_out, tabular_out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order for image and tabular data.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, image_len, tabular_len):
        """
        Apply QKV attention.

        :param qkv: A tensor of Qs, Ks, and Vs concatenated, [batch, 3 * n_heads * channels, length]
        :param image_len: Number of tokens in the image portion.
        :param tabular_len: Number of tokens in the tabular portion.
        :return: Attention outputs for both image and tabular tokens.
        """

        bs, width, _ = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        # Split Q, K, V from concatenated input
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(ch)

        # Separate Q, K, V for image and tabular parts
        img_q, tab_q = q[:, :, :image_len], q[:, :, image_len:]
        img_k, tab_k = k[:, :, :image_len], k[:, :, image_len:]
        img_v, tab_v = v[:, :, :image_len], v[:, :, image_len:]

        # Compute attention for images and tabular data
        img_weight = th.softmax(th.einsum("bct,bcs->bts", img_q * scale, img_k * scale), dim=-1)
        img_a = th.einsum("bts,bcs->bct", img_weight, img_v)

        tab_weight = th.softmax(th.einsum("bct,bcs->bts", tab_q * scale, tab_k * scale), dim=-1)
        tab_a = th.einsum("bts,bcs->bct", tab_weight, tab_v)

        return img_a.reshape(bs, -1, image_len), tab_a.reshape(bs, -1, tabular_len)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for image and tabular data.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        local_window=1,
        window_shift=False,
    ):
        super().__init__()
        self.channels = channels

        # Set the number of heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                self.channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = self.channels // num_head_channels

        self.local_window = local_window
        self.window_shift = window_shift
        self.use_checkpoint = use_checkpoint

        # Normalization and QKV computation for image and tabular data
        self.img_norm = normalization(self.channels)
        self.tab_norm = normalization(self.channels)
        self.img_qkv = conv_nd(1, self.channels, self.channels * 3, kernel_size=1)
        self.tab_qkv = conv_nd(1, self.channels, self.channels * 3, kernel_size=1)
        self.attention = QKVAttention(self.num_heads)

        # Projection layers for image and tabular
        self.img_proj_out = zero_module(conv_nd(2, self.channels, self.channels, kernel_size=1))
        self.tab_proj_out = zero_module(conv_nd(1, self.channels, self.channels, kernel_size=1))

    def forward(self, image, tabular):
        return checkpoint(self._forward, (image, tabular), self.parameters(), True)

    def _forward(self, image, tabular):
        """
        Forward method for cross-attention between image and tabular data.

        :param image: Image tokens of shape [batch, channels, height * width]
        :param tabular: Tabular tokens of shape [batch, channels, features]
        """
        b, c, hw = image.shape  # Flattened image tokens
        _, _, f = tabular.shape  # Tabular features

        # Normalize and compute QKV for image and tabular data
        img_qkv = self.img_qkv(self.img_norm(image))
        tab_qkv = self.tab_qkv(self.tab_norm(tabular))
        qkv = th.cat([img_qkv, tab_qkv], dim=2)

        # Cross attention between image and tabular tokens
        img_h, tab_h = self.attention(qkv, image_len=hw, tabular_len=f)

        # Project output back to original shape
        img_h = self.img_proj_out(img_h).view(b, c, hw)
        tab_h = self.tab_proj_out(tab_h).view(b, c, f)

        # Skip connection
        image_out = image + img_h
        tabular_out = tabular + tab_h

        return image_out, tabular_out
