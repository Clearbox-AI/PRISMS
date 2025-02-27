import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import torch as th
from abc import abstractmethod

from arch_utils import (conv_nd, avg_pool_nd, normalization, zero_module, count_flops_attn, checkpoint,
                        timestep_embedding)
from fp16_util import (convert_module_to_f16, convert_module_to_f32)
import logger

#TODO:
from runtime.runtime_utils import ShapeManager
# shape_manager = ShapeManager()


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, image, tabular, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, image, tabular, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                image, tabular = layer(image, tabular, emb)
            else:
                image, tabular = layer(image, tabular)
        return image, tabular


class ImageConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            dilation=1,
    ):
        super().__init__()

        # Use conv_nd to create a 2D convolution with padding set to "same"
        self.image_conv = conv_nd(
            2,  # Dimension for image data
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

    def forward(self, image):
        # Apply the convolutional layer
        return self.image_conv(image)


class TabularConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            dilation=1,
            conv_type="1d",
    ):
        super().__init__()

        if conv_type == "1d":
            # For sequential-like tabular data
            self.tabular_conv = conv_nd(1, in_channels, out_channels, kernel_size, stride, padding, dilation)

        elif conv_type == "linear":
            # For independent tabular features
            self.tabular_conv = nn.Linear(in_channels, out_channels)

        else:
            raise NotImplementedError("Unsupported conv_type for tabular data")

    def forward(self, tabular):

        tabular = self.tabular_conv(tabular)
        return tabular


class InitialBlock(nn.Module):
    def __init__(
            self,
            image_in_channels,
            tabular_in_features,
            image_out_channels,
            tabular_out_features,
            kernel_size=3
    ):
        super().__init__()

        # Image processing layer: 2D convolution for images
        self.image_conv = ImageConv(image_in_channels, image_out_channels, kernel_size=kernel_size)

        # Tabular data processing layer: fully connected layer
        self.tabular_fc = TabularConv(tabular_in_features, tabular_out_features, kernel_size=kernel_size, conv_type='1d')

    def forward(self, image, tabular):
        # Process image data
        image_out = self.image_conv(image)

        # Process tabular data
        tabular_out = self.tabular_fc(tabular)

        return image_out, tabular_out


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution for images and tabular data.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D (tabular) or 2D (image).
    :param out_channels: the number of output channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.stride = 2 if dims == 1 else 2  # 2x for tabular (1D), 2x for image (2D)

        # Define convolution based on dimensions if needed
        if use_conv:
            if dims == 2:  # For images
                self.conv = conv_nd(2, self.channels, self.out_channels, kernel_size=3)
            elif dims == 1:  # For tabular
                self.conv = conv_nd(1, self.channels, self.out_channels, kernel_size=3) # TAB_LIN

    def forward(self, x):
        target_shape = shape_manager.load_upsample_shape(self.dims)
        # target_shape = None

        if target_shape is not None:
            if self.dims == 2:
                target_size = target_shape[-2:]  # For 2D data (height, width)

                # Upsample for images with height and width dimensions
                # x = F.interpolate(x, scale_factor=(self.stride, self.stride), mode="nearest")
            elif self.dims == 1:
                target_size = (target_shape[-1],)  # For 1D data (features)

                # Upsample for tabular data
                # x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
            x = F.interpolate(x, size=target_size, mode="nearest")

        else:
            if self.dims == 2:
                # Upsample for images with height and width dimensions
                x = F.interpolate(x, scale_factor=(self.stride, self.stride), mode="nearest")
            elif self.dims == 1:
                # Upsample for tabular data
                x = F.interpolate(x, scale_factor=self.stride, mode="nearest")

        # Apply convolution if specified
        if self.use_conv:
            x = self.conv(x)

        return x



class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution for images and tabular data.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D (tabular) or 2D (image).
    :param out_channels: the number of output channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims == 1 else 2  # Downsampling factor: 2 for tabular, 2 for image

        # Define downsampling operation
        if use_conv:
            if dims == 2:  # For images
                self.op = conv_nd(2, self.channels, self.out_channels, kernel_size=3, stride=stride, padding=1)
            elif dims == 1:  # For tabular
                self.op = conv_nd(1, self.channels, self.out_channels, kernel_size=3, stride=stride)  # TAB_LIN
                # self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)
        else:
            # Use average pooling if convolution is not specified
            if dims == 2:
                self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
            elif dims == 1:
                self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
                # self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
                # self.op = conv_nd(0, self.channels, self.out_channels // 2)  # TAB_LIN
                # self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)
    def forward(self, x):
        # Save the shape of the tensor during downsampling if it's smaller than the last recorded shape
        shape_manager.save_downsample_shape(x.shape, self.dims)

        return self.op(x)



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
    An attention block that allows spatial positions to attend to each other for image data.
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

        # Define qkv and projection layers for the image
        self.qkv = conv_nd(1, channels, channels * 3, kernel_size=1)  # Conv1d for flexible shape handling
        self.attention = SingleModalQKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, kernel_size=1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        """
        Forward method for applying spatial attention.

        :param x: Image tensor shaped [batch, channels, height, width]
        :return: x with spatial attention applied.
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
        tabular_type='1d',
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
            ImageConv(channels, self.out_channels, kernel_size=3)
        )

        # Tabular processing layers
        self.tabular_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            TabularConv(channels, self.out_channels, 3, conv_type="1d")
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
            self.tab_upd = Downsample(channels, use_conv, dims=1)
            self.tab_upd_orig = Downsample(channels, use_conv, dims=1)  # For the original tabular input
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
            zero_module(ImageConv(self.out_channels, self.out_channels, kernel_size=1))
        )

        self.tabular_out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(TabularConv(self.out_channels, self.out_channels, 1, conv_type='1d'))
        )

        # Skip connections
        if self.out_channels == channels:
            self.image_skip_connection = nn.Identity()
            self.tabular_skip_connection = nn.Identity()
        elif use_conv:
            self.image_skip_connection = ImageConv(channels, self.out_channels, kernel_size=3)
            self.tabular_skip_connection = TabularConv(channels, self.out_channels, kernel_size=3, conv_type='1d') #
        else:
            self.image_skip_connection = ImageConv(channels, self.out_channels, kernel_size=1)
            self.tabular_skip_connection = TabularConv(channels, self.out_channels, kernel_size=1, conv_type='1d')

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
        # _, f = tabular.shape  # For tabular data, where f is the number of features

        if self.updown:
            # Processed features
            img_h = self.image_in_layers(image)
            img_h = self.img_upd(img_h)

            tab_h = self.tabular_in_layers(tabular)
            tab_h = self.tab_upd(tab_h)

            # Original inputs transformed to match processed features
            image = self.img_upd_orig(image)
            tabular = self.tab_upd_orig(tabular)
        else:
            img_h = self.image_in_layers(image)
            tab_h = self.tabular_in_layers(tabular)

        # Process timestep embedding
        emb_out = self.emb_layers(emb).type(image.dtype)

        if self.use_scale_shift_norm:
            # Scale and shift for images
            img_out_norm, img_out_rest = self.image_out_layers[0], self.image_out_layers[1:]
            img_emb_out = emb_out[:, :, None, None]  # Reshape to broadcast across height and width
            scale, shift = th.chunk(img_emb_out, 2, dim=1)
            img_h = img_out_norm(img_h) * (1 + scale) + shift
            img_h = img_out_rest(img_h)

            # Scale and shift for tabular data
            tab_out_norm, tab_out_rest = self.tabular_out_layers[0], self.tabular_out_layers[1:]
            # tab_emb_out = emb_out  # Keep as [batch, channels]
            tab_emb_out = emb_out[..., None]
            scale, shift = th.chunk(tab_emb_out, 2, dim=1)
            tab_h = tab_out_norm(tab_h) * (1 + scale) + shift
            tab_h = tab_out_rest(tab_h)

            # tab_emb_out = emb_out[..., None]  # Reshape to [batch, channels, 1] to broadcast across features
            # scale, shift = th.chunk(tab_emb_out, 2, dim=1)
            # tab_h = tab_out_norm(tab_h) * (1 + scale) + shift
            # tab_h = tab_out_rest(tab_h)

        else:
            # If no scale-shift normalization, directly add the embedding
            img_emb_out = emb_out[:, :, None, None]  # Broadcast across spatial dimensions for images
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
            # image_out = rearrange(image_out, "b c h w -> b (h w) c")
            # image_out = self.image_attention_block(image_out)
            # image_out = rearrange(image_out, "b (h w) c -> b c h w", h=h, w=w)

            image_out = rearrange(image_out, "b c h w -> b c (h w)")  # Flatten height and width for spatial attention
            image_out = self.image_attention_block(image_out)  # Apply spatial attention
            image_out = rearrange(image_out, "b c (h w) -> b c h w", h=h, w=w)  # Reshape back to original image dimensions

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
        self.tab_qkv = conv_nd(1, self.channels, self.channels * 3, kernel_size=1)  # TAB_LIN
        self.attention = QKVAttention(self.num_heads)

        # Projection layers for image and tabular
        self.img_proj_out = zero_module(ImageConv(self.channels, self.channels, kernel_size=1))
        self.tab_proj_out = zero_module(TabularConv(self.channels, self.channels, kernel_size=1, conv_type = '1d'))

    def forward(self, image, tabular):
        return checkpoint(self._forward, (image, tabular), self.parameters(), True)

    def _forward(self, image, tabular):
        """
        Forward method for cross-attention between image and tabular data.

        :param image: Image tensor of shape [batch, channels, height, width]
        :param tabular: Tabular tensor of shape [batch, channels, features]
        :return: Updated image and tabular data with cross-attention applied.
        """
        b, c, h, w = image.shape  # Image dimensions
        _, _, f = tabular.shape   # Tabular dimensions

        # Flatten spatial dimensions for image tokens
        image_token = rearrange(image, "b c h w -> b c (h w)")
        tabular_token = tabular

        # Compute QKV for image and tabular data
        i_qkv = self.img_qkv(self.img_norm(image_token))  # [batch, 3*channels, h*w]
        t_qkv = self.tab_qkv(self.tab_norm(tabular_token))  # [batch, 3*channels, f]
        qkv = th.cat([i_qkv, t_qkv], dim=2)

        # Apply cross-attention between image and tabular tokens
        image_h, tabular_h = self.attention(qkv, image_len=h * w, tabular_len=f)

        # Reshape back to original dimensions
        image_h = rearrange(image_h, "b c (h w) -> b c h w", h=h, w=w)
        image_h = self.img_proj_out(image_h)
        image_out = image + image_h

        # Apply projection and skip connection for tabular data
        tabular_h = self.tab_proj_out(tabular_h)
        tabular_out = tabular + tabular_h

        return image_out, tabular_out


class MultimodalUNet(nn.Module):
    """
    The full coupled-UNet model with attention and timestep embedding, adapted for image and tabular data.
    """

    def __init__(
            self,
            image_size,
            tabular_size,
            model_channels,
            image_out_channels,
            tabular_out_channels,
            num_res_blocks,
            cross_attention_resolutions,
            cross_attention_windows,
            cross_attention_shift,
            image_attention_resolutions,
            tabular_attention_resolutions,
            # image_type="2d",
            # tabular_type="1d",
            dropout=0,
            channel_mult=(1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.tabular_size = tabular_size
        self.model_channels = model_channels
        self.image_out_channels = image_out_channels
        self.tabular_out_channels = tabular_out_channels
        self.num_res_blocks = num_res_blocks
        self.cross_attention_resolutions = cross_attention_resolutions
        self.cross_attention_windows = cross_attention_windows
        self.cross_attention_shift = cross_attention_shift
        self.image_attention_resolutions = image_attention_resolutions
        self.tabular_attention_resolutions = tabular_attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels
        self.time_embed = nn.Sequential(
            conv_nd(0, model_channels, time_embed_dim),
            nn.SiLU(),
            conv_nd(0, time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self._feature_size = ch
        input_block_chans = [ch]

        # Initial input blocks #TODO
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(InitialBlock(
            self.image_size[0], self.tabular_size[0], image_out_channels=ch, tabular_out_features=ch
        ))])

        ds = 1
        # dilation = 1

        # Build input blocks and cross attention layers
        for level, mult in enumerate(channel_mult):
            for block_id in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        # image_type=image_type,
                        # tabular_type=tabular_type,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        image_attention=ds in self.image_attention_resolutions,
                        tabular_attention=ds in self.tabular_attention_resolutions,
                        num_heads=num_heads,
                    )
                ]

                ch = int(mult * model_channels)

                if ds in cross_attention_resolutions:
                    ds_i = cross_attention_resolutions.index(ds)
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            local_window=cross_attention_windows[ds_i],
                            window_shift=cross_attention_shift,
                            num_head_channels=num_head_channels,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            # image_type=image_type,
                            # tabular_type=tabular_type,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle blocks
        self.middle_blocks = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                # image_type=image_type,
                # tabular_type=tabular_type,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                image_attention=True,
                tabular_attention=True,
                num_heads=num_heads,
            ),
            CrossAttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                # image_type=image_type,
                # tabular_type=tabular_type,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                image_attention=True,
                tabular_attention=True,
                num_heads=num_heads,
            ),
        )
        self._feature_size += ch

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_id in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        # image_type=image_type,
                        # tabular_type=tabular_type,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        image_attention=ds in self.image_attention_resolutions,
                        tabular_attention=ds in self.tabular_attention_resolutions,
                        num_heads=num_heads,
                    )
                ]

                ch = int(model_channels * mult)
                if ds in cross_attention_resolutions:
                    ds_i = cross_attention_resolutions.index(ds)
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            local_window=cross_attention_windows[ds_i],
                            window_shift=cross_attention_shift,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )

                if level and block_id == num_res_blocks:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                # image_type=image_type,
                                # tabular_type=tabular_type,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                        )
                        ds //= 2

                self._feature_size += ch
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output projections
        self.tabular_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(TabularConv(input_ch, tabular_out_channels, 3, conv_type='1d')),
        )
        self.image_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(ImageConv(input_ch, image_out_channels, kernel_size=3)),
        )

    # Methods for FP16 conversion
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.image_out.apply(convert_module_to_f16)
        self.tabular_out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.video_out.apply(convert_module_to_f32)
        self.audio_out.apply(convert_module_to_f32)

    def load_state_dict_(self, state_dict, is_strict=False):

        for key, val in self.state_dict().items():

            if key in state_dict.keys():
                if val.shape == state_dict[key].shape:
                    continue
                else:
                    state_dict.pop(key)
                    logger.log("{} not matchable with state_dict with shape {}".format(key, val.shape))
            else:

                logger.log("{} not exists in state_dict".format(key))

        for key, val in state_dict.items():
            if key in self.state_dict().keys():
                if val.shape == state_dict[key].shape:
                    continue
            else:
                logger.log("{} not used in state_dict".format(key))
        self.load_state_dict(state_dict, strict=is_strict)
        return

    def forward(self, image, tabular, timesteps, label=None):
        """
        Apply the model to an input batch.
        :param image: an [N x C x H x W] Tensor of image inputs.
        :param tabular: an [N x F] Tensor of tabular inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param label: an [N] Tensor of labels, if class-conditional.
        :return: an image output of [N x C x H x W] Tensor, a tabular output of [N x F]
        """

        assert (label is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # Lists to store intermediate outputs for skip connections
        image_hs = []
        tabular_hs = []

        # Generate time embeddings
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # If class-conditional, add class label embedding
        if self.num_classes is not None:
            assert label.shape == (image.shape[0],)
            emb = emb + self.label_emb(label)

        # Ensure inputs are in the correct dtype
        image = image.type(self.dtype)
        tabular = tabular.type(self.dtype)

        # Encoder: Process through input blocks
        for m_id, module in enumerate(self.input_blocks):
            image, tabular = module(image, tabular, emb)
            image_hs.append(image)
            tabular_hs.append(tabular)

        # Middle blocks
        image, tabular = self.middle_blocks(image, tabular, emb)

        # Decoder: Process through output blocks, adding skip connections
        for m_id, module in enumerate(self.output_blocks):
            image = th.cat([image, image_hs.pop()], dim=1)
            tabular = th.cat([tabular, tabular_hs.pop()], dim=1)
            image, tabular = module(image, tabular, emb)

        # Final output layers for image and tabular data
        image = self.image_out(image)
        tabular = self.tabular_out(tabular)

        return image, tabular



if __name__ == '__main__':
    import time
    import torch as th
    import torch.nn.functional as F

    # Set device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Model configuration parameters
    model_channels = 192
    emb_channels = 128
    image_size = [3, 64, 64]  # Channels, Height, Width for image data
    tabular_size = [96, 96]  # Number of features in tabular data
    image_out_channels = 3
    tabular_out_channels = 96 # TODO: set as tabular_size since tabular is handled as [b, c, f]
    num_heads = 2
    num_res_blocks = 1
    cross_attention_resolutions = [4, 8, 16]
    cross_attention_window = [1, 1, 1]
    cross_attention_shift = False
    image_attention_resolutions = [2, 4, 8, 16]
    tabular_attention_resolutions = [2, 4, 8, 16]
    lr = 0.0001
    channel_mult = (1, 2, 3, 4)

    # Initialize the model
    model = MultimodalUNet(
        image_size=image_size,
        tabular_size=tabular_size,
        model_channels=model_channels,
        image_out_channels=image_out_channels,
        tabular_out_channels=tabular_out_channels,
        num_res_blocks=num_res_blocks,
        cross_attention_resolutions=cross_attention_resolutions,
        num_heads=num_heads,
        cross_attention_windows=cross_attention_window,
        cross_attention_shift=cross_attention_shift,
        image_attention_resolutions=image_attention_resolutions,
        tabular_attention_resolutions=tabular_attention_resolutions,
        use_scale_shift_norm=True,
        use_checkpoint=True
    ).to(device)

    #TODO:
    # Define the hook function

    layer_outputs = {}
    def hook_fn(module, input, output):
        layer_outputs[module] = output.shape if isinstance(output, th.Tensor) else [o.shape for o in output]

    # Register hooks for each layer in the model
    for name, layer in model.named_modules():
        layer.register_forward_hook(hook_fn)

    # Optimizer
    optim = th.optim.SGD(model.parameters(), lr=lr)

    # Training loop
    model.train()
    while True:

        # TODO
        shape_manager = ShapeManager()


        time_start = time.time()

        # Dummy data for image and tabular inputs
        image = th.randn([1, 3, 64, 64]).to(device)  # Batch size, Channels, Height, Width
        # tabular = th.randn([1, tabular_size]).to(device)  # Batch size, Features
        # tabular = tabular.unsqueeze(1).repeat(1, tabular_size, 1) # to handle tabular as 2d data
        tabular = th.randn([1, 96, 96]).to(device)

        time_index = th.tensor([1]).to(device)

        # Forward pass
        image_out, tabular_out = model(image, tabular, time_index)

        # Target data and loss
        image_target = th.randn_like(image_out)
        tabular_target = th.randn_like(tabular_out)
        loss = F.mse_loss(image_target, image_out) + F.mse_loss(tabular_target, tabular_out)

        # Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Logging
        print(f"loss: {loss.item()} time: {time.time() - time_start}")

        # TODO
        shape_manager.delete_file()

