import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import torch as th
import torch
from abc import abstractmethod

from multi_modal_diffusion.arch_utils import (conv_nd, avg_pool_nd, normalization, zero_module, count_flops_attn, checkpoint,
                        timestep_embedding)
from multi_modal_diffusion.fp16_util import (convert_module_to_f16, convert_module_to_f32)
from multi_modal_diffusion import logger


from torch import Tensor
def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

################################################################################################


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, image, tabular, emb):
        """
        Apply the module to `image` and `tabular` given `emb` timestep embeddings.
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
        return self.image_conv(image)

class TabularMLP(nn.Module):
    """
    MLP for tabular data processing.
    """
    def __init__(
            self,
            in_features,
            out_features,
            hidden_features=None,
            num_layers=2,
            activation=nn.SiLU(),
    ):
        super().__init__()
        layers = []
        hidden_features = hidden_features or max(in_features, out_features)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
            layers.append(activation)
        layers.append(nn.Linear(hidden_features, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

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

        # Tabular data processing layer: MLP
        self.tabular_mlp = TabularMLP(tabular_in_features, tabular_out_features)

    def forward(self, image, tabular):
        # Process image data
        image_out = self.image_conv(image)

        # Process tabular data
        tabular_out = self.tabular_mlp(tabular)

        return image_out, tabular_out


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution for images.

    Note: For tabular data, upsampling is not typically applicable.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.stride = 2

        # Define convolution based on dimensions if needed
        if use_conv:
            if dims == 2:  # For images
                self.conv = conv_nd(2, self.channels, self.out_channels, kernel_size=3)

    def forward(self, x):
        if self.dims == 2:
            x = F.interpolate(x, scale_factor=(self.stride, self.stride), mode="nearest")
            if self.use_conv:
                x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    For images, use the existing downsampling method.
    For tabular data, we may not need downsampling, but to align with the architecture,
    we can use identity or appropriate MLP layers if necessary.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if dims == 2:
            # Image downsampling
            if use_conv:
                self.op = conv_nd(2, self.channels, self.out_channels, kernel_size=3, stride=2, padding=1)
            else:
                self.op = avg_pool_nd(2, kernel_size=2, stride=2)
        else:
            # For tabular data, we can use an identity layer
            self.op = nn.Identity()

    def forward(self, x):
        if self.dims == 2:
            x = self.op(x)
        else:
            x = self.op(x)
        return x


class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class SingleModalAtten(nn.Module):
    """
    An attention block that allows positions to attend to each other.

    Adapted for image data (spatial attention) and tabular data (feature-wise attention).
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        is_tabular=False,
    ):
        super().__init__()
        self.channels = channels
        self.is_tabular = is_tabular

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

        # Define qkv and projection layers
        if is_tabular:
            # For tabular data, attention over features
            self.qkv = nn.Linear(channels, channels * 3)
            self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=self.num_heads)
            self.proj_out = zero_module(nn.Linear(channels, channels))
        else:
            # For image data, attention over spatial dimensions
            self.qkv = conv_nd(1, channels, channels * 3, kernel_size=1)
            self.attention = SingleModalQKVAttention(self.num_heads)
            self.proj_out = zero_module(conv_nd(1, channels, channels, kernel_size=1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        if self.is_tabular:
            # TODO: check methodology
            x = self.norm(x)
            qkv = self.qkv(x)  # [batch, features * 3]
            qkv = qkv.view(x.shape[0], 3, self.channels)  # [batch, 3, channels]
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)  # [batch, 1, channels]
            h, _ = self.attention(q, k, v)  # [batch, 1, channels]
            h = h.squeeze(1)  # [batch, channels]
            h = self.proj_out(h)
            return x + h
        else:
            # x: [batch, channels, sequence_length]
            b, c, length = x.shape
            x = self.norm(x)
            qkv = self.qkv(x)
            h = self.attention(qkv)
            h = self.proj_out(h)
            return x + h

class ResBlock(TimestepBlock):
    """
    A residual block adapted for image and tabular data that can optionally change the number of channels.
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
            ImageConv(channels, self.out_channels, kernel_size=3)
        )

        # Tabular processing layers
        self.tabular_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            TabularMLP(channels, self.out_channels)
        )

        # Upsampling and Downsampling
        self.updown = up or down
        if up:
            self.img_upd = Upsample(channels, use_conv, dims=2)
            self.img_upd_orig = Upsample(channels, use_conv, dims=2)  # For the original image input
            # For tabular data, upsampling is not applicable
            self.tab_upd = nn.Identity()
            self.tab_upd_orig = nn.Identity()
        elif down:
            self.img_upd = Downsample(channels, use_conv, dims=2)
            self.img_upd_orig = Downsample(channels, use_conv, dims=2)  # For the original image input
            # For tabular data, downsampling is not applicable
            self.tab_upd = nn.Identity()
            self.tab_upd_orig = nn.Identity()
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
            # zero_module(nn.Linear(self.out_channels, self.out_channels))
            nn.Linear(self.out_channels, self.out_channels)
        )

        # Skip connections
        if self.out_channels == channels:
            self.image_skip_connection = nn.Identity()
            self.tabular_skip_connection = nn.Identity()
        elif use_conv:
            self.image_skip_connection = ImageConv(channels, self.out_channels, kernel_size=3)
            self.tabular_skip_connection = nn.Linear(channels, self.out_channels)
        else:
            self.image_skip_connection = ImageConv(channels, self.out_channels, kernel_size=1)
            self.tabular_skip_connection = nn.Linear(channels, self.out_channels)

        # Attention blocks (optional)
        self.image_attention = image_attention
        self.tabular_attention = tabular_attention

        if self.image_attention:
            self.image_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, num_head_channels=-1,
                use_checkpoint=use_checkpoint, is_tabular=False
            )
        if self.tabular_attention:
            self.tabular_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, num_head_channels=-1,
                use_checkpoint=use_checkpoint, is_tabular=True
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

        if self.updown:
            # Processed features
            img_h = self.image_in_layers(image)
            img_h = self.img_upd(img_h)

            tab_h = self.tabular_in_layers(tabular)
            # No upsampling for tabular data
            tab_h = self.tab_upd(tab_h)

            # Original inputs transformed to match processed features
            image = self.img_upd_orig(image)
            # No upsampling for tabular data
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
            tab_emb_out = emb_out
            scale, shift = th.chunk(tab_emb_out, 2, dim=1)
            tab_h = tab_out_norm(tab_h) * (1 + scale) + shift
            tab_h = tab_out_rest(tab_h)
        else:
            # If no scale-shift normalization, directly add the embedding
            img_emb_out = emb_out[:, :, None, None]  # Broadcast across spatial dimensions for images
            img_h = img_h + img_emb_out
            img_h = self.image_out_layers(img_h)

            tab_emb_out = emb_out
            tab_h = tab_h + tab_emb_out
            tab_h = self.tabular_out_layers(tab_h)

        # Add skip connections
        image_out = self.image_skip_connection(image) + img_h
        tabular_out = self.tabular_skip_connection(tabular) + tab_h

        # Apply attention if enabled
        if self.image_attention:
            image_out = rearrange(image_out, "b c h w -> b c (h w)")  # Flatten spatial dimensions
            image_out = self.image_attention_block(image_out)
            image_out = rearrange(image_out, "b c (h w) -> b c h w", h=h, w=w)  # Reshape back

        if self.tabular_attention:
            tabular_out = self.tabular_attention_block(tabular_out)

        return image_out, tabular_out


# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention for cross-attention between image and tabular data.
#     """
#
#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads
#
#     def forward(self, qkv, image_len, tabular_len):
#         """
#         Apply QKV attention over concatenated image and tabular tokens.
#
#         :param qkv: A tensor of Qs, Ks, and Vs concatenated, [batch_size, 3 * n_heads * channels, total_len]
#         :param image_len: Number of tokens in the image portion.
#         :param tabular_len: Number of tokens in the tabular portion.
#         :return: Attention outputs for both image and tabular tokens.
#         """
#
#         bs, width, total_len = qkv.shape
#         assert width % (3 * self.n_heads) == 0, "Width must be divisible by 3 * n_heads"
#         ch = width // (3 * self.n_heads)
#
#         # Reshape and split Q, K, V
#         qkv = qkv.view(bs, self.n_heads, 3 * ch, total_len)
#         q, k, v = qkv.split(ch, dim=2)  # Each: [batch_size, n_heads, ch, total_len]
#         scale = 1 / math.sqrt(ch)
#
#         # Compute attention weights over the concatenated sequence
#         attn_weights = th.softmax(
#             th.einsum("bncl,bncs->bnls", q * scale, k * scale), dim=-1
#         )  # [batch_size, n_heads, total_len, total_len]
#
#         # Apply attention weights to values
#         attn_output = th.einsum("bnls,bncs->bncl", attn_weights, v)  # [batch_size, n_heads, ch, total_len]
#
#         # Split the output back into image and tabular parts
#         image_output = attn_output[..., :image_len].contiguous()
#         tabular_output = attn_output[..., image_len:].contiguous()
#
#         # Reshape back to original dimensions
#         image_output = image_output.view(bs, -1, image_len)  # [batch_size, n_heads * ch, image_len]
#         tabular_output = tabular_output.view(bs, -1, tabular_len)  # [batch_size, n_heads * ch, tabular_len]
#
#         return image_output, tabular_output
#
#
# class CrossAttentionBlock(nn.Module):
#     """
#     Cross-attention block for image and tabular data.
#     """
#
#     def __init__(
#             self,
#             channels,
#             num_heads=1,
#             num_head_channels=-1,
#             use_checkpoint=False,
#     ):
#         super().__init__()
#         self.channels = channels
#
#         # Set the number of heads
#         if num_head_channels == -1:
#             self.num_heads = num_heads
#         else:
#             assert (
#                     channels % num_head_channels == 0
#             ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
#             self.num_heads = channels // num_head_channels
#
#         self.use_checkpoint = use_checkpoint
#
#         # Normalization layers
#         self.img_norm = normalization(self.channels)
#         self.tab_norm = normalization(self.channels)
#
#         # QKV computation for image and tabular data
#         self.img_qkv = nn.Linear(self.channels, self.channels * 3)
#         self.tab_qkv = nn.Linear(self.channels, self.channels * 3)
#
#         # Attention
#         self.attention = QKVAttention(self.num_heads)
#
#         # Projection layers
#         self.img_proj_out = zero_module(nn.Linear(self.channels, self.channels))
#         self.tab_proj_out = zero_module(nn.Linear(self.channels, self.channels))
#
#     def forward(self, image, tabular):
#         return checkpoint(self._forward, (image, tabular), self.parameters(), self.use_checkpoint)
#
#     def _forward(self, image, tabular):
#         """
#         Forward method for cross-attention between image and tabular data.
#
#         :param image: Image tensor of shape [batch_size, channels, height, width]
#         :param tabular: Tabular tensor of shape [batch_size, channels]
#         :return: Updated image and tabular data with cross-attention applied.
#         """
#         b, c, h, w = image.shape  # Image dimensions
#         b_t, f = tabular.shape  # Tabular dimensions (features)
#
#         # Flatten spatial dimensions for image tokens
#         image_token = rearrange(image, "b c h w -> b (h w) c")  # [batch_size, seq_len, channels]
#         tabular_token = tabular.unsqueeze(1)  # [batch_size, 1, channels]
#
#         # Compute QKV for image and tabular data
#         img_qkv = self.img_qkv(self.img_norm(image_token))  # [batch_size, seq_len, 3 * channels]
#         tab_qkv = self.tab_qkv(self.tab_norm(tabular_token))  # [batch_size, 1, 3 * channels]
#
#         # Concatenate QKV tensors along the sequence length
#         qkv = torch.cat([img_qkv, tab_qkv], dim=1)  # [batch_size, seq_len + 1, 3 * channels]
#         qkv = qkv.transpose(1, 2)  # [batch_size, 3 * channels, seq_len + 1]
#
#         # Apply attention
#         total_len = image_token.shape[1] + tabular_token.shape[1]
#         image_len = image_token.shape[1]
#         tabular_len = tabular_token.shape[1]
#
#         img_attn, tab_attn = self.attention(qkv, image_len=image_len, tabular_len=tabular_len)
#
#         # Reshape back to original dimensions
#         image_h = img_attn.transpose(1, 2).view(b, c, h, w)  # [batch_size, channels, h, w]
#         image_h = self.img_proj_out(image_h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [batch_size, channels, h, w]
#         image_out = image + image_h
#
#         tabular_h = tab_attn.transpose(1, 2).squeeze(2)  # [batch_size, channels]
#         tabular_h = self.tab_proj_out(tabular_h)
#         tabular_out = tabular + tabular_h
#
#         return image_out, tabular_out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention for cross-attention between image and tabular data.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, image_len, tabular_len):
        """
        Apply QKV attention.

        :param qkv: A tensor of Qs, Ks, and Vs concatenated, [batch_size, 3 * n_heads * channels, length]
        :param image_len: Number of tokens in the image portion.
        :param tabular_len: Number of tokens in the tabular portion.
        :return: Attention outputs for both image and tabular tokens.
        """

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        # Split Q, K, V from concatenated input
        qkv = qkv.view(bs, self.n_heads, 3 * ch, length)
        q, k, v = qkv.split(ch, dim=2)  # Each has shape [batch_size, n_heads, ch, length]
        scale = 1 / math.sqrt(ch)

        # Separate Q, K, V for image and tabular parts
        img_q = q[..., :image_len]  # [bs, n_heads, ch, image_len]
        img_k = k[..., :image_len]
        img_v = v[..., :image_len]

        tab_q = q[..., image_len:]  # [bs, n_heads, ch, tabular_len]
        tab_k = k[..., image_len:]
        tab_v = v[..., image_len:]

        # Image queries attend to tabular keys/values
        img_weight = th.softmax(
            th.einsum("bncl,bncs->bnls", img_q * scale, tab_k * scale), dim=-1
        )  # [bs, n_heads, image_len, tabular_len]
        img_a = th.einsum("bnls,bncs->bncl", img_weight, tab_v)  # [bs, n_heads, ch, image_len]

        # Tabular queries attend to image keys/values
        tab_weight = th.softmax(
            th.einsum("bncl,bncs->bnls", tab_q * scale, img_k * scale), dim=-1
        )  # [bs, n_heads, tabular_len, image_len]
        tab_a = th.einsum("bnls,bncs->bncl", tab_weight, img_v)  # [bs, n_heads, ch, tabular_len]

        # Reshape back to original dimensions
        img_a = img_a.reshape(bs, -1, image_len)  # [bs, n_heads * ch, image_len]
        tab_a = tab_a.reshape(bs, -1, tabular_len)  # [bs, n_heads * ch, tabular_len]

        return img_a, tab_a

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

        self.use_checkpoint = use_checkpoint

        # Normalization layers
        self.img_norm = normalization(self.channels)
        self.tab_norm = normalization(self.channels)

        # QKV computation for image and tabular data
        # self.img_qkv = nn.Linear(self.channels, self.channels * 3)
        self.img_qkv = conv_nd(1, self.channels, self.channels * 3, kernel_size=1)
        self.tab_qkv = nn.Linear(self.channels, self.channels * 3)

        # Attention
        self.attention = QKVAttention(self.num_heads)

        # Projection layers for image and tabular
        # self.img_proj_out = zero_module(nn.Linear(self.channels, self.channels))
        # TODO: check for initialization instead of zero_module (eg. Kaiming initialization for convolutional layers)
        self.img_proj_out = zero_module(ImageConv(self.channels, self.channels, kernel_size=1))
        # self.tab_proj_out = zero_module(TabularMLP(self.channels, self.channels))
        self.tab_proj_out = TabularMLP(self.channels, self.channels)

    def forward(self, image, tabular):
        return checkpoint(self._forward, (image, tabular), self.parameters(), self.use_checkpoint)

    def _forward(self, image, tabular):
        """
        Forward method for cross-attention between image and tabular data.

        :param image: Image tensor of shape [batch_size, channels, height, width]
        :param tabular: Tabular tensor of shape [batch_size, channels]
        :return: Updated image and tabular data with cross-attention applied.
        """
        b, c, h, w = image.shape  # Image dimensions
        b_t, c_t = tabular.shape   # Tabular dimensions (channels)

        # Flatten spatial dimensions for image tokens
        image_token = rearrange(image, "b c h w -> b c (h w)")  # [batch_size, channels, seq_len]
        seq_len_image = image_token.shape[2]
        # tabular_token = tabular.unsqueeze(1)  # [batch_size, 1, channels]
        tabular_token = tabular

        # Normalize
        image_token = self.img_norm(image_token)
        tabular_token = self.tab_norm(tabular_token)

        # Compute QKV for image and tabular data
        img_qkv = self.img_qkv(image_token)  # [batch_size, seq_len, 3 * channels]
        tab_qkv = self.tab_qkv(tabular_token)  # [batch_size, 1, 3 * channels]

        # Concatenate along sequence length dimension
        # qkv = torch.cat([img_qkv.transpose(1, 2), tab_qkv.transpose(1, 2)], dim=2)  # [batch_size, 3 * channels, total_len]
        qkv = torch.cat([img_qkv, tab_qkv.unsqueeze(2)], dim=2)  # [batch_size, 3 * channels, seq_len + 1]

        # Apply cross-attention
        image_len = seq_len_image
        tabular_len = 1
        img_a, tab_a = self.attention(qkv, image_len=image_len, tabular_len=tabular_len)

        # Reshape back to original dimensions
        image_h = img_a.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)  # [batch_size, channels, h, w]
        image_h = self.img_proj_out(image_h)
        image_out = image + image_h

        tabular_h = tab_a.transpose(1, 2).squeeze(1)  # [batch_size, channels]
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
            image_attention_resolutions,
            tabular_attention_resolutions,
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
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self._feature_size = ch
        input_block_chans = [ch]

        # Initial input blocks
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(InitialBlock(
            self.image_size[0], self.tabular_size, image_out_channels=ch, tabular_out_features=ch
        ))])

        ds = 1

        # Build input blocks and cross attention layers
        for level, mult in enumerate(channel_mult):
            for block_id in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        image_attention=ds in self.image_attention_resolutions,
                        tabular_attention=ds in self.tabular_attention_resolutions,
                        num_heads=num_heads,
                    )
                ]

                ch = int(mult * model_channels)

                if ds in self.cross_attention_resolutions:
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
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
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        image_attention=ds in self.image_attention_resolutions,
                        tabular_attention=ds in self.tabular_attention_resolutions,
                        num_heads=num_heads,
                    )
                ]

                ch = int(model_channels * mult)
                if ds in self.cross_attention_resolutions:
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
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
            # zero_module(TabularMLP(ch, tabular_out_channels)),
            TabularMLP(ch, tabular_out_channels)
        )
        self.image_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(ImageConv(ch, image_out_channels, kernel_size=3)),
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
        self.image_out.apply(convert_module_to_f32)
        self.tabular_out.apply(convert_module_to_f32)

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
            # tabular = tabular + tabular_hs.pop()  # For tabular data, we sum skip connections
            tabular = th.cat([tabular, tabular_hs.pop()], dim=1)
            image, tabular = module(image, tabular, emb)

        # Final output layers for image and tabular data
        image = self.image_out(image)
        tabular = self.tabular_out(tabular)

        return image, tabular




if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from multi_modal_diffusion.scripts.mm_training import ImageTabularDataset
    import time

    # Set device
    device = th.device('cpu')  # Using CPU

    # Model configuration parameters
    model_channels = 192
    emb_channels = 128
    image_size = [3, 64, 64]  # Channels, Height, Width for image data
    tabular_size = 174          # Number of features in tabular data (1D tensor)
    image_out_channels = 3
    tabular_out_channels = 174  # Must match the tabular_size
    num_heads = 2
    num_res_blocks = 1
    cross_attention_resolutions = [4, 8, 16]
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
        image_attention_resolutions=image_attention_resolutions,
        tabular_attention_resolutions=tabular_attention_resolutions,
        use_scale_shift_norm=True,
        use_checkpoint=True
    ).to(device)

    # Optimizer
    optim = th.optim.SGD(model.parameters(), lr=lr)

    # Data loading parameters
    data_dir = r'D:\clearboxAI\NACC\extracted_dataset'  # Replace with the actual data directory
    batch_size = 1  # Adjust as needed
    num_workers = 0  # Number of subprocesses to use for data loading

    # Create dataset and data loader
    dataset = ImageTabularDataset(data_dir, image_size=(64, 64))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    model.train()
    while True:
        for batch in data_loader:
            # Record the start time
            time_start = time.time()

            # Extract image and tabular data, and move to device
            image = batch['image'].to(device)  # [batch_size, channels, height, width]
            tabular = batch['tabular'].to(device)  # [batch_size, features]

            # Define timesteps (using a dummy value of 1)
            timesteps = th.ones(image.size(0), dtype=th.long).to(device)

            # Forward pass
            image_out, tabular_out = model(image, tabular, timesteps)

            # Use the inputs as targets (autoencoder-like setup)
            image_target = image
            tabular_target = tabular

            # Compute loss
            loss = F.mse_loss(image_out, image_target) + F.mse_loss(tabular_out, tabular_target)

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Logging
            print(f"Loss: {loss.item():.6f} | Time: {time.time() - time_start:.4f} seconds")

