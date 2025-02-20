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
from multi_modal_diffusion.utils.custom_logger import (register_gradient_hooks, debug_print_stats, debug_forward)
from multi_modal_diffusion.configs.defaults import (debug_logger, single_attn_active, cross_attn_active)



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

    def __init__(self, *args, debug=False):
        super().__init__(*args)
        self.debug = debug
        if self.debug:
            # Optionally register gradient hooks for your entire sequential
            register_gradient_hooks(self, name_prefix="TimestepEmbedSequential", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, image, tabular, emb):
        if self.debug:
            debug_print_stats(image, "TimestepEmbedSequential - input image", debug_logger=debug_logger)
            debug_print_stats(tabular, "TimestepEmbedSequential - input tabular", debug_logger=debug_logger)
            debug_print_stats(emb, "TimestepEmbedSequential - time emb", debug_logger=debug_logger)

        for idx, layer in enumerate(self):
            if isinstance(layer, TimestepBlock):
                image, tabular = layer(image, tabular, emb)
            else:
                image, tabular = layer(image, tabular)

            if self.debug:
                debug_print_stats(image, f"TimestepEmbedSequential - layer {idx} output image", debug_logger=debug_logger)
                debug_print_stats(tabular, f"TimestepEmbedSequential - layer {idx} output tabular", debug_logger=debug_logger)

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
            debug=False
    ):
        super().__init__()
        self.debug = debug

        self.image_conv = conv_nd(
            2,  # Dimension for image data
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        if self.debug:
            register_gradient_hooks(self, name_prefix="ImageConv", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, image):
        if self.debug:
            debug_print_stats(image, "ImageConv - input", debug_logger=debug_logger)

        out = self.image_conv(image)

        if self.debug:
            debug_print_stats(out, "ImageConv - output", debug_logger=debug_logger)
        return out

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
            debug=False
    ):
        super().__init__()
        self.debug = debug

        layers = []
        hidden_features = hidden_features or max(in_features, out_features)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
            layers.append(activation)
        layers.append(nn.Linear(hidden_features, out_features))
        self.mlp = nn.Sequential(*layers)

        if self.debug:
            register_gradient_hooks(self, name_prefix="TabularMLP", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, x):
        if self.debug:
            debug_print_stats(x, "TabularMLP - input", debug_logger=debug_logger)

        out = self.mlp(x)

        if self.debug:
            debug_print_stats(out, "TabularMLP - output", debug_logger=debug_logger)
        return out

class InitialBlock(nn.Module):
    def __init__(
            self,
            image_in_channels,
            tabular_in_features,
            image_out_channels,
            tabular_out_features,
            kernel_size=3,
            debug=False
    ):
        super().__init__()
        self.debug = debug

        # Image processing layer: 2D convolution for images
        self.image_conv = ImageConv(image_in_channels, image_out_channels, kernel_size=kernel_size, debug=self.debug)

        # Tabular data processing layer: MLP
        self.tabular_mlp = TabularMLP(tabular_in_features, tabular_out_features, debug=self.debug)

        if self.debug:
            register_gradient_hooks(self, name_prefix="InitialBlock", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, image, tabular):

        if self.debug:
            debug_print_stats(image, "InitialBlock - input image", debug_logger=debug_logger)
            debug_print_stats(tabular, "InitialBlock - input tabular", debug_logger=debug_logger)

        # Process image data
        image_out = self.image_conv(image)

        # Process tabular data
        tabular_out = self.tabular_mlp(tabular)

        if self.debug:
            debug_print_stats(image_out, "InitialBlock - output image", debug_logger=debug_logger)
            debug_print_stats(tabular_out, "InitialBlock - output tabular", debug_logger=debug_logger)

        return image_out, tabular_out


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution for images.

    Note: For tabular data, upsampling is not typically applicable.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, debug=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.stride = 2
        self.debug = debug

        # Define convolution based on dimensions if needed
        if use_conv and dims == 2:  # For images
            self.conv = conv_nd(2, self.channels, self.out_channels, kernel_size=3)

        if self.debug:
            register_gradient_hooks(self, name_prefix="Upsample", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, x):
        if self.debug:
            debug_print_stats(x, "Upsample - input", debug_logger=debug_logger)

        if self.dims == 2:
            x = F.interpolate(x, scale_factor=(self.stride, self.stride), mode="nearest")
            if self.use_conv:
                x = self.conv(x)

        if self.debug:
            debug_print_stats(x, "Upsample - output", debug_logger=debug_logger)
        return x

class Downsample(nn.Module):
    """
    For images, use the existing downsampling method.
    For tabular data, we may not need downsampling, but to align with the architecture,
    we can use identity or appropriate MLP layers if necessary.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, debug=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.debug = debug

        if dims == 2:
            # Image downsampling
            if use_conv:
                self.op = conv_nd(2, self.channels, self.out_channels, kernel_size=3, stride=2, padding=1)
            else:
                self.op = avg_pool_nd(2, kernel_size=2, stride=2)
        else:
            # For tabular data, we can use an identity layer
            self.op = nn.Identity()

        if self.debug:
            register_gradient_hooks(self, name_prefix="Downsample", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, x):
        if self.debug:
            debug_print_stats(x, "Downsample - input", debug_logger=debug_logger)

        if self.dims == 2:
            x = self.op(x)
        else:
            x = self.op(x)

        if self.debug:
            debug_print_stats(x, "Downsample - output", debug_logger=debug_logger)
        return x


class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def __init__(self, n_heads, debug=False):
        super().__init__()
        self.n_heads = n_heads
        self.debug = debug

        if self.debug:
            register_gradient_hooks(self, name_prefix="SingleModalQKVAttention", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, qkv):
        """
        qkv: [batch, width, length]
        where width = n_heads * channels * 3  (since Q, K, V are concatenated)
        """

        if self.debug:
            debug_print_stats(qkv, "SingleModalQKVAttention - qkv (input)", debug_logger=debug_logger)

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0, f"width={width} is not divisible by 3*n_heads={3*self.n_heads}"
        ch = width // (3 * self.n_heads)

        # Separate Q, K, V
        q, k, v = qkv.chunk(3, dim=1)

        if self.debug:
            debug_print_stats(q, "Q shape", debug_logger=debug_logger)
            debug_print_stats(k, "K shape", debug_logger=debug_logger)
            debug_print_stats(v, "V shape", debug_logger=debug_logger)

        scale = 1 / math.sqrt(ch)

        # Compute attention weight
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )

        if self.debug:
            debug_print_stats(weight, "attention weight (before softmax)", debug_logger=debug_logger)

        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        if self.debug:
            debug_print_stats(weight, "attention weight (after softmax)", debug_logger=debug_logger)

        # Multiply by V => shape [batch*n_heads, ch, length]
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        out = a.reshape(bs, -1, length)  # [B, n_heads*ch, length]
        if self.debug:
            debug_print_stats(out, "SingleModalQKVAttention - output", debug_logger=debug_logger)

        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class SingleModalAtten(nn.Module):
    """
    An attention block that allows positions to attend to each other.
    Adapted for image data (spatial attention) and tabular data (feature-wise attention).
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, is_tabular=False, debug=False):
        super().__init__()
        self.channels = channels
        self.is_tabular = is_tabular
        self.debug = debug

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
            # 1) Each of the F features becomes a token, dimension=1. We embed to "channels" dimension.
            self.embed_in = nn.Linear(1, channels)
            # 2) Use nn.MultiheadAttention on [F, B, channels].
            self.embed_out = nn.Linear(channels, 1)
            # 3) Project back down to 1 dimension per feature.
            self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=self.num_heads)

        else:
            # For image data, attention over spatial dimensions
            self.qkv = conv_nd(1, channels, channels * 3, kernel_size=1)
            self.attention = SingleModalQKVAttention(self.num_heads, debug=self.debug)
            self.proj_out = zero_module(conv_nd(1, channels, channels, kernel_size=1))

        if self.debug:
            register_gradient_hooks(self, name_prefix="SingleModalAtten", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, x):
        """
        x is either:
        - [batch, channels, seq_len] (if is_tabular=False, 'image' flattened)
        - [batch, channels] (if is_tabular=True)
        """

        if self.debug:
            debug_print_stats(x, "SingleModalAtten - input (raw)", debug_logger=debug_logger)
            if self.is_tabular:
                assert x.ndim == 2, f"Expected [B, C] for tabular, got {x.shape}"
            else:
                assert x.ndim == 3, f"Expected [B, C, seq_len] for image, got {x.shape}"

        # We use checkpoint if needed
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    @debug_forward(debug_logger=debug_logger)
    def _forward(self, x):
        x_norm = self.norm(x)

        if self.debug:
            debug_print_stats(x_norm, "SingleModalAtten - after norm", debug_logger=debug_logger)

        if self.is_tabular:
            # x: [B, F]. Let F = #features => seq_len=F

            B, F = x.shape
            # (1) Expand each feature from 1D -> channels
            #     => [B, F, 1], then flatten to [B*F, 1], pass embed_in => [B*F, channels]
            x_expanded = x.unsqueeze(-1)  # [B, F, 1]
            x_flat = x_expanded.view(B * F, 1)  # [B*F, 1]
            emb_in = self.embed_in(x_flat)  # [B*F, channels]
            x_embed = emb_in.view(B, F, self.channels)  # [B, F, channels]

            # (2) We want MHA => shape [seq_len=F, batch=B, embed_dim=channels]
            x_mha_in = x_embed.permute(1, 0, 2)  # [F, B, channels]

            # (3) Self-attention (Q=K=V)
            #     out => [F, B, channels]
            out, _ = self.attention(x_mha_in, x_mha_in, x_mha_in)

            # (4) Reshape back => [B, F, channels]
            out = out.permute(1, 0, 2).contiguous()

            # (5) Project each token from [channels] -> 1
            out_flat = out.view(B * F, self.channels)  # [B*F, channels]
            out_proj = self.embed_out(out_flat)  # [B*F, 1]
            out_proj = out_proj.view(B, F)  # [B, F]

            # (6) Skip connection: original shape was [B, F]
            return x + out_proj

        else:
            # x_norm: [batch, channels, seq_len], qkv conv: [batch, channels*3, seq_len]
            qkv = self.qkv(x_norm)
            if self.debug:
                debug_print_stats(qkv, "SingleModalAtten (image) - qkv (raw)", debug_logger=debug_logger)

            # SingleModalQKVAttention expects [batch, width, length] where width = 3 * num_heads * ch_for_head
            h = self.attention(qkv)
            if self.debug:
                debug_print_stats(h, "SingleModalAtten (image) - attn output", debug_logger=debug_logger)

            # final projection
            h = self.proj_out(h)
            if self.debug:
                debug_print_stats(h, "SingleModalAtten (image) - proj_out", debug_logger=debug_logger)

            # Skip connection
            assert h.shape == x.shape, \
                f"Skip connection shape mismatch: x={x.shape}, h={h.shape}"
            return x + h

        # OLD
        # if self.is_tabular:
        #     # TODO: check methodology
        #     x = self.norm(x)
        #     qkv = self.qkv(x)  # [batch, features * 3]
        #     qkv = qkv.view(x.shape[0], 3, self.channels)  # [batch, 3, channels]
        #     q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        #     q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)  # [batch, 1, channels]
        #     h, _ = self.attention(q, k, v)  # [batch, 1, channels]
        #     h = h.squeeze(1)  # [batch, channels]
        #     h = self.proj_out(h)
        #     return x + h
        # else:
        #     # x: [batch, channels, sequence_length]
        #     b, c, length = x.shape
        #     x = self.norm(x)
        #     qkv = self.qkv(x)
        #     h = self.attention(qkv)
        #     h = self.proj_out(h)
        #     return x + h

class ResBlock(TimestepBlock):
    """
    A residual block adapted for image and tabular data that can optionally change the number of channels.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_scale_shift_norm=False,
                 use_checkpoint=False, up=False, down=False, use_conv=False, image_attention=False,
                 tabular_attention=False, num_heads=4, debug=False, attention_active=None):

        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.debug = debug
        self.attention_active = single_attn_active if not attention_active else attention_active

        if self.debug:
            register_gradient_hooks(self, name_prefix="ResBlock", debug_logger=debug_logger)

        # Image processing layers
        self.image_in_layers = nn.Sequential(
            # normalization(channels),
            nn.SiLU(),
            ImageConv(channels, self.out_channels, kernel_size=3, debug=self.debug)
        )

        # Tabular processing layers
        self.tabular_in_layers = nn.Sequential(
            # normalization(channels),
            nn.SiLU(),
            TabularMLP(channels, self.out_channels, debug=self.debug)
        )

        # Upsampling and Downsampling
        self.updown = up or down
        if up:
            self.img_upd = Upsample(channels, use_conv, dims=2, debug=self.debug)
            self.img_upd_orig = Upsample(channels, use_conv, dims=2, debug=self.debug)  # For the original image input
            # For tabular data, upsampling is not applicable
            self.tab_upd = nn.Identity()
            self.tab_upd_orig = nn.Identity()
        elif down:
            self.img_upd = Downsample(channels, use_conv, dims=2, debug=self.debug)
            self.img_upd_orig = Downsample(channels, use_conv, dims=2, debug=self.debug)  # For the original image input
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
        if self.debug:
            register_gradient_hooks(self.emb_layers, name_prefix="ResBlock.emb_layers", debug_logger=debug_logger)

        # Output layers for image and tabular data
        self.image_out_layers = nn.Sequential(
            # normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(ImageConv(self.out_channels, self.out_channels, kernel_size=1, debug=self.debug))
        )

        self.tabular_out_layers = nn.Sequential(
            # normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.out_channels, self.out_channels)
        )

        # Skip connections
        if self.out_channels == channels:
            self.image_skip_connection = nn.Identity()
            self.tabular_skip_connection = nn.Identity()
        elif use_conv:
            self.image_skip_connection = ImageConv(channels, self.out_channels, kernel_size=3, debug=self.debug)
            self.tabular_skip_connection = nn.Linear(channels, self.out_channels)
        else:
            self.image_skip_connection = ImageConv(channels, self.out_channels, kernel_size=1, debug=self.debug)
            self.tabular_skip_connection = nn.Linear(channels, self.out_channels)

        # Attention blocks (optional)
        self.image_attention = image_attention
        self.tabular_attention = tabular_attention

        if self.image_attention:
            self.image_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, num_head_channels=-1,
                use_checkpoint=use_checkpoint, is_tabular=False, debug=self.debug
            )
        if self.tabular_attention:
            self.tabular_attention_block = SingleModalAtten(
                channels=self.out_channels, num_heads=num_heads, num_head_channels=-1,
                use_checkpoint=use_checkpoint, is_tabular=True, debug=self.debug
            )

    @debug_forward(debug_logger=debug_logger)
    def forward(self, image, tabular, emb):
        """
        Apply the block to image and tabular data, conditioned on a timestep embedding.
        Applies gradient checkpointing around self._forward
        """

        if self.debug:
            debug_print_stats(image, "ResBlock - forward (image) [before checkpoint]", debug_logger=debug_logger)
            debug_print_stats(tabular, "ResBlock - forward (tabular) [before checkpoint]", debug_logger=debug_logger)

        return checkpoint(self._forward, (image, tabular, emb), self.parameters(), self.use_checkpoint)

    @debug_forward(debug_logger=debug_logger)
    def _forward(self, image, tabular, emb):
        """
        :param image:  [batch, channels, height, width]
        :param tabular:  [batch, features]
        :param emb: time-step embedding [batch, emb_channels]
        """

        if self.debug:
            debug_print_stats(image, "ResBlock - _forward (image in)", debug_logger=debug_logger)
            debug_print_stats(tabular, "ResBlock - _forward (tabular in)", debug_logger=debug_logger)
            debug_print_stats(emb, "ResBlock - _forward (emb in)", debug_logger=debug_logger)

        b, c, h, w = image.shape  # For image

        # 1. Up/Down sampling flow
        if self.updown:
            # Processed features
            img_h = self.image_in_layers(image)
            img_h = self.img_upd(img_h)

            tab_h = self.tabular_in_layers(tabular)
            tab_h = self.tab_upd(tab_h) # No upsampling for tabular data

            # Also transform original inputs so skip-connection matches
            image = self.img_upd_orig(image)
            tabular = self.tab_upd_orig(tabular) # No upsampling for tabular data
        else:
            img_h = self.image_in_layers(image)
            tab_h = self.tabular_in_layers(tabular)

        if self.debug:
            debug_print_stats(img_h, "ResBlock - img_h (after in_layers & up/down)", debug_logger=debug_logger)
            debug_print_stats(tab_h, "ResBlock - tab_h (after in_layers & up/down)", debug_logger=debug_logger)

        # 2. Timestep embedding
        emb_out = self.emb_layers(emb).type(image.dtype)
        if self.debug:
            debug_print_stats(emb_out, "ResBlock - emb_out (after emb_layers)", debug_logger=debug_logger)

        # 3. Scale-Shift Norm or direct add
        if self.use_scale_shift_norm:
            # Scale and shift for images
            img_out_norm, img_out_rest = self.image_out_layers[0], self.image_out_layers[1:]
            img_emb_out = emb_out[:, :, None, None]  # Reshape to broadcast across height and width
            scale, shift = th.chunk(img_emb_out, 2, dim=1)

            # Normalization, then scale/shift
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

        if self.debug:
            debug_print_stats(img_h, "ResBlock - img_h (after scale-shift / out_layers)", debug_logger=debug_logger)
            debug_print_stats(tab_h, "ResBlock - tab_h (after scale-shift / out_layers)", debug_logger=debug_logger)

        # 4. Skip connections
        skip_img = self.image_skip_connection(image)
        skip_tab = self.tabular_skip_connection(tabular)

        if self.debug:
            debug_print_stats(skip_img, "ResBlock - skip_img", debug_logger=debug_logger)
            debug_print_stats(skip_tab, "ResBlock - skip_tab", debug_logger=debug_logger)

        image_out = skip_img + img_h
        tabular_out = skip_tab + tab_h

        # image_out = img_h
        # tabular_out = tab_h

        if self.debug:
            debug_print_stats(image_out, "ResBlock - image_out (skip connected)", debug_logger=debug_logger)
            debug_print_stats(tabular_out, "ResBlock - tabular_out (skip connected)", debug_logger=debug_logger)

        # 5. (Optional) Single-modal attention if enabled & attention_active
        if self.image_attention and self.attention_active:
            # Flatten spatial dims from [B, C, H, W] -> [B, C, H*W]
            image_out = rearrange(image_out, "b c h w -> b c (h w)")
            if self.debug:
                debug_print_stats(image_out, "ResBlock - image_out (flattened for attention)", debug_logger=debug_logger)
            image_out = self.image_attention_block(image_out)
            # Reshape back
            image_out = rearrange(image_out, "b c (h w) -> b c h w", h=h, w=w)
            if self.debug:
                debug_print_stats(image_out, "ResBlock - image_out (after attention)", debug_logger=debug_logger)

        if self.tabular_attention and self.attention_active:
            # For tabular, shape is [B, C].  Direct pass:
            if self.debug:
                debug_print_stats(tabular_out, "ResBlock - tabular_out (before attention)", debug_logger=debug_logger)
            tabular_out = self.tabular_attention_block(tabular_out)
            if self.debug:
                debug_print_stats(tabular_out, "ResBlock - tabular_out (after attention)", debug_logger=debug_logger)

        return image_out, tabular_out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention for cross-attention between image and tabular data.
    """

    def __init__(self, n_heads, debug=False):
        super().__init__()
        self.n_heads = n_heads
        self.debug = debug

        if self.debug:
            register_gradient_hooks(self, name_prefix="QKVAttention", debug_logger=debug_logger)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, qkv, image_len, tabular_len):
        """
        Apply QKV attention.

        :param qkv: A tensor of Qs, Ks, and Vs concatenated, [batch_size, 3 * n_heads * channels, length]
        :param image_len: Number of tokens in the image portion.
        :param tabular_len: Number of tokens in the tabular portion.
        :return: Attention outputs for both image and tabular tokens.
        """

        B, width, total_len = qkv.shape
        assert total_len == (image_len + tabular_len)

        # Each head has 'head_dim' channels
        # width = 3 * n_heads * head_dim
        head_dim = width // (3 * self.n_heads)
        assert (3 * self.n_heads * head_dim) == width

        # [B, n_heads, 3*head_dim, total_len]
        qkv = qkv.view(B, self.n_heads, 3 * head_dim, total_len)
        q, k, v = qkv.split(head_dim, dim=2)  # each [B, n_heads, head_dim, total_len]

        # We use standard Transformer scaling => 1/sqrt(head_dim)
        scale = 1 / math.sqrt(head_dim)

        # Partition the sequence dimension into "image" vs "tab" tokens
        # IMAGE TOKENS
        img_q = q[..., :image_len]  # [B, n_heads, head_dim, image_len]
        img_k = k[..., :image_len]
        img_v = v[..., :image_len]

        # TAB TOKENS
        tab_q = q[..., image_len:]  # [B, n_heads, head_dim, tabular_len]
        tab_k = k[..., image_len:]
        tab_v = v[..., image_len:]

        ###
        # 1) IMAGE queries x TAB keys => shape [B, n_heads, image_len, tabular_len]
        ###
        img_weight = torch.einsum(
            "bncl,bncs->bnls", (img_q * scale), (tab_k * scale)
        )  # => [B, n_heads, image_len, tabular_len]
        img_weight = torch.softmax(img_weight.float(), dim=-1).type(img_weight.dtype)

        # multiply by tab V => [B, n_heads, head_dim, image_len]
        img_a = torch.einsum("bnls,bncs->bncl", img_weight, tab_v)

        ###
        # 2) TAB queries x IMAGE keys => shape [B, n_heads, tabular_len, image_len]
        ###
        tab_weight = torch.einsum(
            "bncl,bncs->bnls", (tab_q * scale), (img_k * scale)
        )
        tab_weight = torch.softmax(tab_weight.float(), dim=-1).type(tab_weight.dtype)

        # multiply by image V => [B, n_heads, head_dim, tabular_len]
        tab_a = torch.einsum("bnls,bncs->bncl", tab_weight, img_v)

        # reshape => [B, n_heads*head_dim, length]
        img_a = img_a.reshape(B, self.n_heads * head_dim, image_len)
        tab_a = tab_a.reshape(B, self.n_heads * head_dim, tabular_len)
        return img_a, tab_a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for image <-> tabular, with multi-token tab.
    The tabular input is shape [B, F] originally.
    We'll embed it into [B, c, F], so F is # of tokens, c is the channel dimension.
    We'll flatten the image to [B, c, H*W].
    Then we cat => shape [B, c, total_tokens].
    Then do QKV, run cross-attention, and split them back.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, debug=False):
        super().__init__()
        self.channels = channels
        self.debug = debug
        self.use_checkpoint = use_checkpoint

        # Determine #heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels) == 0, (
                f"channels {channels} not divisible by {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels

        # Normalization for the image and tabular tokens
        self.img_norm = normalization(channels)
        self.tab_norm = normalization(self.channels)

        # We'll do an embedding for tabular => from [B, F] => [B, c, F]
        self.tab_embed_in = nn.Linear(1, channels)
        # If you want norm for tab, define self.tab_norm if needed

        # QKV
        # We'll process the cat'd tokens with a 1D conv for image dimension:
        # shape [B, c, total_tokens] => [B, 3*c, total_tokens]
        self.qkv_conv = nn.Conv1d(in_channels=channels, out_channels=3 * channels, kernel_size=1)

        # The cross-attention
        self.attention = QKVAttention(self.num_heads, debug=self.debug)

        # Output projection for image
        self.img_proj_out = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        nn.init.zeros_(self.img_proj_out.weight)  # zero init for a "residual" style

        # Output projection for tab
        # We'll go from [B, c, F] -> [B, c, F], then eventually reduce to [B, F] if needed
        self.tab_proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.tab_proj_out.weight)

        # Final MLP to get back to [B, F] if you want to keep same shape as input
        self.tab_final_linear = nn.Linear(channels, 1)

    @debug_forward(debug_logger=debug_logger)
    def forward(self, image, tabular):

        if self.debug:
            debug_print_stats(image, "CrossAttentionBlock - input image", debug_logger=debug_logger)
            debug_print_stats(tabular, "CrossAttentionBlock - input tabular", debug_logger=debug_logger)

        return checkpoint(self._forward, (image, tabular), self.parameters(), self.use_checkpoint)

    @debug_forward(debug_logger=debug_logger)
    def _forward(self, image, tabular):
        """
        Forward method for cross-attention between image and tabular data.

        :param image: Image tensor of shape [batch_size, channels, height, width]
        :param tabular: Tabular tensor of shape [batch_size, channels]
        :return: Updated image and tabular data with cross-attention applied.
        """

        B, c, H, W = image.shape
        # 1) Flatten image => [B, c, H*W]
        img_tokens = rearrange(image, "b c h w -> b c (h w)")
        seq_img = img_tokens.shape[2]

        # Normalize image tokens
        img_tokens = self.img_norm(img_tokens)  # shape [B, c, seq_img]

        # 2) Convert tabular => multi-token
        # B, F = tabular shape
        B2, F = tabular.shape
        assert B2 == B, f"Inconsistent batch size: {B2} vs {B}"

        # Flatten each feature => [B*F, 1], embed => [B*F, c], reshape => [B, c, F]
        tab_flat = tabular.view(B * F, 1)
        tab_emb_in = self.tab_embed_in(tab_flat)  # => [B*F, c]
        tab_tokens = tab_emb_in.view(B, c, F)  # => [B, c, F]

        # Possibly apply tab_norm if needed
        # tab_tokens = self.tab_norm(tab_tokens)

        # 3) Concatenate image & tab along seq dimension
        cat_tokens = torch.cat([img_tokens, tab_tokens], dim=2)
        total_len = seq_img + F

        # 4) QKV => [B, 3*c, total_len]
        qkv = self.qkv_conv(cat_tokens)

        # 5) Cross attention
        img_a, tab_a = self.attention(qkv, image_len=seq_img, tabular_len=F)
        # img_a => [B, c, seq_img]
        # tab_a => [B, c, F]

        # 6) Reshape image back => [B, c, H, W], proj, skip
        img_a_2d = rearrange(img_a, "b c (h w) -> b c h w", h=H, w=W)
        img_proj = self.img_proj_out(img_a_2d)
        image_out = image + img_proj

        # 7) For tab => [B, c, F], proj, skip
        tab_proj = self.tab_proj_out(tab_a)  # => [B, c, F]
        tab_combined = tab_tokens + tab_proj

        # 8) Final => map c->1 per token => shape [B, F]
        tab_flat2 = rearrange(tab_combined, "b c f -> (b f) c")
        tab_out1 = self.tab_final_linear(tab_flat2)  # => [B*F, 1]
        tab_out1 = tab_out1.view(B, F)

        # skip connection to original tab?
        tabular_out = tabular + tab_out1

        return image_out, tabular_out


class MultimodalUNet(nn.Module):
    """
    The full coupled-UNet model with attention and timestep embedding, adapted for image and tabular data.
    """

    def __init__(self, image_size, tabular_size, model_channels, image_out_channels, tabular_out_channels,
                 num_res_blocks, cross_attention_resolutions, image_attention_resolutions,
                 tabular_attention_resolutions, dropout=0, channel_mult=(1, 2, 3, 4), num_classes=None,
                 use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
                 use_scale_shift_norm=False, resblock_updown=True, debug=False, ):
        super().__init__()
        self.debug = debug

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
        self.debug = debug

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

        if self.debug:
            register_gradient_hooks(self, name_prefix="MultimodalUNet", debug_logger=debug_logger)

        # Initial input blocks
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(
                InitialBlock(self.image_size[0], self.tabular_size, image_out_channels=ch,
                             tabular_out_features=ch, debug=self.debug),
                debug=self.debug)
            ])

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
                        debug=self.debug,
                    )
                ]

                ch = int(mult * model_channels)

                if ds in self.cross_attention_resolutions and cross_attn_active:
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            debug=self.debug,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers, debug=self.debug,))
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
                            debug=self.debug,
                        ),
                        debug=self.debug
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
                debug=self.debug,
            ),
            *(
                [
                    CrossAttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        debug=self.debug,
                    )
                ] if cross_attn_active else []
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
                debug=self.debug,
            ),
            debug=self.debug
        )
        self._feature_size += ch

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_id in range(num_res_blocks + 1):
                # ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch, #ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        image_attention=ds in self.image_attention_resolutions,
                        tabular_attention=ds in self.tabular_attention_resolutions,
                        num_heads=num_heads,
                        debug=self.debug,
                    )
                ]

                ch = int(model_channels * mult)
                if ds in self.cross_attention_resolutions and cross_attn_active:
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            debug=self.debug,
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
                                debug=self.debug,
                            )
                        )
                        ds //= 2

                self._feature_size += ch
                self.output_blocks.append(TimestepEmbedSequential(*layers, debug=self.debug,))

        # Output projections
        self.tabular_out = nn.Sequential(
            # normalization(ch),
            nn.SiLU(),
            TabularMLP(ch, tabular_out_channels, debug=self.debug,)
        )
        self.image_out = nn.Sequential(
            # normalization(ch),
            nn.SiLU(),
            zero_module(ImageConv(ch, image_out_channels, kernel_size=3, debug=self.debug)),
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

    @debug_forward(debug_logger=debug_logger)
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

        if self.debug:
            debug_print_stats(image, "MultimodalUNet.forward - image (input)", debug_logger=debug_logger)
            debug_print_stats(tabular, "MultimodalUNet.forward - tabular (input)", debug_logger=debug_logger)

        # Lists to store intermediate outputs for skip connections
        image_hs = []
        tabular_hs = []

        # Generate time embeddings
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # If class-conditional, add class label embedding
        if self.num_classes is not None:
            assert label.shape == (image.shape[0],)
            emb = emb + self.label_emb(label)

        if self.debug:
            debug_print_stats(emb, f"MultimodalUNet.forward - emb (time + num_classes:{self.num_classes})", debug_logger=debug_logger)

        # Ensure inputs are in the correct dtype
        image = image.type(self.dtype)
        tabular = tabular.type(self.dtype)

        # Encoder: Process through input blocks
        for m_id, module in enumerate(self.input_blocks):
            image, tabular = module(image, tabular, emb)
            image_hs.append(image)
            tabular_hs.append(tabular)

            if self.debug:
                debug_print_stats(image, f"MultimodalUNet - input_block {m_id} output (image)", debug_logger=debug_logger)
                debug_print_stats(tabular, f"MultimodalUNet - input_block {m_id} output (tabular)", debug_logger=debug_logger)

        # Middle blocks
        image, tabular = self.middle_blocks(image, tabular, emb)
        if self.debug:
            debug_print_stats(image, "MultimodalUNet - middle_blocks output (image)", debug_logger=debug_logger)
            debug_print_stats(tabular, "MultimodalUNet - middle_blocks output (tabular)", debug_logger=debug_logger)

        # Decoder: Process through output blocks, adding skip connections
        for m_id, module in enumerate(self.output_blocks):
            # Cat skip connection for image
            # skip_img = image_hs.pop()
            # image = th.cat([image, skip_img], dim=1)
            if self.debug:
                debug_print_stats(image, f"MultimodalUNet - image after cat skip {m_id}", debug_logger=debug_logger)

            # Cat skip connection for tabular
            # skip_tab = tabular_hs.pop()
            # # The code uses cat rather than sum for tabular skip
            # # => [N, feats + feats], #TODO: watch out for shape alignment
            # tabular = th.cat([tabular, skip_tab], dim=1)
            if self.debug:
                debug_print_stats(tabular, f"MultimodalUNet - tabular after cat skip {m_id}", debug_logger=debug_logger)

            image, tabular = module(image, tabular, emb)
            if self.debug:
                debug_print_stats(image, f"MultimodalUNet - output_block {m_id} out (image)", debug_logger=debug_logger)
                debug_print_stats(tabular, f"MultimodalUNet - output_block {m_id} out (tabular)", debug_logger=debug_logger)

        # Final output layers for image and tabular data
        image = self.image_out(image)
        tabular = self.tabular_out(tabular)
        if self.debug:
            debug_print_stats(image, "MultimodalUNet.forward - image_out (final)", debug_logger=debug_logger)
            debug_print_stats(tabular, "MultimodalUNet.forward - tabular_out (final)", debug_logger=debug_logger)

        return image, tabular




# if __name__ == '__main__':
#
#     from torch.utils.data import DataLoader
#     from multi_modal_diffusion.scripts.mm_training import ImageTabularDataset
#     import time
#
#     # Set device
#     device = th.device('cpu')  # Using CPU
#
#     # Model configuration parameters
#     model_channels = 192
#     emb_channels = 128
#     image_size = [3, 64, 64]  # Channels, Height, Width for image data
#     tabular_size = 174          # Number of features in tabular data (1D tensor)
#     image_out_channels = 3
#     tabular_out_channels = 174  # Must match the tabular_size
#     num_heads = 2
#     num_res_blocks = 1
#     cross_attention_resolutions = [4, 8, 16]
#     image_attention_resolutions = [2, 4, 8, 16]
#     tabular_attention_resolutions = [2, 4, 8, 16]
#     lr = 0.0001
#     channel_mult = (1, 2, 3, 4)
#
#     # Initialize the model
#     model = MultimodalUNet(
#         image_size=image_size,
#         tabular_size=tabular_size,
#         model_channels=model_channels,
#         image_out_channels=image_out_channels,
#         tabular_out_channels=tabular_out_channels,
#         num_res_blocks=num_res_blocks,
#         cross_attention_resolutions=cross_attention_resolutions,
#         num_heads=num_heads,
#         image_attention_resolutions=image_attention_resolutions,
#         tabular_attention_resolutions=tabular_attention_resolutions,
#         use_scale_shift_norm=True,
#         use_checkpoint=True
#     ).to(device)
#
#     # Optimizer
#     optim = th.optim.SGD(model.parameters(), lr=lr)
#
#     # Data loading parameters
#     data_dir = r'D:\clearboxAI\NACC\extracted_dataset'  # Replace with the actual data directory
#     batch_size = 1  # Adjust as needed
#     num_workers = 0  # Number of subprocesses to use for data loading
#
#     # Create dataset and data loader
#     dataset = ImageTabularDataset(data_dir, image_size=(64, 64))
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#
#     # Training loop
#     model.train()
#     while True:
#         for batch in data_loader:
#             # Record the start time
#             time_start = time.time()
#
#             # Extract image and tabular data, and move to device
#             image = batch['image'].to(device)  # [batch_size, channels, height, width]
#             tabular = batch['tabular'].to(device)  # [batch_size, features]
#
#             # Define timesteps (using a dummy value of 1)
#             timesteps = th.ones(image.size(0), dtype=th.long).to(device)
#
#             # Forward pass
#             image_out, tabular_out = model(image, tabular, timesteps)
#
#             # Use the inputs as targets (autoencoder-like setup)
#             image_target = image
#             tabular_target = tabular
#
#             # Compute loss
#             loss = F.mse_loss(image_out, image_target) + F.mse_loss(tabular_out, tabular_target)
#
#             # Backpropagation
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#
#             # Logging
#             print(f"Loss: {loss.item():.6f} | Time: {time.time() - time_start:.4f} seconds")


if __name__ == '__main__':

    import time
    import torch as th
    import torch.nn.functional as F

    # Set device
    device = th.device("cuda:0") if th.cuda.is_available() else th.device('cpu')

    # Model configuration parameters
    model_channels = 192
    emb_channels = 128
    image_size = [4, 64, 64]  # Channels, Height, Width for image data
    tabular_size = 174          # Number of features in tabular data (1D tensor)
    image_out_channels = 4
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

    # Training loop
    model.train()
    while True:
        # Generate random image and tabular data
        image = th.randn(1, 4, 64, 64).to(device)  # Random image of shape [1, 4, 64, 64]
        tabular = th.randn(1, tabular_size).to(device)  # Random tabular data of shape [1, tabular_size]

        # Record the start time
        time_start = time.time()

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

