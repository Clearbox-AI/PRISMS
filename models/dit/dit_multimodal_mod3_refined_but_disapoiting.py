import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections.abc import Iterable
from itertools import repeat
from typing import Optional, Tuple, Dict, Union, List, Any

from omegaconf import DictConfig
from utils.configurations import apply_overrides


def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Creates a normalization layer of the given type.
    Currently supports only "layernorm" or "np_layernorm".
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN style shift & scale:
      out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Sinusoidal + MLP embed for scalar timesteps.
    """
    def __init__(self, hidden_size, act_layer=nn.GELU, frequency_embedding_size=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            act_layer(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device
            ) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freq_emb = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(freq_emb)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


################################################################################
# FeedForward, FeedForwardECMoe
################################################################################

class FeedForward(nn.Module):
    """
    2-lin style with SILU gating (microdiffusion approach).
    """
    def __init__(self, dim, hidden_dim, multiple_of=256, use_bias=True):
        super().__init__()
        # Similar to the original code
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=init_std)


class FeedForwardECMoe(nn.Module):
    """
    Expert-Choice style MoE feed-forward.
    """
    def __init__(self, num_experts, expert_capacity, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Parameter(torch.ones(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, hidden_dim, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Quick example capacity logic (as in your code):
        tokens_per_expert = int(self.expert_capacity * T / self.num_experts)

        scores = self.gate(x)  # (B, T, E)
        probs = F.softmax(scores, dim=-1)
        g, m = torch.topk(probs.permute(0,2,1), tokens_per_expert, dim=-1)
        p = F.one_hot(m, num_classes=T).float()  # (B, E, tokens_per_expert, T)

        xin = torch.einsum('bekt,btd->bekd', p, x)
        h = torch.einsum('bekd,edh->bekh', xin, self.w1)
        h = self.gelu(h)
        h = torch.einsum('bekh,ehd->bekd', h, self.w2)

        out = g.unsqueeze(dim=-1)*h
        out = torch.einsum('bekt,bekd->btd', p, out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


################################################################################
# CrossAttention, SelfAttention
################################################################################

class CrossAttention(nn.Module):
    """
    One-way cross attention: x attends to cond.
    """
    def __init__(self, dim, num_heads, qkv_bias=True, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        assert hidden_dim % num_heads == 0
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, 2*hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T_x, dim) attends to cond: (B, T_cond, dim)
        B, T_x, _ = x.shape
        T_cond = cond.shape[1]

        q = self.q_linear(x)      # (B, T_x, hidden_dim)
        kv = self.kv_linear(cond) # (B, T_cond, 2*hidden_dim)
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, T_x, self.num_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T_cond, self.num_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T_cond, self.num_heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, num_heads, T_x, head_dim)
        out = out.transpose(1,2).reshape(B, T_x, self.hidden_dim)
        out = self.proj(out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.q_linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.kv_linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    """
    Standard multi-head self-attention.
    """
    def __init__(self, dim, num_heads, qkv_bias=True, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        assert hidden_dim % num_heads == 0
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(dim, 3*hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3*hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, T, self.hidden_dim)
        out = self.proj(out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


################################################################################
# Patch & Tab embeddings
################################################################################


class PatchEmbed(nn.Module):
    """
    Simple image patchify => flatten => (B, T, dim).
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.num_patches = (img_size // patch_size)*(img_size // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1,2)  # => (B, T, embed_dim)
        return x


class TabularProjection(nn.Module):
    """
    Group columns into fewer tokens => each group => MLP => single token
    """
    def __init__(self, in_features: int, hidden_size: int, groups: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups

        self.group_mlps = nn.ModuleList()
        group_sizes = self._get_group_sizes(in_features, groups)
        start = 0
        for gsize in group_sizes:
            mlp = nn.Sequential(
                nn.Linear(gsize, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.group_mlps.append(mlp)
            start += gsize

    def forward(self, tabular_input: torch.Tensor) -> torch.Tensor:
        B, C = tabular_input.shape
        group_sizes = self._get_group_sizes(C, self.groups)

        outputs = []
        start = 0
        for i, gsize in enumerate(group_sizes):
            mlp = self.group_mlps[i]
            subset = tabular_input[:, start:start+gsize]  # (B, gsize)
            out = mlp(subset)                             # (B, hidden_size)
            outputs.append(out.unsqueeze(1))
            start += gsize

        return torch.cat(outputs, dim=1)  # (B, groups, hidden_size)

    @staticmethod
    def _get_group_sizes(num_cols: int, groups: int) -> List[int]:
        base = num_cols // groups
        remainder = num_cols % groups
        sizes = []
        for i in range(groups):
            size = base + (1 if i < remainder else 0)
            sizes.append(size)
        return sizes


###############################################################################
# Block definitions for a "bidirectional" design
###############################################################################

class ImageBlock(nn.Module):
    """
    Image-only block:
      Self-attn + MLP (optionally MoE) + AdaLN from time embed
    """
    def __init__(
        self,
        dim, head_dim, mlp_ratio, qkv_ratio,
        multiple_of, time_emb_dim, norm_eps, use_bias,
        # MoE toggles
        moe=False, num_experts=8, expert_capacity=1.0,
        block_idx=0, total_blocks=1, depth_init=True
    ):
        super().__init__()
        qkv_hidden_dim = int(dim * qkv_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)

        # Only the image MLP can be MoE:
        if moe:
            self.mlp = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # gating: 6×dim
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim)
        )

        if depth_init:
            self.weight_init_std = 0.02 / (2*(block_idx+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*total_blocks)**0.5

    def forward(self, x_img: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Self-Attn
        x_ln = modulate(self.norm1(x_img), shift_msa, scale_msa)
        x_img = x_img + gate_msa.unsqueeze(1) * self.attn(x_ln)

        # MLP
        x_ln2 = modulate(self.norm2(x_img), shift_mlp, scale_mlp)
        x_img = x_img + gate_mlp.unsqueeze(1) * self.mlp(x_ln2)

        return x_img

    def custom_init(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)


class TabBlock(nn.Module):
    """
    Tab-only block:
      Self-attn + MLP (NO MoE for tab side) + AdaLN from time embed
    """
    def __init__(
        self,
        dim, head_dim, mlp_ratio, qkv_ratio,
        multiple_of, time_emb_dim, norm_eps, use_bias,
        block_idx=0, total_blocks=1, depth_init=True
    ):
        super().__init__()
        qkv_hidden_dim = int(dim * qkv_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim)
        )

        if depth_init:
            self.weight_init_std = 0.02 / (2*(block_idx+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*total_blocks)**0.5

    def forward(self, x_tab: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        x_ln = modulate(self.norm1(x_tab), shift_msa, scale_msa)
        x_tab = x_tab + gate_msa.unsqueeze(1) * self.attn(x_ln)

        x_ln2 = modulate(self.norm2(x_tab), shift_mlp, scale_mlp)
        x_tab = x_tab + gate_mlp.unsqueeze(1) * self.mlp(x_ln2)
        return x_tab

    def custom_init(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)


class BiCrossBlock(nn.Module):
    """
    Bidirectional cross-attention:
      - x_img attends to x_tab
      - x_tab attends to x_img
      - MLP on each side
      - MoE possible only on image side
    """
    def __init__(
        self,
        dim, head_dim, qkv_ratio, mlp_ratio,
        multiple_of, time_emb_dim, norm_eps, use_bias,
        # MoE toggles
        moe_img=False, num_experts=8, expert_capacity=1.0,
        # NO MoE on tab side
        block_idx=0, total_blocks=1, depth_init=True
    ):
        super().__init__()
        qkv_hidden_dim = int(dim * qkv_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # cross attn
        self.norm_img = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_img = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )
        self.norm_tab = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_tab = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        # MLP for image (may be MoE)
        if moe_img:
            self.mlp_img = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp_img = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # MLP for tab (no MoE)
        self.mlp_tab = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        self.norm_img2 = create_norm('layernorm', dim, eps=norm_eps)
        self.norm_tab2 = create_norm('layernorm', dim, eps=norm_eps)

        # gating: 4 sets => 12×dim
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 12*dim)
        )

        if depth_init:
            self.weight_init_std = 0.02 / (2*(block_idx+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*total_blocks)**0.5

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor, t_emb: torch.Tensor):
        chunked = self.adaLN_modulation(t_emb).chunk(12, dim=1)
        (shift_xi, scale_xi, gate_xi, shift_mi, scale_mi, gate_mi,
         shift_xt, scale_xt, gate_xt, shift_mt, scale_mt, gate_mt) = chunked

        # image attends to tab
        x_ln_img = modulate(self.norm_img(x_img), shift_xi, scale_xi)
        x_img = x_img + gate_xi.unsqueeze(1) * self.cross_attn_img(x_ln_img, x_tab)

        # tab attends to image
        x_ln_tab = modulate(self.norm_tab(x_tab), shift_xt, scale_xt)
        x_tab = x_tab + gate_xt.unsqueeze(1) * self.cross_attn_tab(x_ln_tab, x_img)

        # image MLP
        x_ln_img2 = modulate(self.norm_img2(x_img), shift_mi, scale_mi)
        x_img = x_img + gate_mi.unsqueeze(1) * self.mlp_img(x_ln_img2)

        # tab MLP
        x_ln_tab2 = modulate(self.norm_tab2(x_tab), shift_mt, scale_mt)
        x_tab = x_tab + gate_mt.unsqueeze(1) * self.mlp_tab(x_ln_tab2)

        return x_img, x_tab

    def custom_init(self):
        self.norm_img.reset_parameters()
        self.norm_tab.reset_parameters()
        self.norm_img2.reset_parameters()
        self.norm_tab2.reset_parameters()

        self.cross_attn_img.custom_init(self.weight_init_std)
        self.cross_attn_tab.custom_init(self.weight_init_std)
        self.mlp_img.custom_init(self.weight_init_std)
        self.mlp_tab.custom_init(self.weight_init_std)


################################################################################
# Final heads for image & tab
################################################################################

class FinalImageLayer(nn.Module):
    """
    Final image generation head:
      AdaLN => linear => reshape => (B, C, H, W)
    """
    def __init__(self, in_dim, time_emb_dim, patch_size, out_chans, act_layer, norm_layer):
        super().__init__()
        self.norm = norm_layer
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2*in_dim)  # shift & scale only
        )
        self.linear = nn.Linear(in_dim, patch_size*patch_size*out_chans)
        self.patch_size = patch_size
        self.out_chans = out_chans

    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)  # (B, T, p^2*out_chans)
        return x


class FinalTabLayer(nn.Module):
    """
    Final tab reconstruction:
      LN => MLP from pooled tokens => (B, out_dim)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_tab: torch.Tensor):
        # x_tab: (B, T_tab, in_dim) => pool => project
        x_pooled = x_tab.mean(dim=1)
        x_pooled = self.norm(x_pooled)
        return self.linear(x_pooled)


################################################################################
# A "2‐way" MultiModal DiT
################################################################################
# class MultiModalDiT(nn.Module):
class MultiModalDiT(nn.Module):
    """
    Final “definitive” design:

      - MoE only for the image MLP (ImageBlock or BiCrossBlock's image side).
      - MoE every "moe_frequency" layers (param).
      - The block schedule:
          ~15% "image_only",
          ~10% "parallel",
          ~25–80% alternate parallel/cross,
          ~80–95% parallel,
          ~95–100% cross
      - CFG via zeroing out both image tokens + tab tokens for unconditional pass.
      - Tab is never None in forward().
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=512,
        depth=12,
        head_dim=64,
        multiple_of=256,
        qkv_ratio=1.0,
        mlp_ratio=4.0,
        norm_eps=1e-6,
        use_bias=True,
        depth_init=True,
        # Tab
        num_tab_columns=10,
        tab_groups=3,
        out_table_features=10,
        # MoE
        moe_frequency=3,    # every N layers on the image side
        num_experts=8,
        expert_capacity=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.depth = depth
        self.head_dim = head_dim
        self.multiple_of = multiple_of
        self.qkv_ratio =qkv_ratio
        self.mlp_ratio =mlp_ratio
        self.norm_eps = norm_eps
        self.use_bias = use_bias
        self.depth_init = depth_init
        self.tab_groups = tab_groups
        self.out_table_features = out_table_features
        self.moe_frequency = moe_frequency
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        # Patchify image
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.num_patches = self.x_embedder.num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels

        # Tab
        self.num_tab_columns = num_tab_columns
        self.tab_proj = TabularProjection(num_tab_columns, dim, tab_groups)

        # Time embed
        self.t_embedder = TimestepEmbedder(hidden_size=dim)

        # Positional embedding for images
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, dim))

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_type = self._which_phase_block(i)

            # Decide if image side should use MoE in this block
            use_moe_img = ((i % moe_frequency) == (moe_frequency - 1))

            if block_type == 'image_only':
                blk = ImageBlock(
                    dim, head_dim, mlp_ratio, qkv_ratio,
                    multiple_of, time_emb_dim=dim,
                    norm_eps=norm_eps, use_bias=use_bias,
                    moe=use_moe_img, num_experts=num_experts,
                    expert_capacity=expert_capacity,
                    block_idx=i, total_blocks=depth, depth_init=depth_init
                )

            elif block_type == 'parallel':
                # parallel => image sub-block, tab sub-block
                blk = nn.ModuleDict({
                    'img': ImageBlock(
                        dim, head_dim, mlp_ratio, qkv_ratio,
                        multiple_of, time_emb_dim=dim,
                        norm_eps=norm_eps, use_bias=use_bias,
                        moe=use_moe_img, num_experts=num_experts,
                        expert_capacity=expert_capacity,
                        block_idx=i, total_blocks=depth, depth_init=depth_init
                    ),
                    'tab': TabBlock(
                        dim, head_dim, mlp_ratio, qkv_ratio,
                        multiple_of, time_emb_dim=dim,
                        norm_eps=norm_eps, use_bias=use_bias,
                        block_idx=i, total_blocks=depth, depth_init=depth_init
                    )
                })

            else:  # 'cross'
                blk = BiCrossBlock(
                    dim, head_dim, qkv_ratio, mlp_ratio,
                    multiple_of, time_emb_dim=dim,
                    norm_eps=norm_eps, use_bias=use_bias,
                    moe_img=use_moe_img, num_experts=num_experts,
                    expert_capacity=expert_capacity,
                    block_idx=i, total_blocks=depth, depth_init=depth_init
                )

            self.blocks.append(blk)

        # Final heads
        self.final_img = FinalImageLayer(
            in_dim=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_chans=self.out_channels,
            act_layer=nn.GELU,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps)
        )
        self.final_tab = FinalTabLayer(in_dim=dim, out_dim=out_table_features)

        self.initialize_weights()

    def _which_phase_block(self, idx: int) -> str:
        d = self.depth
        p1 = int(0.15 * d + 0.5)       # ~15%
        p2 = p1 + int(0.10 * d + 0.5)  # ~25%
        p3 = int(0.80 * d + 0.5)       # ~80%
        p4 = int(0.95 * d + 0.5)       # ~95%

        if idx < p1:
            return 'image_only'
        elif idx < p2:
            return 'parallel'
        elif idx < p3:
            # alternate parallel & cross, e.g. every 3rd block is cross
            offset = idx - p2
            if (offset % 3) == 2:
                return 'cross'
            else:
                return 'parallel'
        elif idx < p4:
            return 'parallel'
        else:
            return 'cross'

    def initialize_weights(self):
        def zero_bias(m):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                zero_bias(module)

        self.apply(_basic_init)

        # sin-cos pos embed
        side = int(self.num_patches**0.5)
        pe = self.get_2d_sincos_pe(side, self.dim)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # custom init for each block
        for blk in self.blocks:
            if isinstance(blk, nn.ModuleDict):
                blk['img'].custom_init()
                blk['tab'].custom_init()
            else:
                blk.custom_init()

        # final_img init
        nn.init.constant_(self.final_img.linear.weight, 0)
        nn.init.constant_(self.final_img.adaLN_modulation[-1].weight, 0)

    @staticmethod
    def get_2d_sincos_pe(grid_size, embed_dim):
        def get_1d_sin_cos(pos, emb_dim):
            half_dim = emb_dim//2
            omega = 1. / (10000**(np.arange(half_dim)/half_dim))
            out = np.einsum('m,d->md', pos, omega)
            emb_sin = np.sin(out)
            emb_cos = np.cos(out)
            return np.concatenate([emb_sin, emb_cos], axis=1)

        h = np.arange(grid_size, dtype=np.float32)
        w = np.arange(grid_size, dtype=np.float32)
        ww, hh = np.meshgrid(w, h)
        ww = ww.reshape(-1)
        hh = hh.reshape(-1)
        assert embed_dim % 2 == 0
        emb_h = get_1d_sin_cos(hh, embed_dim//2)
        emb_w = get_1d_sin_cos(ww, embed_dim//2)
        return np.concatenate([emb_h, emb_w], axis=1)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, patch_dim = x.shape
        p = self.patch_size
        c = self.out_channels
        h = w = int(T**0.5)
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h*p, w*p)
        return x

    def forward(self, x_img: torch.Tensor, t: torch.Tensor, tab: torch.Tensor, cfg: float = 1.0):
        """
        - x_img: (B, C, H, W)
        - t:     (B,) timesteps
        - tab:   (B, num_tab_columns)
        - cfg:   guidance scale
        """
        if cfg == 1.0:
            return self._forward_no_cfg(x_img, t, tab)
        else:
            return self._forward_with_cfg(x_img, t, tab, cfg)

    def _forward_no_cfg(self, x_img, t, tab):
        B = x_img.shape[0]

        # 1) Image patchify + pos
        img_tokens = self.x_embedder(x_img)   # (B, T_img, dim)
        img_tokens = img_tokens + self.pos_embed

        # 2) Tab -> tokens
        tab_tokens = self.tab_proj(tab)       # (B, T_tab, dim)

        # 3) time embed
        t_emb = self.t_embedder(t)

        # 4) pass through blocks
        for blk in self.blocks:
            if isinstance(blk, ImageBlock):
                img_tokens = blk(img_tokens, t_emb)
            elif isinstance(blk, nn.ModuleDict):
                # parallel
                img_tokens = blk['img'](img_tokens, t_emb)
                tab_tokens = blk['tab'](tab_tokens, t_emb)
            else:
                # BiCrossBlock
                img_tokens, tab_tokens = blk(img_tokens, tab_tokens, t_emb)

        # 5) final heads
        img_logits = self.final_img(img_tokens, t_emb)  # (B, T_img, p^2*C)
        x_img_out = self.unpatchify(img_logits)
        tab_out = self.final_tab(tab_tokens)            # (B, out_table_features)

        return {
            "image_sample": x_img_out,
            "table_sample": tab_out
        }

    def _forward_with_cfg(self, x_img, t, tab, cfg):
        """
        CFG by zeroing out image+tab in the unconditional pass (second half).
        """
        B = x_img.shape[0]
        # replicate
        x_img_cat = torch.cat([x_img, torch.zeros_like(x_img)], dim=0)  # half real, half zero
        tab_cat   = torch.cat([tab, torch.zeros_like(tab)], dim=0)
        t_cat     = torch.cat([t, t], dim=0)

        # forward
        out_cat = self._forward_no_cfg(x_img_cat, t_cat, tab_cat)
        img_cat = out_cat['image_sample']     # (2B, C, H, W)
        tab_cat_ = out_cat['table_sample']    # (2B, out_features)

        # split cond/uncond
        cond_img, uncond_img = torch.split(img_cat, B, dim=0)
        cond_tab, uncond_tab = torch.split(tab_cat_, B, dim=0)

        # combine => standard CFG formula
        final_img = uncond_img + cfg * (cond_img - uncond_img)
        final_tab = uncond_tab + cfg * (cond_tab - uncond_tab)

        return {
            "image_sample": final_img,
            "table_sample": final_tab
        }


def load_dit(cfg: DictConfig, **overrides: Any) -> nn.Module:
    """
    Load a MultiModalDiT model based on the provided configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object for the DiT model.
        **overrides (Any): Arbitrary keyword arguments used to override the default configuration.

    Returns:
        nn.Module: The loaded MultiModalDiT model.
    """
    # Apply any overrides to the config before loading
    cfg = apply_overrides(cfg, overrides)

    print("[INFO] Loading MultiModalDiT model with config:", cfg)

    # Instantiate the MultiModalDiT model
    model = MultiModalDiT(**cfg.dit)
    print("[INFO] Loaded DiT")
    return model

###############################################################################
# Usage Example:
###############################################################################
if __name__ == "__main__":

    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth = 16

    model = MultiModalDiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=128,  # smaller dim for a quick test
        depth=6,  # fewer layers for a test
        head_dim=32,
        multiple_of=64,
        qkv_ratio=1.0,
        mlp_ratio=2.0,
        norm_eps=1e-6,
        use_bias=True,
        depth_init=True,
        num_tab_columns=10,
        tab_groups=2,
        out_table_features=5,
        moe_frequency=2,  # MoE every 2 layers on the image side
        num_experts=4,
        expert_capacity=1.0
    )

    # Fake data

    batch_size = 2
    x_img = torch.randn(batch_size, 4, 32, 32)  # (B, in_channels, H, W)
    t = torch.tensor([10, 20])  # (B,) scalar timesteps
    tab = torch.randn(batch_size, 10)  # (B, num_tab_columns)

    # Forward pass (CFG=1.0 => normal pass; or e.g. cfg=1.5)
    out = model(x_img, t, tab, cfg=1.5)

    print("Output keys:", out.keys())
    print("Image sample shape:", out["image_sample"].shape)
    print("Tab sample shape:", out["table_sample"].shape)
