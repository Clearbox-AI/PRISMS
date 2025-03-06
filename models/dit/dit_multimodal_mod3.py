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


def ntuple(n: int):
    """Converts input into an n-tuple."""
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


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


################################################################################
# FeedForwardECMoe, FeedForward
################################################################################

class FeedForwardECMoe(nn.Module):
    """Expert-Choice style MoE feed-forward."""
    def __init__(self, num_experts, expert_capacity, dim, hidden_dim, multiple_of):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dim = dim
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        self.w1 = nn.Parameter(torch.ones(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, hidden_dim, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        tokens_per_expert = int(self.expert_capacity * T / self.num_experts)

        scores = self.gate(x)  # (B, T, E)
        probs = F.softmax(scores, dim=-1)
        g, m = torch.topk(probs.permute(0,2,1), tokens_per_expert, dim=-1)
        p = F.one_hot(m, num_classes=T).float()
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


class FeedForward(nn.Module):
    """
    Same as earlier: 2-lin style with hidden_dim = (2/3)*mlp_ratio, etc.
    """
    def __init__(self, dim, hidden_dim, multiple_of=256, use_bias=True):
        super().__init__()
        # matches original microdiffusion trick
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


################################################################################
# CrossAttention, SelfAttention
################################################################################

class CrossAttention(nn.Module):
    """
    Cross attention from image tokens -> tabular tokens.
    No separate gating or shift/scale for cross-attn in this example.
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
        self.kv_linear = nn.Linear(dim, 2 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T_img, dim)
        cond: (B, T_tab, dim)
        """
        B, T_img, _ = x.shape
        T_tab = cond.shape[1]

        q = self.q_linear(x)            # (B, T_img, hidden_dim)
        kv = self.kv_linear(cond)       # (B, T_tab, 2*hidden_dim)
        k, v = kv.chunk(2, dim=-1)      # each (B, T_tab, hidden_dim)

        q = q.reshape(B, T_img, self.num_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T_tab, self.num_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T_tab, self.num_heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v                     # (B, num_heads, T_img, head_dim)
        out = out.transpose(1, 2).reshape(B, T_img, self.num_heads*self.head_dim)
        out = self.proj(out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.q_linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.kv_linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    """
    Standard self-attention (multi-head).
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
        """
        x: (B, T, dim)
        """
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3*hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, T, self.num_heads*self.head_dim)
        out = self.proj(out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


################################################################################
# MultimodalBlocks
################################################################################

class MultiModalBlock_ImgFromTab(nn.Module):
    """
    One block that updates image tokens from tab tokens (cross-attn),
    while tab tokens only do unimodal self-attn and MLP.
    """

    def __init__(
            self,
            dim: int,
            head_dim: int,
            mlp_ratio: float,
            qkv_ratio: float,
            multiple_of: int,
            time_emb_dim: int,
            norm_eps: float,
            depth_init: bool,
            layer_id: int,
            num_layers: int,
            use_bias: bool,
            # MoE or not for image MLP
            moe_img: bool,
            num_experts: int,
            expert_capacity: float,
            # MoE or not for tab MLP
            moe_tab: bool,
    ):
        super().__init__()
        self.dim = dim

        # QKV dims
        qkv_hidden_dim = (
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        # -------------- IMAGE branch --------------
        # (A) Self-attn (image)
        self.norm1_img = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_img = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )
        # (B) Cross-attn (image from tab)
        self.norm2_img = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_img = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )
        # (C) MLP
        self.norm3_img = create_norm('layernorm', dim, eps=norm_eps)
        if moe_img:
            self.mlp_img = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp_img = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # AdaLN for image: produce 6*dim => (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_mod_img = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6 * dim)
        )

        # -------------- TAB branch --------------
        # Tab tokens only do self-attn + MLP (no cross-attn in this block)
        self.norm1_tab = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_tab = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )
        self.norm2_tab = create_norm('layernorm', dim, eps=norm_eps)
        if moe_tab:
            self.mlp_tab = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp_tab = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # AdaLN for tab
        self.adaLN_mod_tab = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6 * dim)
        )

        # init scale
        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * num_layers) ** 0.5

    def forward(self, x_img, x_tab, t_emb):
        """
        x_img: (B, T_img, dim)
        x_tab: (B, T_tab, dim)
        t_emb: (B, time_emb_dim)
        returns (x_img, x_tab) updated
        """
        B, T_img, D = x_img.shape

        # ---------------- IMAGE BRANCH ----------------
        # get shift/scale/gate for image
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_mod_img(t_emb).chunk(6, dim=1)  # each (B, dim)

        # 1) Image self-attn
        x_ln_img = modulate(self.norm1_img(x_img), shift_msa_i, scale_msa_i)
        x_img = x_img + gate_msa_i.unsqueeze(1) * self.attn_img(x_ln_img)

        # 2) Cross-attn (image from tab)
        x_ln2_img = self.norm2_img(x_img)
        x_img = x_img + self.cross_attn_img(x_ln2_img, x_tab)

        # 3) MLP
        x_ln3_img = modulate(self.norm3_img(x_img), shift_mlp_i, scale_mlp_i)
        x_img = x_img + gate_mlp_i.unsqueeze(1) * self.mlp_img(x_ln3_img)

        # ---------------- TAB BRANCH ----------------
        # get shift/scale/gate for tab
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_mod_tab(t_emb).chunk(6, dim=1)

        # 1) Tab self-attn
        x_ln_tab = modulate(self.norm1_tab(x_tab), shift_msa_t, scale_msa_t)
        x_tab = x_tab + gate_msa_t.unsqueeze(1) * self.attn_tab(x_ln_tab)

        # 2) Tab MLP
        x_ln2_tab = modulate(self.norm2_tab(x_tab), shift_mlp_t, scale_mlp_t)
        x_tab = x_tab + gate_mlp_t.unsqueeze(1) * self.mlp_tab(x_ln2_tab)

        return x_img, x_tab

    def custom_init(self):
        # image branch
        self.norm1_img.reset_parameters()
        self.norm2_img.reset_parameters()
        self.norm3_img.reset_parameters()
        self.attn_img.custom_init(self.weight_init_std)
        self.cross_attn_img.custom_init(self.weight_init_std)
        self.mlp_img.custom_init(self.weight_init_std)
        # tab branch
        self.norm1_tab.reset_parameters()
        self.norm2_tab.reset_parameters()
        self.attn_tab.custom_init(self.weight_init_std)
        self.mlp_tab.custom_init(self.weight_init_std)


class MultiModalBlock_TabFromImg(nn.Module):
    """
    Opposite direction: tab tokens cross-attend to image tokens,
    while image tokens do only unimodal self-attn + MLP.
    """

    def __init__(
            self,
            dim: int,
            head_dim: int,
            mlp_ratio: float,
            qkv_ratio: float,
            multiple_of: int,
            time_emb_dim: int,
            norm_eps: float,
            depth_init: bool,
            layer_id: int,
            num_layers: int,
            use_bias: bool,
            # MoE or not for image MLP
            moe_img: bool,
            num_experts: int,
            expert_capacity: float,
            # MoE or not for tab MLP
            moe_tab: bool,
    ):
        super().__init__()
        self.dim = dim

        # QKV dims
        qkv_hidden_dim = (
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        # -------------- IMAGE branch (no cross-attn) --------------
        self.norm1_img = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_img = SelfAttention(dim, qkv_hidden_dim // head_dim, use_bias, qkv_hidden_dim)
        self.norm2_img = create_norm('layernorm', dim, eps=norm_eps)
        if moe_img:
            self.mlp_img = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp_img = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        self.adaLN_mod_img = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6 * dim)
        )

        # -------------- TAB branch (with cross-attn from image) --------------
        self.norm1_tab = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_tab = SelfAttention(dim, qkv_hidden_dim // head_dim, use_bias, qkv_hidden_dim)
        self.norm2_tab = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_tab = CrossAttention(dim, qkv_hidden_dim // head_dim, use_bias, qkv_hidden_dim)
        self.norm3_tab = create_norm('layernorm', dim, eps=norm_eps)
        if moe_tab:
            self.mlp_tab = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp_tab = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        self.adaLN_mod_tab = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6 * dim)
        )

        # init scale
        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * num_layers) ** 0.5

    def forward(self, x_img, x_tab, t_emb):
        # -------------- IMAGE BRANCH: self-attn + MLP only --------------
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_mod_img(t_emb).chunk(6, dim=1)

        x_ln_img = modulate(self.norm1_img(x_img), shift_msa_i, scale_msa_i)
        x_img = x_img + gate_msa_i.unsqueeze(1) * self.attn_img(x_ln_img)

        x_ln2_img = modulate(self.norm2_img(x_img), shift_mlp_i, scale_mlp_i)
        x_img = x_img + gate_mlp_i.unsqueeze(1) * self.mlp_img(x_ln2_img)

        # -------------- TAB BRANCH: self-attn + cross-attn + MLP --------------
        shift_msa_t, scale_msa_t, gate_msa_t, shift_ca_t, scale_ca_t, gate_ca_t = \
            self.adaLN_mod_tab(t_emb).chunk(6, dim=1)

        # (A) self-attn
        x_ln_tab = modulate(self.norm1_tab(x_tab), shift_msa_t, scale_msa_t)
        x_tab = x_tab + gate_msa_t.unsqueeze(1) * self.attn_tab(x_ln_tab)

        # (B) cross-attn from image
        x_ln2_tab = modulate(self.norm2_tab(x_tab), shift_ca_t, scale_ca_t)
        x_tab = x_tab + self.cross_attn_tab(x_ln2_tab, x_img)

        # (C) MLP
        x_ln3_tab = self.norm3_tab(x_tab)
        x_tab = x_tab + self.mlp_tab(x_ln3_tab)

        return x_img, x_tab

    def custom_init(self):
        # image branch
        self.norm1_img.reset_parameters()
        self.norm2_img.reset_parameters()
        self.attn_img.custom_init(self.weight_init_std)
        self.mlp_img.custom_init(self.weight_init_std)
        # tab branch
        self.norm1_tab.reset_parameters()
        self.norm2_tab.reset_parameters()
        self.norm3_tab.reset_parameters()
        self.attn_tab.custom_init(self.weight_init_std)
        self.cross_attn_tab.custom_init(self.weight_init_std)
        self.mlp_tab.custom_init(self.weight_init_std)


################################################################################
# PatchMixer, PatchEmbed
################################################################################

class PatchMixerBlock(nn.Module):
    """
    A simple DiT-like block that only does self-attn + MLP on the patch tokens,
    modulated by the time embedding (and possibly we can add the tab embedding if desired).
    But typically it's used for image patch mixing alone.

    We'll do the same 6×dim approach for MSA + MLP gating, or we can keep it simpler.
    For demonstration, let's keep it consistent with the main blocks.
    """
    def __init__(
        self,
        dim: int,
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        time_emb_dim: int,   # can use time embed or some other mod
        norm_eps: float,
        depth_init: bool,
        layer_id: int,
        num_layers: int,
        use_bias: bool,
        # MoE or not
        moe_block: bool,
        num_experts: int,
        expert_capacity: float,
    ):
        super().__init__()
        # QKV dims
        qkv_hidden_dim = (
            (head_dim*2)*((int(dim*qkv_ratio) + head_dim*2 -1)//(head_dim*2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim*mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias, hidden_dim=qkv_hidden_dim)

        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        if moe_block:
            self.mlp = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # 6×dim => shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim)
        )

        if depth_init:
            self.weight_init_std = 0.02 / (2*(layer_id+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*num_layers)**0.5

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # x: (B, T, dim)
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Self Attn
        x_ln = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_ln)

        # MLP
        x_ln2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_ln2)
        return x

    def custom_init(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)


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


################################################################################
# Timestep Embedding, TabularProjection, FinalLayer, Masking
################################################################################

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


class TabularProjection(nn.Module):
    """
    Group columns into fewer tokens (slightly coarse grouping)
    and then project each group to the transformer dimension.

    Example:
      num_columns=10, groups=3
      => group_size ~ 4,4,2 => produce (B, 3, hidden_size) tokens
    """
    def __init__(self, in_features: int, hidden_size: int, groups: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups

        # We create an MLP for each group:
        #   or, you could do a single linear that just slices columns.
        #   For demonstration, we'll define a simple approach with multiple modules in a list.
        self.group_mlps = nn.ModuleList()
        # We'll precompute group boundaries
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
        """
        tabular_input: (B, num_columns)
        Returns: (B, groups, hidden_size)
        """
        B, C = tabular_input.shape
        # chunk columns and apply each group MLP
        group_sizes = self._get_group_sizes(C, self.groups)

        outputs = []
        start = 0
        for i, gsize in enumerate(group_sizes):
            mlp = self.group_mlps[i]
            subset = tabular_input[:, start:start+gsize]  # (B, gsize)
            out = mlp(subset)                             # (B, hidden_size)
            outputs.append(out.unsqueeze(1))
            start += gsize

        # concatenate along dim=1 => (B, groups, hidden_size)
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _get_group_sizes(num_cols: int, groups: int) -> List[int]:
        """
        Splits num_cols into 'groups' chunks as evenly as possible.
        Example: num_cols=10, groups=3 => [4,4,2]
        """
        base = num_cols // groups
        remainder = num_cols % groups
        sizes = []
        for i in range(groups):
            size = base + (1 if i < remainder else 0)
            sizes.append(size)
        return sizes


class FinalLayer(nn.Module):
    """
    Final image generation head:
      AdaLN => linear => (B, T, p^2 * out_channels).
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
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)  # each (B, in_dim)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


def get_mask(batch: int, length: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {
        'mask': mask,
        'ids_keep': ids_keep,
        'ids_restore': ids_restore
    }

def mask_out_token(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    index = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=index)

def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    B, L_keep, D = x.shape
    L = ids_restore.shape[1]
    mask_tokens = mask_token.repeat(B, L - L_keep, 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
    return x_


################################################################################
# A "2‐way" MultiModal DiT
################################################################################

class MultiModalDiT(nn.Module):
    """
    Example architecture with:
      - PatchEmbed for images
      - TimestepEmbedder for diffusion time
      - TabularProjection for table inputs
      - Interleaved blocks: (ImgFromTab) -> (TabFromImg) -> ...
      - Optional PatchMixer stage
      - Masking for image tokens
      - Classifier-Free Guidance
      - MoE in certain blocks

    We'll keep the same logic for final_img, final_tab, etc.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=512,
        depth=6,
        head_dim=64,
        multiple_of=256,
        qkv_multipliers=[1.0],
        ffn_multipliers=[4.0],
        norm_eps=1e-6,
        depth_init=True,
        use_bias=True,
        # tabular
        num_tab_columns=10,
        tab_groups=3,
        out_table_features=10,
        # Experts
        num_experts=4,
        expert_capacity=1.0,
        experts_every_n=2,
        # PatchMixer optional
        use_patch_mixer=False,
        patch_mixer_depth=2,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim
        self.num_tab_columns = num_tab_columns
        self.use_patch_mixer = use_patch_mixer

        # Patchify images
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.num_patches = self.x_embedder.num_patches
        self.base_size = input_size // patch_size

        # Time embed
        self.t_embedder = TimestepEmbedder(hidden_size=dim, act_layer=nn.GELU)

        # Tab grouping
        self.table_proj = TabularProjection(
            in_features=num_tab_columns,
            hidden_size=dim,
            groups=tab_groups
        )
        self.table_preproc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Optional patch mixer
        # (like your original PatchMixerBlock code)
        self.patch_mixer = None
        if use_patch_mixer:
            from math import ceil
            pm_expert_blocks_idx = [i for i in range(patch_mixer_depth) if (i+1) % experts_every_n == 0]
            pm_is_moe_block = [(i in pm_expert_blocks_idx) for i in range(patch_mixer_depth)]

            self.patch_mixer = nn.ModuleList([
                PatchMixerBlock(
                    dim=(patch_mixer_dim),
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    time_emb_dim=dim,   # or patch_mixer_dim
                    norm_eps=norm_eps,
                    depth_init=False,   # simpler approach
                    layer_id=i,
                    num_layers=patch_mixer_depth,
                    use_bias=use_bias,
                    moe_block=pm_is_moe_block[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                )
                for i in range(patch_mixer_depth)
            ])

            if patch_mixer_dim != dim:
                self.patch_mixer_map_xin = nn.Sequential(
                    create_norm('layernorm', dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
                self.patch_mixer_map_xout = nn.Sequential(
                    create_norm('layernorm', patch_mixer_dim, eps=norm_eps),
                    nn.Linear(patch_mixer_dim, dim, bias=use_bias)
                )
            else:
                self.patch_mixer_map_xin = nn.Identity()
                self.patch_mixer_map_xout = nn.Identity()

        # Prepare per-layer qkv and ffn multipliers
        total_depth = depth
        if len(ffn_multipliers) == total_depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            # spread out the multipliers
            num_splits = len(ffn_multipliers)
            assert total_depth % num_splits == 0
            dps = total_depth // num_splits
            qkv_ratios = list(np.concatenate([[m]*dps for m in qkv_multipliers]))
            mlp_ratios = list(np.concatenate([[m]*dps for m in ffn_multipliers]))

        # Determine which blocks use MoE for image MLP / tab MLP
        # (For simplicity, we'll say "experts_every_n" => the block uses MoE for both image & tab)
        # but you can pick separate schedules for image vs. tab if you like
        self.blocks = nn.ModuleList()
        for i in range(depth):
            moe_flag = ((i+1) % experts_every_n == 0)
            # pick cross-attn direction
            if i % 2 == 0:
                blk = MultiModalBlock_ImgFromTab(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratios[i],
                    qkv_ratio=qkv_ratios[i],
                    multiple_of=multiple_of,
                    time_emb_dim=dim,
                    norm_eps=norm_eps,
                    depth_init=depth_init,
                    layer_id=i,
                    num_layers=depth,
                    use_bias=use_bias,
                    moe_img=moe_flag,
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                    moe_tab=moe_flag,
                )
            else:
                blk = MultiModalBlock_TabFromImg(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratios[i],
                    qkv_ratio=qkv_ratios[i],
                    multiple_of=multiple_of,
                    time_emb_dim=dim,
                    norm_eps=norm_eps,
                    depth_init=depth_init,
                    layer_id=i,
                    num_layers=depth,
                    use_bias=use_bias,
                    moe_img=moe_flag,
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                    moe_tab=moe_flag,
                )
            self.blocks.append(blk)

        # final heads
        self.final_img = FinalLayer(
            in_dim=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_chans=self.out_channels,
            act_layer=nn.GELU,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps),
        )
        self.final_tab = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_table_features)
        )

        # mask token for re-inserting masked patches
        self.register_buffer("mask_token", torch.zeros(1, 1, patch_size**2*self.out_channels))

        # position embedding for image tokens (if you like sin-cos)
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, dim))

        self.initialize_weights()

    def initialize_weights(self):
        # Basic init
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
        pe = self.get_2d_sincos_pe(side, self.dim, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # conv init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # patch_mixer init
        if self.patch_mixer:
            for blk in self.patch_mixer:
                blk.custom_init()

        # main blocks init
        for blk in self.blocks:
            blk.custom_init()

        # zero out final linear
        nn.init.constant_(self.final_img.linear.weight, 0)
        nn.init.constant_(self.final_img.adaLN_modulation[-1].weight, 0)


    @staticmethod
    def get_2d_sincos_pe(grid_size, embed_dim, base_size=16):
        def get_1d_sin_cos(pos, emb_dim):
            half_dim = emb_dim // 2
            omega = 1. / (10000**(np.arange(half_dim)/half_dim))
            out = np.einsum('m,d->md', pos, omega)
            emb_sin = np.sin(out)
            emb_cos = np.cos(out)
            return np.concatenate([emb_sin, emb_cos], axis=1)

        h = np.arange(grid_size, dtype=np.float32) / (grid_size/base_size)
        w = np.arange(grid_size, dtype=np.float32) / (grid_size/base_size)
        ww, hh = np.meshgrid(w, h)
        ww = ww.reshape(-1)
        hh = hh.reshape(-1)
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


    def forward(
            self,
            x_img: torch.Tensor,   # (B, in_channels, H, W)
            t: torch.Tensor,       # (B,) timesteps
            tab: torch.Tensor = None,  # (B, num_tab_columns)
            cfg: float = 1.0,      # classifier-free guidance scale
            mask_ratio: float = 0.0
    ):
        """
        If tab=None => unconditional generation for tab (just zeros).
        If cfg>1 => do classifier-free guidance approach.
        """
        B = x_img.shape[0]
        if tab is None:
            tab = torch.zeros(B, self.num_tab_columns, device=x_img.device, dtype=x_img.dtype)

        if cfg == 1.0:
            return self.forward_without_cfg(x_img, t, tab, mask_ratio)
        else:
            return self.forward_with_cfg(x_img, t, tab, cfg, mask_ratio)


    def forward_without_cfg(self, x_img, t, tab, mask_ratio=0.0):
        # 1) patchify image
        x_img_tokens = self.x_embedder(x_img)
        # add pos embed
        x_img_tokens = x_img_tokens + self.pos_embed

        # 2) time embedding
        t_emb = self.t_embedder(t)

        # 3) tab => project => preproc
        x_tab_tokens = self.table_proj(tab)
        x_tab_tokens = self.table_preproc(x_tab_tokens)

        # 4) optional patch mixer (on image tokens)
        if self.patch_mixer:
            x_img_tokens = self.patch_mixer_map_xin(x_img_tokens)
            for blk in self.patch_mixer:
                x_img_tokens = blk(x_img_tokens, t_emb)
            x_img_tokens = self.patch_mixer_map_xout(x_img_tokens)

        # 5) optional masking
        mask = None
        ids_restore = None
        if mask_ratio > 0.0:
            B, T_img, D = x_img_tokens.shape
            mask_info = get_mask(B, T_img, mask_ratio, x_img_tokens.device)
            x_img_tokens = mask_out_token(x_img_tokens, mask_info['ids_keep'])
            mask = mask_info['mask']
            ids_restore = mask_info['ids_restore']

        # 6) main blocks (interleaved cross-attn)
        for blk in self.blocks:
            x_img_tokens, x_tab_tokens = blk(x_img_tokens, x_tab_tokens, t_emb)

        # 7) final image
        img_logits = self.final_img(x_img_tokens, t_emb)
        if mask_ratio > 0.0 and ids_restore is not None:
            img_logits = unmask_tokens(img_logits, ids_restore, self.mask_token)
        img_out = self.unpatchify(img_logits)

        # 8) final tab => pool or do some aggregator
        tab_pooled = x_tab_tokens.mean(dim=1)
        tab_out = self.final_tab(tab_pooled)

        return {
            "image_sample": img_out,
            "table_sample": tab_out,
            "mask": mask
        }

    def forward_with_cfg(self, x_img, t, tab, cfg, mask_ratio=0.0):
        B = x_img.shape[0]
        # replicate the batch => 2B
        x_img_cat = torch.cat([x_img, x_img], dim=0)
        zeros_tab = torch.zeros_like(tab)
        tab_cat = torch.cat([tab, zeros_tab], dim=0)
        t_cat = torch.cat([t, t], dim=0)

        out_cat = self.forward_without_cfg(
            x_img_cat, t_cat, tab_cat, mask_ratio=mask_ratio
        )
        # out_cat => dict with image_sample => (2B, C, H, W), table_sample => (2B, out_feats)
        image_cat = out_cat['image_sample']
        table_cat = out_cat['table_sample']

        cond_img, uncond_img = torch.split(image_cat, B, dim=0)
        cond_tab, uncond_tab = torch.split(table_cat, B, dim=0)

        final_img = uncond_img + cfg * (cond_img - uncond_img)
        final_tab = uncond_tab + cfg * (cond_tab - uncond_tab)

        return {
            "image_sample": final_img,
            "table_sample": final_tab,
            "mask": None
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
        input_size=64,
        patch_size=4,
        in_channels=3,
        dim=256,
        depth=depth,
        head_dim=32,
        multiple_of=64,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174
    )

    # Fake data

    N = 2
    x_img = torch.randn(N, 3, 64, 64)  # e.g. 2 images, 3 channels
    tab = torch.randn(N, 174)
    t = torch.randint(0, 1000, (N,))  # random timesteps


    # 3) Forward pass
    res = model(x_img, t, tab, mask_ratio=0.2)
    if res["mask"] is not None:
        print("img_out shape:", res["image_sample"].shape)  # (N, 3, 64, 64)
        print("table_sample:", res['table_sample'].shape)  # (N, 10)

