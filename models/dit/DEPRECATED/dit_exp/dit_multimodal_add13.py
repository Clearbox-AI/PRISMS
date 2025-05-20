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
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        self.aux_loss: Optional[torch.Tensor] = None  # will be filled in fwd
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        scores = self.gate(x)  # (B,T,E)
        probs = F.softmax(scores, dim=-1)  # differentiable

        # auxiliary load‑balancing loss (factor 0.01 is user‑tunable)
        with torch.no_grad():
            importance = probs.sum(dim=(0, 1))  # (E,)
            self.aux_loss = (importance * importance).sum() * self.num_experts / (B * T)

        # expert computations ----------------------------------------------------
        # x ➔ hidden -------------------------------------------------------------
        h = torch.einsum('btd,edh->bteh', x, self.w1)  # (B,T,E,H)
        h = self.gelu(h)
        h = torch.einsum('bteh,ehd->bted', h, self.w2)  # (B,T,E,D)

        out = (probs.unsqueeze(-1) * h).sum(dim=2)  # aggregate to (B,T,D)
        return out

    # def custom_init(self, init_std: float):
    #     nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
    #     nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
    #     nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


class Mlp(nn.Module):
    """
    Basic MLP from timm (without dropout).
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = lambda: nn.GELU(approximate="tanh"),
        norm_layer: Optional[Any] = None,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TabTransformer(nn.Module):
    """Turns a row of *numeric* columns `(B, num_cols)` into transformer tokens."""

    def __init__(
        self,
        num_cols: int,
        dim: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        num_layers: int = 2,
        use_pos_embed: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_cols = num_cols
        self.dim = dim
        self.num_heads = dim // head_dim
        self.use_pos_embed = use_pos_embed

        self.scalar_embed = nn.Linear(1, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed_tab = nn.Parameter(torch.zeros(1, 1 + num_cols, dim))

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, self.num_heads, mlp_ratio, norm_eps)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(dim, eps=norm_eps)
        self._init_weights()

    # ...................................................... utils & forward
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_tab, std=0.02)
        nn.init.xavier_uniform_(self.scalar_embed.weight)
        nn.init.constant_(self.scalar_embed.bias, 0.)

    def forward(self, x_tab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C = x_tab.shape
        if C != self.num_cols:
            raise ValueError(f"Expected {self.num_cols} columns, got {C}")

        col_tokens = self.scalar_embed(x_tab.unsqueeze(-1))       # (B,C,D)
        cls_tok    = self.cls_token.expand(B, -1, -1)            # (B,1,D)
        tokens     = torch.cat([cls_tok, col_tokens], dim=1)      # (B,1+C,D)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed_tab[:, : tokens.size(1)]

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm_final(tokens)
        return tokens, tokens[:, 0]


class TransformerEncoderLayer(nn.Module):
    """Minimal transformer encoder layer (identical to original)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x



################################################################################
# CrossAttention
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


################################################################################
# SelfAttention, FeedForward, etc.
################################################################################

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
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


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
# Timestep Embedding
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


################################################################################
# PatchEmbed, FinalLayer, Masking
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
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.proj.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)                # (B,T,dim)
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

class _GateInitMixin:
    """Utility mix‑in – provides *_zero_gate()* but no custom_init."""
    def _zero_gate(self):
        proj: nn.Linear = self.adaLN_modulation_x[-1]  # type: ignore[attr-defined]
        nn.init.constant_(proj.weight, 0.)
        if proj.bias is not None:
            nn.init.constant_(proj.bias, 0.)

class MultiModalDiTBlockImgToTab(_GateInitMixin, nn.Module):
    """
    Updates *image tokens*:
      1) Self-Attn on image tokens
      2) Cross-Attn from image tokens -> tab tokens (read‐only)
      3) MLP on image tokens
    The tab tokens are not updated here.
    Each sub-layer has shift, scale, gate from a time+other embedding.
    """
    def __init__(
        self,
        dim: int,
        head_dim,
        mlp_ratio,
        qkv_ratio,
        multiple_of,
        time_emb_dim: int,
        tab_pooled_dim: int,
        layer_id: int,
        depth_init: float,
        num_layers: int,
        norm_eps: float = 1e-6,
        use_bias: bool = True,
    ):
        super().__init__()
        # QKV dims
        qkv_hidden_dim = ((head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
                          if qkv_ratio != 1.0 else dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # sub-layers for the image
        self.norm1_x = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_x = SelfAttention(dim, num_heads=(qkv_hidden_dim // head_dim), qkv_bias=use_bias, hidden_dim=qkv_hidden_dim)

        self.norm2_x = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_x = CrossAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias, hidden_dim=qkv_hidden_dim)

        self.norm3_x = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp_x = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # gating: 3 sub-layers => each has SHIFT, SCALE, GATE => 9 * dim total
        self.adaLN_modulation_x = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 9 * dim, bias=True)
        )

        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5 if depth_init else
            0.02 / (2 * num_layers) ** 0.5
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor, t_emb: torch.Tensor, x_tab_pooled: torch.Tensor):
        """
        x_img: (B, T_img, dim)
        x_tab: (B, T_tab, dim)  [read-only, used as K/V in cross-attn]
        t_emb: (B, time_emb_dim)
        Returns updated x_img
        """
        # gating
        combined_img = t_emb + x_tab_pooled
        shift_scale_gate = self.adaLN_modulation_x(combined_img)  # (B, 9*dim)
        (
            shift_msa_x, scale_msa_x, gate_msa_x,
            shift_cross_x, scale_cross_x, gate_cross_x,
            shift_mlp_x, scale_mlp_x, gate_mlp_x
        ) = shift_scale_gate.chunk(9, dim=1)

        # 1) Self-Attn on image
        x_ln = modulate(self.norm1_x(x_img), shift_msa_x, scale_msa_x)
        x_img = x_img + gate_msa_x.unsqueeze(1) * self.attn_x(x_ln)

        # 2) Cross-Attn from image -> tab (x_img queries, x_tab is K/V)
        x_ln2 = modulate(self.norm2_x(x_img), shift_cross_x, scale_cross_x)
        x_img = x_img + gate_cross_x.unsqueeze(1) * self.cross_attn_x(x_ln2, x_tab)

        # 3) MLP
        x_ln3 = modulate(self.norm3_x(x_img), shift_mlp_x, scale_mlp_x)
        x_img = x_img + gate_mlp_x.unsqueeze(1) * self.mlp_x(x_ln3)

        return x_img

    def custom_init(self):
        # original weight init
        for norm in (self.norm1_x, self.norm2_x, self.norm3_x):
            norm.reset_parameters()
        self.attn_x.custom_init(self.weight_init_std)
        self.cross_attn_x.custom_init(self.weight_init_std)
        self.mlp_x.custom_init(self.weight_init_std)
        # the gate gets zeroed globally via `_zero_gate()` in initialize_weights


class MultiModalDiTBlockTabToImg(_GateInitMixin, nn.Module):
    """
    Updates *tab tokens*:
      1) Self-Attn on tab tokens
      2) Cross-Attn from tab tokens -> image tokens (read-only)
      3) MLP on tab tokens
    The image tokens are not updated here.
    """
    def __init__(
        self,
        dim: int,
        head_dim,
        mlp_ratio,
        qkv_ratio,
        multiple_of,
        time_emb_dim: int,
        img_pooled_dim: int,
        layer_id: int,
        depth_init: float,
        num_layers: int,
        norm_eps: float = 1e-6,
        use_bias: bool = True,
    ):
        super().__init__()
        # QKV dims
        qkv_hidden_dim = ((head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
                          if qkv_ratio != 1.0 else dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1_t = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_t = SelfAttention(dim, num_heads=(qkv_hidden_dim // head_dim), qkv_bias=use_bias,
                                    hidden_dim=qkv_hidden_dim)

        self.norm2_t = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_t = CrossAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                           hidden_dim=qkv_hidden_dim)

        self.norm3_t = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp_t = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # gating: 3 sub-layers => SHIFT, SCALE, GATE => 9 * dim total
        self.adaLN_modulation_t = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 9 * dim)
        )

        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5 if depth_init else
            0.02 / (2 * num_layers) ** 0.5
        )

    def forward(self, x_tab: torch.Tensor, x_img: torch.Tensor, t_emb: torch.Tensor, x_img_pooled: torch.Tensor):
        """
        x_tab: (B, T_tab, dim)
        x_img: (B, T_img, dim) [read-only, used as K/V in cross-attn]
        t_emb: (B, time_emb_dim)
        Returns updated x_tab
        """

        combined_tab = t_emb + x_img_pooled
        shift_scale_gate = self.adaLN_modulation_t(combined_tab)  # (B, 9*dim)
        (
            shift_msa_t, scale_msa_t, gate_msa_t,
            shift_cross_t, scale_cross_t, gate_cross_t,
            shift_mlp_t, scale_mlp_t, gate_mlp_t
        ) = shift_scale_gate.chunk(9, dim=1)

        # 1) Self-Attn on tab
        t_ln = modulate(self.norm1_t(x_tab), shift_msa_t, scale_msa_t)
        x_tab = x_tab + gate_msa_t.unsqueeze(1) * self.attn_t(t_ln)

        # 2) Cross-Attn from tab -> image (x_tab queries, x_img is K/V)
        t_ln2 = modulate(self.norm2_t(x_tab), shift_cross_t, scale_cross_t)
        x_tab = x_tab + gate_cross_t.unsqueeze(1) * self.cross_attn_t(t_ln2, x_img)

        # 3) MLP
        t_ln3 = modulate(self.norm3_t(x_tab), shift_mlp_t, scale_mlp_t)
        x_tab = x_tab + gate_mlp_t.unsqueeze(1) * self.mlp_t(t_ln3)

        return x_tab

    def custom_init(self):
        for norm in (self.norm1_t, self.norm2_t, self.norm3_t):
            norm.reset_parameters()
        self.attn_t.custom_init(self.weight_init_std)
        self.cross_attn_t.custom_init(self.weight_init_std)
        self.mlp_t.custom_init(self.weight_init_std)

class FinalTabHead(nn.Module):
    """Maps transformer tokens → numeric prediction for each tab column."""

    def __init__(
        self,
        in_dim: int,
        time_emb_dim: int,
        num_cols: int,
        act_layer=nn.SiLU,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_cols = num_cols
        self.norm_final = nn.LayerNorm(in_dim, eps=norm_eps)
        self.adaLN = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2 * in_dim)
        )
        self.linear = nn.Linear(in_dim, 1)
        nn.init.trunc_normal_(self.linear.weight, std=0.02 / math.sqrt(in_dim))
        nn.init.constant_(self.linear.bias, 0.)

    def forward(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        col_tokens = tokens[:, 1:, :]                          # (B,C,D)
        col_tokens = self.norm_final(col_tokens)
        shift, scale = self.adaLN(t_emb).chunk(2, dim=1)
        col_mod = col_tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.linear(col_mod).squeeze(-1)                 # (B,C)
        return out

# -----------------------------------------------------------------------------
# 4. Helper to zero‑init AdaLN gate parameters (called post‑construction)
# -----------------------------------------------------------------------------

def _zero_gate(module: nn.Module, attr: str) -> None:
    """
    Zero **only the final third** (= gate) of the AdaLN projection so that
    SHIFT and SCALE are kept informative.  Works for every module that uses
    our [act ▸ Linear(out = 9·D)] pattern.
    """
    proj: nn.Linear = getattr(module, attr)[-1]

    # gates exist only when out_features is a multiple of 3 × hidden_dim
    if proj.out_features % 3 != 0:
        return  # no gate ➜ nothing to do

    D = proj.out_features // 3
    with torch.no_grad():
        proj.weight[:, 2 * D:] = 0.0
        if proj.bias is not None:
            proj.bias[2 * D:] = 0.0

class ViTPatchMixerBlock(nn.Module):
    """
    A small block that does:
      1) Self-attn on [CLS + image tokens]
      2) Cross-attn from image tokens -> tab tokens (read-only)
      3) MLP
    Each sub-layer is modulated by t_emb + tab_pooled => (B, patch_mixer_dim).
    """
    def __init__(
        self,
        dim: int,
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        time_tab_dim: int,   # dimension of (t_emb + tab_pooled)
        layer_id: int,
        num_layers: int,
        norm_eps: float = 1e-6,
        depth_init: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        # qkv dimension
        qkv_hidden_dim = (
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Norm + self-attn on [CLS + patches]
        self.norm1 = create_norm("layernorm", dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=(qkv_hidden_dim // head_dim),
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim,
        )

        # Norm + cross-attn from image -> tab
        self.norm2 = create_norm("layernorm", dim, eps=norm_eps)
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=(qkv_hidden_dim // head_dim),
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim,
        )

        # Norm + MLP
        self.norm3 = create_norm("layernorm", dim, eps=norm_eps)
        self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # Gating => 9*dim = SHIFT, SCALE, GATE for 3 sublayers
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 9 * dim, bias=True)
        )

        # Weight init
        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * num_layers) ** 0.5

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor, cond_emb: torch.Tensor):
        """
        Args:
          x_img: (B, T_img+1, dim) => includes [CLS] token at index 0
          x_tab: (B, T_tab, dim)   => read-only tab tokens for cross-attn
          cond_emb: (B, time_tab_dim) => e.g. t_emb + tab_pooled
        Returns:
          updated x_img => (B, T_img+1, dim)
        """
        # gating
        # 9 chunks => shift/scale/gate for self-attn, cross-attn, MLP
        shift_scale_gate = self.adaLN_modulation(cond_emb)  # (B, 9*dim)
        (
            shift_msa_x, scale_msa_x, gate_msa_x,
            shift_cross_x, scale_cross_x, gate_cross_x,
            shift_mlp_x, scale_mlp_x, gate_mlp_x
        ) = shift_scale_gate.chunk(9, dim=1)

        # 1) Self-Attn on [CLS + patches]
        x_ln1 = modulate(self.norm1(x_img), shift_msa_x, scale_msa_x)
        x_img = x_img + gate_msa_x.unsqueeze(1) * self.attn(x_ln1)

        # 2) Cross-Attn => x_tab is K/V, x_img is Q
        x_ln2 = modulate(self.norm2(x_img), shift_cross_x, scale_cross_x)
        x_img = x_img + gate_cross_x.unsqueeze(1) * self.cross_attn(x_ln2, x_tab)

        # 3) MLP
        x_ln3 = modulate(self.norm3(x_img), shift_mlp_x, scale_mlp_x)
        x_img = x_img + gate_mlp_x.unsqueeze(1) * self.mlp(x_ln3)

        return x_img

    def custom_init(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.cross_attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)


class ViTPatchMixer(nn.Module):
    """
    A short stack of patch mixer blocks (like a mini ViT) that does:
      - Insert a [CLS] token among the image patches
      - self-attn + cross-attn with tab tokens
      - gating from cond_emb (t_emb + tab_pooled)
      - returns final patch tokens (excluding CLS) and the final CLS as img_pooled

    We assume the dimension is patch_mixer_dim, i.e. smaller or equal to main dim.
    """
    def __init__(
        self,
        patch_mixer_dim: int,
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        time_tab_dim: int,       # dimension of cond_emb
        patch_mixer_depth: int,
        norm_eps: float = 1e-6,
        depth_init: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        # self.dim = patch_mixer_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_mixer_dim))

        # A small stack of blocks
        self.blocks = nn.ModuleList([
            ViTPatchMixerBlock(
                dim=patch_mixer_dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                qkv_ratio=qkv_ratio,
                multiple_of=multiple_of,
                time_tab_dim=time_tab_dim,
                layer_id=i,
                num_layers=patch_mixer_depth,
                norm_eps=norm_eps,
                depth_init=depth_init,
                use_bias=use_bias
            )
            for i in range(patch_mixer_depth)
        ])

        self.norm_final = nn.LayerNorm(patch_mixer_dim, eps=norm_eps)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def custom_init(self):
        for blk in self.blocks:
            blk.custom_init()

    def forward(
        self,
        img_tokens: torch.Tensor,  # shape (B, T_img, patch_mixer_dim)
        x_tab: torch.Tensor,       # shape (B, T_tab, patch_mixer_dim), read-only
        cond_emb: torch.Tensor,    # shape (B, time_tab_dim) => e.g. t_emb + tab_pooled
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          final_patches: (B, T_img, patch_mixer_dim)    # excludes CLS
          img_cls: (B, patch_mixer_dim)                  # the final CLS
        """
        B, T, D = img_tokens.shape
        # Insert [CLS] at front
        cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls_token, img_tokens], dim=1) # => (B, 1+T, D)

        for block in self.blocks:
            x = block(x, x_tab, cond_emb)

        # final LN
        x = self.norm_final(x)

        # separate [CLS] from patch tokens
        img_cls = x[:, 0, :]         # (B, D)
        final_patches = x[:, 1:, :]  # (B, T, D)
        return final_patches, img_cls


################################################################################
# A "middle‐way" MultiModal DiT with coarse column grouping
################################################################################

class MultiModalDiT(nn.Module):
    """
    Demonstration:
      - Slightly coarse grouping of tab columns => multiple tokens
      - 6×dim for MSA + MLP gating only (no gating for cross-attn)
      - CrossAttn used in main blocks
      - Optional Patch Mixer stage
      - Optional MoE in certain blocks
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
        # tabular settings
        num_tab_columns=157,
        tab_groups=10,
        out_table_features=10,
        # Patch mixer
        use_patch_mixer=True,
        patch_mixer_depth=2,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
        # Experts (MoE)
        num_experts=8,
        expert_capacity=1.0,
        experts_every_n=2
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim
        self.use_patch_mixer = use_patch_mixer
        self.patch_mixer_dim = patch_mixer_dim
        self.num_tab_columns = num_tab_columns

        # Patchify
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.num_patches = self.x_embedder.num_patches
        self.base_size = input_size // patch_size

        # Timestep embed
        self.t_embedder = TimestepEmbedder(hidden_size=dim, act_layer=nn.GELU)

        self.tab_transformer = TabTransformer(
            num_cols=num_tab_columns,
            dim=dim,
            head_dim=head_dim,
            mlp_ratio=4.0,  # or whatever ratio you like
            num_layers=2,  # or more if you want
            use_pos_embed=True,
            norm_eps=norm_eps,
        )

        # pos_embed
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, dim))

        # == 1) Patch Mixer
        if use_patch_mixer: # actually basically we are forced to use it at this point
            # figure out which patch mixer blocks are MoE: THIS IS NOT USED ANYMORE
            # pm_expert_blocks_idx = [i for i in range(patch_mixer_depth-1) if (i+1) % experts_every_n == 0]
            # pm_is_moe_block = [(i in pm_expert_blocks_idx) for i in range(patch_mixer_depth)]

            self.vit_patch_mixer = ViTPatchMixer(
                patch_mixer_dim=patch_mixer_dim,
                head_dim=head_dim,
                mlp_ratio=patch_mixer_mlp_ratio,
                qkv_ratio=patch_mixer_qkv_ratio,
                multiple_of=multiple_of,
                time_tab_dim=dim,  # or patch_mixer_dim if you prefer
                patch_mixer_depth=patch_mixer_depth,
                norm_eps=norm_eps,
                depth_init=False,  # not layering across entire net
                use_bias=use_bias
            )

            # If patch_mixer_dim != dim => linear in/out
            if patch_mixer_dim != dim:
                self.patch_mixer_map_xin = nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, patch_mixer_dim)
                ) if patch_mixer_dim != dim else nn.Identity()

                self.patch_mixer_map_xout = nn.Sequential(
                    nn.LayerNorm(patch_mixer_dim), nn.Linear(patch_mixer_dim, dim)
                ) if patch_mixer_dim != dim else nn.Identity()

                self.cond_map_down = (
                    nn.Linear(dim, patch_mixer_dim, bias=False)
                    if patch_mixer_dim != dim else nn.Identity()
                )
                self.tab_map_down = (
                    nn.Linear(dim, patch_mixer_dim, bias=False)
                    if patch_mixer_dim != dim else nn.Identity()
                )
            else:
                self.patch_mixer_map_xin = self.patch_mixer_map_xout = nn.Identity()
                self.cond_map_down = self.tab_map_down = nn.Identity()

        else:
            self.vit_patch_mixer = None
            self.patch_mixer_map_xin = nn.Identity()
            self.patch_mixer_map_xout = nn.Identity()

        # == 2) Main Blocks
        total_depth = depth
        # spread the multipliers
        if len(ffn_multipliers) == total_depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            num_splits = len(ffn_multipliers)
            assert total_depth % num_splits == 0
            dps = total_depth // num_splits
            qkv_ratios = list(np.concatenate([[m]*dps for m in qkv_multipliers]))
            mlp_ratios = list(np.concatenate([[m]*dps for m in ffn_multipliers]))

        # figure out which blocks are MoE in main
        expert_blocks_idx = [i for i in range(depth-1) if (i+1) % experts_every_n == 0]
        is_moe_block = [(i in expert_blocks_idx) for i in range(depth)]

        # BRANCH 1 IMG TO TAB
        self.blocks_imgtotab = nn.ModuleList()
        for i in range(depth):
            blk = MultiModalDiTBlockImgToTab(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratios[i],
                qkv_ratio=qkv_ratios[i],
                multiple_of=multiple_of,
                time_emb_dim=dim,
                tab_pooled_dim=dim,
                norm_eps=norm_eps,
                depth_init=depth_init,
                layer_id=i,
                num_layers=depth,
                use_bias=use_bias,
            )
            # If this block is MoE => we switch the MLP to FeedForwardECMoe
            if is_moe_block[i]:
                # manually replace the block.mlp with MoE:
                hidden_dim = int(dim * mlp_ratios[i])
                mlp_moe = FeedForwardECMoe(num_experts, expert_capacity, dim, hidden_dim, multiple_of)
                blk.mlp = mlp_moe
            self.blocks_imgtotab.append(blk)

        # BRANCH 2 TAB TO IMG
        self.blocks_tabtoimg = nn.ModuleList()
        for i in range(depth):
            blk = MultiModalDiTBlockTabToImg(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratios[i],
                qkv_ratio=qkv_ratios[i],
                multiple_of=multiple_of,
                time_emb_dim=dim,
                img_pooled_dim=dim,
                norm_eps=norm_eps,
                depth_init=depth_init,
                layer_id=i,
                num_layers=depth,
                use_bias=use_bias,
            )
            # If this block is MoE => we switch the MLP to FeedForwardECMoe
            if is_moe_block[i]:
                # manually replace the block.mlp with MoE:
                hidden_dim = int(dim * mlp_ratios[i])
                mlp_moe = FeedForwardECMoe(num_experts, expert_capacity, dim, hidden_dim, multiple_of)
                blk.mlp = mlp_moe
            self.blocks_tabtoimg.append(blk)

        # final image
        self.final_img = FinalLayer(
            in_dim=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_chans=self.out_channels,
            act_layer=nn.GELU,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps),
        )

        self.final_tab = FinalTabHead(
            in_dim=dim,
            time_emb_dim=dim,
            num_cols=157,
            act_layer=nn.GELU,
            norm_eps=norm_eps
        )

        # mask token
        self.register_buffer("mask_token", torch.zeros(1, 1, patch_size**2*self.out_channels))

        self.initialize_weights()

    def initialize_weights(self):  # overwritten (was original idx 1090‑something)
        """Initialise & then **zero‑gate** cross‑attention AdaLNs."""
        def zeros_bias(m):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        def basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                zeros_bias(m)
        self.apply(basic_init)

        # sinusoidal pos‑embed on correct device (fix‑15) --------------------
        side = int(self.num_patches ** 0.5)
        pe   = torch.from_numpy(self.get_2d_sincos_pe(side, self.dim, base_size=self.base_size)).float()
        self.pos_embed.data.copy_(pe.unsqueeze(0).to(self.pos_embed.device))

        # patch‑embed conv weight already init‑ed in PatchEmbed.reset_parameters

        # patch‑mixer & main blocks init … same as before
        if hasattr(self, "vit_patch_mixer") and self.vit_patch_mixer is not None:
            self.vit_patch_mixer.custom_init()
        for blk in getattr(self, "blocks_imgtotab", []):
            blk.custom_init(); _zero_gate(blk, "adaLN_modulation_x")
        for blk in getattr(self, "blocks_tabtoimg", []):
            blk.custom_init(); _zero_gate(blk, "adaLN_modulation_t")

        # final image head stays non‑zero‑gated on purpose

    def forward(
            self,
            x_img: torch.Tensor,  # shape (B, in_channels, H, W)
            t: torch.Tensor,  # shape (B,) timesteps
            x_tab: torch.Tensor = None,  # shape (B, num_tab_columns) or None
            cfg: float = 1.0,  # guidance scale
            mask_ratio: float = 0.0
    ):
        """
        If tab is None => unconditional generation (the model sees zero tab embeddings).
        If cfg == 1.0 => normal forward (purely cond or purely uncond).
        If cfg > 1.0 => do classifier-free guidance mixing (cond vs. uncond).
        """

        if cfg == 1.0:
            return self.forward_without_cfg(x_img, t, x_tab, mask_ratio=mask_ratio)
        else:
            # Do the standard classifier-free guidance approach:
            return self.forward_with_cfg(x_img, t, x_tab, cfg=cfg, mask_ratio=mask_ratio)

    def forward_without_cfg(
            self,
            x_img: torch.Tensor,
            t: torch.Tensor,
            x_tab: torch.Tensor,
            mask_ratio: float = 0.0
    ):

        # T Embed
        t_emb = self.t_embedder(t)

        # 1) Patchify
        img_tokens = self.x_embedder(x_img) + self.pos_embed

        # 2) tab branch -------------------------------------------------------
        tab_tokens, tab_cls = self.tab_transformer(x_tab)
        tab_kv = tab_tokens[:, 1:, :]  # FIX‑5 (exclude CLS)
        cond_emb = t_emb + tab_cls

        if self.vit_patch_mixer is not None:
            # map image tokens to patch_mixer_dim
            img_tokens_small = self.patch_mixer_map_xin(img_tokens)
            tab_tokens_small = self.tab_map_down(tab_kv)
            cond_emb_small = self.cond_map_down(cond_emb)

            # also map tab_tokens to patch_mixer_dim for cross-attn if needed
            # but note your TabTransformer returns (B,1+num_cols,dim). We only need the column tokens for cross-attn:
            # removing the [CLS], or keep it if you want. Typically we do keep them if we want read-only.
            # let's do a quick linear map:
            # if self.patch_mixer_map_xin is not None and self.patch_mixer_dim != self.dim:
            #     tab_tokens_small = self.patch_mixer_map_xin(tab_tokens)  # shape => (B, 1+num_cols, patch_mixer_dim)
            #     cond_emb_small = self.patch_mixer_map_xin(cond_emb)
            # else:
            #     tab_tokens_small = tab_tokens
            #     cond_emb_small = cond_emb

            # pass to vit_patch_mixer
            img_tokens_small, img_cls_small = self.vit_patch_mixer(
                img_tokens_small, tab_tokens_small, cond_emb_small
            )
            img_tokens = self.patch_mixer_map_xout(img_tokens_small)
            img_cls = self.patch_mixer_map_xout(img_cls_small)
        else:
            img_cls = img_tokens.mean(1)  # cheap pooled rep

        tab_pooled = tab_cls
        img_pooled = img_cls

        # 4) Optional masking
        mask, ids_restore = None, None
        if mask_ratio > 0.0:
            B, T_img, D = img_tokens.shape
            info = get_mask(B, T_img, mask_ratio, x_img.device)
            img_tokens = mask_out_token(img_tokens, info['ids_keep'])
            mask, ids_restore = info['mask'], info['ids_restore']

        # inter‑leaved main blocks
        for blk_img, blk_tab in zip(self.blocks_imgtotab, self.blocks_tabtoimg):
            img_tokens = blk_img(img_tokens, tab_tokens[:, 1:, :], t_emb, tab_pooled)
            tab_tokens = blk_tab(tab_tokens, img_tokens, t_emb, img_pooled)

        img_logits = self.final_img(img_tokens, t_emb)
        if (mask_ratio > 0.0) and (ids_restore is not None):
            img_logits = unmask_tokens(img_logits, ids_restore, self.mask_token)
        img_out = self.unpatchify(img_logits)


        tab_out = self.final_tab(tab_tokens, t_emb)

        return {
            "image_sample": img_out,  # (B, C, H, W)
            "tab_sample": tab_out,
            "mask": mask
        }

    def forward_with_cfg(
            self,
            x_img: torch.Tensor,
            t: torch.Tensor,
            x_tab: torch.Tensor,
            cfg: float,
            mask_ratio: float = 0.0
    ):
        """
        Classifier-free guidance approach:
         1) We replicate the batch => 2B
         2) The first half uses real tab, the second half uses zeros
         3) Pass them through forward_without_cfg
         4) Split outputs => combine => final
        """
        B = x_img.shape[0]

        # concat images => shape (2B, C, H, W)
        x_img_cat = torch.cat([x_img, x_img], dim=0)

        # concat tab => shape (2B, num_cols), second half = zeros
        zeros_tab = torch.zeros_like(x_tab)
        tab_cat = torch.cat([x_tab, zeros_tab], dim=0)

        # if t has shape (B, ), replicate => (2B, )
        # if t.shape[0] != 1:
        t = torch.cat([t, t], dim=0)  # (2B,)

        # single pass with the expanded batch => (2B, ...)
        out_cat = self.forward_without_cfg(
            x_img_cat,
            t,
            tab_cat,
            mask_ratio=mask_ratio
        )
        # out_cat => dict with image_sample => (2B, C,H,W)

        # split
        image_sample_cat = out_cat['image_sample']  # (2B, C,H,W)
        cond_img, uncond_img = torch.split(image_sample_cat, B, dim=0)

        # combine => uncond + cfg*(cond - uncond)
        final_img = uncond_img + cfg * (cond_img - uncond_img)

        return {
            "image_sample": final_img,
            "mask": None
        }

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, patch_dim = x.shape
        p = self.patch_size
        c = self.out_channels
        h = w = int(T**0.5)
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h*p, w*p)
        return x

    @staticmethod
    def get_2d_sincos_pe(grid_size, embed_dim, base_size=16, pos_interp_scale=1.0):
        """
        Minimal utility to generate 2D sin-cos embedding
        """
        def get_1d_sin_cos(pos, emb_dim):
            half_dim = emb_dim//2
            omega = 1. / (10000**(np.arange(half_dim)/half_dim))
            out = np.einsum('m,d->md', pos, omega)
            emb_sin = np.sin(out)
            emb_cos = np.cos(out)
            return np.concatenate([emb_sin, emb_cos], axis=1)

        h = np.arange(grid_size, dtype=np.float32)/(grid_size/base_size)/pos_interp_scale
        w = np.arange(grid_size, dtype=np.float32)/(grid_size/base_size)/pos_interp_scale
        ww, hh = np.meshgrid(w,h)
        ww = ww.reshape(-1)
        hh = hh.reshape(-1)
        assert embed_dim%2==0
        emb_h = get_1d_sin_cos(hh, embed_dim//2)
        emb_w = get_1d_sin_cos(ww, embed_dim//2)
        return np.concatenate([emb_h, emb_w], axis=1)


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
    if "dit" in cfg:
        model = MultiModalDiT(**cfg.dit)
    else:
        model = MultiModalDiT(**cfg)
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
        dim=512,
        depth=depth,
        head_dim=16,
        multiple_of=64,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
        num_tab_columns=157,
        tab_groups=10,
        out_table_features=157
    )

    # Fake data

    N = 2
    x_img = torch.randn(N, 4, 32, 32)  # e.g. 2 images, 3 channels
    tab = torch.randn(N, 157)
    t = torch.randint(0, 1000, (N,))  # random timesteps


    # 3) Forward pass
    res = model(x_img, t, tab, mask_ratio=0.2, cfg=1)
    print("img_out shape:", res["image_sample"].shape)  # (N, 3, 64, 64)
    print("tab_out shape:", res["tab_sample"].shape)  # (N, 174)

