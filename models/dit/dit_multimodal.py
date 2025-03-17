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
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=init_std)
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
        self.num_patches = (img_size // patch_size)*(img_size // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1,2)  # => (B, T, embed_dim)
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


################################################################################
# The DiTBlock with Cross-Attn but 6×dim for MSA + MLP only
################################################################################

class MultiModalDiTBlock(nn.Module):
    """
    - Self-attn with gating
    - Cross-attn (NO gating)
    - MLP with gating
    - can do MoE
    - supports classifier-free guidance by duplicating the batch if cfg>1.0
    => 6 * dim for shift/scale/gate: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    """
    def __init__(
        self,
        dim,
        head_dim,
        mlp_ratio,
        qkv_ratio,
        multiple_of,
        time_emb_dim,
        tab_emb_dim,
        norm_eps,
        depth_init,
        layer_id,
        num_layers,
        use_bias,
    ):
        super().__init__()
        self.dim = dim
        # QKV dims
        qkv_hidden_dim = (
            (head_dim*2)*((int(dim*qkv_ratio) + head_dim*2 -1)//(head_dim*2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim*mlp_ratio)

        # 1) Self-Attn
        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim//head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        # 2) Cross-Attn
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim//head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        # 3) MLP
        self.norm3 = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # We want 6×dim for MSA + MLP gating => (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim + tab_emb_dim, 6*dim)
        )

        if depth_init:
            # like microdiffusion: scaled init by block index
            self.weight_init_std = 0.02 / (2*(layer_id+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*num_layers)**0.5

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, tab_pooled: torch.Tensor, tab_tokens: torch.Tensor):
        """
        x:         (B, T_img, dim)   -> image tokens
        t_emb:     (B, time_emb_dim)
        tab_pooled (B, tab_emb_dim)  -> single vector for AdaLN
        tab_tokens (B, T_tab, dim)   -> used in cross-attn
        """
        combined = torch.cat([t_emb, tab_pooled], dim=1)  # (B, time_emb_dim + tab_emb_dim)
        # chunk => 6 * dim
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(combined).chunk(6, dim=1)

        # 1) Self Attn
        x_ln = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1)*self.attn(x_ln)

        # 2) Cross Attn (NO gating or shift/scale for cross-attn in this example)
        x_ln2 = self.norm2(x)
        x = x + self.cross_attn(x_ln2, tab_tokens)

        # 3) MLP
        x_ln3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1)*self.mlp(x_ln3)

        return x

    def custom_init(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.cross_attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)


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
        num_tab_columns=10,
        tab_groups=3,
        out_table_features=10,
        # Patch mixer
        use_patch_mixer=False,
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

        # Tab grouping
        self.table_proj = TabularProjection(
            in_features=num_tab_columns,
            hidden_size=dim,
            groups=tab_groups
        )
        # small preproc
        self.table_preproc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # pooled
        self.table_pooled_process = Mlp(in_features=dim, hidden_features=dim, out_features=dim)

        # pos_embed
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, dim))

        # == 1) Optional Patch Mixer
        if use_patch_mixer:
            # figure out which patch mixer blocks are MoE
            pm_expert_blocks_idx = [i for i in range(patch_mixer_depth-1) if (i+1) % experts_every_n == 0]
            pm_is_moe_block = [(i in pm_expert_blocks_idx) for i in range(patch_mixer_depth)]

            self.patch_mixer = nn.ModuleList([
                PatchMixerBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    time_emb_dim=dim,    # or patch_mixer_dim, but let's keep it simple
                    norm_eps=norm_eps,
                    depth_init=False,    # not layering across entire net
                    layer_id=i,
                    num_layers=patch_mixer_depth,
                    use_bias=use_bias,
                    moe_block=pm_is_moe_block[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity
                )
                for i in range(patch_mixer_depth)
            ])
            # If patch_mixer_dim != dim => linear in/out
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
        else:
            self.patch_mixer = None

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

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = MultiModalDiTBlock(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratios[i],
                qkv_ratio=qkv_ratios[i],
                multiple_of=multiple_of,
                time_emb_dim=dim,
                tab_emb_dim=dim,
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
            self.blocks.append(blk)

        # final image
        self.final_img = FinalLayer(
            in_dim=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_chans=self.out_channels,
            act_layer=nn.GELU,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps),
        )

        # final tab => for row reconstruction
        self.final_tab = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_table_features)
        )

        # mask token
        self.register_buffer("mask_token", torch.zeros(1, 1, patch_size**2*self.out_channels))

        self.initialize_weights()

    def initialize_weights(self):
        def zero_bias(m):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                zero_bias(module)
        self.apply(_basic_init)

        # sincos pos embed
        side = int(self.num_patches**0.5)
        pe = self.get_2d_sincos_pe(side, self.dim, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # conv init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # patch_mixer init
        if self.patch_mixer:
            for block in self.patch_mixer:
                block.custom_init()

        # main blocks init
        for block in self.blocks:
            block.custom_init()

        # zero out final linear
        nn.init.constant_(self.final_img.linear.weight, 0)
        nn.init.constant_(self.final_img.adaLN_modulation[-1].weight, 0)

    def forward(
            self,
            x_img: torch.Tensor,  # shape (B, in_channels, H, W)
            t: torch.Tensor,  # shape (B,) timesteps
            tab: torch.Tensor = None,  # shape (B, num_tab_columns) or None
            cfg: float = 1.0,  # guidance scale
            mask_ratio: float = 0.0
    ):
        """
        If tab is None => unconditional generation (the model sees zero tab embeddings).
        If cfg == 1.0 => normal forward (purely cond or purely uncond).
        If cfg > 1.0 => do classifier-free guidance mixing (cond vs. uncond).
        """

        if cfg == 1.0:
            return self.forward_without_cfg(x_img, t, tab, mask_ratio=mask_ratio)
        else:
            # Do the standard classifier-free guidance approach:
            return self.forward_with_cfg(x_img, t, tab, cfg=cfg, mask_ratio=mask_ratio)

    def forward_without_cfg(
            self,
            x_img: torch.Tensor,
            t: torch.Tensor,
            tab: torch.Tensor,
            mask_ratio: float = 0.0
    ):
        # 1) Patchify
        x = self.x_embedder(x_img)  # => (B, T, dim)
        x = x + self.pos_embed

        # 2) T Embed
        t_emb = self.t_embedder(t)

        # 3) Tab => grouping => preproc => pooled
        tab_tokens = self.table_proj(tab)
        tab_tokens = self.table_preproc(tab_tokens)
        tab_pooled = tab_tokens.mean(dim=1)
        tab_pooled = self.table_pooled_process(tab_pooled)

        # 4) patch mixer if any
        if self.patch_mixer:
            x = self.patch_mixer_map_xin(x)
            for blk in self.patch_mixer:
                x = blk(x, t_emb)
            x = self.patch_mixer_map_xout(x)

        # 5) optional masking
        mask = None
        ids_restore = None
        if mask_ratio > 0.0:
            B, T_img, D = x.shape
            mask_info = get_mask(B, T_img, mask_ratio, x.device)
            x = mask_out_token(x, mask_info['ids_keep'])
            mask = mask_info['mask']
            ids_restore = mask_info['ids_restore']

        # 6) main blocks
        for blk in self.blocks:
            # each block => x=blk(x, t_emb, tab_pooled, tab_tokens)
            x = blk(x, t_emb, tab_pooled, tab_tokens)

        # 7) final image
        img_logits = self.final_img(x, t_emb)
        if mask_ratio > 0.0 and ids_restore is not None:
            img_logits = unmask_tokens(img_logits, ids_restore, self.mask_token)
        img_out = self.unpatchify(img_logits)

        return {
            "image_sample": img_out,  # (B, C, H, W)
            "mask": mask
        }

    def forward_with_cfg(
            self,
            x_img: torch.Tensor,
            t: torch.Tensor,
            tab: torch.Tensor,
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
        zeros_tab = torch.zeros_like(tab)
        tab_cat = torch.cat([tab, zeros_tab], dim=0)

        # if t has shape (B, ), replicate => (2B, )
        # if t.ndim == 1 and t.shape[0] == B:
        if len(t) != 1:
            t = torch.cat([t, t], dim=0)

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
    res = model(x_img, t, tab, mask_ratio=0.2, cfg=1.2)
    print("img_out shape:", res["image_sample"].shape)  # (N, 3, 64, 64)


