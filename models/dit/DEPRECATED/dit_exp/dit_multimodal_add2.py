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


################################################################################
# Basic Utilities
################################################################################

def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
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


################################################################################
# Patch Embedding + Patch Mixer
################################################################################

class PatchEmbed(nn.Module):
    """
    Patchify the image => (B, num_patches, dim).
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)  # => (B, T, embed_dim)
        return x


class PatchMixerBlock(nn.Module):
    """
    The patch mixer for image tokens, from Sony code: self-attn + MLP with gating.
    Possibly with MoE in the MLP if moe_block=True.
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
        moe_block: bool,
        num_experts: int,
        expert_capacity: float,
    ):
        super().__init__()
        # QKV dimension rounding
        qkv_hidden_dim = (
            (head_dim*2)*((int(dim*qkv_ratio) + head_dim*2 -1)//(head_dim*2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim*mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias, hidden_dim=qkv_hidden_dim)

        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        if moe_block:
            self.mlp = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # AdaLN gating => 6×dim
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim)
        )

        if depth_init:
            self.weight_init_std = 0.02 / (2*(layer_id+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*num_layers)**0.5

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Self Attn
        x_ln = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1)*self.attn(x_ln)

        # MLP
        x_ln2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1)*self.mlp(x_ln2)

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


class ImageAttnPooler(nn.Module):
    """
    A simple attention-based pooling for image tokens:
      - We have a learned 'cls_token' (1,1,dim).
      - We do cross-attention from that token to the image tokens.
      - The output is a single (B, dim) vector, analogous to "pooled" representation.
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, T_img, dim) image patch tokens
        Returns:
          (B, dim) a single pooled vector
        """
        B, _, _ = x.shape
        x = self.norm(x)  # optional LN before pooling
        cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,dim)

        # Cross-attn: queries = cls_token, keys/values = x
        out = self.attn(cls_token, x)  # => (B, 1, dim)
        return out.squeeze(1)          # => (B, dim)

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.01)
        # Optionally init cross-attn:
        self.attn.custom_init(init_std)


################################################################################
# Attention Modules
################################################################################

class SelfAttention(nn.Module):
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


class CrossAttention(nn.Module):
    """
    Cross-attn: queries from x, keys/values from cond.
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
        B, T_x, _ = x.shape
        T_cond = cond.shape[1]

        q = self.q_linear(x)
        kv = self.kv_linear(cond)
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, T_x, self.num_heads, self.head_dim).transpose(1,2)
        k = k.reshape(B, T_cond, self.num_heads, self.head_dim).transpose(1,2)
        v = v.reshape(B, T_cond, self.num_heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, T_x, self.hidden_dim)
        out = self.proj(out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.q_linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.kv_linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


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
# Final Heads
################################################################################

class FinalImageHead(nn.Module):
    """
    Projects from tokens => patch^2*C, then unpatchify.
    Uses a small AdaLN with t_emb for shift/scale.
    """
    def __init__(self, in_dim, time_emb_dim, patch_size, out_chans, act_layer, norm_eps):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2*in_dim)  # shift + scale
        )
        self.linear = nn.Linear(in_dim, patch_size*patch_size*out_chans)
        self.patch_size = patch_size
        self.out_chans = out_chans

    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class FinalTabHead(nn.Module):
    """
    For purely numeric tab data:
      1) Mean‐pool the tokens => shape (B, in_dim).
      2) Apply a layer norm.
      3) Modulate by time embedding via shift & scale (AdaLN).
      4) Feed into a final linear projection => shape (B, out_dim).
    """
    def __init__(self, in_dim, time_emb_dim, out_dim,
                 act_layer=nn.SiLU, norm_eps=1e-6):
        super().__init__()
        # Final LN (or rename to self.norm if you prefer)
        self.norm_final = create_norm("layernorm", in_dim, eps=norm_eps)

        # A simple MLP that maps time_emb_dim -> 2*in_dim => [shift, scale]
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2 * in_dim)
        )

        # Final linear to go from in_dim -> out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, t_emb):
        """
        x: shape (B, N, in_dim)  -- token embeddings
        t_emb: shape (B, time_emb_dim) -- time or latent embedding
        """
        # 1) Mean‐pool over the token dimension
        x_pooled = x.mean(dim=1)  # shape (B, in_dim)

        # 2) Layer norm
        x_normed = self.norm_final(x_pooled)  # shape (B, in_dim)

        # 3) AdaLN shift & scale
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)  # each (B, in_dim)
        x_mod = x_normed * (1.0 + scale) + shift  # shape (B, in_dim)

        # 4) Final linear projection
        out_tab = self.linear(x_mod)  # shape (B, out_dim)
        return out_tab


################################################################################
# A Simple TabTokenizer (numeric only)
################################################################################

class SimpleTabTokenizer(nn.Module):
    """
    For purely numeric columns => one embedding weight per numeric column (+ optional bias).
    We'll produce tokens of dimension 'dim' to match the image tokens.
    """
    def __init__(self, d_numerical, dim, use_bias=True, use_cls=False):
        super().__init__()
        self.d_numerical = d_numerical
        self.dim = dim
        self.use_cls = use_cls

        # e.g., if use_cls => d_numerical+1 rows, else => d_numerical
        nrows = d_numerical + (1 if use_cls else 0)
        self.weight = nn.Parameter(torch.Tensor(nrows, dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(nrows, dim))
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        else:
            self.bias = None

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B, d_numerical)
        Return shape: (B, T_tab, dim)
          where T_tab = d_numerical (+1 if use_cls).
        """
        B = x_num.shape[0]
        if self.use_cls:
            ones = torch.ones(B, 1, device=x_num.device, dtype=x_num.dtype)
            x_num = torch.cat([ones, x_num], dim=1)  # => shape (B, d_numerical+1)

        tokens = x_num.unsqueeze(-1) * self.weight.unsqueeze(0)  # (B, #cols, dim)
        if self.bias is not None:
            tokens = tokens + self.bias.unsqueeze(0)
        return tokens


################################################################################
# The symmetrical multi-modal block
################################################################################

class MultiModalDiTBlock(nn.Module):
    """
    A symmetrical cross-attention block that takes:
      - x_img (B, T_img, dim)
      - img_pooled (B, img_pooled_dim)
      - x_tab (B, T_tab, dim)
      - tab_pooled (B, tab_pooled_dim)
      - t_emb (B, time_emb_dim)

    Then, each modality sees the other's original representation during cross-attn.
    Each side also uses t_emb + the OTHER side's pooled embedding for gating.
    """

    def __init__(
        self,
        dim,            # dimension for BOTH image tokens and tab tokens
        head_dim,
        mlp_ratio,
        qkv_ratio,
        multiple_of,
        time_emb_dim,
        img_pooled_dim,  # dimension of img_pooled
        tab_pooled_dim,  # dimension of tab_pooled
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
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim*2 - 1) // (head_dim * 2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        # --------------------------------------------------------------------
        # IMAGE BRANCH
        # --------------------------------------------------------------------
        self.norm1_x = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_x = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        self.norm2_x = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_x_to_tab = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        self.norm3_x = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp_x = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # Gating for the image side => 6×dim
        # The gating input for image side = [t_emb, tab_pooled]
        self.adaLN_modulation_x = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim + tab_pooled_dim, 6 * dim)
        )

        # --------------------------------------------------------------------
        # TAB BRANCH
        # --------------------------------------------------------------------
        self.norm1_t = create_norm('layernorm', dim, eps=norm_eps)
        self.attn_t = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        self.norm2_t = create_norm('layernorm', dim, eps=norm_eps)
        self.cross_attn_t_to_img = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        self.norm3_t = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp_t = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # Gating for the tab side => 6×dim
        # The gating input for tab side = [t_emb, img_pooled]
        self.adaLN_modulation_t = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim + img_pooled_dim, 6 * dim)
        )

        # --------------------------------------------------------------------
        # Weight init factor
        # --------------------------------------------------------------------
        if depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * num_layers) ** 0.5

    def forward(
        self,
        x_img: torch.Tensor,        # (B, T_img, dim)
        img_pooled: torch.Tensor,   # (B, img_pooled_dim)
        x_tab: torch.Tensor,        # (B, T_tab, dim)
        tab_pooled: torch.Tensor,   # (B, tab_pooled_dim)
        t_emb: torch.Tensor         # (B, time_emb_dim)
    ):
        # Store the original tokens for cross-attn
        x_img_orig = x_img
        x_tab_orig = x_tab

        # --------------------------------------------------------------------
        # IMAGE BRANCH
        # gating input => cat([t_emb, tab_pooled]) => 6×dim
        # --------------------------------------------------------------------
        combined_img = torch.cat([t_emb, tab_pooled], dim=1)  # => (B, time_emb_dim + tab_pooled_dim)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = \
            self.adaLN_modulation_x(combined_img).chunk(6, dim=1)

        # (1) Self-Attn on image
        x_ln = modulate(self.norm1_x(x_img), shift_msa_x, scale_msa_x)
        x_img = x_img + gate_msa_x.unsqueeze(1) * self.attn_x(x_ln)

        # (2) Cross-Attn from image -> tab
        x_ln2 = self.norm2_x(x_img)
        x_img = x_img + self.cross_attn_x_to_tab(x_ln2, x_tab_orig)

        # (3) MLP
        x_ln3 = modulate(self.norm3_x(x_img), shift_mlp_x, scale_mlp_x)
        x_img = x_img + gate_mlp_x.unsqueeze(1) * self.mlp_x(x_ln3)

        # --------------------------------------------------------------------
        # TAB BRANCH
        # gating input => cat([t_emb, img_pooled]) => 6×dim
        # --------------------------------------------------------------------
        combined_tab = torch.cat([t_emb, img_pooled], dim=1)  # => (B, time_emb_dim + img_pooled_dim)
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation_t(combined_tab).chunk(6, dim=1)

        # (1) Self-Attn on tab
        t_ln = modulate(self.norm1_t(x_tab), shift_msa_t, scale_msa_t)
        x_tab = x_tab + gate_msa_t.unsqueeze(1) * self.attn_t(t_ln)

        # (2) Cross-Attn from tab -> image
        t_ln2 = self.norm2_t(x_tab)
        x_tab = x_tab + self.cross_attn_t_to_img(t_ln2, x_img_orig)

        # (3) MLP
        t_ln3 = modulate(self.norm3_t(x_tab), shift_mlp_t, scale_mlp_t)
        x_tab = x_tab + gate_mlp_t.unsqueeze(1) * self.mlp_t(t_ln3)

        return x_img, x_tab

    def custom_init(self):
        self.norm1_x.reset_parameters()
        self.norm2_x.reset_parameters()
        self.norm3_x.reset_parameters()
        self.attn_x.custom_init(self.weight_init_std)
        self.cross_attn_x_to_tab.custom_init(self.weight_init_std)
        self.mlp_x.custom_init(self.weight_init_std)

        self.norm1_t.reset_parameters()
        self.norm2_t.reset_parameters()
        self.norm3_t.reset_parameters()
        self.attn_t.custom_init(self.weight_init_std)
        self.cross_attn_t_to_img.custom_init(self.weight_init_std)
        self.mlp_t.custom_init(self.weight_init_std)


################################################################################
# The final MultiModal DiT
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
            # Image
            img_size=64,
            patch_size=4,
            in_channels=3,
            dim=256,
            patch_mixer_depth=2,
            patch_mixer_dim=256,
            patch_mixer_qkv_ratio=1.0,
            patch_mixer_mlp_ratio=4.0,
            # Tab
            d_numerical=10,
            out_dim_tab=10,
            use_cls_tab=False,
            # Main blocks
            depth=6,
            head_dim=64,
            qkv_multipliers=[1.0],
            ffn_multipliers=[4.0],
            multiple_of=256,
            time_emb_dim=256,
            norm_eps=1e-6,
            use_bias=True,
            # MoE
            experts_every_n=2,
            num_experts=4,
            expert_capacity=1.0,
            # Patch Mixer => optional MoE
            use_patch_mixer=False,
            patch_mixer_use_moe=True,
            # Some init scaling
            depth_init=True
    ):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # 1) Image Patch Embedding
        self.x_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.num_patches = self.x_embedder.num_patches

        # 2) Tab Tokenizer
        self.tab_tokenizer = SimpleTabTokenizer(
            d_numerical=d_numerical,
            dim=dim,
            use_bias=True,
            use_cls=use_cls_tab
        )

        # 3) Positional Embeddings for image
        self.register_buffer("pos_embed_img", torch.zeros(1, self.num_patches, dim))

        # 4) Timestep Embedding
        self.t_embedder = TimestepEmbedder(hidden_size=time_emb_dim)

        # 5) Pooling image and tabular
        self.image_pooler = ImageAttnPooler(dim=dim, num_heads=(dim // head_dim), qkv_bias=use_bias)
        self.table_preproc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.table_pooled_process = FeedForward(dim, dim, multiple_of=multiple_of, use_bias=use_bias)

        # 7) Optional Patch Mixer on image tokens
        if use_patch_mixer:
            pm_expert_idx = [i for i in range(patch_mixer_depth) if (i + 1) % experts_every_n == 0]
            pm_blocks = []
            for i in range(patch_mixer_depth):
                is_moe = (i in pm_expert_idx) and patch_mixer_use_moe
                blk = PatchMixerBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    time_emb_dim=time_emb_dim,
                    norm_eps=norm_eps,
                    depth_init=False,
                    layer_id=i,
                    num_layers=patch_mixer_depth,
                    use_bias=use_bias,
                    moe_block=is_moe,
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                )
                pm_blocks.append(blk)
            self.patch_mixer = nn.ModuleList(pm_blocks)

            if patch_mixer_dim != dim:
                self.patch_mixer_map_in = nn.Sequential(
                    nn.LayerNorm(dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
                self.patch_mixer_map_out = nn.Sequential(
                    nn.LayerNorm(patch_mixer_dim, eps=norm_eps),
                    nn.Linear(patch_mixer_dim, dim, bias=use_bias)
                )
            else:
                self.patch_mixer_map_in = nn.Identity()
                self.patch_mixer_map_out = nn.Identity()
        else:
            self.patch_mixer = None

        # 8) Main Blocks
        total_depth = depth
        if len(ffn_multipliers) == total_depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            # replicate the sequence
            num_splits = len(ffn_multipliers)
            assert total_depth % num_splits == 0
            dps = total_depth // num_splits
            qkv_ratios = list(np.concatenate([[m] * dps for m in qkv_multipliers]))
            mlp_ratios = list(np.concatenate([[m] * dps for m in ffn_multipliers]))

        expert_blocks_idx = [i for i in range(depth) if (i + 1) % experts_every_n == 0]
        is_moe_block = [(i in expert_blocks_idx) for i in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = MultiModalDiTBlock(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratios[i],
                qkv_ratio=qkv_ratios[i],
                multiple_of=multiple_of,
                time_emb_dim=time_emb_dim,
                img_pooled_dim=dim,
                tab_pooled_dim=dim,
                norm_eps=norm_eps,
                depth_init=depth_init,
                layer_id=i,
                num_layers=depth,
                use_bias=use_bias
            )
            # If this block is MoE => replace block MLPs with MoE:
            if is_moe_block[i]:
                hidden_dim = int(dim * mlp_ratios[i])
                blk.mlp_x = FeedForwardECMoe(num_experts, expert_capacity, dim, hidden_dim, multiple_of)
                blk.mlp_t = FeedForwardECMoe(num_experts, expert_capacity, dim, hidden_dim, multiple_of)

            self.blocks.append(blk)

        # 9) Final heads (created once!)
        self.final_img = FinalImageHead(
            in_dim=dim,
            time_emb_dim=time_emb_dim,
            patch_size=patch_size,
            out_chans=in_channels,
            act_layer=nn.GELU,
            norm_eps=norm_eps
        )
        self.final_tab = FinalTabHead(
            in_dim=dim,
            time_emb_dim=time_emb_dim,
            out_dim=out_dim_tab,
            act_layer=nn.GELU,
            norm_eps=norm_eps
        )

        # a single mask token used for masked patches
        self.register_buffer("mask_token", torch.zeros(1, 1, patch_size * patch_size * in_channels))

        # Initialize
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

        # sincos pos embed for image
        side = int(self.num_patches**0.5)
        pe_img = self.get_2d_sincos_pe(side, self.dim, base_size=(self.img_size//self.patch_size))
        self.pos_embed_img.data.copy_(torch.from_numpy(pe_img).float().unsqueeze(0))

        # patch mixer init
        if self.patch_mixer is not None:
            for blk in self.patch_mixer:
                blk.custom_init()

        # main blocks init
        for blk in self.blocks:
            blk.custom_init()

        # image_pooler init
        self.image_pooler.custom_init(0.01)

        # zero out final heads’ linear layers
        nn.init.constant_(self.final_img.linear.weight, 0)
        nn.init.constant_(self.final_tab.linear.weight, 0)
        nn.init.constant_(self.final_img.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_tab.adaLN_modulation[-1].weight, 0)

    def forward(
            self,
            x_img: torch.Tensor,
            t: torch.Tensor,
            x_tab: Optional[torch.Tensor] = None,
            cfg: float = 1.0,
            mask_ratio: float = 0.0
    ):
        if cfg == 1.0:
            return self.forward_without_cfg(x_img, t, x_tab, mask_ratio)
        else:
            return self.forward_with_cfg(x_img, t, x_tab, cfg, mask_ratio)

    def forward_without_cfg(
            self,
            x_img: torch.Tensor,
            t: torch.Tensor,
            x_tab: torch.Tensor,
            mask_ratio: float = 0.0
    ):

        t_emb = self.t_embedder(t)

        # 1) Patchify image => (B, T_img, dim), add pos embed
        img_tokens = self.x_embedder(x_img)
        img_tokens = img_tokens + self.pos_embed_img[:, :img_tokens.shape[1], :]

        # 2) (Optional) patch mixer
        if self.patch_mixer is not None:
            img_tokens = self.patch_mixer_map_in(img_tokens)
            for blk in self.patch_mixer:
                img_tokens = blk(img_tokens, t_emb)
            img_tokens = self.patch_mixer_map_out(img_tokens)

        # 3) Tab embed => (B, T_tab, dim)
        tab_tokens = self.tab_tokenizer(x_tab)
        tab_tokens = self.table_preproc(tab_tokens)
        tab_pooled = tab_tokens.mean(dim=1)
        tab_pooled = self.table_pooled_process(tab_pooled)

        # 4) Image attn‐pool => single vector
        img_pooled = self.image_pooler(img_tokens)  # (B, dim)

        # 5) optional masking
        mask = None
        ids_restore = None
        if mask_ratio > 0.0:
            B, T_img, D = img_tokens.shape
            info = get_mask(B, T_img, mask_ratio, x_img.device)
            img_tokens = mask_out_token(img_tokens, info['ids_keep'])
            mask = info['mask']
            ids_restore = info['ids_restore']

        # 6) main blocks => cross‐attn both ways
        for blk in self.blocks:
            img_tokens, tab_tokens = blk(
                x_img=img_tokens,
                img_pooled=img_pooled,
                x_tab=tab_tokens,
                tab_pooled=tab_pooled,
                t_emb=t_emb
            )

        # 7) Final heads => reconstruct image, tab
        img_logits = self.final_img(img_tokens, t_emb)
        if (mask_ratio > 0.0) and (ids_restore is not None):
            img_logits = unmask_tokens(img_logits, ids_restore, self.mask_token)
        img_out = self.unpatchify(img_logits)

        tab_out = self.final_tab(tab_tokens, t_emb)

        return {
            "image_sample": img_out,
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
        x_img_cat = torch.cat([x_img, x_img], dim=0)
        x_num_cat = torch.cat([x_tab, torch.zeros_like(x_tab)], dim=0)
        t_cat = torch.cat([t, t], dim=0)

        out_cat = self.forward_without_cfg(x_img_cat, t_cat, x_num_cat, mask_ratio)
        img_cat = out_cat["image_sample"]
        tab_cat = out_cat["tab_sample"]
        img_cond, img_uncond = torch.split(img_cat, B, dim=0)
        tab_cond, tab_uncond = torch.split(tab_cat, B, dim=0)

        img_final = img_uncond + cfg * (img_cond - img_uncond)
        tab_final = tab_uncond + cfg * (tab_cond - tab_uncond)

        return {
            "image_sample": img_final,
            "tab_sample": tab_final,
            "mask": None
        }

    @staticmethod
    def get_2d_sincos_pe(grid_size, embed_dim, base_size=16):
        def get_1d_sin_cos(pos, emb_dim):
            half_dim = emb_dim // 2
            omega = 1. / (10000 ** (torch.arange(0, half_dim) / half_dim))
            out = pos.unsqueeze(-1) * omega.unsqueeze(0)
            emb_sin = torch.sin(out)
            emb_cos = torch.cos(out)
            return torch.cat([emb_sin, emb_cos], dim=-1).cpu().numpy()

        h = torch.linspace(0, base_size, steps=grid_size)
        w = torch.linspace(0, base_size, steps=grid_size)
        ww, hh = torch.meshgrid(w, h, indexing='xy')
        ww, hh = ww.flatten(), hh.flatten()
        emb_h = get_1d_sin_cos(hh, embed_dim // 2)
        emb_w = get_1d_sin_cos(ww, embed_dim // 2)
        return np.concatenate([emb_h, emb_w], axis=1)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, patch_dim = x.shape
        p = self.patch_size
        c = self.in_channels
        h = w = int(T ** 0.5)
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h * p, w * p)
        return x


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
        img_size=64,
        patch_size=4,
        in_channels=4,
        dim=256,
        depth=depth,
        head_dim=32,
        multiple_of=64,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_use_moe=False,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        norm_eps=1e-6,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
        d_numerical=174,
        out_dim_tab=174,
        use_cls_tab=False,
        time_emb_dim=256
    )

    # Fake data

    N = 2
    x_img = torch.randn(N, 4, 64, 64)  # e.g. 2 images, 3 channels
    tab = torch.randn(N, 174)
    t = torch.randint(0, 1000, (N,))  # random timesteps


    # 3) Forward pass
    res = model(x_img, t, tab, mask_ratio=0.2, cfg=1.2)
    print("img_out shape:", res["image_sample"].shape)  # (N, 3, 64, 64)
    print("tab_out shape:", res["tab_sample"].shape)  # (N, 174)

