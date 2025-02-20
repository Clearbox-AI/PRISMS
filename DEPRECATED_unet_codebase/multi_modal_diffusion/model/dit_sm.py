import math
from collections.abc import Iterable
from itertools import repeat
from typing import Optional, Tuple, Dict, Union, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Basic Data Types (you can keep or remove depending on your usage)
################################################################################

DATA_TYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32
}

################################################################################
# Core Helpers
################################################################################

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies a learned shift & scale to `x` (e.g., in AdaLN):
       out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """
    MLP block from timm (without dropout).

    Args:
        in_features:  input dimension
        hidden_features: intermediate dim
        out_features: output dimension
        act_layer: activation constructor (defaults to GELU with tanh approx)
        norm_layer: optional normalization
        bias: whether linear layers use bias
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


def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Creates a normalization layer of the given type.
    Currently supports only "layernorm" or "np_layernorm".
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        # Same as layernorm but elementwise_affine=False
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')


################################################################################
# Positional Embeddings (2D Sin-Cos)
################################################################################

def ntuple(n: int):
    """Converts input into an n-tuple."""
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    cls_token: bool = False,
    extra_tokens: int = 0,
    pos_interp_scale: float = 1.0,
    base_size: int = 16
) -> np.ndarray:
    """
    Generates 2D sin-cos positional embeddings of shape (grid_size^2, embed_dim).
    """
    to_2tuple = ntuple(2)
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / pos_interp_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / pos_interp_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[1], grid_size[0]])  # (2,1,h,w)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        # Extra tokens (like [CLS], etc.)
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Splits the embed_dim in half for H and W, then merges."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Creates 1D sin-cos embeddings from pos. (pos is flattened)
    """
    assert embed_dim % 2 == 0
    pos = pos.reshape(-1)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / (10000 ** omega)
    out = np.einsum('m,d->md', pos, omega)  # outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


################################################################################
# Masking Utilities
################################################################################

def get_mask(batch: int, length: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Randomly selects 'mask_ratio' fraction of tokens to mask.
    Returns a dict with:
        - mask: shape (B, length), 0=keep, 1=masked
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore to original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # uniform in [0,1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend order => top are kept
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
    """
    Gathers only the kept tokens.
    x: (B, L, D)
    ids_keep: (B, L_keep)
    """
    B, L, D = x.shape
    index = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=index)


def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """
    Re-inserts masked tokens at their correct positions, filling them with mask_token.
    """
    B, L_keep, D = x.shape
    L = ids_restore.shape[1]
    mask_tokens = mask_token.repeat(B, L - L_keep, 1)
    # Append mask tokens
    x_ = torch.cat([x, mask_tokens], dim=1)
    # Unshuffle to original positions
    x_ = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
    )
    return x_

################################################################################
# Timestep Embedding for Diffusion
################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embed scalar timesteps into vector representations via sinusoidal frequencies + MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        act_layer: Any,
        frequency_embedding_size: int = 512
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Standard sinusoidal embedding for timesteps.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0,
                end=half,
                dtype=torch.float32,
                device=t.device
            ) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # zero-pad if dim is odd
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # If t is (B,) => produce frequency embedding => MLP => (B, hidden_size)
        freq_emb = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(freq_emb)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


################################################################################
# Distributed Loss Metric (optional)
################################################################################

class DistLoss(nn.Module):
    """
    Minimal distributed loss aggregator (like a TorchMetrics.Metric).
    Accumulates the total loss and number of batches, returns average.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("loss_sum", torch.zeros(1))
        self.register_buffer("batches", torch.zeros(1))

    def update(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self.loss_sum += value
            self.batches += 1

    def compute(self) -> torch.Tensor:
        if self.batches.item() == 0:
            return torch.tensor(0.0, device=self.loss_sum.device)
        return self.loss_sum / self.batches

    def reset(self) -> None:
        self.loss_sum.zero_()
        self.batches.zero_()

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, norm_eps=1e-6, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        hidden_dim = hidden_dim if hidden_dim else dim
        self.qkv = nn.Linear(dim, 3 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head
        q = q.reshape(B, T, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.proj(out)
        return out

    def custom_init(self, init_std):
        nn.init.trunc_normal_(self.qkv.weight, std=init_std)
        nn.init.trunc_normal_(self.proj.weight, std=init_std)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256, use_bias=True):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=use_bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def custom_init(self, init_std):
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

    def forward(self, x):
        B, T, D = x.shape
        tokens_per_expert = int(self.expert_capacity * T / self.num_experts)

        scores = self.gate(x)  # [B, T, E]
        probs = F.softmax(scores, dim=-1)
        g, m = torch.topk(probs.permute(0, 2, 1), tokens_per_expert, dim=-1)  # [B, E, k], [B, E, k]
        p = F.one_hot(m, num_classes=T).float()  # [B, E, k, T]

        xin = torch.einsum('bekt,btd->bekd', p, x)  # [B, E, k, D]
        h = torch.einsum('bekd,edh->bekh', xin, self.w1)  # [B, E, k, hidden_dim]
        h = self.gelu(h)
        h = torch.einsum('bekh,ehd->bekd', h, self.w2)  # [B, E, k, D]

        out = g.unsqueeze(dim=-1) * h
        out = torch.einsum('bekt,bekd->btd', p, out)
        return out

    def custom_init(self, init_std):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)

class PatchEmbed(nn.Module):
    """Simple patchify: Conv2d => flatten => (B,T,dim)."""
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        # x: (B,C,H,W)
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1,2)  # (B, T, embed_dim)
        return x

class FinalLayer(nn.Module):
    def __init__(self, in_dim, time_emb_dim, patch_size, out_chans, act_layer, norm_layer):
        super().__init__()
        self.norm = norm_layer
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2*in_dim, bias=True),
        )
        self.linear = nn.Linear(in_dim, patch_size * patch_size * out_chans)
        self.patch_size = patch_size
        self.out_chans = out_chans

    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)  # (B, T, p^2*out_chans)
        return x

class DiTBlock(nn.Module):
    """Single block: Self-Attn + (MoE or Dense) MLP + AdaLN with time embeddings."""
    def __init__(
        self,
        dim,
        head_dim,
        mlp_ratio,
        qkv_ratio,
        multiple_of,
        time_emb_dim,
        norm_eps,
        depth_init,
        layer_id,
        num_layers,
        use_bias,
        moe_block,
        num_experts,
        expert_capacity
    ):
        super().__init__()
        self.dim = dim
        qkv_hidden_dim = (
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim*2 -1)//(head_dim*2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim,
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        if moe_block:
            self.mlp = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim, bias=True),
        )
        if depth_init:
            self.weight_init_std = 0.02 / (2*(layer_id+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*num_layers)**0.5

    def forward(self, x, t_emb):
        # x: (B,T,dim), t_emb: (B, time_emb_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Self Attention
        x = x + gate_msa.unsqueeze(1)*self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MLP
        x = x + gate_mlp.unsqueeze(1)*self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def custom_init(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)

class DiT(nn.Module):
    """Unconditional DiT model: patchify image -> transformer -> unpatchify."""
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=512,
        depth=12,
        head_dim=64,
        multiple_of=256,
        pos_interp_scale=1.0,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=[1.0],
        ffn_multipliers=[4.0],
        use_patch_mixer=True,
        patch_mixer_depth=2,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
        use_bias=True,
        num_experts=8,
        expert_capacity=1.0,
        experts_every_n=2
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.head_dim = head_dim
        self.pos_interp_scale = pos_interp_scale
        self.use_patch_mixer = use_patch_mixer

        # Main modules
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim, nn.GELU)

        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // patch_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, dim))

        # Patch mixer
        if use_patch_mixer:
            pm_expert_blocks_idx = [
                i for i in range(patch_mixer_depth) if (i+1) % experts_every_n == 0
            ]
            is_moe_block = [i in pm_expert_blocks_idx for i in range(patch_mixer_depth)]
            self.patch_mixer = nn.ModuleList([
                DiTBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    time_emb_dim=dim,
                    norm_eps=norm_eps,
                    depth_init=False,
                    layer_id=0,
                    num_layers=depth,
                    use_bias=use_bias,
                    moe_block=is_moe_block[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity
                ) for i in range(patch_mixer_depth)
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
        else:
            self.patch_mixer = None

        # Distribute qkv/ffn multipliers over the blocks
        assert len(ffn_multipliers) == len(qkv_multipliers)
        if len(ffn_multipliers) == depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            num_splits = len(ffn_multipliers)
            assert depth % num_splits == 0
            dps = depth // num_splits
            qkv_ratios = list(np.concatenate([ [m]*dps for m in qkv_multipliers ]))
            mlp_ratios = list(np.concatenate([ [m]*dps for m in ffn_multipliers ]))

        # Identify MoE blocks among main blocks
        expert_blocks_idx = [i for i in range(depth - 1) if (i+1) % experts_every_n == 0]
        is_moe_block = [i in expert_blocks_idx for i in range(depth)]

        self.blocks = nn.ModuleList([
            DiTBlock(
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
                moe_block=is_moe_block[i],
                num_experts=num_experts,
                expert_capacity=expert_capacity
            )
            for i in range(depth)
        ])

        self.register_buffer(
            "mask_token",
            torch.zeros(1, 1, patch_size**2 * self.out_channels)
        )

        self.final_layer = FinalLayer(
            in_dim=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_chans=self.out_channels,
            act_layer=nn.GELU,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps),
        )
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
        # Sin-cos pos embed
        pe = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            pos_interp_scale=self.pos_interp_scale,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # custom init of blocks
        if self.patch_mixer:
            for block in self.patch_mixer:
                block.custom_init()
        for block in self.blocks:
            block.custom_init()

        # zero out final linear
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)

    def forward(self, x, t, mask_ratio=0.0, **kwargs):
        """
        x: (B, C, H, W)
        t: (B,) => must already be in the correct dimension for TimestepEmbedder,
                   or your TimestepEmbedder might do scalar->embedding inside.
        mask_ratio: fraction of patches to mask (for training)
        Returns: dict with 'sample' => (B, C, H, W) denoised image
        """

        if x.dtype != torch.float32:
            x = x.float()
        if t.dtype != torch.float32:
            t = t.float()

        # Patchify
        x = self.x_embedder(x)
        x = x + self.pos_embed  # add positional embedding

        # Timestep embed
        t_emb = self.t_embedder(t)

        # Patch mixer
        if self.use_patch_mixer:
            x = self.patch_mixer_map_xin(x)
            for block in self.patch_mixer:
                x = block(x, t_emb)
            x = self.patch_mixer_map_xout(x)

        # Optionally mask
        mask = None
        if mask_ratio > 0.0:
            B, T, D = x.shape
            mask_info = get_mask(B, T, mask_ratio, x.device)
            x = mask_out_token(x, mask_info['ids_keep'])
            mask = mask_info['mask']
            ids_restore = mask_info['ids_restore']
        else:
            ids_restore = None

        # Main blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final => patchify
        x = self.final_layer(x, t_emb)

        # Unmask if we used masking
        if mask_ratio > 0.0 and ids_restore is not None:
            x = unmask_tokens(x, ids_restore, self.mask_token)

        # Unpatchify
        x = self.unpatchify(x)
        return {'image_sample': x, 'mask': mask}

    def unpatchify(self, x):
        B, T, patch_dim = x.shape
        p = self.patch_size
        c = self.out_channels
        h = w = int(T**0.5)
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h*p, w*p)
        return x




if __name__ == "__main__":

    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth=16

    net = DiT(
        input_size=64,
        patch_size=4,
        in_channels=3,
        dim=512,
        depth=16,
        head_dim=32,
        multiple_of=64,
        pos_interp_scale=1.0,
        norm_eps=1e-6,
        depth_init=True,
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
        )

    N = 2
    x_image = torch.randn(N, 3, 64, 64)  # e.g. 2 images, 3 channels
    t = torch.randint(0, 1000, (N,))  # random timesteps

    # 3) Forward pass
    res = net(x_image, t)
    if res["mask"] is not None:
        print("img_out shape:", res["image_sample"].shape)  # (N, 3, 64, 64)
