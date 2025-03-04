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


class FeedForwardECMoe(nn.Module):
    """
    Expert-Choice style MoE feed-forward: each token route is assigned
    to exactly one expert (via top-k=1 gating).
    """
    def __init__(self, num_experts, expert_capacity, dim, hidden_dim, multiple_of):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dim = dim
        # We'll round hidden_dim up to a multiple of `multiple_of`, just like your code:
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        # Each expert has 2 linear layers (w1, w2)
        self.w1 = nn.Parameter(torch.ones(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, hidden_dim, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

        # Optionally you might want to init them in a custom way.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        For each token, we do top-1 gating => route token to exactly one expert.
        Then do w1->GELU->w2. Then combine outputs.
        """
        B, T, D = x.shape

        # Decide how many tokens go to each expert
        # We'll interpret "expert_capacity" as fraction or as absolute.
        # In the original code, you did something like tokens_per_expert = ...
        tokens_per_expert = int(self.expert_capacity * T / self.num_experts)
        # If you want a simpler approach, you can do tokens_per_expert = int(self.expert_capacity).

        # 1) gating
        scores = self.gate(x)  # (B, T, E)
        probs = F.softmax(scores, dim=-1)  # (B, T, E)
        # pick top-k=1 => gather the top (tokens_per_expert) for each expert
        # One simplistic approach: pick the top tokens_per_expert along T for each expert
        # The original code does "topk" along dimension T but after permuting:
        g, m = torch.topk(
            probs.permute(0,2,1),
            k=tokens_per_expert,
            dim=-1
        )
        # g, m each => (B, E, tokens_per_expert)

        # 2) one-hot for each expert => shape (B, E, tokens_per_expert, T)
        p = F.one_hot(m, num_classes=T).float()

        # 3) gather tokens for each expert => shape (B, E, tokens_per_expert, D)
        xin = torch.einsum('bekt,btd->bekd', p, x)

        # 4) forward each expert
        h = torch.einsum('bekd,edh->bekh', xin, self.w1)  # w1
        h = self.gelu(h)
        h = torch.einsum('bekh,ehd->bekd', h, self.w2)   # w2

        # 5) multiply by the gating factor
        out = g.unsqueeze(dim=-1) * h  # (B,E,tokens_per_expert,D)

        # 6) scatter back to the original token positions
        out = torch.einsum('bekt,bekd->btd', p, out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)



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





class JointBlock(nn.Module):
    """
    A single Transformer block over *all tokens* (image + tab).
    6×dim gating => shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    """
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
        moe_block: bool,
        num_experts: int,
        expert_capacity: float
    ):
        super().__init__()
        # QKV dims
        qkv_hidden_dim = (
            (head_dim*2)*((int(dim*qkv_ratio) + head_dim*2 -1)//(head_dim*2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim//head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)

        # MLP or MoE
        if moe_block:
            self.mlp = FeedForwardECMoe(
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                dim=dim,
                hidden_dim=mlp_hidden_dim,
                multiple_of=multiple_of
            )
        else:
            self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # Gating
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim)
        )

        if depth_init:
            self.weight_init_std = 0.02 / (2*(layer_id+1))**0.5
        else:
            self.weight_init_std = 0.02 / (2*num_layers)**0.5

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # x => (B, T_all, dim)
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # 1) Self-attn
        x_ln = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_ln)

        # 2) MLP / MoE
        x_ln2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_ln2)
        return x


# -------------------------------------------------------------------------
# Final heads
# -------------------------------------------------------------------------
class FinalImageHead(nn.Module):
    """
    Takes final image tokens => produce patch outputs => unpatchify
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

    def forward(self, x_tokens, t_emb):
        """
        x_tokens: (B, T_img, in_dim) after Transformer
        returns: (B, T_img, patch_size^2 * out_chans)
        """
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x_norm = modulate(self.norm(x_tokens), shift, scale)
        return self.linear(x_norm)


class FinalTabHead(nn.Module):
    """
    Takes final tab tokens => produce final tab (e.g. each token or pooled)
    Here we do simple mean-pool then MLP => out_table_features
    """
    def __init__(self, in_dim, out_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_features)
        )

    def forward(self, x_tokens):
        # x_tokens: (B, T_tab, in_dim)
        x_norm = self.norm(x_tokens)
        x_pooled = x_norm.mean(dim=1)  # (B, in_dim)
        return self.mlp(x_pooled)


# -------------------------------------------------------------------------
# The Joint DiT Model (Model C)
# -------------------------------------------------------------------------
class JointDiTWithPatchMixerAndMoe(nn.Module):
    """
    1) Patchify the (noisy) image => pass it through patch_mixer blocks
       (some may be MoE).
    2) Project the (noisy) tab => get tab tokens.
    3) Concat the result => pass through main "joint blocks" (some may be MoE).
    4) Final heads => denoise image + tab.
    """
    def __init__(
        self,
        # image stuff
        input_size=32,
        patch_size=2,
        in_channels=4,
        # overall hidden dim for the main model
        dim=512,
        # patch mixer settings
        use_patch_mixer=True,
        patch_mixer_depth=2,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,
        # main transformer settings
        depth=6,
        head_dim=64,
        qkv_multipliers=[1.0],
        ffn_multipliers=[4.0],
        norm_eps=1e-6,
        depth_init=True,
        use_bias=True,
        multiple_of=256,
        # tab settings
        num_tab_columns=10,
        tab_groups=2,
        out_table_features=10,
        # MoE
        num_experts=8,
        expert_capacity=1.0,
        experts_every_n=2  # apply MoE block for every Nth block
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.use_patch_mixer = use_patch_mixer

        self.register_buffer(
            "mask_token",
            torch.zeros(1, 1, patch_size * patch_size * in_channels),
            persistent=False
        )

        # 1) Patchify image
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.num_img_tokens = self.x_embedder.num_patches

        # 2) Timestep embed
        self.t_embedder = TimestepEmbedder(hidden_size=dim)

        # 3) Patch Mixer blocks (optional)
        if use_patch_mixer:
            # which patch mixer blocks are MoE?
            pm_expert_blocks_idx = [
                i for i in range(patch_mixer_depth)
                if (i+1) % experts_every_n == 0
            ]
            pm_is_moe_block = [(i in pm_expert_blocks_idx) for i in range(patch_mixer_depth)]

            self.patch_mixer = nn.ModuleList([
                PatchMixerBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    time_emb_dim=dim,      # we can keep them separate or unify
                    norm_eps=norm_eps,
                    depth_init=False,      # doesn't matter too much
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
                    nn.LayerNorm(dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
                self.patch_mixer_map_xout = nn.Sequential(
                    nn.LayerNorm(patch_mixer_dim, eps=norm_eps),
                    nn.Linear(patch_mixer_dim, dim, bias=use_bias)
                )
            else:
                self.patch_mixer_map_xin = nn.Identity()
                self.patch_mixer_map_xout = nn.Identity()
        else:
            self.patch_mixer = None

        # 4) Tab projection
        self.tab_proj = TabularProjection(
            in_features=num_tab_columns,
            hidden_size=dim,
            groups=tab_groups
        )
        # figure out T_tab
        dummy_in = torch.zeros(1, num_tab_columns)
        with torch.no_grad():
            T_tab = self.tab_proj(dummy_in).shape[1]
        self.num_tab_tokens = T_tab

        # 5) The main joint blocks
        total_depth = depth
        # replicate multipliers if needed
        if len(ffn_multipliers) == total_depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            num_splits = len(ffn_multipliers)
            assert total_depth % num_splits == 0
            dps = total_depth // num_splits
            qkv_ratios = list(np.concatenate([[m]*dps for m in qkv_multipliers]))
            mlp_ratios = list(np.concatenate([[m]*dps for m in ffn_multipliers]))

        # figure out which main blocks are MoE
        expert_blocks_idx = [
            i for i in range(depth)
            if (i+1) % experts_every_n == 0
        ]
        is_moe_block = [(i in expert_blocks_idx) for i in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = JointBlock(
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
            self.blocks.append(blk)

        # 6) Final heads for image and tab
        self.final_img = FinalImageHead(
            in_dim=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_chans=in_channels,  # same as input
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm(dim, eps=norm_eps)
        )
        self.final_tab = FinalTabHead(in_dim=dim, out_features=out_table_features)

        # 7) Single pos_embed for "image+tab" or separate?
        #    Typically you'd do a single pos_embed for the final combined tokens.
        #    But the patch mixer uses its own self-attn => needs no separate pos embedding there.
        #    For the "main" blocks, let's do a single pos_embed of size (T_img + T_tab).
        self.num_tokens_total = self.num_img_tokens + self.num_tab_tokens
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, self.num_tokens_total, dim),
            persistent=False
        )

        self.initialize_weights()

    def initialize_weights(self):
        # patch mixer init
        if self.patch_mixer:
            for blk in self.patch_mixer:
                # you can define a custom_init if needed
                pass

        # main blocks init
        for blk in self.blocks:
            # similarly, can do custom init
            pass

        # sin-cos for the image portion => zeros for the tab portion
        side = int(self.num_img_tokens**0.5)
        pe_img = self.get_2d_sincos_pe(side, self.dim)  # shape (T_img, dim)
        pe_tab = np.zeros((self.num_tab_tokens, self.dim), dtype=np.float32)
        pe_full = np.concatenate([pe_img, pe_tab], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(pe_full).unsqueeze(0))

        # You can also zero out final_img linear, etc.
        nn.init.constant_(self.final_img.linear.weight, 0)

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
        emb_h = get_1d_sin_cos(hh, embed_dim//2)
        emb_w = get_1d_sin_cos(ww, embed_dim//2)
        return np.concatenate([emb_h, emb_w], axis=1)

    def unpatchify(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: (B, T_img, patch_size^2 * out_chans)
        => (B, out_chans, H, W)
        """
        B, T, patch_dim = x_tokens.shape
        p = self.patch_size
        c = self.in_channels
        h = w = int(T**0.5)
        x = x_tokens.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h*p, w*p)
        return x

    def forward(
            self,
            x_noisy_img: torch.Tensor,  # (B, in_channels, H, W)
            x_noisy_tab: torch.Tensor,  # (B, num_tab_columns)
            t: torch.Tensor,  # (B,) timesteps
            mask_ratio: float = 0.0
    ):
        B = x_noisy_img.shape[0]

        # 1) patchify image => (B, T_img, dim)
        x_img_tokens = self.x_embedder(x_noisy_img)

        # 2) If masking is requested
        mask_info = None
        if mask_ratio > 0:
            mask_info = get_mask(
                B=B,
                T=self.num_img_tokens,  # total patch-tokens
                mask_ratio=mask_ratio,
                device=x_img_tokens.device
            )
            # keep only unmasked tokens
            x_img_tokens = mask_out_token(x_img_tokens, mask_info['ids_keep'])

        # 3) Timestep embed
        t_emb = self.t_embedder(t)

        # 4) Patch mixer if any
        if self.patch_mixer:
            x_img_tokens = self.patch_mixer_map_xin(x_img_tokens)
            for pm_block in self.patch_mixer:
                x_img_tokens = pm_block(x_img_tokens, t_emb)
            x_img_tokens = self.patch_mixer_map_xout(x_img_tokens)

        # 5) tab => (B, T_tab, dim)
        tab_tokens = self.tab_proj(x_noisy_tab)

        # 6) Concat => shape (B, T_img_kept + T_tab, dim)
        x_all = torch.cat([x_img_tokens, tab_tokens], dim=1)

        # 7) Add position embedding (only for the # of tokens we actually have)
        T_all = x_all.shape[1]
        x_all = x_all + self.pos_embed[:, :T_all, :]

        # 8) main blocks
        for blk in self.blocks:
            x_all = blk(x_all, t_emb)

        # 9) Split back out
        T_img_final = x_img_tokens.shape[1]  # may be < self.num_img_tokens if masked
        x_img_final = x_all[:, :T_img_final, :]
        x_tab_final = x_all[:, T_img_final:, :]

        # 10) final image => (B, T_img_final, patch_size^2 * in_channels)
        img_patch_logits = self.final_img(x_img_final, t_emb)

        # 11) If we want to "unmask" => restore to the original T_img
        if mask_ratio > 0 and mask_info is not None:
            img_patch_logits = unmask_tokens(
                x_masked=img_patch_logits,
                ids_restore=mask_info['ids_restore'],
                mask_token=self.mask_token  # shape (1,1, patch_size^2*in_channels)
            )

        # 12) unpatchify => (B, in_channels, H, W)
        denoised_img = self.unpatchify(img_patch_logits)

        # 13) final tab
        denoised_tab = self.final_tab(x_tab_final)

        return {
            "image_sample": denoised_img,
            "table_sample": denoised_tab,
            "mask": mask_info['mask'] if mask_info is not None else None
        }










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



class PatchMixerBlock(nn.Module):
    """
    A mini-block that does (self-attn + MLP) on the *image patch tokens only*,
    modulated by the time embedding.
    Optionally uses MoE for the MLP.
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
        # QKV dims
        qkv_hidden_dim = (
            (head_dim*2)*((int(dim*qkv_ratio) + head_dim*2 -1)//(head_dim*2))
            if qkv_ratio != 1.0 else dim
        )
        mlp_hidden_dim = int(dim*mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,
            qkv_bias=use_bias,
            hidden_dim=qkv_hidden_dim
        )

        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)

        # MoE or standard MLP
        if moe_block:
            self.mlp = FeedForwardECMoe(
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                dim=dim,
                hidden_dim=mlp_hidden_dim,
                multiple_of=multiple_of
            )
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
        # x => (B, T_img, dim)
        B, T, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Self Attn
        x_ln = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_ln)

        # MLP (or MoE)
        x_ln2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_ln2)
        return x




# class Mlp(nn.Module):
#     """
#     Basic MLP from timm (without dropout).
#     """
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: Optional[int] = None,
#         out_features: Optional[int] = None,
#         act_layer: Any = lambda: nn.GELU(approximate="tanh"),
#         norm_layer: Optional[Any] = None,
#         bias: bool = True,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#
#         self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
#         self.act = act_layer()
#         self.norm = norm_layer if norm_layer is not None else nn.Identity()
#         self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.norm(x)
#         x = self.fc2(x)
#         return x


class TabularProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, groups: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups

        # We'll precompute group boundaries
        group_sizes = self._get_group_sizes(in_features, groups)
        self.group_mlps = nn.ModuleList()
        start = 0
        for gsize in group_sizes:
            mlp = nn.Sequential(
                nn.Linear(gsize, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.group_mlps.append(mlp)
            start += gsize

    def forward(self, tab_input: torch.Tensor) -> torch.Tensor:
        B, C = tab_input.shape
        group_sizes = self._get_group_sizes(C, self.groups)
        outputs = []
        start = 0
        for i, gsize in enumerate(group_sizes):
            subset = tab_input[:, start:start+gsize]   # (B, gsize)
            out = self.group_mlps[i](subset)           # (B, hidden_size)
            outputs.append(out.unsqueeze(1))
            start += gsize
        return torch.cat(outputs, dim=1)  # => (B, T_tab, hidden_size)

    @staticmethod
    def _get_group_sizes(num_cols: int, groups: int):
        base = num_cols // groups
        remainder = num_cols % groups
        sizes = []
        for i in range(groups):
            size = base + (1 if i < remainder else 0)
            sizes.append(size)
        return sizes






################################################################################
# PatchEmbed, FinalLayer, Masking
################################################################################




def get_mask(B: int, T: int, mask_ratio: float, device):
    """
    Randomly mask `mask_ratio` fraction of the tokens out of total T.
    Returns a dict with:
      - ids_keep: (B, T_keep) the indices of tokens we keep
      - ids_restore: (B, T) how to restore the ordering
      - mask: (B, T) 0 or 1 indicating which tokens are masked
    """
    T_keep = int(T * (1 - mask_ratio))
    # generate random permutations of the token indices
    noise = torch.rand(B, T, device=device)  # uniform [0,1)
    # sort each row by noise
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :T_keep]
    # prepare the restore indices
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # generate the binary mask
    mask = torch.ones([B, T], device=device)
    mask.scatter_(1, ids_keep, 0)
    return {
        'ids_keep': ids_keep,
        'ids_restore': ids_restore,
        'mask': mask
    }

def mask_out_token(x: torch.Tensor, ids_keep: torch.Tensor):
    """
    x: (B, T, D)
    ids_keep: (B, T_keep)
    returns x_kept => (B, T_keep, D)
    """
    B, T, D = x.shape
    T_keep = ids_keep.shape[1]
    # gather according to ids_keep
    # We expand ids_keep to shape (B, T_keep, 1) then broadcast
    ids_keep_ex = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_masked = torch.gather(x, dim=1, index=ids_keep_ex)
    return x_masked

def unmask_tokens(x_masked: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor):
    """
    x_masked:   (B, T_keep, D)
    ids_restore:(B, T)        - sorted indices telling where each of the T_keep tokens goes.
    mask_token: (1, 1, D)     - the embedding for all masked positions

    Returns x_full: (B, T, D),
      where x_masked tokens are scattered into their correct positions,
      and the masked positions are filled with mask_token.
    """
    B, T_keep, D = x_masked.shape
    T = ids_restore.shape[1]  # the total # of tokens (masked + unmasked)

    # Prepare an empty array for final
    # each masked slot is initially "mask_token"
    x_full = mask_token.repeat(B, T, 1).to(x_masked.device)  # (B, T, D)

    # We only have T_keep tokens in x_masked, so we only slice the first T_keep columns of ids_restore
    ids_restore_ex = ids_restore[:, :T_keep].unsqueeze(-1).expand(-1, -1, D)
    # now ids_restore_ex shape => (B, T_keep, D) matches x_masked shape => (B, T_keep, D)

    # scatter the unmasked tokens back to their correct positions
    x_full.scatter_(dim=1, index=ids_restore_ex, src=x_masked)
    return x_full






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
    model = JointDiTWithPatchMixerAndMoe(**cfg.dit)
    print("[INFO] Loaded DiT")
    return model

###############################################################################
# Usage Example:
###############################################################################
def test_model_flow():
    # instantiate
    model = JointDiTWithPatchMixerAndMoe(
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=64,
        patch_mixer_depth=1,
        depth=2,
        num_tab_columns=10
    ).cuda()

    B = 2
    x_img = torch.randn(B, 4, 32, 32).cuda()
    x_tab = torch.randn(B, 10).cuda()
    t = torch.randint(0, 1000, (B,)).cuda()

    out = model(x_img, x_tab, t, mask_ratio=0.3)
    print("image_sample shape:", out["image_sample"].shape)  # => (2, 4, 32, 32)
    print("table_sample shape:", out["table_sample"].shape)  # => (2, 10)
    if out["mask"] is not None:
        print("mask shape:", out["mask"].shape)  # => (2, #patches=256)
        print("Mask ratio actual:", out["mask"].float().mean().item())  # ~0.3
# Finally, just call:
if __name__ == "__main__":
    test_model_flow()
