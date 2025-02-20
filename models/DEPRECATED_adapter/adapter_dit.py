import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from timm.models.vision_transformer import PatchEmbed


###############################################################################
# 1. Utility Functions
###############################################################################

def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    else:
        raise ValueError(f"Norm type not supported: {norm_type}")

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies modulation to input tensor using shift and scale factors."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """
    MLP implementation from timm (without the dropout layers).
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

    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=init_std)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into a vector representation via sinusoidal -> MLP.
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device
            ) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # if shape (B,) => do sinusoidal => mlp => (B, hidden_size)
        if t.dim() == 1:
            emb = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        else:
            # If user already gave shape (B, freq_emb_size)
            emb = t.to(self.dtype)
        return self.mlp(emb)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]],
        cls_token: bool = False,
        extra_tokens: int = 0,
        pos_interp_scale: float = 1.0,
        base_size: int = 16
) -> np.ndarray:
    """
    2D sinusoidal positional embeddings. Provided by original snippet.
    """

    def ntuple(n):
        def parse(x):
            if isinstance(x, tuple):
                return x
            return tuple([x] * n)

        return parse

    to_2tuple = ntuple(2)
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pos_interp_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pos_interp_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d):
        assert embed_dim % 2 == 0
        H = grid_2d.shape[2]
        W = grid_2d.shape[3]
        idx = 0
        emb = np.zeros((H * W, embed_dim), dtype=np.float32)
        half_dim = embed_dim // 2
        for i in range(H):
            for j in range(W):
                y = grid_2d[1, 0, i, j]
                x = grid_2d[0, 0, i, j]
                for d in range(half_dim // 2):
                    div_term = 10000 ** (2 * d / half_dim)
                    emb[idx, 2 * d] = np.sin(y / div_term)
                    emb[idx, 2 * d + 1] = np.cos(y / div_term)
                for d in range(half_dim // 2):
                    div_term = 10000 ** (2 * d / half_dim)
                    emb[idx, half_dim + 2 * d] = np.sin(x / div_term)
                    emb[idx, half_dim + 2 * d + 1] = np.cos(x / div_term)
                idx += 1
        return emb

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


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
        "mask": mask,
        "ids_keep": ids_keep,
        "ids_restore": ids_restore
    }


def mask_out_token(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    N, L, D = x.shape
    return torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )


def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    N, L, D = x.shape
    L_total = ids_restore.shape[1]

    mask_tokens = mask_token.repeat(N, L_total - L, 1)
    x_ = torch.cat([x, mask_tokens], dim=1)  # => (N, L_total, D)
    x_ = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
    )
    return x_


################################################################################
# 2. Self-Attention and Cross-Attention
################################################################################

class CrossAttention(nn.Module):
    """
    Cross-attention layer:
     Q from x
     K,V from cond
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        norm_eps: float = 1e-6,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        assert hidden_dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_bias = qkv_bias

        self.q_linear = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, hidden_dim*2, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.ln_q = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)

    def forward(self, x_tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens => Q
        cond_tokens => K,V
        Shapes => (B, Nx, C), (B, Ny, C)
        """
        B, Nx, C = x_tokens.shape

        q = self.q_linear(x_tokens).view(B, Nx, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond_tokens).view(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)  # each => (B, Ny, num_heads, head_dim)

        # LN on q,k
        q = self.ln_q(q.reshape(B, Nx, self.num_heads*self.head_dim)).reshape(B, Nx, self.num_heads, self.head_dim)
        k = self.ln_k(k.reshape(B, -1, self.num_heads*self.head_dim)).reshape(B, -1, self.num_heads, self.head_dim)

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1,2),
            k.transpose(1,2),
            v.transpose(1,2),
            is_causal=False
        )
        attn_out = attn_out.transpose(1,2).contiguous().view(B, Nx, self.num_heads*self.head_dim)
        attn_out = self.proj(attn_out)
        return attn_out

    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.q_linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.kv_linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    """
    Self-attention layer using PyTorch's scaled_dot_product_attention.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            norm_eps: float = 1e-6,
            hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        if hidden_dim is None:
            hidden_dim = dim
        # This ensures hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv_bias = qkv_bias

        # qkv: produce [B, N, 3*hidden_dim]
        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        # extra LN for q, k
        self.ln_q = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each => [B, N, num_heads, head_dim]

        # LN on q, k
        q = self.ln_q(q.reshape(B, N, self.num_heads * self.head_dim)).reshape(
            B, N, self.num_heads, self.head_dim
        ).to(q.dtype)
        k = self.ln_k(k.reshape(B, N, self.num_heads * self.head_dim)).reshape(
            B, N, self.num_heads, self.head_dim
        ).to(k.dtype)

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        )
        x_out = attn_out.transpose(1, 2).contiguous()  # => [B, N, num_heads, head_dim]
        x_out = x_out.reshape(B, N, self.num_heads * self.head_dim)
        x_out = self.proj(x_out)  # => [B, N, dim]
        return x_out

    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


################################################################################
# 3. MoE FeedForward
################################################################################

class FeedForwardECMoe(nn.Module):
    """
    Expert-Choice Mixture of Experts feed-forward layer.
    """

    def __init__(self, num_experts: int, expert_capacity: float, dim: int,
                 hidden_dim: int, multiple_of: int):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dim = dim
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Parameter(torch.ones(num_experts, dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, self.hidden_dim, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape
        tokens_per_expert = int(self.expert_capacity * t / self.num_experts)

        scores = self.gate(x)
        probs = F.softmax(scores, dim=-1)

        g, m = torch.topk(probs.permute(0, 2, 1), tokens_per_expert, dim=-1)
        p = F.one_hot(m, num_classes=t).float()

        xin = torch.einsum('nekt,ntd->nekd', p, x)
        h = torch.einsum('nekd,edf->nekf', xin, self.w1)
        h = self.gelu(h)
        h = torch.einsum('nekf,efd->nekd', h, self.w2)

        out = g.unsqueeze(dim=-1) * h
        out = torch.einsum('nekt,nekd->ntd', p, out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


################################################################################
# 4. Final Layer
################################################################################

class T2IFinalLayer(nn.Module):
    """
    Final layer of DiT architecture for time-based conditioning only.
    """

    def __init__(
            self,
            hidden_size: int,
            time_emb_dim: int,
            patch_size: int,
            out_channels: int,
            act_layer: Any,
            norm_final: nn.Module
    ):
        super().__init__()
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(time_emb_dim, 2 * hidden_size, bias=True)
        )
        self.norm_final = norm_final

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


###############################################################################
# 5. DiTBlock
###############################################################################

class DiTBlock(nn.Module):
    """
    Single transformer block that does:
      - Self-Attn on x
      - Cross-Attn with y
      - MLP
      - time-based AdaLN for each sub-layer
    We produce 9 * dim shift/scale/gate from t:
       (shiftS,scaleS,gateS, shiftC,scaleC,gateC, shiftM,scaleM,gateM)
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
        self.dim = dim

        # QKV dimension
        if abs(qkv_ratio - 1.0) < 1e-9:
            qkv_hidden_dim = dim
        else:
            qkv_hidden_dim = (head_dim*2)*(
                (int(dim*qkv_ratio)+ head_dim*2 -1)//(head_dim*2)
            )
        num_heads = qkv_hidden_dim//head_dim

        # Self-Attn
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim
        )

        # Cross-Attn
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim
        )

        # MLP
        mlp_hidden_dim = int(dim*mlp_ratio)
        if moe_block:
            self.mlp = FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                norm_layer=None,
                bias=use_bias
            )

        # We produce 9 * dim => 3 sub-layers × (shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 9*dim, bias=True)
        )

        # We often do separate LN for MLP + one for crossAttn, but for simplicity,
        # we do a single LN for MLP if you want to replicate original code exactly,
        # you might have norm2 + norm3. We'll keep 2 LNs for MLP + crossAttn:
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)

        self.weight_init_std = (
            0.02/(2*(layer_id+1))**0.5
            if depth_init else 0.02/(2*num_layers)**0.5
        )

    def forward(
        self,
        x_tokens: torch.Tensor,   # (B, Nx, dim)
        y_tokens: torch.Tensor,   # (B, Ny, dim)
        t_emb: torch.Tensor       # (B, time_emb_dim)
    ) -> torch.Tensor:
        # chunk => shape => (B, dim)
        shift_s, scale_s, gate_s, shift_c, scale_c, gate_c, shift_m, scale_m, gate_m = \
            self.adaLN_modulation(t_emb).chunk(9, dim=1)

        # 1) Self-Attn
        x_sa = self.self_attn(modulate(x_tokens, shift_s, scale_s))
        x_tokens = x_tokens + gate_s.unsqueeze(1)* x_sa

        # 2) Cross-Attn
        x_ca = self.cross_attn(modulate(x_tokens, shift_c, scale_c), y_tokens)
        x_tokens = x_tokens + gate_c.unsqueeze(1)* x_ca

        # 3) MLP
        x_norm = self.norm2(x_tokens)
        x_mlp = self.mlp(modulate(x_norm, shift_m, scale_m))
        x_tokens = x_tokens + gate_m.unsqueeze(1)* x_mlp

        return x_tokens

    def custom_init(self):
        self.self_attn.custom_init(self.weight_init_std)
        self.cross_attn.custom_init(self.weight_init_std)
        if hasattr(self.mlp, "custom_init"):
            self.mlp.custom_init(self.weight_init_std)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        self.norm2.reset_parameters()

#############################################################################
# PatchMixerBlock (x-only)
#############################################################################

class PatchMixerBlock(nn.Module):
    """
    A simpler patch mixer block that does self-attn + MLP on x only.
    (No cross-attn for y.)
    """
    def __init__(
        self,
        dim:int,
        head_dim:int,
        mlp_ratio:float,
        qkv_ratio:float,
        multiple_of:int,
        time_emb_dim:int,
        norm_eps:float,
        depth_init:bool,
        layer_id:int,
        num_layers:int,
        use_bias:bool,
        moe_block:bool,
        num_experts:int,
        expert_capacity:float
    ):
        super().__init__()
        if abs(qkv_ratio-1.0)<1e-9:
            qkv_hidden_dim=dim
        else:
            qkv_hidden_dim=(head_dim*2)*(
                (int(dim*qkv_ratio)+ head_dim*2 -1)//(head_dim*2)
            )
        num_heads=qkv_hidden_dim//head_dim

        self.self_attn=SelfAttention(dim=dim, num_heads=num_heads,
                                     qkv_bias=use_bias, norm_eps=norm_eps,
                                     hidden_dim=qkv_hidden_dim)

        mlp_hidden_dim=int(dim*mlp_ratio)
        if moe_block:
            self.mlp=FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
        else:
            self.mlp=Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=lambda:nn.GELU(approximate="tanh"),
                norm_layer=None,
                bias=use_bias
            )

        # We'll produce 6*dim => (shiftSA, scaleSA, gateSA, shiftMLP, scaleMLP, gateMLP)
        self.adaLN_modulation=nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 6*dim, bias=True)
        )
        self.norm2=create_norm('layernorm', dim, eps=norm_eps)

        self.weight_init_std=(
            0.02/(2*(layer_id+1))**0.5 if depth_init
            else 0.02/(2*num_layers)**0.5
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor)->torch.Tensor:
        # chunk => 6 => (shSA, scSA, gaSA, shMLP, scMLP, gaMLP)
        shiftS, scaleS, gateS, shiftM, scaleM, gateM=\
            self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # self-attn
        x_sa=self.self_attn(modulate(x, shiftS, scaleS))
        x=x+ gateS.unsqueeze(1)* x_sa

        # MLP
        x_norm=self.norm2(x)
        x_mlp=self.mlp(modulate(x_norm, shiftM, scaleM))
        x=x+ gateM.unsqueeze(1)* x_mlp
        return x

    def custom_init(self):
        self.self_attn.custom_init(self.weight_init_std)
        if hasattr(self.mlp, "custom_init"):
            self.mlp.custom_init(self.weight_init_std)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        self.norm2.reset_parameters()

###############################################################################
# 6. Main DiT
###############################################################################
class DiT(nn.Module):
    """
    Diffusion Transformer that processes:
      - x_img => "source image"
      - y_img => "conditioning image"
      - time t
    Then does self-attn on x tokens + cross-attn with y tokens.
    Finally outputs a reconstructed image for x.
    """

    def __init__(
        self,
        input_size: int=32,
        patch_size: int=2,
        in_channels: int=4,
        dim: int=512,
        depth: int=8,
        head_dim: int=64,
        multiple_of: int=256,
        pos_interp_scale: float=1.0,
        norm_eps: float=1e-6,
        depth_init: bool=True,
        qkv_multipliers:List[float]=[1.0],
        ffn_multipliers:List[float]=[4.0],
        use_patch_mixer: bool=False,
        patch_mixer_depth: int=0,
        patch_mixer_dim: int=512,
        patch_mixer_qkv_ratio: float=1.0,
        patch_mixer_mlp_ratio: float=1.0,
        use_bias: bool=True,
        num_experts: int=8,
        expert_capacity: float=1,
        experts_every_n: int=2
    ):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim

        # Time embed
        self.t_embedder = TimestepEmbedder(
            hidden_size=dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            frequency_embedding_size=512
        )

        # PatchEmbed for x_img
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
            bias=True
        )
        # PatchEmbed for y_img
        self.y_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
            bias=True
        )

        self.num_patches_x = self.x_embedder.num_patches
        self.num_patches_y = self.y_embedder.num_patches

        # Positional embeddings for x,y
        self.register_buffer("pos_embed_x", torch.zeros(1, self.num_patches_x, dim), persistent=False)
        self.register_buffer("pos_embed_y", torch.zeros(1, self.num_patches_y, dim), persistent=False)

        # PatchMixer for x only
        self.use_patch_mixer = use_patch_mixer
        if use_patch_mixer and patch_mixer_depth > 0:
            # We will do x => map_xin => dimension=patch_mixer_dim => patch mixer => map_xout => dimension=dim
            self.patch_mixer_map_xin = nn.Sequential(
                create_norm('layernorm', dim, eps=norm_eps),
                nn.Linear(dim, patch_mixer_dim, bias=use_bias)
            )
            self.patch_mixer_map_xout = nn.Sequential(
                create_norm('layernorm', patch_mixer_dim, eps=norm_eps),
                nn.Linear(patch_mixer_dim, dim, bias=use_bias)
            )

            # build patch_mixer blocks
            expert_blocks_idx = [i for i in range(1, patch_mixer_depth) if (i + 1) % experts_every_n == 0]
            is_moe = [True if i in expert_blocks_idx else False for i in range(patch_mixer_depth)]

            self.patch_mixer = nn.ModuleList([
                PatchMixerBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    time_emb_dim=dim,  # We keep using 'dim' for the time embedding
                    norm_eps=norm_eps,
                    depth_init=False,
                    layer_id=i,
                    num_layers=patch_mixer_depth,
                    use_bias=use_bias,
                    moe_block=is_moe[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity
                ) for i in range(patch_mixer_depth)
            ])
        else:
            self.patch_mixer = None

        # Expand qkv/ffn multipliers across 'depth'
        assert len(ffn_multipliers)==len(qkv_multipliers)
        if len(ffn_multipliers)==depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            num_splits = len(ffn_multipliers)
            assert depth%num_splits==0, "Depth must be multiple of # splits!"
            dps = depth//num_splits
            qkv_ratios = list(np.array([[m]*dps for m in qkv_multipliers]).reshape(-1))
            mlp_ratios = list(np.array([[m]*dps for m in ffn_multipliers]).reshape(-1))

        # Mark MoE blocks
        expert_blocks_idx = [i for i in range(0, depth-1) if (i+1)%experts_every_n==0]
        is_moe_block = [True if i in expert_blocks_idx else False for i in range(depth)]

        # Main blocks (each has self-attn + cross-attn + MLP)
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
            ) for i in range(depth)
        ])

        # Mask token if we do random patch-masking on x
        self.register_buffer(
            "mask_token",
            torch.zeros(1,1, patch_size*patch_size*self.out_channels),
            persistent=False
        )

        # Final projection => produce the output image from x
        self.final_layer = T2IFinalLayer(
            hidden_size=dim,
            time_emb_dim=dim,
            patch_size=patch_size,
            out_channels=in_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            norm_final=create_norm('layernorm', dim, eps=norm_eps)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize all submodules:
          - fill pos_embed_x, pos_embed_y with sin-cos
          - normal init for patch embed
          - custom_init for blocks
          - zero final layer
        """
        def zero_bias(m: nn.Module):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                zero_bias(module)

        self.apply(_basic_init)

        # Sin-cos for x
        px = get_2d_sincos_pos_embed(
            embed_dim=self.dim,
            grid_size=int(self.num_patches_x**0.5),
            cls_token=False,
            extra_tokens=0,
            pos_interp_scale=1.0,
            base_size=(self.input_size//self.patch_size)
        )
        self.pos_embed_x.data.copy_(torch.from_numpy(px).float().unsqueeze(0))

        # Sin-cos for y
        py = get_2d_sincos_pos_embed(
            embed_dim=self.dim,
            grid_size=int(self.num_patches_y**0.5),
            cls_token=False,
            extra_tokens=0,
            pos_interp_scale=1.0,
            base_size=(self.input_size//self.patch_size)
        )
        self.pos_embed_y.data.copy_(torch.from_numpy(py).float().unsqueeze(0))

        # PatchEmbed init
        wx = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(wx.view([wx.shape[0], -1]))
        wy = self.y_embedder.proj.weight.data
        nn.init.xavier_uniform_(wy.view([wy.shape[0], -1]))

        # custom_init blocks
        for block in self.blocks:
            block.custom_init()
        if self.patch_mixer is not None:
            for block in self.patch_mixer:
                block.custom_init()

        # final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse patch embedding => (B,T, p^2*c) => (B,c,H,W)
        """
        B, T, D = x.shape
        p = self.patch_size
        c = self.out_channels
        h = w = int(T**0.5)
        x = x.view(B,h,w,p,p,c)
        x = x.permute(0,5,1,3,2,4).contiguous()
        x = x.view(B,c,h*p,w*p)
        return x

    def forward(
        self,
        x_img: torch.Tensor,  # shape => (B, in_channels, H, W)
        y_img: torch.Tensor,  # shape => (B, in_channels, H, W) for conditioning
        t: torch.Tensor,      # shape => (B,) or (B,dim)
        mask_ratio: float=0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Return => {
          'image_sample': (B,c,H,W),
          'mask': optional
        }
        """
        # 1) Patch embed x + pos
        x_tokens = self.x_embedder(x_img) + self.pos_embed_x
        # 2) Patch embed y + pos
        y_tokens = self.y_embedder(y_img) + self.pos_embed_y

        # 3) T embedding
        t_emb = self.t_embedder(t)

        # 4) optional patch mixer on x
        if self.patch_mixer is not None:
            x_tokens = self.patch_mixer_map_xin(x_tokens)
            for block in self.patch_mixer:
                # block only does self-attn + MLP => expects (x_tokens,t_emb)
                x_tokens = block(x_tokens, t_emb)
            x_tokens = self.patch_mixer_map_xout(x_tokens)

        # 5) optional mask x
        mask=None
        ids_restore=None
        if mask_ratio>0:
            mask_dict = get_mask(
                batch=x_tokens.shape[0],
                length=x_tokens.shape[1],
                mask_ratio=mask_ratio,
                device=x_tokens.device
            )
            ids_keep = mask_dict['ids_keep']
            ids_restore = mask_dict['ids_restore']
            mask = mask_dict['mask']
            x_tokens = mask_out_token(x_tokens, ids_keep)

        # 6) main DiT blocks => each does self + cross + MLP
        for block in self.blocks:
            x_tokens = block(x_tokens, y_tokens, t_emb)

        # 7) final => shape => (B,T, p^2*c)
        x_tokens = self.final_layer(x_tokens, t_emb)

        # unmask
        if mask_ratio>0 and ids_restore is not None:
            x_tokens = unmask_tokens(x_tokens, ids_restore, self.mask_token)

        # unpatchify => (B,c,H,W)
        out_img = self.unpatchify(x_tokens)
        return {"image_sample": out_img, "mask": mask}


###############################################################################
# 5. Factory Functions
###############################################################################
def MicroDiT_Tiny_2(
        qkv_ratio: List[float] = [1.0, 1.0],
        mlp_ratio: List[float] = [4.0, 4.0],
        pos_interp_scale: float = 1.0,
        input_size: int = 64,
        num_experts: int = 8,
        expert_capacity: float = 2.0,
        experts_every_n: int = 2,
        in_channels: int = 4,
        **kwargs
) -> DiT:
    depth = 8
    model = DiT(
        input_size=input_size,
        patch_size=4,
        in_channels=in_channels,
        dim=512,
        depth=depth,
        head_dim=32,
        multiple_of=256,
        pos_interp_scale=pos_interp_scale,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], depth),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], depth),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=True,
        num_experts=num_experts,
        expert_capacity=expert_capacity,
        experts_every_n=experts_every_n,
        **kwargs
    )
    return model


def MicroDiT_XL_2(
        qkv_ratio: List[float] = [0.5, 1.0],
        mlp_ratio: List[float] = [0.5, 4.0],
        pos_interp_scale: float = 1.0,
        input_size: int = 64,
        num_experts: int = 8,
        expert_capacity: float = 2.0,
        experts_every_n: int = 2,
        in_channels: int = 4,
        **kwargs
) -> DiT:
    depth = 28
    model = DiT(
        input_size=input_size,
        patch_size=2,
        in_channels=in_channels,
        dim=1024,
        depth=depth,
        head_dim=64,
        multiple_of=256,
        pos_interp_scale=pos_interp_scale,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], depth),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], depth),
        use_patch_mixer=True,
        patch_mixer_depth=6,
        patch_mixer_dim=768,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=True,
        num_experts=num_experts,
        expert_capacity=expert_capacity,
        experts_every_n=experts_every_n,
        **kwargs
    )
    return model


###############################################################################
# 7. Quick Test
###############################################################################

if __name__=="__main__":
    print("Testing cross-conditioned DiT => 'source' + 'target' images...")

    B=2
    C=4
    H=64
    W=64

    # create a small model
    model=MicroDiT_Tiny_2(input_size=64, in_channels=4)
    x_img=torch.randn(B, C, H, W)
    y_img=torch.randn(B, C, H, W)
    t=torch.randint(0,1000,(B,))

    out_dict=model(
        x_img,  # source
        y_img,  # condition
        t,
        mask_ratio=0.25
    )
    out_img=out_dict["image_sample"]
    print("Check tiny model")
    print("Output image shape:", out_img.shape)  # => (B,4,32,32)
    if out_dict["mask"] is not None:
        print("Mask shape:", out_dict["mask"].shape)


    # create a large model
    model = MicroDiT_XL_2(input_size=64, in_channels=4)

    out_dict = model(
        x_img,  # source
        y_img,  # condition
        t,
        mask_ratio=0.25
    )
    out_img = out_dict["image_sample"]
    print("Check large model")
    print("Output image shape:", out_img.shape)  # => (B,4,32,32)
    if out_dict["mask"] is not None:
        print("Mask shape:", out_dict["mask"].shape)
