import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Union, Tuple, List, Optional, Any, Dict
from timm.models.vision_transformer import PatchEmbed



class TabularEncoder(nn.Module):
    """
    An MLP to embed raw tabular features into an intermediate dimension.
    This stays inside the MultiModalDiT so it trains end-to-end.
    """

    def __init__(self, feature_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),  # approximate
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, feature_dim)
        return self.net(x)  # (N, embed_dim)


class AttentionBlockTabular(nn.Module):
    """
    A small self-attention + MLP block for a single token (N, 1, dim).
    We do this once to refine the tabular embedding before using it as a prompt.
    Adapt this for multiple tab tokens (one per feature column)
    """

    def __init__(
            self,
            dim: int,
            head_dim: int,
            mlp_ratio: float,
            multiple_of: int,
            norm_eps: float,
            use_bias: bool = True,
    ):
        super().__init__()
        assert dim % head_dim == 0, "Hidden dimension must be multiple of head_dim"

        self.num_heads = dim // head_dim
        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=self.num_heads,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            multiple_of=multiple_of,
            use_bias=use_bias,
        )

    def forward(self, x: torch.Tensor):
        # x shape: (N, 1, dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Convert timesteps (batch of scalar values) into sinusoidal embeddings of size `dim`.
    Typical approach: half of 'dim' is sin, half is cos at different frequencies.
    """

    # timesteps is (N,) or (N,1). Ensure it's float
    timesteps = timesteps.float().view(-1)
    half_dim = dim // 2
    # Compute geometric progression of frequencies
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    # Outer product: shape (N, half_dim)
    freqs = timesteps[:, None] * freqs[None, :]
    # Embed sin & cos
    emb = torch.cat([freqs.sin(), freqs.cos()], dim=1)
    if dim % 2 == 1:
        # if odd dim, pad one more channel
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedder(nn.Module):
    """
    Standard technique for diffusion: embed the scalar 't' using sinusoidal embeddings, then an MLP up to 'dim'.

    Args:
        hidden_size (int): Output (model) dimension after MLP
        act_layer (Any): Constructor for the activation (e.g., nn.GELU)
        frequency_embedding_size (int): Size of the initial sinusoidal embedding
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

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        1) Convert timesteps to a sinusoidal embedding of size `frequency_embedding_size`.
        2) Project through an MLP to `hidden_size`.
        Returns (N, hidden_size).
        """
        # (N,) -> (N, frequency_embedding_size)
        fourier_emb = get_timestep_embedding(timesteps, self.frequency_embedding_size)
        # (N, hidden_size)
        return self.mlp(fourier_emb)


def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Creates a normalization layer based on the specified type.
    The code uses 'np_layernorm' for Q,K in attention.

    - 'layernorm': Standard PyTorch LayerNorm with learnable parameters.
    - 'np_layernorm': A LayerNorm with elementwise_affine=False (no learned affine).
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    else:
        raise ValueError(f'Norm type "{norm_type}" not supported!')

def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Unmask tokens using provided mask token."""
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    x_ = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
    )  # unshuffle
    return x_


def get_mask(batch: int, length: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """Get binary mask for input sequence.

    mask: binary mask, 0 is keep, 1 is remove
    ids_keep: indices of tokens to keep
    ids_restore: indices to restore the original order
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
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
    """Mask out tokens specified by ids_keep."""
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )
    return x_masked


class T2IFinalLayer(nn.Module):
    """
    Final injection of the pooled embedding c into the image tokens x with AdaLN,
    then a linear that reshapes tokens -> patch_size^2 * out_channels for each token.
    """

    def __init__(self, in_dim, pooled_emb_dim, patch_size, out_channels, act_layer, norm_layer):
        super().__init__()
        self.norm = norm_layer
        self.adaLN_modulation = nn.Sequential(
            act_layer(approximate="tanh"),
            nn.Linear(pooled_emb_dim, 2 * in_dim, bias=True),
        )
        self.linear = nn.Linear(in_dim, patch_size**2 * out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


def ntuple(n):
    """Helper: convert int -> (int, int) if needed."""

    def parse(x):
        if isinstance(x, tuple):
            return x
        return (x,) * n

    return parse


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Build 2D sin-cos positional embedding from a flattened grid (2, H, W).
    The result is shape (H*W, embed_dim).
    """
    assert grid.shape[0] == 2
    h, w = grid.shape[1], grid.shape[2]
    emb_h = get_1d_sin_cos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sin_cos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sin_cos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """
    Create 1D sin-cos embedding from positions. Output shape: (len(positions), embed_dim).
    """
    assert embed_dim % 2 == 0
    # i.e. half is sin, half is cos
    half_dim = embed_dim // 2
    freqs = np.arange(half_dim, dtype=np.float32)
    freqs = np.expand_dims(positions, 1) * (1.0 / (10000 ** (freqs / half_dim)))
    out = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=1)
    return out


def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]],
        cls_token: bool = False,
        extra_tokens: int = 0,
        pos_interp_scale: float = 1.0,
        base_size: int = 16
) -> np.ndarray:
    """
    Generate 2D sin-cos position embeddings for a (grid_size x grid_size) patch layout.
    Includes optional interpolation scale and base_size logic from the authors.
    """
    to_2tuple = ntuple(2)
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    # e.g. grid_size = (H, W)

    # "Interpolate" position if needed
    # e.g. if pos_interp_scale != 1, the coordinate values are scaled
    # or if base_size differs from grid_size, we do partial re-scaling.
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    # dividing by (grid_size / base_size) -> allows resolution mismatch
    grid_h = grid_h / (grid_size[0] / base_size) / pos_interp_scale
    grid_w = grid_w / (grid_size[1] / base_size) / pos_interp_scale

    # Make mesh
    grid = np.meshgrid(grid_w, grid_h)  # [2, H, W]
    grid = np.stack(grid, axis=0)  # (2, H, W)

    # Flatten to (H*W, embed_dim)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # Optionally prepend extra tokens (e.g. class token, etc.)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class SelfAttention(nn.Module):
    """
    Self-attention layer.
      - We do a single QKV projection of shape (B, N, 3 * hidden_dim).
      - We reshape into (B, N, num_heads, head_dim) for q, k, v.
      - We apply scaled-dot-product-attention, then project back to (B, N, dim).

    Args:
        dim (int): Input and output tensor dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool, True): Whether to use bias in QKV linear layers
        norm_eps (float, 1e-6): Epsilon for normalization layers
        hidden_dim (Optional[int], None): Dimension for qkv space. If None, same as input dim
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
        assert hidden_dim % num_heads == 0, 'hidden_dim should be divisible by num_heads'
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        # Normalization for Q, K
        self.ln_q = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (B, N, C=dim)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, num_heads, head_dim)

        # LN each of q, k
        q = self.ln_q(q.view(B, N, self.num_heads * self.head_dim)).view(B, N, self.num_heads, self.head_dim)
        k = self.ln_k(k.view(B, N, self.num_heads * self.head_dim)).view(B, N, self.num_heads, self.head_dim)

        # scaled_dot_product_attention in PyTorch 2.0 handles the attention
        # q,k,v: (B, num_heads, N, head_dim)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        ).transpose(1, 2)

        # (B, N, num_heads, head_dim) -> (B, N, hidden_dim)
        x = x.reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x

    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class CrossAttention(nn.Module):
    """
    Same idea as SelfAttention, but Q is from x, while K,V come from the conditioning.
    Cross-attention layer:
      - Query (Q) is derived from x,
      - Key (K) / Value (V) are derived from 'cond' (the conditioning tokens).

    Args:
        dim (int): Input and output tensor dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool, True): Whether to use bias in Q/KV linear layers
        norm_eps (float, 1e-6): Epsilon for normalization layers
        hidden_dim (Optional[int], None): Dimension for qkv space
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
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Separate linear layers for Q and KV
        self.q_linear = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.ln_q = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('np_layernorm', dim=hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: shape (B, Nx, dim)  -> will generate Q
        cond: shape (B, Ny, dim) -> will generate K, V
        Returns: (B, Nx, dim)
        """
        B, Nx, C = x.shape

        # Q
        q = self.q_linear(x).reshape(B, Nx, self.num_heads, self.head_dim)

        # K, V
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)  # each shape (B, Ny, num_heads, head_dim)

        # LN Q, K
        q = self.ln_q(q.view(B, Nx, self.num_heads * self.head_dim)).view(B, Nx, self.num_heads, self.head_dim)
        k = self.ln_k(k.view(B, -1, self.num_heads * self.head_dim)).view(B, -1, self.num_heads, self.head_dim)

        # Compute scaled dot-product attention
        # q: (B, Nx, num_heads, head_dim) -> rearranged
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        ).transpose(1, 2)

        # Reshape back to (B, Nx, hidden_dim)
        x = x.reshape(B, Nx, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x

    def custom_init(self, init_std: float) -> None:
        for linear in (self.q_linear, self.kv_linear):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


class FeedForwardECMoe(nn.Module):
    """
    Expert-Choice style Mixture of Experts (MoE) feed-forward layer with GELU activation.
      - We split tokens among 'num_experts' via gating.
      - Each expert is a separate (dim -> hidden_dim -> dim) MLP.

    Args:
        num_experts (int): Number of experts
        expert_capacity (float): Capacity factor (approx tokens per expert)
        dim (int): Input dimension (and output dimension)
        hidden_dim (int): MLP hidden dimension
        multiple_of (int): Round hidden dim up to nearest multiple
    """

    def __init__(
            self,
            num_experts: int,
            expert_capacity: float,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dim = dim
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Each expert has two weight matrices: (dim -> hidden_dim) and (hidden_dim -> dim)
        self.w1 = nn.Parameter(torch.ones(num_experts, dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, self.hidden_dim, dim))
        # Gating function: decides which tokens go to which expert
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (n, t, d)
          - n: batch size
          - t: number of tokens
          - d: embedding dimension
        Returns shape (n, t, d)
        """
        assert len(x.shape) == 3
        n, t, d = x.shape
        # tokens_per_expert = capacity factor * tokens / num_experts
        tokens_per_expert = int(self.expert_capacity * t / self.num_experts)

        # 1) Gating
        scores = self.gate(x)  # (n, t, e)
        probs = F.softmax(scores, dim=-1)  # (n, t, e)

        # 2) Find top-k tokens for each expert
        g, m = torch.topk(probs.permute(0, 2, 1), tokens_per_expert, dim=-1)
        # g: (n, e, k) gating probabilities
        # m: (n, e, k) token indices
        p = F.one_hot(m, num_classes=t).float()  # (n, e, k, t)

        # 3) Dispatch tokens to experts
        # (n, e, k, d)
        xin = torch.einsum('nekt, ntd -> nekd', p, x)
        # MLP part 1
        h = torch.einsum('nekd, edf -> nekf', xin, self.w1)  # (n, e, k, hidden_dim)
        h = self.gelu(h)
        # MLP part 2
        h = torch.einsum('nekf, efd -> nekd', h, self.w2)  # (n, e, k, d)

        # 4) Weighted by gating probabilities
        out = g.unsqueeze(dim=-1) * h  # (n, e, k, d)

        # 5) Combine experts output back to (n, t, d)
        out = torch.einsum('nekt, nekd -> ntd', p, out)
        return out

    def custom_init(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


class FeedForward(nn.Module):
    """
    A standard feed-forward block with SiLU activation.
    Basic MLP with hidden_dim ~ 2/3 the original ratio. A nonstandard scaling, then an elementwise multiplication
    in forward.

    Args:
        dim (int): Input (and output) dimension
        hidden_dim (int): Intermediate dimension
        multiple_of (int): Round hidden_dim up to nearest multiple
        use_bias (bool): Whether linear layers have bias
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            use_bias: bool,
    ):
        super().__init__()
        self.dim = dim
        # The authors do a custom scale: hidden_dim = 2/3 * hidden_dim, then round
        hidden_dim = int(2 * hidden_dim / 3)
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, self.hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, self.hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(self.hidden_dim, dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (n, t, d)
        The authors do F.silu(...) for the first part,
        multiplied by a second linear transform, then
        a final linear w3 to revert back to dim.
        """
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def custom_init(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)



class DiTBlock(nn.Module):
    """
    DiT transformer block comprising:
      - Self-attention
      - Cross-attention (with conditioning tokens)
      - An MLP (dense or MoE)
      - 'AdaLN' style conditioning from a pooled vector `c`

    Args:
        dim (int): Dimension for the block's input/output
        head_dim (int): Dimension of each attention head
        mlp_ratio (float): Factor for the MLP hidden dimension
        qkv_ratio (float): Factor for QKV projection dimension
        multiple_of (int): Round hidden dims up to a multiple of this
        pooled_emb_dim (int): Dimension of the pooled conditioning vector (e.g., tabular, textual, etc.)
        norm_eps (float): Epsilon for normalization
        depth_init (bool): Depth-dependent weight initialization
        layer_id (int): Index of this block
        num_layers (int): Total number of blocks
        compress_xattn (bool): Whether to compress cross-attn QKV dims
        use_bias (bool): Whether linear layers have biases
        moe_block (bool): Whether to use Mixture-of-Experts for the MLP
        num_experts (int): Number of experts if using MoE
        expert_capacity (float): Expert capacity factor
    """
    def __init__(
        self,
        dim: int,
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        pooled_emb_dim: int,
        norm_eps: float,
        depth_init: bool,
        layer_id: int,
        num_layers: int,
        compress_xattn: bool,
        use_bias: bool,
        moe_block: bool,
        num_experts: int,
        expert_capacity: float,
    ):
        super().__init__()
        self.dim = dim

        # QKV dimension for self-attention
        if qkv_ratio != 1:
            qkv_hidden_dim = (head_dim * 2) * (
                (int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2)
            )
        else:
            qkv_hidden_dim = dim

        # Hidden dimension for feed-forward
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Norm layers
        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.norm3 = create_norm('layernorm', dim, eps=norm_eps)

        # Self-attention & cross-attention
        # (assuming you have SelfAttention, CrossAttention classes defined).
        self.attn = SelfAttention(
            dim=dim,
            num_heads=(qkv_hidden_dim // head_dim),
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim,
        )
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=(qkv_hidden_dim // head_dim) if compress_xattn else (dim // head_dim),
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=(qkv_hidden_dim if compress_xattn else dim),
        )

        # MLP (either dense or Mixture-of-Experts)
        if moe_block:
            self.mlp = FeedForwardECMoe(
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                dim=dim,
                hidden_dim=mlp_hidden_dim,
                multiple_of=multiple_of
            )
        else:
            self.mlp = FeedForward(
                dim=dim,
                hidden_dim=mlp_hidden_dim,
                multiple_of=multiple_of,
                use_bias=use_bias
            )

        # AdaLN modulation to incorporate the pooled conditioning vector `c`
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(pooled_emb_dim, 6 * dim, bias=True),
        )

        # Depth-dependent init scaling
        if depth_init:
            self.weight_init_std = 0.02 / ((2 * (layer_id + 1)) ** 0.5)
        else:
            self.weight_init_std = 0.02 / ((2 * num_layers) ** 0.5)

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, dim) - main tokens (e.g. image tokens)
        y: (N, S, dim) - conditioning tokens (e.g. tabular or text)
        c: (N, dim)    - "pooled" condition embedding (AdaLN injection)
        """
        # AdaLN produces shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # 1) Self-attention with gating
        x_normed = modulate(self.norm1(x), shift_msa, scale_msa)
        x_attn = self.attn(x_normed)  # (N, T, dim)
        x = x + gate_msa.unsqueeze(1) * x_attn

        # TODO:
        # 2) Cross-attention
        # x = x + self.cross_attn(self.norm2(x), y)

        # 3) MLP
        x_normed2 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_normed2)
        x = x + gate_mlp.unsqueeze(1) * x_mlp

        return x

    def custom_init(self):
        """
        Initialize weights in a custom manner, as the authors do.
        Typically resetting layer norms and applying truncated normal on attention and MLP.
        """
        for norm in (self.norm1, self.norm2, self.norm3):
            norm.reset_parameters()

        self.attn.custom_init(self.weight_init_std)
        self.cross_attn.custom_init(self.weight_init_std)
        self.mlp.custom_init(self.weight_init_std)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies a shift and scale (per-sample) to x before a linear or attention operation.
    shift, scale shape: (N, dim)
    x shape: (N, T, dim)
    """
    # Add dimension for broadcasting
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################################
# MULTIMODAL MODEL
#################################################################################################


class MultiModalDiT(nn.Module):
    """
    # The main class combining image patch embed + patch mixer + a stack of DiTBlocks + tabular pipeline.

    1) self.x_embedder -> PatchEmbed for images
    2) self.t_embedder -> TimestepEmbedder for diffusion time
    3) self.tab_encoder + self.tab_proj + self.tab_block -> turns tabular row into (N,1,dim) token
    4) optional patch_mixer for images
    5) main DiTBlocks (self.blocks)
    6) final_layer for images
    7) tabular_final_proj for the single tab token.

    A multi-modal DiT that handles:
      - Image tokens (via PatchEmbed + patch mixer + transformer blocks)
      - Tabular data as conditioning (via TabularEncoder, optional self-attn,
        used in cross-attn the same way text was in the original code).
      - MoE for both image + tab tokens if chosen in the DiTBlock definition
      - Classifier-free guidance (CFG) for the tabular data.
    """
    def __init__(
        self,
        # Image-related
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        # Transformer backbone
        dim: int = 512,
        depth: int = 16,
        head_dim: int = 32,
        multiple_of: int = 256,
        norm_eps: float = 1e-6,
        # Tabular
        tabular_feature_dim: int = 64,
        tabular_embed_dim: int = 512,
        # Prompt-like block for tabular
        tab_block_head_dim: int = 64,
        tab_block_mlp_ratio: float = 4.0,
        # QKV & FFN multipliers for DiT blocks
        qkv_multipliers: List[float] = [1.0],
        ffn_multipliers: List[float] = [4.0],
        # Patch mixer settings
        use_patch_mixer: bool = True,
        patch_mixer_depth: int = 4,
        patch_mixer_dim: int = 512,
        patch_mixer_qkv_ratio: float = 1.0,
        patch_mixer_mlp_ratio: float = 1.0,
        # MoE config
        use_bias: bool = True,
        num_experts: int = 8,
        expert_capacity: float = 2.0,
        experts_every_n: int = 2,
        mask_ratio: float = 0.0
        # ...
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.use_patch_mixer = use_patch_mixer
        self.mask_ratio = mask_ratio

        # ---------------------------
        # 1) IMAGE PATCH EMBEDDING
        # ---------------------------
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
        )
        num_patches = self.x_embedder.num_patches
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, dim))
        self.base_size = input_size // patch_size

        # ---------------------------
        # 2) TIMESTEP EMBEDDING
        # ---------------------------
        self.t_embedder = TimestepEmbedder(dim, act_layer=nn.GELU)

        # ---------------------------
        # 3) TABULAR PIPELINE
        # ---------------------------
        self.tab_encoder = TabularEncoder(tabular_feature_dim, tabular_embed_dim)
        # Project tab embedding from tabular_embed_dim -> dim
        self.tab_proj = nn.Linear(tabular_embed_dim, dim)
        # Optional mini self-attn block to refine tabular embeddings
        self.tab_block = AttentionBlockTabular(
            dim=dim,
            head_dim=tab_block_head_dim,
            mlp_ratio=tab_block_mlp_ratio,
            multiple_of=multiple_of,
            norm_eps=norm_eps,
            use_bias=use_bias
        )
        # Pool the tab tokens to produce a single vector
        self.tab_pooled_mlp = nn.Sequential(
            create_norm('layernorm', dim, eps=norm_eps),
            nn.Linear(dim, dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=use_bias),
        )

        # ---------------------------
        # 4) PATCH MIXER for IMAGES
        # ---------------------------
        if self.use_patch_mixer:
            # Possibly define some mixer blocks. This is like the original code:
            # "patch_mixer" is a few DiTBlocks with smaller dimension, etc.
            self.patch_mixer = nn.ModuleList()
            expert_blocks_idx = [
                i for i in range(patch_mixer_depth) if (i+1) % experts_every_n == 0
            ]
            is_moe_block = [True if i in expert_blocks_idx else False
                            for i in range(patch_mixer_depth)]

            for i in range(patch_mixer_depth):
                block = DiTBlock(
                    dim=patch_mixer_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio,
                    qkv_ratio=patch_mixer_qkv_ratio,
                    multiple_of=multiple_of,
                    pooled_emb_dim=dim,  # We still use `dim` for conditioning
                    norm_eps=norm_eps,
                    depth_init=False,       # Not worrying about progressive init
                    layer_id=i,
                    num_layers=patch_mixer_depth,
                    compress_xattn=False,
                    use_bias=use_bias,
                    moe_block=is_moe_block[i],
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                )
                self.patch_mixer.append(block)

            # If patch_mixer_dim != dim, define some linear maps
            if patch_mixer_dim != dim:
                self.patch_mixer_map_xin = nn.Sequential(
                    create_norm('layernorm', dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
                self.patch_mixer_map_xout = nn.Sequential(
                    create_norm('layernorm', patch_mixer_dim, eps=norm_eps),
                    nn.Linear(patch_mixer_dim, dim, bias=use_bias)
                )
                self.patch_mixer_map_tab = nn.Sequential(
                    create_norm('layernorm', dim, eps=norm_eps),
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
            else:
                self.patch_mixer_map_xin = nn.Identity()
                self.patch_mixer_map_xout = nn.Identity()
                self.patch_mixer_map_tab = nn.Identity()

        # ---------------------------
        # 5) MAIN DiT BLOCKS
        # ---------------------------
        if len(ffn_multipliers) == depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            # distribute them
            num_splits = len(ffn_multipliers)
            assert depth % num_splits == 0
            depth_per_split = depth // num_splits
            qkv_ratios = list(np.array([
                [m]*depth_per_split for m in qkv_multipliers
            ]).reshape(-1))
            mlp_ratios = list(np.array([
                [m]*depth_per_split for m in ffn_multipliers
            ]).reshape(-1))

        expert_blocks_idx = [i for i in range(depth - 1) if (i+1) % experts_every_n == 0]
        is_moe_block = [True if i in expert_blocks_idx else False for i in range(depth)]

        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                head_dim=head_dim,
                mlp_ratio=mlp_ratios[i],
                qkv_ratio=qkv_ratios[i],
                multiple_of=multiple_of,
                pooled_emb_dim=dim,
                norm_eps=norm_eps,
                depth_init=True,
                layer_id=i,
                num_layers=depth,
                compress_xattn=False,  # cross-attn always uses dim
                use_bias=use_bias,
                moe_block=is_moe_block[i],
                num_experts=num_experts,
                expert_capacity=expert_capacity,
            )
            for i in range(depth)
        ])

        # Mask token (if we want random patch masking)
        self.register_buffer("mask_token", torch.zeros(1, 1, patch_size**2 * in_channels))

        # 6) Final "image" layer
        self.final_layer = T2IFinalLayer(
            in_dim=dim,
            pooled_emb_dim=dim,
            patch_size=patch_size,
            out_channels=in_channels,
            act_layer=nn.GELU,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps)
        )

        # 7) Final "tabular" projection
        self.tabular_final_proj = nn.Sequential(
            create_norm('layernorm', dim, eps=norm_eps),
            nn.Linear(dim, tabular_feature_dim)  # Output same size as input features
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Similar to original:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            pos_interp_scale=1.0,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Optionally init other modules (MLPs, linear layers, etc.)
        # ...
        for block in self.blocks:
            block.custom_init()
        if self.use_patch_mixer:
            for block in self.patch_mixer:
                block.custom_init()

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.in_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x_image, x_tabular, t, cfg=1.0, **kwargs):
        """
        x_image: (N, C, H, W)
        x_tabular: (N, tabular_feature_dim)
        t: (N,) timesteps
        cfg: classifier-free guidance scale
        returns (image_out, tabular_out)
        """

        if x_image.dtype != torch.float32:
            x_image = x_image.float()
        if x_tabular.dtype != torch.float32:
            x_tabular = x_tabular.float()
        if t.dtype != torch.float32:
            t = t.float()

        if cfg != 1.0:
            return self.forward_with_cfg(x_image, x_tabular, t, cfg, **kwargs)
        else:
            return self.forward_without_cfg(x_image, x_tabular, t, **kwargs)

    def forward_with_cfg(self, x_img, x_tab, t, cfg, **kwargs):
        # Duplicate
        x_img = torch.cat([x_img, x_img], dim=0)
        zero_tab = torch.zeros_like(x_tab)
        x_tab = torch.cat([x_tab, zero_tab], dim=0)
        if len(t) > 1:
            t = torch.cat([t, t], dim=0)

        res = self.forward_without_cfg(x_img, x_tab, t, **kwargs)
        image_out, tab_out, mask = res["image_sample"], res["tabular_sample"], res["mask"]

        # Split cond/uncond
        image_cond, image_uncond = torch.split(image_out, image_out.shape[0]//2, dim=0)
        tab_cond, tab_uncond = torch.split(tab_out, tab_out.shape[0]//2, dim=0)

        # CFG
        img_final = image_uncond + cfg * (image_cond - image_uncond)
        tab_final = tab_uncond + cfg * (tab_cond - tab_uncond)
        return {
            'image_sample': img_final,
            'tabular_sample': tab_final,
            'mask': mask
        }

    def forward_without_cfg(
            self,
            x_img: torch.Tensor,  # (N, C, H, W)
            x_tab: torch.Tensor,  # (N, tabular_feature_dim)
            t: torch.Tensor,  # (N,)
            **kwargs
    ):
        """
        x_img, x_tab -> 2 modalities
        We might do random patch masking on the image tokens if mask_ratio > 0.
        We'll skip masking tabular data unless you specifically want that.
        Returns a dict with 'image_sample' and 'tabular_sample' plus 'mask'.
        """

        # ------------------
        # 1) IMAGE -> PATCHES
        # ------------------
        x = self.x_embedder(x_img) + self.pos_embed  # (N, num_patches, dim)

        # Optionally mask the image tokens
        mask = None
        ids_keep = None
        ids_restore = None

        if not self.training:
            self.mask_ratio = 0
        if self.mask_ratio > 0:
            # Suppose you have the same get_mask utility from the original code.
            # get_mask returns a dict with 'mask', 'ids_keep', 'ids_restore'
            mask_dict = get_mask(x.shape[0], x.shape[1], mask_ratio=self.mask_ratio, device=x.device
            )
            mask = mask_dict['mask']
            ids_keep = mask_dict['ids_keep']
            ids_restore = mask_dict['ids_restore']

            # actually apply the mask to the image tokens
            x = mask_out_token(x, ids_keep)  # Now x has fewer tokens

        # ------------------
        # 2) TIME EMBEDDING
        # ------------------
        t_embed = self.t_embedder(t)  # (N, dim)

        # ------------------
        # 3) TABULAR PIPELINE
        # ------------------
        tab_embed = self.tab_encoder(x_tab)  # (N, tabular_embed_dim)
        tab_embed = self.tab_proj(tab_embed).unsqueeze(1)  # (N, 1, dim)
        tab_embed = self.tab_block(tab_embed)  # self-attn block, shape still (N, 1, dim)
        tab_pooled = tab_embed.mean(dim=1)  # (N, dim)
        tab_pooled = self.tab_pooled_mlp(tab_pooled)

        # Add tab_pooled to the time embedding for conditioning
        # TODO:
        # t_embed = t_embed + tab_pooled

        # ------------------
        # 4) PATCH MIXER
        # ------------------
        if self.use_patch_mixer:
            x = self.patch_mixer_map_xin(x)
            tab_mixer = self.patch_mixer_map_tab(tab_embed)
            for block in self.patch_mixer:
                x = block(x, tab_mixer, t_embed)
            x = self.patch_mixer_map_xout(x)

        # ------------------
        # 5) MAIN DiT BLOCKS
        # ------------------
        for block in self.blocks:
            x = block(x, tab_embed, t_embed)

        # ------------------
        # 6) FINAL IMAGE
        # ------------------
        x_img_out = self.final_layer(x, t_embed)  # (N, num_tokens, patch_size^2 * C)

        # If we masked, we need to unmask to get the correct shape
        if self.mask_ratio > 0:
            x_img_out = unmask_tokens(x_img_out, ids_restore, self.mask_token)

        # Reconstruct (N, C, H, W)
        x_img_out = self.unpatchify(x_img_out)

        # ------------------
        # 7) FINAL TABULAR
        # ------------------
        # TODO:
        x_tab_out = self.tabular_final_proj(tab_pooled)

        # Return everything (you can just return the two outputs if you don't need the mask).
        return {
            'image_sample': x_img_out,
            'tabular_sample': x_tab_out,
            'mask': mask
        }


if __name__ == "__main__":
    # 1) Create the model
    model = MultiModalDiT(
        input_size=64,
        patch_size=4,
        in_channels=3,
        dim=128,
        depth=4,
        head_dim=32,
        multiple_of=64,
        norm_eps=1e-6,
        tabular_feature_dim=174,
        tabular_embed_dim=64,
        tab_block_head_dim=16,
        tab_block_mlp_ratio=2.0,
        use_patch_mixer=False,
        mask_ratio=0.0
    )

    # 2) Generate random inputs
    N = 2
    x_image = torch.randn(N, 3, 64, 64)    # e.g. 2 images, 3 channels, 32x32
    x_tab = torch.randn(N, 174)            # e.g. tabular of size 16
    t = torch.randint(0, 1000, (N,))      # random timesteps

    # 3) Forward pass
    res = model(x_image, x_tab, t, cfg=1)
    if res["mask"] is not None:
        print("img_out shape:", res["image_sample"].shape)   # (N, 3, 32, 32)
        print("tab_out shape:", res["tabular_sample"].shape)   # (N, 16)
