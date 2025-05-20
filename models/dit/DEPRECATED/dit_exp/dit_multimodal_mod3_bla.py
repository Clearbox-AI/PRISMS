import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any
# from timm.models.vision_transformer import PatchEmbed as PatchEmbed2D
from timm.models.vision_transformer import PatchEmbed
from omegaconf import DictConfig
from utils.configurations import apply_overrides

def partial_unpatchify_tokens(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Reshape a (B, N, D) sequence of tokens into a 2D (B, D, h, w) feature map.
    N must be h*w. This is a "partial unpatch" to let us do local 2D ops.
    """
    B, N, D = tokens.shape
    assert N == h * w, f"Expected h*w = {h*w}, got N={N}."
    # (B, N, D) -> (B, h, w, D) -> (B, D, h, w)
    x = tokens.view(B, h, w, D).permute(0, 3, 1, 2).contiguous()
    return x

def partial_patchify_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of partial_unpatchify_tokens.
    x: (B, D, h, w) -> (B, h*w, D).
    """
    B, D, h, w = x.shape
    x = x.permute(0, 2, 3, 1).contiguous().view(B, h*w, D)
    return x

class LocalConvRefinement(nn.Module):
    """
    A small local conv-based refiner that operates on (B, D, H, W),
    e.g. to reduce block-edge artifacts. We'll do a few 3x3 convs + GELU.
    """
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        return self.net(x)

class SmallDecoderConv(nn.Module):
    """
    Optional final local conv-based upsampler or refiner to run
    after unpatchifying. E.g., if still in a 32x32 latent space
    or (B, in_chans, 32, 32), we refine it a bit.
    """
    def __init__(self, in_chans: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_chans, H, W)
        return self.net(x)


#########################################
#           HELPER FUNCTIONS
#########################################

def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates a normalization layer (layernorm).
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    else:
        raise ValueError(f"Unsupported norm: {norm_type}")

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    'AdaLN' style modulation: x * (1 + scale) + shift
    x: (B, N, D)
    shift, scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def custom_init_linear(linear: nn.Linear, std: float = 0.02, mean: float = 0.0):
    """
    Custom init for linear layers (trunc_normal_).
    """
    nn.init.trunc_normal_(linear.weight, mean=mean, std=std)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, 0.0)

def custom_init_layernorm(ln: nn.LayerNorm):
    """
    Reset LayerNorm params.
    """
    ln.reset_parameters()

def get_mask(batch_size: int, length: int, mask_ratio: float, device: torch.device):
    """
    Creates a random patch mask (1 => masked, 0 => keep).
    Returns dict with:
      mask, ids_keep, ids_restore
    """
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch_size, length, device=device)
    # Asc sort => small = keep
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch_size, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {
        'mask': mask,
        'ids_keep': ids_keep,
        'ids_restore': ids_restore
    }

def mask_out_token(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """
    Gather only the tokens at 'ids_keep'.
    x: (B, length, dim)
    """
    B, L, D = x.shape
    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )
    return x_masked

def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """
    Reinsert masked tokens as 'mask_token'.
    x: (B, len_keep, D)
    ids_restore: (B, length)
    mask_token: (1,1,D)
    """
    B, len_keep, D = x.shape
    L_full = ids_restore.shape[1]
    # #masked = L_full - len_keep
    mask_tokens = mask_token.repeat(B, L_full - len_keep, 1)
    x_ = torch.cat([x, mask_tokens], dim=1)  # (B, L_full, D)
    x_ = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
    )
    return x_

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Create 2D sin/cos positional embeddings.
    grid_size: The height=width of the patch grid
    Returns array shape (grid_size^2, embed_dim)
    """
    # generate a grid of (row, col)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)      # (2, grid_size, grid_size)

    grid = grid.reshape(2, 1, grid_size*grid_size)  # (2, 1, N)
    grid = grid.astype(np.float32)

    # each is shape (N, embed_dim/2) if we do half dimension for row, half for col
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    grid is (2, 1, N) where N=grid_size^2
    embed_dim must be divisible by 2
    """
    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2
    emb_h = get_1d_sin_cos(grid[0], half_dim)  # (N, half_dim)
    emb_w = get_1d_sin_cos(grid[1], half_dim)  # (N, half_dim)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (N, embed_dim)
    return emb

def get_1d_sin_cos(position: np.ndarray, dim: int) -> np.ndarray:
    """
    From the standard Transformer formula for sin/cos
    position: (1, N) or (N,) shape
    Output shape => (N, dim)
    """
    if position.ndim == 2:
        position = position.squeeze(0)  # shape (N,)
    N = position.shape[0]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pos_emb = np.zeros((N, dim), dtype=np.float32)
    pos_emb[:, 0::2] = np.sin(position[:, None] * div_term)
    pos_emb[:, 1::2] = np.cos(position[:, None] * div_term)
    return pos_emb

# 1D version for tabular data (like a single row splitted into patches)
def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """
    Returns shape (length, embed_dim).
    """
    positions = np.arange(length, dtype=np.float32)
    return get_1d_sin_cos(positions, embed_dim)

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Args:
        hidden_size (int): Size of hidden dimension
        act_layer (Any): Activation layer constructor
        frequency_embedding_size (int, 512): Size of frequency embedding
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
        """Create sinusoidal timestep embeddings."""
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
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

#########################################
#         PATCH EMBEDDINGS
#########################################

class PatchEmbed2D(nn.Module):
    """
    2D patch embedding for images.
    Typically used in the original DiT code (like timm's PatchEmbed).
    We define a minimal version here.
    """

    def __init__(
            self,
            img_height: int,
            img_width: int,
            patch_size: int,
            in_chans: int,
            embed_dim: int,
            bias: bool = True
    ):
        super().__init__()
        assert img_height % patch_size == 0 and img_width % patch_size == 0, \
            "Image dimension must be divisible by patch_size."
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # A conv with kernel_size=patch_size, stride=patch_size
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=bias
        )
        # number of patches is (img_height/patch_size * img_width/patch_size)
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, in_chans, img_height, img_width)
        Returns: shape (B, num_patches, embed_dim)
        """
        # shape after proj => (B, embed_dim, H//p, W//p)
        x = self.proj(x)
        # flatten spatial => (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

    def custom_init(self, init_std: float = 0.02):
        """
        Custom init for the conv.
        """
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)


class TabularPatchEmbed(nn.Module):
    """
    For tabular data of shape (B, table_width).
    We'll chunk the row into patches of length 'patch_size'.
    Then embed each chunk with a Linear => (B, #patches, embed_dim).
    """
    def __init__(
        self,
        table_width: int,
        patch_size: int,
        embed_dim: int,
        bias: bool = True
    ):
        super().__init__()
        assert (table_width % patch_size)==0, "table_width must be multiple of patch_size"
        self.table_width = table_width
        self.patch_size = patch_size
        self.num_patches = table_width // patch_size
        self.embed_dim = embed_dim

        # Project each chunk of size 'patch_size' into 'embed_dim'
        self.proj = nn.Linear(patch_size, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, table_width)
        => (B, num_patches, embed_dim)
        """
        B, W = x.shape
        # chunk => (B, num_patches, patch_size)
        x = x.view(B, self.num_patches, self.patch_size)
        # pass each chunk through linear => (B, num_patches, embed_dim)
        x = self.proj(x)
        return x

    def custom_init(self, init_std: float=0.02):
        custom_init_linear(self.proj, std=init_std)


#########################################
#     INITIAL IMAGE / TABULAR BLOCKS
#########################################

class LightImageConditionBlock(nn.Module):
    """
    A simple single self-attention + feed-forward block for image tokens.
    This is analogous to a minimal 'transformer block' used to refine the image tokens.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
        use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            bias=use_bias, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        hidden_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, D)
        # 1) Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # 2) Feed-forward
        x_norm = self.norm2(x)
        x = x + self.ff(x_norm)
        return x


# ------------------------------------------------------------------
# Example minimal MLP for tabular condition
class TabularConditionMLP(nn.Module):
    """
    A simple MLP that processes tabular tokens.
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        use_bias: bool = True,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, D)
        x = self.norm(x)
        x = self.net(x)
        return x


#########################################
#             PATCH MIXER
#########################################

class ImgPatchMixerBlock(nn.Module):
    """
    A patch mixer block for images with cross-attn and MLP.
    Now accepts a conditioning dimension `cond_dim` (e.g. main model dim, 256)
    so that the AdaLN modulation layer expects a vector of size cond_dim.
    """
    def __init__(
        self,
        dim: int,         # patch mixer token dimension (e.g. patch_mixer_dim_img)
        cond_dim: int,    # conditioning dimension (main model dim, e.g. 256)
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        norm_eps: float,
        layer_id: int,
        num_layers: int,
        depth_init: bool,
        use_bias: bool
    ):
        super().__init__()
        qkv_dim = int(dim * qkv_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=(qkv_dim // head_dim),
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim
        )
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=(qkv_dim // head_dim),
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim
        )
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.norm3 = create_norm('layernorm', dim, eps=norm_eps)

        self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)
        # The modulation layer takes in the conditioning embedding in main model space.
        self.adaLN_mod = nn.Sequential(
            nn.GELU(),
            nn.Linear(cond_dim, 6 * dim, bias=True)
        )

        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5
            if depth_init else
            0.02 / (2 * num_layers) ** 0.5
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c is the conditioning embedding (e.g. t_emb_img) in main model space.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_mod(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + self.cross_attn(self.norm2(x), y)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x

    def custom_init(self):
        # TODO: Custom initialization logic if needed.
        pass

class TabPatchMixerBlock(nn.Module):
    """
    A patch mixer block for tabular data (no cross-attn) that uses AdaLN modulation.
    It accepts a conditioning dimension `cond_dim` (e.g. main model dim, 256)
    so that the modulation layer uses the unmodified time embedding.
    """
    def __init__(
        self,
        dim: int,         # patch mixer token dimension for tab data (e.g. patch_mixer_dim_tab)
        cond_dim: int,    # conditioning dimension (main model dim, e.g. 256)
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        norm_eps: float,
        layer_id: int,
        num_layers: int,
        depth_init: bool,
        use_bias: bool
    ):
        super().__init__()
        qkv_dim = int(dim * qkv_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Self-attention branch (with AdaLN modulation)
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=(qkv_dim // head_dim),
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim
        )
        # Cross-attention branch to incorporate image conditioning
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=(qkv_dim // head_dim),
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim
        )
        # MLP branch
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)

        # AdaLN modulation layer – produces 6*dim parameters from the conditioning vector
        self.adaLN_mod = nn.Sequential(
            nn.GELU(),
            nn.Linear(cond_dim, 6 * dim, bias=True)
        )

        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5
            if depth_init else
            0.02 / (2 * num_layers) ** 0.5
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TabPatchMixerBlock.

        Args:
            x: Tabular patch tokens of shape (B, N, dim)
            y: Image conditioning tokens of shape (B, M, dim) to be used in cross-attention
            c: Conditioning vector (time and/or table-related) of shape (B, cond_dim)

        Returns:
            Updated tabular patch tokens of shape (B, N, dim)
        """
        # Compute modulation parameters from conditioning vector.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_mod(c).chunk(6, dim=1)

        # Self-attention branch with AdaLN modulation.
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # Cross-attention branch incorporating image conditioning tokens.
        x = x + self.cross_attn(self.norm2(x), y)
        # MLP branch with AdaLN modulation.
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

    def custom_init(self):
        # TODO: Custom initialization logic if needed.
        pass


def build_img_patch_mixer(
    depth: int,
    dim: int,
    cond_dim: int,  # main model dimension (e.g. 256)
    head_dim: int,
    mlp_ratio: float,
    qkv_ratio: float,
    multiple_of: int,
    norm_eps: float,
    depth_init: bool,
    use_bias: bool,
    base_layer_id: int = 0,
):
    blocks = nn.ModuleList()
    for i in range(depth):
        blk = ImgPatchMixerBlock(
            dim=dim,
            cond_dim=cond_dim,  # pass main model dimension here
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            qkv_ratio=qkv_ratio,
            multiple_of=multiple_of,
            norm_eps=norm_eps,
            layer_id=(base_layer_id + i),
            num_layers=depth,
            depth_init=depth_init,
            use_bias=use_bias
        )
        blocks.append(blk)
    return blocks


def build_tab_patch_mixer(
    depth: int,
    dim: int,
    cond_dim: int,  # main model dimension (e.g. 256)
    head_dim: int,
    mlp_ratio: float,
    qkv_ratio: float,
    multiple_of: int,
    norm_eps: float,
    depth_init: bool,
    use_bias: bool,
    base_layer_id: int = 0
):
    blocks = nn.ModuleList()
    for i in range(depth):
        blk = TabPatchMixerBlock(
            dim=dim,
            cond_dim=cond_dim,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            qkv_ratio=qkv_ratio,
            multiple_of=multiple_of,
            norm_eps=norm_eps,
            layer_id=(base_layer_id + i),
            num_layers=depth,
            depth_init=depth_init,
            use_bias=use_bias
        )
        blocks.append(blk)
    return blocks


#########################################
#         FEED-FORWARD LAYERS
#########################################

class FeedForward(nn.Module):
    """
    Basic feed-forward with 'SiLU' activation,
    from the original code logic:
      hidden_dim ~ mlp_ratio * dim
      then w1, w2, w3 as in the snippet
    """

    def __init__(self, dim, hidden_dim, multiple_of=256, use_bias=True):
        super().__init__()
        # the original code does hidden_dim = int(2*hidden_dim / 3) then round up to multiple_of
        h_dim = int(2 * hidden_dim / 3)
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, h_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, h_dim, bias=use_bias)
        self.w3 = nn.Linear(h_dim, dim, bias=use_bias)

    def forward(self, x: torch.Tensor):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def custom_init(self, init_std: float = 0.02):
        custom_init_linear(self.w1, std=init_std)
        custom_init_linear(self.w2, std=init_std)
        custom_init_linear(self.w3, std=init_std)


class FeedForwardECMoe(nn.Module):
    """
    Expert-Choice style Mixture-of-Experts from original code.
    """

    def __init__(
            self,
            num_experts: int,
            expert_capacity: float,
            dim: int,
            hidden_dim: int,
            multiple_of: int
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        h_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Parameter(torch.ones(num_experts, dim, h_dim))
        self.w2 = nn.Parameter(torch.ones(num_experts, h_dim, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (N, T, D)
        """
        n, t, d = x.shape
        tokens_per_expert = int(self.expert_capacity * t / self.num_experts)

        scores = self.gate(x)  # (n, t, e)
        probs = F.softmax(scores, dim=-1)  # (n, t, e)
        # topk along the time dimension => k = tokens_per_expert
        g, m = torch.topk(probs.permute(0, 2, 1), tokens_per_expert, dim=-1)
        # m => indices
        # p => one-hot
        p = F.one_hot(m, num_classes=t).float()  # (n, e, k, t)

        xin = torch.einsum('nekt, ntd->nekd', p, x)  # (n, e, k, d)
        h = torch.einsum('nekd, edf->nekf', xin, self.w1)  # => (n, e, k, h_dim)
        h = self.gelu(h)
        h = torch.einsum('nekf, efd->nekd', h, self.w2)  # => (n, e, k, d)

        out = g.unsqueeze(-1) * h
        out = torch.einsum('nekt, nekd->ntd', p, out)
        return out

    def custom_init(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


#########################################
#        ATTENTION LAYERS
#########################################

class SelfAttention(nn.Module):
    """
    Basic self-attention for each modality (image or table).
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
        assert hidden_dim % num_heads == 0

        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.ln_q = create_norm('layernorm', hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('layernorm', hidden_dim, eps=norm_eps)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # LN on q, k
        q = self.ln_q(q.reshape(B, N, -1)).reshape(B, N, self.num_heads, self.head_dim)
        k = self.ln_k(k.reshape(B, N, -1)).reshape(B, N, self.num_heads, self.head_dim)

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        )
        x = attn_out.transpose(1, 2).contiguous().reshape(B, N, self.hidden_dim)
        x = self.proj(x)
        return x

    def custom_init(self, init_std: float = 0.02):
        # for the qkv and proj
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0.0)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)
        # LN
        custom_init_layernorm(self.ln_q)
        custom_init_layernorm(self.ln_k)


class CrossAttention(nn.Module):
    """
    Cross-attention from x->cond.
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
        assert hidden_dim % num_heads == 0

        self.q_linear = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.ln_q = create_norm('layernorm', hidden_dim, eps=norm_eps)
        self.ln_k = create_norm('layernorm', hidden_dim, eps=norm_eps)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Nx, dim) -> queries
        cond: (B, Ny, dim) -> keys, values
        """
        B, Nx, C = x.shape
        q = self.q_linear(x).reshape(B, Nx, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        # LN
        q = self.ln_q(q.reshape(B, Nx, -1)).reshape(B, Nx, self.num_heads, self.head_dim)
        k = self.ln_k(k.reshape(B, -1, self.num_heads * self.head_dim)).reshape(
            B, -1, self.num_heads, self.head_dim
        )

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        )
        x = attn_out.transpose(1, 2).contiguous().reshape(B, Nx, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x

    def custom_init(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.q_linear.weight, mean=0.0, std=0.02)
        if self.q_linear.bias is not None:
            nn.init.constant_(self.q_linear.bias, 0.0)
        nn.init.trunc_normal_(self.kv_linear.weight, mean=0.0, std=0.02)
        if self.kv_linear.bias is not None:
            nn.init.constant_(self.kv_linear.bias, 0.0)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)
        custom_init_layernorm(self.ln_q)
        custom_init_layernorm(self.ln_k)


#########################################
#    MULTI-MODAL DiT BLOCK
#########################################

class DiTBlockMultiModal(nn.Module):
    """
    Symmetrical block that updates x_img and x_tab in parallel:
     - x_img: self-attn, cross-attn from x_tab, MLP
     - x_tab: self-attn, cross-attn from x_img, MLP
     - Distinct AdaLN mod for each domain
    """

    def __init__(
            self,
            dim_img: int,
            dim_tab: int,
            head_dim: int,
            mlp_ratio_img: float,
            mlp_ratio_tab: float,
            qkv_ratio_img: float,
            qkv_ratio_tab: float,
            multiple_of: int,
            norm_eps: float,
            depth_init: bool,
            layer_id: int,
            num_layers: int,
            use_bias: bool,
            moe_block_img: bool,
            moe_block_tab: bool,
            num_experts: int,
            expert_capacity: float
    ):
        super().__init__()

        # -- Image sub-block
        qkv_dim_img = int(dim_img * qkv_ratio_img)
        self.norm1_img = create_norm('layernorm', dim_img, eps=norm_eps)
        self.attn_img = SelfAttention(
            dim=dim_img,
            num_heads=qkv_dim_img // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim_img
        )
        self.cross_attn_img = CrossAttention(
            dim=dim_img,
            num_heads=qkv_dim_img // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim_img
        )
        self.norm2_img = create_norm('layernorm', dim_img, eps=norm_eps)
        self.norm3_img = create_norm('layernorm', dim_img, eps=norm_eps)

        hidden_img = int(dim_img * mlp_ratio_img)
        if moe_block_img:
            self.mlp_img = FeedForwardECMoe(
                num_experts, expert_capacity, dim_img, hidden_img, multiple_of
            )
        else:
            self.mlp_img = FeedForward(dim_img, hidden_img, multiple_of, use_bias)

        self.adaLN_mod_img = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_img, 6 * dim_img, bias=True)
        )

        # -- Table sub-block
        qkv_dim_tab = int(dim_tab * qkv_ratio_tab)
        self.norm1_tab = create_norm('layernorm', dim_tab, eps=norm_eps)
        self.attn_tab = SelfAttention(
            dim=dim_tab,
            num_heads=qkv_dim_tab // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim_tab
        )
        self.cross_attn_tab = CrossAttention(
            dim=dim_tab,
            num_heads=qkv_dim_tab // head_dim,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_dim_tab
        )
        self.norm2_tab = create_norm('layernorm', dim_tab, eps=norm_eps)
        self.norm3_tab = create_norm('layernorm', dim_tab, eps=norm_eps)

        hidden_tab = int(dim_tab * mlp_ratio_tab)
        if moe_block_tab:
            self.mlp_tab = FeedForwardECMoe(
                num_experts, expert_capacity, dim_tab, hidden_tab, multiple_of
            )
        else:
            self.mlp_tab = FeedForward(dim_tab, hidden_tab, multiple_of, use_bias)

        self.adaLN_mod_tab = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_tab, 6 * dim_tab, bias=True)
        )

        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5 if depth_init
            else 0.02 / (2 * num_layers) ** 0.5
        )

    def forward(
            self,
            x_img: torch.Tensor,  # (B, Ni, dim_img)
            x_tab: torch.Tensor,  # (B, Nt, dim_tab)
            t_img: torch.Tensor,  # (B, dim_img)
            t_tab: torch.Tensor  # (B, dim_tab)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ============ Image path =============
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_mod_img(t_img).chunk(6, dim=1)

        # (1) Self-attn
        x_img = x_img + gate_msa_i.unsqueeze(1) * self.attn_img(
            modulate(self.norm1_img(x_img), shift_msa_i, scale_msa_i)
        )
        # (2) cross-attn from x_img -> x_tab
        x_img = x_img + self.cross_attn_img(
            self.norm2_img(x_img),
            x_tab
        )
        # (3) MLP
        x_img = x_img + gate_mlp_i.unsqueeze(1) * self.mlp_img(
            modulate(self.norm3_img(x_img), shift_mlp_i, scale_mlp_i)
        )

        # ============ Table path =============
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_mod_tab(t_tab).chunk(6, dim=1)

        # (1) Self-attn
        x_tab = x_tab + gate_msa_t.unsqueeze(1) * self.attn_tab(
            modulate(self.norm1_tab(x_tab), shift_msa_t, scale_msa_t)
        )
        # (2) cross-attn from x_tab -> x_img
        x_tab = x_tab + self.cross_attn_tab(
            self.norm2_tab(x_tab),
            x_img
        )
        # (3) MLP
        x_tab = x_tab + gate_mlp_t.unsqueeze(1) * self.mlp_tab(
            modulate(self.norm3_tab(x_tab), shift_mlp_t, scale_mlp_t)
        )

        return x_img, x_tab

    def custom_init(self):
        """
        Apply custom init to all sub-layers.
        """
        # norms
        for ln in (self.norm1_img, self.norm2_img, self.norm3_img,
                   self.norm1_tab, self.norm2_tab, self.norm3_tab):
            custom_init_layernorm(ln)

        # attn
        self.attn_img.custom_init(self.weight_init_std)
        self.cross_attn_img.custom_init(self.weight_init_std)
        self.attn_tab.custom_init(self.weight_init_std)
        self.cross_attn_tab.custom_init(self.weight_init_std)

        # mlp
        if hasattr(self.mlp_img, 'custom_init'):
            self.mlp_img.custom_init(self.weight_init_std)
        if hasattr(self.mlp_tab, 'custom_init'):
            self.mlp_tab.custom_init(self.weight_init_std)

        # adaLN_mod
        for mod in self.adaLN_mod_img:
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)
        for mod in self.adaLN_mod_tab:
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)


#########################################
# FINAL LAYERS + UNPATCHIFY
#########################################

class FinalLayer(nn.Module):
    """
    Minimal T2IFinalLayer-like.
    Goes from (B, N, hidden_size) -> (B, N, out_token_size)
    with an AdaLN mod using a "time_emb_dim" sized vector.
    """

    def __init__(self, hidden_size: int, out_token_size: int, time_emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_token_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 2*hidden_size, bias=True)
        )
        self.norm_final = create_norm('layernorm', hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def custom_init(self, init_std: float = 0.02):
        custom_init_linear(self.linear, std=init_std)
        if isinstance(self.norm_final, nn.LayerNorm):
            custom_init_layernorm(self.norm_final)
        for mod in self.adaLN_modulation:
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)


def unpatchify_image(tokens: torch.Tensor, patch_size: int, in_chans: int, height: int, width: int):
    """
    tokens: (B, N, patch_size^2 * in_chans) => (B, in_chans, height, width)
    """
    B, L, _ = tokens.shape
    p = patch_size
    hp = height // p
    wp = width // p

    tokens = tokens.view(B, hp, wp, p, p, in_chans)
    # or permute(0, 5, 1, 3, 2, 4) depending on your dimension order:
    tokens = tokens.permute(0, 5, 1, 3, 2, 4).contiguous()
    return tokens.view(B, in_chans, height, width)


def unpatchify_table_1d(tokens: torch.Tensor, patch_size: int, total_width: int):
    """
    tokens: (B, n_tab_patches, patch_size) => (B, total_width)
    """
    B, L, C = tokens.shape
    assert C == patch_size
    assert L * patch_size == total_width
    return tokens.view(B, total_width)


#########################################
#       MULTI-MODAL DIT MODEL
#########################################

class MultiModalDiT(nn.Module):
    """
    Final symmetrical multi-modal DiT architecture
    that can generate both images and tables.

    A symmetrical multi-modal DiT that can:
     - embed images (2D) and tables (1D)
     - optionally mask patches for each domain
     - run a stack of multi-modal blocks (self/cross-attn) on both
     - produce final predicted noise for each domain
     - handle classifier-free guidance by domain toggles
    """

    def __init__(
            self,
            # Image config
            img_in_channels: int,
            img_height: int,
            img_width: int,
            img_patch_size: int,

            # Table config
            table_width: int,
            table_patch_size: int,

            # Shared backbone config
            hidden_dim: int = 256,
            depth: int = 6,
            head_dim: int = 64,
            mlp_ratio_img: float = 4.0,
            mlp_ratio_tab: float = 4.0,
            qkv_ratio_img: float = 1.0,
            qkv_ratio_tab: float = 1.0,
            multiple_of: int = 256,
            use_bias: bool = True,
            depth_init: bool = True,
            experts_every_n: int = 2,
            num_experts: int = 8,
            expert_capacity: float = 1.0,

            # patch mixers
            use_patch_mixer_img: bool = True,
            patch_mixer_depth_img: int = 2,
            patch_mixer_dim_img: int = 128,
            patch_mixer_mlp_ratio_img: float = 4.0,
            patch_mixer_qkv_ratio_img: float = 1.0,

            use_patch_mixer_tab: bool = True,
            patch_mixer_depth_tab: int = 2,
            patch_mixer_dim_tab: int = 128,
            patch_mixer_mlp_ratio_tab: float = 4.0,
            patch_mixer_qkv_ratio_tab: float = 1.0
    ):
        super().__init__()

        self.img_in_channels = img_in_channels
        self.img_height = img_height
        self.img_width = img_width
        self.img_patch_size = img_patch_size
        self.table_width = table_width
        self.table_patch_size = table_patch_size
        self.hidden_dim = hidden_dim

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_embedder = TimestepEmbedder(hidden_dim, approx_gelu)

        self.img_patch_embed = PatchEmbed(
            img_height, img_patch_size, img_in_channels, hidden_dim, bias=True
        )
        self.img_num_patches = self.img_patch_embed.num_patches

        # -- 2) Table patch embed
        self.tab_patch_embed = TabularPatchEmbed(
            table_width=table_width,
            patch_size=table_patch_size,
            embed_dim=hidden_dim
        )
        self.tab_num_patches = self.tab_patch_embed.num_patches

        # -- (A) Positional Embeddings
        # We'll store them as buffers and fill with sin/cos in initialize_weights()
        self.register_buffer(
            "pos_embed_img",
            torch.zeros(1, self.img_num_patches, hidden_dim),
            persistent=False
        )
        self.register_buffer(
            "pos_embed_tab",
            torch.zeros(1, self.tab_num_patches, hidden_dim),
            persistent=False
        )

        # -- 3) Condition "preprocessing" blocks
        # For images: a light transformer block to refine image tokens
        num_heads_img = hidden_dim // head_dim
        self.img_emb_preprocess = LightImageConditionBlock(
            dim=hidden_dim,
            num_heads=num_heads_img,
            mlp_ratio=2.0,  # or 4.0, up to you
            norm_eps=1e-6,
            use_bias=use_bias
        )
        # For tables: a simpler MLP
        self.tab_emb_preprocess = TabularConditionMLP(
            dim=hidden_dim,
            mlp_ratio=2.0,  # or 4.0
            use_bias=use_bias,
            norm_eps=1e-6
        )

        # -- 4) Pooled embedding MLP => merges into time embedding
        # We'll do a short MLP for each
        def create_pooled_mlp():
            # Simple two-layer MLP to produce a single vector in hidden_dim
            return nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=1e-6),
                nn.Linear(hidden_dim, hidden_dim, bias=use_bias),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim, bias=use_bias),
            )

        self.pooled_img_emb_process = create_pooled_mlp()
        self.pooled_tab_emb_process = create_pooled_mlp()

        # ---- TRY: local refiners for image tokens ----
        # We'll do a partial unpatch of the (B, N, D) tokens to (B, D, h, w).
        # For a 32x32 image with patch_size=2, N=256 => h=16, w=16.
        self.h_img = img_height // img_patch_size
        self.w_img = img_width // img_patch_size

        self.local_refiner_img_1 = LocalConvRefinement(dim=hidden_dim, num_layers=2)
        self.local_refiner_img_2 = LocalConvRefinement(dim=hidden_dim, num_layers=2)
        # Optional final local conv if you want to refine (B, C, H, W) after unpatchify:
        self.small_decoder_conv = SmallDecoderConv(in_chans=img_in_channels, num_layers=2)
        # ---- TRY: local refiners for image tokens ----

        # -- 5) Patch mixers
        # image
        self.use_patch_mixer_img = use_patch_mixer_img
        if self.use_patch_mixer_img:
            if patch_mixer_dim_img != hidden_dim:
                self.patch_mixer_map_xin_img = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-6),
                    nn.Linear(hidden_dim, patch_mixer_dim_img, bias=use_bias)
                )
                self.patch_mixer_map_xout_img = nn.Sequential(
                    nn.LayerNorm(patch_mixer_dim_img, eps=1e-6),
                    nn.Linear(patch_mixer_dim_img, hidden_dim, bias=use_bias)
                )
                # This is how you could pass a “condition” to the patch mixer
                self.patch_mixer_map_y_img = nn.Sequential(
                    create_norm('layernorm', hidden_dim),
                    nn.Linear(hidden_dim, patch_mixer_dim_img, bias=use_bias)
                )
            else:
                self.patch_mixer_map_xin_img = nn.Identity()
                self.patch_mixer_map_xout_img = nn.Identity()
                self.patch_mixer_map_y_img = nn.Identity()

            # build the actual stack, referencing some external function
            self.patch_mixer_img = nn.ModuleList(
                build_img_patch_mixer(
                    depth=patch_mixer_depth_img,
                    dim=patch_mixer_dim_img,
                    cond_dim=hidden_dim,  # used if you do cross-attn
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio_img,
                    qkv_ratio=patch_mixer_qkv_ratio_img,
                    multiple_of=multiple_of,
                    norm_eps=1e-6,
                    depth_init=depth_init,
                    use_bias=use_bias,
                    base_layer_id=0
                )
            )
        else:
            self.patch_mixer_img = None

        # table
        self.use_patch_mixer_tab = use_patch_mixer_tab
        if self.use_patch_mixer_tab:
            if patch_mixer_dim_tab != hidden_dim:
                self.patch_mixer_map_xin_tab = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-6),
                    nn.Linear(hidden_dim, patch_mixer_dim_tab, bias=use_bias)
                )
                self.patch_mixer_map_xout_tab = nn.Sequential(
                    nn.LayerNorm(patch_mixer_dim_tab, eps=1e-6),
                    nn.Linear(patch_mixer_dim_tab, hidden_dim, bias=use_bias)
                )
                self.patch_mixer_map_y_tab = nn.Sequential(
                    create_norm('layernorm', hidden_dim),
                    nn.Linear(hidden_dim, patch_mixer_dim_tab, bias=use_bias)
                )
            else:
                self.patch_mixer_map_xin_tab = nn.Identity()
                self.patch_mixer_map_xout_tab = nn.Identity()
                self.patch_mixer_map_y_tab = nn.Identity()

            self.patch_mixer_tab = nn.ModuleList(
                build_tab_patch_mixer(
                    depth=patch_mixer_depth_tab,
                    dim=patch_mixer_dim_tab,
                    cond_dim=hidden_dim,
                    head_dim=head_dim,
                    mlp_ratio=patch_mixer_mlp_ratio_tab,
                    qkv_ratio=patch_mixer_qkv_ratio_tab,
                    multiple_of=multiple_of,
                    norm_eps=1e-6,
                    depth_init=depth_init,
                    use_bias=use_bias,
                    base_layer_id=0
                )
            )
        else:
            self.patch_mixer_tab = None

        # -- 6) Build main multi-modal DiT blocks
        expert_blocks_idx = [i for i in range(depth - 1) if (i + 1) % experts_every_n == 0]
        is_moe_block_img = [i in expert_blocks_idx for i in range(depth)]
        is_moe_block_tab = [i in expert_blocks_idx for i in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = DiTBlockMultiModal(
                dim_img=hidden_dim,
                dim_tab=hidden_dim,
                head_dim=head_dim,
                mlp_ratio_img=mlp_ratio_img,
                mlp_ratio_tab=mlp_ratio_tab,
                qkv_ratio_img=qkv_ratio_img,
                qkv_ratio_tab=qkv_ratio_tab,
                multiple_of=multiple_of,
                norm_eps=1e-6,
                depth_init=depth_init,
                layer_id=i,
                num_layers=depth,
                use_bias=use_bias,
                moe_block_img=is_moe_block_img[i],
                moe_block_tab=is_moe_block_tab[i],
                num_experts=num_experts,
                expert_capacity=expert_capacity
            )
            self.blocks.append(blk)

        # -- 7) Final layers
        out_img_token_size = (img_patch_size ** 2) * img_in_channels
        out_tab_token_size = table_patch_size

        self.final_layer_img = FinalLayer(hidden_dim, out_img_token_size, time_emb_dim=hidden_dim)
        self.final_layer_tab = FinalLayer(hidden_dim, out_tab_token_size, time_emb_dim=hidden_dim)

        # -- 8) Mask tokens
        self.register_buffer("mask_token_img", torch.zeros(1, 1, out_img_token_size))
        self.register_buffer("mask_token_tab", torch.zeros(1, 1, out_tab_token_size))

        # done, init
        self.initialize_weights()

    def initialize_weights(self):
        # time embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # patch embeddings
        # self.img_patch_embed.custom_init()
        self.tab_patch_embed.custom_init()

        # fill pos_embed_img with 2D sin/cos
        gs_img = int(self.img_num_patches ** 0.5)
        pos2d = get_2d_sincos_pos_embed(self.hidden_dim, gs_img)
        self.pos_embed_img[0, :, :] = torch.from_numpy(pos2d)

        # fill pos_embed_tab with 1D sin/cos
        pos1d_tab = get_1d_sincos_pos_embed(self.hidden_dim, self.tab_num_patches)
        self.pos_embed_tab[0, :, :] = torch.from_numpy(pos1d_tab)

        # patch mixers
        if self.use_patch_mixer_img and self.patch_mixer_img is not None:
            for blk in self.patch_mixer_img:
                blk.custom_init()
        if self.use_patch_mixer_tab and self.patch_mixer_tab is not None:
            for blk in self.patch_mixer_tab:
                blk.custom_init()

        # main blocks
        for blk in self.blocks:
            blk.custom_init()

        # final layers
        self.final_layer_img.custom_init()
        self.final_layer_tab.custom_init()

        # light blocks
        for p in self.img_emb_preprocess.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.tab_emb_preprocess.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # pooled mlps
        for m in [self.pooled_img_emb_process, self.pooled_tab_emb_process]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    ####################################################
    #  forward_without_cfg
    ####################################################
    def forward_without_cfg(
        self,
        x_img: torch.Tensor,  # (B, C, H, W)
        x_tab: torch.Tensor,  # (B, table_width) or (B, 1, 1, table_width), depending on patch embed
        t_emb_img: torch.Tensor,
        t_emb_tab: torch.Tensor,
        mask_ratio_img: float = 0.0,
        mask_ratio_tab: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass without classifier-free guidance.
        With the "project out after masking" approach from the original code.
        """
        B, device = x_img.shape[0], x_img.device

        # 1.1) Patchify image
        img_tokens = self.img_patch_embed(x_img)  # (B, Nimg, D)

        # (A) local refine #1
        x_2d = partial_unpatchify_tokens(img_tokens, self.h_img, self.w_img)
        x_2d = self.local_refiner_img_1(x_2d)
        img_tokens = partial_patchify_tokens(x_2d)

        # 1.2) Patchify table
        tab_tokens = self.tab_patch_embed(x_tab)  # (B, Ntab, D)

        # 2) Add positional embeddings (if shape matches)
        if img_tokens.shape[1] == self.img_num_patches:
            img_tokens = img_tokens + self.pos_embed_img[:, :self.img_num_patches, :]
        if tab_tokens.shape[1] == self.tab_num_patches:
            tab_tokens = tab_tokens + self.pos_embed_tab[:, :self.tab_num_patches, :]

        # 3) Time embeddings
        t_emb_img = self.t_embedder(t_emb_img)  # (B, D)
        t_emb_tab = self.t_embedder(t_emb_tab)  # (B, D)

        # 4) Condition Preprocess
        img_tokens = self.img_emb_preprocess(img_tokens)  # light transf block
        tab_tokens = self.tab_emb_preprocess(tab_tokens)  # MLP

        # 5) Pooled => add to time embedding
        pooled_img = img_tokens.mean(dim=1)  # (B, D)
        pooled_img = self.pooled_img_emb_process(pooled_img)
        t_emb_img = t_emb_img + pooled_img

        pooled_tab = tab_tokens.mean(dim=1)  # (B, D)
        pooled_tab = self.pooled_tab_emb_process(pooled_tab)
        t_emb_tab = t_emb_tab + pooled_tab

        # 6) Patch Mixers
        # (A) Image side
        if self.use_patch_mixer_img and self.patch_mixer_img is not None:
            img_tokens = self.patch_mixer_map_xin_img(img_tokens)
            pooled_tab = self.patch_mixer_map_y_img(pooled_tab.unsqueeze(1))
            for blk in self.patch_mixer_img:
                img_tokens = blk(img_tokens, pooled_tab.squeeze(1), t_emb_img)
            # Mask AFTER patch mixer => save compute
            mask_img, ids_restore_img = None, None
            if mask_ratio_img > 0.0:
                n_img_patches = img_tokens.shape[1]
                mask_dict_img = get_mask(B, n_img_patches, mask_ratio_img, device)
                mask_img = mask_dict_img['mask']
                ids_keep_img = mask_dict_img['ids_keep']
                ids_restore_img = mask_dict_img['ids_restore']
                img_tokens = mask_out_token(img_tokens, ids_keep_img)
            img_tokens = self.patch_mixer_map_xout_img(img_tokens)
        else:
            # if no patch mixer, do masking directly
            mask_img, ids_restore_img = None, None
            if mask_ratio_img > 0.0:
                n_img_patches = img_tokens.shape[1]
                mask_dict_img = get_mask(B, n_img_patches, mask_ratio_img, device)
                mask_img = mask_dict_img['mask']
                ids_keep_img = mask_dict_img['ids_keep']
                ids_restore_img = mask_dict_img['ids_restore']
                img_tokens = mask_out_token(img_tokens, ids_keep_img)

        # (B) Table side
        if self.use_patch_mixer_tab and self.patch_mixer_tab is not None:
            tab_tokens = self.patch_mixer_map_xin_tab(tab_tokens)
            pooled_img = self.patch_mixer_map_y_tab(pooled_img.unsqueeze(1))
            for blk in self.patch_mixer_tab:
                tab_tokens = blk(tab_tokens, pooled_img, t_emb_tab)
            mask_tab, ids_restore_tab = None, None
            if mask_ratio_tab > 0.0:
                n_tab_patches = tab_tokens.shape[1]
                mask_dict_tab = get_mask(B, n_tab_patches, mask_ratio_tab, device)
                mask_tab = mask_dict_tab['mask']
                ids_keep_tab = mask_dict_tab['ids_keep']
                ids_restore_tab = mask_dict_tab['ids_restore']
                tab_tokens = mask_out_token(tab_tokens, ids_keep_tab)
            tab_tokens = self.patch_mixer_map_xout_tab(tab_tokens)
        else:
            mask_tab, ids_restore_tab = None, None
            if mask_ratio_tab > 0.0:
                n_tab_patches = tab_tokens.shape[1]
                mask_dict_tab = get_mask(B, n_tab_patches, mask_ratio_tab, device)
                mask_tab = mask_dict_tab['mask']
                ids_keep_tab = mask_dict_tab['ids_keep']
                ids_restore_tab = mask_dict_tab['ids_restore']
                tab_tokens = mask_out_token(tab_tokens, ids_keep_tab)

        # 7) Main multi-modal DiT blocks
        for blk in self.blocks:
            img_tokens, tab_tokens = blk(img_tokens, tab_tokens, t_emb_img, t_emb_tab)

        # (B) local refine #2 for image
        x_2d = partial_unpatchify_tokens(img_tokens, self.h_img, self.w_img)
        x_2d = self.local_refiner_img_2(x_2d)
        img_tokens = partial_patchify_tokens(x_2d)

        # 8) Final => predicted noise tokens
        img_pred_tokens = self.final_layer_img(img_tokens, t_emb_img)
        tab_pred_tokens = self.final_layer_tab(tab_tokens, t_emb_tab)

        # 9) Unmask
        if mask_ratio_img > 0.0 and ids_restore_img is not None:
            img_pred_tokens = unmask_tokens(img_pred_tokens, ids_restore_img, self.mask_token_img)
        if mask_ratio_tab > 0.0 and ids_restore_tab is not None:
            tab_pred_tokens = unmask_tokens(tab_pred_tokens, ids_restore_tab, self.mask_token_tab)

        # 10) Unpatchify
        sample_img = unpatchify_image(
            img_pred_tokens,
            patch_size=self.img_patch_size,
            in_chans=self.img_in_channels,
            height=self.img_height,
            width=self.img_width
        )
        sample_tab = unpatchify_table_1d(
            tab_pred_tokens,
            patch_size=self.table_patch_size,
            total_width=self.table_width
        )

        # (C) optional final local conv in (B, C, H, W)
        sample_img = self.small_decoder_conv(sample_img)

        return {
            'sample_img': sample_img,
            'sample_tab': sample_tab,
            'mask_img': mask_img,
            'mask_tab': mask_tab
        }

    # -------------------------------------------------
    def forward_with_cfg(
            self,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            t_emb_img: torch.Tensor,
            t_emb_tab: torch.Tensor,
            cfg_scale: float,
            cond_image: bool = True,
            cond_table: bool = True,
            mask_ratio_img: float = 0.0,
            mask_ratio_tab: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Classifier-free guidance pass with domain toggles.
        """
        B = x_img.shape[0]

        # 1) Build unconditional inputs
        x_img_uncond = torch.zeros_like(x_img) if cond_image else x_img
        x_tab_uncond = torch.zeros_like(x_tab) if cond_table else x_tab

        # 2) Concatenate so we have "cond" batch first, "uncond" batch second => shape (2B, ...)
        x_img_all = torch.cat([x_img, x_img_uncond], dim=0)
        x_tab_all = torch.cat([x_tab, x_tab_uncond], dim=0)

        # 3) For time embeddings, replicate only if shape != (1,)
        #    This exactly matches the single‐modal approach:
        if t_emb_img.shape[0] != 1:
            t_emb_img_all = torch.cat([t_emb_img, t_emb_img], dim=0)  # (2B,)
        else:
            t_emb_img_all = t_emb_img  # remains (1,)

        if t_emb_tab.shape[0] != 1:
            t_emb_tab_all = torch.cat([t_emb_tab, t_emb_tab], dim=0)  # (2B,)
        else:
            t_emb_tab_all = t_emb_tab  # remains (1,)

        # 4) Single pass through forward_without_cfg with batch size = 2B
        out_all = self.forward_without_cfg(
            x_img_all,
            x_tab_all,
            t_emb_img_all,
            t_emb_tab_all,
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab
        )

        sample_img_all = out_all['sample_img']  # (2B, ...)
        sample_tab_all = out_all['sample_tab']  # (2B, ...)

        # 4) Split the result back into "cond" and "uncond"
        img_cond, img_uncond = sample_img_all[:B], sample_img_all[B:]
        tab_cond, tab_uncond = sample_tab_all[:B], sample_tab_all[B:]

        # 5) Merge them with classifier-free guidance
        final_img = img_uncond + cfg_scale * (img_cond - img_uncond)
        final_tab = tab_uncond + cfg_scale * (tab_cond - tab_uncond)

        return {
            'sample_img': final_img,
            'sample_tab': final_tab
        }

    # -------------------------------------------------
    def forward(
            self,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            t_emb_img: torch.Tensor,
            t_emb_tab: torch.Tensor,
            cfg: float = 1.0,
            cond_image: bool = True,
            cond_table: bool = True,
            mask_ratio_img: float = 0.0,
            mask_ratio_tab: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Routes to with/without CFG.
        """
        if cfg == 1.0:
            return self.forward_without_cfg(
                x_img, x_tab,
                t_emb_img, t_emb_tab,
                mask_ratio_img=mask_ratio_img,
                mask_ratio_tab=mask_ratio_tab
            )
        else:
            return self.forward_with_cfg(
                x_img, x_tab,
                t_emb_img, t_emb_tab,
                cfg_scale=cfg,
                cond_image=cond_image,
                cond_table=cond_table,
                mask_ratio_img=mask_ratio_img,
                mask_ratio_tab=mask_ratio_tab
            )

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


if __name__ == "__main__":

    # Instantiate the model
    net = MultiModalDiT(
        img_in_channels=4,
        img_height=32,
        img_width=32,
        img_patch_size=2,  # => 8x8 = 64 patches
        table_width=16,
        table_patch_size=2,  # => 8 patches
        hidden_dim=256,
        depth=24,
        head_dim=32,
        mlp_ratio_img=4.0,
        mlp_ratio_tab=4.0,
        qkv_ratio_img=1.0,
        qkv_ratio_tab=1.0,
        multiple_of=256,
        use_bias=True,
        depth_init=True
    )

    # Create dummy inputs
    B = 4
    x_img = torch.randn(B, 4, 32, 32)
    x_tab = torch.randn(B, 16)
    t_emb_img = torch.full((B,), 0.5, dtype=torch.float32)
    t_emb_tab = torch.full((B,), 0.5, dtype=torch.float32)

    # # Forward pass without CFG and with patch masking
    # mask_ratio_img = 0.3
    # mask_ratio_tab = 0.2
    # out_no_cfg = net(
    #     x_img, x_tab,
    #     t_emb_img, t_emb_tab,
    #     cfg=1.0,
    #     mask_ratio_img=mask_ratio_img,
    #     mask_ratio_tab=mask_ratio_tab
    # )
    # print("== No CFG, mask 30% img & 20% tab patches ==")
    # print("sample_img shape:", out_no_cfg["sample_img"].shape)
    # print("sample_tab shape:", out_no_cfg["sample_tab"].shape)
    # # 'mask_img' and 'mask_tab' will exist only in no-CFG scenario
    # print("mask_img shape:",
    #       out_no_cfg["mask_img"].shape if out_no_cfg["mask_img"] is not None else None)
    # print("mask_tab shape:",
    #       out_no_cfg["mask_tab"].shape if out_no_cfg["mask_tab"] is not None else None)

    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # Forward pass with CFG: image->table scenario
    out_cfg = net(
        x_img, x_tab,
        t_emb_img, t_emb_tab,
        cfg=4.0,
        cond_image=True,
        cond_table=False,
        mask_ratio_img=0.0,
        mask_ratio_tab=0.0
    )
    print("\n== CFG (image->table): cond_image=True, cond_table=False, scale=4.0 ==")
    print("sample_img shape:", out_cfg["sample_img"].shape)
    print("sample_tab shape:", out_cfg["sample_tab"].shape)
    # 'mask_img' and 'mask_tab' won't be returned here because it's forward_with_cfg

    # Cross-Conditional CFG: both modalities conditioned
    out_cross = net(
        x_img, x_tab,
        t_emb_img, t_emb_tab,
        cfg=5.0,
        cond_image=True,
        cond_table=True
    )
    print("\n== Cross-Conditional: cond_image=True, cond_table=True, scale=5.0 ==")
    print("sample_img shape:", out_cross["sample_img"].shape)
    print("sample_tab shape:", out_cross["sample_tab"].shape)

    # --- Testing Patch Mixer & MoE ---
    # Print out which main blocks use MoE for each modality.
    print("\n== Testing MoE in Multi-modal Blocks ==")
    for i, blk in enumerate(net.blocks):
        # Example: if your DiTBlockMultiModal has .mlp_img / .mlp_tab attributes
        mlp_img_type = type(blk.mlp_img).__name__
        mlp_tab_type = type(blk.mlp_tab).__name__
        print(f"Block {i}: Image MLP -> {mlp_img_type}, Table MLP -> {mlp_tab_type}")

    # Print out the patch mixer block types (if used)
    if net.use_patch_mixer_img and net.patch_mixer_img is not None:
        print("\n== Image Patch Mixer Blocks ==")
        for i, blk in enumerate(net.patch_mixer_img):
            print(f"Image Patch Mixer Block {i}: {blk.__class__.__name__}")
    if net.use_patch_mixer_tab and net.patch_mixer_tab is not None:
        print("\n== Table Patch Mixer Blocks ==")
        for i, blk in enumerate(net.patch_mixer_tab):
            print(f"Table Patch Mixer Block {i}: {blk.__class__.__name__}")

    # Additionally, test a version of the model with patch mixer disabled.
    net_no_pm = MultiModalDiT(
        img_in_channels=4,
        img_height=32,
        img_width=32,
        img_patch_size=4,
        table_width=16,
        table_patch_size=2,
        hidden_dim=256,
        depth=6,
        head_dim=64,
        mlp_ratio_img=4.0,
        mlp_ratio_tab=4.0,
        qkv_ratio_img=1.0,
        qkv_ratio_tab=1.0,
        multiple_of=256,
        use_bias=True,
        depth_init=True,
        use_patch_mixer_img=False,
        use_patch_mixer_tab=False
    )
    out_no_pm = net_no_pm(x_img, x_tab, t_emb_img, t_emb_tab, cfg=1.0)
    print("\n== Model without Patch Mixer ==")
    print("sample_img shape:", out_no_pm["sample_img"].shape)
    print("sample_tab shape:", out_no_pm["sample_tab"].shape)