import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# 1. Helper Functions
# ------------------------
def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Create a layer normalization module. In Sony's code, they used some variants
    like 'layernorm' vs. 'np_layernorm'. Here we stick to standard layernorm.
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Sony 'modulate': x * (1 + scale) + shift
    shift, scale are shape (B, D), we broadcast them over the sequence dimension.
    """
    # x shape: (B, N, D)
    # shift, scale shape: (B, D)
    # We unsqueeze dim=1 so shift, scale broadcast over tokens
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ------------------------
# 2. Timestep Embedding (AdaLN style)
# ------------------------
class TimestepEmbedder(nn.Module):
    """
    Sony's approach:
    - First create a frequency embedding from the time index t.
    - Pass it through a small MLP.
    - Then each Transformer block uses an 'adaLN_modulation' to chunk and do shift/scale/gate.

    We keep the baseline sinusoidal time embedding but you may adapt as needed.
    """

    def __init__(self, embed_dim: int, frequency_embedding_size: int = 512):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # MLP that converts frequency embedding -> final time embedding
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Standard sinusoidal embedding: for each scalar t, produce a 'dim'-dim vector
        of sin/cos frequencies.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (B,) or (B,1) containing discrete timesteps
        returns: shape (B, embed_dim)
        """
        freq_emb = self.timestep_embedding(t, self.frequency_embedding_size, max_period=10000)
        return self.mlp(freq_emb)


# ------------------------
# 3. Attention / MLP Building Blocks (From Sony’s DiT)
# ------------------------
class SelfAttention(nn.Module):
    """
    Standard multi-head self-attention (scaled_dot_product_attention, PyTorch 2.0).
    We re-center queries/keys with optional small layernorm (Sony's code).
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, norm_eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        # tiny layernorm for q,k (similar to Sony code)
        self.ln_q = create_norm("layernorm", dim, eps=norm_eps)
        self.ln_k = create_norm("layernorm", dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project to qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each => (B, N, num_heads, head_dim)

        # re-center q, k with LN
        q = self.ln_q(q.reshape(B, N, -1)).reshape(B, N, self.num_heads, self.head_dim)
        k = self.ln_k(k.reshape(B, N, -1)).reshape(B, N, self.num_heads, self.head_dim)

        # Dot-product attention (PyTorch 2.0)
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, heads, N, head_dim)
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        )
        # shape => (B, heads, N, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


class CrossAttention(nn.Module):
    """
    Cross-attention from x -> cond.
    We form queries from x, keys/values from cond.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, norm_eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        # small LN for q, k
        self.ln_q = create_norm("layernorm", dim, eps=norm_eps)
        self.ln_k = create_norm("layernorm", dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Nx, C)  => queries
        cond: (B, Ny, C) => keys/values
        returns cross-attended x
        """
        B, Nx, C = x.shape

        # form queries from x
        q = self.q_linear(x).reshape(B, Nx, self.num_heads, self.head_dim)
        # form keys, values from cond
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)  # each => (B, Ny, heads, head_dim)

        # LN
        q = self.ln_q(q.reshape(B, Nx, -1)).reshape(B, Nx, self.num_heads, self.head_dim)
        k = self.ln_k(k.reshape(B, -1, self.num_heads * self.head_dim)) \
            .reshape(B, -1, self.num_heads, self.head_dim)

        # Dot-product attn
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False
        )
        # => (B, heads, Nx, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, Nx, C)
        return self.proj(attn_out)


class FeedForward(nn.Module):
    """
    Simple MLP from Sony code: out = w3( SiLU(w1(x)) * w2(x) )
    """

    def __init__(self, dim: int, hidden_dim: int, use_bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ------------------------
# 4. Image Patch Embedding + Unpatchify
# ------------------------
class PatchEmbed(nn.Module):
    """
    Projects an image [B, in_chans, H, W] into a sequence of patch tokens [B, N_patches, embed_dim].
    This is standard for Vision Transformers.
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=4, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # number of patches
        num_patches_h = img_size // patch_size
        num_patches_w = img_size // patch_size
        self.num_patches = num_patches_h * num_patches_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => (B, in_chans, H, W)
        # output => (B, num_patches, embed_dim)
        out = self.proj(x)  # => (B, embed_dim, H/patch, W/patch)
        out = out.flatten(2).transpose(1, 2)  # => (B, num_patches, embed_dim)
        return out


def unpatchify(x: torch.Tensor, patch_size: int, out_chans: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse the patch embedding.
    x => (B, N, patch_size^2 * out_chans).
    Return => (B, out_chans, H, W).
    """
    B, N, D = x.shape
    h = H // patch_size
    w = W // patch_size
    assert h * w == N, "Mismatch in patch count"
    # reshape => (B, h, w, patch_size, patch_size, out_chans)
    x = x.reshape(B, h, w, patch_size, patch_size, out_chans)
    # reorder => (B, out_chans, h*patch_size, w*patch_size)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.reshape(B, out_chans, H, W)
    return x


# ------------------------
# 5. Tabular Tokenization
# ------------------------
class TabularTokenizer(nn.Module):
    """
    Simplified version of Minkai’s approach:
      - We add one [CLS] numeric token (value=1.0).
      - We have `weight_num` of shape (d_numeric+1, d_token).
      - We optionally have a bias of shape (d_numeric, d_token) (the CLS won't get bias).
      - Output => (B, T_tab, d_token), T_tab = d_numeric + 1.
    """
    def __init__(self, d_numeric: int, d_token: int, bias=True):
        super().__init__()
        self.d_numeric = d_numeric
        self.d_token = d_token

        # [CLS] + numeric columns => total (d_numeric+1) embeddings
        self.weight_num = nn.Parameter(torch.empty(d_numeric + 1, d_token))
        nn.init.kaiming_uniform_(self.weight_num, a=math.sqrt(5))

        if bias:
            # Minkai's code: one bias row per numeric column (excluding [CLS]).
            self.bias = nn.Parameter(torch.empty(d_numeric, d_token))
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        else:
            self.bias = None

    def forward(self, x_tab: torch.Tensor) -> torch.Tensor:
        """
        x_tab: shape (B, d_numeric) => purely numeric columns
        Return => tokens shape (B, d_numeric+1, d_token), with index 0 = [CLS].
        """
        B, Dn = x_tab.shape
        assert Dn == self.d_numeric, "Mismatch in numeric columns"

        # Insert [CLS] at front => shape (B, d_numeric+1)
        # [CLS] can be set to 1.0 or 0.0 – Minkai used 1.0
        x_tab_cls = torch.cat([torch.ones(B, 1, device=x_tab.device), x_tab], dim=1)

        # Multiply each numeric column by its embedding row
        # => shape: (B, d_numeric+1, d_token)
        out_num = x_tab_cls.unsqueeze(-1) * self.weight_num  # broadcast multiply

        if self.bias is not None:
            # bias shape = (d_numeric, d_token)
            # we do not add bias for the [CLS] row => so we can pad it with zeros
            zeros_for_cls = torch.zeros(1, self.d_token, device=x_tab.device)
            bias_full = torch.cat([zeros_for_cls, self.bias], dim=0)  # => (d_numeric+1, d_token)
            out_num = out_num + bias_full.unsqueeze(0)  # broadcast over batch

        return out_num


# ------------------------
# 6. Multimodal Block: image + tab in parallel (with Sony's AdaLN)
# ------------------------
class MultimodalBlock(nn.Module):
    """
    One “parallel” block that processes both:
      - image tokens: self-attn, cross-attn to tab, feed-forward
      - tab tokens: self-attn, cross-attn to image, feed-forward
    with Sony's AdaLN modulation for self-attn & feed-forward.

    We do not apply AdaLN to cross-attention (as per Sony code approach).
    We produce separate LN modules for image vs. tab.
    """

    def __init__(
            self,
            dim: int,
            n_heads: int,
            mlp_ratio: float,
            ada_embed_dim: int,
            norm_eps: float = 1e-6,
            use_bias: bool = True,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # Image path
        self.norm_i1 = create_norm("layernorm", dim, eps=norm_eps)
        self.self_attn_img = SelfAttention(dim, n_heads, qkv_bias=use_bias, norm_eps=norm_eps)

        self.norm_i2 = create_norm("layernorm", dim, eps=norm_eps)
        self.cross_attn_img = CrossAttention(dim, n_heads, qkv_bias=use_bias, norm_eps=norm_eps)

        self.norm_i3 = create_norm("layernorm", dim, eps=norm_eps)
        self.ff_img = FeedForward(dim, hidden_dim, use_bias=use_bias)

        # Tab path
        self.norm_t1 = create_norm("layernorm", dim, eps=norm_eps)
        self.self_attn_tab = SelfAttention(dim, n_heads, qkv_bias=use_bias, norm_eps=norm_eps)

        self.norm_t2 = create_norm("layernorm", dim, eps=norm_eps)
        self.cross_attn_tab = CrossAttention(dim, n_heads, qkv_bias=use_bias, norm_eps=norm_eps)

        self.norm_t3 = create_norm("layernorm", dim, eps=norm_eps)
        self.ff_tab = FeedForward(dim, hidden_dim, use_bias=use_bias)

        # Single linear for all AdaLN modulations for this block:
        # We want 6 scalars for image (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        # and 6 for tab. => total 12 * dim
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(ada_embed_dim, 12 * dim, bias=True),
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor, c: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        x_img: (B, Ni, dim)
        x_tab: (B, Nt, dim)
        c: (B, ada_embed_dim) => time embedding (or pooled embedding) that we chunk
        Return => updated (x_img, x_tab)
        """
        B, _, D = x_img.shape

        # 1) get shift/scale/gate from time embedding c
        #    => shape (B, 12*D)
        ada_vals = self.adaLN_modulation(c)  # (B, 12D)
        # chunk into 12 pieces => each piece is shape (B, D)
        #   [image MSA shift, image MSA scale, image MSA gate, image MLP shift, ...]
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
            shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t \
            = ada_vals.chunk(12, dim=1)

        # ---------------------------
        # Image path
        # 1) Self-attn (with AdaLN)
        xi = modulate(self.norm_i1(x_img), shift_msa_i, scale_msa_i)
        xi = self.self_attn_img(xi)
        x_img = x_img + gate_msa_i.unsqueeze(1) * xi

        # 2) Cross-attn (no AdaLN in original Sony for cross-attn)
        xi = self.norm_i2(x_img)
        xi = self.cross_attn_img(xi, x_tab)
        x_img = x_img + xi

        # 3) Feed-forward (with AdaLN)
        xi = modulate(self.norm_i3(x_img), shift_mlp_i, scale_mlp_i)
        xi = self.ff_img(xi)
        x_img = x_img + gate_mlp_i.unsqueeze(1) * xi

        # ---------------------------
        # Tab path
        # 1) Self-attn (with AdaLN)
        xt = modulate(self.norm_t1(x_tab), shift_msa_t, scale_msa_t)
        xt = self.self_attn_tab(xt)
        x_tab = x_tab + gate_msa_t.unsqueeze(1) * xt

        # 2) Cross-attn (no AdaLN by analogy)
        xt = self.norm_t2(x_tab)
        xt = self.cross_attn_tab(xt, x_img)
        x_tab = x_tab + xt

        # 3) Feed-forward (with AdaLN)
        xt = modulate(self.norm_t3(x_tab), shift_mlp_t, scale_mlp_t)
        xt = self.ff_tab(xt)
        x_tab = x_tab + gate_mlp_t.unsqueeze(1) * xt

        return x_img, x_tab


# ------------------------
# 7. Final “Heads” for output (image & tab)
# ------------------------
class FinalImageLayer(nn.Module):
    """
    Analogous to Sony's T2IFinalLayer: a final AdaLN + linear that maps each image token
    to patch_size^2 * out_chans, which we then unpatchify.
    """

    def __init__(self, dim: int, time_embed_dim: int, patch_size: int, out_chans: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        # linear that maps from (dim) -> patch_size^2 * out_chans
        self.linear = nn.Linear(dim, patch_size * patch_size * out_chans, bias=True)

        # Sony-style AdaLN mod
        # (2*dim for shift,scale)
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_dim, 2 * dim, bias=True)
        )
        # final LN
        self.norm_final = create_norm("layernorm", dim, eps=1e-6)

    def forward(self, x: torch.Tensor, c: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x => (B, N, dim)
        c => (B, time_embed_dim)
        returns => (B, out_chans, H, W)
        """
        B, N, D = x.shape
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_mod = modulate(self.norm_final(x), shift, scale)
        x_patches = self.linear(x_mod)  # => (B, N, patch_size^2 * out_chans)
        # unpatchify
        out_img = unpatchify(x_patches, self.patch_size, self.out_chans, H, W)
        return out_img


class FinalTabLayer(nn.Module):
    """
    A simple final layer that produces a single vector of shape (d_numerical + sum_of_cat),
    or you can do something else. We do an AdaLN + linear on the *CLS* token
    (the first tab token). This matches a typical approach: we treat the first tab token
    as the "pooled" representation.
    """

    def __init__(self, dim: int, time_embed_dim: int, d_out: int):
        super().__init__()
        self.d_out = d_out
        # AdaLN for final
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_dim, 2 * dim, bias=True)
        )
        self.norm_final = create_norm("layernorm", dim, eps=1e-6)
        self.linear = nn.Linear(dim, d_out, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x => (B, T_tab, dim). We use x[:,0,:] as "CLS" to produce final vector.
        c => (B, time_embed_dim)
        return => (B, d_out)
        """
        B, T, D = x.shape
        x_cls = x[:, 0, :]  # take the first token
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_mod = self.norm_final(x_cls) * (1.0 + scale) + shift  # shape (B, D)
        out_tab = self.linear(x_mod)  # (B, d_out)
        return out_tab


# ------------------------
# 8. Full MultimodalDiT
# ------------------------
class MultiModalDiT(nn.Module):
    """
    A combined architecture that:
      - Takes an image (latent or normal) + tabular data + time t
      - Produces final image and final tab vector
        for a diffusion model to compute a loss vs. real (image, tab).
    """

    def __init__(
            self,
            image_size=32,
            patch_size=4,
            in_channels=4,
            dim=256,  # embed dimension
            depth=4,  # number of blocks
            n_heads=4,
            mlp_ratio=4.0,
            d_tab=8,  # number of numeric columns
            time_embed_dim=256,
            d_out_tab=8,  # dimension of final tab vector
    ):
        super().__init__()

        # 1) Timestep embedder
        self.time_embedder = TimestepEmbedder(embed_dim=time_embed_dim, frequency_embedding_size=512)

        # 2) Image patch embedding
        self.img_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = in_channels  # final output channels same as input

        # Learnable position embedding for image patches
        num_patches = self.img_embed.num_patches
        self.pos_embed_img = nn.Parameter(torch.zeros(1, num_patches, dim))

        # 3) Tab tokenizer (numeric only)
        self.tab_tokenizer = TabularTokenizer(d_numeric=d_tab, d_token=dim, bias=True)

        # 4) Blocks
        self.blocks = nn.ModuleList([
            MultimodalBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                ada_embed_dim=time_embed_dim,
            )
            for _ in range(depth)
        ])

        # 5) Final image + tab heads
        self.final_image = FinalImageLayer(
            dim=dim,
            time_embed_dim=time_embed_dim,
            patch_size=patch_size,
            out_chans=in_channels
        )
        self.final_tab = FinalTabLayer(
            dim=dim,
            time_embed_dim=time_embed_dim,
            d_out=d_out_tab
        )

        # init position embedding
        nn.init.normal_(self.pos_embed_img, std=0.02)

    def forward(
            self,
            x_img: torch.Tensor,  # (B, in_chans, H, W)
            x_tab: torch.Tensor,  # (B, d_tab)
            time_scalar: torch.Tensor,  # (B,) integer timesteps
            **kwargs
    ) -> dict:
        """
        x_img: shape (B, in_channels, H, W)
        x_num: (B, d_numerical)
        x_cat: (B, num_categorical)
        t: (B,) - timesteps
        Returns => {"img_out": ..., "tab_out": ...}
        """
        B, C, H, W = x_img.shape
        # 1) time embedding
        t_emb = self.time_embedder(time_scalar)  # => (B, time_embed_dim)

        # 2) Patchify image + add pos embed
        x_i = self.img_embed(x_img)  # => (B, N_patches, dim)
        x_i = x_i + self.pos_embed_img[:, : x_i.shape[1], :]

        # 3) Tokenize tab numeric
        x_t = self.tab_tokenizer(x_tab)  # => (B, d_tab+1, dim)

        # 4) Forward through each block
        for block in self.blocks:
            x_i, x_t = block(x_i, x_t, t_emb)

        # 5) Final image + tab
        img_out = self.final_image(x_i, t_emb, H, W)  # => (B, in_chans, H, W)
        tab_out = self.final_tab(x_t, t_emb)  # => (B, d_out_tab)

        return {
            "img_out": img_out,
            "tab_out": tab_out
        }


from omegaconf import DictConfig
from utils.configurations import apply_overrides
from typing import Any
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
    # Suppose we have:
    B = 2
    H, W = 32, 32
    in_chans = 4
    d_tab = 174  # e.g. 5 numeric columns

    x_img = torch.randn(B, in_chans, H, W)
    x_tab = torch.randn(B, d_tab)
    t_emb_img = torch.full((B,), 0.5, dtype=torch.float32)

    model = MultiModalDiT(
        image_size=H,
        patch_size=4,
        in_channels=in_chans,
        dim=128,
        depth=3,
        n_heads=4,
        mlp_ratio=4.0,
        d_tab=d_tab,
        time_embed_dim=128,
        d_out_tab=5,  # e.g. same dimension as x_tab or whatever you prefer
    )

    out = model(x_img, x_tab, t_emb_img)
    print("img_out shape:", out["img_out"].shape)  # => (B, in_chans, H, W)
    print("tab_out shape:", out["tab_out"].shape)  # => (B, 5)
