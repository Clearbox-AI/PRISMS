# -*- coding: utf-8 -*-
"""
Multi-modal DiT for images and tabular data (token-per-column / optional VAE),
with bidirectional cross-attention, an optional mini-ViT-style patch mixer, and optional MoE.

File layout:
  1) Imports & utilities (norm layers, modulation, initialization helpers)
  2) Generic modules (MoE FFN, minimal TransformerEncoder, Attention, FF)
  3) Tabular components (MDN head, TabSynVAE)
  4) Vision components (PatchEmbed, masking, FinalLayer)
  5) Multimodal DiT blocks (Img→Tab and Tab→Img) plus tokenizer / tab head
  6) PatchMixer (conditional mini-ViT)
  7) MultiModalDiT model + config loader + standalone example

Note: initialization and gating choices follow the DiT/AdaLN-Zero literature
(see details and references at the end of the file).
"""

# -----------------------------------------------------------------------------
# 1) IMPORTS & UTILITIES
# -----------------------------------------------------------------------------
import math
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# ---- Normalization factory --------------------------------------------------
def create_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """
    Create a normalization layer:
      - 'layernorm'  : LayerNorm with affine parameters
      - 'np_layernorm'/'adanorm': LayerNorm without affine (more neutral for AdaLN)
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
    elif norm_type in {"np_layernorm", "adanorm"}:
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

# ---- Modulation helper ------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applica modulazione stile AdaLN: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ---- Initialization helpers for AdaLN projections ---------------------------
def _scale_shift_weights(module: nn.Module, attr: str, dim: int) -> None:
    """
    Scale the weights of the Linear that produces [shift|scale] by 1/sqrt(dim)
    to start close to the identity (stabilizes AdaLN).
    Also supports nn.Sequential(..., nn.Linear).
    """
    lin_mod = getattr(module, attr)
    if isinstance(lin_mod, nn.Sequential):
        lin_mod = lin_mod[-1]
    assert isinstance(lin_mod, nn.Linear), f"{attr} must end with nn.Linear"
    factor = 1.0 / math.sqrt(dim)
    with torch.no_grad():
        lin_mod.weight.mul_(factor)
        if lin_mod.bias is not None:
            lin_mod.bias.mul_(factor)

def _zero_gate(module: nn.Module, attr: str) -> None:
    """
    Zero the rows corresponding to the last third of out_features for the Linear `attr`.
    In this codebase, this initially closes one of the three gates produced by the
    [shift, scale, gate]×3 projection, implementing a 'start-closed gate' behavior.
    """
    proj_mod = getattr(module, attr)
    if isinstance(proj_mod, nn.Sequential):
        proj_mod = proj_mod[-1]
    if not isinstance(proj_mod, nn.Linear):
        return
    if proj_mod.out_features % 3:
        return  # not a 3·D projection; no gate in the expected format
    D = proj_mod.out_features // 3
    with torch.no_grad():
        proj_mod.weight[2 * D:, :].zero_()
        if proj_mod.bias is not None:
            proj_mod.bias[2 * D:].zero_()

# -----------------------------------------------------------------------------
# 2) GENERIC MODULES
# -----------------------------------------------------------------------------

class FeedForwardECMoe(nn.Module):
    """
    Expert-Choice style MoE: each expert is an MLP [D→H→D] with dense softmax routing.
    """
    def __init__(self, num_experts, expert_capacity, dim, hidden_dim, multiple_of, init_std=0.02):
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
        self.aux_loss: Optional[torch.Tensor] = None

        nn.init.trunc_normal_(self.gate.weight, std=init_std)
        nn.init.trunc_normal_(self.w1, std=init_std)
        nn.init.trunc_normal_(self.w2, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if self.training and hasattr(self, "step_counter"):
            scale = min(self.step_counter.item() / 500.0, 1.0)
        else:
            scale = 1.0

        scores = scale * self.gate(x)
        probs = F.softmax(scores, dim=-1)  # (B,T,E)

        # Auxiliary loss to encourage balanced expert utilization
        with torch.no_grad():
            importance = probs.sum(dim=(0, 1))  # (E,)
            self.aux_loss = (importance * importance).sum() * self.num_experts / (B * T)

        h = torch.einsum('btd,edh->bteh', x, self.w1)  # (B,T,E,H)
        h = self.gelu(h)
        h = torch.einsum('bteh,ehd->bted', h, self.w2)  # (B,T,E,D)
        return (probs.unsqueeze(-1) * h).sum(dim=2)  # (B,T,D)


class TransformerEncoderLayer(nn.Module):
    """
    Encoder transformer minimale (LayerNorm → MHA → residuo → MLP → residuo).
    """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        return x + self.mlp(h)


# ---- Attention & Feed-Forward ------------------------------------------------
class CrossAttention(nn.Module):
    """
    Cross-attention: queries from `x` (e.g., images), keys/values from `cond` (e.g., tabular).
    Computed with PyTorch SDPA (stable and internally scaled).
    """
    def __init__(self, dim, num_heads, qkv_bias=True, hidden_dim=None, init_std=0.02):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        assert hidden_dim % num_heads == 0
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, 2 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=qkv_bias)

        self.init_std = init_std
        self._init()

    def _init(self) -> None:
        for p in (self.q_linear, self.kv_linear, self.proj):
            nn.init.trunc_normal_(p.weight, std=self.init_std)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0.)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T_x, D)
        cond: (B, T_c, D)
        """
        B, T_x, _ = x.shape
        T_c = cond.shape[1]

        q = self.q_linear(x)             # (B, T_x, H)
        kv = self.kv_linear(cond)        # (B, T_c, 2H)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, T_x, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_c, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_c, self.num_heads, self.head_dim).transpose(1, 2)

        # Run SDPA in fp32 for stability, then cast back
        attn_out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                  dropout_p=0.0, is_causal=False)
        out = attn_out.to(q.dtype).transpose(1, 2).reshape(B, T_x, self.num_heads * self.head_dim)
        return self.proj(out)


class SelfAttention(nn.Module):
    """Self‑attention multi‑testa standard (SDPA)."""
    def __init__(self, dim, num_heads, qkv_bias=True, hidden_dim=None, init_std=0.02):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        assert hidden_dim % num_heads == 0
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(dim, 3 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, dim)
        self.init_std = init_std
        self._init()

    def _init(self) -> None:
        for p in (self.qkv, self.proj):
            nn.init.trunc_normal_(p.weight, std=self.init_std)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B,T,3H)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                  dropout_p=0.0, is_causal=False)
        out = attn_out.to(q.dtype).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    FFN stile SwiGLU‑like: w3( SiLU(w1(x)) ⊙ w2(x) ).
    La dimensione nascosta segue la formula (2/3)*mlp_ratio*D e viene allineata a `multiple_of`.
    """
    def __init__(self, dim, hidden_dim, multiple_of=256, use_bias=True, init_std=0.02):
        super().__init__()
        h = int(2 * hidden_dim / 3)
        h = multiple_of * ((h + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, h, bias=use_bias)
        self.w2 = nn.Linear(dim, h, bias=use_bias)
        self.w3 = nn.Linear(h, dim, bias=use_bias)

        self.init_std = init_std
        for p in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(p.weight, mean=0.0, std=self.init_std)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# -----------------------------------------------------------------------------
# 3) TABULAR COMPONENTS (MDN + TabSyn-style VAE)
# -----------------------------------------------------------------------------

class NumericMDNHead(nn.Module):
    """
    Mixture Density Network (MDN) head for a single numeric column: projects to
    [mu, log_sigma, logit_pi] for K mixture components and applies variance constraints
    and temperature scaling on mixture weights (classic MDN setup).
    """
    def __init__(self, in_dim: int, n_mixtures: int = 3,
                 sigma_min: float = 2e-2, sigma_max: float = 5.0, pi_temp: float = 1.0):
        super().__init__()
        self.n_mix = int(n_mixtures)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.pi_temp = float(pi_temp)

        self.proj = nn.Linear(in_dim, 3 * self.n_mix)  # [mu|log_sigma|logit_pi]
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma, logit_pi = self.proj(h).chunk(3, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        if self.pi_temp != 1.0:
            logit_pi = logit_pi / self.pi_temp
        pi = torch.softmax(logit_pi, dim=-1)
        return mu, sigma, pi


class FinalTabLatentHead(nn.Module):
    """
    Per-column latent head: produce d_token for each tab token.
    Returns (B, n_tokens, d_token) which we flatten upstream.
    """
    def __init__(self, in_dim: int, time_emb_dim: int, n_tokens: int, d_token: int, act=nn.SiLU, eps=1e-6):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        self.ada = nn.Sequential(act(), nn.Linear(time_emb_dim, 2*in_dim))
        self.out = nn.Linear(in_dim, d_token)
        nn.init.trunc_normal_(self.out.weight, std=0.02 / math.sqrt(in_dim)); nn.init.constant_(self.out.bias, 0.)

    def forward(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # tokens = [CLS + per-column]; drop CLS
        cols = tokens[:, 1:, :]
        cols = self.norm(cols)
        shift, scale = self.ada(t_emb).chunk(2, dim=1)
        cols = cols * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out(cols) # (B, n_tokens, d_token)


class TabSynVAE(nn.Module):
    """
    Token-per-column VAE (categorical + numeric):
      - Encoder: categorical soft-embedding (p @ W), numeric via a small MLP 1→d_token
      - Backbone: Transformer over [CLS + tokens]
      - Latent: per-token diagonal Gaussian (μ, logσ²)
      - Decoder: categorical logits; numeric MDN (K components), trained with NLL
    """
    def __init__(self,
                 cat_dims: List[int],
                 num_numeric: int,
                 d_token: int = 4,
                 hidden_dim: int = 128,
                 depth: int = 2,
                 num_heads: int = 4,
                 beta: float = 0.5,
                 num_mixtures: int = 3,
                 dropout_p: float = 0.0,
                 kl_free_bits: float = 0.0,
                 eps: float = 1e-6,
                 mdn_sigma_min: float = 2e-2,
                 mdn_sigma_max: float = 5.0,
                 mdn_pi_temp: float = 1.25,
                 mdn_sample_scale: float = 1.0,
                 mdn_kind: str = "gaussian_mixture"):
        super().__init__()
        # Inference behavior: stochastic MDN sampling for numeric columns
        self.stochastic_inference: bool = True

        self.cat_dims = cat_dims
        self.num_numeric = num_numeric
        self.n_tokens = len(cat_dims) + num_numeric
        self.d_token = d_token
        self.beta = float(beta)
        self.kl_free_bits = float(kl_free_bits)
        self.num_mixtures = int(num_mixtures)
        self.eps = eps
        self.mdn_sample_scale = float(mdn_sample_scale)
        self.mdn_kind = str(mdn_kind)

        # Categorical: soft embedding (p @ W, no bias)
        self.cat_proj = nn.ModuleList([nn.Linear(k, d_token, bias=False) for k in cat_dims])

        # Numeric: small MLP 1→d_token
        self.num_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_token), nn.GELU(), nn.Linear(d_token, d_token))
            for _ in range(num_numeric)
        ])

        for m in self.cat_proj:
            nn.init.xavier_uniform_(m.weight)
        for mlp in self.num_proj:
            for sub in mlp.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    nn.init.constant_(sub.bias, 0.)

        # Token encoder/decoder
        self.to_enc = nn.Linear(d_token, hidden_dim)
        self.to_dec = nn.Linear(d_token, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n_tokens, hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(depth)])
        self.dec_layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(depth)])

        # Per-token latent head
        self.mu_logvar = nn.Linear(hidden_dim, 2 * d_token)
        nn.init.trunc_normal_(self.mu_logvar.weight, std=0.02)
        nn.init.constant_(self.mu_logvar.bias, 0.)

        # Reconstruction head
        self.cat_heads = nn.ModuleList([nn.Linear(hidden_dim, k) for k in cat_dims])
        self.num_heads = nn.ModuleList([
            NumericMDNHead(hidden_dim, num_mixtures, sigma_min=mdn_sigma_min,
                           sigma_max=mdn_sigma_max, pi_temp=mdn_pi_temp)
            for _ in range(num_numeric)
        ])
        for hh in list(self.cat_heads):
            nn.init.trunc_normal_(hh.weight, std=0.02)
            nn.init.constant_(hh.bias, 0.)

        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    # ---------- helpers ----------
    def _split_cat_num(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Split the concatenated model vector [cat_logits || numeric] into per-column tensors."""
        n_cat = sum(self.cat_dims)
        cat_logits = x[:, :n_cat]
        num = x[:, n_cat:]
        cats, off = [], 0
        for k in self.cat_dims:
            cats.append(cat_logits[:, off:off + k])
            off += k
        return cats, num

    def _build_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        cats, num = self._split_cat_num(x)
        toks = []
        # Categorical: soft embedding using probabilities
        for slc, proj in zip(cats, self.cat_proj):
            p = torch.softmax(slc, dim=-1)
            toks.append(proj(p))
        # Numeric: MLP 1→d_token
        for j, mlp in enumerate(self.num_proj):
            toks.append(mlp(num[:, j].unsqueeze(-1)))
        toks = torch.stack(toks, dim=1)  # (B, T, d_token)

        h = self.to_enc(toks)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos[:, :h.size(1) + 1]
        for layer in self.enc_layers:
            h = layer(h)
        return self.norm(h)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return z, mu, logvar as (B, T, d)."""
        h = self._build_tokens(x)
        per_tok = h[:, 1:, :]  # drop CLS
        mu_log = self.mu_logvar(per_tok)
        mu, logv = mu_log.chunk(2, dim=-1)
        std = (0.5 * logv).exp()
        z = mu + torch.randn_like(std) * std
        return z, mu, logv

    def decode(self, z_tokens: torch.Tensor):
        B, T, _ = z_tokens.shape
        h = self.to_dec(z_tokens)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos[:, :T + 1]
        for layer in self.dec_layers:
            h = layer(h)
        h = self.norm(h)

        cols = h[:, 1:, :]
        outs_cat, outs_num, idx = [], [], 0
        for head in self.cat_heads:
            outs_cat.append(head(cols[:, idx, :]))
            idx += 1
        for head in self.num_heads:
            mu, sigma, pi = head(cols[:, idx, :])
            outs_num.append((mu, sigma, pi))
            idx += 1
        return cols, outs_cat, outs_num

    def loss_forward(self, x: torch.Tensor, beta: Optional[float] = None, free_bits: Optional[float] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE loss over model-space vector x: total = CE_cats + NLL_nums + beta*KL.
        Returns: (total, ce_cat, nll_num, kl)
        """
        z, mu, logv = self.encode(x)
        _, cats_hat, nums_hat = self.decode(z)
        cats, nums = self._split_cat_num(x)

        # Categorical: soft cross-entropy with a softened target p = softmax(x_cat)
        ce = 0.0
        for slc, pred in zip(cats, cats_hat):
            p = torch.softmax(slc, dim=-1).clamp_min(1e-6)
            ce = ce + torch.sum(-p * torch.log_softmax(pred, dim=-1), dim=-1).mean()

        # Numeric: MDN NLL (gaussian/logistic)
        nll = 0.0
        for j, (mu_j, sigma_j, pi_j) in enumerate(nums_hat):
            y = nums[:, j].unsqueeze(-1)  # (B,1)
            if self.mdn_kind.startswith("logistic"):
                u = (y - mu_j) / (sigma_j + 1e-8)
                log_probs = -torch.log(sigma_j + 1e-8) - (u + 2.0 * F.softplus(-u))
            else:
                log_probs = -0.5 * ((y - mu_j) ** 2 / (sigma_j ** 2 + 1e-8)) \
                            - torch.log(sigma_j + 1e-8) - 0.5 * math.log(2 * math.pi)
            log_mix = torch.logsumexp(torch.log(pi_j + 1e-8) + log_probs, dim=-1)
            nll = nll - log_mix.mean()

        # KL with per-latent free-bits
        kl_per = -0.5 * (1 + logv - mu.pow(2) - logv.exp())  # (B,T,d)
        fb = self.kl_free_bits if (free_bits is None) else float(free_bits)
        if fb > 0.0:
            kl_per = torch.clamp(kl_per - fb, min=0.0)
        kl = kl_per.sum(dim=(1, 2)).mean()

        b = self.beta if (beta is None) else float(beta)
        loss = ce + nll + b * kl
        return loss, ce.detach(), nll.detach(), kl.detach()

    # --- convenience methods ---
    def encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = self.encode(x)
        return z.reshape(x.size(0), -1)

    def encode_flat_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latents μ (useful for diffusion training on tabular data)."""
        _, mu, _ = self.encode(x)
        return mu.reshape(x.size(0), -1)

    def decode_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Decode flattened latents → model-space vector: categorical logits + numeric features."""
        B, T = z_flat.size(0), self.n_tokens
        z = z_flat.view(B, T, self.d_token)
        _, cats_hat, nums_hat = self.decode(z)

        outs = [pred for pred in cats_hat]
        for (mu_j, sigma_j, pi_j) in nums_hat:
            if (not self.training) and getattr(self, "stochastic_inference", True):
                # Sample component ~ Categorical(pi), then sample from that component
                idx = torch.multinomial(pi_j.clamp_min(1e-8), num_samples=1).squeeze(-1)  # (B,)
                mu_sel = mu_j.gather(1, idx.unsqueeze(-1))
                sig_sel = sigma_j.gather(1, idx.unsqueeze(-1))
                scale = float(getattr(self, "mdn_sample_scale", 1.0))
                if self.mdn_kind.startswith("logistic"):
                    u = torch.rand_like(mu_sel).clamp(1e-6, 1 - 1e-6)
                    eps = torch.log(u) - torch.log(1.0 - u)  # Logistic(0,1)
                    y = mu_sel + (sig_sel * scale) * eps
                else:
                    y = mu_sel + (sig_sel * scale) * torch.randn_like(mu_sel)
                outs.append(y)
            else:
                outs.append((pi_j * mu_j).sum(dim=-1, keepdim=True))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def quick_metrics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Lightweight post-pretrain diagnostics: categorical accuracy, numeric MAE, variance ratio.
        Useful as an end-of-epoch sanity check.
        """
        zmu = self.encode_flat_mean(x)
        recon = self.decode_flat(zmu)
        cats, nums = self._split_cat_num(x)
        cats_r, nums_r = self._split_cat_num(recon)

        acc_sum, acc_cnt = 0.0, 0
        for slc_t, slc_p in zip(cats, cats_r):
            acc_sum += (slc_t.argmax(dim=-1) == slc_p.argmax(dim=-1)).float().mean()
            acc_cnt += 1
        cat_acc = torch.tensor(0.0, device=x.device) if acc_cnt == 0 else (acc_sum / acc_cnt)

        if nums.numel():
            num_mae = torch.mean(torch.abs(nums - nums_r))
            vr = (nums_r.var(dim=0, unbiased=False) / (nums.var(dim=0, unbiased=False) + 1e-8)).mean()
        else:
            num_mae = torch.tensor(0.0, device=x.device)
            vr = torch.tensor(1.0, device=x.device)
        return {"cat_acc": cat_acc, "num_mae": num_mae, "num_var_ratio": vr}

# -----------------------------------------------------------------------------
# 4) VISION COMPONENTS (patchify, masking, final head)
# -----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding followed by an MLP (standard in diffusion models).
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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
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


class PatchEmbed(nn.Module):
    """Patchify immagini via Conv2d stride=patch → (B, T, D)."""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="linear")
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # (B,T,D)


def get_mask(batch: int, length: int, mask_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """MAE‑like random masking sugli indici dei token (0=keep, 1=mask)."""
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.rand(batch, length, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    mask = torch.ones([batch, length], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {'mask': mask, 'ids_keep': ids_keep, 'ids_restore': ids_restore}

def mask_out_token(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    index = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=index)

def unmask_tokens(x: torch.Tensor, ids_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    B, L_keep, D = x.shape
    L = ids_restore.shape[1]
    mask_tokens = mask_token.repeat(B, L - L_keep, 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    return torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))


class FinalLayer(nn.Module):
    """
    Head finale immagine: AdaLN + Linear → (B, T, p^2*C_out).
    """
    def __init__(self, in_dim, time_emb_dim, patch_size, out_chans, act_layer, norm_layer):
        super().__init__()
        self.norm = norm_layer
        self.adaLN_modulation = nn.Sequential(act_layer(), nn.Linear(time_emb_dim, 2 * in_dim))  # shift&scale
        self.linear = nn.Linear(in_dim, patch_size * patch_size * out_chans)
        self.patch_size = patch_size
        self.out_chans = out_chans

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

# -----------------------------------------------------------------------------
# 5) MULTIMODAL DIT BLOCKS + TOKENIZER / TAB HEAD
# -----------------------------------------------------------------------------

class _AdaLNBlockBase(nn.Module):
    """Mixin per inizializzazione AdaLN (zero‑gate parziale + small‑shift)."""
    def _post_init_gating(self, dim: int):
        _zero_gate(self, "adaLN_modulation")
        _scale_shift_weights(self, "adaLN_modulation", dim)


class MultiModalDiTBlockImgToTab(_AdaLNBlockBase):
    """
    Aggiorna token immagine:
      1) Self‑Attn su token immagine
      2) Cross‑Attn (img ← tab, read‑only)
      3) MLP
    Ogni sotto‑strato: AdaLN (shift, scale, gate) dalla condizione t_emb + pooled tab.
    """
    def __init__(self, dim, head_dim, mlp_ratio, qkv_ratio, multiple_of, time_emb_dim, tab_pooled_dim,
                 layer_id, depth_init, num_layers, norm_eps=1e-6, use_bias=True):
        super().__init__()
        qkv_hidden_dim = ((head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
                          if qkv_ratio != 1.0 else dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1_x = create_norm("np_layernorm", dim, eps=norm_eps)
        self.attn_x = SelfAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                    hidden_dim=qkv_hidden_dim,
                                    init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                    else 0.02 / math.sqrt(2 * num_layers))
        self.norm2_x = create_norm("np_layernorm", dim, eps=norm_eps)
        self.cross_x = CrossAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                      hidden_dim=qkv_hidden_dim,
                                      init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                      else 0.02 / math.sqrt(2 * num_layers))
        self.norm3_x = create_norm("np_layernorm", dim, eps=norm_eps)
        self.mlp_x = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias,
                                 init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                 else 0.02 / math.sqrt(2 * num_layers))

        self.adaLN_modulation = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, 9 * dim, bias=True))
        self._post_init_gating(dim)
        self.cond_norm = create_norm("np_layernorm", dim, eps=norm_eps)

    @staticmethod
    def _mod(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _split_mod(self, cond):
        ssg = self.adaLN_modulation(cond).chunk(9, dim=1)
        shift = ssg[0], ssg[3], ssg[6]
        scale = ssg[1], ssg[4], ssg[7]
        gate = ssg[2], ssg[5], ssg[8]
        return shift, scale, gate

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor, t_emb: torch.Tensor,
                x_tab_pooled: torch.Tensor, xattn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.cond_norm(t_emb + x_tab_pooled)
        shift, scale, gate = self._split_mod(cond)
        gate = tuple(torch.sigmoid(g) for g in gate)

        if xattn_mask is not None:  # optional: disable cross-attention only
            mask = xattn_mask.view(-1, 1).to(gate[1].dtype)
            gate = (gate[0], gate[1] * mask, gate[2])

        x_ln = self._mod(self.norm1_x(x_img), shift[0], scale[0])
        x_img = x_img + gate[0].unsqueeze(1) * self.attn_x(x_ln)

        x_ln = self._mod(self.norm2_x(x_img), shift[1], scale[1])
        x_img = x_img + gate[1].unsqueeze(1) * self.cross_x(x_ln, x_tab)

        x_ln = self._mod(self.norm3_x(x_img), shift[2], scale[2])
        return x_img + gate[2].unsqueeze(1) * self.mlp_x(x_ln)


class MultiModalDiTBlockTabToImg(_AdaLNBlockBase):
    """Simmetrico del precedente: aggiorna token tab leggendo token immagine."""
    def __init__(self, dim, head_dim, mlp_ratio, qkv_ratio, multiple_of, time_emb_dim, img_pooled_dim,
                 layer_id, depth_init, num_layers, norm_eps=1e-6, use_bias=True):
        super().__init__()
        qkv_hidden_dim = ((head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
                          if qkv_ratio != 1.0 else dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1_t = create_norm("np_layernorm", dim, eps=norm_eps)
        self.attn_t  = SelfAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                     hidden_dim=qkv_hidden_dim,
                                     init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                     else 0.02 / math.sqrt(2 * num_layers))
        self.norm2_t = create_norm("np_layernorm", dim, eps=norm_eps)
        self.cross_t = CrossAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                      hidden_dim=qkv_hidden_dim,
                                      init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                      else 0.02 / math.sqrt(2 * num_layers))
        self.norm3_t = create_norm("np_layernorm", dim, eps=norm_eps)
        self.mlp_t   = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias,
                                   init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                   else 0.02 / math.sqrt(2 * num_layers))
        self.adaLN_modulation = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, 9 * dim, bias=True))
        self._post_init_gating(dim)
        self.cond_norm = create_norm("np_layernorm", dim, eps=norm_eps)

    @staticmethod
    def _mod(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _split_mod(self, cond):
        ssg = self.adaLN_modulation(cond).chunk(9, dim=1)
        shift = ssg[0], ssg[3], ssg[6]
        scale = ssg[1], ssg[4], ssg[7]
        gate  = ssg[2], ssg[5], ssg[8]
        return shift, scale, gate

    def forward(self, x_tab: torch.Tensor, x_img: torch.Tensor, t_emb: torch.Tensor,
                img_pooled: torch.Tensor, xattn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.cond_norm(t_emb + img_pooled)
        shift, scale, gate = self._split_mod(cond)
        gate = tuple(torch.sigmoid(g) for g in gate)

        if xattn_mask is not None:
            mask = xattn_mask.view(-1, 1).to(gate[1].dtype)
            gate = (gate[0], gate[1] * mask, gate[2])

        t_ln = self._mod(self.norm1_t(x_tab), shift[0], scale[0])
        x_tab = x_tab + gate[0].unsqueeze(1) * self.attn_t(t_ln)

        t_ln = self._mod(self.norm2_t(x_tab), shift[1], scale[1])
        x_tab = x_tab + gate[1].unsqueeze(1) * self.cross_t(t_ln, x_img)

        t_ln = self._mod(self.norm3_t(x_tab), shift[2], scale[2])
        return x_tab + gate[2].unsqueeze(1) * self.mlp_t(t_ln)


class GroupedTabTokenizer(nn.Module):
    """
    Costruisce un token per feature categ. (k_i→D) e un token per numerica (1→D).
    Ordine: [CLS, cat_0..cat_{m-1}, num_0..num_{n-1}]
    """
    def __init__(self, cat_dims: List[int], num_numeric: int, dim: int,
                 use_pos_embed: bool = True, norm_eps: float = 1e-6):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_numeric = num_numeric
        self.dim = dim
        self.use_pos_embed = use_pos_embed

        self.cat_proj = nn.ModuleList([nn.Linear(k, dim) for k in cat_dims])
        self.num_proj = nn.Linear(1, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + len(cat_dims) + num_numeric, dim))
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.cat_proj:
            nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0.)
        nn.init.xavier_uniform_(self.num_proj.weight); nn.init.constant_(self.num_proj.bias, 0.)

    def forward(self, x_tab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tab = x_tab.to(self.cls_token.dtype)
        B, _ = x_tab.shape
        offs = 0
        toks: List[torch.Tensor] = []

        # Categorical (slices)
        for k, proj in zip(self.cat_dims, self.cat_proj):
            toks.append(proj(x_tab[:, offs:offs + k]))
            offs += k
        # Numeric
        for j in range(self.num_numeric):
            v = x_tab[:, offs + j].unsqueeze(-1)  # (B,1)
            toks.append(self.num_proj(v))

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, *[t.unsqueeze(1) for t in toks]], dim=1)  # (B, 1+m+n, D)
        if self.use_pos_embed:
            x = x + self.pos_embed[:, :x.size(1)]
        return self.norm(x), x[:, 0]  # tokens, CLS


class FinalTabHeadGrouped(nn.Module):
    """Head finale tab raggruppata: logits per categ e 1 scalare per colonna numerica."""
    def __init__(self, in_dim: int, time_emb_dim: int, cat_dims: List[int], num_numeric: int,
                 act=nn.SiLU, eps=1e-6):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_numeric = num_numeric
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        self.ada = nn.Sequential(act(), nn.Linear(time_emb_dim, 2 * in_dim))

        self.cat_heads = nn.ModuleList([nn.Linear(in_dim, k) for k in cat_dims])
        self.num_heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_numeric)])
        for m in list(self.cat_heads) + list(self.num_heads):
            nn.init.trunc_normal_(m.weight, std=0.02 / math.sqrt(in_dim))
            nn.init.constant_(m.bias, 0.)

    def forward(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        cols = tokens[:, 1:, :]  # drop CLS
        cols = self.norm(cols)
        shift, scale = self.ada(t_emb).chunk(2, dim=1)
        cols = cols * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        outs, idx = [], 0
        for head in self.cat_heads:
            outs.append(head(cols[:, idx, :])); idx += 1
        for head in self.num_heads:
            outs.append(head(cols[:, idx, :])); idx += 1
        return torch.cat(outs, dim=1)

# -----------------------------------------------------------------------------
# 6) VIT PATCH MIXER (conditional mini-ViT with t_emb + pooled tab)
# -----------------------------------------------------------------------------

class ViTPatchMixerBlock(_AdaLNBlockBase):
    """
    Operates on [CLS + patch tokens] with embedding dimension `dim`:
      1) Self-attention
      2) Cross-attention (image ← tabular)
      3) MLP
    Each sublayer is modulated by `cond_emb` (e.g., t_emb + pooled tab).
    """
    def __init__(self, dim, head_dim, mlp_ratio, qkv_ratio, multiple_of, time_tab_dim,
                 layer_id, num_layers, norm_eps=1e-6, depth_init=False, use_bias=True):
        super().__init__()
        qkv_hidden_dim = ((head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
                          if qkv_ratio != 1.0 else dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = create_norm("np_layernorm", dim, eps=norm_eps)
        self.attn  = SelfAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                   hidden_dim=qkv_hidden_dim,
                                   init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                   else 0.02 / math.sqrt(2 * num_layers))
        self.norm2 = create_norm("np_layernorm", dim, eps=norm_eps)
        self.cross = CrossAttention(dim, qkv_hidden_dim // head_dim, qkv_bias=use_bias,
                                    hidden_dim=qkv_hidden_dim,
                                    init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                    else 0.02 / math.sqrt(2 * num_layers))
        self.norm3 = create_norm("np_layernorm", dim, eps=norm_eps)
        self.mlp   = FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias,
                                 init_std=0.02 / math.sqrt(2 * (layer_id + 1)) if depth_init
                                 else 0.02 / math.sqrt(2 * num_layers))

        self.adaLN_modulation = nn.Sequential(nn.GELU(), nn.Linear(dim, 9 * dim, bias=True))
        self._post_init_gating(dim)
        self.cond_norm = create_norm("np_layernorm", dim, eps=norm_eps)

    @staticmethod
    def _mod(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _split_mod(self, cond):
        ssg = self.adaLN_modulation(cond).chunk(9, dim=1)
        shift = ssg[0], ssg[3], ssg[6]
        scale = ssg[1], ssg[4], ssg[7]
        gate  = ssg[2], ssg[5], ssg[8]
        return shift, scale, gate

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        cond = self.cond_norm(cond_emb)
        shift, scale, gate = self._split_mod(cond)
        gate = tuple(torch.sigmoid(g) for g in gate)

        x_ln = self._mod(self.norm1(x_img), shift[0], scale[0])
        x_img = x_img + gate[0].unsqueeze(1) * self.attn(x_ln)

        x_ln = self._mod(self.norm2(x_img), shift[1], scale[1])
        x_img = x_img + gate[1].unsqueeze(1) * self.cross(x_ln, x_tab)

        x_ln = self._mod(self.norm3(x_img), shift[2], scale[2])
        return x_img + gate[2].unsqueeze(1) * self.mlp(x_ln)


class ViTPatchMixer(nn.Module):
    """
    Mini‑ViT patch mixer:
      - Inserisce [CLS] prima dei patch token
      - self‑attn + cross‑attn con token tab
      - gating da cond_emb (t_emb + tab_pooled)
      - restituisce (patch finali, img_cls)
    """
    def __init__(self, patch_mixer_dim: int, head_dim: int, mlp_ratio: float, qkv_ratio: float,
                 multiple_of: int, time_tab_dim: int, patch_mixer_depth: int,
                 norm_eps: float = 1e-6, depth_init: bool = False, use_bias: bool = True):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_mixer_dim))
        self.blocks = nn.ModuleList([
            ViTPatchMixerBlock(
                dim=patch_mixer_dim, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_ratio=qkv_ratio,
                multiple_of=multiple_of, time_tab_dim=time_tab_dim, layer_id=i,
                num_layers=patch_mixer_depth, norm_eps=norm_eps, depth_init=depth_init, use_bias=use_bias
            )
            for i in range(patch_mixer_depth)
        ])
        self.norm_final = nn.LayerNorm(patch_mixer_dim, eps=norm_eps)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img_tokens: torch.Tensor, x_tab: torch.Tensor, cond_emb: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = img_tokens.shape
        x = torch.cat([self.cls_token.expand(B, 1, D), img_tokens], dim=1)  # (B,1+T,D)
        for block in self.blocks:
            x = block(x, x_tab, cond_emb)
        x = self.norm_final(x)
        img_cls = x[:, 0, :]
        final_patches = x[:, 1:, :]
        return final_patches, img_cls

# -----------------------------------------------------------------------------
# 7) MULTI-MODAL DIT
# -----------------------------------------------------------------------------

class MultiModalDiT(nn.Module):
    """
    Multimodal DiT with:
      - Coarse grouping of tabular columns into tokens
      - Cross-attention in the main blocks (img↔tab)
      - Optional patch mixer
      - Optional MoE on selected blocks
      - Classifier-Free Guidance support (mix cond/uncond).
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

        # --- Tab settings ---
        num_numeric: int = 156,
        categorical_cardinalities: Optional[List[int]] = [2],
        num_classes: int = 2,  # {0,1} + NULL
        tab_groups=10,                 # (unused; retained for config compatibility)
        out_table_features=10,         # (unused; retained for config compatibility)

        # TabSyn latent mode
        use_tabsyn_vae: bool = True,
        tabsyn_d_token: int = 4,

        # Patch mixer
        use_patch_mixer=True,
        patch_mixer_depth=2,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=1.0,

        # MoE
        num_experts=8,
        expert_capacity=1.0,
        experts_every_n=2,
    ):
        super().__init__()

        # ---- label conditioning ------------------------------------------------
        self.NULL_ID = num_classes  # = 2 for {0,1}+NULL
        self.class_emb = nn.Embedding(num_classes + 1, dim)
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)

        # ---- Simple diagnostic head (on tab CLS) -----------------------------
        self.diag_head = nn.Linear(dim, 1)

        # ---- Tabular setup ---------------------------------------------------
        if categorical_cardinalities is None:
            categorical_cardinalities = []
        self.num_numeric = num_numeric
        self.cat_dims = categorical_cardinalities or []
        self.num_tab_columns = self.num_numeric + sum(self.cat_dims)

        self.tab_tokenizer = GroupedTabTokenizer(self.cat_dims, self.num_numeric, dim=dim, use_pos_embed=True,
                                                 norm_eps=norm_eps)
        self.use_tabsyn_vae = use_tabsyn_vae
        self.tabsyn_d_token = tabsyn_d_token
        self.n_tab_tokens = len(self.cat_dims) + self.num_numeric

        if self.use_tabsyn_vae:
            self.final_tab_latent = FinalTabLatentHead(
                in_dim=dim, time_emb_dim=dim, n_tokens=self.n_tab_tokens, d_token=self.tabsyn_d_token
            )
            self.latent_proj = nn.Linear(self.tabsyn_d_token, dim)
            nn.init.trunc_normal_(self.latent_proj.weight, std=0.02)
            nn.init.constant_(self.latent_proj.bias, 0.)
        else:
            self.final_tab = FinalTabHeadGrouped(in_dim=dim, time_emb_dim=dim,
                                                 cat_dims=self.cat_dims, num_numeric=self.num_numeric)

        # ---- Vision ----------------------------------------------------------
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim

        # Patch embedder
        self.x_embedder = PatchEmbed(img_size=input_size, patch_size=patch_size,
                                     in_chans=in_channels, embed_dim=dim)
        self.num_patches = self.x_embedder.num_patches
        self.base_size = input_size // patch_size

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size=dim, act_layer=nn.GELU)

        # Precomputed 2D sinusoidal position embedding (MAE-style).
        self.register_buffer("pos_embed", torch.zeros(1, self.num_patches, dim))

        # InfoNCE projections
        self.coh_img = nn.Linear(dim, 128, bias=False)
        self.coh_tab = nn.Linear(dim, 128, bias=False)

        # ---- Optional Patch Mixer --------------------------------------------
        self.use_patch_mixer = use_patch_mixer
        self.patch_mixer_dim = patch_mixer_dim
        if use_patch_mixer:
            self.vit_patch_mixer = ViTPatchMixer(
                patch_mixer_dim=patch_mixer_dim, head_dim=head_dim,
                mlp_ratio=patch_mixer_mlp_ratio, qkv_ratio=patch_mixer_qkv_ratio,
                multiple_of=multiple_of, time_tab_dim=dim, patch_mixer_depth=patch_mixer_depth,
                norm_eps=norm_eps, depth_init=False, use_bias=use_bias
            )
            if patch_mixer_dim != dim:
                self.patch_mixer_map_xin  = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_mixer_dim))
                self.patch_mixer_map_xout = nn.Sequential(nn.LayerNorm(patch_mixer_dim), nn.Linear(patch_mixer_dim, dim))
                self.cond_map_down = nn.Linear(dim, patch_mixer_dim, bias=False)
                self.tab_map_down  = nn.Linear(dim, patch_mixer_dim, bias=False)
            else:
                self.patch_mixer_map_xin  = nn.Identity()
                self.patch_mixer_map_xout = nn.Identity()
                self.cond_map_down = nn.Identity()
                self.tab_map_down  = nn.Identity()
        else:
            self.vit_patch_mixer = nn.Identity()
            self.patch_mixer_map_xin = self.patch_mixer_map_xout = nn.Identity()
            self.cond_map_down = self.tab_map_down = nn.Identity()

        # ---- Main interleaved blocks (img↔tab) --------------------------------
        total_depth = depth
        if len(ffn_multipliers) == total_depth:
            qkv_ratios = qkv_multipliers
            mlp_ratios = ffn_multipliers
        else:
            num_splits = len(ffn_multipliers)
            assert total_depth % num_splits == 0
            dps = total_depth // num_splits
            qkv_ratios = list(np.concatenate([[m] * dps for m in qkv_multipliers]))
            mlp_ratios = list(np.concatenate([[m] * dps for m in ffn_multipliers]))

        def _to_pos_int(x) -> int:
            try:
                xi = int(x)
            except Exception:
                return 0
            return max(0, xi)

        _ne = _to_pos_int(num_experts)
        _een = _to_pos_int(experts_every_n)
        enable_moe = (_ne > 0) and (_een > 0)
        is_moe_block = [False] * depth
        if enable_moe:
            # Do not mark the last block
            expert_blocks_idx = [i for i in range(depth - 1) if ((i + 1) % _een) == 0]
            for _i in expert_blocks_idx:
                is_moe_block[_i] = True

        self.blocks_imgtotab = nn.ModuleList()
        self.blocks_tabtoimg = nn.ModuleList()
        for i in range(depth):
            blk_img = MultiModalDiTBlockImgToTab(dim=dim, head_dim=head_dim,
                                                 mlp_ratio=mlp_ratios[i], qkv_ratio=qkv_ratios[i],
                                                 multiple_of=multiple_of, time_emb_dim=dim, tab_pooled_dim=dim,
                                                 norm_eps=norm_eps, depth_init=depth_init, layer_id=i,
                                                 num_layers=depth, use_bias=use_bias)
            blk_tab = MultiModalDiTBlockTabToImg(dim=dim, head_dim=head_dim,
                                                 mlp_ratio=mlp_ratios[i], qkv_ratio=qkv_ratios[i],
                                                 multiple_of=multiple_of, time_emb_dim=dim, img_pooled_dim=dim,
                                                 norm_eps=norm_eps, depth_init=depth_init, layer_id=i,
                                                 num_layers=depth, use_bias=use_bias)
            if is_moe_block[i]:
                hidden_dim = int(dim * mlp_ratios[i])
                blk_img.mlp_x = FeedForwardECMoe(_ne, expert_capacity, dim, hidden_dim, multiple_of)
                blk_tab.mlp_t = FeedForwardECMoe(_ne, expert_capacity, dim, hidden_dim, multiple_of)

            self.blocks_imgtotab.append(blk_img)
            self.blocks_tabtoimg.append(blk_tab)

        # ---- Final image head ------------------------------------------------
        self.final_img = FinalLayer(in_dim=dim, time_emb_dim=dim, patch_size=patch_size,
                                    out_chans=self.out_channels, act_layer=nn.GELU,
                                    norm_layer=create_norm('np_layernorm', dim, eps=norm_eps))
        # Mask token (for unmasking)
        self.register_buffer("mask_token", torch.zeros(1, 1, patch_size ** 2 * self.out_channels))

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Consistent initialization: set the 2D sinusoidal position embedding (MAE-style)."""
        side = int(self.num_patches ** 0.5)
        pe = torch.from_numpy(self.get_2d_sincos_pe(side, self.dim, base_size=self.base_size)).float()
        with torch.no_grad():
            self.pos_embed.copy_(pe.unsqueeze(0).to(self.pos_embed.device))

    # ---- Forward --------------------------------------------------------------
    def forward(self,
                x_img: torch.Tensor,     # (B, C, H, W)
                x_tab: torch.Tensor,     # (B, num_tab_columns) or flattened VAE latents
                t_img: torch.Tensor,     # (B,)
                t_tab: torch.Tensor,     # (B,)
                labels: Optional[torch.Tensor] = None,
                cfg: float = 1.0,
                mask_ratio: float = 0.0,
                self_cond_img: Optional[torch.Tensor] = None,
                self_cond_tab: Optional[torch.Tensor] = None,
                disable_xattn_img2tab: bool = False,
                disable_xattn_tab2img: bool = False,
                detach_xattn_img2tab: bool = False,
                detach_xattn_tab2img: bool = False,
                gradscale_xattn_img2tab: float = 1.0,
                gradscale_xattn_tab2img: float = 1.0):
        """
        If x_tab is None → unconditional generation (uses label=NULL).
        If cfg>1.0 → Classifier-Free Guidance (mix cond/uncond).
        """
        if labels is None:
            labels = torch.full_like(t_img, self.NULL_ID, dtype=torch.long)

        if cfg == 1.0:
            return self._forward_no_cfg(
                x_img, x_tab, t_img, t_tab, labels,
                mask_ratio=mask_ratio,
                self_cond_img=self_cond_img, self_cond_tab=self_cond_tab,
                disable_xattn_img2tab=disable_xattn_img2tab,
                disable_xattn_tab2img=disable_xattn_tab2img,
                detach_xattn_img2tab=detach_xattn_img2tab,
                detach_xattn_tab2img=detach_xattn_tab2img,
                gradscale_xattn_img2tab=gradscale_xattn_img2tab,
                gradscale_xattn_tab2img=gradscale_xattn_tab2img
            )
        else:
            return self._forward_with_cfg(x_img, x_tab, t_img, t_tab, labels, cfg=cfg, mask_ratio=mask_ratio)

    def _forward_no_cfg(self,
                        x_img: torch.Tensor, x_tab: Optional[torch.Tensor],
                        t_img: torch.Tensor, t_tab: torch.Tensor, labels: Optional[torch.Tensor],
                        mask_ratio: float = 0.0,
                        self_cond_img: Optional[torch.Tensor] = None,
                        self_cond_tab: Optional[torch.Tensor] = None,
                        disable_xattn_img2tab: bool = False,
                        disable_xattn_tab2img: bool = False,
                        detach_xattn_img2tab: bool = False,
                        detach_xattn_tab2img: bool = False,
                        gradscale_xattn_img2tab: float = 1.0,
                        gradscale_xattn_tab2img: float = 1.0) -> Dict[str, torch.Tensor]:
        # Scale gradients only (identity forward)
        def _grad_scale_only(x: torch.Tensor, s: float) -> torch.Tensor:
            if s >= 1.0:
                return x
            return x * s + x.detach() * (1.0 - s)

        # Masks to disable cross-attention (1=keep, 0=drop)
        m_img2tab = None if not disable_xattn_img2tab else torch.zeros(x_img.size(0), device=x_img.device)
        m_tab2img = None if not disable_xattn_tab2img else torch.zeros(x_img.size(0), device=x_img.device)

        # Timestep + label embedding
        label_emb = self.class_emb(labels)
        t_emb_img = self.t_embedder(t_img) + label_emb
        t_emb_tab = self.t_embedder(t_tab) + label_emb

        # Optional self-conditioning
        if self_cond_img is not None:
            x_img = x_img + self_cond_img
        if self_cond_tab is not None:
            x_tab = x_tab + self_cond_tab

        # Patchify + position embedding
        img_tokens = self.x_embedder(x_img) + self.pos_embed

        # Tab tokens (two modes)
        if not self.use_tabsyn_vae:
            tab_tokens, tab_cls = self.tab_tokenizer(x_tab)
        else:
            B = x_tab.size(0)
            z = x_tab.view(B, self.n_tab_tokens, self.tabsyn_d_token)
            cols = self.latent_proj(z)
            cls = self.tab_tokenizer.cls_token.expand(B, 1, -1)
            xcat = torch.cat([cls, cols], dim=1)
            if getattr(self.tab_tokenizer, "use_pos_embed", True):
                xcat = xcat + self.tab_tokenizer.pos_embed[:, :xcat.size(1)]
            tab_tokens = self.tab_tokenizer.norm(xcat)
            tab_cls = tab_tokens[:, 0]

        tab_kv = tab_tokens[:, 1:, :]
        tab_pooled = tab_cls

        s_i2t = 0.0 if detach_xattn_img2tab else float(gradscale_xattn_img2tab)
        s_t2i = 0.0 if detach_xattn_tab2img else float(gradscale_xattn_tab2img)
        tab_kv_ctx = _grad_scale_only(tab_kv, s_i2t)

        # Patch mixer (optional)
        if isinstance(self.vit_patch_mixer, ViTPatchMixer):
            img_small = self.patch_mixer_map_xin(img_tokens)
            tab_small = self.tab_map_down(tab_kv_ctx)
            cond_small = self.cond_map_down(t_emb_img + tab_cls)
            cond_small = _grad_scale_only(cond_small, s_i2t)
            img_small, img_cls_small = self.vit_patch_mixer(img_small, tab_small, cond_small)
            img_tokens = self.patch_mixer_map_xout(img_small)

        # Masking (MAE‑like)
        mask, ids_restore = None, None
        if mask_ratio > 0:
            B, T_img, D = img_tokens.shape
            info = get_mask(B, T_img, mask_ratio, x_img.device)
            img_tokens = mask_out_token(img_tokens, info['ids_keep'])
            mask, ids_restore = info['mask'], info['ids_restore']

        # Interleaved blocks (img↔tab)
        for i, (blk_img, blk_tab) in enumerate(zip(self.blocks_imgtotab, self.blocks_tabtoimg)):
            tab_kv = tab_tokens[:, 1:, :]
            tab_pooled = tab_tokens[:, 0, :]
            img_tokens = blk_img(img_tokens, _grad_scale_only(tab_kv, s_i2t), t_emb_img,
                                 _grad_scale_only(tab_pooled, s_i2t), xattn_mask=m_img2tab)
            img_pooled = img_tokens.mean(1)
            tab_tokens = blk_tab(tab_tokens, _grad_scale_only(img_tokens, s_t2i),
                                 t_emb_tab, _grad_scale_only(img_pooled, s_t2i), xattn_mask=m_tab2img)

        # Final heads
        img_logits = self.final_img(img_tokens, t_emb_img)  # (B,T,p^2*C)
        if mask_ratio > 0 and ids_restore is not None:
            img_logits = unmask_tokens(img_logits, ids_restore, self.mask_token)
        img_sample = self.unpatchify(img_logits)  # (B,C,H,W)

        if not self.use_tabsyn_vae:
            tab_sample = self.final_tab(tab_tokens, t_emb_tab)  # (B, sum k_i + n)
        else:
            lat_tokens = self.final_tab_latent(tab_tokens, t_emb_tab)  # (B,T,d_token)
            tab_sample = lat_tokens.reshape(lat_tokens.size(0), -1)

        # Diagnostic head on CLS (detached by default to avoid disrupting the generative trunk)
        if not hasattr(self, "detach_diag_from_trunk"):
            self.detach_diag_from_trunk = True
        diag_feat = tab_tokens[:, 0, :].detach() if self.detach_diag_from_trunk else tab_tokens[:, 0, :]
        diag_logits = self.diag_head(diag_feat)  # (B,1)

        # Coherence (InfoNCE): projection + L2 normalization
        img_pooled_final = img_tokens.mean(1)
        tab_pooled_final = tab_tokens[:, 0, :]
        img_emb = F.normalize(self.coh_img(img_pooled_final), dim=-1)
        tab_emb = F.normalize(self.coh_tab(tab_pooled_final), dim=-1)

        return {"image_sample": img_sample, "tab_sample": tab_sample,
                "diag_logits": diag_logits.squeeze(-1), "mask": mask,
                "img_emb": img_emb, "tab_emb": tab_emb}

    def _forward_with_cfg(self, x_img: torch.Tensor, x_tab: Optional[torch.Tensor],
                          t_img: torch.Tensor, t_tab: torch.Tensor, labels: torch.Tensor,
                          cfg: float, mask_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Classifier-Free Guidance (mix cond/uncond) as in Ho & Salimans (2022).
        Important: the 'denoised' state (x_img, x_tab) is identical in both branches;
        only the conditioning is dropped (labels→NULL).
        """
        B = x_img.shape[0]

        x_img_cat = torch.cat([x_img, x_img], dim=0)
        t_img_cat = torch.cat([t_img, t_img], dim=0)
        t_tab_cat = torch.cat([t_tab, t_tab], dim=0)
        labels_cat = torch.cat([labels, labels.new_full(labels.shape, self.NULL_ID)], 0) if labels.shape[0] == B else labels
        tab_cat = torch.cat([x_tab, x_tab], dim=0) if x_tab is not None else None

        out_cat = self._forward_no_cfg(x_img_cat, tab_cat, t_img_cat, t_tab_cat, labels_cat, mask_ratio=mask_ratio)
        cond_img, uncond_img = out_cat["image_sample"].split(B, dim=0)
        cond_tab, uncond_tab = out_cat["tab_sample"].split(B, dim=0)
        cond_d, uncond_d = out_cat["diag_logits"].split(B, dim=0)

        guided_img = uncond_img + cfg * (cond_img - uncond_img)
        guided_tab = uncond_tab + cfg * (cond_tab - uncond_tab)
        guided_d = uncond_d + cfg * (cond_d - uncond_d)
        return {"image_sample": guided_img, "tab_sample": guided_tab, "diag_logits": guided_d, "mask": None}

    # ---- utilities -----------------------------------------------------------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, patch_dim = x.shape
        p = self.patch_size
        c = self.out_channels
        h = w = int(T ** 0.5)
        x = x.reshape(B, h, w, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h * p, w * p)

    @staticmethod
    def get_2d_sincos_pe(grid_size, embed_dim, base_size=16, pos_interp_scale=1.0):
        """
        Generate 2D sinusoidal embeddings (as used in MAE/ViT).
        """
        def get_1d_sin_cos(pos, emb_dim):
            half_dim = emb_dim // 2
            omega = 1. / (10000 ** (np.arange(half_dim) / half_dim))
            out = np.einsum('m,d->md', pos, omega)
            emb_sin = np.sin(out)
            emb_cos = np.cos(out)
            return np.concatenate([emb_sin, emb_cos], axis=1)

        h = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size) / pos_interp_scale
        w = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size) / pos_interp_scale
        ww, hh = np.meshgrid(w, h)
        ww = ww.reshape(-1); hh = hh.reshape(-1)
        assert embed_dim % 2 == 0
        emb_h = get_1d_sin_cos(hh, embed_dim // 2)
        emb_w = get_1d_sin_cos(ww, embed_dim // 2)
        return np.concatenate([emb_h, emb_w], axis=1)


# -----------------------------------------------------------------------------
# Config loader + usage example
# -----------------------------------------------------------------------------
from utils.configurations import _merge_cfg

def load_dit(cfg: DictConfig, **overrides: Any) -> nn.Module:
    """Instantiate MultiModalDiT from a DictConfig (OmegaConf)."""
    final_cfg = _merge_cfg(cfg, overrides)
    print("[INFO] Loading MultiModalDiT model with config:", final_cfg)
    model = MultiModalDiT(**final_cfg)
    print("[INFO] Loaded DiT")
    return model


# -----------------------------------------------------------------------------
# Standalone execution example
# -----------------------------------------------------------------------------
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
        num_numeric=156,
        categorical_cardinalities=[2]
    )

    # Dummy data
    N = 2
    x_img = torch.randn(N, 4, 32, 32)   # batch=2, 4 channels
    tab = torch.randn(N, 158)
    t_img = torch.randint(0, 1000, (N,))
    t_tab = torch.randint(0, 1000, (N,))

    res = model(x_img=x_img, x_tab=tab, t_img=t_img, t_tab=t_tab, mask_ratio=0.2, cfg=1.0)
    print("img_out shape:", res["image_sample"].shape)  # (N, 4, 32, 32)
    print("tab_out shape:", res["tab_sample"].shape)    # (N, 174) with default VAE latent mode