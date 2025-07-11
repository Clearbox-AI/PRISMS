from omegaconf import DictConfig
from omegaconf import OmegaConf
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, List, Any, Dict, Union
from easydict import EasyDict
from pathlib import Path
import json
import sklearn.preprocessing


# -----------------------------------------------------------------------------
# Column‑wise reversible transforms
# -----------------------------------------------------------------------------
class ColumnTransform(nn.Module):
    """Base class – subclasses must implement forward / inverse."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
    def inverse(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class StandardiseMixin:
    """Adds μ/σ buffers so every transformed column is ~N(0,1)."""
    def fit_stats(self, col: torch.Tensor):
        mu, sig = col.mean(), col.std().clamp_min(1e-9)
        self.register_buffer("mu", mu)
        self.register_buffer("sig", sig)

    # helper
    def _fwd_std(self, x):
        return (x - self.mu) / self.sig
    def _inv_std(self, z):
        return z * self.sig + self.mu

class IdentityTransform(StandardiseMixin, ColumnTransform):
    def __init__(self, apply_std: bool = True):
        super().__init__()
        self.apply_std = apply_std
    def forward(self, x):
        return self._fwd_std(x) if self.apply_std else x
    def inverse(self, z):
        return self._inv_std(z) if self.apply_std else z

# --- bounded scalar ------------------------------------------------------
_MARGIN = 5e-3  # 0.5 %

def _widen(lo: float, hi: float) -> Tuple[float, float]:
    span = hi - lo
    return lo - _MARGIN*span, hi + _MARGIN*span

class BoundedScalar(StandardiseMixin, ColumnTransform):
    """Map *lo≤x≤hi* → ℝ via ε‑logit, then standardise.
    Replaces the old tanh mapping which saturated.
    """
    def __init__(self, lo: float, hi: float, eps: float = 1e-3, apply_std: bool = True):
        super().__init__()
        assert hi > lo, "hi must be larger than lo"
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))
        self.eps = eps
        self.apply_std = apply_std

    def _to_unit(self, x: torch.Tensor):
        y = (x - self.lo) / (self.hi - self.lo)
        return y.clamp(self.eps, 1.0 - self.eps)

    def forward(self, x):
        z = torch.logit(self._to_unit(x))
        return self._fwd_std(z) if self.apply_std else z

    def inverse(self, z):
        z = self._inv_std(z) if self.apply_std else z
        y = torch.sigmoid(z)
        return (self.lo + y * (self.hi - self.lo)).clamp(self.lo, self.hi)

class NonNegativeQN(StandardiseMixin, ColumnTransform):
    """Non‑negative column with heavy tail – quantile‑normalise."""
    def __init__(self, n_quant: int = 1000):
        super().__init__()
        self.qtf = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal", n_quantiles=n_quant, subsample=int(1e9)
        )
    def forward(self, x):
        if not hasattr(self, "_fitted"):
            self.qtf.fit(x.cpu().numpy())
            self._fitted = True
            self.fit_stats(torch.from_numpy(self.qtf.transform(x.cpu().numpy())))
        z_np = self.qtf.transform(x.cpu().numpy())
        z = torch.from_numpy(z_np).to(x.device, dtype=x.dtype)
        return self._fwd_std(z)
    def inverse(self, z):
        z = self._inv_std(z)
        x_np = self.qtf.inverse_transform(z.cpu().numpy())
        return torch.from_numpy(x_np).to(z.device, dtype=z.dtype)

class NonPositive(StandardiseMixin, ColumnTransform):
    def __init__(self, apply_std: bool = True):
        super().__init__(); self.apply_std = apply_std
    def forward(self, x):
        z = torch.log1p(-x)
        return self._fwd_std(z) if self.apply_std else z
    def inverse(self, z):
        z = self._inv_std(z) if self.apply_std else z
        return -torch.expm1(z).clamp_min_(0.)

class CategoricalOrdinal(ColumnTransform):
    def __init__(self, categories: List[Any]):
        super().__init__()
        self.categories = categories
        self.lookup = {v: i for i, v in enumerate(categories)}
        self.mask_val = len(categories)      # last index is MASK/UNK
    def forward(self, x):
        idx = torch.tensor(
            [self.lookup.get(int(v.item()), self.mask_val) for v in x],
            device=x.device, dtype=x.dtype
        )
        return idx[:, None]
    def inverse(self, z):
        zint = z.round().clamp_(0, self.mask_val).long().squeeze(1)
        vals = [self.categories[i] if i < self.mask_val else self.categories[0]
                for i in zint]
        return torch.tensor(vals, device=z.device, dtype=torch.float32)[:, None]

# mapping helper --------------------------------------------------------------
_SIGN2TF = {
    "non-negative": NonNegativeQN,
    "non-positive": NonPositive,
    "mixed": IdentityTransform,
}

def build_transforms_from_schema(path: Union[str, Path]) -> List[ColumnTransform]:
    with open(path) as f: schema: List[Dict[str, Any]] = json.load(f)
    tf: List[ColumnTransform] = []
    for col in schema:
        dtype = col.get("dtype", "continuous").lower()

        if dtype == "categorical":           # ---------------- categorical
            cats = col.get("categories")
            if not cats:
                lo, hi = int(col["min"]), int(col["max"])
                cats = list(range(lo, hi + 1))
            tf.append(CategoricalOrdinal(cats))
            continue

        lo, hi = col.get("min"), col.get("max")
        if lo is not None and hi is not None:        # bounded scalar
            lo, hi = _widen(lo, hi)
            tf.append(BoundedScalar(lo, hi))
        else:                                        # signed / non-pos / etc.
            sign = col.get("sign", "mixed").lower()
            tf.append(_SIGN2TF.get(sign, IdentityTransform)())
    return tf


# helper
def _make_t(batch, sigma_scalar):
    """return (B,) float32 tensor with log(σ)/4 repeated B times"""
    return (sigma_scalar.log() / 4).float().repeat(batch)


class MultiModalDiffusion(nn.Module):
    def __init__(
        self,
        dit: nn.Module,
        num_tab_features: int,
            tab_transforms: List[ColumnTransform],
        *,
        sigma_min: float, sigma_max: float, num_steps: int,
        rho: int, P_mean: float, P_std: float,
        S_churn: float, S_min: float, S_max: float, S_noise: float,
        dtype: str = "bfloat16",
    ):
        super().__init__()
        assert len(tab_transforms) == num_tab_features, "Mismatch transforms ↔ features"

        self.dit = dit
        self.num_tab_features = num_tab_features
        self.dtype = dtype
        self.tab_tf = nn.ModuleList(tab_transforms)

        # remember which columns are categorical for later masks
        self.cat_indices: List[int] = [i for i, tf in enumerate(tab_transforms) if isinstance(tf, CategoricalOrdinal)]
        self.cont_indices: List[int] = [i for i in range(num_tab_features) if i not in self.cat_indices]
        self.register_buffer("cat_mask_tensor", torch.tensor(self.cat_indices, dtype=torch.long), persistent=False)

        self.edm_img = EasyDict(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_steps=num_steps,
            rho=rho,
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=0.9,
            S_churn=S_churn,
            S_min=S_min,
            S_max=S_max,
            S_noise=S_noise,
        )

        self.edm_tab = EasyDict(
            sigma_min=0.002,
            sigma_max=30,
            num_steps=40,
            rho=rho, P_mean=P_mean, P_std=0.5,
            sigma_data=0.9,
            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
        )

        self.randn_like = torch.randn_like

    # ------------------------------------------------------------------
    # Encode / decode helpers
    # ------------------------------------------------------------------
    def encode_tab(self, x: torch.Tensor) -> torch.Tensor:
        cols = [tf(x[:, i: i + 1]) for i, tf in enumerate(self.tab_tf)]
        return torch.cat(cols, dim=1)

    def decode_tab(self, z):
        out, cur = [], 0
        for tf in self.tab_tf:
            out.append(tf.inverse(z[:, cur:cur+1])); cur += 1
        # return self._validate(torch.cat(out, 1))
        return torch.cat(out, 1)

    # ---------------------------------------------------------------------
    # Forward – training
    # ---------------------------------------------------------------------
    def forward(
        self,
        x_img: torch.Tensor,         # latent image  (B,4,32,32)  already scaled
        x_tab: torch.Tensor,         # standardised  (B,F)
        mask_ratio: float = 0.,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss, loss_img, loss_tab
        """

        # encode numeric columns to R
        x_tab = self.encode_tab(x_tab)

        device = x_img.device
        B = x_img.size(0)

        # TODO separare le due sigma in base al branch -> tutto il resto, anche sampling obv
        # 1) sample a *shared* σ  ~ logN(P_mean,P_std)
        sigma = ((torch.randn(B, 1, device=device) * self.edm_tab.P_std + self.edm_tab.P_mean).exp())  # (B,1)
        σ_img = sigma.view(B, 1, 1, 1)
        σ_tab = sigma.view(B, 1)

        # 2a) continuous columns: add Gaussian noise
        x_tab_noisy = x_tab.clone()

        if self.cont_indices:
            # cont = torch.tensor(self.cont_indices, device=device)
            # ε_cont = torch.randn(B, len(cont), device=device) * σ_tab
            # x_tab_noisy[:, cont] = x_tab[:, cont] + ε_cont

            cont = torch.tensor(self.cont_indices, device=device)
            x_tab_noisy[:, cont] += torch.randn(B, len(cont), device=device) * σ_tab

        # 2b)  categorical  (TabDiff-style masking)
        if self.cat_indices:
            move_p = (1. - torch.exp(-σ_tab)).squeeze(1)  # (B,)
            rnd = torch.rand(B, len(self.cat_indices), device=device)
            for j, idx in enumerate(self.cat_indices):
                mask_val = float(self.tab_tf[idx].mask_val)
                x_tab_noisy[:, idx] = torch.where(
                    rnd[:, j] < move_p,  # both (B,)
                    torch.full_like(x_tab_noisy[:, idx], mask_val),
                    x_tab_noisy[:, idx]
                )

        # 2c) image noise
        x_img_noisy = x_img + torch.randn_like(x_img) * σ_img


        # 2) inject Gaussian noise (ε drawn *independently*)
        # ε_img = torch.randn_like(x_img) * σ_img
        # x_img_noisy = x_img + ε_img
        # ε_tab = torch.randn_like(x_tab) * σ_tab
        # x_tab_noisy = x_tab + ε_tab

        # 3) prepare EDM conditioning factors
        c_in_img = 1. / (self.edm_tab.sigma_data ** 2 + σ_img ** 2).sqrt()
        c_in_tab = 1. / (self.edm_tab.sigma_data ** 2 + σ_tab ** 2).sqrt()
        c_noise = σ_tab.log() / 4.  # (B,1)   identical for img/tab

        # 4) DiT forward ----------------------------------------------------
        out = self.dit(
            x_img=c_in_img * x_img_noisy,
            x_tab=c_in_tab * x_tab_noisy,
            t=c_noise.squeeze(-1),  # keep DiT sig‑shape agnostic
            mask_ratio=mask_ratio,
        )
        F_img = out["image_sample"]
        F_tab = out["tab_sample"]

        # 5) Convert ε‑prediction → denoised estimate D(x)
        D_img = self._to_D(x_img_noisy, σ_img, out["image_sample"], branch="img")
        D_tab = self._to_D(x_tab_noisy, σ_tab, out["tab_sample"], branch="tab")

        # 6) Weighted EDM MSE – modality‑balanced
        w = ((σ_tab ** 2 + self.edm_tab.sigma_data ** 2) / (σ_tab * self.edm_tab.sigma_data) ** 2)  # (B,1)
        loss_img = (w.view(B, 1, 1, 1) * (D_img - x_img).square()).mean()
        loss_tab = (w * (D_tab - x_tab).square()).mean() / self.num_tab_features
        total_loss = loss_img + loss_tab

        return total_loss, loss_img, loss_tab

    # ---------------------------------------------------------------------
    # Sampling EDM schedule, but categorical columns receive no Gaussian noise and are initialised with MASK code.
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        guidance_scale: float = 1.0,
        num_steps: int = 40,
        seed: Optional[int] = None,
        return_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one pair per batch element.
        If `return_latents=False` the tabular output is de‑standardised and
        the image latent is *not* decoded – leave decoding to the caller.
        """
        device = next(self.dit.parameters()).device
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        # Initial x at largest σ
        x_img = torch.randn((batch_size, 4, 32, 32), generator=g, device=device)
        x_tab = torch.zeros((batch_size, self.num_tab_features), device=device)
        # continuous part starts StdNormal * σ_max
        if self.cont_indices:
            cont = torch.tensor(self.cont_indices, device=device)
            x_tab[:, cont] = torch.randn((batch_size, len(cont)), generator=g, device=device)
        # categorical part initialised to MASK
        for idx in self.cat_indices:
            x_tab[:, idx] = self.tab_tf[idx].mask_val

        cfg_fwd = partial(self.dit.forward, cfg=guidance_scale) \
            if guidance_scale > 1. else self.dit.forward

        # pre‑compute σ schedule
        N = num_steps
        step = torch.arange(N, device=device, dtype=torch.float64)
        σ = (self.edm_tab.sigma_max ** (1 / self.edm_tab.rho) +
             step / (N - 1) * (self.edm_tab.sigma_min ** (1 / self.edm_tab.rho) -
                               self.edm_tab.sigma_max ** (1 / self.edm_tab.rho))) ** self.edm_tab.rho
        σ = torch.cat([σ, σ.new_zeros(1)])  # append 0 for last update

        # main loop ---------------------------------------------------------
        x_img = x_img.double() * σ[0]
        x_tab = x_tab.double() * σ[0]

        for i, (σ_cur, σ_next) in enumerate(zip(σ[:-1], σ[1:])):
            # σ‑churn (continuous part only)
            γ = min(self.edm_tab.S_churn / N, np.sqrt(2) - 1) if self.edm_tab.S_min <= σ_cur <= self.edm_tab.S_max else 0.0
            σ_hat = σ_cur + γ * σ_cur
            g_noise = self.edm_tab.S_noise

            x_img_hat = x_img + (σ_hat ** 2 - σ_cur ** 2).sqrt() * g_noise * torch.randn_like(x_img)
            x_tab_hat = x_tab.clone()
            if self.cont_indices:
                cont = torch.tensor(self.cont_indices, device=device)
                x_tab_hat[:, cont] += (σ_hat ** 2 - σ_cur ** 2).sqrt() * g_noise * torch.randn_like(x_tab_hat[:, cont])

            # predict ε (or v) and map to D(x)
            t_hat_vec = _make_t(batch_size, σ_hat)
            out = cfg_fwd(
                x_img=x_img_hat.float(),
                x_tab=x_tab_hat.float(),
                t=t_hat_vec,
                mask_ratio=0.,
            )
            D_img = self._to_D(x_img_hat, σ_hat, out["image_sample"], branch="img")
            D_tab = self._to_D(x_tab_hat, σ_hat, out["tab_sample"], branch="tab")

            # Euler step
            d_img = (x_img_hat - D_img) / σ_hat
            d_tab = (x_tab_hat - D_tab) / σ_hat
            x_img_next = x_img_hat + (σ_next - σ_hat) * d_img
            x_tab_next = x_tab_hat + (σ_next - σ_hat) * d_tab

            # 2nd‑order corrector (disabled at final step)
            if i < N - 1:
                t_next_vec = _make_t(batch_size, σ_next)
                out = cfg_fwd(
                    x_img=x_img_next.float(),
                    x_tab=x_tab_next.float(),
                    t=t_next_vec,
                    mask_ratio=0.,
                )
                D_img_prime = self._to_D(x_img_next, σ_next, out["image_sample"], branch="img")
                D_tab_prime = self._to_D(x_tab_next, σ_next, out["tab_sample"], branch="tab")
                d_img_prime = (x_img_next - D_img_prime) / σ_next
                d_tab_prime = (x_tab_next - D_tab_prime) / σ_next
                x_img_next = x_img_hat + (σ_next - σ_hat) * 0.5 * (d_img + d_img_prime)
                x_tab_next = x_tab_hat + (σ_next - σ_hat) * 0.5 * (d_tab + d_tab_prime)

            x_img, x_tab = x_img_next, x_tab_next

        x_img = x_img.float()
        x_tab = x_tab.float()

        if return_latents:
            return x_img, x_tab
        return x_img, self.decode_tab(x_tab)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _to_D(self, x_noisy, σ, F_x, *, branch: str):
        # edm = self.edm_img if branch == 'img' else self.edm_tab
        edm = self.edm_tab
        c_skip = edm.sigma_data ** 2 / (σ ** 2 + edm.sigma_data ** 2)
        c_out = σ * edm.sigma_data / torch.sqrt(σ ** 2 + edm.sigma_data ** 2)
        return c_skip * x_noisy + c_out * F_x

    def _validate(self, x):
        for i, tf in enumerate(self.tab_tf):
            if isinstance(tf, NonNegativeQN):
                x[:, i].clamp_min_(0.)
            elif isinstance(tf, NonPositive):
                x[:, i].clamp_max_(0.)
            elif isinstance(tf, CategoricalOrdinal):
                x[:, i].round_().clamp_(0, tf.mask_val)
        return x



# def load_diffusion(cfg: DictConfig, dit_model: nn.Module, tmp_param: Any = None, **overrides: Any) -> nn.Module:
#     """
#     Load a MultiModalDiffusion model from config, injecting a pre-loaded DiT.
#     """
#     from utils.configurations import apply_overrides
#     cfg = apply_overrides(cfg, overrides)
#     print("[INFO] Loading Diffusion model with config:", cfg)
#
#     if "diffusion" in cfg:
#         diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg.diffusion)
#     else:
#         diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg)
#
#     print("[INFO] Loaded Diffusion Model")
#     return diffusion_model

from utils.configurations import _merge_cfg
def load_diffusion(cfg: DictConfig, dit_model, tab_transforms, **overrides):
    """
    Instantiate ``MultiModalDiffusion`` with an already-built *dit_model*.

    Extra keyword args override (or add) fields in *cfg* exactly like the other
    loaders.
    """
    final_cfg = _merge_cfg(cfg, overrides)
    diffusion_model = MultiModalDiffusion(dit=dit_model, tab_transforms=tab_transforms, **final_cfg)
    return diffusion_model



if __name__ == "__main__":
    # Suppose we have:
    from torch import optim, Tensor

    # 1) A "MultiModalDiT" instance that expects:
    #    model(x_img, x_tab, time_scalar) -> {"img_out":..., "tab_out":...}
    from models.dit.dit_multimodal_add16 import MultiModalDiT
    import numpy as np

    B = 4
    H, W = 32, 32
    in_chans = 4
    d_tab = 157

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

    # 2) Our EDM diffuser
    diffuser = MultiModalDiffusion(dit=model)

    # 3) Some example training batch
    x_img = torch.randn(B, in_chans, H, W)
    x_tab = torch.randn(B, d_tab)

    # 4) forward => get loss
    loss_total, loss_img, loss_tab = diffuser(x_img, x_tab)
    print(f"total loss={loss_total}, img loss={loss_img}, tab loss={loss_tab}")