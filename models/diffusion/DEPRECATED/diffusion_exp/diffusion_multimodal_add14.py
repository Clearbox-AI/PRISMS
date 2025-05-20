from typing import Any
from omegaconf import DictConfig
import torch
import torch.nn as nn


################################################################################
#                                 EMA helper                                   #
################################################################################

class EMA:
    """Exponential moving average of model parameters for more stable sampling."""

    def __init__(
        self,
        dit: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10,
    ) -> None:
        self.dit = dit
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every

        # clone parameters
        self.shadow_params = [p.clone().detach() for p in dit.parameters() if p.requires_grad]
        self.collected_params: Optional[list[torch.Tensor]] = None
        self.num_updates = 0

    @torch.no_grad()
    def update(self, dit: nn.Module) -> None:
        if self.num_updates < self.update_after_step:
            self.num_updates += 1
            return

        if (self.num_updates - self.update_after_step) % self.update_every != 0:
            self.num_updates += 1
            return

        for s, p in zip(self.shadow_params, dit.parameters(), strict=True):
            if not p.requires_grad:
                continue
            s.data.lerp_(p.data, 1.0 - self.decay)
        self.num_updates += 1

    def as_model(self) -> nn.Module:
        """
        Returns a detached, eval‑mode clone that carries the EMA parameters.
        The original network (self.dit) is never modified.
        """
        import copy
        ema_clone = copy.deepcopy(self.dit)  # ← keeps architecture only
        for s, p in zip(self.shadow_params, ema_clone.parameters(), strict=True):
            if p.requires_grad:
                p.data.copy_(s.data)
        ema_clone.eval()
        for p in ema_clone.parameters():
            p.requires_grad_(False)
        return ema_clone

    def copy_to(self, dit: nn.Module) -> None:
        """Load EMA parameters into `model` (in‑place)."""
        for s, p in zip(self.shadow_params, dit.parameters(), strict=True):
            if not p.requires_grad:
                continue
            p.data.copy_(s.data)

    def ema_model_inference(self) -> nn.Module:
        """
        Returns an *evaluation‑only* clone of ``self.dit`` that carries the
        shadow (EMA) parameters.  The original model is left untouched, so
        training can continue straight after sampling.
        """

        import copy
        ema_dit = copy.deepcopy(self.dit)
        for s, p in zip(self.shadow_params, ema_dit.parameters(), strict=True):
            if not p.requires_grad:
                continue
            p.data.copy_(s.data)
        ema_dit.eval()
        for p in ema_dit.parameters():
            p.requires_grad_(False)
        return ema_dit

################################################################################
#                           Multi‑modal Diffusion                              #
################################################################################

# multimodal_diffusion.py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple
from easydict import EasyDict

# helper
def _make_t(batch, sigma_scalar):
    """return (B,) float32 tensor with log(σ)/4 repeated B times"""
    return (sigma_scalar.log() / 4).float().repeat(batch)

# -------------------------------------------------------------------------
# EDM hyper‑parameter container
# -------------------------------------------------------------------------
def edm_defaults(p_mean: float = -0.6, p_std: float = 1.2) -> EasyDict:
    return EasyDict(
        sigma_min = 2e-3,
        sigma_max = 80.,
        P_mean    = p_mean,
        P_std     = p_std,
        sigma_data= 0.9,
        num_steps = 18,
        rho       = 7,
        S_churn   = 0.,
        S_min     = 0.,
        S_max     = float("inf"),
        S_noise   = 1.,
    )


# -------------------------------------------------------------------------
# Main model
# -------------------------------------------------------------------------
class MultiModalDiffusion(nn.Module):
    """
    DiT‑based EDM that *jointly* diffuses an image latent (B,4,32,32)
    and a tabular vector (B,F).

    The wrapped `dit_model` **must** expose
        forward(x_img, x_tab, c_noise, mask_ratio=0., cfg=1.0)
    and return a dict with keys 'image_sample', 'tab_sample', plus
    optional 'mask' for MAE‑style training.
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(
        self,
        dit: nn.Module,
        num_tab_features: int = 157,
        *,
        edm_cfg: Optional[dict] = None,
        dtype: str = "bfloat16",
    ):
        """
        Parameters
        ----------
        dit        : multimodal DiT ‑‑ the *only* trainable component.
        num_tab_features : dimensionality of tabular vector.
        edm_cfg          : override EDM defaults (optional).
        dtype            : compute dtype for DiT ('bfloat16' | 'fp16' | 'fp32').
        """
        super().__init__()
        self.dit = dit
        self.num_tab_features = num_tab_features
        self.dtype = dtype
        self.edm = edm_defaults() if edm_cfg is None else EasyDict(edm_cfg)

        # convenience helpers
        self.randn_like = torch.randn_like

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
        device = x_img.device
        B = x_img.size(0)

        # 1) sample a *shared* σ  ~ log 𝒩(P_mean,P_std)
        sigma = ((torch.randn(B, 1, device=device) * self.edm.P_std + self.edm.P_mean)
                 .exp())                          # (B,1)
        σ_img = sigma.view(B, 1, 1, 1)
        σ_tab = sigma.view(B, 1)

        # 2) inject Gaussian noise (ε drawn *independently*)
        ε_img = torch.randn_like(x_img) * σ_img
        ε_tab = torch.randn_like(x_tab) * σ_tab
        x_img_noisy = x_img + ε_img
        x_tab_noisy = x_tab + ε_tab

        # 3) prepare EDM conditioning factors
        c_in_img  = 1. / (self.edm.sigma_data ** 2 + σ_img ** 2).sqrt()
        c_in_tab  = 1. / (self.edm.sigma_data ** 2 + σ_tab ** 2).sqrt()
        c_noise   = σ_tab.log() / 4.               # (B,1)   identical for img/tab

        # 4) DiT forward ----------------------------------------------------
        out = self.dit(
            x_img = c_in_img * x_img_noisy,
            x_tab = c_in_tab * x_tab_noisy,
            t     = c_noise.squeeze(-1),           # keep DiT sig‑shape agnostic
            mask_ratio = mask_ratio,
        )
        F_img = out["image_sample"]
        F_tab = out["tab_sample"]

        # 5) Convert ε‑prediction → denoised estimate D(x)
        D_img = self._to_D(x_img_noisy, σ_img, F_img)
        D_tab = self._to_D(x_tab_noisy, σ_tab, F_tab)

        # 6) Weighted EDM MSE – modality‑balanced
        w = ((σ_tab ** 2 + self.edm.sigma_data**2) /
             (σ_tab * self.edm.sigma_data) ** 2)   # (B,1)

        loss_img = (w.view(B,1,1,1) * (D_img - x_img).square()).mean()
        loss_tab = (w * (D_tab - x_tab).square()).mean() / self.num_tab_features
        total_loss = loss_img + loss_tab

        return total_loss, loss_img, loss_tab

    # ---------------------------------------------------------------------
    # Sampling (EDM tracker identical to training, but in a loop)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        guidance_scale: float = 1.0,
        num_steps: int = 30,
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

        # x(t=σ_max)       (shared init noise)
        x_img = torch.randn((batch_size, 4, 32, 32), device=device, generator=g)
        x_tab = torch.randn((batch_size, self.num_tab_features), device=device, generator=g)

        cfg_fwd = partial(self.dit.forward, cfg=guidance_scale) \
                  if guidance_scale > 1. else self.dit.forward

        # pre‑compute σ schedule
        N = num_steps
        step = torch.arange(N, device=device, dtype=torch.float64)
        σ = (self.edm.sigma_max**(1/self.edm.rho) +
             step / (N-1) * (self.edm.sigma_min**(1/self.edm.rho) -
                             self.edm.sigma_max**(1/self.edm.rho))) ** self.edm.rho
        σ = torch.cat([σ, σ.new_zeros(1)])        # append 0 for last update

        # main loop ---------------------------------------------------------
        x_img = x_img.double() * σ[0]
        x_tab = x_tab.double() * σ[0]

        for i, (σ_cur, σ_next) in enumerate(zip(σ[:-1], σ[1:])):
            # optional σ‑churn  (turned off by default)
            γ = min(self.edm.S_churn/N, np.sqrt(2)-1) \
                if self.edm.S_min <= σ_cur <= self.edm.S_max else 0.
            σ_hat = σ_cur + γ*σ_cur
            g_noise = self.edm.S_noise
            x_img_hat = x_img + (σ_hat**2 - σ_cur**2).sqrt() * g_noise * torch.randn_like(x_img)
            x_tab_hat = x_tab + (σ_hat**2 - σ_cur**2).sqrt() * g_noise * torch.randn_like(x_tab)

            # predict ε (or v) and map to D(x)
            t_hat_vec = _make_t(batch_size, σ_hat)
            out = cfg_fwd(
                x_img = x_img_hat.float(),
                x_tab = x_tab_hat.float(),
                t     = t_hat_vec,
                mask_ratio = 0.,
            )
            D_img = self._to_D(x_img_hat, σ_hat, out["image_sample"])
            D_tab = self._to_D(x_tab_hat, σ_hat, out["tab_sample"])

            # Euler step
            d_img = (x_img_hat - D_img) / σ_hat
            d_tab = (x_tab_hat - D_tab) / σ_hat
            x_img_next = x_img_hat + (σ_next - σ_hat) * d_img
            x_tab_next = x_tab_hat + (σ_next - σ_hat) * d_tab

            # 2nd‑order corrector (disabled at final step)
            if i < N-1:
                t_next_vec = _make_t(batch_size, σ_next)
                out = cfg_fwd(
                    x_img = x_img_next.float(),
                    x_tab = x_tab_next.float(),
                    t     = t_next_vec,
                    mask_ratio = 0.,
                )
                D_img_prime = self._to_D(x_img_next, σ_next, out["image_sample"])
                D_tab_prime = self._to_D(x_tab_next, σ_next, out["tab_sample"])
                d_img_prime = (x_img_next - D_img_prime) / σ_next
                d_tab_prime = (x_tab_next - D_tab_prime) / σ_next
                x_img_next = x_img_hat + (σ_next - σ_hat) * 0.5 * (d_img + d_img_prime)
                x_tab_next = x_tab_hat + (σ_next - σ_hat) * 0.5 * (d_tab + d_tab_prime)

            x_img, x_tab = x_img_next, x_tab_next

        x_img = x_img.float()
        x_tab = x_tab.float()

        if return_latents:
            return x_img, x_tab

        # -------- post‑process to original spaces --------------------------
        # x_tab = x_tab * self.tab_scaler_std               # un‑standardise
        # leave x_img as latent; caller can VAE.decode(...)
        # return x_img / self.img_latent_scale, x_tab
        return x_img, x_tab

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _to_D(self, x_noisy, σ, F_x):
        """EDM skip‑connection formula – works for scalar or tensor σ."""
        c_skip = self.edm.sigma_data**2 / (σ**2 + self.edm.sigma_data**2)
        c_out  = σ * self.edm.sigma_data / (σ**2 + self.edm.sigma_data**2).sqrt()
        return c_skip * x_noisy + c_out * F_x



def load_diffusion(cfg: DictConfig, dit_model: nn.Module, tmp_param: Any = None, **overrides: Any) -> nn.Module:
    """
    Load a MultiModalDiffusion model from config, injecting a pre-loaded DiT.
    """
    from utils.configurations import apply_overrides
    cfg = apply_overrides(cfg, overrides)
    print("[INFO] Loading Diffusion model with config:", cfg)

    if "diffusion" in cfg:
        diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg.diffusion)
    else:
        diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg)

    print("[INFO] Loaded Diffusion Model")
    return diffusion_model



if __name__ == "__main__":
    # Suppose we have:
    from torch import optim, Tensor

    # 1) A "MultiModalDiT" instance that expects:
    #    model(x_img, x_tab, time_scalar) -> {"img_out":..., "tab_out":...}
    from models.dit.dit_multimodal_add12 import MultiModalDiT
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