from typing import Any
from omegaconf import DictConfig

from diffusers import DPMSolverMultistepScheduler


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


################################################################################
#                              Helper functions                                #
################################################################################


from diffusers.schedulers.scheduling_ddpm import betas_for_alpha_bar
def cosine_betas_1000():
    """
    1 000‑step discrete version of the continuous cosine log‑SNR schedule
    (a.k.a. ‘squaredcos_cap_v2’ used by Stable‑Diffusion and your training loop).
    """
    betas = betas_for_alpha_bar(1000, alpha_transform_type="cosine", max_beta=0.999)
    return torch.tensor(betas, dtype=torch.float32)

COSINE_BETAS_1000 = cosine_betas_1000()

def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def right_pad_dims_to(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Pad `t` with singleton dimensions so it can be broadcast with `x`."""
    padding_dims = x.ndim - t.ndim
    return t.view(*t.shape, *((1,) * padding_dims)) if padding_dims > 0 else t


def log(t, eps: float = 1e-20):
    return torch.log(t.clamp(min=eps))


def alpha_cosine_log_snr(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:  # noqa: N802
    """Continuous cosine noise schedule from https://arxiv.org/abs/2202.00512."""
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

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

class MultiModalDiffusion(nn.Module):
    """Joint diffusion model for latent images (Stable‑Diffusion style) and
    continuous tabular rows (z‑score standardised).

    Parameters
    ----------
    dit_model: nn.Module
        Denoising U‑Net / DiT‑like backbone that must expose the signature
        `dit_model(x_img, x_tab, t) -> dict` with keys ``v_img`` and ``v_tab``
        containing v‑predictions for each modality.
    img_latent_shape: Tuple[int, int, int]
        Spatial dimensions of latent images, e.g. ``(4, 32, 32)``.
    num_tab_features: int
        Number of columns in the tabular data.
    lambda_tab: float, optional
        Relative weight of tabular loss w.r.t. image loss.
    num_sample_steps: int, optional
        Default number of sampling steps for the built‑in Karras DPM++ 2M solver.
    device: str | torch.device, optional
    """

    def __init__(
        self,
        dit: nn.Module,
        img_latent_shape: Tuple[int, int, int] = (4, 32, 32),
        num_tab_features: int = 157,
        *,
        lambda_tab: float = 1.0,
        num_sample_steps: int = 30,
        device: Optional[torch.device] = None,
            warmup_steps=None
    ) -> None:
        super().__init__()
        self.dit = dit
        self.img_channels, self.img_h, self.img_w = img_latent_shape
        self.num_tab_features = num_tab_features
        self.lambda_tab = lambda_tab
        self.num_sample_steps = num_sample_steps

        # noise schedule (continuous cosine)
        self.log_snr_fn = alpha_cosine_log_snr

        # register a buffer so it moves with the module; useful for saving state_dict
        self.register_buffer("_dummy", torch.empty(0))

        # pre‑compute per‑modality scaling for MSE to balance contributions
        self.pixel_count = self.img_channels * self.img_h * self.img_w
        self.tab_size = self.num_tab_features
        if lambda_tab is None:
            # heuristic balancing
            self.lambda_tab = self.pixel_count / self.tab_size

        self._device_override = device

    # ---------------------------------------------------------------------
    # properties
    # ---------------------------------------------------------------------
    @property
    def device(self):  # noqa: D401
        return self._device_override or self._dummy.device

    # ---------------------------------------------------------------------
    # noise helpers – continuous v‑prediction following https://arxiv.org/abs/2202.00512
    # ---------------------------------------------------------------------
    def _q_sample(self, x_start: torch.Tensor, times: torch.Tensor, *, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to `x_start` at timestep `times` (in [0,1]). Returns (x_noised, alpha, sigma)."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr_fn(times)
        log_snr = right_pad_dims_to(x_start, log_snr)

        alpha = (log_snr.sigmoid()).sqrt()
        sigma = ((-log_snr).sigmoid()).sqrt()
        x_noised = alpha * x_start + sigma * noise
        return x_noised, alpha, sigma

    def _random_times(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, device=self.device)

    # ---------------------------------------------------------------------
    # forward / loss (training)
    # ---------------------------------------------------------------------
    def forward(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        *,
        times: Optional[torch.Tensor] = None,
        noise_img: Optional[torch.Tensor] = None,
        noise_tab: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint training loss.

        Returned tuple: (total_loss, loss_img, loss_tab)
        """
        b = x_img.shape[0]
        assert x_tab.shape[0] == b, "batch size mismatch between image and tabular inputs"
        if times is None:
            times = self._random_times(b)

        # noise the inputs (cosine continuous schedule)
        x_img_noised, alpha_img, sigma_img = self._q_sample(x_img, times, noise=noise_img)
        x_tab_noised, alpha_tab, sigma_tab = self._q_sample(x_tab, times, noise=noise_tab)

        # v targets
        v_img_target = alpha_img * (noise_img if noise_img is not None else (x_img_noised - alpha_img * x_img) / sigma_img) - sigma_img * x_img  # simplifies to alpha*noise - sigma*x0
        v_tab_target = alpha_tab * (noise_tab if noise_tab is not None else (x_tab_noised - alpha_tab * x_tab) / sigma_tab) - sigma_tab * x_tab

        # model predicts v directly (stable‑diffusion style)
        out = self.dit(x_img=x_img_noised, x_tab=x_tab_noised, t=times)
        v_img_pred = out["image_sample"]
        v_tab_pred = out["tab_sample"]

        # losses
        loss_img = F.mse_loss(v_img_pred, v_img_target)
        loss_tab = F.mse_loss(v_tab_pred, v_tab_target)
        total_loss = loss_img + self.lambda_tab * loss_tab
        return total_loss, loss_img.detach(), loss_tab.detach()

    # ---------------------------------------------------------------------
    # sampling (DPM++ 2M Karras)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        *,
        model_ema: nn.Module,
        batch_size: int = 4,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate (`latent_img`, `latent_tab`) via Karras DPM++ 2M sampler."""
        n_steps = num_steps or self.num_sample_steps
        device = self.device

        # initialise latents
        latents_img = torch.randn(
            batch_size,
            self.img_channels,
            self.img_h,
            self.img_w,
            device=device,
            generator=generator,
        )
        latents_tab = torch.randn(batch_size, self.num_tab_features, device=device, generator=generator)

        # separate schedulers – identical hyper‑params, distinct internal state
        def make_scheduler():
            return DPMSolverMultistepScheduler(
                trained_betas=np.asarray(COSINE_BETAS_1000),
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=False,
                prediction_type="v_prediction",
            )

        scheduler_img = make_scheduler()
        scheduler_tab = make_scheduler()
        scheduler_img.set_timesteps(n_steps, device=device)
        scheduler_tab.set_timesteps(n_steps, device=device)

        timesteps = scheduler_img.timesteps

        for i, t in enumerate(timesteps):
            # scale model inputs per diffusers requirement
            latent_img_in = scheduler_img.scale_model_input(latents_img, t)
            latent_tab_in = scheduler_tab.scale_model_input(latents_tab, t)
            t_batch = t.unsqueeze(0).expand(latents_img.shape[0])

            # DiT forward expects the *same* timestep tensor for both modalities
            with torch.autocast(device_type="cuda", enabled=torch.is_autocast_enabled()):
                preds = self.dit(x_img=latent_img_in, x_tab=latent_tab_in, t=t_batch)
            v_img_pred, v_tab_pred = preds["image_sample"], preds["tab_sample"]

            # scheduler step for each modality (independent but shared timestep)
            latents_img = scheduler_img.step(v_img_pred, t, latents_img).prev_sample
            latents_tab = scheduler_tab.step(v_tab_pred, t, latents_tab).prev_sample

        return latents_img, latents_tab


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