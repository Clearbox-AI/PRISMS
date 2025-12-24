"""
Multimodal diffusion wrapper.

This module implements an EDM-Karras diffuser over a DiT multimodal backbone, with:
- optional TabDiff-style feature-wise noise scaling
- optional TabSyn VAE latent for tabular data
- branch balancing (GradNorm) and cross-modal stabilization utilities
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from omegaconf import DictConfig

from data.tabular_transforms import FittedTransforms, inverse_transform
from models.dit.dit_multimodal import FeedForwardECMoe, TabSynVAE
from models.diffusion.mm_diffusion_utils import (
    coral_loss,
    info_nce,
    log_sigma_to_t,
    min_snr_weight,
)


# -----------------------------------------------------------------------------#
# Main class
# -----------------------------------------------------------------------------#
class MultiModalDiffusion(nn.Module):
    """
    EDM-Karras multimodal diffuser with:
    - optional feature-wise σ for tabular inputs (TabDiff-style)
    - GradNorm-based dynamic loss balancing
    - dual timestamp support (t_img, t_tab) in the DiT backbone
    """

    def __init__(
            self,
            *,
            dit: nn.Module,
            tab_transforms: FittedTransforms,
            num_tab_features: int,
            # EDM hyper-parameters (image)
            sigma_min: float = 0.002,
            sigma_max: float = 50.0,
            num_steps: int = 40,
            rho: int = 7,
            P_mean: float = -1.2,
            P_std: float = 1.2,
            S_churn: float = 0.0,
            S_min: float = 0.0,
            S_max: float = float("inf"),
            S_noise: float = 1.0,
            # EDM hyper-parameters (tabular, narrower schedule)
            tab_sigma_min: float = 0.15,  # sweep: 0.10–0.25
            tab_sigma_max_sched: float = 1.0,
            tab_rho: int = 5,
            tab_P_mean: float = -0.1,
            tab_P_std: float = 0.5,
            # Min-SNR weighting (0 disables)
            min_snr_gamma_img: float = 5.0,
            min_snr_gamma_tab: float = 2.0,  # clamps tiny-σ dominance for tab
            # TabDiff
            tab_sigma_max: float = float("inf"),
            feature_wise_sigma: bool = True,
            lambda_sigma: float = 1e-5,
            lambda_out: float = 0,  # 1e-5,
            freeze_alpha_at: int = 8_000,
            lambda_corr: float = 0.05,
            lambda_nce: float = 0.1,
            # misc
            class_counts: Tuple[int, int] = (1085, 307),  # (negatives, positives)
            dtype: str = "bfloat16",
            noise_select=True,
            cond_dropout_p: float = 0.0,
            xattn_drop_p: float = 0.12,
            img_loss_anneal: Tuple[int, int, float] = (15_000, 45_000, 0.85),  # from, to, min_factor
            lambda_var: float = 0.025,
            lambda_cat_ce: float = 0.08,
            lambda_swd: float = 0.07,
            lambda_skewkurt: float = 0.015,
            self_cond_p: float = 0,
            # branch pacing & balancing
            detach_cross_until: int = 2000,
            tab_update_prob_start: float = 0.25,
            tab_update_prob_end: float = 1.0,
            tab_update_warmup_to: int = 25_000,
            xattn_hard_off_until: int = 0,
            freeze_sigma_data_img: bool = True,
            freeze_sigma_data_tab: bool = True,
            gradnorm_mode: str = "heads",  # {"heads","all"}
            gradnorm_penalty: float = 0.10,
            # gradient-only cross-modal ramp (forward intact; gradients scaled)
            cross_grad_scale_start: float = 0.2,
            cross_grad_scale_end: float = 1.0,
            cross_grad_warmup_to: int = 25_000,
            cross_grad_scale_mode: str = "adaptive",  # {"adaptive","ramp"}
            # GradNorm clamp
            gradnorm_w_tab_min: float = 0.5,
            gradnorm_w_tab_max: float = 2.0,
            light_metrics_every: int = 500,
            # TabSyn VAE integration
            use_tabsyn_vae: bool = True,
            tabsyn_d_token: int = 8,
            tabsyn_beta: float = 0.02,
            tabsyn_pretrain_steps: int = 10000,
            tabsyn_num_mixtures: int = 5,
            tabsyn_dropout_p: float = 0.0,
            tabsyn_beta_start: float = 1e-5,
            tabsyn_beta_warmup: Optional[int] = 6000,
            tabsyn_free_bits: float = 0.1,  # nats per latent dim
            # stochastic latent training (posterior temperature τ)
            tabsyn_latent_tau_start: float = 0.0,
            tabsyn_latent_tau_end: float = 0.6,  # sweep: 0.4–0.8
            tabsyn_latent_tau_warmup_to: int = 20_000,  # sweep: 15–30k
            tabsyn_latent_deterministic: bool = True,  # prefer μ-only during diffusion training
            # warm-up for tab EDM / Min-SNR weights
            tab_edm_weight_warmup_to: int = 20_000,
            # latent α (cat/num groups) when TabSyn VAE is enabled
            latent_alpha_mode: str = "group",  # {"none","group"}
            latent_alpha_init: float = 1.0,
            # image regularizers (disabled by default in latent space training)
            lambda_tv_img: float = 0.0,
            lambda_fft_img: float = 0.0,
    ):
        super().__init__()
        self.dit = dit
        self.dtype = dtype

        self.noise_select = bool(noise_select)

        # --- tab/image auxiliary losses and training knobs ---
        self.lambda_corr = lambda_corr
        self.lambda_nce = lambda_nce
        self.cond_dropout_p = float(cond_dropout_p)
        self.xattn_drop_p = float(xattn_drop_p)
        self.img_loss_anneal = img_loss_anneal
        self.lambda_var = float(lambda_var)
        self.lambda_cat_ce = float(lambda_cat_ce)
        self.lambda_swd = float(lambda_swd)
        self.lambda_skewkurt = float(lambda_skewkurt)

        # Image anti-artifact regularizers (used in forward)
        self.lambda_tv_img = float(lambda_tv_img)
        self.lambda_fft_img = float(lambda_fft_img)

        # Additional knobs (kept identical to original behavior)
        self.lambda_mmd_tab = 0.08
        self.calibrate_tab_var = True
        self.calibrate_tab_var_clip = 0.10
        self.posterior_sigma_data = True

        # Relative SNR floor for tab weights (weights only; forward unchanged)
        self._snr_floor_rel = 0.25

        # Per-feature SNR equalization on the tab branch
        self.snr_equalize_exp = 1.0  # κ: 0=off, 0.5=partial, 1.0=full
        # Categorical marginal matching
        self.lambda_cat_marg = 0.05
        # Light mean matching for numerics (in addition to variance)
        self.lambda_mean = 0.02
        # Mean calibration during sampling (in addition to variance)
        self.calibrate_tab_mean = True
        self.calibrate_tab_mean_clip = 0.12
        # Dynamic thresholding (disabled in latent space; pixel-space can apply downstream)
        self.dynamic_threshold_img = False
        self.dynamic_threshold_p = 0.995
        self.dynamic_threshold_rescale = False
        # Parametric ramp for image regularizers
        self.img_reg_ramp = (10_000, 20_000)
        # Deterministic tab refinement at the end of sampling
        self.tab_refine_steps = 2
        self.tab_refine_mix = 0.30

        # pacing / balancing
        self.detach_cross_until = int(detach_cross_until)
        self.tab_update_prob_start = float(tab_update_prob_start)
        self.tab_update_prob_end = float(tab_update_prob_end)
        self.tab_update_warmup_to = int(tab_update_warmup_to)
        self.gradnorm_mode = str(gradnorm_mode)
        self.gradnorm_penalty = float(gradnorm_penalty)
        self.cross_grad_scale_start = float(cross_grad_scale_start)
        self.cross_grad_scale_end = float(cross_grad_scale_end)
        self.cross_grad_warmup_to = int(cross_grad_warmup_to)

        # runtime gates / flags
        self._xattn_hard_off_until = int(xattn_hard_off_until)
        self._freeze_sigma_data_img = bool(freeze_sigma_data_img)
        self._freeze_sigma_data_tab = bool(freeze_sigma_data_tab)
        self.cross_grad_scale_mode = str(cross_grad_scale_mode)
        self.gradnorm_w_tab_min = float(gradnorm_w_tab_min)
        self.gradnorm_w_tab_max = float(gradnorm_w_tab_max)

        # Optional: normalize per-feature Min-SNR weights across features (intra-sample)
        self.normalize_tab_min_snr = True

        # ---------------- tabular dims & transforms ------------------ #
        self.ft = tab_transforms
        cat_dim = (sum(len(c) for c in tab_transforms.cat_encoder.categories_) if tab_transforms.cat_features else 0)
        n_cat_feats = len(tab_transforms.cat_features) if tab_transforms.cat_features else 0
        self._n_cat_feats = int(n_cat_feats)

        n_num_feats = len(tab_transforms.num_features)

        self._tabsyn_use = bool(use_tabsyn_vae)
        self._latent_alpha_mode = str(latent_alpha_mode).lower()
        self._latent_alpha_init = float(latent_alpha_init)
        self._tabsyn_d = int(tabsyn_d_token)
        if use_tabsyn_vae and hasattr(self.dit, "tabsyn_d_token"):
            self._tabsyn_d = int(getattr(self.dit, "tabsyn_d_token"))
        self._tabsyn_pretrain = int(tabsyn_pretrain_steps)
        self._tabsyn_beta = float(tabsyn_beta)
        self._tabsyn_num_mixtures = int(tabsyn_num_mixtures)
        self._tabsyn_dropout_p = float(tabsyn_dropout_p)
        self._tabsyn_beta_start = float(tabsyn_beta_start)
        self._tabsyn_beta_warmup = int(tabsyn_beta_warmup or max(1, int(0.3 * self._tabsyn_pretrain)))

        # Cyclical β-annealing (triangular). Period defaults to warmup.
        self._tabsyn_beta_cyc = True
        self._tabsyn_beta_cyc_period = int(
            self._tabsyn_beta_warmup or max(1, int(0.25 * self._tabsyn_pretrain))
        )

        self._tabsyn_free_bits = float(tabsyn_free_bits)

        # posterior temperature schedule
        self._tau_start = float(tabsyn_latent_tau_start)
        self._tau_end = float(tabsyn_latent_tau_end)
        self._tau_wu = int(tabsyn_latent_tau_warmup_to)
        self._tabsyn_deterministic = bool(tabsyn_latent_deterministic)

        # EDM warm-up for tab weights
        self._tab_edm_wu = int(tab_edm_weight_warmup_to)

        if self._tabsyn_use:
            # VAE latent dimension: tokens_per_row * d_token
            self.n_tab_tokens = n_cat_feats + n_num_feats
            self.tab_outdim = self.n_tab_tokens * self._tabsyn_d

            # Consistency guard: ensure dataset transforms match the DiT tokenization.
            if hasattr(self.dit, "n_tab_tokens") and int(self.dit.n_tab_tokens) != int(self.n_tab_tokens):
                raise ValueError(
                    f"Inconsistent n_tab_tokens: DiT={int(self.dit.n_tab_tokens)} "
                    f"but transforms/dataset={int(self.n_tab_tokens)}. "
                    "Align `num_numeric` and `categorical_cardinalities` in the DiT with dataset transforms."
                )
        else:
            self.tab_outdim = n_num_feats + cat_dim

        self.num_tab_features = num_tab_features  # raw feature count (pre one-hot)

        # ---------------- EDM configuration --------------------------- #
        self.edm_img = EasyDict(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_steps=num_steps,
            rho=rho,
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=0.9,  # kept for backward compatibility; weights use sigma_data_img buffer
            S_churn=S_churn,
            S_min=S_min,
            S_max=S_max,
            S_noise=S_noise,
        )

        self.edm_tab = EasyDict(
            sigma_min=tab_sigma_min,
            sigma_max=tab_sigma_max_sched,
            num_steps=num_steps,
            rho=tab_rho,
            P_mean=tab_P_mean,
            P_std=tab_P_std,
            S_churn=0.0,
            S_min=0.0,
            S_max=float("inf"),
            S_noise=1.0,
        )

        self._metrics_every = light_metrics_every

        # Branch-specific sigma_data with EMA calibration
        self.register_buffer("sigma_data_img", torch.tensor(0.5))
        self.register_buffer("sigma_data_tab_vec", torch.ones(self.tab_outdim))
        self.register_buffer("mu_data_tab_vec", torch.zeros(self.tab_outdim))
        self._sigma_ema = 0.999
        self.self_cond_p = float(self_cond_p)

        # Min-SNR-γ
        self.min_snr_gamma_img = float(min_snr_gamma_img)
        self.min_snr_gamma_tab = float(min_snr_gamma_tab)

        # ---------------- TabDiff setup ------------------------------ #
        self.tab_sigma_max = float(tab_sigma_max)
        # In latent mode, feature-wise alpha is handled separately (group α), so disable tab_alpha vector.
        self.feature_wise = bool(feature_wise_sigma and (not self._tabsyn_use))
        self.lambda_sigma = lambda_sigma * num_tab_features / 250.0
        self.lambda_out = lambda_out * num_tab_features / 250.0
        self.freeze_alpha_at = freeze_alpha_at

        if self.feature_wise:
            # Will be initialized from data stats on first forward pass.
            self.tab_alpha = nn.Parameter(torch.full((self.tab_outdim,), 0.8))

        # ---------------- GradNorm variables ------------------------- #
        # Single learnable log-weight for tab branch scaling
        self.log_w_tab = nn.Parameter(torch.zeros(()))

        # Global step counter (for ramps, freezes, etc.)
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))

        # Lightweight EMA of recent MSEs used for adaptive cross-grad gating
        self.register_buffer("_ema_img_mse", torch.ones((), dtype=torch.float32))
        self.register_buffer("_ema_tab_mse", torch.ones((), dtype=torch.float32))
        self._ema_mse_beta = 0.95

        self.n_neg, self.n_pos = class_counts

        # Broadcast-safety flag: set after alpha initialization and (optional) broadcast
        self._alpha_init_done = False

        # Wire the global step counter into each MoE MLP (if present)
        for m in self.dit.modules():
            if isinstance(m, FeedForwardECMoe):
                m.step_counter = self.step_counter

        # Per-dimension weights so each categorical feature contributes ~1 total weight
        dim_w = []
        if not self._tabsyn_use:
            if getattr(self.ft, "cat_dims", None):
                for k in self.ft.cat_dims:
                    dim_w += [1.0 / max(1, k)] * k
            dim_w += [1.0] * len(self.ft.num_features)
        else:
            # Latent mode: uniform weighting across latent dimensions
            dim_w = [1.0] * self.tab_outdim
        self.register_buffer("tab_dim_weights", torch.tensor(dim_w, dtype=torch.float32))

        # ---------------- TabSyn VAE module -------------------------- #
        if self._tabsyn_use:
            self.tab_vae = TabSynVAE(
                self.ft.cat_dims,
                len(self.ft.num_features),
                d_token=self._tabsyn_d,
                beta=self._tabsyn_beta,
                num_mixtures=self._tabsyn_num_mixtures,
                dropout_p=self._tabsyn_dropout_p,
                kl_free_bits=self._tabsyn_free_bits,
            )

            # Decoder sampling preferences (inference only)
            self.tab_vae.mdn_sample_scale = 1.15
            self.tab_vae.mdn_kind = "logistic_mixture"

            # When using VAE latent, prefer EDM weights for the tab branch
            self.use_tab_edm_weights = True

            # Group α for latent dims (cat vs num), frozen at the same time as tab_alpha
            if self._latent_alpha_mode != "none":
                self.tab_alpha_latent_cat = nn.Parameter(torch.tensor(self._latent_alpha_init))
                self.tab_alpha_latent_num = nn.Parameter(torch.tensor(self._latent_alpha_init))

        # EMA of categorical marginals as a stable target for marginal matching
        n_cat_oh = sum(self.ft.cat_dims) if getattr(self.ft, "cat_dims", None) else 0
        if n_cat_oh > 0:
            self.register_buffer("_cat_marg_ema", torch.full((n_cat_oh,), 0.0))
        else:
            self.register_buffer("_cat_marg_ema", torch.zeros(0))

        # Runtime helper for external TabSyn-VAE pretraining
        self._tabsyn_pretrained = False

    # ==========================
    # TabSyn-VAE helper API
    # ==========================
    def has_tabsyn_vae(self) -> bool:
        return bool(getattr(self, "_tabsyn_use", False))

    @torch.no_grad()
    def freeze_tab_vae(self, requires_grad: bool = False):
        """
        Freeze (default) or unfreeze TabSyn-VAE parameters.
          - requires_grad=True  -> trainable (used during VAE pretraining)
          - requires_grad=False -> frozen (used during diffusion training)
        """
        if not self.has_tabsyn_vae():
            return
        for p in self.tab_vae.parameters():
            p.requires_grad_(requires_grad)

    def tabsyn_vae_parameters(self):
        """Iterator over TabSyn-VAE parameters (empty if disabled)."""
        if not self.has_tabsyn_vae():
            return []
        return self.tab_vae.parameters()

    def tabsyn_pretrain_step(self, x_tab: torch.Tensor, *, advance_counter: bool = True):
        """
        One standalone TabSyn-VAE pretraining step with β-annealing and free-bits.

        Returns: loss, ce, nll, kl (tensors on x_tab.device).
        Does not touch `self.step_counter`.
        """
        assert self.has_tabsyn_vae(), "tabsyn_pretrain_step called but use_tabsyn_vae=False"
        self.tab_vae.train()

        if not hasattr(self, "_tabsyn_pt_step"):
            # Ensure this buffer is created on the same device as the module
            dev = next(self.parameters()).device
            self.register_buffer("_tabsyn_pt_step", torch.zeros((), dtype=torch.long, device=dev))

        s = int(self._tabsyn_pt_step.item())

        # β schedule: cyclical triangular (Fu et al., 2019) or linear warm-up fallback
        if getattr(self, "_tabsyn_beta_cyc", False) and self._tabsyn_beta_warmup > 0:
            T = max(1, int(getattr(self, "_tabsyn_beta_cyc_period", self._tabsyn_beta_warmup)))
            phase = (s % (2 * T)) / float(T)  # 0→1→0 over 2T steps
            r = phase if phase <= 1.0 else (2.0 - phase)
            beta_now = self._tabsyn_beta_start + (self._tabsyn_beta - self._tabsyn_beta_start) * float(r)
        else:
            if self._tabsyn_beta_warmup <= 0:
                beta_now = self._tabsyn_beta
            else:
                r = min(1.0, s / float(self._tabsyn_beta_warmup))
                beta_now = self._tabsyn_beta_start + (self._tabsyn_beta - self._tabsyn_beta_start) * r

        loss, ce, nll, kl = self.tab_vae.loss_forward(x_tab, beta=beta_now, free_bits=self._tabsyn_free_bits)
        if advance_counter:
            self._tabsyn_pt_step += 1
        return loss, ce, nll, kl

    def tabsyn_save(self, path: str):
        """Save only the TabSyn-VAE state_dict plus minimal metadata."""
        assert self.has_tabsyn_vae(), "tabsyn_save called but use_tabsyn_vae=False"
        payload = {
            "state_dict": {k: v.cpu() for k, v in self.tab_vae.state_dict().items()},
            "meta": {
                "cat_dims": list(self.ft.cat_dims),
                "num_numeric": len(self.ft.num_features),
                "d_token": int(self._tabsyn_d),
                "beta": float(self._tabsyn_beta),
                "num_mixtures": int(self._tabsyn_num_mixtures),
            },
        }
        torch.save(payload, path)

    def tabsyn_load(self, path: str, map_location: str | torch.device | None = None, strict: bool = True):
        """Load only TabSyn-VAE weights; leaves the rest of the diffuser intact."""
        assert self.has_tabsyn_vae(), "tabsyn_load called but use_tabsyn_vae=False"
        payload = torch.load(path, map_location=map_location or "cpu")
        self.tab_vae.load_state_dict(payload["state_dict"], strict=strict)
        self._tabsyn_pretrained = True

    # -----------------------------------------------------------------
    # Forward – training
    # -----------------------------------------------------------------
    def forward(
            self,
            x_img: torch.Tensor,  # (B,4,32,32) latent-space image
            x_tab: torch.Tensor,  # (B,F) model-space (transforms) OR VAE input (if enabled)
            *,
            mask_ratio: float = 0.0,
            labels: Optional[torch.Tensor] = None,
            stage: str = "diffusion",
    ) -> Tuple[torch.Tensor, ...]:

        # Special mode: TabSyn-VAE pretraining (run through DDP normally)
        if stage == "vae_pretrain":
            assert self.has_tabsyn_vae(), "vae_pretrain stage requested but use_tabsyn_vae=False"
            loss, ce, nll, kl = self.tabsyn_pretrain_step(x_tab, advance_counter=True)
            return loss, ce, nll, kl

        # Label-dropout for CFG (10%)
        if labels is not None:
            drop = torch.rand_like(labels, dtype=torch.float) < 0.10
            labels = labels.clone()
            labels[drop] = self.dit.NULL_ID
        else:
            labels = None

        B, device = x_img.size(0), x_img.device
        self.step_counter += 1
        step_i = int(self.step_counter.item())

        # Gradient-only cross-modal warm-up gate (no hard detach)
        if self.cross_grad_scale_mode.lower() == "adaptive":
            prev_img = float(self._ema_img_mse.item())
            prev_tab = float(self._ema_tab_mse.item())
            denom = max(1e-8, prev_img + prev_tab)
            p_img = prev_img / denom  # fraction of image error
            g_cross = self.cross_grad_scale_start + (self.cross_grad_scale_end - self.cross_grad_scale_start) * (
                (1.0 - p_img) ** 2
            )
            g_cross = float(max(0.0, min(1.0, g_cross)))
        else:
            r = (self.step_counter.float() / max(1, self.cross_grad_warmup_to)).clamp_(0, 1)
            g_cross = float(self.cross_grad_scale_start + (self.cross_grad_scale_end - self.cross_grad_scale_start) * r)

        # Detach cross-attention gradients during the initial stabilization window
        detach_img2tab = step_i < int(self.detach_cross_until)
        detach_tab2img = step_i < int(self.detach_cross_until)

        # Encode tab input to latent (no grad through VAE during diffusion training)
        if self._tabsyn_use:
            with torch.no_grad():
                z_tokens, mu, logv = self.tab_vae.encode(x_tab)  # (B,T,d), (B,T,d), (B,T,d)
                if self._tabsyn_deterministic:
                    z = mu
                else:
                    std = (0.5 * logv).exp()
                    t = min(1.0, float(self.step_counter.item()) / max(1, self._tau_wu))
                    tau = self._tau_start + (self._tau_end - self._tau_start) * t
                    z = mu + float(tau) * std * torch.randn_like(std)
                x_tab_lat = z.reshape(z.size(0), -1)  # (B, E)
        else:
            x_tab_lat = x_tab

        # Update EMA estimates for sigma_data and mean (optionally frozen)
        with torch.no_grad():
            sd_img = x_img.float().std()
            if self._tabsyn_use and self.posterior_sigma_data:
                mu_flat = mu.reshape(mu.size(0), -1).float()
                std_flat = (0.5 * logv).exp().reshape(logv.size(0), -1).float()
                var_mu = mu_flat.var(dim=0, unbiased=False)
                mean_var = (std_flat ** 2).mean(dim=0)
                sd_tab_vec = (var_mu + mean_var).sqrt()
                mean_tab_vec = mu_flat.mean(dim=0)
            else:
                sd_tab_vec = x_tab_lat.float().std(dim=0)
                mean_tab_vec = x_tab_lat.float().mean(dim=0)

            if not self._freeze_sigma_data_img:
                self.sigma_data_img.mul_(self._sigma_ema).add_((1 - self._sigma_ema) * sd_img)
            if not self._freeze_sigma_data_tab:
                self.sigma_data_tab_vec.mul_(self._sigma_ema).add_((1 - self._sigma_ema) * sd_tab_vec)

            # Mean EMA is maintained regardless; it is not used in EDM weights
            self.mu_data_tab_vec.mul_(self._sigma_ema).add_((1 - self._sigma_ema) * mean_tab_vec)

        # Draw separate σ per modality (branch-specific log-normal)
        sigma_img = (torch.randn(B, 1, device=device) * self.edm_img.P_std + self.edm_img.P_mean).exp()
        sigma_tab = (torch.randn(B, 1, device=device) * self.edm_tab.P_std + self.edm_tab.P_mean).exp()

        # Injected σ shapes
        σ_img_inj = sigma_img.view(B, 1, 1, 1)

        # Tab base σ (with optional hard cap)
        σ_tab_base = sigma_tab
        if torch.isfinite(torch.tensor(self.tab_sigma_max)):
            σ_tab_base = σ_tab_base.clamp(max=self.tab_sigma_max)

        # One-time feature-wise alpha init (TabDiff mode only)
        if self.feature_wise and not self._alpha_init_done:
            with torch.no_grad():
                std_ratio = x_tab.std(0) / (x_tab.std() + 1e-8)
                std_ratio = std_ratio.clamp(min=1e-3)
                self.tab_alpha.data.copy_(std_ratio)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(self.tab_alpha.data, src=0)
            self._alpha_init_done = True

        # Feature-wise σ for tabular: latent group α (TabSyn) or TabDiff α vector
        if self._tabsyn_use and (self._latent_alpha_mode != "none"):
            T_cat = self._n_cat_feats * self._tabsyn_d
            α_cat = self.tab_alpha_latent_cat.abs().clamp(0.75, 1.75)
            α_num = self.tab_alpha_latent_num.abs().clamp(0.75, 1.75)
            αvec = torch.ones(self.tab_outdim, device=device, dtype=σ_tab_base.dtype)
            if T_cat > 0:
                αvec[:T_cat] = α_cat
            if T_cat < self.tab_outdim:
                αvec[T_cat:] = α_num
            σ_tab_inj = σ_tab_base * αvec.view(1, -1)
        elif self.feature_wise:
            α = self.tab_alpha.abs().clamp(1.0, 2.0).view(1, -1).to(x_tab)
            σ_tab_inj = σ_tab_base * α
        else:
            σ_tab_inj = σ_tab_base.expand(-1, self.tab_outdim)

        # Optional per-feature SNR equalization
        if self.snr_equalize_exp > 0.0:
            snr_eq = (self.sigma_data_tab_vec / (self.sigma_data_tab_vec.mean() + 1e-8))
            snr_eq = snr_eq.clamp(min=1e-3, max=1e3).pow(self.snr_equalize_exp).to(σ_tab_inj)
            σ_tab_inj = σ_tab_inj * snr_eq.view(1, -1)

        # Add Gaussian noise
        x_img_noisy = x_img + torch.randn_like(x_img) * σ_img_inj
        x_tab_noisy = x_tab_lat + torch.randn_like(x_tab_lat) * σ_tab_inj

        # EDM time embeddings:
        # - image uses scalar σ
        # - tab uses RMS(σ_tab_inj) per batch item
        t_img = log_sigma_to_t(sigma_img.squeeze(1))
        σ_tab_rms = σ_tab_inj.double().pow(2).mean(dim=1).sqrt().float()  # (B,)
        t_tab = log_sigma_to_t(σ_tab_rms)

        # Modality dropout: disable cross-attention gates only (annealed at start)
        disable_img2tab = False
        disable_tab2img = False
        if self.training and self.xattn_drop_p > 0.0:
            warm = (self.step_counter.float() / 2000.0).clamp(max=1.0)
            p_now = self.xattn_drop_p * (1.0 - warm)
            if torch.rand((), device=device) < p_now:
                disable_img2tab = True
            if torch.rand((), device=device) < p_now:
                disable_tab2img = True

        # Optional hard warm-up: disable both bridges for the first N iterations
        if step_i < self._xattn_hard_off_until:
            disable_img2tab = True
            disable_tab2img = True

        # Self-conditioning in preconditioned space (teacher pass, stop-grad)
        if (self.self_cond_p > 0.0) and (torch.rand((), device=device) < self.self_cond_p):
            with torch.no_grad():
                den_img = (self.sigma_data_img ** 2 + σ_img_inj ** 2).sqrt()
                den_tab = (self.sigma_data_tab_vec.view(1, -1) ** 2 + σ_tab_inj ** 2).sqrt()
                mp_dtype = next(self.dit.parameters()).dtype
                x_img_in = (x_img_noisy * (1.0 / den_img)).to(dtype=mp_dtype)
                x_tab_in = (x_tab_noisy * (1.0 / den_tab)).to(dtype=mp_dtype)

                sc_out = self.dit(
                    x_img=x_img_in,
                    x_tab=x_tab_in,
                    t_img=t_img, t_tab=t_tab, labels=labels,
                    mask_ratio=mask_ratio, self_cond_img=None, self_cond_tab=None, cfg=1.0,
                    disable_xattn_img2tab=disable_img2tab, disable_xattn_tab2img=disable_tab2img,
                    detach_xattn_img2tab=detach_img2tab, detach_xattn_tab2img=detach_tab2img,
                    gradscale_xattn_img2tab=g_cross, gradscale_xattn_tab2img=g_cross
                )

                D_img_sc = self._to_D(x_img_noisy, σ_img_inj, sc_out["image_sample"], self.sigma_data_img)
                D_tab_sc = self._to_D(x_tab_noisy, σ_tab_inj, sc_out["tab_sample"], self.sigma_data_tab_vec.view(1, -1))

                # Map back to the preconditioned input domain and pass residuals
                self_cond_img = ((D_img_sc / den_img) - x_img_in).to(dtype=mp_dtype).detach()
                self_cond_tab = ((D_tab_sc / den_tab) - x_tab_in).to(dtype=mp_dtype).detach()
        else:
            self_cond_img = None
            self_cond_tab = None

        # Backbone call (preconditioning matches EDM)
        den_img = (self.sigma_data_img ** 2 + σ_img_inj ** 2).sqrt()
        den_tab = (self.sigma_data_tab_vec.view(1, -1) ** 2 + σ_tab_inj ** 2).sqrt()

        mp_dtype = next(self.dit.parameters()).dtype
        x_img_in = (x_img_noisy * (1.0 / den_img)).to(dtype=mp_dtype)
        x_tab_in = (x_tab_noisy * (1.0 / den_tab)).to(dtype=mp_dtype)

        out = self.dit(
            x_img=x_img_in,
            x_tab=x_tab_in,
            t_img=t_img,
            t_tab=t_tab,
            labels=labels,
            mask_ratio=mask_ratio,
            self_cond_img=self_cond_img,
            self_cond_tab=self_cond_tab,
            disable_xattn_img2tab=disable_img2tab, disable_xattn_tab2img=disable_tab2img,
            detach_xattn_img2tab=detach_img2tab, detach_xattn_tab2img=detach_tab2img,
            gradscale_xattn_img2tab=g_cross, gradscale_xattn_tab2img=g_cross
        )
        F_img, F_tab, diag_logits = out["image_sample"], out["tab_sample"], out["diag_logits"]

        # ε-to-D(x) mapping
        D_img = self._to_D(x_img_noisy, σ_img_inj, F_img, self.sigma_data_img)
        D_tab = self._to_D(x_tab_noisy, σ_tab_inj, F_tab, self.sigma_data_tab_vec.view(1, -1))

        # Weighted MSE losses (EDM)
        w_img = ((σ_img_inj ** 2 + self.sigma_data_img ** 2) / (σ_img_inj * self.sigma_data_img) ** 2)

        # Tab branch: optionally delay EDM/Min-SNR weighting to avoid early tiny-σ dominance
        use_tab_edm_now = bool(getattr(self, "use_tab_edm_weights", False))
        if use_tab_edm_now and (step_i >= self._tab_edm_wu):
            sd_vec = self.sigma_data_tab_vec.view(1, -1)
            σ_eff = torch.maximum(σ_tab_inj, self._snr_floor_rel * sd_vec)
            w_tab = ((σ_eff ** 2 + sd_vec ** 2) / (σ_eff * sd_vec) ** 2)
            w_tab = w_tab * min_snr_weight(σ_eff, sd_vec, self.min_snr_gamma_tab)

            if self.normalize_tab_min_snr:
                w_mean = w_tab.mean(dim=1, keepdim=True).clamp_min(1e-8)
                w_tab = w_tab / w_mean
        else:
            w_tab = torch.ones_like(D_tab)

        # Image: EDM + Min-SNR (with late γ annealing)
        gamma_img_now = float(self.min_snr_gamma_img)
        if step_i >= 30_000:
            r = min(1.0, (step_i - 30_000) / 20_000.0)
            gamma_img_now = self.min_snr_gamma_img * (1.0 - 0.8 * r)  # down to ~20% of initial
        w_img = w_img * min_snr_weight(σ_img_inj, self.sigma_data_img, gamma_img_now)

        # Image loss anneal for long training runs
        s_from, s_to, min_fac = self.img_loss_anneal
        prog = ((self.step_counter.item() - s_from) / max(1, (s_to - s_from)))
        img_fac = 1.0 if prog <= 0 else (1.0 - (1.0 - min_fac) * min(1.0, prog))

        mse_img = (w_img * (D_img - x_img).square()).mean()
        loss_img = img_fac * mse_img

        # Tab MSE with per-dimension weights (categoricals normalized per feature)
        tab_w = self.tab_dim_weights.view(1, -1).to(D_tab)
        loss_tab_raw = (w_tab * tab_w * (D_tab - x_tab_lat).square()).mean()

        # TabDiff regularizers
        reg = torch.tensor(0.0, device=device)
        ramp = (self.step_counter.float() / 2_000.0).clamp(max=1.0)

        if self.feature_wise and self.lambda_sigma > 0 and (self.step_counter < self.freeze_alpha_at):
            reg = reg + ramp * self.lambda_sigma * (self.tab_alpha.abs() - 1.0).pow(2).mean()

        # Class-balanced focal loss (binary)
        beta = 0.999
        gamma = 2.0
        n_neg, n_pos = self.n_neg, self.n_pos
        eff_num = torch.tensor([n_neg, n_pos], device=device, dtype=torch.float)
        eff_num = (1 - beta ** eff_num) / (1 - beta)
        cb_w = (1 - beta) / eff_num

        # Mask unconditional rows (label == NULL_ID)
        if labels is None:
            focal = torch.tensor(0.0, device=device)
        else:
            valid = labels < self.dit.NULL_ID
            if valid.any():
                lbl = labels[valid]
                logit = diag_logits[valid]
                alpha = cb_w[lbl]
                prob = torch.sigmoid(logit)
                focal = (
                    alpha
                    * (1 - prob).pow(gamma)
                    * F.binary_cross_entropy_with_logits(logit, lbl.float(), reduction="none")
                ).mean()
            else:
                focal = torch.tensor(0.0, device=device)

        # Lightweight diagnostics: MSE binned by σ (per-batch, no extra forwards)
        with torch.no_grad():
            mse_img_vec = (D_img - x_img).pow(2).flatten(1).mean(dim=1).float()
            mse_tab_vec = (D_tab - x_tab_lat).pow(2).mean(dim=1).float()
            sigma_img_1d = σ_img_inj.view(-1).float()
            sigma_tab_rms = σ_tab_inj.pow(2).mean(dim=1).sqrt().float()

            def _bins(sigma_1d):
                b_small = sigma_1d < 0.75
                b_mid = (sigma_1d >= 0.75) & (sigma_1d < 1.75)
                b_large = sigma_1d >= 1.75
                return b_small, b_mid, b_large

            bsi, bmi, bli = _bins(sigma_img_1d)
            bst, bmt, blt = _bins(sigma_tab_rms)

            def _safe_mean(x, m):
                return float(x[m].mean().item()) if m.any() else float("nan")

            self._last_img_mse_b_small = _safe_mean(mse_img_vec, bsi)
            self._last_img_mse_b_mid = _safe_mean(mse_img_vec, bmi)
            self._last_img_mse_b_large = _safe_mean(mse_img_vec, bli)
            self._last_tab_mse_b_small = _safe_mean(mse_tab_vec, bst)
            self._last_tab_mse_b_mid = _safe_mean(mse_tab_vec, bmt)
            self._last_tab_mse_b_large = _safe_mean(mse_tab_vec, blt)

        # Dependency alignment (Deep CORAL) computed in model space
        if self._tabsyn_use:
            x_tab_dec = self.tab_vae.decode_flat(D_tab)  # (B, sum k_i + n)
            n_cat = sum(self.ft.cat_dims) if getattr(self.ft, "cat_dims", None) else 0
            if x_tab_dec.shape[1] > n_cat:
                dep_loss = coral_loss(x_tab_dec[:, n_cat:], x_tab[:, n_cat:].detach())
            else:
                dep_loss = torch.tensor(0.0, device=device)
        else:
            dep_loss = coral_loss(D_tab, x_tab.detach())

        # Cross-modal InfoNCE
        img_e = out.get("img_emb", None)
        tab_e = out.get("tab_emb", None)
        if (img_e is not None) and (tab_e is not None):
            nce = info_nce(img_e, tab_e, temp=0.07)
        else:
            nce = torch.tensor(0.0, device=device)

        # Numeric variance/mean match (latent slice if TabSyn is enabled)
        n_cat = sum(self.ft.cat_dims)
        var_loss = torch.tensor(0.0, device=device)
        mean_loss = torch.tensor(0.0, device=device)
        if D_tab.shape[1] > n_cat:
            if self._tabsyn_use:
                start = self._n_cat_feats * self._tabsyn_d
                D_num = D_tab[:, start:]
                X_num = x_tab_lat[:, start:].detach()
            else:
                n_cat_oh = sum(self.ft.cat_dims) if getattr(self.ft, "cat_dims", None) else 0
                D_num = D_tab[:, n_cat_oh:]
                X_num = x_tab[:, n_cat_oh:].detach()
            var_loss = (D_num.var(dim=0, unbiased=False) - X_num.var(dim=0, unbiased=False)).abs().mean()
            mean_loss = (D_num.mean(dim=0) - X_num.mean(dim=0)).abs().mean()

        # Optional MMD (RBF) on decoded numerics (model space)
        mmd_loss = torch.tensor(0.0, device=device)
        if self.lambda_mmd_tab > 0.0:

            def _mmd_rbf(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    d = torch.cdist(A, B, p=2).pow(2)
                    med = torch.median(d)
                    if not torch.isfinite(med) or med <= 0:
                        med = torch.tensor(1.0, device=A.device, dtype=A.dtype)
                gamma = 1.0 / (2.0 * med)
                kxx = torch.exp(-gamma * torch.cdist(A, A, p=2).pow(2))
                kyy = torch.exp(-gamma * torch.cdist(B, B, p=2).pow(2))
                kxy = torch.exp(-gamma * torch.cdist(A, B, p=2).pow(2))
                return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()

            if self._tabsyn_use:
                dec_pred = self.tab_vae.decode_flat(D_tab).float()
                dec_real = x_tab.float()
            else:
                dec_pred = D_tab.float()
                dec_real = x_tab.float()

            if dec_pred.shape[1] > n_cat and dec_real.shape[1] > n_cat:
                A = dec_pred[:, n_cat:]
                B = dec_real[:, n_cat:].detach()
                keep = (A.std(0, unbiased=False) > 1e-6) & (B.std(0, unbiased=False) > 1e-6)
                if keep.any():
                    A, B = A[:, keep], B[:, keep]
                    mmd_loss = _mmd_rbf(A, B)

        # Categorical CE on decoded outputs + marginal KL (EMA target)
        cat_ce_loss = torch.tensor(0.0, device=device)
        cat_marg_kl = torch.tensor(0.0, device=device)
        n_cat_oh = sum(self.ft.cat_dims) if getattr(self.ft, "cat_dims", None) else 0
        if n_cat_oh > 0:
            if self._tabsyn_use:
                dec_pred = self.tab_vae.decode_flat(D_tab)
            else:
                dec_pred = D_tab

            off = 0
            ncols = 0
            x_cat_oh = x_tab[:, :n_cat_oh]
            for k in self.ft.cat_dims:
                logits = dec_pred[:, off:off + k]
                tgt = x_cat_oh[:, off:off + k].argmax(dim=1)
                cat_ce_loss = cat_ce_loss + F.cross_entropy(logits, tgt, reduction="mean")

                with torch.no_grad():
                    cur = x_cat_oh[:, off:off + k].float().mean(dim=0)
                    self._cat_marg_ema[off:off + k].mul_(self._sigma_ema).add_((1 - self._sigma_ema) * cur)

                p_hat = F.softmax(logits, dim=1).mean(dim=0).clamp(1e-6, 1 - 1e-6)
                p_tgt = self._cat_marg_ema[off:off + k]
                p_tgt = (p_tgt / (p_tgt.sum() + 1e-8)).clamp(1e-6, 1 - 1e-6)
                cat_marg_kl = cat_marg_kl + F.kl_div(p_hat.log(), p_tgt, reduction="sum")

                off += k
                ncols += 1

            if ncols > 0:
                cat_ce_loss = cat_ce_loss / float(ncols)
                cat_marg_kl = cat_marg_kl / float(ncols)

        # SWD on decoded numerics (model space)
        swd_loss = torch.tensor(0.0, device=device)
        if dec_pred.shape[1] > n_cat_oh and dec_real.shape[1] > n_cat_oh:
            A = dec_pred[:, n_cat_oh:].float()
            B = dec_real[:, n_cat_oh:].float()
            keep = (A.std(0, unbiased=False) > 1e-6) & (B.std(0, unbiased=False) > 1e-6)
            if keep.any():
                A, B = A[:, keep], B[:, keep]

                def _swd(a, b, n_proj=64):
                    d = a.size(1)
                    v = F.normalize(torch.randn(n_proj, d, device=a.device), dim=1)
                    ap = a @ v.t()
                    bp = b @ v.t()
                    ap, _ = ap.sort(dim=0)
                    bp, _ = bp.sort(dim=0)
                    return (ap - bp).pow(2).mean()

                swd_loss = _swd(A, B)

        # Skew/kurtosis matching on numerics (lightweight)
        sk_loss = torch.tensor(0.0, device=device)
        if dec_pred.shape[1] > n_cat_oh and dec_real.shape[1] > n_cat_oh:
            Ap = dec_pred[:, n_cat_oh:].float()
            Bp = dec_real[:, n_cat_oh:].float()
            keep = (Ap.std(0, unbiased=False) > 1e-6) & (Bp.std(0, unbiased=False) > 1e-6)
            if keep.any():
                Ap = Ap[:, keep]
                Bp = Bp[:, keep]

                def _sk_kurt(x, eps=1e-6):
                    m = x.mean(0, keepdim=True)
                    c2 = ((x - m) ** 2).mean(0) + eps
                    c3 = ((x - m) ** 3).mean(0)
                    c4 = ((x - m) ** 4).mean(0)
                    skew = c3 / (c2.sqrt() ** 3 + eps)
                    kurt = c4 / (c2 ** 2)
                    return skew, kurt

                sA, kA = _sk_kurt(Ap)
                sB, kB = _sk_kurt(Bp)
                sk_loss = (sA - sB).abs().mean() + 0.25 * (kA - kB).abs().mean()

        # Soft curriculum for tab-side updates (TTUR-like)
        def _lin_ramp(st, v0, v1, to):
            if to <= 0:
                return v1
            r = max(0.0, min(1.0, st / float(to)))
            return v0 + (v1 - v0) * r

        g_tab = _lin_ramp(step_i, self.tab_update_prob_start, self.tab_update_prob_end, self.tab_update_warmup_to)

        # Group all tab-side objectives (scaled by g_tab)
        tab_group = g_tab * (
            loss_tab_raw
            + self.lambda_corr * dep_loss
            + 0.5 * focal
            + self.lambda_nce * nce
            + self.lambda_var * var_loss
            + self.lambda_mean * mean_loss
            + self.lambda_mmd_tab * mmd_loss
            + self.lambda_cat_ce * cat_ce_loss
            + self.lambda_cat_marg * cat_marg_kl
            + self.lambda_swd * swd_loss
            + self.lambda_skewkurt * sk_loss
        )

        # GradNorm: choose anchor parameters
        if self.gradnorm_mode == "heads":
            img_params = [p for p in self.dit.final_img.parameters() if p.requires_grad]
            if hasattr(self.dit, "final_tab"):
                tab_params = [p for p in self.dit.final_tab.parameters() if p.requires_grad]
            elif hasattr(self.dit, "final_tab_latent"):
                tab_params = [p for p in self.dit.final_tab_latent.parameters() if p.requires_grad]
            else:
                tab_params = [p for p in self.dit.parameters() if p.requires_grad]
        else:
            img_params = [p for p in self.dit.parameters() if p.requires_grad]
            tab_params = img_params

        def _safe_norm(grads, device_):
            flats = [g.reshape(-1) for g in grads if g is not None]
            if not flats:
                return torch.tensor(0.0, device=device_)
            return torch.norm(torch.cat(flats), p=2)

        # Include λ_nce * nce in the image anchor (once) for consistent balancing
        grads_img = torch.autograd.grad(
            loss_img + self.lambda_nce * nce,
            img_params,
            retain_graph=True,
            allow_unused=True,
        )
        G_img = _safe_norm(grads_img, device).detach()

        grads_tab_base = torch.autograd.grad(tab_group, tab_params, retain_graph=True, allow_unused=True)
        G_tab_base = _safe_norm(grads_tab_base, device).detach()

        # Tab grad scaling factor (clamped exp(log_w_tab))
        w_tab_gn = torch.exp(
            self.log_w_tab.clamp(
                min=math.log(self.gradnorm_w_tab_min),
                max=math.log(self.gradnorm_w_tab_max),
            )
        )

        G_tab_scaled = w_tab_gn * G_tab_base

        # GradNorm penalty: encourage G_tab_scaled ≈ G_img
        grad_penalty = (G_img - G_tab_scaled).abs()

        # Slightly down-weight CORAL early (marginals first, deps later)
        ramp = (self.step_counter.float() / 2_000.0).clamp(max=1.0)
        coral_w = float(0.5 + 0.5 * ramp)

        # Define the coherent "core" and ensure g_tab applies consistently
        tab_core = (
            loss_tab_raw
            + coral_w * self.lambda_corr * dep_loss
            + 0.5 * focal
            + self.lambda_nce * nce
            + self.lambda_var * var_loss
            + self.lambda_mean * mean_loss
            + self.lambda_mmd_tab * mmd_loss
            + self.lambda_cat_marg * cat_marg_kl
        )
        tab_group = g_tab * tab_core

        total = loss_img + w_tab_gn * tab_group + reg + self.gradnorm_penalty * grad_penalty

        # Freeze alpha parameters at the configured iteration
        if self.feature_wise and self.step_counter == self.freeze_alpha_at:
            self.tab_alpha.requires_grad_(False)
        if self._tabsyn_use and (self._latent_alpha_mode != "none") and self.step_counter == self.freeze_alpha_at:
            self.tab_alpha_latent_cat.requires_grad_(False)
            self.tab_alpha_latent_num.requires_grad_(False)

        # Update EMAs for adaptive cross-grad gate
        with torch.no_grad():
            mse_img_now = (D_img - x_img).pow(2).mean()
            mse_tab_now = (D_tab - x_tab_lat).pow(2).mean()
            self._ema_img_mse.mul_(self._ema_mse_beta).add_((1 - self._ema_mse_beta) * mse_img_now.detach())
            self._ema_tab_mse.mul_(self._ema_mse_beta).add_((1 - self._ema_mse_beta) * mse_tab_now.detach())

        return total, loss_img.detach(), loss_tab_raw.detach()

    # ---------------------------------------------------------------------
    # Sampling (EDM-Karras / Heun)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
            self,
            batch_size: int,
            *,
            labels: Optional[torch.Tensor] = None,
            guidance_scale: float | dict | tuple = 1.0,
            num_steps: int = 40,
            seed: Optional[int] = None,
            return_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EDM-Karras sampling with separate image/tab schedules matching training.
        Feature-wise α is applied to the tab schedule (as in training).

        The implementation is intentionally identical to the original behavior.
        """

        device = next(self.dit.parameters()).device
        mp_dtype = next(self.dit.parameters()).dtype
        sched_dtype = torch.float32
        g = torch.Generator(device=device).manual_seed(seed or torch.seed())

        def _karras_sigmas(sigma_min: float, sigma_max: float, rho: int, N: int, dtype):
            t = torch.arange(N, device=device, dtype=sched_dtype)
            s = (sigma_max ** (1 / rho)
                 + t / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            return torch.cat([s.to(dtype), s.new_zeros(1)])

        if hasattr(self, "sigma_data_tab_vec"):
            sigma_data_tab_vec = self.sigma_data_tab_vec.to(device=device, dtype=torch.float32)
        else:
            scalar = float(getattr(self, "sigma_data_tab", 1.0))
            sigma_data_tab_vec = torch.full((self.tab_outdim,), scalar, device=device, dtype=torch.float32)

        N = int(num_steps)
        sigmas_img = _karras_sigmas(self.edm_img.sigma_min, self.edm_img.sigma_max, self.edm_img.rho, N, torch.float32)
        sigmas_tab = _karras_sigmas(self.edm_tab.sigma_min, self.edm_tab.sigma_max, self.edm_tab.rho, N, torch.float32)

        def choose_best_noise(shape):
            n1 = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
            n2 = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
            sel = (n2.square().mean(dim=tuple(range(1, n2.ndim))) >
                   n1.square().mean(dim=tuple(range(1, n1.ndim))))
            return torch.where(sel.view(-1, *([1] * (len(shape) - 1))), n2, n1)

        if getattr(self, "noise_select", False):
            x_img = choose_best_noise((batch_size, 4, 32, 32))
            x_tab = choose_best_noise((batch_size, self.tab_outdim))
        else:
            x_img = torch.randn(batch_size, 4, 32, 32, generator=g, device=device, dtype=torch.float32)
            x_tab = torch.randn(batch_size, self.tab_outdim, generator=g, device=device, dtype=torch.float32)

        if self.feature_wise:
            alpha = self.tab_alpha.abs().clamp(1.0, 2.0).to(device=device, dtype=mp_dtype)
        else:
            alpha = torch.ones(self.tab_outdim, device=device, dtype=mp_dtype)

        σ0_img = sigmas_img[0]
        σ0_tab_scalar = sigmas_tab[0].clamp(max=float(self.tab_sigma_max))
        σ0_tab_vec = (σ0_tab_scalar * alpha).view(1, -1).expand(batch_size, -1)

        x_img = x_img * σ0_img
        x_tab = x_tab * σ0_tab_vec

        def _is_main():
            import torch.distributed as dist_
            return (not dist_.is_available()) or (not dist_.is_initialized()) or (dist_.get_rank() == 0)

        def _rms_sigma(vec: torch.Tensor) -> torch.Tensor:
            if vec.ndim == 1:
                return vec.pow(2).mean().sqrt()
            return vec.double().pow(2).mean(dim=1).sqrt().float()

        def _denoise(xi_img, xi_tab, sigma_scalar, sigma_tab_vec, labs):
            B = xi_img.shape[0]
            t_img = (sigma_scalar.float().log().div(4)).repeat(B)
            t_tab = (_rms_sigma(sigma_tab_vec).log().div(4))

            den_img = (self.sigma_data_img.float() ** 2 + sigma_scalar.float() ** 2).sqrt().view(1, 1, 1, 1)
            den_tab = (sigma_data_tab_vec.view(1, -1).float() ** 2 + sigma_tab_vec.float() ** 2).sqrt()

            out = self.dit(
                x_img=(xi_img / den_img).to(dtype=mp_dtype),
                x_tab=(xi_tab / den_tab).to(dtype=mp_dtype),
                t_img=t_img.to(dtype=mp_dtype),
                t_tab=t_tab.to(dtype=mp_dtype),
                labels=labs,
                mask_ratio=0.0,
                self_cond_img=None,
                self_cond_tab=None,
            )
            F_img, F_tab = out["image_sample"].float(), out["tab_sample"].float()
            D_img = self._to_D(xi_img, sigma_scalar.view(1, 1, 1, 1).float(), F_img, self.sigma_data_img.float())
            D_tab = self._to_D(xi_tab, sigma_tab_vec.float(), F_tab, sigma_data_tab_vec.view(1, -1).float())
            return D_img, D_tab

        def _split_scales(gs):
            if isinstance(gs, dict):
                return float(gs.get("img", 1.0)), float(gs.get("tab", 1.0))
            if isinstance(gs, (tuple, list)) and len(gs) >= 2:
                return float(gs[0]), float(gs[1])
            return float(gs), float(gs)

        def _denoise_cfg(xi_img, xi_tab, sigma_scalar, sigma_tab_vec):
            s_img_max, s_tab_max = _split_scales(guidance_scale)

            rms_tab = sigma_tab_vec.double().pow(2).mean(dim=1).sqrt().float().view(-1, 1)
            tab_decay = (rms_tab / (self.edm_tab.sigma_max + 1e-8)).clamp(min=0.0, max=1.0)
            s_tab = 1.0 + (s_tab_max - 1.0) * (1.0 - tab_decay)

            ratio = (sigma_scalar / (self.edm_img.sigma_max + 1e-8)).clamp(min=0.0, max=1.0)
            s_img = 1.0 + (s_img_max - 1.0) * (1.0 - ratio) ** 2

            if labels is None:
                labs = torch.full((batch_size,), self.dit.NULL_ID, device=device, dtype=torch.long)
                return _denoise(xi_img, xi_tab, sigma_scalar, sigma_tab_vec, labs)

            if (s_img == 1.0) and torch.allclose(s_tab, torch.ones_like(s_tab)):
                return _denoise(xi_img, xi_tab, sigma_scalar, sigma_tab_vec, labels.to(device))

            labs_u = torch.full((batch_size,), self.dit.NULL_ID, device=device, dtype=torch.long)
            labs_c = labels.to(device=device)
            Du_img, Du_tab = _denoise(xi_img, xi_tab, sigma_scalar, sigma_tab_vec, labs_u)
            Dc_img, Dc_tab = _denoise(xi_img, xi_tab, sigma_scalar, sigma_tab_vec, labs_c)
            return (Du_img + s_img * (Dc_img - Du_img), Du_tab + s_tab * (Dc_tab - Du_tab))

        if _is_main():
            snr_img0 = (self.sigma_data_img.to(mp_dtype) ** 2 / (σ0_img ** 2 + 1e-12)).item()
            snr_tab0 = (sigma_data_tab_vec ** 2 / (σ0_tab_vec ** 2 + 1e-12)).mean().item()
            σ0_tab_rms = _rms_sigma(σ0_tab_vec).mean()
            print(
                f"[Sampler] σ0_img={σ0_img.item():.5f} | σ0_tab_rms≈{σ0_tab_rms.item():.5f} "
                f"| SNR_img0={snr_img0:.5f} | SNR_tab0≈{snr_tab0:.5f}"
            )

        for i in range(N):
            σ_i_img = sigmas_img[i]
            σ_ip1_img = sigmas_img[i + 1]
            σ_i_tab_scalar = sigmas_tab[i]
            σ_ip1_tab_scalar = sigmas_tab[i + 1]

            σ_i_tab = (σ_i_tab_scalar.clamp(max=float(self.tab_sigma_max)) * alpha).view(1, -1).expand(batch_size, -1)

            gamma_img = 0.0
            if (self.edm_img.S_churn > 0.0) and (self.edm_img.S_min <= float(σ_i_img) <= self.edm_img.S_max):
                gamma_img = min(self.edm_img.S_churn / N, math.sqrt(2.0) - 1.0)

            gamma_tab = 0.0
            if (self.edm_tab.S_churn > 0.0) and (self.edm_tab.S_min <= float(σ_i_tab_scalar) <= self.edm_tab.S_max):
                gamma_tab = min(self.edm_tab.S_churn / N, math.sqrt(2.0) - 1.0)

            σ_hat_img = σ_i_img * (1.0 + gamma_img)
            σ_hat_tab_scalar = σ_i_tab_scalar * (1.0 + gamma_tab)
            σ_hat_tab = (σ_hat_tab_scalar.clamp(max=float(self.tab_sigma_max)) * alpha).view(1, -1).expand(batch_size, -1)

            if (gamma_img > 0.0) or (gamma_tab > 0.0):
                d_img = (σ_hat_img ** 2 - σ_i_img ** 2).clamp(min=0).sqrt()
                d_tab = (σ_hat_tab ** 2 - σ_i_tab ** 2).clamp(min=0).sqrt()
                x_img = x_img + torch.randn_like(x_img) * d_img
                x_tab = x_tab + torch.randn_like(x_tab) * d_tab

            D_img, D_tab = _denoise_cfg(x_img, x_tab, σ_hat_img, σ_hat_tab)
            d_img = (x_img - D_img) / σ_hat_img
            d_tab = (x_tab - D_tab) / σ_hat_tab

            σ_ip1_tab = (σ_ip1_tab_scalar.clamp(max=float(self.tab_sigma_max)) * alpha).view(1, -1).expand(batch_size, -1)
            x_img_e = x_img + (σ_ip1_img - σ_hat_img) * d_img
            x_tab_e = x_tab + (σ_ip1_tab - σ_hat_tab) * d_tab

            if i < N - 1:
                D_img2, D_tab2 = _denoise_cfg(x_img_e, x_tab_e, σ_ip1_img, σ_ip1_tab)
                d_img2 = (x_img_e - D_img2) / σ_ip1_img
                d_tab2 = (x_tab_e - D_tab2) / σ_ip1_tab
                x_img = x_img + 0.5 * (σ_ip1_img - σ_hat_img) * (d_img + d_img2)
                x_tab = x_tab + 0.5 * (σ_ip1_tab - σ_hat_tab) * (d_tab + d_tab2)
            else:
                x_img, x_tab = x_img_e, x_tab_e

        # Optional low-σ refinement on the tab branch (image remains unchanged)
        if self.tab_refine_steps > 0:
            σ_small = torch.tensor(max(float(self.edm_tab.sigma_min) * 0.8, 0.03), device=device, dtype=mp_dtype)
            for _ in range(int(self.tab_refine_steps)):
                σ_tab_vec = (σ_small.clamp(max=float(self.tab_sigma_max)) * alpha).view(1, -1).expand(batch_size, -1)
                σ_img_small = max(float(self.edm_img.sigma_min) * 0.8, 0.02)
                D_img_ref, D_tab_ref = _denoise_cfg(
                    x_img,
                    x_tab,
                    torch.tensor(σ_img_small, device=device, dtype=mp_dtype),
                    σ_tab_vec,
                )
                x_tab = x_tab + self.tab_refine_mix * (D_tab_ref - x_tab)

        # Diagnostics (rank-0 only)
        if _is_main():
            try:
                if getattr(self, "_tabsyn_use", False):
                    x_tab_diag_model = self.tab_vae.decode_flat(x_tab).float()
                else:
                    x_tab_diag_model = x_tab.float()

                n_cat = sum(self.ft.cat_dims) if getattr(self.ft, "cat_dims", None) else 0
                if x_tab_diag_model.shape[1] > n_cat:
                    z_num = x_tab[:, n_cat:]
                    frac_abs_gt4 = (z_num.abs() > 4.0).float().mean().item()
                else:
                    frac_abs_gt4 = 0.0

                raw_np = inverse_transform(self.ft, x_tab_diag_model.cpu().numpy())

                near_min, near_max, tot = 0, 0, 0
                mins, maxs, names = [], [], []
                for name in self.ft.num_features:
                    spec = self.ft.num_specs[name]
                    if spec.kind == "logit" and spec.min is not None and spec.max is not None:
                        mins.append(spec.min)
                        maxs.append(spec.max)
                        names.append(name)

                if names:
                    mins = torch.tensor(mins, dtype=torch.float32)
                    maxs = torch.tensor(maxs, dtype=torch.float32)
                    vals = torch.from_numpy(raw_np[:, [self.ft.feature_list.index(n) for n in names]].astype("float32"))
                    rng = (maxs - mins).clamp(min=1e-8)
                    lo = mins + 0.01 * rng
                    hi = maxs - 0.01 * rng
                    near_min = (vals < lo).float().mean().item()
                    near_max = (vals > hi).float().mean().item()
                    tot = len(names)

                print(
                    f"[Sampler] end-step: frac(|z_num|>4)={frac_abs_gt4:.3f} | "
                    f"bounded-cols={tot} | near-min≈{near_min:.3f} | near-max≈{near_max:.3f}"
                )
            except Exception as e:
                print(f"[Sampler] diagnostics failed (non-fatal): {e}")

        if return_latents:
            return x_img, x_tab

        # Optional variance calibration on numeric latent dims (TabSyn only)
        if self.calibrate_tab_var and getattr(self, "_tabsyn_use", False):
            start = getattr(self, "_n_cat_feats", 0) * getattr(self, "_tabsyn_d", 1)
            if start < x_tab.size(1):
                num = x_tab[:, start:]
                m = num.mean(0, keepdim=True)
                v = num.var(0, unbiased=False, keepdim=True)
                tgt_v = (self.sigma_data_tab_vec[start:].to(num) ** 2).view(1, -1)
                s = (tgt_v / (v + 1e-8)).sqrt().clamp(
                    1.0 - self.calibrate_tab_var_clip,
                    1.0 + self.calibrate_tab_var_clip,
                )
                x_tab[:, start:] = m + (num - m) * s

        # Optional mean calibration on numeric latent dims (TabSyn only)
        if self.calibrate_tab_mean and getattr(self, "_tabsyn_use", False):
            start = getattr(self, "_n_cat_feats", 0) * getattr(self, "_tabsyn_d", 1)
            if start < x_tab.size(1):
                num = x_tab[:, start:]
                m = num.mean(0, keepdim=True)
                tgt_m = self.mu_data_tab_vec[start:].to(num).view(1, -1)
                delta = (tgt_m - m).clamp(-self.calibrate_tab_mean_clip, self.calibrate_tab_mean_clip)
                x_tab[:, start:] = num + delta

        # Decode latent to model-space and then inverse transform
        if self._tabsyn_use:
            x_tab_model = self.tab_vae.decode_flat(x_tab).float()
        else:
            x_tab_model = x_tab.float()

        # Clamp QT numerics before inverse transform (existing behavior)
        n_cat = sum(self.ft.cat_dims) if getattr(self.ft, "cat_dims", None) else 0
        if x_tab_model.shape[1] > n_cat:
            qt_idx = [i for i, col in enumerate(self.ft.num_features) if self.ft.num_specs[col].kind == "qt"]
            for j in qt_idx:
                x_tab_model[:, n_cat + j] = x_tab_model[:, n_cat + j].clamp_(-6.0, 6.0)

        tab_np = inverse_transform(self.ft, x_tab_model.cpu().numpy())
        return x_img, torch.from_numpy(tab_np.astype(np.float32)).to(device)

    @property
    def class_counts(self) -> Tuple[int, int]:
        """Returns (n_negatives, n_positives) as observed during training."""
        return (self.n_neg, self.n_pos)

    # ------------------- internal helpers --------------------------- #
    def _to_D(self, x_noisy: torch.Tensor, σ: torch.Tensor, F_x: torch.Tensor, sigma_data: torch.Tensor):
        c_skip = sigma_data ** 2 / (σ ** 2 + sigma_data ** 2)
        c_out = σ * sigma_data / (σ ** 2 + sigma_data ** 2).sqrt()
        return c_skip * x_noisy + c_out * F_x


from utils.configurations import _merge_cfg
def load_diffusion(cfg: DictConfig, dit_model, **overrides):
    """
    Instantiate `MultiModalDiffusion` using an already-built `dit_model`.

    Keyword overrides are merged into `cfg` consistently with the other loaders.
    """
    final_cfg = _merge_cfg(cfg, overrides)
    diffusion_model = MultiModalDiffusion(dit=dit_model, **final_cfg)

    if "noise_select" in cfg and cfg.noise_select:
        diffusion_model.noise_select = True
    return diffusion_model
