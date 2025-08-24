from omegaconf import DictConfig
from omegaconf import OmegaConf
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple
from easydict import EasyDict
from data.tabular_transforms import FittedTransforms, inverse_transform
import math
from models.dit.dit_multimodal import FeedForwardECMoe
import torch.distributed as dist


def _log_sigma_to_t(sigma: torch.Tensor) -> torch.Tensor:
    """EDM time‑embedding (log σ)/4, preserving shape (B,) or scalar."""
    return sigma.log().div(4).float()

def _flatten_grads(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list if g is not None])

# -----------------------------------------------------------------------------#
# Main class
# -----------------------------------------------------------------------------#
class MultiModalDiffusion(nn.Module):
    """
    EDM‑Karras multimodal diffuser with
    * TabDiff feature‑wise σ (optional)
    * GradNorm dynamic loss scaling
    * Dual timestamp support in the DiT backbone
    """

    def __init__(
            self,
            *,
            dit: nn.Module,
            tab_transforms: FittedTransforms,
            num_tab_features: int,
            # EDM hyper‑params
            sigma_min: float = 0.002,
            sigma_max: float = 50.0,
            num_steps: int = 40,
            rho: int = 7,
            P_mean: float = 0.0,
            P_std: float = 1.0,
            S_churn: float = 0.0,
            S_min: float = 0.0,
            S_max: float = float("inf"),
            S_noise: float = 1.0,
            # TabDiff
            tab_sigma_max: float = 0.93,
            feature_wise_sigma: bool = True,
            lambda_sigma: float = 1e-5,
            lambda_out: float = 1e-5,
            freeze_alpha_at: int = 20_000,
            # misc
            class_counts: Tuple[int, int] = (1085, 307),  # (negatives, positives)
            dtype: str = "bfloat16",
            noise_select = True
    ):
        super().__init__()
        self.dit = dit

        self.dtype = dtype

        # ---------------- tabular dims & transforms ------------------ #
        self.ft = tab_transforms
        cat_dim = (
            sum(len(c) for c in tab_transforms.cat_encoder.categories_)
            if tab_transforms.cat_features
            else 0
        )
        self.tab_outdim = len(tab_transforms.num_features) + cat_dim
        self.num_tab_features = num_tab_features  # raw count (before one‑hot)

        # ---------------- EDM tracker -------------------------------- #
        self.edm = EasyDict(
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

        # ---------------- TabDiff setup ------------------------------ #
        self.tab_sigma_max = float(tab_sigma_max)
        self.feature_wise = bool(feature_wise_sigma)
        self.lambda_sigma = lambda_sigma * num_tab_features / 250.0
        self.lambda_out = lambda_out * num_tab_features / 250.0
        self.freeze_alpha_at = freeze_alpha_at

        if self.feature_wise:
            # will be overwritten on the first forward pass
            self.tab_alpha = nn.Parameter(torch.full((self.tab_outdim,), 0.8))

        # ---------------- GradNorm variables ------------------------- #
        # single learnable log‑weight for the tab branch
        self.log_w_tab = nn.Parameter(torch.zeros(()))

        # iteration counter (for ramp‑up, freeze, etc.)
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))

        self.n_neg, self.n_pos = class_counts

        # --- broadcast-safety flags ------------------------------------------- #
        self._alpha_init_done = False  # will be set after broadcast

        # ── wire global step counter to each MoE mlp ────────────────────────────
        for m in self.dit.modules():
            if isinstance(m, FeedForwardECMoe):
                m.step_counter = self.step_counter

    # -----------------------------------------------------------------
    # Forward – training
    # -----------------------------------------------------------------
    def forward(
            self,
            x_img: torch.Tensor,  # (B,4,32,32) latent‑space image
            x_tab: torch.Tensor,  # (B,F)        standardised table
            *,
            mask_ratio: float = 0.0,
            labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ------- 30 % label-dropout for CFG --------------------------------
        if labels is not None:
            drop = torch.rand_like(labels, dtype=torch.float) < 0.30
            labels = labels.clone()
            labels[drop] = self.dit.NULL_ID

        B, device = x_img.size(0), x_img.device
        self.step_counter += 1

        # (1) draw shared scalar σ  ~ LogNormal
        sigma_scalar = (torch.randn(B, 1, device=device) * self.edm.P_std + self.edm.P_mean).exp()  # (B,1)

        # (2) branch‑specific injected σ
        σ_img_inj = sigma_scalar.view(B, 1, 1, 1)  # (B,1,1,1)
        σ_tab_base = sigma_scalar.clamp(max=self.tab_sigma_max)  # (B,1)

        # ---------- ONE-TIME, ALL-RANK initialisation & broadcast -------------- #
        if self.feature_wise and not self._alpha_init_done:
            with torch.no_grad():
                # compute per-feature std-ratio on *this* mini-batch
                std_ratio = x_tab.std(0) / (x_tab.std() + 1e-8)
                std_ratio = std_ratio.clamp(min=1e-3)
                self.tab_alpha.data.copy_(std_ratio)

                # ---- synchronise so every rank sees identical weights -------- #

                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(self.tab_alpha.data, src=0)
            self._alpha_init_done = True
            σ_tab_inj = (σ_tab_base * self.tab_alpha.abs() if self.feature_wise else σ_tab_base)
        else:
            σ_tab_inj = σ_tab_base  # (B,1)


        # (3) add Gaussian noise
        x_img_noisy = x_img + torch.randn_like(x_img) * σ_img_inj
        x_tab_noisy = x_tab + torch.randn_like(x_tab) * σ_tab_inj

        # (4) EDM time‑embeddings (no clamping here)
        t_img = _log_sigma_to_t(sigma_scalar.squeeze(1))  # (B,)
        t_tab = t_img.clone()

        # SELF-CONDITIONING (50 % of mini-batches)                   #
        if torch.rand((), device=device) < 0.5:
            with torch.no_grad():  # stop-gradient teacher pass
                sc_out = self.dit(  # first (teacher) call
                    x_img = x_img_noisy * (1.0 / (self.edm.sigma_data ** 2 + σ_img_inj ** 2).sqrt()),
                    x_tab = x_tab_noisy * (1.0 / (self.edm.sigma_data ** 2 + σ_tab_inj ** 2).sqrt()),
                    t_img = t_img,
                    t_tab = t_tab,
                    labels = labels,
                    mask_ratio = mask_ratio,
                    self_cond_img = None,
                    self_cond_tab = None,
                    cfg = 1.0  # always unconditional here
                )
            self_cond_img = sc_out["image_sample"].detach()
            self_cond_tab = sc_out["tab_sample"].detach()
        else:
            self_cond_img = None
            self_cond_tab = None

        # (5) backbone
        out = self.dit(
            x_img=x_img_noisy * (1.0 / (self.edm.sigma_data ** 2 + σ_img_inj ** 2).sqrt()),
            x_tab=x_tab_noisy * (1.0 / (self.edm.sigma_data ** 2 + σ_tab_inj ** 2).sqrt()),
            t_img=t_img,
            t_tab=t_tab,
            labels=labels,
            mask_ratio=mask_ratio,
            self_cond_img = self_cond_img,
            self_cond_tab = self_cond_tab,
        )
        F_img, F_tab, diag_logits = out["image_sample"], out["tab_sample"], out["diag_logits"]

        # ----- (6) ε‑to‑D(x) mapping -------------------------------- #
        D_img = self._to_D(x_img_noisy, σ_img_inj, F_img)
        D_tab = self._to_D(x_tab_noisy, σ_tab_inj, F_tab)

        # ----- (7) weighted MSE losses ------------------------------ #
        w_img = ((σ_img_inj ** 2 + self.edm.sigma_data ** 2) / (σ_img_inj * self.edm.sigma_data) ** 2)
        w_tab = ((σ_tab_inj ** 2 + self.edm.sigma_data ** 2) / (σ_tab_inj * self.edm.sigma_data) ** 2)

        loss_img = (w_img * (D_img - x_img).square()).mean()
        loss_tab_raw = (w_tab * (D_tab - x_tab).square()).mean()

        # ----- (8) TabDiff regularisers ----------------------------- #
        reg = torch.tensor(0.0, device=device)

        # ramp‑up schedule for λ’s (first 2k steps)
        ramp = (self.step_counter.float() / 2_000.0).clamp(max=1.0) # linear 0→1 over 2k iters

        if self.feature_wise and self.lambda_sigma > 0 and (self.step_counter < self.freeze_alpha_at):
            reg = reg + ramp * self.lambda_sigma * (self.tab_alpha.abs() - 1.0).pow(2).mean()

        if self.lambda_out > 0:
            n_num = len(self.ft.num_features)
            if n_num:
                num_pred = D_tab[..., -n_num:]
                mins = torch.tensor([self.ft.num_specs[c].min for c in self.ft.num_features], device=device).view(1, n_num)
                maxs = torch.tensor([self.ft.num_specs[c].max for c in self.ft.num_features], device=device).view(1, n_num)
                # overflow = torch.clamp(num_pred - maxs, min=0.)
                # underflow = torch.clamp(mins - num_pred, min=0.)
                # reg = reg + ramp * self.lambda_out * (overflow + underflow).mean()
                v = num_pred
                reg_range = ((torch.relu(v - maxs)) ** 2 + (torch.relu(mins - v)) ** 2).mean()
                reg = reg + ramp * self.lambda_out * reg_range

        # ================== Class-Balanced Focal Loss =====================
        beta = 0.999
        gamma = 2.0
        n_neg, n_pos = self.n_neg, self.n_pos  # from class_counts
        eff_num = torch.tensor([n_neg, n_pos], device=device, dtype=torch.float)
        eff_num = (1 - beta ** eff_num) / (1 - beta)
        cb_w = (1 - beta) / eff_num  # length-2 tensor

        # ----- mask-out unconditional rows (label == NULL_ID = 2) --------
        valid = labels < self.dit.NULL_ID  # keep 0 & 1 only
        if valid.any():
            lbl = labels[valid]
            logit = diag_logits[valid]

            alpha = cb_w[lbl]  # safe index
            prob = torch.sigmoid(logit)
            focal = (alpha * (1 - prob).pow(gamma) *
                     F.binary_cross_entropy_with_logits(
                         logit, lbl.float(),
                         reduction='none')
                     ).mean()
        else:
            focal = torch.tensor(0., device=device)


        # ----- (9) GradNorm weighting ------------------------------- #
        loss_tab = torch.exp(self.log_w_tab) * loss_tab_raw
        total = loss_img + loss_tab + focal + reg

        # --- compute GradNorm penalty (α = 1.5) every step --------- #
        grads_img = torch.autograd.grad(loss_img, list(self.dit.parameters()), retain_graph=True, allow_unused=True)
        grads_tab = torch.autograd.grad(loss_tab, list(self.dit.parameters()), retain_graph=True, allow_unused=True)
        G_img = _flatten_grads(grads_img).norm()
        G_tab = _flatten_grads(grads_tab).norm()
        grad_penalty = (G_img - G_tab.detach()).abs() ** 0.5
        total = total + 0.01 * grad_penalty

        # --- optionally freeze tab_alpha after warm‑up ------------- #
        if self.feature_wise and self.step_counter == self.freeze_alpha_at:
            self.tab_alpha.requires_grad_(False)

        return total, loss_img.detach(), loss_tab_raw.detach()

    # ---------------------------------------------------------------------
    # Sampling (EDM tracker identical to training, but in a loop)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
            self,
            batch_size: int,
            *,
            labels: Optional[torch.Tensor] = None,
            guidance_scale: float = 1.0,
            num_steps: int = 40,
            seed: Optional[int] = None,
            return_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EDM Karras sampling with clamped tab noise injection.
        Returns: image_latent (B,4,32,32), tab_tensor (B,F)
        """

        device = next(self.dit.parameters()).device
        g = torch.Generator(device=device).manual_seed(seed or torch.seed())

        if guidance_scale > 1.0:
            labels_cat = (labels if labels is not None else torch.zeros(batch_size, dtype=torch.long, device=device))
        else:
            labels_cat = labels

        # -------- σ schedule (Karras) --------------------------- #
        t_arr = torch.arange(num_steps, device=device, dtype=torch.float64)
        sigmas = (self.edm.sigma_max ** (1 / self.edm.rho)
                  + t_arr / (num_steps - 1)
                  * (self.edm.sigma_min ** (1 / self.edm.rho) - self.edm.sigma_max ** (
                            1 / self.edm.rho))) ** self.edm.rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # final 0
        N = num_steps

        # -------- initial noise --------------------------------- #
        def choose_best_noise(shape):
            n1, n2 = torch.randn(shape, generator=g, device=device), torch.randn(shape, generator=g, device=device)
            return torch.where((n2.square().mean(dim=tuple(range(1, n2.ndim))) > n1.square().mean(dim=tuple(range(1, n1.ndim)))).view(-1, *([1] * (len(shape) - 1))), n2, n1)

        if getattr(self, "noise_select", False):
            x_img = choose_best_noise((batch_size, 4, 32, 32)).double()
            x_tab = choose_best_noise((batch_size, self.tab_outdim)).double()
        else:
            x_img = torch.randn(batch_size, 4, 32, 32, generator=g, device=device).double()
            x_tab = torch.randn(batch_size, self.tab_outdim, generator=g, device=device).double()

        # -------- branch‑specific σ0 ----------------------------- #
        σ0 = sigmas[0]
        σ0_tab = torch.clamp(σ0, max=self.tab_sigma_max)
        if self.feature_wise:
            σ0_tab = σ0_tab * self.tab_alpha.abs()
        σ0_tab = σ0_tab.unsqueeze(0).expand(batch_size, -1)

        x_img *= σ0
        x_tab *= σ0_tab

        # -------- cfg wrapper ----------------------------------- #
        cfg_fwd = (partial(self.dit.forward, cfg=guidance_scale) if guidance_scale > 1.0 else self.dit.forward)

        # -------- main loop ------------------------------------- #
        prev_img, prev_tab = None, None
        for i in range(N):
            σ_i = sigmas[i].double()
            σ_ip1 = sigmas[i + 1].double()

            # σ_i_tab_inj = torch.clamp(σ_i, max=self.tab_sigma_max)

            # σ for current step
            σ_i_tab = torch.clamp(σ_i, max=self.tab_sigma_max)
            if self.feature_wise:
                σ_i_tab = σ_i_tab * self.tab_alpha.abs()
            σ_i_tab = σ_i_tab.unsqueeze(0).expand(batch_size, -1)

            # σ‑churn (Karras §C.2)
            γ = (min(self.edm.S_churn / N, math.sqrt(2) - 1.0) if self.edm.S_min <= σ_i <= self.edm.S_max else 0.0)
            σ_hat = σ_i + γ * σ_i  # always ≥ σ_i

            σ_hat_tab = torch.clamp(σ_hat, max=self.tab_sigma_max)
            if self.feature_wise:
                σ_hat_tab = σ_hat_tab * self.tab_alpha.abs()
            σ_hat_tab = σ_hat_tab.unsqueeze(0).expand(batch_size, -1)  # (B,F) or (B,1)

            # noise perturbation
            noise_img = torch.randn_like(x_img)
            noise_tab = torch.randn_like(x_tab)
            Δ_img = (σ_hat ** 2 - σ_i ** 2).sqrt()
            Δ_tab = (σ_hat_tab ** 2 - σ_i_tab ** 2).clamp(min=0.0).sqrt()

            x_img_hat = x_img + self.edm.S_noise * noise_img * Δ_img
            x_tab_hat = x_tab + self.edm.S_noise * noise_tab * Δ_tab

            # dual timesteps
            t_img = _log_sigma_to_t(σ_hat).repeat(batch_size)
            t_tab = t_img.clone()

            # --------------- balanced guidance -----------------
            if guidance_scale > 1.0 and self.edm is not None:
                scale_step = 1.0 + (guidance_scale - 1.0) * i / (N - 1)
            else:
                scale_step = guidance_scale
            # predict ε
            out = cfg_fwd(
                x_img=x_img_hat.float() * (1.0 / (self.edm.sigma_data ** 2 + σ_hat ** 2).sqrt()),
                x_tab=x_tab_hat.float() * (1.0 / (self.edm.sigma_data ** 2 + σ_hat_tab ** 2).sqrt()),
                t_img=t_img,
                t_tab=t_tab,
                labels=labels_cat,
                mask_ratio=0.0,
                self_cond_img = prev_img,
                self_cond_tab = prev_tab,
                cfg=scale_step
            )
            prev_img, prev_tab = out["image_sample"].detach(), out["tab_sample"].detach()

            D_img = self._to_D(x_img_hat, σ_hat, out["image_sample"])
            D_tab = self._to_D(x_tab_hat, σ_hat_tab, out["tab_sample"])

            d_img = (x_img_hat - D_img) / σ_hat
            d_tab = (x_tab_hat - D_tab) / σ_hat_tab

            # Euler step
            σ_ip1_tab = torch.clamp(σ_ip1, max=self.tab_sigma_max)
            if self.feature_wise:
                σ_ip1_tab = σ_ip1_tab * self.tab_alpha.abs()
            σ_ip1_tab = σ_ip1_tab.unsqueeze(0).expand(batch_size, -1)

            x_img_next = x_img_hat + (σ_ip1 - σ_hat) * d_img
            x_tab_next = x_tab_hat + (σ_ip1_tab - σ_hat_tab) * d_tab

            # 2nd order corrector (disabled at final step)
            if i < N - 1:
                t_img_2 = _log_sigma_to_t(σ_ip1).repeat(batch_size)
                t_tab_2 = t_img_2.clone()

                out2 = cfg_fwd(
                    x_img=x_img_next.float() * (1.0 / (self.edm.sigma_data ** 2 + σ_ip1 ** 2).sqrt()),
                    x_tab=x_tab_next.float() * (1.0 / (self.edm.sigma_data ** 2 + σ_ip1_tab ** 2).sqrt()),
                    t_img=t_img_2,
                    t_tab=t_tab_2,
                    labels=labels_cat,
                    mask_ratio=0.0,
                )
                D_img_2 = self._to_D(x_img_next, σ_ip1, out2["image_sample"])
                D_tab_2 = self._to_D(x_tab_next, σ_ip1_tab, out2["tab_sample"])

                d_img_2 = (x_img_next - D_img_2) / σ_ip1
                d_tab_2 = (x_tab_next - D_tab_2) / σ_ip1_tab

                x_img_next = x_img_hat + 0.5 * (σ_ip1 - σ_hat) * (d_img + d_img_2)
                x_tab_next = x_tab_hat + 0.5 * (σ_ip1_tab - σ_hat_tab) * (d_tab + d_tab_2)

            x_img, x_tab = x_img_next, x_tab_next

        # back to fp32
        x_img, x_tab = x_img.float(), x_tab.float()

        if return_latents:
            return x_img, x_tab

        # inverse transform the table branch
        tab_np = inverse_transform(self.ft, x_tab.cpu().numpy())
        return x_img, torch.from_numpy(tab_np.astype(np.float32)).to(device)

    # ------------------- helpers ----------------------------------------- #
    def _to_D(self, x_noisy: torch.Tensor, σ: torch.Tensor, F_x: torch.Tensor):
        c_skip = self.edm.sigma_data ** 2 / (σ ** 2 + self.edm.sigma_data ** 2)
        c_out = σ * self.edm.sigma_data / (σ ** 2 + self.edm.sigma_data ** 2).sqrt()
        return c_skip * x_noisy + c_out * F_x


from utils.configurations import _merge_cfg
def load_diffusion(cfg: DictConfig, dit_model, **overrides):
    """
    Instantiate ``MultiModalDiffusion`` with an already-built *dit_model*.

    Extra keyword args override (or add) fields in *cfg* exactly like the other
    loaders.
    """

    final_cfg = _merge_cfg(cfg, overrides)
    diffusion_model = MultiModalDiffusion(dit=dit_model, **final_cfg)

    if "noise_select" in cfg and cfg.noise_select:
        diffusion_model.noise_select = True
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