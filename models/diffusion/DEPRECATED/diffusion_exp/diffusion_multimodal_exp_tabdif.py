import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


##############################
# MultiModalDiffusion
##############################

class MultiModalDiffusion(nn.Module):
    """
    A diffusion wrapper for a MultimodalDiT model that:
     - Noises both the image latents (x_img) and tabular data (x_tab).
     - Denoises them jointly, letting the model do cross-attention in both directions.
     - Uses an EDM-style approach (sigma-based, second-order Heun steps) for sampling.

    Expected usage:
      1) 'forward(...)' for training => returns MSE loss across both image & tab 
         (i.e., model tries to reconstruct the clean x_img, x_tab from their noised versions).
      2) 'sample(...)' => given random initial noise for both image & tab, do the
         EDM sampling loop to produce denoised pairs.

    Args:
        model: The MultimodalDiT or similar that implements:
            model(x_img_t, x_tab_t, time_t) -> 
                {"img_out": (B, in_img_channels, H, W or shape), 
                 "tab_out": (B, d_tab)}
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        sigma_data: Data SNR scale (typical EDM param).
        p_mean, p_std: lognormal distribution parameters to sample \(\sigma\) from.
        num_steps: default # of steps in sampling.
        rho: exponent in the sigma schedule (EDM style).
        S_churn, S_min, S_max, S_noise: parameters for stochasticity in sampling steps.
    """

    def __init__(
            self,
            model: nn.Module,
            sigma_min: float = 0.002,
            sigma_max: float = 80.0,
            sigma_data: float = 0.9,
            p_mean: float = -0.6,
            p_std: float = 1.2,
            num_steps: int = 18,
            rho: float = 7.0,
            S_churn: float = 0.0,
            S_min: float = 0.0,
            S_max: float = float('inf'),
            S_noise: float = 1.0,
    ):
        super().__init__()
        self.model = model

        # EDM parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        self.num_steps = num_steps
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        # We'll define a helper for random noise generation
        # in typical PyTorch code you'd use "torch.randn_like(...)"
        self.randn_like = torch.randn_like

    def forward(
            self,
            x_img_clean: torch.Tensor,
            # shape (B, in_img_channels, H, W) - these are "latents" if you pre-encoded images
            x_tab_clean: torch.Tensor,  # shape (B, d_tab)
            **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass => returns MSE loss for EDM training. 
        1) sample lognormal sigma
        2) add noise to (x_img_clean, x_tab_clean)
        3) pass noised data to model
        4) combine model output with c_skip/c_out to get D_x
        5) compute MSE vs. x_img_clean, x_tab_clean
        """
        B = x_img_clean.shape[0]
        device = x_img_clean.device

        # 1) sample \sigma from lognormal
        rnd_normal = torch.randn(B, device=device)
        log_sigma = rnd_normal * self.p_std + self.p_mean
        sigma = log_sigma.exp().reshape(B, 1)  # shape (B,1)

        # 2) add noise
        noise_img = self.randn_like(x_img_clean)
        noise_tab = torch.randn_like(x_tab_clean)
        # broadcast sigma to match shapes
        # for images: shape (B,1,1,1)
        sigma_img = sigma.reshape(B, 1, 1, 1)
        x_img_noisy = x_img_clean + noise_img * sigma_img
        # for tab: shape (B,1)
        x_tab_noisy = x_tab_clean + noise_tab * sigma

        # 3) run model forward. We pass t = log_sigma / 4 (like Sony code) or directly sigma
        #    We'll do c_noise = log_sigma / 4. That means model is expecting a scalar time for each sample.
        c_noise = log_sigma / 4.0  # shape (B,)
        out = self.model(  # This should return e.g. {"img_out":..., "tab_out":...}
            x_img_noisy.float(),
            x_tab_noisy.float(),
            c_noise.float(),
            **kwargs
        )
        # out["img_out"] shape => (B, in_img_channels, H, W)
        # out["tab_out"] shape => (B, d_tab)

        # 4) c_skip / c_out logic
        # c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
        # c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        sigma_sq = sigma ** 2
        sd_sq = self.sigma_data ** 2
        c_skip = sd_sq / (sigma_sq + sd_sq)  # shape (B,1)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma_sq + sd_sq)  # shape (B,1)

        # broadcast c_skip, c_out for image
        c_skip_img = c_skip.reshape(B, 1, 1, 1)
        c_out_img = c_out.reshape(B, 1, 1, 1)

        # compute D_x
        #    D_x_img = c_skip_img * x_img_noisy + c_out_img * out["img_out"]
        #    D_x_tab = c_skip     * x_tab_noisy + c_out     * out["tab_out"]
        D_x_img = c_skip_img * x_img_noisy + c_out_img * out["img_out"]
        D_x_tab = c_skip * x_tab_noisy + c_out * out["tab_out"]

        # 5) compute MSE
        loss_img = F.mse_loss(D_x_img, x_img_clean, reduction='none').mean(dim=[1, 2, 3])
        loss_tab = F.mse_loss(D_x_tab, x_tab_clean, reduction='none').mean(dim=1)
        # sum or average them
        loss = (loss_img + loss_tab).mean()
        return {"loss": loss, "loss_img": loss_img.mean(), "loss_tab": loss_tab.mean()}

    @torch.no_grad()
    def sample(
            self,
            real_img: torch.Tensor,  # shape (B, in_chans, H, W)
            real_tab: torch.Tensor,  # shape (B, d_tab)
            steps: int = 30,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Modified sample method that:
          1) Takes real image & table data to determine shapes (batch, channels, etc.).
          2) Creates random noise of the same shapes as starting points.
          3) Runs the standard EDM sampling loop from x_img_init, x_tab_init.
          4) Returns the final outputs.

        This DOES NOT denoise 'real_img' or 'real_tab' – it simply uses them to get shapes.
        """
        if steps is None:
            steps = self.num_steps  # fallback to default

        device = real_img.device
        B, in_chans, H, W = real_img.shape
        d_tab = real_tab.shape[1]

        # 1) Create random initial states from shape
        x_img_init = torch.randn_like(real_img)  # shape (B, in_chans, H, W)
        x_tab_init = torch.randn_like(real_tab)  # shape (B, d_tab)

        # 2) define time-step schedule
        step_indices = torch.arange(steps, dtype=torch.float64, device=device)

        def s2alpha(sigma):
            return sigma ** (1.0 / self.rho)

        t_steps = (
                          s2alpha(self.sigma_max)
                          + (step_indices / (steps - 1)) * (s2alpha(self.sigma_min) - s2alpha(self.sigma_max))
                  ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])], dim=0)  # (steps+1,)

        # 3) multiply init by t_steps[0]
        x_img_next = x_img_init.double() * t_steps[0]
        x_tab_next = x_tab_init.double() * t_steps[0]

        for i in range(steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            x_img_cur = x_img_next
            x_tab_cur = x_tab_next

            # "churn" step
            gamma = (
                min(self.S_churn / steps, math.sqrt(2) - 1)
                if (self.S_min <= t_cur <= self.S_max)
                else 0.0
            )
            t_hat_scalar = t_cur + gamma * t_cur

            # Expand t_hat to shape (B,)
            t_hat = t_hat_scalar * torch.ones((B,), device=device)

            # Possibly add noise if gamma>0
            if gamma > 0.0:
                eps_img = self.randn_like(x_img_cur)
                eps_tab = torch.randn_like(x_tab_cur)
                sigma_extra = (t_hat_scalar ** 2 - t_cur ** 2).sqrt() * self.S_noise

                # Broadcast shapes
                sigma_extra_img = sigma_extra.view(1, 1, 1)
                x_img_hat = x_img_cur + sigma_extra_img * eps_img
                sigma_extra_tab = sigma_extra  # scalar, multiplied per row
                x_tab_hat = x_tab_cur + sigma_extra_tab * eps_tab
            else:
                x_img_hat = x_img_cur
                x_tab_hat = x_tab_cur

            # Euler step
            log_sigma_hat = torch.log(t_hat + 1e-12)
            c_noise = log_sigma_hat / 4.0
            out = self.model_forward_wrapper(
                x_img_hat, x_tab_hat, t_hat, c_noise, **kwargs
            )
            D_x_img = out["D_x_img"]
            D_x_tab = out["D_x_tab"]

            eps = 1e-12
            denom_img = t_hat.view(B, 1, 1, 1).clamp_min(eps)
            denom_tab = t_hat.view(B, 1).clamp_min(eps)
            d_cur_img = (x_img_hat - D_x_img) / denom_img
            d_cur_tab = (x_tab_hat - D_x_tab) / denom_tab

            # second-order correction if i < steps-1
            t_next_scalar = t_next
            t_next_vec = t_next_scalar * torch.ones((B,), device=device)

            x_img_next = x_img_hat + (t_next_scalar - t_hat_scalar) * d_cur_img
            x_tab_next = x_tab_hat + (t_next_scalar - t_hat_scalar) * d_cur_tab

            if i < steps - 1:
                log_sigma_next = torch.log(t_next_vec + eps)
                c_noise_next = log_sigma_next / 4.0
                out2 = self.model_forward_wrapper(
                    x_img_next, x_tab_next, t_next_vec, c_noise_next, **kwargs
                )
                D_x_img2 = out2["D_x_img"]
                D_x_tab2 = out2["D_x_tab"]

                d_prime_img = (x_img_next - D_x_img2) / t_next_vec.view(B, 1, 1, 1).clamp_min(eps)
                d_prime_tab = (x_tab_next - D_x_tab2) / t_next_vec.view(B, 1).clamp_min(eps)
                x_img_next = x_img_hat + (t_next_scalar - t_hat_scalar) * 0.5 * (d_cur_img + d_prime_img)
                x_tab_next = x_tab_hat + (t_next_scalar - t_hat_scalar) * 0.5 * (d_cur_tab + d_prime_tab)

        x_img_final = x_img_next.float()
        x_tab_final = x_tab_next.float()

        return x_img_final, x_tab_final

    def model_forward_wrapper(
            self,
            x_img_t: torch.Tensor,  # (B, in_chans, H, W), current noised
            x_tab_t: torch.Tensor,  # (B, d_tab), current noised
            sigma_val: torch.Tensor,  # shape (B,) or scalar
            c_noise: torch.Tensor,  # shape (B,)
            **kwargs
    ) -> dict:
        """
        Similar to Sony's 'model_forward_wrapper', but for joint image+tab.
        We'll produce:
          D_x_img = c_skip * x_img_t + c_out * model_out["img_out"]
          D_x_tab = c_skip * x_tab_t + c_out * model_out["tab_out"]
        and return them in a dict.

        In practice, the model is the MultimodalDiT, expecting:
          model(x_img_t, x_tab_t, c_noise, ...)
          => {"img_out": ..., "tab_out": ...}
        """
        B = x_img_t.shape[0]
        device = x_img_t.device
        sigma_val = sigma_val.view(B)  # ensure shape (B,)

        # c_skip, c_out
        sigma_sq = sigma_val ** 2
        sd_sq = self.sigma_data ** 2
        c_skip_val = sd_sq / (sigma_sq + sd_sq)
        c_out_val = sigma_val * self.sigma_data / torch.sqrt(sigma_sq + sd_sq)

        # broadcast for image
        c_skip_img = c_skip_val.view(B, 1, 1, 1)
        c_out_img = c_out_val.view(B, 1, 1, 1)
        # broadcast for tab
        c_skip_tab = c_skip_val.view(B, 1)
        c_out_tab = c_out_val.view(B, 1)

        # forward pass
        out = self.model(x_img_t.float(), x_tab_t.float(), c_noise.float(), **kwargs)

        # combine => D_x
        D_x_img = c_skip_img * x_img_t + c_out_img * out["img_out"]
        D_x_tab = c_skip_tab * x_tab_t + c_out_tab * out["tab_out"]

        return {
            "D_x_img": D_x_img,
            "D_x_tab": D_x_tab
        }


from omegaconf import DictConfig
from utils.configurations import apply_overrides
from typing import Any
def load_diffusion(cfg: DictConfig, dit_model: nn.Module, **overrides: Any) -> nn.Module:
    """
    Load a MultiModalDiffusion model based on the provided configuration.

    The config is expected to have a top-level 'diffusion' section containing the parameters
    for the MultiModalDiffusion. The 'dit_model' argument is required because
    MultiModalDiffusion depends on a pre-loaded DiT model.

    Args:
        cfg (DictConfig): The Hydra configuration object (must contain a 'diffusion' section).
        dit_model (nn.Module): The already loaded DiT model, required by the Diffusion model.
        **overrides (Any): Arbitrary keyword arguments used to override the default configuration.

    Returns:
        nn.Module: The loaded MultiModalDiffusion model.
    """
    # Apply any overrides to the config before loading
    cfg = apply_overrides(cfg, overrides)

    print("[INFO] Loading Diffusion model with config:", cfg)

    # Instantiate the Diffusion model, injecting the loaded DiT
    diffusion_model = MultiModalDiffusion(model=dit_model, **cfg.diffusion)
    print("[INFO] Loaded Diffusion Model")
    return diffusion_model

if __name__ == "__main__":
    # Suppose we have:
    from torch import optim, Tensor

    # 1) A "MultiModalDiT" instance that expects:
    #    model(x_img, x_tab, time_scalar) -> {"img_out":..., "tab_out":...}
    from models.dit.dit_multimodal_exp_tabdif import MultiModalDiT

    B = 4
    H, W = 32, 32
    in_chans = 4
    d_tab = 174

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
        d_out_tab=d_tab,  # e.g. same dimension as x_tab or whatever you prefer
    )

    # 2) Our EDM diffuser
    diffuser = MultiModalDiffusion(model)

    # 3) Some example training batch
    x_img = torch.randn(B, in_chans, H, W)
    x_tab = torch.randn(B, d_tab)

    # 4) forward => get loss
    loss = diffuser.forward(x_img, x_tab)
    print("Training loss:", loss.item())

    # 5) do a gradient step
    opt = optim.Adam(diffuser.parameters(), lr=1e-4)
    loss.backward()
    opt.step()

    # 6) sampling: let's say we want to sample from random noise
    # e.g. x_img_init, x_tab_init ~ N(0,1)
    x_img_init = torch.randn_like(x_img)
    x_tab_init = torch.randn_like(x_tab)

    out = diffuser.sample(x_img_init, x_tab_init, steps=20)
    print("Sampled image shape:", out["img_out"].shape)
    print("Sampled tab shape:", out["tab_out"].shape)
