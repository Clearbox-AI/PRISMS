import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Any
from torchvision.utils import save_image
from omegaconf import DictConfig
from pathlib import Path
from hydra import compose, initialize_config_dir

from utils.ddp import is_main_process
from utils.configurations import apply_overrides
from models.dit.dit_multimodal import load_dit
import random


class MultiModalDiffusion(nn.Module):
    """
    EDM-based multi-modal diffusion for images + tabular data, optionally using a VAE.
    This module implements a training forward pass (EDM loss) and a sampling procedure.
    """

    def __init__(
        self,
        dit: nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -0.6,
        p_std: float = 1.2,
        sigma_data: float = 0.9,
        num_steps: int = 18,
        rho: float = 7.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float('inf'),
        s_noise: float = 1.0,
        train_mask_ratio: float = 0.0,
        latent_reg_weight: float = 0.0
    ) -> None:
        """
        Args:
            dit (nn.Module): The underlying diffusion model (e.g., DiT).
            sigma_min (float): Minimum noise level.
            sigma_max (float): Maximum noise level.
            p_mean (float): Mean for log-normal sampling of sigma.
            p_std (float): Std for log-normal sampling of sigma.
            sigma_data (float): Data noise level for EDM scaling.
            num_steps (int): Number of sampling steps.
            rho (float): Rho exponent for noise schedule.
            s_churn (float): Stochasticity weight for noise schedule.
            s_min (float): Lower bound of noise ramp.
            s_max (float): Upper bound of noise ramp.
            s_noise (float): Additional noise scaling.
            train_mask_ratio (float): Mask ratio for training (e.g., token dropping).
            latent_reg_weight (float): Weight for an optional latent regularization term.
        """
        super().__init__()
        self.dit = dit

        # EDM hyperparams
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.num_steps = num_steps
        self.rho = rho
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise
        self.train_mask_ratio = train_mask_ratio
        self.latent_reg_weight = latent_reg_weight

    def forward(self, images: torch.Tensor, table_data: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform a forward pass to compute the EDM loss.

        Args:
            images (torch.Tensor): The real image tensors of shape (B, C, H, W).
            table_data (torch.Tensor): Tabular data of shape (B, T) or similar.

        Returns:
            (total_loss, image_loss, tab_loss) as a tuple of:
                - total_loss (torch.Tensor): combined image + table loss
                - image_loss (torch.Tensor): image-only diffusion loss
                - tab_loss (Optional[torch.Tensor]): table-only diffusion loss (if table data is provided)
        """

        device = images.device
        B = images.shape[0]

        # ------------------------------------------------
        # 1) Sample log-normal noise scale for images
        # ------------------------------------------------
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma_img = (rnd_normal * self.p_std + self.p_mean).exp()  # shape [B,1,1,1]

        # EDM weighting factor
        weight_img = ((sigma_img ** 2 + self.sigma_data ** 2) / (sigma_img * self.sigma_data) ** 2)

        # Add noise to images
        noise_img = torch.randn_like(images)
        noised_images = images + sigma_img * noise_img

        # Decide if we do "joint" or "conditional" for the table:
        # (We always have a real table_data, but we may or may not add noise.)
        do_joint = (random.random() < 1)
        if do_joint:
            # Full joint mode => same sigma as images
            sigma_tab = sigma_img.view(B, 1)  # shape [B,1]
        else:
            # Conditional mode => "sigma=0" => keep table clean
            sigma_tab = torch.zeros_like(sigma_img.view(B, 1))

        # Prepare table noise
        noise_tab = torch.randn_like(table_data)
        noised_table = table_data + sigma_tab * noise_tab

        # EDM weighting factor for table
        weight_tab = ((sigma_tab ** 2 + self.sigma_data ** 2) / (sigma_tab * self.sigma_data) ** 2)
        # Avoid division by zero if sigma_tab=0 => set weight_tab=1 when in conditional mode
        # or do it more carefully with .where() logic:
        weight_tab = torch.where(sigma_tab > 1e-8, weight_tab, torch.ones_like(weight_tab))

        # ------------------------------------------------
        # 2) EDM scaling for images
        # ------------------------------------------------
        c_in_img = 1.0 / (self.sigma_data ** 2 + sigma_img ** 2).sqrt()
        c_skip_img = self.sigma_data ** 2 / (sigma_img ** 2 + self.sigma_data ** 2)
        c_out_img = sigma_img * self.sigma_data / (sigma_img ** 2 + self.sigma_data ** 2).sqrt()
        t_img = (sigma_img.log() / 4.0).squeeze()  # shape (B,)

        # EDM scaling for table
        c_in_tab = 1.0 / (self.sigma_data ** 2 + sigma_tab ** 2).sqrt()
        c_skip_tab = self.sigma_data ** 2 / (sigma_tab ** 2 + self.sigma_data ** 2)
        c_out_tab = sigma_tab * self.sigma_data / (sigma_tab ** 2 + self.sigma_data ** 2).sqrt()

        # ------------------------------------------------
        # 3) Forward pass in the model
        # ------------------------------------------------
        # We pass scaled/noised images & scaled/noised table
        out = self.dit(
            x_img=c_in_img * noised_images,
            t=t_img,
            tab=c_in_tab * noised_table,
            cfg=1.0,
            mask_ratio=self.train_mask_ratio
        )

        # The model must return out["image_sample"] & out["table_sample"]
        pred_img = out["image_sample"]  # shape (B, C, H, W)
        pred_tab = out["table_sample"]  # shape (B, T)

        # ------------------------------------------------
        # 4) "Denoised" final predictions via EDM formula
        # ------------------------------------------------
        # For images
        denoised_img = c_skip_img * noised_images + c_out_img * pred_img
        # For tables
        denoised_tab = c_skip_tab * noised_table + c_out_tab * pred_tab

        # ------------------------------------------------
        # 5) MSE Loss vs. ground truth
        # ------------------------------------------------
        # Image loss
        img_loss_val = weight_img * ((denoised_img - images) ** 2)
        image_loss = img_loss_val.mean(dim=[1, 2, 3]).mean()

        # Optional latent regularization
        if self.latent_reg_weight > 0:
            real_mean = images.mean(dim=(0, 2, 3), keepdim=True)
            real_std = images.std(dim=(0, 2, 3), keepdim=True)
            pred_mean = pred_img.mean(dim=(0, 2, 3), keepdim=True)
            pred_std = pred_img.std(dim=(0, 2, 3), keepdim=True)
            reg_loss = F.mse_loss(pred_mean, real_mean) + F.mse_loss(pred_std, real_std)
            image_loss = image_loss + self.latent_reg_weight * reg_loss

        # Table loss
        tab_loss_val = weight_tab * ((denoised_tab - table_data) ** 2)  # shape [B, T]
        table_loss = tab_loss_val.mean()

        # Weighted sum
        total_loss = image_loss + 0.7 * table_loss

        return total_loss, image_loss, table_loss

    @torch.no_grad()
    def sample(
            self, batch_size: int = 4, table_data: Optional[torch.Tensor] = None, cfg: float = 1.0,
            steps: Optional[int] = None, height: int = 64, width: int = 64, device: str = 'cuda',
            save_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Produce samples using the EDM sampler.

        Args:
            batch_size (int): Number of images to sample.
            table_data (Optional[torch.Tensor]): Conditioning tabular data.
            cfg (float): Classifier-Free Guidance scale.
            steps (int, optional): Number of sampling steps. Defaults to self.num_steps.
            height (int): Image height.
            width (int): Image width.
            device (str): The device to run sampling on.
            save_path (str, optional): If provided, will save a grid of the final latents only on main process.

        Returns:
            final_latents (torch.Tensor): The final latents of shape (B, C, H, W).
            latest_tab_sample (Optional[torch.Tensor]): The final predicted table sample if available.
        """

        self.eval()
        steps = steps or self.num_steps
        c = self.dit.in_channels  # e.g. 3 or 4 for images

        # 1) Image latents start from random noise
        x_img = torch.randn(batch_size, c, height, width, device=device, dtype=torch.float64)

        # 2) Table latents: if None => unconditional => random noise; else => "clean condition"
        if table_data is None:
            # We'll generate the table from noise
            table_in_dim = 174
            x_tab = torch.randn(batch_size, table_in_dim, device=device, dtype=torch.float64)
            table_is_uncond = True
        else:
            # We'll treat the table as condition => effectively sigma=0 => no random variation
            x_tab = table_data.to(device).double()
            table_is_uncond = False

        # Create time schedule
        t_vals = self.create_edm_timesteps(steps, device)
        # Start from the largest sigma
        x_img = x_img * t_vals[0]
        if table_is_uncond:
            x_tab = x_tab * t_vals[0]

        # Sampler loop
        x_img_next = x_img
        x_tab_next = x_tab

        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_i = x_img_next
            x_t = x_tab_next

            # s_churn
            gamma = (min(self.s_churn / steps, (2 ** 0.5) - 1)
                     if (self.s_min <= t_cur <= self.s_max) else 0.0)
            t_hat = t_cur + gamma * t_cur

            # Add noise (stochasticity)
            x_i_hat = x_i + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * torch.randn_like(x_i)
            if table_is_uncond:
                x_t_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * torch.randn_like(x_t)
            else:
                x_t_hat = x_t

            # ---- First pass / Euler step ----
            x_img_denoised, x_tab_denoised = self.model_forward_edm(x_i_hat, x_t_hat, t_hat, cfg, table_is_uncond)
            d_cur_img = (x_i_hat - x_img_denoised) / t_hat
            x_img_next = x_i_hat + (t_next - t_hat) * d_cur_img

            d_cur_tab = None
            if x_tab_denoised is not None:
                d_cur_tab = (x_t_hat - x_tab_denoised) / t_hat
                x_tab_next = x_t_hat + (t_next - t_hat) * d_cur_tab

            # ---- Second pass / Heun correction ----
            if i < steps - 1:
                x_img_denoised2, x_tab_denoised2 = self.model_forward_edm(x_img_next, x_tab_next, t_next, cfg,
                                                                          table_is_uncond)

                d_prime_img = (x_img_next - x_img_denoised2) / t_next
                x_img_next = x_i_hat + (t_next - t_hat) * 0.5 * (d_cur_img + d_prime_img)

                if x_tab_denoised2 is not None and d_cur_tab is not None:
                    d_prime_tab = (x_tab_next - x_tab_denoised2) / t_next
                    x_tab_next = x_t_hat + (t_next - t_hat) * 0.5 * (d_cur_tab + d_prime_tab)

        final_imgs = x_img_next.float()
        final_tabs = x_tab_next.float()  # always produce a final table

        return final_imgs, final_tabs

    def model_forward_edm(
            self,
            x_img_in: torch.Tensor,
            x_tab_in: torch.Tensor,
            sigma_val: float,
            cfg_val: float,
            table_is_uncond: bool
    ):
        """
        Single step of EDM scaling and forward pass. We unify image+table:
          - If table_is_uncond=True => apply same sigma to table
          - Otherwise => treat table as condition => effectively sigma=0
        """
        B = x_img_in.shape[0]
        sigma_img = torch.full((B, 1, 1, 1), sigma_val, device=x_img_in.device, dtype=torch.float32)

        # EDM factors for images
        c_in = 1.0 / (sigma_img ** 2 + self.sigma_data ** 2).sqrt()
        c_skip = self.sigma_data ** 2 / (sigma_img ** 2 + self.sigma_data ** 2)
        c_out = sigma_img * self.sigma_data / (sigma_img ** 2 + self.sigma_data ** 2).sqrt()
        t_embed = (sigma_img.log() / 4.0).view(-1)

        if table_is_uncond:
            # Table has same sigma
            sigma_tab = torch.full((B, 1), sigma_val, device=x_tab_in.device, dtype=torch.float32)
            c_in_tab = 1.0 / (sigma_tab ** 2 + self.sigma_data ** 2).sqrt()
            c_skip_tab = self.sigma_data ** 2 / (sigma_tab ** 2 + self.sigma_data ** 2)
            c_out_tab = sigma_tab * self.sigma_data / (sigma_tab ** 2 + self.sigma_data ** 2).sqrt()
        else:
            # Clean table => sigma=0 => skip
            c_in_tab = None
            c_skip_tab = None
            c_out_tab = None

        # Scale inputs
        x_img_scaled = c_in * x_img_in.float()
        if table_is_uncond:
            x_tab_scaled = c_in_tab * x_tab_in.float()
        else:
            x_tab_scaled = x_tab_in.float()

        out = self.dit(
            x_img=x_img_scaled,
            t=t_embed,
            tab=x_tab_scaled,
            cfg=cfg_val,
            mask_ratio=0.0
        )
        Fx_img = out["image_sample"].float()
        Fx_tab = out.get("table_sample", None)

        # "denoised" outputs
        denoised_img = c_skip * x_img_in + c_out * Fx_img

        denoised_tab = None
        if Fx_tab is not None:
            if table_is_uncond:
                denoised_tab = c_skip_tab * x_tab_in + c_out_tab * Fx_tab
            else:
                # Even in conditional mode, we can still produce a table output,
                # but it won't differ much from x_tab_in if sigma=0
                denoised_tab = x_tab_in

        return denoised_img, denoised_tab

    def create_edm_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        """
        Create a time schedule for EDM sampling steps.

        Args:
            steps (int): Number of steps.
            device (torch.device): The device to construct the tensor on.

        Returns:
            (torch.Tensor): Time steps of shape (steps+1,).
        """
        step_indices = torch.arange(steps, dtype=torch.float64, device=device)
        inv_rho = 1.0 / self.rho
        t_values = (
            self.sigma_max**inv_rho +
            step_indices / (steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        )**self.rho
        # Append zero for the final step
        t_values = torch.cat([t_values, torch.zeros_like(t_values[:1])])
        return t_values


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
    diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg.diffusion)
    print("[INFO] Loaded Diffusion Model")
    return diffusion_model


if __name__ == "__main__":
    """
    Simple test driver for MultiModalDiffusion with Hydra-based config.
    """

    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        dit_cfg = compose(config_name="dit")
        dit_model = load_dit(dit_cfg)
        diffusion_cfg = compose(config_name="diffusion")
        diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

        # Make up some dummy data
        images = torch.randn(4, 4, 64, 64).cuda()
        table_data = torch.randn(4, 174).cuda()  # (B=4, some tab dim=10)

        # Forward pass
        total_loss, image_loss, tab_loss = diffusion_model(images, table_data)
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Image loss: {image_loss.item():.4f}")
        print(f"Table loss: {tab_loss.item():.4f}" if tab_loss is not None else "No table loss")

        # Sampling demonstration
        with torch.no_grad():
            latents, latest_tab = diffusion_model.sample(
                batch_size=4,
                table_data=table_data,
                steps=10,
                height=64,
                width=64,
                device='cuda'
            )
            print("Sampled latents shape:", latents.shape)
            if latest_tab is not None:
                print("Sampled table shape:", latest_tab.shape)