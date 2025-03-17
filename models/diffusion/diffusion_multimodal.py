import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Any

from torch import Tensor
from torchvision.utils import save_image
from omegaconf import DictConfig
from pathlib import Path
from hydra import compose, initialize_config_dir

from utils.ddp import is_main_process
from utils.configurations import apply_overrides
from models.dit.dit_multimodal import load_dit


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

    def forward(self, images: torch.Tensor, table_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform a forward pass to compute the EDM loss.

        Args:
            images (torch.Tensor): The real image tensors of shape (B, C, H, W).
            table_data (torch.Tensor, optional): Tabular data of shape (B, T) or similar.

        Returns:
            (total_loss, image_loss, tab_loss) as a tuple of:
                - total_loss (torch.Tensor): combined image + table loss
                - image_loss (torch.Tensor): image-only diffusion loss
                - tab_loss (Optional[torch.Tensor]): table-only diffusion loss (if table data is provided)
        """
        device = images.device
        B = images.shape[0]

        # 1) Sample sigma from a log-normal distribution
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()

        # 2) Compute the weight
        weight = ((sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2)

        # 3) Add noise to input
        noise = torch.randn_like(images)
        noised_input = images + sigma * noise

        # 4) EDM scaling
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1.0 / (self.sigma_data**2 + sigma_in**2).sqrt()
        c_skip = self.sigma_data**2 / (sigma_in**2 + self.sigma_data**2)
        c_out = sigma_in * self.sigma_data / (sigma_in**2 + self.sigma_data**2).sqrt()
        t = (sigma_in.log() / 4.0).squeeze()

        # 5) Forward pass through the model
        out = self.dit(
            x_img=c_in * noised_input,
            t=t,
            tab=table_data,
            cfg=1.0,  # CFG not typically used during training
            mask_ratio=self.train_mask_ratio
        )
        F_x = out['image_sample']

        # Combine for denoised prediction
        D_xn = c_skip * noised_input + c_out * F_x
        loss_img = weight * ((D_xn - images) ** 2)
        image_loss = loss_img.mean(dim=[1, 2, 3]).mean()

        # Optional latent regularization
        if self.latent_reg_weight > 0:
            real_mean = images.mean(dim=(0, 2, 3), keepdim=True)
            real_std = images.std(dim=(0, 2, 3), keepdim=True)
            pred_mean = F_x.mean(dim=(0, 2, 3), keepdim=True)
            pred_std = F_x.std(dim=(0, 2, 3), keepdim=True)
            mean_loss = F.mse_loss(pred_mean, real_mean)
            std_loss = F.mse_loss(pred_std, real_std)
            reg_loss = mean_loss + std_loss
            image_loss = image_loss + self.latent_reg_weight * reg_loss


        return image_loss

    @torch.no_grad()
    def sample(
            self, batch_size: int = 4, table_data: torch.Tensor = None, cfg: float = 1.0,
            steps: Optional[int] = None, height: int = 64, width: int = 64, device: str = 'cuda',
            save_path: Optional[str] = None
    ) -> Tensor:
        """
        Produce samples using the EDM sampler.

        Args:
            batch_size (int): Number of images to sample.
            table_data [torch.Tensor]: Conditioning tabular data.
            cfg (float): Classifier-Free Guidance scale.
            steps (int, optional): Number of sampling steps. Defaults to self.num_steps.
            height (int): Image height.
            width (int): Image width.
            device (str): The device to run sampling on.
            save_path (str, optional): If provided, will save a grid of the final latents only on main process.

        Returns:
            final_latents (torch.Tensor): The final latents of shape (B, C, H, W).
        """
        self.eval()
        steps = steps or self.num_steps
        c = self.dit.in_channels

        # Start from pure noise
        x = torch.randn((batch_size, c, height, width), device=device)
        t_vals = self.create_edm_timesteps(steps, device)
        x_next = x.double() * t_vals[0]

        def model_forward(
                x_in: torch.Tensor, t_sigma: torch.Tensor, tab_data: torch.Tensor, cfg_val: float
        ) -> torch.Tensor:
            """
            Internal utility function to run the underlying model forward with the EDM scaling.
            """
            B = x_in.shape[0]
            sigma_in = t_sigma.reshape(-1, 1, 1, 1).float()
            c_in = 1.0 / (sigma_in**2 + self.sigma_data**2).sqrt()
            c_skip = self.sigma_data**2 / (sigma_in**2 + self.sigma_data**2)
            c_out = sigma_in * self.sigma_data / (sigma_in**2 + self.sigma_data**2).sqrt()

            t_embed = (sigma_in.log() / 4.0).reshape(-1)
            if t_embed.numel() == 1 and B > 1:
                t_embed = t_embed.expand(B)

            out = self.dit(
                x_img=c_in * x_in.float(),
                t=t_embed,
                tab=tab_data,
                cfg=cfg_val,
                mask_ratio=0.0
            )
            Fx = out['image_sample'].float()
            denoised = c_skip * x_in + c_out * Fx
            return denoised

        # Sampler loop
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_cur = x_next
            gamma = (min(self.s_churn / steps, np.sqrt(2) - 1)
                     if (self.s_min <= t_cur <= self.s_max) else 0.0)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.s_noise * torch.randn_like(x_cur)

            # Euler step
            denoised = model_forward(x_hat, t_hat, table_data, cfg)
            denoised = denoised.double()
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2nd order correction
            if i < steps - 1:
                denoised2 = model_forward(x_next, t_next, table_data, cfg)
                denoised2 = denoised2.double()
                d_prime = (x_next - denoised2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        final_latents = x_next.float()

        # File I/O only on main process (rank 0) to avoid collisions in DDP
        if save_path and is_main_process():
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            latents_for_vis = (final_latents - final_latents.min()) / (
                final_latents.max() - final_latents.min() + 1e-7
            )
            save_image(latents_for_vis, save_path, nrow=int(batch_size**0.5))

        return final_latents

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
        total_loss = diffusion_model(images, table_data)
        print(f"Total loss: {total_loss.item():.4f}")

        # Sampling demonstration
        with torch.no_grad():
            latents = diffusion_model.sample(
                batch_size=4,
                table_data=table_data,
                steps=10,
                height=64,
                width=64,
                device='cuda'
            )
            print("Sampled latents shape:", latents.shape)