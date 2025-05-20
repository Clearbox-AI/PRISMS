import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Any, Dict, List

from torch import Tensor
from torchvision.utils import save_image
from omegaconf import DictConfig
from pathlib import Path
from hydra import compose, initialize_config_dir
from easydict import EasyDict
import torch.distributed as dist

from utils.ddp import is_main_process
from utils.configurations import apply_overrides

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

# class MultiModalDiffusion(nn.Module):
#     """
#     EDM-based multi-modal diffusion for images + tabular data, optionally using a VAE.
#     This module implements a training forward pass (EDM loss) and a sampling procedure.
#     """
#
#     def __init__(
#             self,
#             dit: nn.Module,
#             train_mask_ratio: float = 0.0,
#             latent_reg_weight: float = 0.0,
#             edm_config: dict = None
#     ) -> None:
#         """
#         Initializes the MultiModalDiffusion model using Hydra configs.
#
#         Args:
#             dit (nn.Module): The underlying diffusion model (e.g., DiT).
#             train_mask_ratio (float): Mask ratio for training (e.g., token dropping).
#             latent_reg_weight (float): Weight for an optional latent regularization term.
#         """
#         super().__init__()
#         self.dit = dit
#         self.train_mask_ratio = train_mask_ratio
#         self.latent_reg_weight = latent_reg_weight
#
#         # EDM configuration
#         # Example structure of edm_config:
#         edm_config = dict(
#               sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2,
#               sigma_data=0.5, num_steps=40, rho=7,
#               S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
#           )
#
#         self.edm_config = edm_config or {}
#         self._dtype = DTYPE_MAP["float32"]
#
#         self.image_pixel_channels = 4
#         self.image_pixel_image_height = 32
#         self.image_pixel_image_width = 32
#         self.tab_size = 174
#
#
#     def forward(self, images: torch.Tensor, table_data: torch.Tensor) -> dict[str, Any]:
#         """
#         Perform a forward pass to compute the EDM loss.
#
#         Args:
#             images (torch.Tensor): The real image tensors of shape (B, C, H, W).
#             table_data (torch.Tensor): Tabular data of shape (B, T) or similar.
#
#         Returns:
#             total_loss (torch.Tensor)
#         """
#         device = images.device
#         B = images.shape[0]
#
#         # --------------------------------------------------
#         # 1) Sample sigma from log-normal
#         # --------------------------------------------------
#         rnd_normal = torch.randn([B, 1, 1, 1], device=device)
#         sigma = (rnd_normal * self.p_std + self.p_mean).exp()  # shape [B, 1, 1, 1]
#
#         # --------------------------------------------------
#         # 2) Compute EDM weighting
#         # --------------------------------------------------
#         # shape [B, 1, 1, 1]
#         weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)
#
#         # --------------------------------------------------
#         # 3) Add noise to images and tab
#         # --------------------------------------------------
#         noise_img = torch.randn_like(images)  # [B, C, H, W]
#         noised_imgs = images + sigma * noise_img  # [B, C, H, W]
#
#         noise_tab = torch.randn_like(table_data)  # [B, T]
#         # Flatten sigma to [B,1] for tab data
#         sigma_tab = sigma.view(B, 1)  # [B, 1]
#         noised_tabs = table_data + sigma_tab * noise_tab  # [B, T]
#
#         # --------------------------------------------------
#         # 4) EDM scaling factors
#         # --------------------------------------------------
#         # c_in, c_skip, c_out all shape [B,1,1,1]
#         sigma_in = sigma
#         c_in = 1.0 / (self.sigma_data ** 2 + sigma_in ** 2).sqrt()  # [B,1,1,1]
#         c_skip = self.sigma_data ** 2 / (sigma_in ** 2 + self.sigma_data ** 2)
#         c_out = sigma_in * self.sigma_data / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()
#
#         # For the tab data, reshape them to [B] or [B,1] where needed
#         c_in_tab = c_in.view(B)  # shape [B]
#         c_skip_tab = c_skip.view(B)
#         c_out_tab = c_out.view(B)
#
#         # --------------------------------------------------
#         # 5) Time embedding for DiT
#         # --------------------------------------------------
#         # shape [B]
#         t_embed = (sigma_in.log() / 4.0).view(B)
#
#         # --------------------------------------------------
#         # 6) Forward pass through the multi-modal DiT
#         #    scaled noised inputs -> predicted “clean” outputs
#         # --------------------------------------------------
#         out = self.dit(
#             # scale the image by c_in (still [B,1,1,1]), so broadcast is okay
#             x_img=c_in * noised_imgs,  # [B,3,H,W]
#             t=t_embed,  # [B]
#             # scale the tab data by c_in_tab[:, None] to get [B,T]
#             x_tab=c_in_tab[:, None] * noised_tabs,  # [B, T]
#             cfg=1.0,  # or your training CFG
#             mask_ratio=self.train_mask_ratio
#         )
#         pred_imgs = out["image_sample"]  # [B, C, H, W]
#         pred_tabs = out["tab_sample"]  # [B, T]
#
#         # --------------------------------------------------
#         # 7) Combine for “denoised” output (EDM style)
#         # --------------------------------------------------
#         # (A) Images
#         D_xn_img = (c_skip * noised_imgs) + (c_out * pred_imgs)
#         loss_img = weight * ((D_xn_img - images) ** 2)  # shape [B, C, H, W]
#         loss_img = loss_img.mean(dim=[1, 2, 3])  # shape [B]
#
#         # (B) Tabs
#         # Expand c_skip_tab, c_out_tab to [B,1] so we can multiply [B,T]
#         c_skip_tab = c_skip_tab[:, None]  # [B,1]
#         c_out_tab = c_out_tab[:, None]  # [B,1]
#         weight_tab = weight.view(B, 1)  # [B,1]
#         D_xn_tab = (c_skip_tab * noised_tabs) + (c_out_tab * pred_tabs)  # [B, T]
#         loss_tab = weight_tab * ((D_xn_tab - table_data) ** 2)  # [B, T]
#         loss_tab = loss_tab.mean(dim=1)  # [B]
#
#         # Sum and average
#         total_loss = (loss_img + loss_tab).mean()
#
#         # --------------------------------------------------
#         # 8) Optional latent regularization
#         # --------------------------------------------------
#         if self.latent_reg_weight > 0:
#             # Example: match predicted vs real image mean+std
#             real_mean_img = images.mean(dim=(0, 2, 3), keepdim=True)
#             real_std_img = images.std(dim=(0, 2, 3), keepdim=True)
#             pred_mean_img = pred_imgs.mean(dim=(0, 2, 3), keepdim=True)
#             pred_std_img = pred_imgs.std(dim=(0, 2, 3), keepdim=True)
#
#             mean_loss_img = F.mse_loss(pred_mean_img, real_mean_img)
#             std_loss_img = F.mse_loss(pred_std_img, real_std_img)
#             reg_loss_img = mean_loss_img + std_loss_img
#
#             # Similarly for tab data
#             real_mean_tab = table_data.mean(dim=0, keepdim=True)
#             real_std_tab = table_data.std(dim=0, keepdim=True)
#             pred_mean_tab = pred_tabs.mean(dim=0, keepdim=True)
#             pred_std_tab = pred_tabs.std(dim=0, keepdim=True)
#
#             mean_loss_tab = F.mse_loss(pred_mean_tab, real_mean_tab)
#             std_loss_tab = F.mse_loss(pred_std_tab, real_std_tab)
#             reg_loss_tab = mean_loss_tab + std_loss_tab
#
#             total_loss += self.latent_reg_weight * (reg_loss_img + reg_loss_tab)
#
#         return {"loss": total_loss, "loss_img": loss_img.mean(), "loss_tab": loss_tab.mean()}
#
#
#     @torch.no_grad()
#     def _sample_edm(
#             self,
#             batch_size: int,
#             cfg: float = 1.0,
#             steps: Optional[int] = None,
#             device: str = 'cuda',
#             init_img: Optional[torch.Tensor] = None,
#             init_tab: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Sampling procedure for both images and tab data.
#
#         Returns final (image_latents, tab_latents).
#         By default starts from random noise, but you can pass init_img/tab if you want
#         e.g. partial diffusion or "inpainting" style approaches.
#
#         This is an EDM sampler using an Euler update (with optional second-order correction).
#         """
#         self.eval()
#         steps = steps or self.num_steps
#
#         # If no init given, start from pure noise
#         if init_img is None:
#             # Suppose the DiT expects shape [B, 4, H, W] for the image
#             init_img = torch.randn((batch_size, 4, self.image_size, self.image_size),
#                                    device=device, dtype=torch.float32)
#         if init_tab is None:
#             # Suppose tab shape [B, tab_size]
#             init_tab = torch.randn((batch_size, self.tab_size),
#                                    device=device, dtype=torch.float32)
#
#         # Create the time schedule
#         t_vals = self.create_edm_timesteps(steps, device)  # shape [steps+1]
#
#         # We'll track the latents in double precision for stability
#         x_next = init_img.double() * t_vals[0]
#         y_next = init_tab.double() * t_vals[0]  # tab latents
#
#         for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
#             x_cur = x_next
#             y_cur = y_next
#             gamma = (
#                 min(self.S_churn / steps, np.sqrt(2) - 1)
#                 if (self.S_min <= t_cur <= self.S_max)
#                 else 0.0
#             )
#             t_hat = t_cur + gamma * t_cur
#
#             if gamma > 0:
#                 # add extra noise
#                 eps_img = self.S_noise * torch.randn_like(x_cur)
#                 eps_tab = self.S_noise * torch.randn_like(y_cur)
#                 x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * eps_img
#                 y_hat = y_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * eps_tab
#             else:
#                 x_hat = x_cur
#                 y_hat = y_cur
#
#             # 1) Euler step: predict Denoised
#             denoised_img, denoised_tab = self.model_step(
#                 x_in=x_hat, y_in=y_hat, sigma_in=t_hat, cfg=cfg
#             )
#             # d_cur
#             d_cur_img = (x_hat - denoised_img) / t_hat
#             d_cur_tab = (y_hat - denoised_tab) / t_hat
#
#             x_next = x_hat + (t_next - t_hat) * d_cur_img
#             y_next = y_hat + (t_next - t_hat) * d_cur_tab
#
#             # 2) 2nd order correction if we want it
#             if i < steps - 1:
#                 denoised_img2, denoised_tab2 = self.model_step(
#                     x_in=x_next, y_in=y_next, sigma_in=t_next, cfg=cfg
#                 )
#                 d_prime_img = (x_next - denoised_img2) / t_next
#                 d_prime_tab = (y_next - denoised_tab2) / t_next
#
#                 x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur_img + 0.5 * d_prime_img)
#                 y_next = y_hat + (t_next - t_hat) * (0.5 * d_cur_tab + 0.5 * d_prime_tab)
#
#         final_imgs = x_next.float()
#         final_tabs = y_next.float()
#         return final_imgs, final_tabs
#
#     def create_edm_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
#         """
#         Create a time schedule for EDM sampling steps.
#         Returns shape [steps+1].
#         """
#         step_indices = torch.arange(steps, dtype=torch.float64, device=device)
#         inv_rho = 1.0 / self.rho
#         t_values = (
#                            self.sigma_max ** inv_rho +
#                            step_indices / (steps - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)
#                    ) ** self.rho
#         # Append zero at the end
#         t_values = torch.cat([t_values, torch.zeros_like(t_values[:1])])
#         return t_values
#
#     def model_step(
#             self,
#             x_in: torch.Tensor,
#             y_in: torch.Tensor,
#             sigma_in: float,
#             cfg: float = 1.0
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Single forward pass that predicts the *clean* image + tab.
#         We do the usual EDM scalars (c_in, c_skip, c_out) and combine the model output.
#
#         Args:
#             x_in (torch.Tensor): Current noised image latents, shape [B, C, H, W].
#             y_in (torch.Tensor): Current noised tab latents, shape [B, T].
#             sigma_in (float): Current diffusion sigma value.
#             cfg (float): CFG scale.
#
#         Returns:
#             denoised_img (torch.Tensor), denoised_tab (torch.Tensor)
#         """
#         B = x_in.shape[0]
#         sigma_in = sigma_in.reshape(-1).float().to(x_in.device)  # shape [1], or [B]
#         if sigma_in.numel() == 1 and B > 1:
#             sigma_in = sigma_in.repeat(B)
#
#         # EDM scalars
#         sigma_data = self.sigma_data
#         c_in = 1.0 / (sigma_in ** 2 + sigma_data ** 2).sqrt()
#         c_skip = sigma_data ** 2 / (sigma_in ** 2 + sigma_data ** 2)
#         c_out = sigma_in * sigma_data / (sigma_in ** 2 + sigma_data ** 2).sqrt()
#
#         # Reshape for broadcast
#         c_in_img = c_in.view(-1, 1, 1, 1)  # shape [B, 1, 1, 1]
#         c_in_tab = c_in.view(-1, 1)  # shape [B, 1]
#         c_skip_img = c_skip.view(-1, 1, 1, 1)
#         c_out_img = c_out.view(-1, 1, 1, 1)
#
#         # Similarly for tab
#         c_skip_tab = c_skip.view(-1, 1)
#         c_out_tab = c_out.view(-1, 1)
#
#         # Time embedding for DiT
#         t_embed = (sigma_in.log() / 4.0)
#
#         # Forward pass through DiT
#         out = self.dit(
#             x_img=c_in_img * x_in.float(),
#             t=t_embed,
#             x_tab=c_in_tab * y_in.float(),
#             cfg=cfg,
#             mask_ratio=0.0
#         )
#         Fx = out["image_sample"].float()
#         Fy = out["tab_sample"].float()
#
#         # Denoised image
#         denoised_img = c_skip_img * x_in + c_out_img * Fx
#         # Denoised tab
#         denoised_tab = c_skip_tab * y_in + c_out_tab * Fy
#
#         return denoised_img, denoised_tab


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torchvision.utils import save_image

# If you use Hydra + OmegaConf, or some other config system:
# from omegaconf import DictConfig
# from hydra import initialize_config_dir, compose
# from easydict import EasyDict

# If you have these from your original snippet, keep them:
DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

class MultiModalDiffusion(nn.Module):
    """
    EDM-based multi-modal diffusion for images + tabular data,
    optionally using a weighting factor to make image loss more/less important.
    """

    def __init__(
        self,
        dit: nn.Module,
        train_mask_ratio: float = 0.0,
        latent_reg_weight: float = 0.0,
        image_loss_weight: float = 1.0,
        # Insert or pass your config here (edm_cfg, data_cfg, etc.)
        edm_config=None,
        image_size=32,
        latent_channels=4,
        tab_size=174,
    ):
        """
        Args:
            dit (nn.Module): The underlying diffusion model (e.g., DiT) that outputs
                             both image_sample and tab_sample.
            train_mask_ratio (float): Mask ratio for training (e.g., token dropping).
            latent_reg_weight (float): Weight for an optional latent regularization term.
            image_loss_weight (float): Weighting factor for the image loss (>= 1 => more emphasis).
            edm_config (dict): Contains EDM hyperparameters like sigma_min, sigma_max, etc.
            image_size (int): Spatial size for images (H == W here for simplicity).
            latent_channels (int): Number of latent channels if working in latent image space.
            tab_size (int): Dimensionality of tabular data.
        """
        super().__init__()
        self.dit = dit
        self.train_mask_ratio = train_mask_ratio
        self.latent_reg_weight = latent_reg_weight
        self.image_loss_weight = image_loss_weight

        # EDM config – set defaults or load from Hydra, etc.
        # E.g., edm_config might be EasyDict with keys:
        #   sigma_min, sigma_max, p_mean, p_std, sigma_data, num_steps, rho,
        #   s_churn, s_min, s_max, s_noise, dtype
        if edm_config is None:
            # Provide a default minimal config
            edm_config = {
                'sigma_min': 0.002,
                'sigma_max': 80,
                'P_mean': -1.2,
                'P_std': 1.2,
                'sigma_data': 0.5,
                'num_steps': 18,
                'rho': 7,
                'S_churn': 80,
                'S_min': 0.05,
                'S_max': 50,
                'S_noise': 1.003,
                'dtype': 'float32',
            }
        # Convert to a standard container if you like
        # self.edm_config = EasyDict(edm_config)  # if using EasyDict
        self.edm_config = edm_config

        # PyTorch dtype
        self._dtype = DTYPE_MAP[self.edm_config["dtype"]]

        # Image, table shape info (from your data config or pass in explicitly)
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.tab_size = tab_size

    def forward(self, images: torch.Tensor, table_data: torch.Tensor) -> dict[str, Tensor | float | Any]:
        """
        Forward pass to compute the EDM loss for BOTH images and tabular data.
        Both are noised (same sigma), then denoised by self.dit.
        Returns the total MSE-based diffusion loss with a weighting factor for images.
        """
        device = images.device
        B = images.shape[0]

        # 1) Sample sigma from log-normal distribution
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.edm_config["P_std"] + self.edm_config["P_mean"]).exp()  # [B,1,1,1]

        # 2) Compute MSE weight
        sigma_data = self.edm_config["sigma_data"]
        weight = ((sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2)  # [B,1,1,1]

        # 3) Noise both images and tabular data
        noise_img = torch.randn_like(images)  # same shape as images
        noised_imgs = images + sigma * noise_img

        noise_tab = torch.randn_like(table_data)  # [B, T]
        # reshape sigma to broadcast over [B,T] => [B,1]
        sigma_tab = sigma.view(B, 1)
        noised_tabs = table_data + sigma_tab * noise_tab

        # 4) EDM scaling
        # Flatten to [B,1,1,1] for images
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1.0 / (sigma_data**2 + sigma_in**2).sqrt()
        c_skip = sigma_data**2 / (sigma_in**2 + sigma_data**2)
        c_out = sigma_in * sigma_data / (sigma_in**2 + sigma_data**2).sqrt()
        t_embed = (sigma_in.log() / 4.0).squeeze()  # shape [B]

        # For tables, we can reuse c_in[:, 0, 0, 0] for 1D data => shape [B]
        c_in_tab = c_in[:, 0, 0, 0]   # shape [B]
        c_skip_tab = c_skip[:, 0, 0, 0]  # [B]
        c_out_tab = c_out[:, 0, 0, 0]    # [B]

        # 5) Forward pass through the DiT model
        # Our DiT should accept both x_img and x_tab (already scaled by c_in, c_in_tab).
        out = self.dit(
            x_img=c_in * noised_imgs,
            t=t_embed,
            x_tab=c_in_tab[:, None] * noised_tabs,  # broadcast shape [B,T]
            cfg=1.0,  # or pass user-specified CFG if you want
            mask_ratio=self.train_mask_ratio
        )
        pred_imgs = out["image_sample"]  # [B, C, H, W]
        pred_tabs = out["tab_sample"]    # [B, T]

        # 6) Combine => denoised predictions
        D_xn_imgs = c_skip * noised_imgs + c_out * pred_imgs
        D_xn_tabs = (c_skip_tab[:, None] * noised_tabs +
                     c_out_tab[:, None]  * pred_tabs)

        # 7) MSE losses
        # images: [B,C,H,W], weight: [B,1,1,1]
        loss_img = weight * ((D_xn_imgs - images) ** 2)
        image_loss = loss_img.mean(dim=[1, 2, 3]).mean()
        # tab: [B,T], weight: [B,1]
        loss_tab = weight.view(B, 1) * ((D_xn_tabs - table_data) ** 2)
        tab_loss = loss_tab.mean(dim=1).mean()

        # 8) Apply weighting factor for images
        image_loss = self.image_loss_weight * image_loss
        total_loss = image_loss + tab_loss

        # Optional latent regularization if you want
        if self.latent_reg_weight > 0:
            real_mean = images.mean(dim=(0, 2, 3), keepdim=True)
            real_std = images.std(dim=(0, 2, 3), keepdim=True)
            pred_mean = pred_imgs.mean(dim=(0, 2, 3), keepdim=True)
            pred_std = pred_imgs.std(dim=(0, 2, 3), keepdim=True)
            mean_loss = F.mse_loss(pred_mean, real_mean)
            std_loss = F.mse_loss(pred_std, real_std)
            reg_loss = mean_loss + std_loss
            total_loss = total_loss + self.latent_reg_weight * reg_loss

        return {"loss": total_loss, "loss_img": image_loss, "loss_tab": tab_loss}

    @torch.no_grad()
    def _sample_edm(
        self,
        batch_size: int,
        cfg: float = 1.0,
        steps: Optional[int] = None,
        height: int = 32,
        width: int = 32,
        tab_size: int = 174,
        device: str = "cuda"
    ):
        """
        Jointly sample images and tab data from noise, using EDM steps (Euler or Heun).
        For simplicity, we use the same schedule for both images and tab.

        Returns (final_images, final_tabs), each shaped [B, ...].
        """
        self.eval()
        steps = steps or self.edm_config["num_steps"]
        sigma_data = self.edm_config["sigma_data"]
        c = self.latent_channels

        # Start from noise: images => shape [B, C, H, W], tabs => [B, T]
        x_img = torch.randn((batch_size, c, height, width), device=device, dtype=torch.float32)
        x_tab = torch.randn((batch_size, tab_size), device=device, dtype=torch.float32)

        # time schedule
        t_vals = self.create_edm_timesteps(steps, device)
        # Convert x -> x*sigma for the first step
        x_img_next = x_img.double() * t_vals[0]
        x_tab_next = x_tab.double() * t_vals[0]  # same factor

        def model_forward(img_in, tab_in, t_sigma, cfg_val):
            """
            Single forward of the DiT to get denoised predictions for images & tables.
            """
            B = img_in.shape[0]
            # shape: t_sigma => [B] or [1], make it [B,1,1,1] for images
            if t_sigma.numel() == 1:  # broadcast if needed
                t_sigma = t_sigma.expand(B)
            t_sigma_img = t_sigma.view(B, 1, 1, 1).float()

            # EDM scaling
            c_in = 1.0 / (t_sigma_img**2 + sigma_data**2).sqrt()
            c_skip = sigma_data**2 / (t_sigma_img**2 + sigma_data**2)
            c_out = t_sigma_img * sigma_data / (t_sigma_img**2 + sigma_data**2).sqrt()

            t_embed = (t_sigma_img.log() / 4.0).reshape(B)  # shape [B]

            # scale inputs
            x_in = c_in * img_in.float()
            tab_in_scaled = c_in[:, 0, 0, 0].unsqueeze(-1) * tab_in.float()

            # pass through the model
            out = self.dit(
                x_img=x_in,
                t=t_embed,
                x_tab=tab_in_scaled,
                cfg=cfg_val,
                mask_ratio=0.0
            )
            pred_imgs = out["image_sample"].float()
            pred_tabs = out["tab_sample"].float()

            # combine => denoised
            denoised_imgs = c_skip * img_in + c_out * pred_imgs
            denoised_tabs = (c_skip[:, 0, 0, 0].unsqueeze(-1) * tab_in +
                             c_out[:, 0, 0, 0].unsqueeze(-1)  * pred_tabs)
            return denoised_imgs, denoised_tabs

        # Main EDM loop
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_img_cur = x_img_next
            x_tab_cur = x_tab_next
            gamma = (
                min(self.edm_config["S_churn"] / steps, np.sqrt(2) - 1)
                if (self.edm_config["S_min"] <= t_cur <= self.edm_config["S_max"])
                else 0.0
            )
            t_hat = t_cur + gamma * t_cur

            if gamma > 0:
                # sample new noise
                eps_img = torch.randn_like(x_img_cur)
                eps_tab = torch.randn_like(x_tab_cur)
                # Expand t_hat to match shape
                # shape for images => [B,1,1,1]; for tab => [B,1]
                # t_hat is a single scalar (0-dim). Convert it to a shape [1,1,1,1] for images:
                t_hat_img = t_hat.view(1, 1, 1, 1)
                x_img_hat = x_img_cur + (t_hat_img ** 2 - t_cur ** 2).sqrt() * self.edm_config["S_noise"] * eps_img

                # For tab data, create shape [batch_size, 1] by expanding or repeating:
                t_hat_tab = t_hat.expand(batch_size, 1)
                x_tab_hat = x_tab_cur + (t_hat_tab**2 - t_cur**2).sqrt() * self.edm_config["S_noise"] * eps_tab
            else:
                x_img_hat = x_img_cur
                x_tab_hat = x_tab_cur

            # Euler step
            denoised_imgs, denoised_tabs = model_forward(x_img_hat, x_tab_hat, t_hat, cfg)
            denoised_imgs = denoised_imgs.double()
            denoised_tabs = denoised_tabs.double()

            d_cur_img = (x_img_hat - denoised_imgs) / t_hat
            d_cur_tab = (x_tab_hat - denoised_tabs) / t_hat

            x_img_next = x_img_hat + (t_next - t_hat) * d_cur_img
            x_tab_next = x_tab_hat + (t_next - t_hat) * d_cur_tab

            # 2nd order correction
            if i < steps - 1:
                denoised_imgs2, denoised_tabs2 = model_forward(x_img_next, x_tab_next, t_next, cfg)
                denoised_imgs2 = denoised_imgs2.double()
                denoised_tabs2 = denoised_tabs2.double()

                d_prime_img = (x_img_next - denoised_imgs2) / t_next
                d_prime_tab = (x_tab_next - denoised_tabs2) / t_next

                x_img_next = x_img_hat + (t_next - t_hat) * (0.5 * d_cur_img + 0.5 * d_prime_img)
                x_tab_next = x_tab_hat + (t_next - t_hat) * (0.5 * d_cur_tab + 0.5 * d_prime_tab)

        final_imgs = x_img_next.float()
        final_tabs = x_tab_next.float()

        return final_imgs, final_tabs

    def create_edm_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        """
        Create a time schedule for EDM sampling steps. Returns shape [steps+1].
        """
        rho = self.edm_config["rho"]
        sigma_min = self.edm_config["sigma_min"]
        sigma_max = self.edm_config["sigma_max"]

        step_indices = torch.arange(steps, dtype=torch.float64, device=device)
        inv_rho = 1.0 / rho

        t_values = (
            sigma_max**inv_rho +
            step_indices / (steps - 1) * (sigma_min**inv_rho - sigma_max**inv_rho)
        )**rho

        # Append zero for final step
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

    # Suppose we have:
    from torch import optim, Tensor

    # 1) A "MultiModalDiT" instance that expects:
    #    model(x_img, x_tab, time_scalar) -> {"img_out":..., "tab_out":...}
    from models.dit.dit_multimodal_add import MultiModalDiT

    B = 4
    H, W = 32, 32
    in_chans = 4
    d_tab = 174

    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth = 16

    model = MultiModalDiT(
        img_size=64,
        patch_size=4,
        in_channels=4,
        dim=256,
        depth=depth,
        head_dim=32,
        multiple_of=64,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_use_moe=False,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        norm_eps=1e-6,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
        d_numerical=174,
        out_dim_tab=174,
        use_cls_tab=False,
        time_emb_dim=256
    )

    # 2) Our EDM diffuser
    diffuser = MultiModalDiffusion(model)

    # 3) Some example training batch
    x_img = torch.randn(B, in_chans, H, W)
    x_tab = torch.randn(B, d_tab)

    # 4) forward => get loss
    loss_dict = diffuser.forward(x_img, x_tab)
    print(f"total loss={loss_dict["loss"]}, img loss={loss_dict["loss_img"]}, tab loss={loss_dict["loss_tab"]}")

    # 5) do a gradient step
    opt = optim.Adam(diffuser.parameters(), lr=1e-4)
    loss_dict["loss"].backward()
    opt.step()
