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
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple, Union

from utils.ddp import is_main_process
from utils.configurations import apply_overrides
from models.dit.dit_multimodal import load_dit
import random
from torch.utils.data import DataLoader


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

    def forward(self, images: torch.Tensor, table_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Add noise to images
        noise_img = torch.randn_like(images)
        noised_images = images + sigma_img * noise_img

        sigma_tab = sigma_img.view(B, 1)  # => shape (B,1)
        noise_tab = torch.randn_like(table_data)
        noised_table = table_data + sigma_tab * noise_tab

        # EDM weighting
        weight_img = ((sigma_img ** 2 + self.sigma_data ** 2) / (sigma_img * self.sigma_data) ** 2)
        weight_tab = ((sigma_tab ** 2 + self.sigma_data ** 2) / (sigma_tab * self.sigma_data) ** 2)

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
        out = self.dit.forward(
            x_noisy_img=c_in_img * noised_images,  # scaled/noised image
            x_noisy_tab=c_in_tab * noised_table,  # scaled/noised tab
            t=t_img,  # same shape (B,)
            mask_ratio=self.train_mask_ratio
        )

        # The model must return out["image_sample"] & out["table_sample"]
        pred_img = out["image_sample"]  # shape (B, C, H, W)
        pred_tab = out["table_sample"]  # shape (B, T)

        # ------------------------------------------------
        # 4) "Denoised" final predictions via EDM formula
        # ------------------------------------------------
        denoised_img = c_skip_img * noised_images + c_out_img * pred_img
        denoised_tab = c_skip_tab * noised_table + c_out_tab * pred_tab

        # ------------------------------------------------
        # 5) MSE Loss vs. ground truth
        # ------------------------------------------------
        # Image loss
        img_loss_val = weight_img * ((denoised_img - images) ** 2)
        image_loss = img_loss_val.mean(dim=[1, 2, 3]).mean()

        # # Optional latent regularization
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
        total_loss = image_loss + 1 * table_loss

        return total_loss, image_loss, table_loss

    @torch.no_grad()
    def sample(
            self,
            condition_modality: str = 'none', #'none', 'image', or 'tab'
            condition_data: Optional[torch.Tensor] = None,
            partial_condition: bool = False,
            partial_noise_factor: float = 0.0,
            cfg: float = 1.0,
            steps: Optional[int] = None,
            height: int = 64,
            width: int = 64,
            tab_n: int = 10,
            batch_size: int = 4,
            device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A unified sample method that always passes x_img and x_tab:
         - If 'none' => unconditional for both
         - If 'image' => condition on given image (real or partial noise),
                         unconditional for tab
         - If 'tab' => condition on given tab data (real or partial noise),
                       unconditional for image

        partial_condition => whether to add small noise to the 'condition_data'
        partial_noise_factor => how much noise

        Returns:
          final_imgs: (B, C, H, W)
          final_tabs: (B, T_tab)
        """

        self.eval()
        steps = steps or self.num_steps

        # For demonstration, assume:
        c = self.dit.in_channels  # e.g. 4

        # 1) Prepare latents for image and tab
        #    We'll always create 'x_img' and 'x_tab' with shape:
        #     x_img: (batch_size, c, H, W)
        #     x_tab: (batch_size, tab_n)
        #    Then fill them according to 'condition_modality'.
        x_img = torch.randn(batch_size, c, height, width, device=device, dtype=torch.float64)  # unconditional by default
        x_tab = torch.randn(batch_size, tab_n, device=device, dtype=torch.float64)  # unconditional by default

        if condition_modality == 'image':
            # => fix or partially fix the image data
            if condition_data is not None:
                # shape check: condition_data => (B, c, H, W)
                real_img = condition_data.to(device, dtype=torch.float64)
                if partial_condition and partial_noise_factor > 0.0:
                    # partial condition => add random noise scaled by partial_noise_factor
                    noise = partial_noise_factor * torch.randn_like(real_img)
                    x_img = real_img + noise
                else:
                    # pinned => no noise
                    x_img = real_img
            # tab remains random => unconditional
            # We'll pass a flag to model_forward_edm to treat the tab as unconditional
            tab_is_uncond = True
            img_is_uncond = False  # means "we are not noising the image in model_forward_edm"

        elif condition_modality == 'tab':
            # => the user wants to fix or partially fix the tab data
            if condition_data is not None:
                # shape => (B, T_tab)
                real_tab = condition_data.to(device, dtype=torch.float64)
                if partial_condition and partial_noise_factor > 0.0:
                    noise = partial_noise_factor * torch.randn_like(real_tab)
                    x_tab = real_tab + noise
                else:
                    x_tab = real_tab
            # image remains random => unconditional
            img_is_uncond = True
            tab_is_uncond = False  # pinned or partial

        else:  # condition_modality == 'none'
            # unconditional for both => do nothing, x_img & x_tab are random
            img_is_uncond = True
            tab_is_uncond = True

        # 2) Create the time schedule
        t_vals = self.create_edm_timesteps(steps, device=device)  # shape => (steps+1,)
        # multiply latents by largest sigma => t_vals[0]
        x_img *= t_vals[0]
        x_tab *= t_vals[0]

        # We'll store them in double precision for numeric stability
        x_img_next = x_img
        x_tab_next = x_tab

        # ---------------------------------------------------------
        # 3) Sampler loop with Heun or Euler steps
        # ---------------------------------------------------------
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):

            # s_churn: add extra noise if s_min <= t_cur <= s_max
            gamma = (min(self.s_churn / steps, (2 ** 0.5) - 1)
                     if (self.s_min <= t_cur <= self.s_max) else 0.0)
            t_hat = t_cur + gamma * t_cur

            # -----------------------------------------------------
            # 3a) Add "churn" noise
            # -----------------------------------------------------
            x_i_hat = x_img_next + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * torch.randn_like(x_img_next)
            x_t_hat = x_tab_next + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * torch.randn_like(x_tab_next) \
                if (tab_is_uncond) else x_tab_next
            # if 'tab_is_uncond' => we treat tab the same as image => add churn noise
            # if tab is pinned => skip

            # if 'img_is_uncond' => we treat image the same as above,
            # but in this snippet we always do it for x_img => we can do a little if check:
            if not img_is_uncond:
                # pinned => no churn noise for image
                x_i_hat = x_img_next

            # -----------------------------------------------------
            # 3b) 1st pass (Euler)
            # -----------------------------------------------------
            # We do an EDM forward => denoised => get derivative
            x_img_denoised, x_tab_denoised = self.model_forward_edm(
                x_i_hat, x_t_hat, t_hat, cfg,
                img_is_uncond=img_is_uncond,
                tab_is_uncond=tab_is_uncond
            )
            d_cur_img = (x_i_hat - x_img_denoised) / t_hat
            x_img_next = x_i_hat + (t_next - t_hat) * d_cur_img

            d_cur_tab = None
            if x_tab_denoised is not None:
                d_cur_tab = (x_t_hat - x_tab_denoised) / t_hat
                x_tab_next = x_t_hat + (t_next - t_hat) * d_cur_tab

            # -----------------------------------------------------
            # 3c) 2nd pass (Heun correction)
            # -----------------------------------------------------
            if i < steps - 1:
                x_img_denoised2, x_tab_denoised2 = self.model_forward_edm(
                    x_img_next, x_tab_next, t_next, cfg,
                    img_is_uncond=img_is_uncond,
                    tab_is_uncond=tab_is_uncond
                )
                d_prime_img = (x_img_next - x_img_denoised2) / t_next
                x_img_next = x_i_hat + (t_next - t_hat) * 0.5 * (d_cur_img + d_prime_img)

                if x_tab_denoised2 is not None and d_cur_tab is not None:
                    d_prime_tab = (x_tab_next - x_tab_denoised2) / t_next
                    x_tab_next = x_t_hat + (t_next - t_hat) * 0.5 * (d_cur_tab + d_prime_tab)

        final_imgs = x_img_next.float()
        final_tabs = x_tab_next.float()
        return final_imgs, final_tabs

    def model_forward_edm(
            self,
            x_img_in: torch.Tensor,
            x_tab_in: torch.Tensor,
            sigma_val: float,
            cfg_val: float, # TODO: re-add implementation to dit model
            img_is_uncond: bool = True,
            tab_is_uncond: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of EDM scaling + forward pass in our new joint DiT,
        but with separate booleans for image uncond vs. tab uncond.

        If 'img_is_uncond=True', we treat x_img_in with a typical sigma approach,
        else pinned => sigma=0 for the image.

        Same for tab_is_uncond.
        """

        B = x_img_in.shape[0]
        sigma_img = torch.full((B, 1, 1, 1), sigma_val, device=x_img_in.device, dtype=torch.float32)

        # image EDM factors:
        if img_is_uncond:
            c_in = 1.0 / (sigma_img ** 2 + self.sigma_data ** 2).sqrt()
            c_skip = self.sigma_data ** 2 / (sigma_img ** 2 + self.sigma_data ** 2)
            c_out = sigma_img * self.sigma_data / (sigma_img ** 2 + self.sigma_data ** 2).sqrt()
        else:
            # pinned => sigma=0
            c_in = 1.0
            c_skip = 1.0
            c_out = 0.0

        t_embed = (sigma_img.log() / 4.0).view(-1)  # shape (B,)

        # tab EDM factors
        if tab_is_uncond:
            sigma_tab = sigma_img.view(B, 1)
            c_in_tab = 1.0 / (sigma_tab ** 2 + self.sigma_data ** 2).sqrt()
            c_skip_tab = self.sigma_data ** 2 / (sigma_tab ** 2 + self.sigma_data ** 2)
            c_out_tab = sigma_tab * self.sigma_data / (sigma_tab ** 2 + self.sigma_data ** 2).sqrt()
        else:
            # pinned => sigma=0
            c_in_tab = 1.0
            c_skip_tab = 1.0
            c_out_tab = 0.0

        x_img_scaled = c_in * x_img_in.float()
        x_tab_scaled = c_in_tab * x_tab_in.float()

        out = self.dit.forward(
            x_noisy_img=x_img_scaled,
            x_noisy_tab=x_tab_scaled,
            t=t_embed,
            mask_ratio=0.0
        )
        Fx_img = out["image_sample"].float()
        Fx_tab = out["table_sample"].float()

        denoised_img = c_skip * x_img_in + c_out * Fx_img
        denoised_tab = c_skip_tab * x_tab_in + c_out_tab * Fx_tab

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


    def generate_samples(
            self,
            n_samples: int,
            device: str,
            condition_modality: str = "none",  # 'none', 'image', or 'tab'
            partial_condition: bool = False,  # Meaningful only if condition_modality != 'none'
            partial_noise_factor: float = 0.0,  # Used only with partial_condition
            cfg_scale: float = 1.0,  # Classifier-Free Guidance scaling
            height: int = 64,
            width: int = 64,
            n_tab: int = 10,
            batch_size: int = 4,
            dataloader: Optional[DataLoader] = None,  # If provided, used for conditional generation
            condition_data: Optional[torch.Tensor] = None,  # External data for conditional generation
            patient_dirs: Optional[List[str]] = None  # Optional mapping to maintain patient-to-sample indices
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, List[int]]]]:
        """
        Generate a specified number of samples, with optional conditional guidance from a dataloader
        or an external tensor. Also supports unconditional generation when 'condition_modality' is 'none'.

        Args:
            n_samples (int): Total number of samples to generate.
            device (str): Device to use (e.g., 'cuda' or 'cpu').
            condition_modality (str):
                'none'   -> Unconditional generation.
                'image'  -> Condition on image data.
                'tab'    -> Condition on tabular data.
            partial_condition (bool):
                If True, partial noise is added to conditioning. Only relevant if condition_modality != 'none'.
            partial_noise_factor (float):
                Level of noise applied for partial conditioning. Meaningful only if partial_condition is True.
            cfg_scale (float):
                Classifier-Free Guidance scale.
            height (int): Height of generated images.
            width (int): Width of generated images.
            n_tab (int): Number of tabular features to generate (or condition on).
            batch_size (int): Batch size used during generation.
            dataloader (Optional[DataLoader]):
                Dataloader for conditional generation. If provided, `condition_data` must be None.
            condition_data (Optional[torch.Tensor]):
                External data for conditional generation. If provided, `dataloader` must be None.
            patient_dirs (Optional[List[str]]):
                If provided during conditional generation, a mapping of {patient_dir: [sample_indices]}
                is returned to indicate which generated samples correspond to each patient_dir.

        Returns:
            (torch.Tensor, torch.Tensor, Optional[Dict[str, List[int]]]):
                - Generated images of shape (n_samples, C, H, W).
                - Generated tabular data of shape (n_samples, n_tab).
                - Optional dictionary mapping patient directories to generated sample indices.
                  Returned only if patient information is provided in the conditioning data.
        """

        # -------------------------------------------------------------------------
        # 0. Preliminary checks and setup
        # -------------------------------------------------------------------------
        if dataloader is not None and condition_data is not None:
            raise ValueError(
                "You cannot provide both 'dataloader' and 'condition_data'. "
                "Please choose one conditional source."
            )

        # partial_condition is only relevant if there's a condition (image or tab)
        use_partial_condition = partial_condition and (condition_modality in ["image", "tab"])

        # Prepare accumulators
        all_imgs = []
        all_tabs = []
        sample_mapping: Dict[str, List[int]] = {}
        total_generated = 0
        sample_index = 0

        # -------------------------------------------------------------------------
        # 1. If external condition_data is provided
        # -------------------------------------------------------------------------
        if condition_data is not None:
            print("[INFO] Generating samples using external 'condition_data'.")

            # We'll keep picking random slices of 'condition_data' until we reach n_samples
            cond_size = condition_data.size(0)
            use_patient_dirs = (patient_dirs is not None)
            while total_generated < n_samples:
                current_bsz = min(batch_size, n_samples - total_generated)

                # Randomly pick from condition_data
                if cond_size > current_bsz:
                    idx = torch.randint(0, cond_size, (current_bsz,))
                    batch_condition_data = condition_data[idx].to(device)
                    # Map chosen indices to patient_dirs if available
                    batch_patient_dirs = (
                        [patient_dirs[i] for i in idx] if use_patient_dirs else [None] * current_bsz
                    )
                else:
                    # If the dataset is smaller than the batch, just take all
                    # (this can repeat multiple times until n_samples is reached)
                    batch_condition_data = condition_data.to(device)
                    batch_patient_dirs = patient_dirs if use_patient_dirs else [None] * cond_size

                # Generate a batch of samples
                imgs_batch, tabs_batch = self.sample(
                    batch_size=current_bsz,
                    condition_modality=condition_modality,
                    condition_data=batch_condition_data,
                    partial_condition=use_partial_condition,
                    partial_noise_factor=partial_noise_factor,
                    cfg=cfg_scale,
                    height=height,
                    width=width,
                    n_tab=n_tab,
                    device=device,
                )

                all_imgs.append(imgs_batch)
                all_tabs.append(tabs_batch)

                # Update the mapping if patient directories were provided
                if use_patient_dirs:
                    for i, pd in enumerate(batch_patient_dirs):
                        if pd is not None:  # ignoring if it somehow doesn't exist
                            if pd not in sample_mapping:
                                sample_mapping[pd] = []
                            sample_mapping[pd].append(sample_index + i)

                # Update counters
                total_generated += current_bsz
                sample_index += current_bsz

        # -------------------------------------------------------------------------
        # 2. If a dataloader is provided for conditional generation
        # -------------------------------------------------------------------------
        elif dataloader is not None:
            print("[INFO] Generating samples using a 'dataloader' for conditional generation.")

            # We assume the dataloader is (ideally) shuffled externally to ensure randomness
            data_iter = iter(dataloader)

            while total_generated < n_samples:
                current_bsz = min(batch_size, n_samples - total_generated)

                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Restart the dataloader if exhausted
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                # Determine the relevant conditional data
                if condition_modality == 'tab' and 'tab' in batch:
                    batch_condition_data = batch['tab'][:current_bsz].to(device)
                elif condition_modality == 'image' and 'image' in batch:
                    batch_condition_data = batch['image'][:current_bsz].to(device)
                else:
                    batch_condition_data = None

                # Pull out patient directories if they exist; otherwise fill with Nones
                batch_patient_dirs = batch.get('dir', [None] * current_bsz)

                # Generate samples for this batch
                imgs_batch, tabs_batch = self.sample(
                    batch_size=current_bsz,
                    condition_modality=condition_modality,
                    condition_data=batch_condition_data,
                    partial_condition=use_partial_condition,
                    partial_noise_factor=partial_noise_factor,
                    cfg=cfg_scale,
                    height=height,
                    width=width,
                    n_tab=n_tab,
                    device=device,
                )

                all_imgs.append(imgs_batch)
                all_tabs.append(tabs_batch)

                # Update the mapping
                for i, pd in enumerate(batch_patient_dirs):
                    if pd not in sample_mapping:
                        sample_mapping[pd] = []
                    sample_mapping[pd].append(sample_index + i)

                # Update counters
                total_generated += current_bsz
                sample_index += current_bsz

        # -------------------------------------------------------------------------
        # 3. Otherwise, unconditional generation
        # -------------------------------------------------------------------------
        else:
            print("[INFO] Generating samples in an unconditional manner (condition_modality='none').")

            while total_generated < n_samples:
                current_bsz = min(batch_size, n_samples - total_generated)

                imgs_batch, tabs_batch = self.sample(
                    batch_size=current_bsz,
                    condition_modality="none",  # Force no conditioning
                    condition_data=None,
                    partial_condition=False,  # partial_condition irrelevant here
                    partial_noise_factor=0.0,  # irrelevant
                    cfg=cfg_scale,
                    height=height,
                    width=width,
                    n_tab=n_tab,
                    device=device,
                )

                all_imgs.append(imgs_batch)
                all_tabs.append(tabs_batch)

                total_generated += current_bsz
                sample_index += current_bsz

        # -------------------------------------------------------------------------
        # 4. Final concatenation and return
        # -------------------------------------------------------------------------
        final_imgs = torch.cat(all_imgs, dim=0)[:n_samples]
        final_tabs = torch.cat(all_tabs, dim=0)[:n_samples]

        # If we have a non-empty mapping, return it; otherwise return None
        if sample_mapping:
            return final_imgs, final_tabs, sample_mapping
        else:
            return final_imgs, final_tabs, None


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