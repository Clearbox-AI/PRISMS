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
from utils.data import infinite_loader, DataLabel
from data.data_bucket import DataBucket
from models.dit.dit_multimodal import load_dit
from models.vae.vae import decode_latents, load_vae
from data.loader import load_training_data
from utils.ddp import (is_dist_available_and_initialized, get_world_size, get_rank, all_gather_tensor,
                       all_gather_object, setup_distributed)
from enums.generation import SourceType

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

class MultiModalDiffusion(nn.Module):
    """
    EDM-based multi-modal diffusion for images + tabular data, optionally using a VAE.
    This module implements a training forward pass (EDM loss) and a sampling procedure.
    """

    def __init__(
        self,
        dit: nn.Module,
        train_mask_ratio: float = 0.0,
        latent_reg_weight: float = 0.0
    ) -> None:
        """
        Initializes the MultiModalDiffusion model using Hydra configs.

        Args:
            dit (nn.Module): The underlying diffusion model (e.g., DiT).
            train_mask_ratio (float): Mask ratio for training (e.g., token dropping).
            latent_reg_weight (float): Weight for an optional latent regularization term.
        """
        super().__init__()
        self.dit = dit
        self.train_mask_ratio = train_mask_ratio
        self.latent_reg_weight = latent_reg_weight

        from utils.configurations import set_project_root
        set_project_root()

        with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
            edm_cfg = compose(config_name="diffusion")

        with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
            data_cfg = compose(config_name="nacc")

        # Extract base EDM parameters from the config
        EDM_params = edm_cfg.EDM
        self.edm_config = EasyDict({
            'sigma_min': EDM_params.sigma_min,
            'sigma_max': EDM_params.sigma_max,
            'P_mean': EDM_params.p_mean,
            'P_std': EDM_params.p_std,
            'sigma_data': EDM_params.sigma_data,
            'num_steps': EDM_params.num_steps,
            'rho': EDM_params.rho,
            'S_churn': EDM_params.s_churn,
            'S_min': EDM_params.s_min,
            'S_max': EDM_params.s_max,
            'S_noise': EDM_params.s_noise,
        })

        self._dtype = DTYPE_MAP[EDM_params.dtype]

        # From data config, get shapes
        self.image_size = data_cfg.data.image_size
        self.latent_channels = data_cfg.data.latent_channels

        self.image_pixel_image_width = data_cfg.data.image_width
        self.image_pixel_image_height = data_cfg.data.image_height
        self.image_pixel_channels = data_cfg.data.target_channels

        self.tab_size = data_cfg.data.tab_size


    def forward(self, images: torch.Tensor, table_data: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute the EDM loss.

        Args:
            images (torch.Tensor): The real image tensors of shape (B, C, H, W).
            table_data (torch.Tensor): Tabular data of shape (B, T) or similar.

        Returns:
            total_loss (torch.Tensor)
        """
        device = images.device
        B = images.shape[0]

        # 1) Sample sigma from a log-normal distribution
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()

        # 2) Compute the weight
        weight = ((sigma**2 + self.edm_config.sigma_data**2) / (sigma * self.edm_config.sigma_data)**2)

        # 3) Add noise to input
        noise = torch.randn_like(images)
        noised_input = images + sigma * noise

        # 4) EDM scaling
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1.0 / (self.edm_config.sigma_data**2 + sigma_in**2).sqrt()
        c_skip = self.edm_config.sigma_data**2 / (sigma_in**2 + self.edm_config.sigma_data**2)
        c_out = sigma_in * self.edm_config.sigma_data / (sigma_in**2 + self.edm_config.sigma_data**2).sqrt()
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
    def _sample_edm(
            self,
            table_data: torch.Tensor,
            batch_size: int,
            cfg: float = 1.0,
            steps: Optional[int] = None,
            height: int = 32,
            width: int = 32,
            device: str = 'cuda',
            save_path: Optional[str] = None
    ) -> torch.Tensor:
        self.eval()
        steps = steps or self.edm_config.num_steps
        c = self.latent_channels

        x = torch.randn((batch_size, c, height, width), device=device, dtype=torch.float32)
        t_vals = self.create_edm_timesteps(steps, device)
        x_next = x.double() * t_vals[0]

        def model_forward(x_in, t_sigma, tab_data, cfg_val):
            B = x_in.shape[0]
            sigma_in = t_sigma.reshape(-1, 1, 1, 1).float()

            c_in = 1.0 / (sigma_in ** 2 + self.edm_config.sigma_data ** 2).sqrt()
            c_skip = self.edm_config.sigma_data ** 2 / (sigma_in ** 2 + self.edm_config.sigma_data ** 2)
            c_out = sigma_in * self.edm_config.sigma_data / (sigma_in ** 2 + self.edm_config.sigma_data ** 2).sqrt()

            # Force the time embedding to match batch size
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

        # Main EDM loop
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_cur = x_next
            gamma = (
                min(self.edm_config.S_churn / steps, np.sqrt(2) - 1)
                if (self.edm_config.S_min <= t_cur <= self.edm_config.S_max)
                else 0.0
            )
            t_hat = t_cur + gamma * t_cur
            if gamma > 0:
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.edm_config.S_noise * torch.randn_like(x_cur)
            else:
                x_hat = x_cur

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

        # Optional save
        if save_path and is_main_process():
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            latents_for_vis = (final_latents - final_latents.min()) / (
                    final_latents.max() - final_latents.min() + 1e-7
            )
            save_image(latents_for_vis, save_path, nrow=int(batch_size ** 0.5))

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
        inv_rho = 1.0 / self.edm_config.rho
        t_values = (
            self.edm_config.sigma_max**inv_rho +
            step_indices / (steps - 1) * (self.edm_config.sigma_min**inv_rho - self.edm_config.sigma_max**inv_rho)
        )**self.edm_config.rho
        # Append zero for the final step
        t_values = torch.cat([t_values, torch.zeros_like(t_values[:1])])
        return t_values

    @torch.no_grad()
    def generate_samples(
            self,
            n_samples: int,
            data_bucket: DataBucket,
            vae: nn.Module,
            batch_size: int = 4,
            cfg: float = 1.0,
            steps: Optional[int] = None,
            device: str = "cuda",
    ) -> Tuple[DataBucket, Dict[Any, List[int]], DataBucket]:
        """
        Generate images conditioned on tabular data from a DataBucket, and also return
        a DataBucket containing the input tabular data used in the generation.

        Returns:
          - A DataBucket containing pixel-space images (DataLabel.IMAGE).
          - A dict mapping from condition key -> list of sample indices.
          - A DataBucket containing the tabular data used for generation (DataLabel.TAB).

        Args:
            n_samples (int): Total number of samples to generate.
            data_bucket (DataBucket): Must contain tabular data (label=TAB or BOTH).
            vae (nn.Module): The VAE used to decode the final latents into pixel space.
            batch_size (int): Batch size for each generation chunk.
            cfg (float): Classifier-Free Guidance scale.
            steps (int, optional): Number of EDM steps. Defaults to self.edm_config.num_steps.
            device (str): The device to run on.

        Returns:
            (DataBucket, Dict[Any, List[int]], DataBucket):
              - A DataBucket of shape (#samples, [C,H,W]) containing decoded images.
              - A dictionary mapping from condition key -> list of sample indices.
              - A DataBucket of shape (#samples, <tabular_dim>) containing the conditioning data.
        """

        # 1) Check that data_bucket is labeled for tabular usage
        if data_bucket.label not in (DataLabel.TAB, DataLabel.BOTH):
            raise ValueError(
                "DataBucket must have label=TAB or BOTH for table conditioning."
            )

        # 2) Create a DataLoader from the data_bucket
        loader = data_bucket.get_dataloader(batch_size=batch_size, shuffle=True)
        cond_iter = infinite_loader(loader)

        # Some config details
        if steps is None:
            steps = self.edm_config.num_steps  # example usage
        dtype = self._dtype  # e.g., torch.float32

        cond_mapping: Dict[Any, List[int]] = {}
        all_decoded_imgs = []
        all_tab_data = []

        total_generated = 0
        global_index = 0

        # 3) Loop until we generate n_samples
        while total_generated < n_samples:
            batch = next(cond_iter)  # get next batch from loader

            # batch is a dict with "image", "tabular", "dir"
            # We only need tabular data for conditioning
            tab_data = batch["tabular"]  # shape [B, ...], if label=TAB or BOTH
            current_bsz = tab_data.shape[0]

            # If generating would exceed n_samples, clip the batch
            if total_generated + current_bsz > n_samples:
                current_bsz = n_samples - total_generated
                tab_data = tab_data[:current_bsz]

            # Move tab_data to device, and cast dtype if needed
            tab_data = tab_data.to(device=device, dtype=dtype)

            # 4) Run EDM sampling with the tab_data
            final_latents = self._sample_edm(
                table_data=tab_data,
                batch_size=current_bsz,
                cfg=cfg,
                steps=steps,
                device=device,
            )

            # 5) Decode pixel-space images with the VAE
            decoded_imgs = decode_latents(vae, final_latents, vae.config.scaling_factor)
            # decoded_imgs shape: [current_bsz, C, H, W]

            # 6) Accumulate
            all_decoded_imgs.append(decoded_imgs.cpu())  # store on CPU
            all_tab_data.append(tab_data.cpu())

            # 7) Cond_mapping: we use "dir" or some ID if available
            if "dir" in batch:
                for i in range(current_bsz):
                    cond_key = batch["dir"][i]
                    if cond_key not in cond_mapping:
                        cond_mapping[cond_key] = []
                    cond_mapping[cond_key].append(global_index + i)
            else:
                # if no 'dir', fallback to an integer index or something
                for i in range(current_bsz):
                    ck = f"cond_{global_index + i}"
                    if ck not in cond_mapping:
                        cond_mapping[ck] = []
                    cond_mapping[ck].append(global_index + i)

            total_generated += current_bsz
            global_index += current_bsz

        # 8) Merge all decoded images / tab data
        # final_images shape: [n_samples, C, H, W]
        final_images = torch.cat(all_decoded_imgs, dim=0)[:n_samples]
        final_tab_data = torch.cat(all_tab_data, dim=0)[:n_samples]

        # 9) Build a DataBucket for the generated images
        # We'll store them as a list of Tensors. Each item is shape [C,H,W].
        img_list = [final_images[i] for i in range(n_samples)]
        result_bucket = DataBucket(
            source_type=SourceType.LIST,
            label=DataLabel.IMAGE,
            data_list=img_list,
            metadata={"is_generated": True}
        )

        # 10) Build a DataBucket for the tab data used in generation
        # i.e. the "conditioning" table that was actually used.
        tab_list = [final_tab_data[i] for i in range(n_samples)]
        input_bucket = DataBucket(
            source_type=SourceType.LIST,
            label=DataLabel.TAB,
            data_list=tab_list
        )

        return result_bucket, cond_mapping, input_bucket