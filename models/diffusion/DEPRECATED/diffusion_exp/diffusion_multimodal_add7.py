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
from models.dit.dit_multimodal import load_dit
from models.vae.vae import decode_latents, load_vae
from data.loader import load_training_data

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

class MultiModalDiffusion(nn.Module):
    """
    Modified version of your base EDM-based multi-modal diffusion that now also
    applies diffusion (noise/denoise) to tabular data, rather than using it
    only as a conditioning signal.
    """

    def __init__(
        self,
        dit: nn.Module,
        train_mask_ratio: float = 0.0,
        latent_reg_weight: float = 0.0,
        diffuse_tables: bool = True,  # <--- new flag to control table diffusion
    ) -> None:
        """
        Args:
            dit (nn.Module): The underlying diffusion model (e.g., DiT).
            train_mask_ratio (float): Mask ratio for training (e.g., token dropping).
            latent_reg_weight (float): Weight for an optional latent regularization term.
            diffuse_tables (bool): If True, also diffuse the tabular data.
        """
        super().__init__()
        self.dit = dit
        self.train_mask_ratio = train_mask_ratio
        self.latent_reg_weight = latent_reg_weight
        self.diffuse_tables = diffuse_tables  # new

        self.train_mask_ratio = train_mask_ratio
        self.train_mask_ratio_tab = train_mask_ratio
        self.latent_reg_weight = latent_reg_weight

        self.edm_config = EasyDict({
            'sigma_min': 0.002,
            'sigma_max': 40,
            'P_mean': -0.6,
            'P_std': 1.2,
            'sigma_data': 0.9,
            'num_steps': 32,
            'rho': 7,
            'S_churn': 0,
            'S_min': 0,
            'S_max': float('inf'),
            'S_noise': 1
        })

        self._dtype = 'bfloat16'
        self.latent_channels = 4

    def forward(self, images: torch.Tensor, table_data: torch.Tensor) -> dict[str, float | Tensor | Any]:
        """
        Perform a forward pass to compute the EDM loss for *both* images and table data
        (if self.diffuse_tables==True). If you only want to condition on table without
        noising it, you can set diffuse_tables=False or adapt the logic below.
        """
        device = images.device
        B = images.shape[0]

        # 1) Sample sigma from a log-normal distribution
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()

        # 2) Compute the weight (same for image and table)
        weight = ((sigma**2 + self.edm_config.sigma_data**2) / (sigma * self.edm_config.sigma_data)**2)

        # 3) Add noise to input *for both image and table*
        noise_img = torch.randn_like(images)
        if self.diffuse_tables:
            # shape match: table_data is [B, T], so expand sigma if needed
            noise_tab = torch.randn_like(table_data)
            noised_table = table_data + sigma.view(-1, 1) * noise_tab
        else:
            noised_table = table_data  # if you want to keep table as purely conditioning

        noised_img = images + sigma * noise_img

        # 4) EDM scaling factors
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1.0 / (self.edm_config.sigma_data**2 + sigma_in**2).sqrt()
        c_skip = self.edm_config.sigma_data**2 / (sigma_in**2 + self.edm_config.sigma_data**2)
        c_out = sigma_in * self.edm_config.sigma_data / (sigma_in**2 + self.edm_config.sigma_data**2).sqrt()
        # t = (sigma_in.log() / 4.0).squeeze()
        c_noise = sigma.log() / 4.0
        time_scalar = c_noise.flatten()

        # IMPORTANT for table dimension: reshape c_in etc.
        # We'll broadcast a (B,1) factor for table_data, or handle inside the model.
        # In practice, your model might handle these differently.

        # 5) Forward pass through the model
        out = self.dit(
            x_img=c_in * noised_img,
            x_tab=c_in.squeeze(-1).squeeze(-1) * noised_table,
            t=time_scalar
        )

        F_x = out['image_sample']                     # predicted "noise-free" image
        F_tab = out['tab_sample']         # predicted "noise-free" table

        # 6) Combine for denoised prediction
        D_xn_img = c_skip * noised_img + c_out * F_x
        D_xn_tab = c_skip.view(-1,1,1,1).squeeze(-1).squeeze(-1) * noised_table + \
                       c_out.view(-1,1,1,1).squeeze(-1).squeeze(-1) * F_tab

        # 7) Compute the MSE for image
        loss_img = weight * ((D_xn_img - images) ** 2)
        image_loss = loss_img.mean(dim=[1, 2, 3]).mean()

        # 8) Compute the MSE for table if we're diffusing it
        if self.diffuse_tables and F_tab is not None:
            # shape: D_xn_tab and table_data are [B, T]
            table_weight = weight.view(-1,1)
            loss_tab = table_weight * ((D_xn_tab - table_data) ** 2)
            table_loss = loss_tab.mean(dim=1).mean()
        else:
            table_loss = 0.0

        total_loss = 0.5 * (image_loss + table_loss)

        # 9) Optional latent regularization
        if self.latent_reg_weight > 0:
            real_mean = images.mean(dim=(0, 2, 3), keepdim=True)
            real_std = images.std(dim=(0, 2, 3), keepdim=True)
            pred_mean = F_x.mean(dim=(0, 2, 3), keepdim=True)
            pred_std = F_x.std(dim=(0, 2, 3), keepdim=True)
            mean_loss = F.mse_loss(pred_mean, real_mean)
            std_loss = F.mse_loss(pred_std, real_std)
            reg_loss = mean_loss + std_loss
            total_loss = total_loss + self.latent_reg_weight * reg_loss

        return {"loss": total_loss, "loss_img": image_loss, "loss_tab": table_loss}

    @torch.no_grad()
    def _sample_edm(
            self,
            batch_size: int,
            cfg: float = 1.0,
            steps: Optional[int] = None,
            height: int = 32,
            width: int = 32,
            table_dim: int = 174,  # or self.tab_size
            device: str = 'cuda',
            save_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Now returns final latents for BOTH image and table if self.diffuse_tables=True.

        If diffuse_tables=True, we start from random noise for the table latents as well
        and do the same Euler steps. If you want to *condition* on some table data instead,
        you'd adapt by partially or not noising the table data.
        """
        self.eval()
        steps = steps or self.edm_config.num_steps
        c = self.latent_channels

        # 1) Start from pure noise for image latents
        x_img = torch.randn((batch_size, c, height, width), device=device, dtype=torch.float32)
        # 2) Start from pure noise for table latents
        x_tab = torch.randn((batch_size, table_dim), device=device, dtype=torch.float32)

        t_vals = self.create_edm_timesteps(steps, device)
        x_img_next = x_img.double() * t_vals[0]
        x_tab_next = x_tab.double() * t_vals[0]  # scale table noise

        def model_forward(x_img_in, x_tab_in, t_sigma, cfg_val):
            """
            A helper that replicates the c_in/c_skip/c_out logic for both image and table.
            """
            B = x_img_in.shape[0]
            # sigma_in = t_sigma.reshape(-1, 1, 1, 1).float()
            t_hat_vec = torch.full((B,), float(t_sigma), device=x_img_in.device)
            sigma_in = t_hat_vec.view(B, 1, 1, 1).float()  # shape [B,1,1,1]

            c_in = 1.0 / (sigma_in ** 2 + self.edm_config.sigma_data ** 2).sqrt()
            c_skip = self.edm_config.sigma_data ** 2 / (sigma_in ** 2 + self.edm_config.sigma_data ** 2)
            c_out = sigma_in * self.edm_config.sigma_data / (sigma_in ** 2 + self.edm_config.sigma_data ** 2).sqrt()

            # For table: broadcast the same factor over the [B, D] dimension
            # c_in_tab = c_in.view(B, 1)
            c_in_tab = c_in.squeeze(-1).squeeze(-1)
            c_skip_tab = c_skip.view(B, 1)
            c_out_tab = c_out.view(B, 1)

            # Time embedding
            t_embed = (sigma_in.log() / 4.0).reshape(-1)
            if t_embed.numel() == 1 and B > 1:
                t_embed = t_embed.expand(B)

            # forward model
            out = self.dit(
                x_img=c_in * x_img_in.float(),
                x_tab=c_in_tab * x_tab_in.float(),
                t=t_embed,
                cfg=cfg_val,
                mask_ratio=0.0
            )
            F_img = out['image_sample'].float()
            F_tab = out['tab_sample'].float()

            # combine
            denoised_img = c_skip * x_img_in + c_out * F_img
            denoised_tab = c_skip_tab * x_tab_in + c_out_tab * F_tab

            return denoised_img, denoised_tab

        # Main EDM sampling loop
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_img_cur = x_img_next
            x_tab_cur = x_tab_next

            gamma = (
                min(self.edm_config.S_churn / steps, np.sqrt(2) - 1)
                if (self.edm_config.S_min <= t_cur <= self.edm_config.S_max)
                else 0.0
            )
            t_hat = t_cur + gamma * t_cur
            if gamma > 0:
                noise_img = torch.randn_like(x_img_cur)
                noise_tab = torch.randn_like(x_tab_cur)
                x_img_hat = x_img_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.edm_config.S_noise * noise_img
                x_tab_hat = x_tab_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.edm_config.S_noise * noise_tab
            else:
                x_img_hat = x_img_cur
                x_tab_hat = x_tab_cur

            # Euler step
            denoised_img, denoised_tab = model_forward(x_img_hat, x_tab_hat, t_hat, cfg)
            d_img_cur = (x_img_hat - denoised_img) / t_hat
            d_tab_cur = (x_tab_hat - denoised_tab) / t_hat

            x_img_next = x_img_hat + (t_next - t_hat) * d_img_cur
            x_tab_next = x_tab_hat + (t_next - t_hat) * d_tab_cur

            # 2nd-order correction
            if i < steps - 1:
                denoised_img2, denoised_tab2 = model_forward(x_img_next, x_tab_next, t_next, cfg)
                d_img_prime = (x_img_next - denoised_img2) / t_next
                d_tab_prime = (x_tab_next - denoised_tab2) / t_next

                x_img_next = x_img_hat + (t_next - t_hat) * (0.5 * d_img_cur + 0.5 * d_img_prime)
                x_tab_next = x_tab_hat + (t_next - t_hat) * (0.5 * d_tab_cur + 0.5 * d_tab_prime)

        final_img_latents = x_img_next.float()
        final_tab_latents = x_tab_next.float()

        return final_img_latents, final_tab_latents


    def create_edm_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        """
        Create a time schedule for EDM sampling steps.
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

    # @torch.no_grad()
    # def generate_samples(
    #         self,
    #         n_samples: int,
    #         data_bucket: DataBucket,
    #         vae: nn.Module,
    #         batch_size: int = 4,
    #         cfg: float = 1.0,
    #         steps: Optional[int] = None,
    #         device: str = "cuda",
    # ) -> Tuple[DataBucket, Dict[Any, List[int]], DataBucket, DataBucket]:
    #     """
    #     Generate BOTH images and tables from scratch (if self.diffuse_tables=True).
    #     Now we also return a DataBucket of the *generated tables*.
    #
    #     Returns:
    #       (DataBucket for images, dict cond_mapping, DataBucket for the conditioning table,
    #        DataBucket for the *generated/diffused* table).
    #     """
    #     if data_bucket.label not in (DataLabel.TAB, DataLabel.BOTH):
    #         raise ValueError(
    #             "DataBucket must have label=TAB or BOTH for table conditioning."
    #         )
    #
    #     loader = data_bucket.get_dataloader(batch_size=batch_size, shuffle=True)
    #     cond_iter = infinite_loader(loader)
    #
    #     if steps is None:
    #         steps = self.edm_config.num_steps
    #     dtype = self._dtype
    #
    #     cond_mapping: Dict[Any, List[int]] = {}
    #     all_decoded_imgs = []
    #     all_tab_data = []
    #     all_generated_tabs = []
    #
    #     total_generated = 0
    #     global_index = 0
    #
    #     while total_generated < n_samples:
    #         batch = next(cond_iter)
    #         tab_data = batch["tabular"]
    #
    #         current_bsz = tab_data.shape[0]
    #         if total_generated + current_bsz > n_samples:
    #             current_bsz = n_samples - total_generated
    #             tab_data = tab_data[:current_bsz]
    #
    #         tab_data = tab_data.to(device=device, dtype=dtype)
    #
    #         # We'll sample from noise for image & table
    #         img_latents, tab_latents = self._sample_edm(
    #             batch_size=current_bsz,
    #             cfg=cfg,
    #             steps=steps,
    #             height=self.image_pixel_image_height,
    #             width=self.image_pixel_image_width,
    #             table_dim=tab_data.shape[1],
    #             device=device,
    #         )
    #
    #         # decode the images via VAE
    #         decoded_imgs = decode_latents(vae, img_latents, vae.config.scaling_factor)
    #
    #         # store
    #         all_decoded_imgs.append(decoded_imgs.cpu())
    #         all_tab_data.append(tab_data.cpu())
    #         # The final table latents are in tab_latents, but if you want them as "generated" data directly:
    #         all_generated_tabs.append(tab_latents.cpu())
    #
    #         # cond_mapping
    #         if "dir" in batch:
    #             for i in range(current_bsz):
    #                 cond_key = batch["dir"][i]
    #                 if cond_key not in cond_mapping:
    #                     cond_mapping[cond_key] = []
    #                 cond_mapping[cond_key].append(global_index + i)
    #         else:
    #             for i in range(current_bsz):
    #                 ck = f"cond_{global_index + i}"
    #                 if ck not in cond_mapping:
    #                     cond_mapping[ck] = []
    #                 cond_mapping[ck].append(global_index + i)
    #
    #         total_generated += current_bsz
    #         global_index += current_bsz
    #
    #     # Merge all
    #     final_images = torch.cat(all_decoded_imgs, dim=0)[:n_samples]
    #     final_tab_data = torch.cat(all_tab_data, dim=0)[:n_samples]
    #     final_gen_tab = torch.cat(all_generated_tabs, dim=0)[:n_samples]
    #
    #     # Build DataBucket for generated images
    #     img_list = [final_images[i] for i in range(n_samples)]
    #     image_bucket = DataBucket(
    #         source_type=SourceType.LIST,
    #         label=DataLabel.IMAGE,
    #         data_list=img_list,
    #         metadata={"is_generated": True}
    #     )
    #
    #     # Build DataBucket for the conditioning table used
    #     tab_list = [final_tab_data[i] for i in range(n_samples)]
    #     cond_table_bucket = DataBucket(
    #         source_type=SourceType.LIST,
    #         label=DataLabel.TAB,
    #         data_list=tab_list
    #     )
    #
    #     # Build DataBucket for the newly diffused/generated table latents
    #     # (If you prefer them in real space, you need a "decode" step for tables too.)
    #     gen_tab_list = [final_gen_tab[i] for i in range(n_samples)]
    #     gen_table_bucket = DataBucket(
    #         source_type=SourceType.LIST,
    #         label=DataLabel.TAB,
    #         data_list=gen_tab_list,
    #         metadata={"is_generated": True}
    #     )
    #
    #     return image_bucket, cond_mapping, cond_table_bucket, gen_table_bucket

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
