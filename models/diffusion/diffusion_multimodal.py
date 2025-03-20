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
from utils.data import DataBucket, infinite_loader, DataLabel
from models.dit.dit_multimodal import load_dit
from models.vae.vae import decode_latents, load_vae
from data.loader import load_training_data
from utils.ddp import (is_dist_available_and_initialized, get_world_size, get_rank, all_gather_tensor,
                       all_gather_object, setup_distributed)

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
        if data_bucket.label not in (DataLabel.TAB, DataLabel.BOTH):
            raise ValueError("DataBucket must be labeled TAB or BOTH for table conditioning.")

        # Turn data_source into an iterator if it's a DataLoader
        if isinstance(data_bucket.data_source, torch.utils.data.DataLoader):
            cond_iter = infinite_loader(data_bucket.data_source)
        else:
            cond_iter = None

        cond_mapping: Dict[Any, List[int]] = {}
        all_decoded_imgs = []
        all_tab_data = []  # Will store each batch's tab_data here

        total_generated = 0
        global_index = 0

        while total_generated < n_samples:
            current_bsz = min(batch_size, n_samples - total_generated)

            # Fetch conditioning data
            if cond_iter is not None:
                batch = next(cond_iter)
                tab_data = batch["tabular"][:current_bsz].to(device, dtype=self._dtype)
                condition_keys = batch.get("dir", None)  # e.g. file paths / IDs
                if condition_keys is None:
                    condition_keys = [f"cond_{i}" for i in range(current_bsz)]
                else:
                    condition_keys = condition_keys[:current_bsz]
            else:
                ds = data_bucket.data_source
                ds_size = len(ds)
                idxs = torch.randint(0, ds_size, (current_bsz,))
                idxs = idxs.cpu().numpy()

                tab_list = []
                condition_keys = []
                for i_idx in idxs:
                    item = ds[i_idx]
                    if data_bucket.label == DataLabel.TAB:
                        tab_list.append(item)
                    else:
                        tab_list.append(item[1])
                    condition_keys.append(i_idx)  # track the index

                tab_data = torch.stack(tab_list, dim=0).to(device, dtype=self._dtype)

            # Store this batch's tab_data for later
            all_tab_data.append(tab_data.clone().cpu())

            # Run EDM sampling for this batch
            final_latents = self._sample_edm(
                table_data=tab_data,
                batch_size=current_bsz,
                cfg=cfg,
                steps=steps,
                device=device,
            )

            # Decode pixel-space images with VAE
            decoded_imgs = decode_latents(vae, final_latents, vae.config.scaling_factor)
            all_decoded_imgs.append(decoded_imgs)

            # Track mapping from condition keys -> generated sample indices
            for i, ck in enumerate(condition_keys):
                ck = str(ck)  # ensure it's hashable (e.g. string)
                if ck not in cond_mapping:
                    cond_mapping[ck] = []
                cond_mapping[ck].append(global_index + i)

            global_index += current_bsz
            total_generated += current_bsz

        # Combine all decoded images into one tensor: (#samples, C, H, W)
        final_images = torch.cat(all_decoded_imgs, dim=0)[:n_samples]

        # Combine all tab data (so that it has the same #samples shape)
        final_tab_data = torch.cat(all_tab_data, dim=0)[:n_samples]

        # Build DataBucket for the generated images
        output_data = [final_images[i] for i in range(n_samples)]
        result_bucket = DataBucket(data_source=output_data, label=DataLabel.IMAGE)

        # Build DataBucket for the input tab data
        input_data = [final_tab_data[i] for i in range(n_samples)]
        input_bucket = DataBucket(data_source=input_data, label=DataLabel.TAB)

        return result_bucket, cond_mapping, input_bucket

    # TODO: check, something it's not working
    @torch.no_grad()
    def generate_samples_distributed(
            self,
            n_samples: int,
            data_bucket: DataBucket,
            vae: nn.Module,
            batch_size: int = 4,
            cfg: float = 1.0,
            steps: Optional[int] = None,
            device: str = "cuda",
    ) -> Tuple[Optional[DataBucket], Optional[Dict[Any, List[int]]]]:
        """
        Generate images conditioned on tabular data, with optional DDP usage.
        - If not in distributed mode, just calls `generate_samples` (single GPU).
        - If in distributed mode, each rank generates a subset, then we gather.

        Returns:
            (DataBucket, dict) on rank 0, (None, None) on other ranks.
        """
        # 1. If not in distributed mode, fall back to single-GPU
        if not is_dist_available_and_initialized():
            return self.generate_samples(
                n_samples=n_samples,
                data_bucket=data_bucket,
                vae=vae,
                batch_size=batch_size,
                cfg=cfg,
                steps=steps,
                device=device
            )

        # 2. Figure out how many samples this rank should produce
        world_size = get_world_size()
        rank = get_rank()
        base_samples_per_rank = n_samples // world_size
        remainder = n_samples % world_size
        local_n_samples = base_samples_per_rank + (1 if rank < remainder else 0)

        # 3. Generate the local portion
        local_bucket, local_cond_map = self.generate_samples(
            n_samples=local_n_samples,
            data_bucket=data_bucket,
            vae=vae,
            batch_size=batch_size,
            cfg=cfg,
            steps=steps,
            device=device
        )
        # `local_bucket` is a DataBucket whose .data_source likely is a list of images (Tensors).

        # Convert local images to a single Tensor for gathering
        if local_bucket is not None and len(local_bucket.data_source) > 0:
            # For example, shape: (local_count, C, H, W)
            local_images = torch.stack(local_bucket.data_source, dim=0).to(device=device)
        else:
            # If no images generated, create an empty Tensor
            local_images = torch.empty(
                (0, self.image_pixel_channels, self.image_pixel_image_height,
                 self.image_pixel_image_width), device=device
            )

        # 4. Gather the per-rank counts so we can index properly
        local_count = torch.tensor([local_images.size(0)], dtype=torch.long, device=device)
        # Gather all counts into a single Tensor of shape (world_size, 1)
        gathered_counts = all_gather_tensor(local_count.unsqueeze(0))  # [world_size, 1]
        if gathered_counts.dim() == 2:
            gathered_counts = gathered_counts.squeeze(1)  # [world_size]

        # Convert to Python list
        counts_list = gathered_counts.cpu().tolist()
        offsets = []
        running_total = 0
        for c in counts_list:
            offsets.append(running_total)
            running_total += c
        # 'running_total' is the total number of images across all ranks

        # 5. Shift the local_cond_map so that local indexes become global indexes
        local_offset = offsets[rank]
        if local_cond_map is not None:
            for key in local_cond_map:
                # shift each index by local_offset
                local_cond_map[key] = [local_offset + idx for idx in local_cond_map[key]]

        # 6. Gather all local_images into a big Tensor on each rank, then slice on rank 0
        all_images = all_gather_tensor(local_images)  # shape (sum_of_all, C, H, W) with possible internal padding

        # 7. Gather all local_cond_map dictionaries onto every rank
        #    (each rank ends up with a list of cond_maps, one per rank)
        cond_maps = all_gather_object(local_cond_map)  # list of size world_size

        # 8. On rank 0, combine everything
        if is_main_process():
            # Use 'running_total' to slice out any padding from all_images
            final_list = [all_images[i] for i in range(running_total)]

            # Merge all cond_maps into a single dictionary
            combined_map: Dict[Any, List[int]] = {}
            for cm in cond_maps:
                if cm is None:
                    continue
                for k, idx_list in cm.items():
                    if k not in combined_map:
                        combined_map[k] = []
                    combined_map[k].extend(idx_list)

            # Build the final DataBucket
            final_bucket = DataBucket(data_source=final_list, label=DataLabel.IMAGE)
            return final_bucket, combined_map
        else:
            # Non-main ranks return nothing
            return None, None


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

    from utils.configurations import set_project_root

    # Comment out if you want to run in DDP mode.
    # ddp_cfg = DictConfig({"distributed": {"backend": "nccl"}})
    # local_rank = setup_distributed(ddp_cfg)

    set_project_root()

    # 1) Load config(s)
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        dit_cfg = compose(config_name="base_dit_training")
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        diffusion_cfg = compose(config_name="diffusion")

    # 2) Load the underlying DiT model
    dit_model = load_dit(dit_cfg)

    # 3) Build the diffusion model
    diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

    # 4) Load VAE
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        vae_cfg = compose(config_name="vae")
    vae = load_vae(vae_cfg)
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cuda")

    # 5) Load train dataloader
    train_dataloader = load_training_data(dit_cfg)

    # --------------------------
    # EXAMPLE 1: Using a DataLoader
    # --------------------------
    tab_bucket_dataloader = DataBucket(data_source=train_dataloader, label=DataLabel.TAB)

    # Generate 5 images from this DataLoader
    gen_bucket_dl, cond_map_dl = diffusion_model.generate_samples_distributed(
        n_samples=9,
        data_bucket=tab_bucket_dataloader,
        vae=vae,
        batch_size=2,
        cfg=1.5,
        steps=6,
        device='cuda'
    )
    if is_main_process() and gen_bucket_dl is not None:
        print("[DIST GEN:DataLoader] Output label:", gen_bucket_dl.label)
        print("[DIST GEN:DataLoader] # of samples:", len(gen_bucket_dl.data_source))
        print("[DIST GEN:DataLoader] cond_map:", cond_map_dl)

    # --------------------------
    # EXAMPLE 2: Using a list of tabulars
    # --------------------------
    list_of_tab_tensors = []
    for i, batch in enumerate(train_dataloader):
        tabs = batch['tabular']  # shape (B, 174)
        for t in tabs:
            list_of_tab_tensors.append(t.cpu())
        if len(list_of_tab_tensors) >= 6:
            break

    tab_bucket_list = DataBucket(
        data_source=list_of_tab_tensors,
        label=DataLabel.TAB
    )

    # Generate 6 images from this list
    gen_bucket_list, cond_map_list = diffusion_model.generate_samples_distributed(
        n_samples=10,
        data_bucket=tab_bucket_list,
        vae=vae,
        batch_size=2,
        cfg=1.2,
        steps=5,
        device='cuda'
    )
    if is_main_process() and gen_bucket_list is not None:
        print("[DIST GEN:List] Output label:", gen_bucket_list.label)
        print("[DIST GEN:List] # of samples:", len(gen_bucket_list.data_source))
        print("[DIST GEN:List] cond_map:", cond_map_list)