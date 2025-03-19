import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from easydict import EasyDict
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
from composer.models import ComposerModel
from torch import Tensor
from torch.utils.data import DataLoader

from enums.models.diffusion import ScenarioType, DataLabel
from enums.models.model_types import ModelType
from utils.configurations import apply_overrides, set_project_root
from utils.data import DataBucket, infinite_loader
from utils.model import load_checkpoint
# from models.utils.model_loader import load_model
from models.dit.dit_multimodal import load_dit
from models.vae.vae import encode_images, decode_latents, load_vae
from data.loader import load_training_data

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


class MultiModalDiffusion(ComposerModel):
    """
    Multi-modal EDM that can handle images + tabular data jointly, loaded from Hydra configs.

    This model:
      - Uses an EDM (Elucidated Diffusion) approach for noise scheduling and MSE losses.
      - Handles multiple scenarios (unconditional, cond_image, cond_table, cond_both).
      - Optionally applies patch/token masking during training to images/tables.
      - Provides generation routines (`generate_samples`) that can do unconditional or
        conditional sampling on images/tables/both with the EDM sampler loop.

    Attributes:
        dit (nn.Module): The DiT model that processes images + tabular data + time embeddings.
    """

    def __init__(
            self,
            dit: nn.Module,
            train_mask_ratio_img: float = 0.0,
            train_mask_ratio_tab: float = 0.0,
    ):
        """
        Initializes the MultiModalDiffusion model using Hydra configs.
        """
        super().__init__()

        from utils.configurations import set_project_root
        set_project_root()

        with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
            edm_cfg = compose(config_name="diffusion")

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

        self.train_mask_ratio_img = train_mask_ratio_img
        self.train_mask_ratio_tab = train_mask_ratio_tab
        self._dtype = DTYPE_MAP[EDM_params.dtype]

        # From data config, get shapes
        self.image_size = data_cfg.data.image_size
        self.latent_channels = data_cfg.data.latent_channels
        self.tab_size = data_cfg.data.tab_size

        self.dit = dit

        # For convenience
        self.randn_like = torch.randn_like


    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        """
        Forward pass for training. Retrieves images/tables, decides scenario, applies EDM loss.

        Args:
            batch (Dict[str, torch.Tensor]):
                Expected to contain "image" and "tabular" keys. May also contain scenario key.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (loss_dict, images, tables):
                - loss_dict includes "loss", "loss_img", "loss_tab".
                - images are the raw images from the batch.
                - tables are the raw table data from the batch.
        """
        if self.dit is None:
            raise RuntimeError("DiT model not set. Call `set_dit_model(...)` before training.")

        images = batch["image"].to(self._dtype)
        tables = batch["tabular"].to(self._dtype)
        scenario = batch.get("scenario", ScenarioType.UNCOND)

        loss_dict = self.edm_loss(
            x_img=images,
            x_tab=tables,
            scenario=scenario,
            mask_ratio_img=self.train_mask_ratio_img,
            mask_ratio_tab=self.train_mask_ratio_tab
        )
        return loss_dict, images, tables

    def edm_loss(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        scenario: ScenarioType,
        mask_ratio_img: float,
        mask_ratio_tab: float,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        The core EDM loss function:
          1) Sample sigma from lognormal.
          2) Noise image/table based on scenario.
          3) Forward pass through the model with time conditioning.
          4) Compute weighted MSE loss.
          5) Unmask if patch or token masking is used.

        Args:
            x_img (torch.Tensor): Image data of shape (N, C, H, W).
            x_tab (torch.Tensor): Tabular data of shape (N, D).
            scenario (ScenarioType): Enum specifying unconditional or various conditionings.
            mask_ratio_img (float): Probability of masking image patches in training.
            mask_ratio_tab (float): Probability of masking tabular tokens in training.

        Returns:
            Dict[str, torch.Tensor]:
                A dictionary with keys {"loss", "loss_img", "loss_tab"}.
        """
        batch_size = x_img.shape[0]

        # 1) Sample sigma
        sigma = self._sample_sigma(n=batch_size, device=x_img.device)

        # 2) EDM weighting factor
        weight = (sigma ** 2 + self.edm_config.sigma_data ** 2) / ((sigma * self.edm_config.sigma_data) ** 2)

        # 3) Noise the inputs according to scenario
        x_noisy_img, x_noisy_tab = self._apply_scenario_noise(x_img, x_tab, scenario, sigma)

        # 4) Forward pass with partial for DiT
        model_out = self.model_forward_wrapper(
            x_noisy_img=x_noisy_img,
            x_noisy_tab=x_noisy_tab,
            sigma=sigma,
            scenario=scenario,
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab,
            model_forward_fxn=partial(self.dit.forward, cfg=1.0),
        )

        denoised_img = model_out["sample_img"]
        denoised_tab = model_out["sample_tab"]

        # Weighted MSE: reduce across (C,H,W) for images, across dim=1 for tab
        loss_img = weight * (denoised_img - x_img).square()
        loss_img_unmasked = loss_img.mean(dim=(1, 2, 3))

        weight_tab = weight.view(-1, 1)
        loss_tab = weight_tab * (denoised_tab - x_tab).square()
        loss_tab_unmasked = loss_tab.mean(dim=1)

        final_loss_img = loss_img_unmasked
        final_loss_tab = loss_tab_unmasked

        # 5) Unmask if the model returns "mask_img" or "mask_tab"
        # Image masking
        if mask_ratio_img > 0.0 and "mask_img" in model_out:
            mask_img = model_out["mask_img"]
            patch_size = getattr(self.dit, "patch_size", 4)

            # MSE across channels => shape (N,1,H,W)
            mse_per_pixel = (denoised_img - x_img).square().mean(dim=1, keepdim=True)
            mse_per_patch = F.avg_pool2d(mse_per_pixel, kernel_size=patch_size).squeeze(1)
            mse_per_patch = mse_per_patch.flatten(start_dim=1)  # (N, #patches)

            unmask_img = 1.0 - mask_img.flatten(start_dim=1)
            unmasked_mse_img = (mse_per_patch * unmask_img).sum(dim=1) / unmask_img.sum(dim=1)
            final_loss_img = unmasked_mse_img

        # Table masking
        if mask_ratio_tab > 0.0 and "mask_tab" in model_out:
            mask_tab = model_out["mask_tab"]  # (N, D)
            mse_per_token = weight_tab * (denoised_tab - x_tab).square()
            unmask_tab = 1.0 - mask_tab
            unmasked_mse_tab = (mse_per_token * unmask_tab).sum(dim=1) / unmask_tab.sum(dim=1)
            final_loss_tab = unmasked_mse_tab

        loss_per_sample = 0.5 * (final_loss_img + final_loss_tab)

        return {
            "loss": loss_per_sample.mean(),
            "loss_img": final_loss_img.mean(),
            "loss_tab": final_loss_tab.mean(),
        }

    def model_forward_wrapper(
        self,
        x_noisy_img: torch.Tensor,
        x_noisy_tab: torch.Tensor,
        sigma: torch.Tensor,
        scenario: ScenarioType,
        mask_ratio_img: float,
        mask_ratio_tab: float,
        model_forward_fxn,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Wraps the DiT forward pass with c_in, c_skip, c_out logic from EDM.

        Args:
            x_noisy_img (torch.Tensor): Noised images.
            x_noisy_tab (torch.Tensor): Noised tab data.
            sigma (torch.Tensor): Noise level (N, 1, 1, 1).
            scenario (ScenarioType): Condition scenario.
            mask_ratio_img (float): If > 0, image patch masking.
            mask_ratio_tab (float): If > 0, table token masking.
            model_forward_fxn (Callable): Typically partial(self.dit.forward, cfg=...).

        Returns:
            Dict[str, torch.Tensor]:
                Keys: {"sample_img", "sample_tab", "mask_img", "mask_tab"}.
                The "mask_*" keys might be absent if no masking is performed.
        """
        device = x_noisy_img.device
        sigma_data = self.edm_config.sigma_data

        # c_in, c_skip, c_out, c_noise
        c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_skip = (sigma_data ** 2) / (sigma_data ** 2 + sigma ** 2)
        c_out = (sigma_data * sigma) / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_noise = sigma.log() / 4.0

        # Scale inputs
        x_img_in = c_in * x_noisy_img
        x_tab_in = c_in.squeeze(-1).squeeze(-1) * x_noisy_tab

        # Forward
        out = model_forward_fxn(
            x_img_in,
            x_tab_in,
            c_noise.flatten(),  # time_embed for images
            c_noise.flatten(),  # time_embed for tables
            cond_image=(scenario in [ScenarioType.COND_IMAGE, ScenarioType.COND_BOTH]),
            cond_table=(scenario in [ScenarioType.COND_TABLE, ScenarioType.COND_BOTH]),
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab,
            **kwargs
        )

        F_img = out["sample_img"].to(device)
        F_tab = out["sample_tab"].to(device)

        c_skip = c_skip.to(device)
        c_out = c_out.to(device)
        denoised_img = c_skip * x_noisy_img + c_out * F_img

        c_skip_tab = c_skip.view(-1)
        c_out_tab = c_out.view(-1)
        denoised_tab = c_skip_tab.unsqueeze(-1) * x_noisy_tab + c_out_tab.unsqueeze(-1) * F_tab

        out["sample_img"] = denoised_img
        out["sample_tab"] = denoised_tab
        return out

    def _apply_scenario_noise(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        scenario: ScenarioType,
        sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to each modality according to the specified scenario.

        Args:
            x_img (torch.Tensor): Clean image latents (N, C, H, W).
            x_tab (torch.Tensor): Clean table latents (N, D).
            scenario (ScenarioType): Condition scenario.
            sigma (torch.Tensor): The noise level (N, 1, 1, 1).

        Returns:
            (torch.Tensor, torch.Tensor):
                The noised images and the noised tabular data.
        """
        if scenario == ScenarioType.UNCOND:
            x_noisy_img = x_img + self.randn_like(x_img) * sigma
            x_noisy_tab = x_tab + torch.randn_like(x_tab) * sigma.view(-1, 1)

        elif scenario == ScenarioType.COND_IMAGE:
            # Keep image unnoised
            x_noisy_img = x_img
            x_noisy_tab = x_tab + torch.randn_like(x_tab) * sigma.view(-1, 1)

        elif scenario == ScenarioType.COND_TABLE:
            # Keep table unnoised
            x_noisy_img = x_img + self.randn_like(x_img) * sigma
            x_noisy_tab = x_tab

        else:  # scenario == COND_BOTH
            scale = 0.5
            x_noisy_img = x_img + self.randn_like(x_img) * (sigma * scale)
            x_noisy_tab = x_tab + torch.randn_like(x_tab) * (sigma.view(-1, 1) * scale)

        return x_noisy_img, x_noisy_tab

    def _sample_sigma(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Samples the noise scale `sigma` from a lognormal distribution.

        Args:
            n (int): Batch size.
            device (torch.device): Device for the resulting tensor.

        Returns:
            torch.Tensor of shape (n, 1, 1, 1) with sampled sigma values.
        """
        normal_samples = torch.randn((n, 1, 1, 1), device=device)
        sigma = (normal_samples * self.edm_config.P_std + self.edm_config.P_mean).exp()
        return sigma

    # ------------------------------------------------------------------------
    # Composer Integration
    # ------------------------------------------------------------------------

    def loss(self, outputs, batch) -> torch.Tensor:
        """
        The standard Composer API to retrieve the final loss.

        Args:
            outputs: Output from `forward`.
            batch: The input batch.

        Returns:
            torch.Tensor: The scalar loss.
        """
        return outputs[0]["loss"]

    def eval_forward(self, batch, outputs: Optional[tuple] = None):
        """
        The standard Composer API for evaluation steps.

        Args:
            batch: Input batch for evaluation.
            outputs (optional): If already computed, skip forward.

        Returns:
            Model outputs.
        """
        if outputs is not None:
            return outputs
        return self.forward(batch)

    def get_metrics(self, is_train: bool = False):
        """
        Composer API to define metrics. No-op here.

        Args:
            is_train (bool): True if training, false otherwise.

        Returns:
            An empty dictionary.
        """
        return {}

    def update_metric(self, batch, outputs, metric):
        """
        Composer API for updating metrics. No-op here.

        Args:
            batch: Input batch.
            outputs: Model outputs.
            metric: The metric to update.
        """
        pass

    # ------------------------------------------------------------------------
    # EDM Sampler
    # ------------------------------------------------------------------------

    @torch.no_grad()
    def _edm_sampler_loop(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        scenario: ScenarioType,
        steps: Optional[int] = None,
        cfg: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the EDM sampling loop to iteratively denoise latents.

        Implements:
          1) Discretization of time steps.
          2) For each step, possible noise (churn), Euler step, and 2nd-order correction.

        Args:
            x_img (torch.Tensor): Initial latents for images (N, C, H, W).
            x_tab (torch.Tensor): Initial latents for tables (N, D).
            scenario (ScenarioType): Condition scenario.
            steps (int, optional): Number of diffusion steps. Default to `edm_config.num_steps`.
            cfg (float, optional): Guidance scale for classifier-free guidance.

        Returns:
            (torch.Tensor, torch.Tensor):
                The final denoised image latents, and final denoised table latents.
        """
        if self.dit is None:
            raise RuntimeError("DiT model not set. Call `set_dit_model(...)` before sampling.")

        mask_ratio_img = 0.0
        mask_ratio_tab = 0.0

        # Possibly use CFG if cfg > 1.0
        model_forward_fxn = (partial(self.dit.forward, cfg=cfg) if cfg > 1.0 else self.dit.forward)

        num_steps = self.edm_config.num_steps if steps is None else steps
        device = x_img.device

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            self.edm_config.sigma_max ** (1 / self.edm_config.rho)
            + step_indices / (num_steps - 1)
            * (self.edm_config.sigma_min ** (1 / self.edm_config.rho)
               - self.edm_config.sigma_max ** (1 / self.edm_config.rho))
        ) ** self.edm_config.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # Scale initial latents
        x_img_next = x_img.to(torch.float64) * t_steps[0]
        x_tab_next = x_tab.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_img_cur = x_img_next
            x_tab_cur = x_tab_next

            # 1) Churn
            gamma = (min(self.edm_config.S_churn / num_steps, np.sqrt(2) - 1)
                     if (self.edm_config.S_min <= t_cur <= self.edm_config.S_max) else 0)
            t_hat_val = t_cur + gamma * t_cur
            t_hat = torch.as_tensor(t_hat_val, device=device).reshape(1)

            if gamma > 0:
                noise_scale = (t_hat_val ** 2 - t_cur ** 2).sqrt() * self.edm_config.S_noise
                x_img_hat = x_img_cur + noise_scale * torch.randn_like(x_img_cur)
                x_tab_hat = x_tab_cur + noise_scale * torch.randn_like(x_tab_cur)
            else:
                x_img_hat = x_img_cur
                x_tab_hat = x_tab_cur

            # 2) Euler step
            denoised_1 = self.model_forward_wrapper(
                x_noisy_img=x_img_hat.to(torch.float32),
                x_noisy_tab=x_tab_hat.to(torch.float32),
                sigma=t_hat.to(torch.float32),
                scenario=scenario,
                mask_ratio_img=mask_ratio_img,
                mask_ratio_tab=mask_ratio_tab,
                model_forward_fxn=model_forward_fxn,
                **kwargs
            )
            F_img1 = denoised_1["sample_img"].to(torch.float64)
            F_tab1 = denoised_1["sample_tab"].to(torch.float64)

            d_img_cur = (x_img_hat - F_img1) / t_hat_val
            d_tab_cur = (x_tab_hat - F_tab1) / t_hat_val

            x_img_next = x_img_hat + (t_next - t_hat_val) * d_img_cur
            x_tab_next = x_tab_hat + (t_next - t_hat_val) * d_tab_cur

            # 3) 2nd-order correction
            if i < num_steps - 1:
                denoised_2 = self.model_forward_wrapper(
                    x_noisy_img=x_img_next.to(torch.float32),
                    x_noisy_tab=x_tab_next.to(torch.float32),
                    sigma=t_next.to(torch.float32),
                    scenario=scenario,
                    mask_ratio_img=mask_ratio_img,
                    mask_ratio_tab=mask_ratio_tab,
                    model_forward_fxn=model_forward_fxn,
                    **kwargs
                )
                F_img2 = denoised_2["sample_img"].to(torch.float64)
                F_tab2 = denoised_2["sample_tab"].to(torch.float64)

                d_img_prime = (x_img_next - F_img2) / t_next
                d_tab_prime = (x_tab_next - F_tab2) / t_next

                x_img_next = x_img_hat + (t_next - t_hat_val) * (0.5 * d_img_cur + 0.5 * d_img_prime)
                x_tab_next = x_tab_hat + (t_next - t_hat_val) * (0.5 * d_tab_cur + 0.5 * d_tab_prime)

        return x_img_next.float(), x_tab_next.float()

    # ------------------------------------------------------------------------
    # Generation API
    # ------------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples(
        self,
        n_samples: int,
        scenario: ScenarioType,
        vae: nn.Module,
        data_bucket: Optional[DataBucket] = None,
        batch_size: int = 4,
        cfg: float = 1.0,
        steps: int = 10,
        device: Union[torch.device, str] = "cuda",
    ) -> Tuple[DataBucket, Optional[Dict[Any, List[int]]]]:
        """
        High-level entry point that dispatches to unconditional or conditional generation.
        Returns:
          - A DataBucket holding the *pixel-space* generated images + tabular data (if both).
            - A mapping (dict) from "condition key" -> list of generated sample indices, or None if UNCOND.

        Generate samples according to the given scenario. Optionally uses conditional data from a DataBucket
        (which can be a dataloader or a list).

        Usage Scenarios:
        1) Unconditional generation:
        - scenario='uncond'
        - data_bucket=None
        => returns n_samples of purely random generation from the model.

        2) Conditional on images only:
        - scenario='cond_image'
        - data_bucket.label must be 'image' or 'both'
        => The method will repeatedly sample from data_bucket to build each batch of conditioning images, until
        n_samples are reached.

        3) Conditional on tables only:
        - scenario='cond_table'
        - data_bucket.label must be 'tab' or 'both'
        => The method will repeatedly sample from data_bucket for tabular data.

        4) Conditional on both images + tables:
        - scenario='cond_both'
        - data_bucket.label must be 'both'
        => The method will repeatedly sample from data_bucket, which must provide pairs of (image, table) data.

        Args:
            n_samples (int): How many total samples to generate.
            scenario (str): One of {'uncond', 'cond_image', 'cond_table', 'cond_both'}.
            batch_size (int): How many samples in each generation batch.
            data_bucket (Optional[DataBucket]):
                The conditional data source if scenario != 'uncond'.
                If scenario='uncond', this must be None.
            guidance_scale (float): Classifier-free guidance scale.
            num_inference_steps (int): Number of EDM steps in sampling.
            device (torch.device, optional): Torch device to use. If None, uses self.dit's device.
            img_shape (tuple): Shape of image latents, e.g. (C,H,W).
            tab_shape (int): Dimensionality for tabular data.

            Returns:
                A tuple:
                    (1) DataBucket containing the generated data. The label will be
                    'image', 'tab', or 'both', matching the scenario outputs.
                    - shape: (n_samples, C, H, W) for images
                    - shape: (n_samples, tab_shape) for tables
                    (2) A dictionary mapping from the conditioning keys to lists of
                    generated sample indices, or None if scenario='uncond'.

                If the `data_bucket` had a list of paths, the keys will be those paths. If the `data_bucket` had a list
                of objects, the keys will be the *indices* in that list. If the `data_bucket` was a dataloader with
                e.g. patient directories, the keys might be the directory strings, etc.
        """

        # No conditional data => unconditional
        if scenario == ScenarioType.UNCOND:
            if data_bucket is not None:
                raise ValueError("For UNCOND scenario, data_bucket must be None.")
            return self._generate_samples_unconditional(
                n_samples=n_samples,
                batch_size=batch_size,
                vae=vae,
                steps=steps,
                device=device
            )
        else:
            if data_bucket is None:
                raise ValueError(f"For scenario={scenario}, must provide a DataBucket.")
            return self._generate_samples_conditional(
                n_samples=n_samples,
                scenario=scenario,
                data_bucket=data_bucket,
                batch_size=batch_size,
                vae=vae,
                cfg=cfg,
                steps=steps,
                device=device
            )

    @torch.no_grad()
    def _generate_samples_unconditional(
        self,
        n_samples: int,
        batch_size: int,
        vae: nn.Module,
        steps: int,
        device: torch.device,
    ) -> Tuple[DataBucket, None]:
        """
        Unconditional generation of samples.

        Args:
            n_samples (int):
                Number of samples to generate.
            batch_size (int):
                Batch size to generate in each iteration.
            vae (nn.Module):
                VAE used to decode latents to images.
            steps (int):
                Number of EDM sampling steps.
            device (torch.device):
                Device for tensor operations.

        Returns:
            (DataBucket, None):
                A DataBucket with image+tab latents decoded to pixel space,
                and None as no condition mapping is used for unconditional generation.
        """
        all_images = []
        all_tables = []
        total_generated = 0

        while total_generated < n_samples:
            current_bsz = min(batch_size, n_samples - total_generated)

            # Create random initial latents for unconditional scenario
            latent_shape_img = (current_bsz, self.latent_channels, self.image_size, self.image_size)
            latent_shape_tab = (current_bsz, self.tab_size)

            x_img = torch.randn(latent_shape_img, device=device, dtype=self._dtype)
            x_tab = torch.randn(latent_shape_tab, device=device, dtype=self._dtype)

            # Random lognormal sigma
            sigma = self._sample_sigma(n=current_bsz, device=device)
            x_img_noisy, x_tab_noisy = self._apply_scenario_noise(
                x_img, x_tab, ScenarioType.UNCOND, sigma
            )

            # EDM sampler
            final_latents_img, final_latents_tab = self._edm_sampler_loop(
                x_img_noisy, x_tab_noisy, scenario=ScenarioType.UNCOND, steps=steps
            )

            # Decode images
            decoded_imgs = decode_latents(vae, final_latents_img, vae.config.scaling_factor)

            all_images.append(decoded_imgs)
            all_tables.append(final_latents_tab)
            total_generated += current_bsz

        # Stack
        final_imgs = torch.cat(all_images, dim=0)[:n_samples]
        final_tabs = torch.cat(all_tables, dim=0)[:n_samples]

        # Wrap as a DataBucket
        out_data = []
        for i in range(n_samples):
            out_data.append((final_imgs[i], final_tabs[i]))

        result_bucket = DataBucket(data_source=out_data, label=DataLabel.BOTH)
        return result_bucket, None

    @torch.no_grad()
    def _generate_samples_conditional(
        self,
        n_samples: int,
        scenario: ScenarioType,
        data_bucket: DataBucket,
        batch_size: int,
        vae: nn.Module,
        device: torch.device,
        steps: int,
        cfg: float = 1.0
    ) -> Tuple[DataBucket, Dict[Any, List[int]]]:
        """
        Generates samples conditioned on images, tables, or both.

        Args:
            n_samples (int):
                Total samples desired.
            scenario (ScenarioType):
                COND_IMAGE, COND_TABLE, or COND_BOTH.
            data_bucket (DataBucket):
                The data source (DataLoader or list) containing the conditioning data.
            batch_size (int):
                Batch size for each generation step.
            vae (nn.Module):
                VAE used to decode latents.
            device (torch.device):
                Device for computations.
            steps (int):
                Number of EDM sampling steps.
            cfg (float):
                Guidance scale for classifier-free guidance.

        Returns:
            (DataBucket, Dict[Any, List[int]]):
                A DataBucket of final generated data and a mapping from condition keys
                (e.g. file paths or data IDs) to the list of sample indices.
        """
        if self.dit is None:
            raise RuntimeError("DiT model not set. Call `set_dit_model(...)` before generation.")

        # Make an infinite iterator if the data source is a DataLoader
        if isinstance(data_bucket.data_source, DataLoader):
            cond_iter = infinite_loader(data_bucket.data_source)
        else:
            cond_iter = None

        cond_mapping: Dict[Any, List[int]] = {}
        total_generated = 0
        global_index = 0

        # We will store either images, tables, or pairs depending on scenario
        all_generated = []

        while total_generated < n_samples:
            current_bsz = min(batch_size, n_samples - total_generated)

            if cond_iter is not None:
                # Dataloader path
                batch = next(cond_iter)
                cond_images = batch["image"][:current_bsz].to(device, dtype=self._dtype)
                cond_tables = batch["tabular"][:current_bsz].to(device, dtype=self._dtype)
                cond_dirs = batch["dir"][:current_bsz]

                condition_keys = cond_dirs
            else:
                # List path
                ds = data_bucket.data_source
                ds_size = len(ds)

                idxs = torch.randint(0, ds_size, (current_bsz,))
                idxs = idxs.cpu().numpy()

                cond_images_list = []
                cond_tables_list = []
                condition_keys = []

                for i_idx in idxs:
                    item = ds[i_idx]
                    key = i_idx

                    if scenario == ScenarioType.COND_IMAGE:
                        # We have images from the data source
                        if data_bucket.label == DataLabel.IMAGE:
                            cond_images_list.append(item)
                            cond_tables_list.append(None)
                        elif data_bucket.label == DataLabel.BOTH:
                            cond_images_list.append(item[0])
                            cond_tables_list.append(None)
                        else:
                            raise RuntimeError("DataBucket label must be IMAGE or BOTH for COND_IMAGE.")
                    elif scenario == ScenarioType.COND_TABLE:
                        if data_bucket.label == DataLabel.TAB:
                            cond_images_list.append(None)
                            cond_tables_list.append(item)
                        elif data_bucket.label == DataLabel.BOTH:
                            cond_images_list.append(None)
                            cond_tables_list.append(item[1])
                        else:
                            raise RuntimeError("DataBucket label must be TAB or BOTH for COND_TABLE.")
                    else:  # scenario == COND_BOTH
                        if data_bucket.label != DataLabel.BOTH:
                            raise RuntimeError("DataBucket must be labeled BOTH for COND_BOTH.")
                        cond_images_list.append(item[0])
                        cond_tables_list.append(item[1])

                    condition_keys.append(key)

                # Stack or create None
                if any(ci is not None for ci in cond_images_list):
                    cond_images = torch.stack([ci for ci in cond_images_list if ci is not None], dim=0)
                    cond_images = cond_images.to(device, dtype=self._dtype)
                else:
                    cond_images = None

                if any(ct is not None for ct in cond_tables_list):
                    cond_tables = torch.stack([ct for ct in cond_tables_list if ct is not None], dim=0)
                    cond_tables = cond_tables.to(device, dtype=self._dtype)
                else:
                    cond_tables = None

            # Convert conditioning images to latents if scenario includes images
            if scenario in (ScenarioType.COND_IMAGE, ScenarioType.COND_BOTH) and cond_images is not None:
                latents_img = encode_images(vae, cond_images, vae.config.scaling_factor)
            else:
                # Randomly init image latents if scenario doesn't use real images
                latent_shape_img = (current_bsz, self.latent_channels, self.image_size, self.image_size)
                latents_img = torch.randn(latent_shape_img, device=device, dtype=self._dtype)

            # Use tables if scenario includes them
            if scenario in (ScenarioType.COND_TABLE, ScenarioType.COND_BOTH) and cond_tables is not None:
                latents_tab = cond_tables
            else:
                latents_tab = torch.randn((current_bsz, self.tab_size), device=device, dtype=self._dtype)

            # Noise them
            sigma = self._sample_sigma(n=current_bsz, device=device)
            latents_img_noisy, latents_tab_noisy = self._apply_scenario_noise(
                latents_img, latents_tab, scenario, sigma
            )

            # Sample
            final_latents_img, final_latents_tab = self._edm_sampler_loop(
                latents_img_noisy,
                latents_tab_noisy,
                scenario=scenario,
                cfg=cfg,
                steps=steps
            )

            # Decode whichever is newly generated
            if scenario == ScenarioType.COND_IMAGE:
                generated_tab = final_latents_tab
                for i in range(current_bsz):
                    all_generated.append(generated_tab[i].cpu())  # store just tab
            elif scenario == ScenarioType.COND_TABLE:
                decoded_imgs = decode_latents(vae, final_latents_img, vae.config.scaling_factor)
                for i in range(current_bsz):
                    all_generated.append(decoded_imgs[i].cpu())  # store just image
            else:
                # scenario == COND_BOTH
                decoded_imgs = decode_latents(vae, final_latents_img, vae.config.scaling_factor)
                for i in range(current_bsz):
                    all_generated.append((decoded_imgs[i].cpu(), final_latents_tab[i].cpu()))

            # Track sample indices for the condition mapping
            for i, ck in enumerate(condition_keys):
                if ck not in cond_mapping:
                    cond_mapping[ck] = []
                cond_mapping[ck].append(global_index + i)

            global_index += current_bsz
            total_generated += current_bsz

        all_generated = all_generated[:n_samples]

        # Build final DataBucket with correct label
        if scenario == ScenarioType.COND_IMAGE:
            # We only have tabular data
            label = DataLabel.TAB
        elif scenario == ScenarioType.COND_TABLE:
            # We only have image data
            label = DataLabel.IMAGE
        else:
            label = DataLabel.BOTH

        result_bucket = DataBucket(data_source=all_generated, label=label)
        return result_bucket, cond_mapping


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
    # Example usage

    # 1) Set project root if needed
    set_project_root()

    # 2) Load configs
    #    (Adjust the paths to your local structure; below is just an example.)
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        diffusion_cfg = compose(config_name="diffusion")

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        data_cfg = compose(config_name="nacc")

    # 3) Load DiT
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        dit_cfg = compose(config_name="base_dit_training")
    dit_model = load_dit(dit_cfg)

    # 4) Build diffusion model
    diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

    # 5) Load VAE
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        vae_cfg = compose(config_name="vae")
    vae = load_vae(vae_cfg)
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cuda")

    # 6) Load checkpoint for the diffusion model
    ckpt_path = "/mnt/storage/nacc_sub/dit/2025-03-13_13-57-21_redone/checkpoints/checkpoint_step_45000_final.pt"
    load_checkpoint(diffusion_model, ckpt_path, "cuda")
    diffusion_model.eval()

    # 7) Load train dataloader
    train_dataloader = load_training_data(dit_cfg)

    # ----- EXAMPLE 1: UNCONDITIONAL GENERATION DATALOADER-----
    generated_data_bucket, _ = diffusion_model.generate_samples(
        n_samples=8,
        scenario=ScenarioType.UNCOND,
        vae=vae,
        batch_size=4,
        device=torch.device("cuda")
    )

    print("[INFO] Unconditional generation done. #samples:", len(generated_data_bucket.data_source))
    for i, (img, tab) in enumerate(generated_data_bucket.data_source):
        print(f"Sample {i}, image shape={img.shape}, tab shape={tab.shape}")

    # ----- EXAMPLE 2A: IMAGE-CONDITIONAL USING A DATALOADER -----
    # Suppose we create a small DataBucket from our training dataloader directly:
    image_cond_bucket = DataBucket(
        data_source=train_dataloader,  # The entire DataLoader
        label=DataLabel.IMAGE  # We're only conditioning on images
    )

    # Generate 7 samples total, in batches of 4, with guidance scale embedded in the model (or specified).
    generated_data_bucket, cond_map = diffusion_model.generate_samples(
        n_samples=7,
        scenario=ScenarioType.COND_IMAGE,
        data_bucket=image_cond_bucket,
        batch_size=4,
        vae=vae,
        device=torch.device("cuda")
    )

    print("Generated data label:", generated_data_bucket.label)  # often DataLabel.BOTH
    print("Number of generated samples:", len(generated_data_bucket.data_source))
    print("Condition map keys (e.g. 'dir'):", list(cond_map.keys())[:5], "...")

    # ----- EXAMPLE 2B: IMAGE-CONDITIONAL USING A LIST OF TENSORS/PATHS -----
    # Maybe you manually fetched a few image batches from your dataloader and stored them in a list:
    list_of_image_tensors = []
    for batch in train_dataloader:
        imgs = batch['image']
        for img in imgs:
            list_of_image_tensors.append(img.to("cuda"))
        if len(list_of_image_tensors) >= 3:
            break

    image_cond_bucket = DataBucket(
        data_source=list_of_image_tensors,
        label=DataLabel.IMAGE
    )

    generated_data_bucket, cond_map = diffusion_model.generate_samples(
        n_samples=6,
        scenario=ScenarioType.COND_IMAGE,
        data_bucket=image_cond_bucket,
        batch_size=4,
        vae=vae,
        device=torch.device("cuda")
    )

    print("Generated data label:", generated_data_bucket.label)
    print("Condition map example:", dict(list(cond_map.items())[:3]))

    # ----- EXAMPLE 3: TABLE-CONDITIONAL GENERATION -----
    # Suppose we gather tab data from the training dataloader into a list
    list_of_tab_tensors = []
    for i, batch in enumerate(train_dataloader):
        tabs = batch['tabular']  # shape (B, 174)
        for t in tabs:
            list_of_tab_tensors.append(t.cpu())
        if len(list_of_tab_tensors) >= 6:
            break

    tab_cond_bucket = DataBucket(
        data_source=list_of_tab_tensors,
        label=DataLabel.TAB
    )

    generated_data_bucket, cond_map = diffusion_model.generate_samples(
        n_samples=9,
        scenario=ScenarioType.COND_TABLE,
        data_bucket=tab_cond_bucket,
        batch_size=5,
        vae=vae,
        cfg=1.2,
        steps=25,
        device='cuda',
    )

    print("Generated data label:", generated_data_bucket.label)  # 'both'
    print("First condition key => indices:", next(iter(cond_map.items())))

    # ----- EXAMPLE 4A: BOTH-CONDITIONAL GENERATION -----
    list_of_pairs = []
    for i, batch in enumerate(train_dataloader):
        imgs = batch['image']  # shape (B, C, H, W)
        tabs = batch['tabular']  # shape (B, 174)
        for img, tab in zip(imgs, tabs):
            list_of_pairs.append((img.cpu(), tab.cpu()))
        if len(list_of_pairs) >= 7:
            break

    both_cond_bucket = DataBucket(
        data_source=list_of_pairs,
        label=DataLabel.BOTH
    )

    generated_data_bucket, cond_map = diffusion_model.generate_samples(
        n_samples=9,
        scenario=ScenarioType.COND_BOTH,
        data_bucket=both_cond_bucket,
        batch_size=4,
        vae=vae,
        cfg=2.0,
        steps=30,
        device='cuda'
    )

    print("cond_map:", cond_map)

    # ----- EXAMPLE 4B: BOTH-CONDITIONAL GENERATION -----
    # Suppose we create a small DataBucket from our training dataloader directly:
    both_cond_bucket = DataBucket(
        data_source=train_dataloader,  # The entire DataLoader
        label=DataLabel.BOTH
    )

    generated_data_bucket, cond_map = diffusion_model.generate_samples(
        n_samples=8,
        scenario=ScenarioType.COND_BOTH,
        data_bucket=both_cond_bucket,
        batch_size=5,
        vae=vae,
        cfg=1.2,
        steps=25,
        device='cuda',
    )

    print("cond_map:", cond_map)
