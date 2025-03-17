from functools import partial

from composer.models import ComposerModel
from easydict import EasyDict
import torch.nn as nn
import numpy as np

from typing import Any, Tuple
from omegaconf import DictConfig
import torch
from typing import List, Dict, Optional

from torch import Tensor

import torch
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any, Tuple, Union
import random

from utils.configurations import apply_overrides
import itertools

DATA_TYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

from enum import Enum

class ScenarioType(Enum):
    UNCOND = "uncond"
    COND_IMAGE = "cond_image"
    COND_TABLE = "cond_table"
    COND_BOTH = "cond_both"

class DataLabel(Enum):
    IMAGE = "image"
    TAB = "tab"
    BOTH = "both"

class DataBucket:
    """
    A container for conditional data or for the final generated samples.
    - data_source can be:
        1) A PyTorch DataLoader,
        2) A list of data items (e.g., images, tabular features, or (img, tab) pairs).
    - label: indicates what type of data is in data_source (image, tab, or both).
    """

    def __init__(self, data_source: Union[DataLoader, List[Any]], label: DataLabel):
        self.data_source = data_source
        self.label = label

    def __len__(self):
        if isinstance(self.data_source, DataLoader):
            # length in terms of #batches (not always exact). For “infinite” iteration, this is less relevant.
            return len(self.data_source)
        return len(self.data_source)


def infinite_loader(dataloader: DataLoader):
    """Create a persistent iterator that cycles through a dataloader indefinitely."""
    while True:
        for batch in dataloader:
            yield batch


class MultiModalDiffusion(ComposerModel):
    """
    Multi-modal EDM that can handle images + tabular data jointly.

    Key Points:
      - Uses EDM (Elucidated Diffusion) approach for sampling sigma, computing MSE loss.
      - Incorporates multiple 'scenarios' (unconditional vs. conditional) in training,
        so that at inference, you can choose to generate images or tables from scratch or from partial noise.
      - Maintains a structure similar to your original single‐modal diffusion code:
         * edm_loss(...) does the main noise sampling, weighting, MSE.
         * model_forward_wrapper(...) calls self.dit with c_in/c_out/c_skip logic if needed.
      - Image patch masking and tabular token masking are both handled; if the model returns
        mask information, we unmask the losses accordingly.

    Args:
        dit (nn.Module): Your DiT model that accepts:
            (x_img, x_tab, time_emb_img, time_emb_tab, cfg, cond_image, cond_table, mask_ratio_img, mask_ratio_tab)
            and returns a dict including 'sample_img', 'sample_tab', possibly 'mask_img', 'mask_tab',
            or other fields for unmasking.
        image_key (str): Batch key for images. Default: 'images'.
        table_key (str): Batch key for tables. Default: 'tables'.
        scenario (str): Batch key for scenario, e.g. 'scenario'. Default: 'scenario'.
        dtype (str): Floating dtype. Default: 'bfloat16'.
        p_mean (float): Mean of lognormal noise in EDM. Default: -0.6.
        p_std (float): Std of lognormal noise in EDM. Default: 1.2.
        train_mask_ratio_img (float): Patch masking ratio for images in training.
        train_mask_ratio_tab (float): Token masking ratio for tables in training.
    """

    def __init__(
        self,
        dit: nn.Module,
        scenario: str = 'scenario',
        dtype: str = 'bfloat16',
        p_mean: float = -0.6,
        p_std: float = 1.2,
        train_mask_ratio_img: float = 0.0,
        train_mask_ratio_tab: float = 0.0,
    ):
        super().__init__()
        self.dit = dit
        self.scenario = scenario
        self.dtype = dtype

        # EDM hyperparameters
        self.edm_config = EasyDict({
            'sigma_min': 0.002,
            'sigma_max': 80,
            'P_mean': p_mean,
            'P_std': p_std,
            'sigma_data': 0.9,
            'num_steps': 18,
            'rho': 7,
            'S_churn': 0,
            'S_min': 0,
            'S_max': float('inf'),
            'S_noise': 1
        })

        # Masking ratios used in training
        self.train_mask_ratio_img = train_mask_ratio_img
        self.train_mask_ratio_tab = train_mask_ratio_tab
        # Typically zero out in eval
        self.eval_mask_ratio_img = 0.0
        self.eval_mask_ratio_tab = 0.0

        self.randn_like = torch.randn_like  # convenience

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        In training, we:
         1) retrieve images/tables,
         2) choose the scenario (uncond, cond_image, cond_table, cond_both),
         3) compute the EDM loss.
        """
        images = batch["image"].to(DATA_TYPES[self.dtype])
        tables = batch["tabular"].to(DATA_TYPES[self.dtype])

        # If scenario is in batch, use it; else pick a default (uncond) or random approach
        scenario = batch.get(self.scenario, 'uncond')  # or custom logic if you want

        # mask ratios differ between training and eval
        mask_ratio_img = self.train_mask_ratio_img if self.training else self.eval_mask_ratio_img
        mask_ratio_tab = self.train_mask_ratio_tab if self.training else self.eval_mask_ratio_tab

        loss = self.edm_loss(images, tables, scenario, mask_ratio_img, mask_ratio_tab)
        return (loss, images, tables)

    def edm_loss(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        scenario: ScenarioType,
        mask_ratio_img: float,
        mask_ratio_tab: float,
        **kwargs
    ) -> dict[str, Tensor | Any]:
        """
        The core EDM loss:
          1) Sample sigma from lognormal,
          2) Decide how to noise image/table based on the scenario,
          3) Call the DiT via `model_forward_wrapper(...)`,
          4) Compute weighted MSE,
          5) If masking is used, unmask the MSE accordingly.
        """
        N = x_img.shape[0]

        # 1) Sample lognormal sigma
        sigma = self._sample_sigma(n=N, device=x_img.device)
        # rnd_normal = torch.randn([N, 1, 1, 1], device=x_img.device)
        # sigma = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()

        # Weight factor from EDM
        weight = (sigma ** 2 + self.edm_config.sigma_data ** 2) / ((sigma * self.edm_config.sigma_data) ** 2)

        # 2) Apply scenario noising
        x_noisy_img, x_noisy_tab = self._apply_scenario_noise(x_img, x_tab, scenario, sigma)

        # 3) Forward pass (cfg=1.0 in training, no extra guidance)
        model_out = self.model_forward_wrapper(
            x_noisy_img,
            x_noisy_tab,
            sigma,
            scenario=scenario,
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab,
            model_forward_fxn=partial(self.dit.forward, cfg=1.0),
        )

        # Denoised outputs
        denoised_img = model_out['sample_img']
        denoised_tab = model_out['sample_tab']

        # 4) Weighted MSE
        # -- images
        loss_img = weight * (denoised_img - x_img) ** 2
        # average over (C,H,W)
        loss_img_unmasked = loss_img.mean(dim=(1, 2, 3))

        # -- tables
        weight_tab = weight.view(-1, 1)  # shape (N,1)
        loss_tab = weight_tab * (denoised_tab - x_tab) ** 2
        loss_tab_unmasked = loss_tab.mean(dim=1)

        final_loss_img = loss_img_unmasked
        final_loss_tab = loss_tab_unmasked

        # If model returns image mask => unmask:
        if mask_ratio_img > 0.0 and 'mask_img' in model_out:
            mask_img = model_out['mask_img']  # shape (N, H'*W') or (N, H', W')
            # Replicate the single‐modal approach of averaging MSE across unmasked patches
            patch_size = getattr(self.dit, 'patch_size', 4)
            # 1) MSE across channels => shape (N,1,H,W)
            mse_per_pixel = (denoised_img - x_img).square().mean(dim=1, keepdim=True)
            # 2) avg_pool2d to patch scale => shape (N,1,H',W')
            mse_per_patch = F.avg_pool2d(mse_per_pixel, kernel_size=patch_size).squeeze(1)
            # Flatten => (N, #patches)
            mse_per_patch = mse_per_patch.flatten(start_dim=1)

            unmask_img = 1.0 - mask_img.flatten(start_dim=1)  # shape (N, #patches)
            unmasked_mse_img = (mse_per_patch * unmask_img).sum(dim=1) / unmask_img.sum(dim=1)
            final_loss_img = unmasked_mse_img

        # If model returns table mask => unmask:
        if mask_ratio_tab > 0.0 and 'mask_tab' in model_out:
            mask_tab = model_out['mask_tab']  # shape (N, D)
            # Weighted MSE => (N,D)
            mse_per_token = weight_tab * (denoised_tab - x_tab).square()
            unmask_tab = 1.0 - mask_tab
            unmasked_mse_tab = (mse_per_token * unmask_tab).sum(dim=1) / unmask_tab.sum(dim=1)
            final_loss_tab = unmasked_mse_tab

        # Combine final losses
        loss_per_sample = 0.5 * (final_loss_img + final_loss_tab)
        return {"loss": loss_per_sample.mean(), "loss_img": final_loss_img.mean(), "loss_tab": final_loss_tab.mean()}

    def model_forward_wrapper(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        sigma: torch.Tensor,
        scenario: ScenarioType,
        mask_ratio_img: float,
        mask_ratio_tab: float,
        model_forward_fxn,
        **kwargs
    ) -> dict:
        """
        This is the wrapper that, if needed, can apply c_skip/c_in/c_out logic (like in EDM).
        However, since you said you pass the *same* time embedding to each data, we won't do
        separate c_skip for each modality. We'll apply the standard c_in approach as in your single-late code,
        but do so for the image input. For the tabular part, we also pass the same time embedding
        if your DiT expects that.

        If your model does per-token or per-pixel normalization with c_skip/c_out, you'll replicate that.
        Here, we'll just show how you might do it if the model is expecting:
           net(img, tab, time_emb_img, time_emb_tab, cfg=..., cond_image=..., cond_table=..., mask_ratio_img=..., mask_ratio_tab=...)
        """
        sigma_data = self.edm_config.sigma_data

        c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_skip = (sigma_data ** 2) / (sigma_data ** 2 + sigma ** 2)
        c_out = (sigma_data * sigma) / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_noise = sigma.log() / 4.0
        time_scalar = c_noise.flatten()  # shape (N,)

        # scale inputs
        x_img_in = c_in * x_img
        x_tab_in = c_in.squeeze(-1).squeeze(-1) * x_tab  # shape (N,D)

        # Forward pass
        out = model_forward_fxn(
            x_img_in,
            x_tab_in,
            time_scalar,
            time_scalar,
            cond_image=(scenario in ['cond_image', 'cond_both']),
            cond_table=(scenario in ['cond_table', 'cond_both']),
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab,
            **kwargs
        )
        # Expect out: {'sample_img', 'sample_tab', 'mask_img'?, 'mask_tab'?}

        # Recombine the model outputs with c_skip, c_out
        F_img = out['sample_img']
        F_tab = out['sample_tab']

        # Ensure consistent device
        device = x_img.device
        F_img = F_img.to(device)
        F_tab = F_tab.to(device)
        c_skip = c_skip.to(device)
        c_out = c_out.to(device)

        # Denoised image
        denoised_img = c_skip * x_img + c_out * F_img

        # Denoised table (reshape c_skip, c_out => (N,1) to match (N,D))
        c_skip_tab = c_skip.view(-1)
        c_out_tab = c_out.view(-1)
        denoised_tab = c_skip_tab.unsqueeze(-1) * x_tab + c_out_tab.unsqueeze(-1) * F_tab

        out['sample_img'] = denoised_img
        out['sample_tab'] = denoised_tab
        return out

    def _apply_scenario_noise(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        scenario: ScenarioType,
        sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine how to noise each modality depending on scenario.
        """
        if scenario == 'uncond':
            x_noisy_img = x_img + self.randn_like(x_img) * sigma
            x_noisy_tab = x_tab + torch.randn_like(x_tab) * sigma.view(-1,1)

        elif scenario == 'cond_image':
            # keep image nearly unnoised
            x_noisy_img = x_img
            x_noisy_tab = x_tab + torch.randn_like(x_tab) * sigma.view(-1,1)

        elif scenario == 'cond_table':
            x_noisy_img = x_img + self.randn_like(x_img) * sigma
            x_noisy_tab = x_tab

        else:  # 'cond_both'
            scale = 0.5
            x_noisy_img = x_img + self.randn_like(x_img) * (sigma * scale)
            x_noisy_tab = x_tab + torch.randn_like(x_tab) * (sigma.view(-1,1) * scale)

        return x_noisy_img, x_noisy_tab

    # ------------------------------
    # Composer integration
    # ------------------------------

    def loss(self, outputs, batch) -> torch.Tensor:
        # The first item in outputs is the scalar loss
        return outputs[0]

    def eval_forward(self, batch, outputs: Optional[tuple] = None):
        # If outputs is already computed, just return it.
        # Otherwise, do a forward pass.
        if outputs is not None:
            return outputs
        return self.forward(batch)

    def get_metrics(self, is_train: bool = False):
        """
        Return a dict of any metrics you want to track.
        If you don't want to track anything, return {}.
        """
        # For instance, you could return something like:
        # return {'loss': DistLoss()} if you had a DistLoss aggregator.
        return {}

    def update_metric(self, batch, outputs, metric):
        """
        If you do have a metric aggregator, you'd update it here.
        For a simple scenario, we do nothing.
        """
        pass

    @torch.no_grad()
    def _edm_sampler_loop(
            self,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            scenario: ScenarioType,
            steps: Optional[int] = None,
            cfg: float = 1.0,
            **kwargs
    ) -> (torch.Tensor, torch.Tensor):
        """

        We do:
          1) Time step discretization (t_steps).
          2) For each t_cur -> t_next:
             - Possibly add temporary noise (churn).
             - Euler step (first-order).
             - 2nd-order correction step.
          3) Return the final x_img, x_tab.

        Args:
            x_img (torch.Tensor): Initial latents for images (N, C, H, W).
            x_tab (torch.Tensor): Initial latents for tables (N, D).
            scenario (str): 'uncond', 'cond_image', 'cond_table', or 'cond_both'.
            steps (int, optional): Number of diffusion steps. Defaults to `self.edm_config.num_steps`.
            cfg (float, optional): Guidance scale for classifier-free guidance.
                                   If <= 1.0, no extra guidance is applied.
            **kwargs: Extra arguments forwarded to `model_forward_wrapper`.

        Returns:
            (x_img_out, x_tab_out): The final denoised latents for each modality.
        """
        # No masking during generation
        mask_ratio_img = 0.0
        mask_ratio_tab = 0.0

        # Choose the correct DiT forward function (with or without CFG).
        model_forward_fxn = (
            partial(self.dit.forward, cfg=cfg)
            if cfg > 1.0 else self.dit.forward
        )

        # Time steps
        num_steps = self.edm_config.num_steps if steps is None else steps
        device = x_img.device

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        # t_steps shaped (num_steps,) + final extra 0
        t_steps = (
                          self.edm_config.sigma_max ** (1 / self.edm_config.rho)
                          + step_indices / (num_steps - 1)
                          * (
                                  self.edm_config.sigma_min ** (1 / self.edm_config.rho)
                                  - self.edm_config.sigma_max ** (1 / self.edm_config.rho)
                          )
                  ) ** self.edm_config.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # e.g.  [t_0, ..., t_{N-1}, 0]

        # We'll track x_img, x_tab in float64 for numeric precision
        x_img_next = x_img.to(torch.float64) * t_steps[0]
        x_tab_next = x_tab.to(torch.float64) * t_steps[0]

        # Main sampling loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_img_cur = x_img_next
            x_tab_cur = x_tab_next

            # Increase noise temporarily (EDM "churn") if t_cur in [S_min, S_max]
            gamma = (
                min(self.edm_config.S_churn / num_steps, np.sqrt(2) - 1)
                if (self.edm_config.S_min <= t_cur <= self.edm_config.S_max)
                else 0
            )
            t_hat_val = t_cur + gamma * t_cur
            t_hat = torch.as_tensor(t_hat_val, device=device).reshape(1)

            # Add noise if gamma > 0
            if gamma > 0:
                noise_scale = (t_hat_val**2 - t_cur**2).sqrt() * self.edm_config.S_noise
                # random normal
                x_img_hat = x_img_cur + noise_scale * torch.randn_like(x_img_cur)
                x_tab_hat = x_tab_cur + noise_scale * torch.randn_like(x_tab_cur)
            else:
                x_img_hat = x_img_cur
                x_tab_hat = x_tab_cur

            # -----------------------
            # Euler step
            # -----------------------
            denoised = self.model_forward_wrapper(
                x_img_hat.to(torch.float32),
                x_tab_hat.to(torch.float32),
                t_hat.to(torch.float32),
                scenario=scenario,
                mask_ratio_img=mask_ratio_img,
                mask_ratio_tab=mask_ratio_tab,
                model_forward_fxn=model_forward_fxn,
                **kwargs
            )
            F_img = denoised['sample_img'].to(torch.float64)
            F_tab = denoised['sample_tab'].to(torch.float64)

            d_img_cur = (x_img_hat - F_img) / t_hat_val
            d_tab_cur = (x_tab_hat - F_tab) / t_hat_val

            x_img_next = x_img_hat + (t_next - t_hat_val) * d_img_cur
            x_tab_next = x_tab_hat + (t_next - t_hat_val) * d_tab_cur

            # -----------------------
            # 2nd order correction
            # -----------------------
            if i < num_steps - 1:
                denoised_2 = self.model_forward_wrapper(
                    x_img_next.to(torch.float32),
                    x_tab_next.to(torch.float32),
                    t_next.to(torch.float32),
                    scenario=scenario,
                    mask_ratio_img=mask_ratio_img,
                    mask_ratio_tab=mask_ratio_tab,
                    model_forward_fxn=model_forward_fxn,
                    **kwargs
                )
                F_img2 = denoised_2['sample_img'].to(torch.float64)
                F_tab2 = denoised_2['sample_tab'].to(torch.float64)

                d_img_prime = (x_img_next - F_img2) / t_next
                d_tab_prime = (x_tab_next - F_tab2) / t_next

                x_img_next = x_img_hat + (t_next - t_hat_val) * (0.5 * d_img_cur + 0.5 * d_img_prime)
                x_tab_next = x_tab_hat + (t_next - t_hat_val) * (0.5 * d_tab_cur + 0.5 * d_tab_prime)

        # Return final latents as float32
        x_img_final = x_img_next.to(torch.float32)
        x_tab_final = x_tab_next.to(torch.float32)
        return x_img_final, x_tab_final

    # @torch.no_grad()
    # def generate(
    #         self,
    #         scenario: str = 'uncond',
    #         image_init: Optional[torch.Tensor] = None,
    #         table_init: Optional[torch.Tensor] = None,
    #         guidance_scale: float = 1.0,
    #         num_inference_steps: int = 30,
    #         seed: Optional[int] = None,
    #         device: Optional[torch.device] = None,
    #         **kwargs
    # ):
    #     """
    #     High-level generation method that sets up latents for each scenario,
    #     then calls edm_sampler_loop. Similar to 'generate' in your single‐modal code,
    #     but now we have multi‐modal data.
    #
    #     Args:
    #         scenario (str): 'uncond', 'cond_image', 'cond_table', 'cond_both'.
    #         image_init (torch.Tensor, optional): If scenario involves an image condition,
    #                                              pass your initial image latents here.
    #         table_init (torch.Tensor, optional): If scenario involves a table condition,
    #                                              pass your initial table data here.
    #         guidance_scale (float): CFG scale.
    #         num_inference_steps (int): # of EDM steps in sampling.
    #         seed (int, optional): RNG seed for reproducible noise.
    #         device (torch.device, optional): Torch device. If None, we infer from self.dit.
    #
    #     Returns:
    #         (final_img, final_tab): The final latents for each modality after sampling.
    #     """
    #
    #     # Decide device
    #     if device is None:
    #         device = next(self.dit.parameters()).device
    #
    #     # Prepare a torch Generator if needed
    #     rng_generator = torch.Generator(device=device)
    #     if seed is not None:
    #         rng_generator.manual_seed(seed)
    #
    #     default_batch = 4
    #     shape_img = (default_batch, 4, 32, 32)
    #     shape_tab = (default_batch, 174)
    #
    #     # 1) Initialize latents according to scenario
    #     if scenario == 'uncond':
    #         x_img = torch.randn(shape_img, device=device, generator=rng_generator)
    #         x_tab = torch.randn(shape_tab, device=device, generator=rng_generator)
    #
    #     elif scenario == 'cond_image':
    #         if image_init is None:
    #             raise ValueError("Must provide image_init for cond_image scenario.")
    #         shape_img = image_init.shape
    #         batch_size = shape_img[0]
    #         x_img = image_init.to(device)
    #         # random for table => same batch size
    #         shape_tab = (batch_size, 174)
    #         x_tab = torch.randn(shape_tab, device=device, generator=rng_generator)
    #
    #     elif scenario == 'cond_table':
    #         if table_init is None:
    #             raise ValueError("Must provide table_init for cond_table scenario.")
    #         shape_tab = table_init.shape  # (B,174)
    #         batch_size = shape_tab[0]
    #         x_tab = table_init.to(device)
    #         # random for image
    #         shape_img = (batch_size, 4, 32, 32)
    #         x_img = torch.randn(shape_img, device=device, generator=rng_generator)
    #
    #     else:  # 'cond_both'
    #         if image_init is None or table_init is None:
    #             raise ValueError("Must provide image_init and table_init for scenario='cond_both'.")
    #         shape_img = image_init.shape  # (N,4,32,32)
    #         shape_tab = table_init.shape  # (N,174)
    #         # Optionally check they have the same batch size:
    #         if shape_img[0] != shape_tab[0]:
    #             raise ValueError(f"In cond_both, mismatch in batch size: {shape_img[0]} vs {shape_tab[0]}")
    #
    #         # Partially noise both
    #         scale = 0.5
    #         x_img = image_init.to(device) + scale * torch.randn(shape_img, device=device, generator=rng_generator)
    #         x_tab = table_init.to(device) + scale * torch.randn(shape_tab, device=device, generator=rng_generator)
    #
    #     # 2) Call the sampler loop
    #     final_img, final_tab = self.edm_sampler_loop(
    #         x_img,
    #         x_tab,
    #         scenario=scenario,
    #         steps=num_inference_steps,
    #         cfg=guidance_scale,
    #         **kwargs
    #     )
    #
    #     return final_img, final_tab

    @torch.no_grad()
    def generate_samples(
            self,
            n_samples: int,
            scenario: ScenarioType,
            data_bucket: Optional[DataBucket] = None,
            batch_size: int = 4,
            vae: nn.Module = None,
            device: torch.device = torch.device("cuda"),
    ) -> Tuple[DataBucket, Optional[Dict[Any, List[int]]]]:
        """
        High-level entry point that dispatches to unconditional or conditional generation.
        Returns:
          - A DataBucket holding the *pixel-space* generated images + tabular data (if both).
          - A mapping (dict) from "condition key" -> list of generated sample indices, or None if UNCOND.

        Generate samples according to the given scenario. Optionally uses
        conditional data from a DataBucket (which can be a dataloader or a list).

        Usage Scenarios:
        1) Unconditional generation:
           - scenario='uncond'
           - data_bucket=None
           => returns n_samples of purely random generation from the model.

        2) Conditional on images only:
           - scenario='cond_image'
           - data_bucket.label must be 'image' or 'both'
           => The method will repeatedly sample from data_bucket to build
              each batch of conditioning images, until n_samples are reached.

        3) Conditional on tables only:
           - scenario='cond_table'
           - data_bucket.label must be 'tab' or 'both'
           => The method will repeatedly sample from data_bucket for tabular data.

        4) Conditional on both images + tables:
           - scenario='cond_both'
           - data_bucket.label must be 'both'
           => The method will repeatedly sample from data_bucket, which must
              provide pairs of (image, table) data.

        Args:
            n_samples (int): How many total samples to generate.
            scenario (str): One of {'uncond', 'cond_image', 'cond_table', 'cond_both'}.
            batch_size (int): How many samples in each generation batch.
            data_bucket (Optional[DataBucket]):
                The conditional data source if scenario != 'uncond'.
                If scenario='uncond', this must be None.
            guidance_scale (float): Classifier-free guidance scale.
            num_inference_steps (int): Number of EDM steps in sampling.
            seed (int, optional): Random seed for reproducibility.
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

                If the `data_bucket` had a list of paths, the keys will be those
                paths. If the `data_bucket` had a list of objects, the keys will be
                the *indices* in that list. If the `data_bucket` was a dataloader
                with e.g. patient directories, the keys might be the directory
                strings, etc.
        """

        if scenario == ScenarioType.UNCOND:
            if data_bucket is not None:
                raise ValueError("For UNCOND scenario, data_bucket must be None.")
            return self._generate_samples_unconditional(
                n_samples=n_samples,
                batch_size=batch_size,
                vae=vae,
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
                device=device
            )

    @torch.no_grad()
    def _generate_samples_unconditional(
            self,
            n_samples: int,
            batch_size: int,
            vae: nn.Module,
            device: torch.device,
    ) -> Tuple[DataBucket, None]:
        """Generate random latents from scratch, decode to pixel space, return as DataBucket."""
        all_images = []
        all_tables = []
        total_generated = 0

        while total_generated < n_samples:
            current_bsz = min(batch_size, n_samples - total_generated)

            # (A) Generate random pixel images (completely random) or directly random latents.
            # Usually we do random latents, then pass them to the sampler.
            # We'll do it by making random latents of shape e.g. (B, 4, H_lat, W_lat).
            # For demonstration, let's say  (4, 32, 32) is your latent shape:
            # If your DIT expects a certain shape, adjust accordingly.

            latent_shape_img = (current_bsz, 4, 32, 32)  # example latent shape
            latent_shape_tab = (current_bsz, 174)  # example table dimension

            x_img = torch.randn(latent_shape_img, device=device)
            x_tab = torch.randn(latent_shape_tab, device=device)

            # (B) Sample sigma from lognormal
            sigma = self._sample_sigma(n=current_bsz, device=device)
            # (C) Apply scenario noise => in uncond, we add noise to both, but we are
            # already in latent space. Let's call it for consistency:
            x_img, x_tab = self._apply_scenario_noise(x_img, x_tab, ScenarioType.UNCOND, sigma)

            # (D) EDM Sampler loop. Imagine your existing code that refines x_img, x_tab.
            final_latents_img, final_latents_tab = self._edm_sampler_loop(
                x_img,
                x_tab,
                scenario=ScenarioType.UNCOND
            )

            # (E) Decode latents back to pixel images (and keep table as is or interpret it).
            decoded_imgs = decode_latents(vae, final_latents_img, vae.config.scaling_factor)
            # Store results
            all_images.append(decoded_imgs)
            all_tables.append(final_latents_tab)  # maybe your table is also in latent space,
            # or directly the final tab data.

            total_generated += current_bsz

        # Concatenate
        final_imgs = torch.cat(all_images, dim=0)[:n_samples]
        final_tabs = torch.cat(all_tables, dim=0)[:n_samples]

        # Build a DataBucket. Typically you produce "both" if you always generate image+tab
        # or just "image" if your uncond scenario only yields images. Adjust as needed.
        out_data = []
        for i in range(n_samples):
            out_data.append((final_imgs[i], final_tabs[i]))

        result_bucket = DataBucket(data_source=out_data, label=DataLabel.BOTH)

        # For unconditional scenario, there's no condition mapping
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
    ) -> Tuple[DataBucket, Dict[Any, List[int]]]:
        """
        Generate samples conditioned on images, tables, or both.
        We fetch data from data_bucket (which might be a DataLoader or a list).
        Returns:
          - DataBucket with final pixel images & tab data.
          - A dict: condition_key -> list of sample indices.
        """
        # 1) Prepare persistent iterator if data_source is a DataLoader
        if isinstance(data_bucket.data_source, DataLoader):
            cond_iter = infinite_loader(data_bucket.data_source)
        else:
            cond_iter = None  # we'll pick randomly from a list

        cond_mapping: Dict[Any, List[int]] = {}
        all_images = []
        all_tables = []
        total_generated = 0
        global_index = 0

        while total_generated < n_samples:
            current_bsz = min(batch_size, n_samples - total_generated)

            # 2) Acquire a batch of data from the bucket
            if cond_iter is not None:
                # Dataloader path
                batch = next(cond_iter)
                # keys: batch['image'], batch['tabular'], batch['dir']
                cond_images = batch['image'][:current_bsz].to(device)
                cond_tables = batch['tabular'][:current_bsz].to(device)
                cond_dirs = batch['dir'][:current_bsz]  # typically list of str

                # Make the "condition keys" for the mapping
                # e.g., we'll just use 'dir' strings
                condition_keys = cond_dirs
            else:
                # We have a list
                ds = data_bucket.data_source
                ds_size = len(ds)
                # pick random indices
                idxs = torch.randint(0, ds_size, (current_bsz,), generator=torch.Generator(device=device))

                cond_images = []
                cond_tables = []
                condition_keys = []

                for i_idx in idxs:
                    i_idx = i_idx.item()
                    item = ds[i_idx]  # item might be an image, or (image, tab), etc.
                    # If scenario=COND_IMAGE => item is an image
                    # If scenario=COND_TABLE => item is table
                    # If scenario=COND_BOTH => item is (image, table)

                    # Use i_idx as the key if no path is available
                    key = i_idx

                    if scenario == ScenarioType.COND_IMAGE:
                        cond_images.append(item)  # item is presumably pixel image
                        cond_tables.append(None)  # we'll fill a placeholder
                    elif scenario == ScenarioType.COND_TABLE:
                        cond_images.append(None)
                        cond_tables.append(item)
                    else:  # scenario == ScenarioType.COND_BOTH
                        # item is a tuple (pixel_image, tab_data)
                        cond_images.append(item[0])
                        cond_tables.append(item[1])

                    condition_keys.append(key)

                # Convert to Tensors if not already
                # ignoring None in the list for cond_images or cond_tables
                # we do something like:
                cond_images = [x for x in cond_images if x is not None]
                cond_tables = [x for x in cond_tables if x is not None]
                if len(cond_images) > 0:
                    cond_images = torch.stack(cond_images, dim=0).to(device)
                else:
                    # If scenario=COND_TABLE, no images at all
                    cond_images = None
                if len(cond_tables) > 0:
                    cond_tables = torch.stack(cond_tables, dim=0).to(device)
                else:
                    cond_tables = None

            # 3) Encode images if scenario != COND_TABLE
            if scenario in (ScenarioType.COND_IMAGE, ScenarioType.COND_BOTH):
                # cond_images shape: (current_bsz, 3, H, W) for pixel images
                latents_img = encode_images(vae, cond_images, vae.config.scaling_factor)
            else:
                # cond_tables only => generate random latents for image
                # or you might do a zero latents, etc., depending on your logic
                latent_shape_img = (current_bsz, 4, 32, 32)
                latents_img = torch.randn(latent_shape_img, device=device)

            # 4) If scenario != COND_IMAGE, we have actual table data in cond_tables
            if scenario in (ScenarioType.COND_TABLE, ScenarioType.COND_BOTH):
                latents_tab = cond_tables
            else:
                # scenario=COND_IMAGE => random table latents
                latents_tab = torch.randn((current_bsz, 128), device=device)

            # 5) Sample sigma from lognormal
            sigma = self._sample_sigma(n=current_bsz, device=device)

            # 6) Apply scenario-specific noise
            latents_img_noisy, latents_tab_noisy = self._apply_scenario_noise(
                latents_img, latents_tab, scenario, sigma
            )

            # 7) EDM Sampler
            final_latents_img, final_latents_tab = self._edm_sampler_loop(
                latents_img_noisy,
                latents_tab_noisy,
                scenario=scenario
            )

            # 8) Decode images from final latents if needed
            decoded_imgs = decode_latents(vae, final_latents_img, vae.config.scaling_factor)

            # 9) Accumulate
            all_images.append(decoded_imgs)
            all_tables.append(final_latents_tab)

            # 10) Update cond_mapping
            for i, ck in enumerate(condition_keys):
                if ck not in cond_mapping:
                    cond_mapping[ck] = []
                cond_mapping[ck].append(global_index + i)

            global_index += current_bsz
            total_generated += current_bsz

        # Collate final
        final_imgs = torch.cat(all_images, dim=0)[:n_samples]
        final_tabs = torch.cat(all_tables, dim=0)[:n_samples]

        out_data = []
        for i in range(n_samples):
            out_data.append((final_imgs[i], final_tabs[i]))

        # Build a data bucket
        result_bucket = DataBucket(data_source=out_data, label=DataLabel.BOTH)

        return result_bucket, cond_mapping

    def _sample_sigma(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Sample sigma from a lognormal distribution:
          sigma = exp( normal * P_std + P_mean ).
        Example:  P_mean=-0.6, P_std=1.2
        Shape: (n, 1, 1, 1), or whatever you require for broadcasting
        """
        rnd_normal = torch.randn([n, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.edm_config["P_std"] + self.edm_config["P_mean"]).exp()
        return sigma

    # @torch.no_grad()
    # def generate_samples(
    #         self,
    #         n_samples: int,
    #         scenario: str = 'uncond',
    #         batch_size: int = 4,
    #         data_bucket: Optional[DataBucket] = None,
    #         guidance_scale: float = 1.0,
    #         num_inference_steps: int = 30,
    #         seed: Optional[int] = None,
    #         device: Optional[torch.device] = None,
    #         # shape of the output images and tables (can be changed as needed)
    #         img_shape: Tuple[int, int, int] = (4, 32, 32),
    #         tab_shape: int = 174,
    # ) -> Tuple[DataBucket, Optional[Dict[Any, List[int]]]]:
    #     """
    #
    #     """
    #
    #     # ---------------------------------------------------------------------
    #     # 0. Preliminary checks and set up
    #     # ---------------------------------------------------------------------
    #     scenarios_allowed = {'uncond', 'cond_image', 'cond_table', 'cond_both'}
    #     if scenario not in scenarios_allowed:
    #         raise ValueError(
    #             f"scenario must be one of {scenarios_allowed}, got: {scenario}"
    #         )
    #
    #     # If unconditional, we do not expect a DataBucket
    #     if scenario == 'uncond' and data_bucket is not None:
    #         raise ValueError(
    #             "For scenario='uncond', data_bucket must be None."
    #         )
    #     # If conditional, we do expect a DataBucket
    #     if scenario != 'uncond' and data_bucket is None:
    #         raise ValueError(
    #             f"For scenario='{scenario}', you must provide a DataBucket."
    #         )
    #
    #     # Additional checks on data_bucket.label
    #     if scenario == 'cond_image':
    #         if data_bucket and data_bucket.label not in {'image', 'both'}:
    #             raise ValueError(
    #                 "DataBucket.label must be 'image' or 'both' "
    #                 f"when scenario='{scenario}'. Got: {data_bucket.label}"
    #             )
    #     if scenario == 'cond_table':
    #         if data_bucket and data_bucket.label not in {'tab', 'both'}:
    #             raise ValueError(
    #                 "DataBucket.label must be 'tab' or 'both' "
    #                 f"when scenario='{scenario}'. Got: {data_bucket.label}"
    #             )
    #     if scenario == 'cond_both':
    #         if data_bucket and data_bucket.label != 'both':
    #             raise ValueError(
    #                 "DataBucket.label must be 'both' when scenario='cond_both'. "
    #                 f"Got: {data_bucket.label}"
    #             )
    #
    #     # Prepare an RNG if needed
    #     rng_generator = torch.Generator(device=device)
    #     if seed is not None:
    #         rng_generator.manual_seed(seed)
    #
    #     # Prepare accumulators
    #     all_imgs = []
    #     all_tabs = []
    #
    #     # A mapping from "conditioning key" -> list of generated sample indices
    #     # This key might be a path, an integer index, or something from the dataloader.
    #     cond_mapping: Dict[Any, List[int]] = {}
    #
    #     total_generated = 0
    #     global_sample_index = 0  # indexes each newly generated sample
    #
    #     # ---------------------------------------------------------------------
    #     # 1. Function to fetch one batch of data from the DataBucket
    #     # ---------------------------------------------------------------------
    #     def fetch_conditional_batch(
    #             bucket: DataBucket,
    #             current_bsz: int
    #     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[Any]]:
    #         """
    #         Returns (cond_image_batch, cond_table_batch, condition_keys).
    #         condition_keys is a list that identifies each item (e.g. paths or indices)
    #         so we can store them in the mapping.
    #         - If scenario='cond_image', tab == None
    #         - If scenario='cond_table', img == None
    #         - If scenario='cond_both', both are filled
    #         """
    #
    #         # For simplicity, we will pick random samples from the entire bucket if it's a list.
    #         # If it's a dataloader, we iterate. The approach can vary based on your preference.
    #
    #         if isinstance(bucket.data_source, DataLoader):
    #             # We'll take a random batch from the dataloader by re-iterating or randomizing
    #             # NOTE: For huge data, you might want a persistent iterator or handle StopIteration
    #             data_iter = iter(bucket.data_source)
    #             batch = next(data_iter)  # in real code, handle StopIteration properly
    #
    #             # We'll assume 'cond_image' is in batch if label has 'image',
    #             # and 'cond_tab' is in batch if label has 'tab'.
    #             # If your dataloader uses different keys, adapt accordingly.
    #             if scenario == 'cond_image':
    #                 cond_img = batch['image'][:current_bsz].to(device)
    #                 cond_tab = None
    #             elif scenario == 'cond_table':
    #                 cond_img = None
    #                 cond_tab = batch['cond_tab'][:current_bsz].to(device)
    #             else:  # 'cond_both'
    #                 cond_img = batch['image'][:current_bsz].to(device)
    #                 cond_tab = batch['tabular'][:current_bsz].to(device)
    #
    #             # For the "key", suppose the dataloader includes a "key" or "dir" field
    #             # describing each data item:
    #             condition_keys = batch.get('dir', [None] * current_bsz)
    #             # Trim if more than current_bsz
    #             condition_keys = condition_keys[:current_bsz]
    #
    #         else:
    #             # It's a list of items or paths. We'll pick random indices.
    #             ds = bucket.data_source
    #             ds_size = len(ds)
    #             idxs = torch.randint(low=0, high=ds_size, size=(current_bsz,), generator=rng_generator)
    #
    #             cond_img = None
    #             cond_tab = None
    #             condition_keys = []
    #
    #             for i in idxs:
    #                 i = i.item()
    #                 item = ds[i]
    #
    #                 # If label='image', item might be an image path or an image array
    #                 # If label='tab', item might be a table path or a table array
    #                 # If label='both', item might be (img_obj, tab_obj)
    #
    #                 # We also store the "key" as either the path (if it's a path)
    #                 # or the integer index (if it's a raw object).
    #                 # Adapt this to however your data is structured.
    #                 if isinstance(item, str):
    #                     # We assume it's a path, so the key is the path
    #                     key = item
    #                 else:
    #                     # Otherwise, it's some data object, so we just keep the index i
    #                     key = i
    #
    #                 condition_keys.append(key)
    #
    #                 if scenario == 'cond_image':
    #                     # Suppose item is an image or path. Load or convert to tensor if needed
    #                     # This simplistic code just assumes item is already a tensor
    #                     # Real code might do: cond_img_tensor = load_image(item)
    #                     # but here we just do:
    #                     img_tensor = (item if isinstance(item, torch.Tensor)
    #                                   else torch.tensor(item, dtype=torch.float, device=device))
    #                     img_tensor = img_tensor.unsqueeze(0)  # shape => (1, C, H, W)
    #                     if cond_img is None:
    #                         cond_img = img_tensor
    #                     else:
    #                         cond_img = torch.cat([cond_img, img_tensor], dim=0)
    #
    #                 elif scenario == 'cond_table':
    #                     tab_tensor = (item if isinstance(item, torch.Tensor)
    #                                   else torch.tensor(item, dtype=torch.float, device=device))
    #                     tab_tensor = tab_tensor.unsqueeze(0)  # shape => (1, tab_shape)
    #                     if cond_tab is None:
    #                         cond_tab = tab_tensor
    #                     else:
    #                         cond_tab = torch.cat([cond_tab, tab_tensor], dim=0)
    #
    #                 else:  # scenario == 'cond_both'
    #                     # item might be a tuple (img_data, tab_data)
    #                     # Adjust to your actual structure
    #                     img_data, tab_data = item
    #                     img_tensor = (img_data if isinstance(img_data, torch.Tensor)
    #                                   else torch.tensor(img_data, dtype=torch.float, device=device))
    #                     tab_tensor = (tab_data if isinstance(tab_data, torch.Tensor)
    #                                   else torch.tensor(tab_data, dtype=torch.float, device=device))
    #                     img_tensor = img_tensor.unsqueeze(0)
    #                     tab_tensor = tab_tensor.unsqueeze(0)
    #                     if cond_img is None:
    #                         cond_img = img_tensor
    #                         cond_tab = tab_tensor
    #                     else:
    #                         cond_img = torch.cat([cond_img, img_tensor], dim=0)
    #                         cond_tab = torch.cat([cond_tab, tab_tensor], dim=0)
    #
    #         return cond_img, cond_tab, condition_keys
    #
    #     # ---------------------------------------------------------------------
    #     # 2. Main loop: keep generating until n_samples are reached
    #     # ---------------------------------------------------------------------
    #     while total_generated < n_samples:
    #         current_bsz = min(batch_size, n_samples - total_generated)
    #
    #         # (A) For unconditional scenario
    #         if scenario == 'uncond':
    #             # Random latents as a starting point
    #             x_img = torch.randn(
    #                 (current_bsz,) + img_shape, device=device, generator=rng_generator
    #             )
    #             x_tab = torch.randn(
    #                 (current_bsz, tab_shape), device=device, generator=rng_generator
    #             )
    #
    #             # Sample from EDM
    #             final_img, final_tab = self.edm_sampler_loop(
    #                 x_img, x_tab,
    #                 scenario='uncond',
    #                 steps=num_inference_steps,
    #                 cfg=guidance_scale
    #             )
    #
    #             # Accumulate
    #             all_imgs.append(final_img)
    #             all_tabs.append(final_tab)
    #             total_generated += current_bsz
    #             global_sample_index += current_bsz
    #
    #         # (B) For conditional scenarios
    #         else:
    #             # 1) Grab condition data for this batch
    #             cond_img, cond_tab, condition_keys = fetch_conditional_batch(
    #                 data_bucket, current_bsz
    #             )
    #
    #             # 2) Initialize latents for the scenario
    #             if scenario == 'cond_image':
    #                 # The image is the condition; table is random
    #                 shape_img = cond_img.shape
    #                 x_img = cond_img.to(device)  # no extra noise added at start
    #                 shape_tab = (current_bsz, tab_shape)
    #                 x_tab = torch.randn(shape_tab, device=device, generator=rng_generator)
    #
    #                 final_img, final_tab = self.edm_sampler_loop(
    #                     x_img, x_tab,
    #                     scenario='cond_image',
    #                     steps=num_inference_steps,
    #                     cfg=guidance_scale
    #                 )
    #
    #             elif scenario == 'cond_table':
    #                 # The table is the condition; image is random
    #                 shape_tab = cond_tab.shape
    #                 x_tab = cond_tab.to(device)
    #                 shape_img = (current_bsz,) + img_shape
    #                 x_img = torch.randn(shape_img, device=device, generator=rng_generator)
    #
    #                 final_img, final_tab = self.edm_sampler_loop(
    #                     x_img, x_tab,
    #                     scenario='cond_table',
    #                     steps=num_inference_steps,
    #                     cfg=guidance_scale
    #                 )
    #
    #             else:  # scenario == 'cond_both'
    #                 # Both image and table are the condition
    #                 shape_img = cond_img.shape
    #                 shape_tab = cond_tab.shape
    #                 # For example, you might add small noise if you want partial noise:
    #                 # But we said we remove user-level partial noise control. The
    #                 # training code might do partial noise internally. So here we
    #                 # just pass them directly or add minimal noise as in your original code.
    #                 x_img = cond_img.to(device)
    #                 x_tab = cond_tab.to(device)
    #
    #                 final_img, final_tab = self.edm_sampler_loop(
    #                     x_img, x_tab,
    #                     scenario='cond_both',
    #                     steps=num_inference_steps,
    #                     cfg=guidance_scale
    #                 )
    #
    #             # 3) Accumulate results
    #             all_imgs.append(final_img)
    #             all_tabs.append(final_tab)
    #
    #             # 4) Update the cond_mapping so that each condition key gets
    #             #    mapped to the newly generated sample indices
    #             #    e.g. if condition_keys=[keyA, keyB, ...] for this batch,
    #             #    then cond_mapping[keyA] = [global_sample_index], ...
    #             for i, ck in enumerate(condition_keys):
    #                 if ck not in cond_mapping:
    #                     cond_mapping[ck] = []
    #                 cond_mapping[ck].append(global_sample_index + i)
    #
    #             total_generated += current_bsz
    #             global_sample_index += current_bsz
    #
    #     # ---------------------------------------------------------------------
    #     # 3. Final concatenation
    #     # ---------------------------------------------------------------------
    #     final_imgs = torch.cat(all_imgs, dim=0)
    #     final_tabs = torch.cat(all_tabs, dim=0)
    #
    #     # Trim if there’s any overshoot (normally won't happen, but just in case)
    #     final_imgs = final_imgs[:n_samples]
    #     final_tabs = final_tabs[:n_samples]
    #
    #     # ---------------------------------------------------------------------
    #     # 4. Build the output DataBucket with the generated data
    #     # ---------------------------------------------------------------------
    #     # Figure out which label to assign to the output.
    #     if scenario == 'uncond':
    #         out_label = 'both'  # by default we have (img + tab)
    #     elif scenario == 'cond_image':
    #         out_label = 'both'
    #     elif scenario == 'cond_table':
    #         out_label = 'both'
    #     else:  # scenario == 'cond_both'
    #         out_label = 'both'
    #
    #     # If, in your usage, you only generate images or only generate tables,
    #     # you can refine 'out_label' to 'image' or 'tab' respectively. But from the
    #     # original code, even uncond always produces both. Adjust as needed.
    #     #
    #     # For example:
    #     # if scenario == 'cond_image':
    #     #    out_label = 'image_tab' or 'both' or maybe 'image' if you never want the table.
    #     # etc.
    #
    #     # We'll store the final images/tables as a list of Tensors (or a single Tensor),
    #     # since the user might want to keep them in memory. Another approach is to store
    #     # them directly in the DataBucket's data_source as a list, e.g. [ (img_i, tab_i), ... ].
    #
    #     generated_data_list = []
    #     for i in range(n_samples):
    #         generated_data_list.append((final_imgs[i], final_tabs[i]))
    #     generated_data_bucket = DataBucket(
    #         data_source=generated_data_list,
    #         label=out_label
    #     )
    #
    #     # If unconditional, we have no cond_mapping
    #     if scenario == 'uncond':
    #         return generated_data_bucket, None
    #     else:
    #         return generated_data_bucket, cond_mapping


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
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from omegaconf import DictConfig
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    import torch
    from utils.configurations import apply_overrides
    from models.dit.dit_multimodal_mod3 import load_dit
    from utils.configurations import set_project_root
    from data.loader import load_training_data
    from utils.model import load_checkpoint
    from models.utils.model_loader import load_model
    from enums.models.model_types import ModelType
    from models.vae.vae import encode_images, decode_latents

    # Set the project root
    set_project_root()

    # Load model configurations using Hydra
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        dit_cfg = compose(config_name="base_dit_training")
        dit_model = load_dit(dit_cfg)
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        diffusion_cfg = compose(config_name="diffusion")
        diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        vae_cfg = compose(config_name="vae")

    vae = load_model(model_type=ModelType.VAE)
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cuda")
    # how to encode images: latents = encode_images(vae, images, vae_cfg.vae.scaling_factor)
    # how to decode latents: images = decode_latents(vae, latents, vae_cfg.vae.scaling_factor)

    load_checkpoint(diffusion_model, "/mnt/storage/nacc_sub/dit/2025-03-13_13-57-21_redone/checkpoints/checkpoint_step_45000_final.pt", "cuda")
    diffusion_model.eval()

    train_dataloader = load_training_data(dit_cfg)


    # # Create dummy data
    # batch_size = 4
    #
    # images = torch.randn(4, 4, 32, 32).cuda()
    # table_data = torch.randn(4, 174).cuda()  # (B=4, some tab dim=174)
    #
    # # Prepare batch as expected by the model
    # batch = {
    #     'image': images,
    #     'tabular': table_data,
    #     'drop_image_mask': torch.zeros(batch_size, 1, 1, 1).cuda(),
    #     'drop_table_mask': torch.zeros(batch_size, 1).cuda()
    # }
    #
    # # --------------------------
    # # 1) Test forward pass (training-like)
    # # --------------------------
    # outputs = diffusion_model(batch)
    # loss, image_latents, table_latents = outputs
    # loss = loss["loss"]
    #
    # print(f"[TRAINING FORWARD] Loss: {loss.item():.4f}")
    # print(f"[TRAINING FORWARD] Image latents shape: {image_latents.shape}")
    # print(f"[TRAINING FORWARD] Table latents shape: {table_latents.shape}")
    #
    # # --------------------------
    # # 2) Test the sampling methods
    # # --------------------------
    # # We'll do a small sampling check in a couple of scenarios.
    #
    # # (a) UNCONDITIONAL SCENARIO
    # with torch.no_grad():
    #     # Hardcode shapes if your generate method allows them. Otherwise, adapt to your code.
    #     # Typically shape_img = (B, 4, 32, 32), shape_tab=(B, 174).
    #     # We'll do a smaller batch here, e.g. batch size = 2
    #     uncond_img, uncond_tab = diffusion_model.generate(
    #         scenario='uncond',
    #         guidance_scale=1.0,
    #         num_inference_steps=10,  # fewer steps for test
    #         seed=1234,
    #         device='cuda'
    #     )
    #
    #     print(f"[SAMPLING: uncond]  Image shape: {uncond_img.shape}")
    #     print(f"[SAMPLING: uncond]  Table shape: {uncond_tab.shape}")
    #
    # # (b) IMAGE-CONDITIONED SCENARIO
    # with torch.no_grad():
    #     # Suppose we want to condition on an image we already have:
    #     # We'll take 2 images from 'images' as 'image_init'
    #     image_init = images[:4].clone()  # shape (4,4,32,32)
    #     cond_img, cond_tab = diffusion_model.generate(
    #         scenario='cond_image',
    #         image_init=images,
    #         guidance_scale=2.0,
    #         num_inference_steps=10,
    #         seed=5678,
    #         device='cuda'
    #     )
    #
    #     print(f"[SAMPLING: cond_image]  Image shape: {cond_img.shape}")
    #     print(f"[SAMPLING: cond_image]  Table shape: {cond_tab.shape}")
    #
    # print("Sampling tests completed successfully.")


    # ----- EXAMPLE 1: UNCONDITIONAL GENERATION -----
    # Generate 16 samples unconditionally, in batches of 4
    generated_data_bucket, mapping = diffusion_model.generate_samples(
        n_samples=16,
        scenario=ScenarioType.UNCOND,  # Using the Enum
        batch_size=4,
        vae=vae,
        device=torch.device("cuda")
    )

    # mapping == None for uncond scenario
    print("Generated Data Label:", generated_data_bucket.label)  # likely 'both'
    print("Total samples generated:", len(generated_data_bucket.data_source))

    # Because your model might generate latent vectors, you can decode them if needed:
    # E.g., if each generated_data_bucket.data_source[i] == (img_latent, tab_latent),
    # you can decode the 'img_latent' into pixel space:
    for i, (img_latent, tab_latent) in enumerate(generated_data_bucket.data_source):
        # Suppose you want to decode the image with your VAE:
        decoded_img = decode_latents(vae, img_latent.unsqueeze(0), vae_cfg.vae.scaling_factor)
        print(f"Decoded sample {i}, shape={decoded_img.shape}, tab.shape={tab_latent.shape}")
        # do something with the result (e.g. visualize, save to disk, etc.)

    # # ----- EXAMPLE 2A: IMAGE-CONDITIONAL USING A DATALOADER -----
    # # Suppose we create a small DataBucket from our training dataloader directly:
    # image_cond_bucket = DataBucket(
    #     data_source=train_dataloader,  # the entire dataloader
    #     label='image'
    # )
    #
    # # Now we generate 20 samples total
    # generated_data_bucket, cond_map = diffusion_model.generate_samples(
    #     n_samples=20,
    #     scenario='cond_image',  # crucial
    #     batch_size=4,
    #     data_bucket=image_cond_bucket,
    #     guidance_scale=1.5,  # example CFG scale
    #     num_inference_steps=30,
    #     seed=42,
    #     device='cuda'
    # )
    #
    # print("Generated data label:", generated_data_bucket.label)  # likely 'both'
    # print("Mapping keys:", list(cond_map.keys())[:5], "...")  # shows some condition keys
    #
    # # ----- EXAMPLE 2B: IMAGE-CONDITIONAL USING A LIST OF TENSORS/PATHS -----
    # # Let's pretend we extracted 10 images from somewhere:
    # list_of_image_tensors = []
    # for i, batch in enumerate(train_dataloader):
    #     imgs = batch['image']
    #     for img in imgs:
    #         list_of_image_tensors.append(img.cpu())  # store on CPU just for example
    #     if len(list_of_image_tensors) >= 10:
    #         break
    #
    # # Build a DataBucket
    # image_cond_bucket = DataBucket(
    #     data_source=list_of_image_tensors,  # a list of Tensors
    #     label='image'
    # )
    #
    # generated_data_bucket, cond_map = diffusion_model.generate_samples(
    #     n_samples=12,
    #     scenario='cond_image',
    #     data_bucket=image_cond_bucket,
    #     batch_size=4,
    #     guidance_scale=2.0,
    #     num_inference_steps=30,
    #     device='cuda',
    #     seed=999
    # )
    #
    # print("cond_map example:", cond_map)
    # # cond_map keys will be the integer indices in the list (0..9) if you used Tensors,
    # # or the actual path strings if you used file paths.
    #
    # # ----- EXAMPLE 3: TABLE-CONDITIONAL GENERATION -----
    # # Suppose we gather tab data from the training dataloader into a list
    # list_of_tab_tensors = []
    # for i, batch in enumerate(train_dataloader):
    #     tabs = batch['tabular']  # shape (B, 174)
    #     for t in tabs:
    #         list_of_tab_tensors.append(t.cpu())
    #     if len(list_of_tab_tensors) >= 20:
    #         break
    #
    # tab_cond_bucket = DataBucket(
    #     data_source=list_of_tab_tensors,
    #     label='tab'
    # )
    #
    # generated_data_bucket, cond_map = diffusion_model.generate_samples(
    #     n_samples=15,
    #     scenario='cond_table',
    #     data_bucket=tab_cond_bucket,
    #     batch_size=5,
    #     guidance_scale=1.2,
    #     num_inference_steps=25,
    #     device='cuda',
    # )
    #
    # print("Generated data label:", generated_data_bucket.label)  # 'both'
    # print("First condition key => indices:", next(iter(cond_map.items())))
    #
    # # ----- EXAMPLE 4: BOTH-CONDITIONAL GENERATION -----
    #
    # list_of_pairs = []
    # for i, batch in enumerate(train_dataloader):
    #     imgs = batch['image']  # shape (B, C, H, W)
    #     tabs = batch['tabular']  # shape (B, 174)
    #     for img, tab in zip(imgs, tabs):
    #         list_of_pairs.append((img.cpu(), tab.cpu()))
    #     if len(list_of_pairs) >= 10:
    #         break
    #
    # both_cond_bucket = DataBucket(
    #     data_source=list_of_pairs,
    #     label='both'
    # )
    #
    # generated_data_bucket, cond_map = diffusion_model.generate_samples(
    #     n_samples=12,
    #     scenario='cond_both',
    #     data_bucket=both_cond_bucket,
    #     batch_size=4,
    #     guidance_scale=2.0,
    #     num_inference_steps=30,
    #     device='cuda',
    #     seed=2024
    # )
    #
    # print("cond_map:", cond_map)
    # # Each key is either an index or a path (depending on your data_source)
    # # Each value is a list of sample indices generated from that condition.


