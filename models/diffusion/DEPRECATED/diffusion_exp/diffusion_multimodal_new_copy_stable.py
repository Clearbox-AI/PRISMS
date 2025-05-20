from functools import partial

from composer.models import ComposerModel
from easydict import EasyDict
import torch.nn as nn
import numpy as np

from typing import Any, Tuple
from omegaconf import DictConfig
import torch
from typing import List, Dict, Optional
import torch.nn.functional as F

from torch import Tensor

from utils.configurations import apply_overrides

DATA_TYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

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
            'sigma_max': 40,
            'P_mean': p_mean,
            'P_std': p_std,
            'sigma_data': 0.9,
            'num_steps': 32,
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
        scenario: str,
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
        rnd_normal = torch.randn([N, 1, 1, 1], device=x_img.device)
        sigma = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()

        # Weight factor from EDM
        weight = (sigma ** 2 + self.edm_config.sigma_data ** 2) / ((sigma * self.edm_config.sigma_data) ** 2)

        # 2) Apply scenario noising
        x_noisy_img, x_noisy_tab = self.apply_scenario_noise(x_img, x_tab, scenario, sigma)

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
        scenario: str,
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

    def apply_scenario_noise(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        scenario: str,
        sigma: torch.Tensor
    ):
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
    def edm_sampler_loop(
            self,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            scenario: str = 'uncond',
            steps: Optional[int] = None,
            cfg: float = 1.0,
            **kwargs
    ) -> (torch.Tensor, torch.Tensor):
        """
        Multi-modal sampling loop, analogous to the single-modal `edm_sampler_loop`.

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

    @torch.no_grad()
    def generate(
            self,
            scenario: str = 'uncond',
            image_init: Optional[torch.Tensor] = None,
            table_init: Optional[torch.Tensor] = None,
            guidance_scale: float = 1.0,
            num_inference_steps: int = 30,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        High-level generation method that sets up latents for each scenario,
        then calls edm_sampler_loop. Similar to 'generate' in your single‐modal code,
        but now we have multi‐modal data.

        Args:
            scenario (str): 'uncond', 'cond_image', 'cond_table', 'cond_both'.
            image_init (torch.Tensor, optional): If scenario involves an image condition,
                                                 pass your initial image latents here.
            table_init (torch.Tensor, optional): If scenario involves a table condition,
                                                 pass your initial table data here.
            guidance_scale (float): CFG scale.
            num_inference_steps (int): # of EDM steps in sampling.
            seed (int, optional): RNG seed for reproducible noise.
            device (torch.device, optional): Torch device. If None, we infer from self.dit.

        Returns:
            (final_img, final_tab): The final latents for each modality after sampling.
        """

        # Decide device
        if device is None:
            device = next(self.dit.parameters()).device

        # Prepare a torch Generator if needed
        rng_generator = torch.Generator(device=device)
        if seed is not None:
            rng_generator.manual_seed(seed)

        default_batch = 4
        shape_img = (default_batch, 4, 32, 32)
        shape_tab = (default_batch, 174)

        # 1) Initialize latents according to scenario
        if scenario == 'uncond':
            x_img = torch.randn(shape_img, device=device, generator=rng_generator)
            x_tab = torch.randn(shape_tab, device=device, generator=rng_generator)

        elif scenario == 'cond_image':
            if image_init is None:
                raise ValueError("Must provide image_init for cond_image scenario.")
            shape_img = image_init.shape
            batch_size = shape_img[0]
            x_img = image_init.to(device)
            # random for table => same batch size
            shape_tab = (batch_size, 174)
            x_tab = torch.randn(shape_tab, device=device, generator=rng_generator)

        elif scenario == 'cond_table':
            if table_init is None:
                raise ValueError("Must provide table_init for cond_table scenario.")
            shape_tab = table_init.shape  # (B,174)
            batch_size = shape_tab[0]
            x_tab = table_init.to(device)
            # random for image
            shape_img = (batch_size, 4, 32, 32)
            x_img = torch.randn(shape_img, device=device, generator=rng_generator)

        else:  # 'cond_both'
            if image_init is None or table_init is None:
                raise ValueError("Must provide image_init and table_init for scenario='cond_both'.")
            shape_img = image_init.shape  # (N,4,32,32)
            shape_tab = table_init.shape  # (N,174)
            # Optionally check they have the same batch size:
            if shape_img[0] != shape_tab[0]:
                raise ValueError(f"In cond_both, mismatch in batch size: {shape_img[0]} vs {shape_tab[0]}")

            # Partially noise both
            scale = 0.5
            x_img = image_init.to(device) + scale * torch.randn(shape_img, device=device, generator=rng_generator)
            x_tab = table_init.to(device) + scale * torch.randn(shape_tab, device=device, generator=rng_generator)

        # 2) Call the sampler loop
        final_img, final_tab = self.edm_sampler_loop(
            x_img,
            x_tab,
            scenario=scenario,
            steps=num_inference_steps,
            cfg=guidance_scale,
            **kwargs
        )

        return final_img, final_tab

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

    # Set the project root
    set_project_root()

    # Load model configurations using Hydra
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        dit_cfg = compose(config_name="base_dit_training")
        dit_model = load_dit(dit_cfg)
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        diffusion_cfg = compose(config_name="diffusion")
        diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

    # Create dummy data
    batch_size = 4

    images = torch.randn(4, 4, 32, 32).cuda()
    table_data = torch.randn(4, 174).cuda()  # (B=4, some tab dim=174)

    # Prepare batch as expected by the model
    batch = {
        'image': images,
        'tabular': table_data,
        'drop_image_mask': torch.zeros(batch_size, 1, 1, 1).cuda(),
        'drop_table_mask': torch.zeros(batch_size, 1).cuda()
    }

    # --------------------------
    # 1) Test forward pass (training-like)
    # --------------------------
    outputs = diffusion_model(batch)
    loss, image_latents, table_latents = outputs
    loss = loss["loss"]

    print(f"[TRAINING FORWARD] Loss: {loss.item():.4f}")
    print(f"[TRAINING FORWARD] Image latents shape: {image_latents.shape}")
    print(f"[TRAINING FORWARD] Table latents shape: {table_latents.shape}")

    # --------------------------
    # 2) Test the sampling methods
    # --------------------------
    # We'll do a small sampling check in a couple of scenarios.

    # (a) UNCONDITIONAL SCENARIO
    with torch.no_grad():
        # Hardcode shapes if your generate method allows them. Otherwise, adapt to your code.
        # Typically shape_img = (B, 4, 32, 32), shape_tab=(B, 174).
        # We'll do a smaller batch here, e.g. batch size = 2
        uncond_img, uncond_tab = diffusion_model.generate(
            scenario='uncond',
            guidance_scale=1.0,
            num_inference_steps=10,  # fewer steps for test
            seed=1234,
            device='cuda'
        )

        print(f"[SAMPLING: uncond]  Image shape: {uncond_img.shape}")
        print(f"[SAMPLING: uncond]  Table shape: {uncond_tab.shape}")

    # (b) IMAGE-CONDITIONED SCENARIO
    with torch.no_grad():
        # Suppose we want to condition on an image we already have:
        # We'll take 2 images from 'images' as 'image_init'
        image_init = images[:4].clone()  # shape (4,4,32,32)
        cond_img, cond_tab = diffusion_model.generate(
            scenario='cond_image',
            image_init=images,
            guidance_scale=2.0,
            num_inference_steps=10,
            seed=5678,
            device='cuda'
        )

        print(f"[SAMPLING: cond_image]  Image shape: {cond_img.shape}")
        print(f"[SAMPLING: cond_image]  Table shape: {cond_tab.shape}")

    print("Sampling tests completed successfully.")