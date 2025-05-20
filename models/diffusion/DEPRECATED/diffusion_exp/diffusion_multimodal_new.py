import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from composer.models import ComposerModel
from typing import Optional, Dict, Tuple, Any
import numpy as np
from omegaconf import DictConfig

###################################################
# Example data-type mapping; remove or adapt as needed
###################################################
DATA_TYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}


class MultiModalDiffusion(ComposerModel):
    """
    Multi-modal Latent Diffusion Model for images and tabular data, *without* VAE references.

    This class:
      - Expects *already-prepared latents* for both image and table in the forward pass.
      - Can apply classifier-free dropout if 'drop_image_mask' or 'drop_table_mask' exist in batch.
      - Implements an EDM (Elucidated Diffusion) loss with optional patch masking.
      - The `generate` method can produce latents from noise or use provided latents,
        then returns the final latents (no decoding to images/tables here).

    Args:
        dit (nn.Module):
            Multi-modal diffusion transformer that predicts noise for both image and table latents.
            Must accept `forward(img_latents, tab_latents, noise_levels, mask_ratio=...)`
            and return `{'sample_img': ..., 'sample_tab': ..., 'mask_img': ..., 'mask_tab': ...}`.
        image_in_channels (int):
            Number of latent channels for the image latents.
        latent_res (int):
            Resolution of the image latents (H=W=latent_res).
        table_latent_dim (int):
            The "width" or "length" dimension for the table latents (e.g., [N, C, 1, W]).
        dtype (str):
            One of 'float16', 'float32', or 'bfloat16' for internal computations.
        p_mean (float):
            EDM log-normal noise mean.
        p_std (float):
            EDM log-normal noise std.
        train_mask_ratio (float):
            Ratio for patch masking while training (0 for none).
    """

    def __init__(
            self,
            dit: nn.Module,
            *,
            image_in_channels: int,
            latent_res: int = 32,
            table_latent_dim: int = 16,
            dtype: str = 'bfloat16',
            p_mean: float = -0.6,
            p_std: float = 1.2,
            train_mask_ratio_img: float = 0.0,
            train_mask_ratio_tab: float = 0.0
    ):
        super().__init__()
        self.dit = dit
        self.image_in_channels = image_in_channels
        self.latent_res = latent_res
        self.table_latent_dim = table_latent_dim
        self.dtype = dtype

        # EDM hyperparameters
        self.edm_config = {
            'sigma_min': 0.002,
            'sigma_max': 80.0,
            'P_mean': p_mean,
            'P_std': p_std,
            'sigma_data': 0.9,
            'num_steps': 18,
            'rho': 7,
            'S_churn': 0.0,
            'S_min': 0.0,
            'S_max': float('inf'),
            'S_noise': 1.0
        }
        self.table_width = 174

        self.train_mask_ratio_img, self.train_mask_ratio_tab = train_mask_ratio_img, train_mask_ratio_tab
        self.eval_mask_ratio_img, self.eval_mask_ratio_tab = 0.0, 0.0  # no patch masking in eval/generation

        self.randn_like = torch.randn_like

        # If using FSDP or distributed, adapt ._fsdp_wrap as needed
        self.dit._fsdp_wrap = True

    ###########################################
    # Forward Pass (Training)
    ###########################################
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Expects:
            batch['image_latents']: (N, C, H, W) latents for images
            batch['table_latents']: (B, table_width)

        Optionally:
            batch['drop_image_mask']: shape (B, 1, 1, 1), 1 => drop entire image latents
          batch['drop_table_mask']: shape (B, 1), 1 => drop entire row latents

        Returns:
            (loss, image_latents, table_latents)
        """
        # 1) Retrieve latents from the batch
        latents_img = batch['image_latents']
        latents_tab = batch['table_latents']

        # 2) Possibly do CF dropout
        if 'drop_image_mask' in batch:
            mask_img = batch['drop_image_mask'].view(-1, 1, 1, 1).to(latents_img.device)
            latents_img = latents_img * (1.0 - mask_img)

        if 'drop_table_mask' in batch:
            mask_tab = batch['drop_table_mask'].view(-1, 1).to(latents_tab.device)
            latents_tab = latents_tab * (1.0 - mask_tab)

        # 3) Compute EDM loss
        mask_ratio_img = self.train_mask_ratio_img if self.training else self.eval_mask_ratio_img
        mask_ratio_tab = self.train_mask_ratio_tab if self.training else self.eval_mask_ratio_tab
        loss = self.edm_loss(
            latents_img.float(),
            latents_tab.float(),
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab
        )
        return (loss, latents_img, latents_tab)

    ###########################################
    # EDM Loss
    ###########################################
    def edm_loss(self, x_img: torch.Tensor, x_tab: torch.Tensor, mask_ratio_img: float = 0.0, mask_ratio_tab: float = 0.0) -> torch.Tensor:
        """
        EDM loss for both image + table latents.
        x_img: (B, in_channels, H, W)
        x_tab: (B, table_width)
        """

        device = x_img.device
        B = x_img.shape[0]
        sigma_data = self.edm_config['sigma_data']

        # 1) Sample random log-normal sigma
        rnd_normal = torch.randn([x_img.shape[0], 1, 1, 1], device=device)
        sigma = (rnd_normal * self.edm_config['P_std'] + self.edm_config['P_mean']).exp()

        # 2) Weight factor
        weight_scalar  = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

        # But we want TWO shapes:
        weight_img = weight_scalar  # => (B,1,1,1), matches (B,C,H,W)
        weight_tab = weight_scalar.view(B, 1)  # => (B,1), matches (B,table_width)

        # 3) Add noise
        n_img = self.randn_like(x_img) * sigma  # shape => (B,C,H,W)
        n_tab = torch.randn_like(x_tab) * sigma.view(B, 1)  # shape => (B,table_width)

        # 4) Model forward => predicted noise
        model_out = self.model_forward_wrapper(
            x_img + n_img,
            x_tab + n_tab,
            sigma,
            self.dit,
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab
        )

        D_xn_img = model_out['sample_img']
        D_xn_tab = model_out['sample_tab']

        # 5) MSE
        loss_img = weight_img * (D_xn_img - x_img) ** 2  # shape => (B,C,H,W)
        loss_img = loss_img.mean(dim=[1, 2, 3])  # => (B,)

        loss_tab = weight_tab * (D_xn_tab - x_tab) ** 2  # shape => (B,table_width)
        loss_tab = loss_tab.mean(dim=1)  # => (B,)

        loss_per_sample = 0.5 * (loss_img + loss_tab)  # => (B,)

        # 5) Optional patch masking
        if mask_ratio_img > 0.0 or mask_ratio_tab > 0.0:
            # We expect patch masks in model_out if training with patch masking
            # E.g. model_out['mask_img'], model_out['mask_tab']
            assert self.dit.training, "Patch masking typically used in training only."

            # Recompute for image in an un-reduced form
            if mask_ratio_img > 0.0 and 'mask_img' in model_out:
                mses_img = weight * (D_xn_img - x_img) ** 2  # [N, C, H, W]
                mses_img = mses_img.mean(dim=1, keepdim=True)  # [N, 1, H, W]
                patch_size_img = getattr(self.dit, 'patch_size_img', 2)
                pooled_img = F.avg_pool2d(mses_img, kernel_size=patch_size_img)  # [N, 1, H/ps, W/ps]
                pooled_img = pooled_img.flatten(1)  # [N, #patches]
                unmask_img = 1.0 - model_out['mask_img']  # [N, #patches]
                loss_masked_img = (pooled_img * unmask_img).sum(dim=1) / (unmask_img.sum(dim=1) + 1e-8)
                loss_img = loss_masked_img

            if mask_ratio_tab > 0.0 and 'mask_tab' in model_out:
                mses_tab = weight * (D_xn_tab - x_tab) ** 2
                mses_tab = mses_tab.unsqueeze(1)  # (B,1,table_width)
                patch_size_tab = getattr(self.dit, 'patch_size_tab', 1)
                pooled_tab = F.avg_pool2d(mses_tab, kernel_size=(1, patch_size_tab))
                pooled_tab = pooled_tab.view(pooled_tab.shape[0], -1)
                unmask_tab = 1.0 - model_out['mask_tab']
                loss_masked_tab = (pooled_tab * unmask_tab).sum(dim=1) / (unmask_tab.sum(dim=1) + 1e-8)
                loss_tab = loss_masked_tab

            # Recombine final
            loss_per_sample = 0.5 * (loss_img + loss_tab)

        return loss_per_sample.mean()

    ###########################################
    # Model Forward Wrapper for EDM
    ###########################################
    def model_forward_wrapper(
            self,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            sigma: torch.Tensor,
            model_forward_fxn: callable,
            mask_ratio_img: float = 0.0,
            mask_ratio_tab: float = 0.0
    ) -> dict:
        """
        Scales inputs according to EDM formula, calls self.dit, and returns denoised latents.
        Expects the DiT to return dict with {'sample_img', 'sample_tab', 'mask_img', 'mask_tab'} optionally.
        """
        # shapes:
        #   x_img => (B, in_ch, H, W)
        #   x_tab => (B, table_width)
        #   sigma => shape => broadcast => (B,1,1,1) or (B,1)

        # Ensure sigma is (B,1,1,1) for the image domain
        # Usually sigma is something like (B,1,1,1) from the random log-normal sampling
        B = x_img.shape[0]
        device = x_img.device

        # Convert sigma to shape (B,1,1,1) if not already
        sigma_img = sigma.reshape(B, 1, 1, 1)  # for image
        # For table, we want (B,1)
        sigma_tab = sigma.reshape(B, 1)

        # EDM constants
        sigma_data = 0.9  # or wherever you store it
        # Compute c_skip, c_out, c_in in shape (B,1,1,1) for the image domain
        c_skip_img = sigma_data ** 2 / (sigma_img ** 2 + sigma_data ** 2)
        c_out_img = sigma_img * sigma_data / torch.sqrt(sigma_img ** 2 + sigma_data ** 2)
        c_in_img = 1.0 / torch.sqrt(sigma_data ** 2 + sigma_img ** 2)

        # Reshape them for table domain => (B,1)
        c_skip_tab = c_skip_img.view(B, 1)
        c_out_tab = c_out_img.view(B, 1)
        c_in_tab = c_in_img.view(B, 1)

        # Scale inputs
        x_img_in = x_img * c_in_img  # => shape (B, C, H, W)
        x_tab_in = x_tab * c_in_tab  # => shape (B, table_width)

        # Pass to your DiT forward (or any model)
        # NOTE: We pass the "noise levels" as well if needed
        noise_level_img = sigma_img.log() / 4.0
        noise_level_img = noise_level_img.view(B)  # (B,)
        noise_level_tab = noise_level_img  # identical for table

        out = model_forward_fxn(
            x_img_in,  # scaled image latents (B, C, H, W)
            x_tab_in,  # scaled table latents  (B, table_width)
            noise_level_img,  # (B,)
            noise_level_tab,  # (B,)
            mask_ratio_img=mask_ratio_img,
            mask_ratio_tab=mask_ratio_tab,
        )
        # out['sample_img'] => shape (B, C, H, W)
        # out['sample_tab'] => shape (B, table_width) if your model is consistent

        F_x_img = out['sample_img']
        F_x_tab = out['sample_tab']

        # Final EDM formula => "skip" + "scaled model output"
        #   D_x = c_skip*x + c_out*F_x
        D_x_img = c_skip_img * x_img + c_out_img * F_x_img  # => (B, C, H, W)
        D_x_tab = c_skip_tab * x_tab + c_out_tab * F_x_tab  # => (B, table_width)

        return {
            'sample_img': D_x_img,
            'sample_tab': D_x_tab,
            'mask_img': out.get('mask_img', None),
            'mask_tab': out.get('mask_tab', None),
        }

    ###########################################
    # Composer Hooks
    ###########################################
    def loss(self, outputs: Tuple[torch.Tensor, ...], batch: Dict[str, torch.Tensor])->torch.Tensor:
        return outputs[0]

    def eval_forward(self, batch: Dict[str, torch.Tensor], outputs: Optional[Tuple] = None):
        if outputs is not None:
            return outputs
        return self.forward(batch)

    ###########################################
    # EDM Sampler Loop
    ###########################################
    @torch.no_grad()
    def edm_sampler_loop(
            self,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            steps: Optional[int] = None,
            cfg: float = 1.0,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard EDM sampling for image + table latents.
        If cfg>1.0, we assume the DiT implements CFG internally
        (i.e., does unconditional+conditional blending).
        """
        # No patch masking during generation
        mask_ratio = 0.0

        # If cfg>1.0, the model_forward_fxn can do classifier-free guidance internally
        model_forward_fxn = (
            partial(self.dit.forward, cfg=cfg) if cfg > 1.0
            else self.dit.forward
        )

        # Unpack EDM config
        num_steps = steps or self.edm_config['num_steps']
        rho = self.edm_config['rho']
        sigma_min = self.edm_config['sigma_min']
        sigma_max = self.edm_config['sigma_max']
        S_churn = self.edm_config['S_churn']
        S_min = self.edm_config['S_min']
        S_max = self.edm_config['S_max']
        S_noise = self.edm_config['S_noise']

        B = x_img.shape[0]  # batch size

        # Time steps t_0, t_1, ..., t_{num_steps} plus an extra zero at the end
        # (We store them in t_steps, shape [num_steps+1])
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x_img.device)
        t_steps = (
                          sigma_max ** (1 / rho)
                          + step_indices / (num_steps - 1)
                          * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
                  ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # final t_{num_steps} = 0

        # Multiply initial latents by t_steps[0]
        x_img_next = x_img.to(torch.float64) * t_steps[0]
        x_tab_next = x_tab.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_img_cur = x_img_next
            x_tab_cur = x_tab_next

            # If S_churn > 0, we can add extra noise to t_cur -> t_hat
            gamma = 0.0
            if (S_churn > 0.0) and (S_min <= t_cur <= S_max):
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1)

            t_hat = t_cur + gamma * t_cur

            # Add noise to go from t_cur to t_hat
            noise_img = self.randn_like(x_img_cur) * S_noise
            noise_tab = self.randn_like(x_tab_cur) * S_noise

            x_img_hat = x_img_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * noise_img
            x_tab_hat = x_tab_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * noise_tab

            # -- Convert scalars t_hat, t_next into shape [B] so the model can reshape internally
            t_hat_b = t_hat.view(1).expand(B).to(x_img_hat.device).float()

            # 1) Euler step (first-order)
            out_hat = self.model_forward_wrapper(
                x_img_hat.float(),
                x_tab_hat.float(),
                t_hat_b,
                model_forward_fxn,
                mask_ratio_img=mask_ratio,
                mask_ratio_tab=mask_ratio,
                **kwargs
            )

            # Divide by t_hat for d_cur
            d_cur_img = (x_img_hat - out_hat['sample_img'].to(torch.float64)) / t_hat_b[:, None, None, None]
            d_cur_tab = (x_tab_hat - out_hat['sample_tab'].to(torch.float64)) / t_hat_b[:, None]

            # Update x_img_next, x_tab_next
            x_img_next = x_img_hat + (t_next - t_hat) * d_cur_img
            x_tab_next = x_tab_hat + (t_next - t_hat) * d_cur_tab

            # 2) 2nd-order correction (Heun method)
            if i < num_steps - 1:
                t_next_b = t_next.view(1).expand(B).to(x_img_hat.device).float()

                out_next = self.model_forward_wrapper(
                    x_img_next.float(),
                    x_tab_next.float(),
                    t_next_b,
                    model_forward_fxn,
                    mask_ratio_img=mask_ratio,
                    mask_ratio_tab=mask_ratio,
                    **kwargs
                )

                d_prime_img = (x_img_next - out_next['sample_img'].to(torch.float64)) / t_next_b[:, None, None, None]
                d_prime_tab = (x_tab_next - out_next['sample_tab'].to(torch.float64)) / t_next_b[:, None]

                x_img_next = x_img_hat + (t_next - t_hat) * (0.5 * d_cur_img + 0.5 * d_prime_img)
                x_tab_next = x_tab_hat + (t_next - t_hat) * (0.5 * d_cur_tab + 0.5 * d_prime_tab)

        # Return final latents (float32)
        return x_img_next.float(), x_tab_next.float()

    ###########################################
    # Generate Method
    ###########################################
    @torch.no_grad()
    def generate(
            self,
            batch_size: int = 1,
            image_latents: Optional[torch.Tensor] = None,
            table_latents: Optional[torch.Tensor] = None,
            guidance_scale: float = 1.0,
            num_inference_steps: int = 30,
            seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate final latents for images & tables, returning them for external decode.

        If image_latents/table_latents are provided => direct conditioning.
        If None => random noise (unconditional).
        If guidance_scale>1 => internal CFG if the model supports it.

        Returns: (final_img_latents, final_tab_latents)
        """

        device = (image_latents.device if image_latents is not None
                  else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        dtype = DATA_TYPES[self.dtype]

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        # 1) Prepare image latents
        if image_latents is None:
            # Unconditional for images
            x_img = torch.randn(
                (batch_size, self.image_in_channels, self.latent_res, self.latent_res),
                generator=rng,
                device=device
            ).to(dtype)
        else:
            x_img = image_latents.to(device, dtype=dtype)

        # 2) Prepare table latents
        if table_latents is None:
            # Unconditional for tables
            x_tab = torch.randn(
                (batch_size, self.table_width),
                generator=rng,
                device=device
            ).to(dtype)
        else:
            x_tab = table_latents.to(device, dtype=dtype)

        # 3) Run EDM sampling
        x_img_final, x_tab_final = self.edm_sampler_loop(
            x_img, x_tab,
            steps=num_inference_steps,
            cfg=guidance_scale
        )

        # 4) Return final latents only (no decoding)
        return x_img_final, x_tab_final

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
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    from typing import Optional, Tuple, Any
    from omegaconf import DictConfig
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    import torch
    from utils.configurations import apply_overrides
    from models.dit.dit_multimodal import load_dit
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
        'image_latents': images,
        'table_latents': table_data,
        'drop_image_mask': torch.zeros(batch_size, 1, 1, 1).cuda(),
        'drop_table_mask': torch.zeros(batch_size, 1).cuda()
    }

    # Forward pass
    outputs = diffusion_model(batch)
    loss, image_latents, table_latents = outputs
    print(f"Loss: {loss.item():.4f}")
    print(f"Image latents shape: {image_latents.shape}")
    print(f"Table latents shape: {table_latents.shape}")

    # Sampling demonstration
    with torch.no_grad():
        sampled_img_latents, sampled_tab_latents = diffusion_model.generate(batch_size=batch_size)
        print("Sampled image latents shape:", sampled_img_latents.shape)
        print("Sampled table latents shape:", sampled_tab_latents.shape)
