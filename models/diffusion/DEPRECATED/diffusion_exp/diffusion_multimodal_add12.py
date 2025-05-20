import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from omegaconf import DictConfig
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt
from einops import reduce
from diffusers import DPMSolverMultistepScheduler


###################################
# Exponential Moving Average (EMA)
###################################
class EMA:
    """
    Maintains an exponential moving average of a model's parameters.
    When you want to sample, you typically use the EMA copy for better stability.
    """
    def __init__(self, model: nn.Module, decay=0.9999):
        """
        Args:
            model (nn.Module): The original model to track.
            decay (float): EMA decay factor.
        """
        self.model = model
        self.decay = decay
        self.model_ema = self._make_ema_model()

    def _make_ema_model(self) -> nn.Module:
        """
        Create a copy of the model with identical architecture,
        then copy over all parameters from self.model.
        """
        from models.dit.dit_multimodal_add12 import MultiModalDiT
        import os
        from pathlib import Path
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from models.dit.dit_multimodal_add12 import load_dit
        from models.diffusion.diffusion_multimodal_add12 import load_diffusion

        with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
            cfg = compose(config_name="base_dit_training")  # or the actual config
            OmegaConf.set_struct(cfg, False)

        # Re-create the same diffusion + DiT architecture
        dit_model = load_dit(cfg.dit)
        ema_model = load_diffusion(cfg, dit_model)

        device = next(self.model.parameters()).device
        ema_model.to(device)

        # Now load the current state from self.model into the new model:
        ema_model.load_state_dict(self.model.state_dict())

        # Then ensure the EMA model is in eval mode and does not require grads:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False

        return ema_model

    def update(self):
        """
        Call this after every train step to update EMA weights.
        """
        with torch.no_grad():
            msd = self.model.state_dict()
            esd = self.model_ema.state_dict()
            for k in msd.keys():
                if msd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)

    def ema_model_inference(self) -> nn.Module:
        """
        Returns the EMA model to use for sampling.
        """
        return self.model_ema


import torch
import torch.nn as nn
import torch.nn.functional as F

# from diffusers import DPMSolverMultistepScheduler

class MultiModalDiffusion(nn.Module):
    """
    A diffusion model that handles both an image latent (B,4,32,32) and tabular data (B,num_columns).
    Uses v-prediction for training, DPM++ 2M Karras for sampling.
    """

    def __init__(self, dit: nn.Module, num_train_timesteps: int = 1000):
        """
        Args:
            dit (nn.Module): Your multimodal DiT-style model returning {"image_sample", "tab_sample"}.
            num_train_timesteps (int): The number of timesteps for the schedule.
        """
        super().__init__()
        self.dit = dit

        # Create a DPMSolverMultistepScheduler in "v_prediction" mode, second-order, Karras sigma.
        self.train_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2"
        )

        self.sample_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2"
        )



        #
        # We'll store partial model outputs for the 2nd (or 3rd) order method.
        # Because we want a single "step" that updates *both* x_img and x_tab
        # at once, we track model_outputs for each modality.
        #
        # These are used in "multimodal_step" for the multi-step formula.
        #
        # self.model_outputs_img = [None] * self.scheduler.config.solver_order
        # self.model_outputs_tab = [None] * self.scheduler.config.solver_order
        # self.lower_order_nums = 0  # how many low-order steps we've done
        # self.step_index = None

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor):
        """
        Training loop for v-parameterization:
          1) Sample random timesteps t
          2) Add noise x_t = alpha_t*x_0 + sigma_t*eps
          3) Model predicts v_img, v_tab
          4) Compute v-target and MSE loss
        """
        device = x_img.device
        batch_size = x_img.size(0)

        # (1) sample timesteps in [0, T-1]
        t = torch.randint(
            low=0,
            high=self.train_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=device
        )

        # (2) Add noise according to the training scheduler's sigmas
        #     We'll gather sigmas from train_scheduler.
        with torch.no_grad():
            sigmas_all = self.train_scheduler.sigmas.to(device=device)
        sig_t = sigmas_all[t]  # shape (B,)

        # Convert sigma -> (alpha_t, sigma_t)
        alpha_t, sigma_t_ = self.train_scheduler._sigma_to_alpha_sigma_t(sig_t)
        noise_img = torch.randn_like(x_img)
        noise_tab = torch.randn_like(x_tab)

        # expand to match shapes
        alpha_img = alpha_t.view(-1, 1, 1, 1)
        sigma_img = sigma_t_.view(-1, 1, 1, 1)
        x_t_img = alpha_img * x_img + sigma_img * noise_img

        alpha_tab = alpha_t.unsqueeze(-1)
        sigma_tab = sigma_t_.unsqueeze(-1)
        x_t_tab = alpha_tab * x_tab + sigma_tab * noise_tab

        # (3) Model prediction => v_img, v_tab
        out = self.dit(x_img=x_t_img, x_tab=x_t_tab, t=t)
        v_img = out["image_sample"]
        v_tab = out["tab_sample"]

        # (4) v-target
        #   v_target = (alpha_t*x_t - x_0) / sigma_t
        # We do that separately for image & tab
        v_img_target = (alpha_img * x_t_img - x_img) / sigma_img
        v_tab_target = (alpha_tab * x_t_tab - x_tab) / sigma_tab

        loss_img = F.mse_loss(v_img, v_img_target)
        loss_tab = F.mse_loss(v_tab, v_tab_target)
        loss_total = loss_img + loss_tab

        return loss_total, loss_img, loss_tab

    @torch.no_grad()
    def sample(
            self,
            model_ema: nn.Module,
            batch_size: int,
            x_img_shape=(4, 32, 32),
            x_tab_shape=(8,),
            num_steps: int = 20,
            device: str = "cuda",
    ):
        """
        DPM++ 2M Karras sampling using a custom `multimodal_step` that updates
        both x_img and x_tab in one pass of the solver.
        """
        shape_img = (batch_size,) + x_img_shape
        shape_tab = (batch_size,) + x_tab_shape

        # We'll use the *sample* scheduler now (which is separate from the training one).
        sample_scheduler = self.sample_scheduler

        # 1) Initialize x_img, x_tab as pure Gaussian noise
        x_img = torch.randn(shape_img, device=device) * sample_scheduler.init_noise_sigma
        x_tab = torch.randn(shape_tab, device=device) * sample_scheduler.init_noise_sigma

        # 2) Set the timesteps for sampling (fewer steps, e.g. 20)
        sample_scheduler.set_timesteps(num_steps, device=device)
        sample_scheduler.set_begin_index(0)

        # 3) We keep local lists for the model outputs across steps for the DPMSolver logic.
        model_outputs_img = [None] * sample_scheduler.config.solver_order
        model_outputs_tab = [None] * sample_scheduler.config.solver_order
        lower_order_nums = 0

        # 4) Iteratively denoise
        for i, t in enumerate(sample_scheduler.timesteps):
            sample_scheduler._step_index = i

            # Model forward (EMA model)
            out = model_ema.dit(
                x_img=x_img,
                x_tab=x_tab,
                t=t.expand(x_img.shape[0])  # shape (B,)
            )
            v_img = out["image_sample"]
            v_tab = out["tab_sample"]

            # Single step update for both x_img and x_tab
            x_img, x_tab, model_outputs_img, model_outputs_tab, lower_order_nums = \
                self.multimodal_step(
                    v_img, v_tab,
                    x_img, x_tab,
                    sample_scheduler,
                    model_outputs_img,
                    model_outputs_tab,
                    lower_order_nums
                )

        return x_img, x_tab

    @torch.no_grad()
    def multimodal_step(
            self,
            model_output_img: torch.Tensor,
            model_output_tab: torch.Tensor,
            x_img: torch.Tensor,
            x_tab: torch.Tensor,
            scheduler: DPMSolverMultistepScheduler,
            model_outputs_img: list,
            model_outputs_tab: list,
            lower_order_nums: int,
            variance_noise: torch.Tensor = None,
    ):
        """
        A custom method that reproduces what `scheduler.step(...)` does — but for
        multiple modalities at once. It uses DPM++ 2M logic, applying
        the same step formula to x_img and x_tab in one go.
        """
        # 1) Convert v-pred => "eps" or x0_pred as needed by DPM++ (the scheduler does that).
        v_img = scheduler.convert_model_output(model_output_img, sample=x_img)
        v_tab = scheduler.convert_model_output(model_output_tab, sample=x_tab)

        # 2) Shift old model outputs forward
        for j in range(scheduler.config.solver_order - 1):
            model_outputs_img[j] = model_outputs_img[j + 1]
            model_outputs_tab[j] = model_outputs_tab[j + 1]
        model_outputs_img[-1] = v_img
        model_outputs_tab[-1] = v_tab

        noise = None  # For DPM++ (non-SDE), we don't usually add noise in the step.

        # 3) Decide if we do 1st or 2nd (or 3rd) order update
        step_index = scheduler._step_index
        total_steps = len(scheduler.timesteps)

        # stable-diffusion-ish logic for "lower_order_final" or euler_at_final
        lower_order_final = (
                (step_index == total_steps - 1)  # last step
                and (
                        scheduler.config.euler_at_final
                        or (scheduler.config.lower_order_final and total_steps < 15)
                        or scheduler.config.final_sigmas_type == "zero"
                )
        )
        second_to_last = (
                (step_index == total_steps - 2)
                and scheduler.config.lower_order_final
                and (total_steps < 15)
        )

        if (
                scheduler.config.solver_order == 1
                or lower_order_nums < 1
                or lower_order_final
        ):
            # do 1st-order update
            x_img_new = scheduler.dpm_solver_first_order_update(
                model_outputs_img[-1], sample=x_img, noise=noise
            )
            x_tab_new = scheduler.dpm_solver_first_order_update(
                model_outputs_tab[-1], sample=x_tab, noise=noise
            )
        elif (
                scheduler.config.solver_order == 2
                or lower_order_nums < 2
                or second_to_last
        ):
            # do 2nd-order update
            x_img_new = scheduler.multistep_dpm_solver_second_order_update(
                model_outputs_img, sample=x_img, noise=noise
            )
            x_tab_new = scheduler.multistep_dpm_solver_second_order_update(
                model_outputs_tab, sample=x_tab, noise=noise
            )
        else:
            # 3rd order (if configured)
            x_img_new = scheduler.multistep_dpm_solver_third_order_update(
                model_outputs_img, sample=x_img, noise=noise
            )
            x_tab_new = scheduler.multistep_dpm_solver_third_order_update(
                model_outputs_tab, sample=x_tab, noise=noise
            )

        # increment "lower_order_nums"
        if lower_order_nums < scheduler.config.solver_order:
            lower_order_nums += 1

        return x_img_new, x_tab_new, model_outputs_img, model_outputs_tab, lower_order_nums



def load_diffusion(cfg: DictConfig, dit_model: nn.Module, **overrides: Any) -> nn.Module:
    """
    Load a MultiModalDiffusion model from config, injecting a pre-loaded DiT.
    """
    from utils.configurations import apply_overrides
    cfg = apply_overrides(cfg, overrides)
    print("[INFO] Loading Diffusion model with config:", cfg)

    if "diffusion" in cfg:
        diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg.diffusion)
    else:
        diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg)

    print("[INFO] Loaded Diffusion Model")
    return diffusion_model



if __name__ == "__main__":
    # Suppose we have:
    from torch import optim, Tensor

    # 1) A "MultiModalDiT" instance that expects:
    #    model(x_img, x_tab, time_scalar) -> {"img_out":..., "tab_out":...}
    from models.dit.dit_multimodal_add12 import MultiModalDiT
    import numpy as np

    B = 4
    H, W = 32, 32
    in_chans = 4
    d_tab = 157

    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth = 16

    model = MultiModalDiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=512,
        depth=depth,
        head_dim=16,
        multiple_of=64,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=256,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
        num_tab_columns=157,
        tab_groups=10,
        out_table_features=157
    )

    # 2) Our EDM diffuser
    diffuser = MultiModalDiffusion(model)

    # 3) Some example training batch
    x_img = torch.randn(B, in_chans, H, W)
    x_tab = torch.randn(B, d_tab)

    # 4) forward => get loss
    loss_total, loss_img, loss_tab = diffuser(x_img, x_tab)
    print(f"total loss={loss_total}, img loss={loss_img}, tab loss={loss_tab}")