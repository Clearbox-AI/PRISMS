import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import save_image

def is_main_process():
    """Return True iff the current process is global rank 0."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


class LatentsDiffusion(nn.Module):
    """
    An EDM-based diffusion class that learns a mapping from src_latents -> tgt_latents.

    The 'forward(...)' method implements the standard EDM loss:
      1) Sample random log-normal sigma
      2) Add noise to src_latents
      3) Pass the noised src_latents to the diffusion model
      4) Compare final denoised output to tgt_latents

    The 'sample(...)' method performs EDM sampling to generate target-like latents,
    typically starting from random noise.
    """

    def __init__(
        self,
        dit: nn.Module,
        sigma_min=0.002,
        sigma_max=80,
        p_mean=-0.6,
        p_std=1.2,
        sigma_data=0.9,
        num_steps=18,
        rho=7,
        s_churn=0,
        s_min=0,
        s_max=float('inf'),
        s_noise=1.0,
        train_mask_ratio=0.0,
    ):
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

        # If you still use patch masking in DiT, you can set a ratio here
        self.train_mask_ratio = train_mask_ratio

    def forward(
        self,
        src_latents: torch.Tensor,
        tgt_latents: torch.Tensor,
        global_step: int = 0
    ) -> torch.Tensor:
        """
        EDM loss for latents -> latents.
        We treat src_latents as the "input" domain, but we want to predict tgt_latents.

        Args:
            src_latents: (B, C, H, W)
            tgt_latents: (B, C, H, W) same shape as src_latents
            global_step:  current training step index (for logging)

        Returns:
            A scalar loss (mean over batch).
        """
        device = src_latents.device
        B = src_latents.shape[0]
        # 1) Sample random log-normal sigma
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()  # => shape (B,1,1,1)

        # 2) Compute weighting factor from EDM
        weight = ((sigma**2 + self.sigma_data**2) / (sigma*self.sigma_data)**2)

        # 3) Add noise to the *source* latents
        noise = torch.randn_like(tgt_latents)
        noised_tgt = tgt_latents + sigma * noise

        # 4) Compute the EDM scaling factors
        sigma_in = sigma.reshape(-1, 1, 1, 1) # (B,1,1,1)
        c_in = 1.0 / (self.sigma_data ** 2 + sigma_in ** 2).sqrt()
        c_skip = self.sigma_data ** 2 / (sigma_in ** 2 + self.sigma_data ** 2)
        c_out = sigma_in * self.sigma_data / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()
        t = (sigma_in.log() / 4.0).squeeze()  # => shape (B,)

        # 5) Run DiT forward
        out = self.dit(
            x_img=c_in * noised_tgt,  # shape (B, C, H, W)
            y_img=src_latents,
            t=t,
            mask_ratio=self.train_mask_ratio
        )
        F_src = out['image_sample']  # the model's unscaled output, shape (B,C,H,W)

        # 6) Combine with skip connection (Karras eqn.)
        D_xn = c_skip * noised_tgt + c_out * F_src

        # 7) The MSE is vs. the *target* latents, not the source latents
        loss_img = weight * ((D_xn - tgt_latents) ** 2)  # => shape (B,C,H,W)
        image_loss = loss_img.mean(dim=[1,2,3]).mean()  # scalar

        # Optionally do some logging
        if is_main_process() and (global_step % 50 == 0):
            # Example: log stats
            print(f"[Step={global_step}]  Loss={image_loss.item():.4f}")

        return image_loss

    @torch.no_grad()
    def sample(self, batch_size=4, y=None, steps=None, height=64, width=64, device='cuda', save_path=None) -> torch.Tensor:
        """
        EDM sampler that generates "target latents" from noise.

        Returns:
            final_latents: shape (B, C, H, W)
        """
        self.eval()
        steps = steps or self.num_steps

        c = self.dit.in_channels  # typically 4 or similar
        x = torch.randn((batch_size, c, height, width), device=device)

        # Create the timesteps
        t_vals = self.create_edm_timesteps(steps, device)
        x_next = x.double() * t_vals[0]

        # Helper for each model step
        def model_forward(x_in, condition, t_sigma):
            B_ = x_in.shape[0]
            sigma_in = t_sigma.view(-1,1,1,1).float()
            c_in = 1.0 / (sigma_in**2 + self.sigma_data**2).sqrt()
            c_skip = self.sigma_data**2 / (sigma_in**2 + self.sigma_data**2)
            c_out = sigma_in * self.sigma_data / (sigma_in**2 + self.sigma_data**2).sqrt()

            t_embed = (sigma_in.log() / 4.0).reshape(-1)
            if t_embed.numel() == 1 and B_ > 1:
                t_embed = t_embed.expand(B_)

            out_ = self.dit(
                x_img=c_in * x_in.float(),
                y_img=condition,
                t=t_embed,
                mask_ratio=0.0
            )
            Fx = out_['image_sample'].float()
            denoised = c_skip * x_in + c_out * Fx
            return denoised

        # Sampler loop (Euler + Heun)
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_cur = x_next
            gamma = (
                min(self.s_churn/steps, math.sqrt(2)-1)
                if (self.s_min <= t_cur <= self.s_max) else 0
            )
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.s_noise * torch.randn_like(x_cur)

            # Euler step
            denoised = model_forward(x_hat, y, t_hat).double()
            d_cur = (x_hat - denoised)/t_hat
            x_next = x_hat + (t_next - t_hat)*d_cur

            # 2nd order correction
            if i < steps-1:
                denoised2 = model_forward(x_next, y, t_next).double()
                d_prime = (x_next - denoised2)/t_next
                x_next = x_hat + (t_next - t_hat)*(0.5*d_cur + 0.5*d_prime)

        final_latents = x_next.float()

        # Optionally save latents to an image (purely for debug)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            lat_for_vis = (final_latents - final_latents.min()) / (final_latents.max()-final_latents.min() +1e-7)
            save_image(lat_for_vis, save_path, nrow=int(batch_size**0.5))

        return final_latents

    def create_edm_timesteps(self, steps: int, device: str):
        """
        Create t-values as in Karras et al. (EDM).
        """
        step_indices = torch.arange(steps, dtype=torch.float64, device=device)
        inv_rho = 1.0/self.rho
        t_values = (
            self.sigma_max**inv_rho
            + step_indices/(steps-1)*(self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        )**self.rho
        t_values = torch.cat([t_values, torch.zeros_like(t_values[:1])])
        return t_values
