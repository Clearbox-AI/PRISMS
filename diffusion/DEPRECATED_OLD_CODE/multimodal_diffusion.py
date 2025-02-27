# models/multimodal_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchvision.utils import save_image
import torch.distributed as dist
from utils.ddp import is_main_process


class MultiModalDiffusion(nn.Module):
    """
    EDM-based multi-modal diffusion for images + tabular data,
    optionally using a VAE.
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
        latent_reg_weight=0.0
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
        self.train_mask_ratio = train_mask_ratio
        self.latent_reg_weight = latent_reg_weight


    # def forward(self, images: torch.Tensor, table_data: torch.Tensor):
    def forward(self, images: torch.Tensor, table_data: torch.Tensor, global_step: int = 0):
        """
        EDM loss: sample sigma, noise, pass to DiT, compute MSE.
        """
        device = images.device
        B = images.shape[0]

        # 1) log-normal sigma
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()

        # 2) Weight
        weight = ((sigma**2 + self.sigma_data**2) / (sigma*self.sigma_data)**2)
        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # 3) Noise
        noise = torch.randn_like(images)
        noised_input = images + sigma * noise

        # 4) EDM scaling
        # c_in, c_skip, c_out from Karras et al. (EDM)
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1.0 / (self.sigma_data**2 + sigma_in**2).sqrt()
        c_skip = self.sigma_data**2 / (sigma_in**2 + self.sigma_data**2)
        c_out = sigma_in * self.sigma_data / (sigma_in**2 + self.sigma_data**2).sqrt()
        t = (sigma_in.log() / 4.0).squeeze()

        # 5) DiT forward
        out = self.dit(
            x_img=c_in * noised_input,
            t=t,
            tab=table_data,
            cfg=1.0,  # no CFG in training
            mask_ratio=self.train_mask_ratio
        )
        F_x = out['image_sample']

        # ---------------------------
        #  DEBUG: LOG SCALE OF F_x
        # ---------------------------
        # For instance, log every 50 steps only on the main process.
        log_interval = 50
        if is_main_process() and (global_step % log_interval == 0):
            log_dir = "/mnt/storage/nacc_sub/mm_dit_con_vae/tmp"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "debug_scales.csv")

            # Collect simple stats
            f_mean = F_x.mean().item()
            f_std = F_x.std().item()
            lat_mean = images.mean().item()
            lat_std = images.std().item()

            # Append to CSV
            with open(log_file, "a") as f:
                # step, f_mean, f_std, lat_mean, lat_std
                f.write(f"{global_step},latents: {f_mean:.5f},{f_std:.5f}, real: {lat_mean:.5f},{lat_std:.5f}\n")

        # Combine
        D_xn = c_skip * noised_input + c_out * F_x
        loss_img = weight * ((D_xn - images) ** 2)
        image_loss = loss_img.mean(dim=[1, 2, 3]).mean()

        # ================================
        #  SIMPLE LATENT MSE REGULARIZATION
        # ================================
        if self.latent_reg_weight > 0:
            # Encourage the raw DiT output (F_x) to match real latents `images`
            # reg_loss = F.mse_loss(F_x, images, reduction='mean')
            # image_loss = image_loss + self.latent_reg_weight * reg_loss

            # or, if you only want to constrain the last 2 channels:
            # reg_loss = F.mse_loss(F_x[:, 2:, :, :], images[:, 2:, :, :])
            # image_loss = image_loss + self.latent_reg_weight * reg_loss

            real_mean = images.mean(dim=(0, 2, 3), keepdim=True)
            real_std = images.std(dim=(0, 2, 3), keepdim=True)
            pred_mean = F_x.mean(dim=(0, 2, 3), keepdim=True)
            pred_std = F_x.std(dim=(0, 2, 3), keepdim=True)
            mean_loss = F.mse_loss(pred_mean, real_mean)
            std_loss = F.mse_loss(pred_std, real_std)
            reg_loss = mean_loss + std_loss
            image_loss = image_loss + self.latent_reg_weight * reg_loss
        # ================================

        tab_loss = None
        if torch.is_tensor(table_data):
            pred_tab = out['table_sample']
            tab_loss_val = F.mse_loss(pred_tab, table_data, reduction='none').mean(dim=1)
            tab_loss = tab_loss_val.mean()
            total_loss = image_loss + tab_loss
        else:
            total_loss = image_loss

        return total_loss, image_loss, tab_loss

    @torch.no_grad()
    def sample(self, batch_size=4, table_data=None, cfg=1.0,
               steps=None, height=64, width=64, device='cuda',
               save_path=None):
        """
        EDM sampler: produce latents & decode to images (if VAE).
        """
        self.eval()
        steps = steps or self.num_steps
        c = self.dit.in_channels

        x = torch.randn((batch_size, c, height, width), device=device)
        t_vals = self.create_edm_timesteps(steps, device)
        x_next = x.double() * t_vals[0]
        latest_tab_sample = None

        def model_forward(x_in, t_sigma, tab_data, cfg_val):
            B = x_in.shape[0]
            sigma_in = t_sigma.reshape(-1,1,1,1).float()
            c_in = 1.0 / (sigma_in**2 + self.sigma_data**2).sqrt()
            c_skip = self.sigma_data**2 / (sigma_in**2 + self.sigma_data**2)
            c_out = sigma_in * self.sigma_data / (sigma_in**2 + self.sigma_data**2).sqrt()

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
            tab_sample = out.get('table_sample', None)
            denoised = c_skip * x_in + c_out * Fx
            return denoised, tab_sample

        # Sampler loop
        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_cur = x_next
            gamma = (min(self.s_churn/steps, np.sqrt(2)-1)
                     if (self.s_min<=t_cur<=self.s_max) else 0)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.s_noise*torch.randn_like(x_cur)

            # Euler
            denoised, tab_sample = model_forward(x_hat, t_hat, table_data, cfg)
            denoised = denoised.double()
            if tab_sample is not None:
                tab_sample = tab_sample.double()
                latest_tab_sample = tab_sample.detach().cpu()
            d_cur = (x_hat - denoised)/t_hat
            x_next = x_hat + (t_next - t_hat)*d_cur

            # 2nd order
            if i < steps-1:
                denoised2, tab_sample2 = model_forward(x_next, t_next, table_data, cfg)
                denoised2 = denoised2.double()
                if tab_sample2 is not None:
                    tab_sample2 = tab_sample2.double()
                    latest_tab_sample = tab_sample2.detach().cpu()
                d_prime = (x_next - denoised2)/t_next
                x_next = x_hat + (t_next - t_hat)*(0.5*d_cur + 0.5*d_prime)

        final_latents = x_next.float()

        # Optionally save
        if save_path is not None:
            # This is purely optional for debugging; not typical to visualize raw latents
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # e.g., just clamp and do some fake normalization?
            latents_for_vis = (final_latents - final_latents.min()) / (final_latents.max() - final_latents.min() + 1e-7)
            save_image(latents_for_vis, save_path, nrow=int(batch_size ** 0.5))

        return final_latents, latest_tab_sample

    def create_edm_timesteps(self, steps, device):
        step_indices = torch.arange(steps, dtype=torch.float64, device=device)
        inv_rho = 1.0/self.rho
        t_values = (
            self.sigma_max**inv_rho +
            step_indices/(steps-1)*(self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        )**self.rho
        # append zero
        t_values = torch.cat([t_values, torch.zeros_like(t_values[:1])])
        return t_values