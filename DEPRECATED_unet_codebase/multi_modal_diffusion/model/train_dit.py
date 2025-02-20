import torch
import torch.nn as nn
import numpy as np


class UncondDiffusion(nn.Module):
    """
    Minimal unconditional diffusion model that can optionally run in latent space
    via a VAE (if provided).
    """

    def __init__(self, dit: nn.Module, vae: nn.Module = None,
                 sigma_min=0.002, sigma_max=80, p_mean=-0.6, p_std=1.2,
                 sigma_data=0.9, num_steps=18, rho=7, s_churn=0,
                 s_min=0, s_max=float('inf'), s_noise=1.0):
        """
        Args:
            dit:      The DiT backbone (unconditional).
            vae:      Optional VAE for encoding/decoding. If None, trains in pixel space.
            sigma_min, sigma_max: The EDM sigma range.
            p_mean, p_std: Log-normal parameters for sampling training sigmas.
            sigma_data: EDM constant for weighting the loss.
            num_steps: default # steps if you do sampling with `sample()`.
            rho, s_churn, s_min, s_max, s_noise: Additional EDM sampling hyperparams.
        """
        super().__init__()
        self.dit = dit
        self.vae = vae

        # Freeze VAE if given
        if self.vae is not None:
            self.vae.requires_grad_(False)

        # If your VAE has a scaling_factor, store it. Else default to 1.0
        self.latent_scale = getattr(self.vae.config, 'scaling_factor', 1.0) if vae else 1.0

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass is not used directly as a standard "nn.Module forward".
        We'll have a separate training_step(...) to compute the EDM loss.
        But you can keep a pass if you want to do a direct forward for something else.
        """
        return x  # No default logic here

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given *clean* images or latents x, returns the EDM loss scalar.

        If you have a VAE and want to train in latent space:
            x is latents from the VAE (like shape [B, 4, 32, 32]).
        If no VAE, x is the raw images (like [B, 3, 64, 64]) scaled in [-1,1] or [0,1].
        """
        # 1) Sample a log-normal sigma
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        # 2) Noise weight factor
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)
        # 3) Add noise
        noise = torch.randn_like(x)
        noised_input = x + sigma * noise

        # 4) EDM model forward => predicted denoised
        #   We use c_in, c_skip, c_out approach from original code
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1 / (self.sigma_data ** 2 + sigma_in ** 2).sqrt()
        c_skip = (self.sigma_data ** 2 / (sigma_in ** 2 + self.sigma_data ** 2))
        c_out = sigma_in * self.sigma_data / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()

        model_out = self.dit(
            x=c_in * noised_input,  # scale input
            t=(sigma_in.log() / 4).squeeze(),  # shape (B,)
        )  # your DiT returns {'sample': predicted_noise} or predicted x0?
        F_x = model_out['image_sample']  # shape same as x

        # Combine
        D_xn = c_skip * noised_input + c_out * F_x

        # 5) Compute MSE
        loss = weight * ((D_xn - x) ** 2)
        return loss.mean()

    @torch.no_grad()
    def encode_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        If a VAE is present, encode x->latents. Otherwise, x is used as-is.
        """
        if self.vae is not None:
            latents = self.vae.encode(x)['latent_dist'].sample()
            return latents * self.latent_scale
        else:
            return x

    @torch.no_grad()
    def decode_if_needed(self, latents: torch.Tensor) -> torch.Tensor:
        """
        If a VAE is present, decode latents->image. Otherwise, latents is already final.
        """
        if self.vae is not None:
            latents = latents / self.latent_scale
            img = self.vae.decode(latents).sample
            # Typically many VAEs produce images in [-1,1], or [-0.5, 0.5].
            # But you can clamp or scale if you prefer:
            img = (img / 2 + 0.5).clamp(0, 1)
            return img
        else:
            return latents  # assume it's already in 0..1 or something

    ########################
    #  EDM Sampling Loop  #
    ########################
    @torch.no_grad()
    def sample(
            self, batch_size=4, height=64, width=64,
            steps=None, device='cuda', save_path=None
    ) -> torch.Tensor:
        """
        Sample images unconditionally from pure noise using an EDM approach.
        If a VAE is used, we sample in latent space, then decode.

        Args:
            batch_size: how many images in a batch
            height, width: final image size if no VAE; if using VAE, must match latent dims
            steps: number of steps in sampling loop
            device: device to run
            save_path: if not None, a file path (or directory) to save the final grid

        Returns:
            A tensor of shape (B,C,H,W) in [0,1] range.
        """
        steps = steps or self.num_steps
        # If using a VAE with e.g. in_channels=4, resolution=32, do that
        c = self.dit.in_channels
        if self.vae is not None:
            # typical SD latents => (4, 32, 32) by default
            H = W = int(height)  # assume user knows the latents shape or store them
        else:
            # pixel space
            H, W = height, width

        shape = (batch_size, c, H, W)
        x = torch.randn(shape, device=device)

        # Create time steps
        t_steps = self.create_edm_timesteps(steps, device)

        x_next = x.double() * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = (
                min(self.s_churn / steps, np.sqrt(2) - 1)
                if (self.s_min <= t_cur <= self.s_max) else 0
            )
            t_hat = t_cur + gamma * t_cur
            # Add extra noise
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * torch.randn_like(x_cur)

            # 1) Euler step
            denoised = self.edm_model_forward(x_hat, t_hat).double()
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2) 2nd order correction
            if i < steps - 1:
                denoised = self.edm_model_forward(x_next, t_next).double()
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # final latents or images
        samples = self.decode_if_needed(x_next.float())
        if save_path is not None:
            # Save a grid of images
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(samples, save_path, nrow=int(batch_size ** 0.5))
        return samples

    @torch.no_grad()
    def edm_model_forward(self, x_in, t_sigma):
        """
        Tiny helper for the sampling loop that calls self.dit with the scaled input + time emb.
        Return the "denoised" or "prediction".
        """
        # # from the original code:
        # sigma_in = t_sigma.reshape(-1, 1, 1, 1).float()
        # c_in = 1.0 / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()
        # c_skip = self.sigma_data ** 2 / (sigma_in ** 2 + self.sigma_data ** 2)
        # c_out = sigma_in * self.sigma_data / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()
        #
        # out = self.dit(
        #     x=c_in * x_in.float(),
        #     t=(sigma_in.log() / 4).squeeze()
        # )
        # F_x = out['sample']
        # return c_skip * x_in + c_out * F_x
        sigma_in = t_sigma.reshape(-1, 1, 1, 1).float()
        c_in = 1.0 / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()
        c_skip = self.sigma_data ** 2 / (sigma_in ** 2 + self.sigma_data ** 2)
        c_out = sigma_in * self.sigma_data / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()

        # Compute t and expand to batch size
        t = (sigma_in.log() / 4).squeeze()  # This becomes a scalar if sigma_in is (1,1,1,1)
        batch_size = x_in.size(0)
        t_expanded = t.expand(batch_size)  # Shape (batch_size,)

        out = self.dit(
            x=c_in * x_in.float(),
            t=t_expanded
        )
        F_x = out['image_sample']  # Adjust key if necessary
        return c_skip * x_in + c_out * F_x

    def create_edm_timesteps(self, steps, device):
        """
        Make time steps t0...tN using the EDM method.
        """
        step_indices = torch.arange(steps, dtype=torch.float64, device=device)
        inv_rho = 1.0 / self.rho
        t_values = (
                           self.sigma_max ** inv_rho +
                           step_indices / (steps - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)
                   ) ** self.rho
        # Add zero final step
        t_values = torch.cat([t_values, torch.zeros_like(t_values[:1])])
        return t_values


import torch
from torch import optim
from torchvision.utils import save_image
import os

def train_uncond_diffusion(
    model: UncondDiffusion,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 200,
    lr: float = 1e-4,
    device: str = 'cuda',
    log_interval: int = 1,
    sample_interval: int = 100,
    sample_save_dir: str = "/mnt/storage/nacc_sub/mm_dit_NO_LAT"
):
    """
    Minimal training loop for UncondDiffusion with EDM.
    """
    os.makedirs(sample_save_dir, exist_ok=True)
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    step = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            step += 1

            # Suppose your batch is just images
            images = batch['image'].to(device)  # or however your dataset is structured

            # If using a VAE, encode to latents:
            latents = model.encode_if_needed(images)

            # Compute the EDM loss
            loss = model.training_step(latents)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training loss
            if step % log_interval == 0:
                print(f"[Epoch {epoch+1}/{num_epochs} | Step {step}] loss: {loss.item():.4f}")

            # Occasionally generate and save samples
            if step % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    samples = model.sample(batch_size=4, device=device, save_path=None)
                    # Save a grid of the samples
                    sample_file = os.path.join(sample_save_dir, f'sample_step_{step}.png')
                    save_image(samples, sample_file, nrow=2)
                    print(f"Saved sample image at step {step} -> {sample_file}")
                model.train()

        # End of epoch, optionally save a checkpoint
        # torch.save(model.state_dict(), f"uncond_diff_ckpt_epoch_{epoch+1}.pt")

    print("Training complete!")


def main():
    # 1) Build your unconditional DiT (from your code)

    from multi_modal_diffusion.model.dit_sm import DiT
    from diffusion_process.dataloaders import load_training_data
    from diffusion_process.enums import DatasetType
    from unittest.mock import MagicMock

    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth = 16

    my_dit = DiT(
        input_size=64,
        patch_size=4,
        in_channels=3,
        dim=512,
        depth=16,
        head_dim=32,
        multiple_of=64,
        pos_interp_scale=1.0,
        norm_eps=1e-6,
        depth_init=True,
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),
        use_patch_mixer=True,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
        use_bias=False,
        num_experts=8,
        expert_capacity=2.0,
        experts_every_n=2,
    )

    # 3) Wrap them in UncondDiffusion
    uncond_model = UncondDiffusion(dit=my_dit)

    # 4) Create your dataset/dataloader
    mock_args = MagicMock()
    mock_args.dataset_type = DatasetType.IMAGE_TABULAR
    mock_args.data_dir = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset/middle_slice"
    mock_args.batch_size = 8
    mock_args.num_workers = 0
    train_loader = load_training_data(mock_args)

    # 5) Train
    train_uncond_diffusion(
        model=uncond_model,
        train_loader=train_loader,
        num_epochs=200,
        lr=1e-4,
        device='cuda'
    )

if __name__ == "__main__":
    main()



