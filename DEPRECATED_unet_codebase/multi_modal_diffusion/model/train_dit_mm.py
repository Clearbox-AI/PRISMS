import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision.utils import save_image
import pandas as pd
import torch
import os
from torch import optim
from torchvision.utils import save_image
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
from diffusion_process.dataloaders import load_training_data
from diffusion_process.enums import DatasetType
from unittest.mock import MagicMock
import random


@torch.no_grad()
def sample_cond_and_uncond(
    self,
    batch_size: int = 4,
    table_data: torch.Tensor = None,
    cfg: float = 1.0,
    steps: int = None,
    height: int = 64,
    width: int = 64,
    device: str = 'cuda',
    save_path: str = None,
    save_prefix: str = "sample"
):
    """
    Generates *two* sets of outputs in a single function call:
      1) Conditional (using `table_data`)
      2) Unconditional (using `None`)

    Returns:
      (cond_imgs, cond_tab_out, uncond_imgs, uncond_tab_out)
    """
    self.eval()

    # --------- 1) Conditional sampling --------- #
    # If table_data is not None, we do the normal conditional sample.
    cond_imgs, cond_tab_out = self.sample(
        batch_size=batch_size,
        table_data=table_data,  # Use the provided table data
        cfg=cfg,
        steps=steps,
        height=height,
        width=width,
        device=device,
        save_path=None,  # We'll handle saving ourselves below
    )

    # --------- 2) Unconditional sampling --------- #
    # For the unconditional version, we pass None
    uncond_imgs, uncond_tab_out = self.sample(
        batch_size=batch_size,
        table_data=None,  # unconditional
        cfg=cfg,
        steps=steps,
        height=height,
        width=width,
        device=device,
        save_path=None,
    )

    # Optionally save images to disk
    if save_path is not None:
        # e.g. save the conditional images
        cond_file = os.path.join(os.path.dirname(save_path), f"{save_prefix}_cond.png")
        save_image(cond_imgs, cond_file, nrow=int(batch_size**0.5))
        print(f"Saved conditional images to {cond_file}")

        # e.g. save the unconditional images
        uncond_file = os.path.join(os.path.dirname(save_path), f"{save_prefix}_uncond.png")
        save_image(uncond_imgs, uncond_file, nrow=int(batch_size**0.5))
        print(f"Saved unconditional images to {uncond_file}")

        # If you also want to save tabular outputs
        if cond_tab_out is not None:
            cond_csv = os.path.join(os.path.dirname(save_path), f"{save_prefix}_cond_table.csv")
            pd.DataFrame(cond_tab_out.numpy()).to_csv(cond_csv, index=False)
            print(f"Saved conditional table data to {cond_csv}")

        if uncond_tab_out is not None:
            uncond_csv = os.path.join(os.path.dirname(save_path), f"{save_prefix}_uncond_table.csv")
            pd.DataFrame(uncond_tab_out.numpy()).to_csv(uncond_csv, index=False)
            print(f"Saved unconditional table data to {uncond_csv}")

    return cond_imgs, cond_tab_out, uncond_imgs, uncond_tab_out


class MultiModalDiffusion(nn.Module):
    """
    EDM-based multi-modal diffusion for images + tabular.
    Optionally uses a patch masking ratio *only* during training.
    """

    def __init__(
        self,
        dit: nn.Module,              # your cross-attn DiT
        vae: nn.Module = None,       # optional VAE
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
        train_mask_ratio=0.0
    ):
        """
        train_mask_ratio: fraction of patches to randomly mask during training.
                          (0 => no masking)
        """
        super().__init__()
        self.dit = dit
        self.vae = vae
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.latent_scale = getattr(self.vae.config, 'scaling_factor', 1.0)
        else:
            self.latent_scale = 1.0

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

        self.train_mask_ratio = train_mask_ratio  # only used if training

    def encode_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is not None:
            latents = self.vae.encode(x)['latent_dist'].sample()
            return latents * self.latent_scale
        else:
            return x

    def decode_if_needed(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is not None:
            latents = latents / self.latent_scale
            img = self.vae.decode(latents).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            return img
        else:
            return latents

    def forward(self, x, table_data=None, cfg=1.0, mask_ratio=0.0):
        """
        Typically unused direct forward.
        The training uses training_step(...),
        sampling uses sample(...).
        """
        return x

    def training_step(self, images: torch.Tensor, table_data: torch.Tensor) -> torch.Tensor:
        """
        EDM loss.  We add a random log‐normal sigma => noised input => pass to DiT.
        Incorporate optional patch masking via `self.train_mask_ratio`.
        """
        device = images.device
        B = images.shape[0]

        # 1) Sample log-normal sigma
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()

        # 2) Weight factor
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)

        # 3) Add noise
        noise = torch.randn_like(images)
        noised_input = images + sigma * noise

        # 4) EDM scaling
        sigma_in = sigma.reshape(-1, 1, 1, 1)
        c_in = 1 / (self.sigma_data ** 2 + sigma_in ** 2).sqrt()
        c_skip = self.sigma_data ** 2 / (sigma_in ** 2 + self.sigma_data ** 2)
        c_out = sigma_in * self.sigma_data / (sigma_in ** 2 + self.sigma_data ** 2).sqrt()
        t = (sigma_in.log() / 4).squeeze()

        # 5) Pass to DiT => pass `mask_ratio=self.train_mask_ratio`
        out = self.dit(
            x_img=(c_in * noised_input),
            t=t,
            tab=table_data,
            cfg=1.0,  # typically no guidance during training
            mask_ratio=self.train_mask_ratio
        )
        F_x = out['image_sample']  # shape (B, C, H, W)

        # Combine
        D_xn = c_skip * noised_input + c_out * F_x
        loss_img = weight * ((D_xn - images) ** 2)
        image_loss_per_sample = loss_img.mean(dim=[1, 2, 3])
        image_loss = image_loss_per_sample.mean()

        # Initialize tabular loss to None
        tab_loss = None

        if torch.is_tensor(table_data):
            pred_tab = out['table_sample']
            loss_tab_per_sample = F.mse_loss(pred_tab, table_data, reduction='none').mean(dim=1)
            tab_loss = loss_tab_per_sample.mean()
            total_loss = image_loss + tab_loss
        else:
            total_loss = image_loss

        # total_loss = image_loss + tab_loss
        # else:
        #     total_loss = image_loss

        return total_loss, image_loss, tab_loss

    @torch.no_grad()
    def sample(
        self,
        batch_size=4,
        table_data=None,
        cfg=1.0,
        steps=None,
        height=32,
        width=32,
        device='cuda',
        save_path=None
    ):
        """
        EDM sampling loop. We forcibly set `mask_ratio=0` => no patch masking in inference.
        """
        self.eval()
        steps = steps or self.num_steps

        c = self.dit.in_channels
        if self.vae is not None:
            shape = (batch_size, c, height, width)
        else:
            shape = (batch_size, c, height, width)

        x = torch.randn(shape, device=device)
        t_vals = self.create_edm_timesteps(steps, device)
        x_next = x.double()*t_vals[0]
        latest_tab_sample = None

        def model_forward(x_in, t_sigma, tab_data, cfg):
            B = x_in.shape[0]

            sigma_in = t_sigma.reshape(-1,1,1,1).float()
            c_in = 1.0/(sigma_in**2 + self.sigma_data**2).sqrt()
            c_skip = self.sigma_data**2/(sigma_in**2 + self.sigma_data**2)
            c_out = sigma_in*self.sigma_data/(sigma_in**2 + self.sigma_data**2).sqrt()

            # A single scalar => shape (1,) after reshape
            t_embed = (sigma_in.log() / 4).reshape(-1)  # => typically (1,)
            # expand to match the batch B => shape (B,)
            if t_embed.numel() == 1 and B > 1:
                t_embed = t_embed.expand(B)

            out = self.dit(
                x_img=c_in*x_in.float(),
                t=t_embed,
                tab=tab_data,
                cfg=cfg,
                mask_ratio=0.0  # forced to 0 in inference
            )
            Fx = out['image_sample'].float()

            tab_sample = out.get('table_sample', None)
            return c_skip * x_in + c_out * Fx, tab_sample

        for i, (t_cur, t_next) in enumerate(zip(t_vals[:-1], t_vals[1:])):
            x_cur = x_next
            gamma = (min(self.s_churn/steps, np.sqrt(2)-1)
                     if (self.s_min<=t_cur<=self.s_max) else 0)
            t_hat = t_cur + gamma*t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt()*self.s_noise*torch.randn_like(x_cur)

            # euler
            denoised, tab_sample = model_forward(x_hat, t_hat, table_data, cfg)
            denoised, tab_sample = denoised.double(), tab_sample.double()
            latest_tab_sample = tab_sample.detach().cpu() if tab_sample is not None else None
            d_cur = (x_hat - denoised)/t_hat
            x_next = x_hat + (t_next - t_hat)*d_cur

            # second order
            if i < steps-1:
                denoised2, tab_sample2 = model_forward(x_next, t_next, table_data, cfg)
                denoised2, tab_sample2 = denoised2.double(), tab_sample2.double()
                latest_tab_sample = tab_sample2.detach().cpu() if tab_sample2 is not None else None
                d_prime = (x_next - denoised2)/t_next
                x_next = x_hat + (t_next - t_hat)*(0.5*d_cur + 0.5*d_prime)

        samples = self.decode_if_needed(x_next.float())
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(samples, save_path, nrow=int(batch_size**0.5))
        return samples, latest_tab_sample

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



def train_multimodal_diffusion(
    model: MultiModalDiffusion,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-4,
    device: str = 'cuda',
    log_interval: int = 10,
    sample_interval: int = 100,
    uncond_prob: float = 0.2,
    sample_save_dir: str = "/mnt/storage/nacc_sub/mm_dit_NO_LAT/samples",
    save_model_interval: int = None,  # New: Interval to save models (steps)
    model_save_dir: str = "/mnt/storage/nacc_sub/mm_dit_NO_LAT/models"
):


    os.makedirs(sample_save_dir, exist_ok=True)
    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)

    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    step = 0
    last_total_loss = None  # Track the last loss value

    for epoch in range(num_epochs):
        for batch in train_loader:
            step += 1
            # batch => {'image':(B,C,H,W), 'table':(B,num_cols)}
            images = batch['image'].to(device)
            tab_data = batch['tabular'].to(device)

            # Possibly drop the tabular data
            if random.random() < uncond_prob:
                tab_data = None  # unconditional

            # Encode images to latents if using VAE
            latents = model.encode_if_needed(images)

            # Compute loss and backpropagate
            total_loss, image_loss, tab_loss = model.training_step(latents, tab_data)
            last_total_loss = total_loss.item()  # Update last loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if step % log_interval == 0:
                log_msg = f"Epoch {epoch + 1}, Step {step}, Image Loss: {image_loss.item():.4f}"
                if tab_loss is not None:
                    log_msg += f", Tabular Loss: {tab_loss.item():.4f}"
                log_msg += f", Total Loss: {total_loss.item():.4f}"
                if tab_data is None:
                    log_msg += " [UNCOND]"
                print(log_msg)

            # Sampling
            if step % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    sub_tab = batch['tabular'][:4].to(device)

                    # (A) Conditional sampling
                    cond_samples, cond_tab_samples = model.sample(
                        batch_size=4,
                        table_data=sub_tab,  # use real tab data
                        cfg=1.0,
                        steps=None,
                        height=64,
                        width=64,
                        device=device,
                        save_path=None
                    )
                    # Save conditional samples
                    cond_img_file = os.path.join(sample_save_dir, f"samples_step_{step}_cond.png")
                    save_image(cond_samples, cond_img_file, nrow=2)
                    print(f"[Sampling] Saved *conditional* sample at {cond_img_file}")

                    cond_csv = os.path.join(sample_save_dir, f"tabular_step_{step}_cond.csv")
                    pd.DataFrame(cond_tab_samples.numpy()).to_csv(cond_csv, index=False)
                    print(f"[Sampling] Saved *conditional* tabular data at {cond_csv}")

                    # (B) Unconditional sampling
                    uncond_samples, uncond_tab_samples = model.sample(
                        batch_size=4,
                        table_data=None,  # no tab => unconditional
                        cfg=1.0,
                        steps=None,
                        height=64,
                        width=64,
                        device=device,
                        save_path=None
                    )
                    # Save unconditional samples
                    uncond_img_file = os.path.join(sample_save_dir, f"samples_step_{step}_uncond.png")
                    save_image(uncond_samples, uncond_img_file, nrow=2)
                    print(f"[Sampling] Saved *unconditional* sample at {uncond_img_file}")

                    uncond_csv = os.path.join(sample_save_dir, f"tabular_step_{step}_uncond.csv")
                    pd.DataFrame(uncond_tab_samples.numpy()).to_csv(uncond_csv, index=False)
                    print(f"[Sampling] Saved *unconditional* tabular data at {uncond_csv}")

                    # sub_tab = tab_data[:4]
                    # samples, tab_samples  = model.sample(
                    #     batch_size=4,
                    #     table_data=sub_tab,
                    #     cfg=1.0,
                    #     steps=None,
                    #     height=64,
                    #     width=64,
                    #     device=device,
                    #     save_path=None
                    # )
                    # sample_file = os.path.join(sample_save_dir, f"samples_step_{step}.png")
                    # save_image(samples, sample_file, nrow=2)
                    # print(f"Saved sample at {sample_file}")
                    #
                    # if tab_samples is not None:
                    #     tab_np = tab_samples.numpy()
                    #     csv_file = os.path.join(sample_save_dir, f"tabular_step_{step}.csv")
                    #     pd.DataFrame(tab_np).to_csv(csv_file, index=False)
                    #     print(f"Saved tabular data at {csv_file}")
                model.train()

            # Interval-based model saving
            if save_model_interval is not None and (step % save_model_interval == 0):
                checkpoint_path = os.path.join(model_save_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                }, checkpoint_path)
                print(f"Saved model checkpoint at step {step} to {checkpoint_path}")

    # Save the last checkpoint if it wasn't already saved.
    if (step - 1) % save_model_interval != 0:
        final_checkpoint_path = os.path.join(model_save_dir, f"checkpoint_step_{step}_final.pt")
        torch.save({
            'step': step,
            'epoch': num_epochs,  # Total epochs completed
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': last_total_loss,
        }, final_checkpoint_path)
        print(f"\nSaved FINAL model checkpoint at step {step}")
    print("Training complete!")



def main():
    # 1) Build your multi-modal DiT that does cross-attn

    qkv_ratio = [0.5, 1.0]
    mlp_ratio = [0.5, 4.0]
    depth = 16

    dit_model = MultiModalDiT(
        input_size=64,
        patch_size=4,
        in_channels=4,
        dim=256,
        depth=depth,
        head_dim=32,
        multiple_of=64,
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
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174
    )

    # 2) Optional: load a pretrained VAE (like from stable-diffusion)
    ...

    # 3) Create MultiModalDiffusion
    mm_diff_model = MultiModalDiffusion(
        dit=dit_model,
        vae=None,       # or None if training in pixel space
        sigma_min=0.002,
        sigma_max=80,
        p_mean=-0.6,
        p_std=1.2,
        sigma_data=0.9,
        num_steps=18
    )

    # 4) Build your training DataLoader
    mock_args = MagicMock()
    mock_args.dataset_type = DatasetType.NACC_LATENTS
    mock_args.data_dir = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset_latents"
    mock_args.batch_size = 8
    mock_args.num_workers = 0
    train_loader = load_training_data(mock_args)

    # 5) Train
    train_multimodal_diffusion(
        model=mm_diff_model,
        train_loader=train_loader,
        num_epochs=200,
        lr=1e-4,
        device='cuda',
        log_interval=1,
        sample_interval=300,
        uncond_prob=0.2,
        sample_save_dir="/mnt/storage/nacc_sub/mm_dit_YES_LAT/samples",
        save_model_interval = 10000,
        model_save_dir = "/mnt/storage/nacc_sub/mm_dit_YES_LAT/models"
    )


if __name__ == "__main__":
    main()
