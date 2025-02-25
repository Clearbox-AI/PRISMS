# trainers/multimodal_trainer.py

import os
import torch
import random
import pandas as pd
from torch import optim
from torchvision.utils import save_image

import hydra
from omegaconf import DictConfig, OmegaConf

# Import your modules
from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae
from diffusion.multimodal_diffusion import MultiModalDiffusion
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
from diffusion_process.dataloaders import load_training_data

def train_multimodal_diffusion(model, train_loader, cfg: DictConfig):
    """
    Main training loop.
    """
    os.makedirs(cfg.training.sample_save_dir, exist_ok=True)
    os.makedirs(cfg.training.model_save_dir, exist_ok=True)

    model.to(cfg.training.device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)

    step = 0
    last_total_loss = None

    for epoch in range(cfg.training.epochs):
        for batch in train_loader:
            step += 1

            # 1) Get data
            images = batch['image'].to(cfg.training.device)  # [B, 3, H, W] in [0,1]
            tab_data = batch['tabular'].to(cfg.training.device)

            # Possibly drop tab => unconditional
            if random.random() < cfg.training.uncond_prob:
                tab_data = None

            # 2) Encode images -> latents if VAE is used
            latents = model.latents_encode(images)

            # 3) training_step
            total_loss, image_loss, tab_loss = model.training_step(latents, tab_data)
            last_total_loss = total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if step % cfg.training.log_interval == 0:
                msg = f"[Epoch {epoch+1} | Step {step}] "
                msg += f"Img Loss: {image_loss.item():.4f}"
                if tab_loss is not None:
                    msg += f" | Tab Loss: {tab_loss.item():.4f}"
                msg += f" | Total: {total_loss.item():.4f}"
                if tab_data is None:
                    msg += " [UNCOND]"
                print(msg)

            # Sampling
            if step % cfg.training.sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    # (A) Conditional sample
                    sub_tab = batch['tabular'][:4].to(cfg.training.device) if tab_data is not None else None
                    cond_imgs, cond_tab_out = model.sample(
                        batch_size=4,
                        table_data=sub_tab,
                        cfg=1.0,
                        steps=None,
                        height=cfg.data.image_size,
                        width=cfg.data.image_size,
                        device=cfg.training.device,
                        save_path=None
                    )
                    cond_file = os.path.join(cfg.training.sample_save_dir, f"samples_step_{step}_cond.png")
                    save_image(cond_imgs, cond_file, nrow=2)
                    print(f"Saved conditional images => {cond_file}")

                    if cond_tab_out is not None:
                        cond_csv = os.path.join(cfg.training.sample_save_dir, f"tabular_step_{step}_cond.csv")
                        pd.DataFrame(cond_tab_out.numpy()).to_csv(cond_csv, index=False)
                        print(f"Saved conditional table => {cond_csv}")

                    # (B) Unconditional sample
                    uncond_imgs, uncond_tab_out = model.sample(
                        batch_size=4,
                        table_data=None,
                        cfg=1.0,
                        steps=None,
                        height=cfg.data.image_size,
                        width=cfg.data.image_size,
                        device=cfg.training.device,
                        save_path=None
                    )
                    uncond_file = os.path.join(cfg.training.sample_save_dir, f"samples_step_{step}_uncond.png")
                    save_image(uncond_imgs, uncond_file, nrow=2)
                    print(f"Saved unconditional images => {uncond_file}")

                    if uncond_tab_out is not None:
                        uncond_csv = os.path.join(cfg.training.sample_save_dir, f"tabular_step_{step}_uncond.csv")
                        pd.DataFrame(uncond_tab_out.numpy()).to_csv(uncond_csv, index=False)
                        print(f"Saved unconditional table => {uncond_csv}")

                model.train()

            # Checkpoint saving
            if cfg.training.save_model_interval is not None and (step % cfg.training.save_model_interval == 0):
                ckpt_path = os.path.join(cfg.training.model_save_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                }, ckpt_path)
                print(f"Saved checkpoint => {ckpt_path}")

    # Final checkpoint
    if cfg.training.save_model_interval is not None:
        final_ckpt = os.path.join(cfg.training.model_save_dir, f"checkpoint_step_{step}_final.pt")
        torch.save({
            'step': step,
            'epoch': cfg.training.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': last_total_loss,
        }, final_ckpt)
        print(f"Saved FINAL checkpoint => {final_ckpt}")

    print("Training complete!")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Entry point. Hydra will automatically load configs/config.yaml,
    parse them into 'cfg', and switch to an output directory (unless disabled).
    """
    print("Full config:\n", OmegaConf.to_yaml(cfg))

    # 1) DataLoader
    # Instead of mocking, pass the config items directly
    train_loader = load_training_data(cfg)

    # 2) Load VAE
    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=cfg.training.device,
        dtype_str=cfg.vae.dtype
    )

    # 3) Build your DiT
    dit_model = MultiModalDiT(
        input_size=cfg.data.image_size,
        patch_size=4,
        in_channels=4,         # for SD latents
        dim=256,
        depth=16,
        head_dim=32,
        # etc. Adjust your actual constructor...
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174
    )

    # 4) Create MultiModalDiffusion
    mm_diff_model = MultiModalDiffusion(
        dit=dit_model,
        vae=vae,
        sigma_min=cfg.diffusion.sigma_min,
        sigma_max=cfg.diffusion.sigma_max,
        p_mean=cfg.diffusion.p_mean,
        p_std=cfg.diffusion.p_std,
        sigma_data=cfg.diffusion.sigma_data,
        num_steps=cfg.diffusion.num_steps,
        train_mask_ratio=cfg.diffusion.train_mask_ratio
    )

    # 5) Train
    train_multimodal_diffusion(mm_diff_model, train_loader, cfg)


if __name__ == "__main__":
    main()