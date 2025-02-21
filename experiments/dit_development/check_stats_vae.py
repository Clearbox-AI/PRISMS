# import utils.project_setup
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import hydra
from omegaconf import DictConfig, OmegaConf

from diffusion.multimodal_diffusion_ddp import MultiModalDiffusion
from models.dit.dit_multimodal import MultiModalDiT
from data.loader import load_training_data
from utils.ddp import strip_ddp_prefix

from models.utils.model_loader import load_model
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion



@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="vae_vs_dit_dist")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------------------
    # 1) Build or load the same architecture as used in training
    # ------------------------------------------------------------------------------
    vae = load_model(model_type=ModelType.VAE)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device=cfg.execution_params.device)

    mm_diff_model = load_model(model_type=ModelType.DIFFUSION, model_variant=DiTTrainingVersion.base_dit_training)
    mm_diff_model.to(cfg.execution_params.device)

    # dit_model = load_model(model_type=ModelType.DIT, model_variant=DiTTrainingVersion.base_dit_training)
    dit_model = mm_diff_model.dit
    dit_model.to(cfg.execution_params.device)

    # ------------------------------------------------------------------------------
    # 2) Load checkpoint
    # ------------------------------------------------------------------------------

    checkpoint_path = cfg.execution_params.dit_checkpoint
    ckpt = torch.load(checkpoint_path, map_location=cfg.execution_params.device)

    raw_sd = ckpt["model_state_dict"]
    mm_diff_model.load_state_dict(raw_sd, strict=True)

    mm_diff_model.eval()

    num_batches = cfg.execution_params.num_batches
    all_generated_latents = []

    with torch.no_grad():
        for _ in range(num_batches):
            latents, _ = mm_diff_model.sample(
                batch_size=cfg.execution_params.batch_size,
                table_data=torch.randn(cfg.execution_params.batch_size, 174, device=cfg.execution_params.device),
                cfg=1.0,
                steps=None,
                height=64,
                width=64,
                device=cfg.execution_params.device,
                save_path=None
            )
            all_generated_latents.append(latents)

    latents = torch.cat(all_generated_latents, dim=0)  # Accumulate batches

    # Calculate statistics for generated latents
    generated_latents_mean = torch.mean(latents)
    generated_latents_var = torch.var(latents)
    generated_mean_per_channel = torch.mean(latents, dim=(0, 2, 3))
    generated_var_per_channel = torch.var(latents, dim=(0, 2, 3))

    # -------------------------------------------------------------------------------
    # 2) Get original latents
    # -------------------------------------------------------------------------------
    # Collect multiple batches for better statistics
    all_original_latents = []

    train_loader = load_training_data(cfg)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        loaded_images = batch['image'].to(cfg.execution_params.device)
        with torch.no_grad():
            encoded_original = vae.encode(loaded_images)
            latents_batch = encoded_original.latent_dist.sample() * vae.config.scaling_factor
        all_original_latents.append(latents_batch)

    latents_original = torch.cat(all_original_latents, dim=0)

    # Calculate statistics for original latents
    original_latents_mean = torch.mean(latents_original)
    original_latents_var = torch.var(latents_original)
    original_mean_per_channel = torch.mean(latents_original, dim=(0, 2, 3))
    original_var_per_channel = torch.var(latents_original, dim=(0, 2, 3))

    # Print statistics comparison
    print("\n=== Latent Statistics Comparison ===")
    print(
        f"Generated Global Mean: {generated_latents_mean.item():.4f}, Original Global Mean: {original_latents_mean.item():.4f}")
    print(
        f"Generated Global Variance: {generated_latents_var.item():.4f}, Original Global Variance: {original_latents_var.item():.4f}\n")

    print("Generated Per-Channel Means:", generated_mean_per_channel.cpu().numpy().round(4))
    print("Original Per-Channel Means:", original_mean_per_channel.cpu().numpy().round(4))
    print("\nGenerated Per-Channel Variances:", generated_var_per_channel.cpu().numpy().round(4))
    print("Original Per-Channel Variances:", original_var_per_channel.cpu().numpy().round(4))

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(latents_original.flatten().cpu().numpy(),
             bins=200, alpha=0.5, density=True, label='Original')
    plt.hist(latents.flatten().cpu().numpy(),
             bins=200, alpha=0.5, density=True, label='Generated')
    plt.title("Latent Value Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    hist_path = os.path.join("/mnt/storage/nacc_sub/mm_dit_con_vae/tmp", "latent_distributions.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"\nSaved distribution comparison at {hist_path}")

if __name__ == "__main__":
    main()
