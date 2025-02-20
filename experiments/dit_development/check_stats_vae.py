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
from diffusion.dataloaders import load_training_data
from models.utils.model_loader import load_model
from enums.models.model_types import ModelType



@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="vae_vs_dit_dist")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------------------
    # 1) Build or load the same architecture as used in training
    # ------------------------------------------------------------------------------
    vae = load_model(ModelType.VAE, model_alias="blabla")

    vae = load_stable_diffusion_xl_vae()
    vae.requires_grad_(False)
    vae.eval()

    dit_model = MultiModalDiT()

    mm_diff_model = MultiModalDiffusion(
        dit=dit_model,
        sigma_min=0.002,        # placeholders—match your training
        sigma_max=20,
        p_mean=-0.6,
        p_std=1.2,
        sigma_data=0.9,
        num_steps=18,
        train_mask_ratio=0.0
    )

    # Move to device
    mm_diff_model.to(device)

    # ------------------------------------------------------------------------------
    # 2) Load checkpoint
    # ------------------------------------------------------------------------------

    checkpoint_path = cfg.execution_params.dit_checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    # load from a ddp model need some precaution
    raw_sd = ckpt["model_state_dict"]
    sd = strip_ddp_prefix(raw_sd)
    mm_diff_model.load_state_dict(sd, strict=True)

    mm_diff_model.eval()

    # ------------------------------------------------------------------------------
    # 3) Build a conditioning row if requested
    # ------------------------------------------------------------------------------
    if condition:
        # Suppose your table has 174 features, just as an example
        # Create a random normal row for each sample
        condition_tab = torch.randn(batch_size, 174, device=device)
        print("Using a random tabular row for conditioning.")
    else:
        condition_tab = None
        print("No conditioning (unconditional).")

    # ------------------------------------------------------------------------------
    # 4) Sample from the model. We want latents & the final images.
    #    -> This requires you to modify your .sample() method to return latents
    #       or we do an alternative approach. We'll assume you have a param "return_latents=True".
    # ------------------------------------------------------------------------------

    num_sample_batches = 100  # Match num_stat_batches for fair comparison
    num_stat_batches = 100  # Adjust based on available memory/data
    all_generated_latents = []

    with torch.no_grad():
        for _ in range(num_sample_batches):
            latents, _ = mm_diff_model.sample(
                batch_size=batch_size,
                table_data=condition_tab,
                cfg=1.0,
                steps=None,
                height=64,
                width=64,
                device=device,
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
        if batch_idx >= num_stat_batches:
            break
        loaded_images = batch['image'].to(device)
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
