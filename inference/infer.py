import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import hydra
from omegaconf import DictConfig, OmegaConf

from diffusion.multimodal_diffusion_ddp import MultiModalDiffusion
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae


def strip_ddp_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    checkpoint_path = "/mnt/storage/nacc_sub/mm_dit_con_vae/checkpoints/checkpoint_step_18000_final_ddp.pt"
    condition = False
    batch_size = 4
    device = "cuda"

    # ------------------------------------------------------------------------------
    # 1) Build or load the same architecture as used in training
    # ------------------------------------------------------------------------------
    # Example: Adjust these hyperparameters to match your training config

    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=device,
        dtype_str=cfg.vae.dtype
    )
    vae.requires_grad_(False)  # Usually we don't train the SD-VAE
    vae.eval()

    dit_model = MultiModalDiT(
        input_size=64,          # match your train config
        patch_size=4,
        in_channels=4,          # for SD latents
        dim=256,
        depth=16,
        head_dim=32,
        num_tab_columns=174,    # example, match your train config
        tab_groups=10,
        out_table_features=174
    )

    mm_diff_model = MultiModalDiffusion(
        dit=dit_model,
        sigma_min=0.002,        # placeholders—match your training
        sigma_max=80,
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


    with torch.no_grad():
        # Adjust the sample() call as needed for your implementation
        latents, tabular_out = mm_diff_model.sample(
            batch_size=batch_size,
            table_data=condition_tab,
            cfg=1.0,
            steps=None,
            height=64,
            width=64,
            device=device,
            save_path=None
        )

    # Decode the generated latents
    decoded_imgs = vae.decode(latents / vae.scaling_factor).sample
    final_images = (decoded_imgs * 0.5 + 0.5).clamp(0, 1)

    # -------------------------------------------------------------------------------
    # 2) Get "original" latents by encoding a batch from the training loader, and decode them
    # -------------------------------------------------------------------------------
    from diffusion_process.dataloaders import load_training_data

    train_loader = load_training_data(cfg)
    for batch_idx, batch in enumerate(train_loader):
        loaded_images = batch['image'].to(device, non_blocking=True)  # shape: [B, 3, H, W]
        # Encode the images to get the original latents
        with torch.no_grad():
            encoded_original = vae.encode(loaded_images)
            latents_original = encoded_original.latent_dist.sample() * vae.config.scaling_factor
        # Process only one batch
        break

    # Decode the original latents
    decoded_original = vae.decode(latents_original / vae.scaling_factor).sample
    final_original = (decoded_original * 0.5 + 0.5).clamp(0, 1)

    # -------------------------------------------------------------------------------
    # 3) Visualization: Compare latents and decoded images (generated vs. original)
    # -------------------------------------------------------------------------------
    os.makedirs("/mnt/storage/nacc_sub/mm_dit_con_vae/tmp", exist_ok=True)

    for i in range(batch_size):
        # -----------------------------
        # Extract the latent representations:
        # -----------------------------
        # Both latents and latents_original have shape [C, H, W] (here C==4)
        latent_gen = latents[i].detach().cpu().numpy()
        latent_org = latents_original[i].detach().cpu().numpy()
        # For visualization, select channel 0 from each latent (feel free to change the channel if desired)
        latent_gen_ch0 = latent_gen[0]
        latent_org_ch0 = latent_org[0]

        # -----------------------------
        # Extract the decoded images:
        # -----------------------------
        image_gen = final_images[i].detach().cpu().numpy()  # e.g., shape [3, H, W]
        image_org = final_original[i].detach().cpu().numpy()  # e.g., shape [3, H, W]
        # If images have 3 channels, transpose to [H, W, C] for matplotlib
        if image_gen.shape[0] == 3:
            image_gen = np.transpose(image_gen, (1, 2, 0))
        if image_org.shape[0] == 3:
            image_org = np.transpose(image_org, (1, 2, 0))
        # Ensure the pixel values are in [0,1]
        image_gen = np.clip(image_gen, 0, 1)
        image_org = np.clip(image_org, 0, 1)

        # -----------------------------
        # Create a 2x2 subplot for comparisons:
        # -----------------------------
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Top row: Latents (visualizing only channel 0)
        axs[0, 0].imshow(latent_gen_ch0, cmap="gray")
        axs[0, 0].set_title("Generated Latent (ch0)")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(latent_org_ch0, cmap="gray")
        axs[0, 1].set_title("Original Latent (ch0)")
        axs[0, 1].axis("off")

        # Bottom row: Decoded Images
        axs[1, 0].imshow(image_gen, cmap=None if (image_gen.ndim == 3 and image_gen.shape[2] == 3) else "gray")
        axs[1, 0].set_title("Decoded Generated Image")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(image_org, cmap=None if (image_org.ndim == 3 and image_org.shape[2] == 3) else "gray")
        axs[1, 1].set_title("Decoded Original Image")
        axs[1, 1].axis("off")

        # Optionally, include a super-title with additional info (e.g., the condition)
        fig.suptitle(f"Sample {i} Comparison (Condition={condition})")
        plt.tight_layout()

        # Save the comparison figure
        outpath = os.path.join("/mnt/storage/nacc_sub/mm_dit_con_vae/tmp", f"comparison_sample_{i}.png")
        plt.savefig(outpath)
        plt.close(fig)
        print(f"Saved {outpath}")

    # -------------------------------------------------------------------------------
    # 4) Optionally, save the tabular output if it exists
    # -------------------------------------------------------------------------------
    if tabular_out is not None:
        tab_out_np = tabular_out.cpu().numpy()  # shape: [batch_size, num_features]
        csv_path = os.path.join("/mnt/storage/nacc_sub/mm_dit_con_vae/tmp", "sample_tab_out.csv")
        np.savetxt(csv_path, tab_out_np, delimiter=",", header="Tabular Output", comments="")
        print(f"Saved generated tabular data => {csv_path}")

    print("Sampling completed!")

if __name__ == "__main__":
    main()
