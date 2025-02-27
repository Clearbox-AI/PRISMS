import torch
import hydra
from omegaconf import DictConfig
import os
import json
import numpy as np
from tqdm import tqdm

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


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    # Adjustable user params
    # ------------------------------------------------------------------
    checkpoint_path = "/mnt/storage/nacc_sub/mm_dit_con_vae_800/checkpoints/checkpoint_step_24000_final.pt"
    dataset_output_path = "/mnt/dataset_storage/data/nacc_dataset"
    batch_size = 4
    device = "cuda"

    # Set the starting index for patient folders (e.g., start from 50)
    starting_patient_index = 1

    # ------------------------------------------------------------------
    # 1) Build or load the same architecture as used in training
    # ------------------------------------------------------------------
    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=device,
        dtype_str=cfg.vae.dtype
    )
    vae.requires_grad_(False)
    vae.eval()

    dit_model = MultiModalDiT(
        input_size=64,
        patch_size=4,
        in_channels=4,
        dim=256,
        depth=16,
        head_dim=32,
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174
    )

    mm_diff_model = MultiModalDiffusion(
        dit=dit_model,
        sigma_min=cfg.diffusion.sigma_min,
        sigma_max=cfg.diffusion.sigma_max,
        p_mean=cfg.diffusion.p_mean,
        p_std=cfg.diffusion.p_std,
        sigma_data=cfg.diffusion.sigma_data,
        num_steps=cfg.diffusion.num_steps,
        train_mask_ratio=cfg.diffusion.train_mask_ratio,
    )

    mm_diff_model.to(device)

    # ------------------------------------------------------------------
    # 2) Load DiT checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_sd = ckpt["model_state_dict"]
    sd = strip_ddp_prefix(raw_sd)
    mm_diff_model.load_state_dict(sd, strict=True)
    mm_diff_model.eval()

    # -------------------------------------------------------------------------------
    # 4) Get "original" latents from your training loader, for reference
    # -------------------------------------------------------------------------------
    from diffusion_process.dataloaders import load_training_data
    train_loader = load_training_data(cfg)

    # Create the main output folder if it doesn't exist
    output_dir = os.path.join(dataset_output_path, "nacc_dataset_ditvae_adapter")
    os.makedirs(output_dir, exist_ok=True)

    # Determine the total number of patients if available
    total_patients = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else None

    # Global patient counter, starting from the defined index
    global_patient_idx = starting_patient_index - 1

    # Create a tqdm progress bar that counts individual patients
    pbar = tqdm(total=total_patients, desc="Patients Processed", unit="patient")

    for batch_idx, batch in enumerate(train_loader):
        loaded_images = batch['image'].to(device, non_blocking=True)
        # Safely get the number of patients in this batch
        current_batch_size = batch['tabular'].shape[0]
        condition_tab = batch['tabular'][:current_batch_size].to(device, non_blocking=True)
        with torch.no_grad():
            encoded_original = vae.encode(loaded_images)
            latents_original = encoded_original.latent_dist.sample() * vae.config.scaling_factor

        # ------------------------------------------------------------------
        # 5) Sample from the DiT model -> latents
        # ------------------------------------------------------------------
        with torch.no_grad():
            dit_latents, dit_tabulars = mm_diff_model.sample(
                batch_size=current_batch_size,
                table_data=condition_tab,
                cfg=1.0,
                steps=None,
                height=64,
                width=64,
                device=device,
                save_path=None
            )

        # ------------------------------------------------------------------
        # 6) Save patient data: for each patient in the batch, create a folder and store:
        #     - condition_tab (JSON)
        #     - dit_tabulars (JSON)
        #     - latents_dit (.npy)
        #     - latents_original (.npy)
        # ------------------------------------------------------------------
        for i in range(current_batch_size):
            global_patient_idx += 1
            patient_folder = os.path.join(output_dir, f"patient_{global_patient_idx}")
            os.makedirs(patient_folder, exist_ok=True)

            # Convert tensors to Python lists for JSON dumping
            condition_tab_list = condition_tab[i].cpu().numpy().tolist()
            dit_tabular_list = dit_tabulars[i].cpu().numpy().tolist()

            # Save JSON files
            with open(os.path.join(patient_folder, "condition_tab.json"), "w") as f:
                json.dump(condition_tab_list, f)
            with open(os.path.join(patient_folder, "dit_tabular.json"), "w") as f:
                json.dump(dit_tabular_list, f)

            # Save decoded images as .npy files
            np.save(os.path.join(patient_folder, "latents_dit.npy"), dit_latents[i].cpu().numpy())
            np.save(os.path.join(patient_folder, "latents_original.npy"), latents_original[i].cpu().numpy())

            # Update progress bar for each patient processed
            pbar.update(1)

        # Optionally, break after one batch if you want to test with a single iteration
        # break

    pbar.close()


if __name__ == "__main__":
    main()
