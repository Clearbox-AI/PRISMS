
import json
import os
import numpy as np
import torch

from typing import Optional, Tuple, Any, Dict, List
from omegaconf import DictConfig
from pathlib import Path
from hydra import compose, initialize_config_dir

from utils.data import DataBucket, infinite_loader, DataLabel
from models.dit.dit_multimodal import load_dit
from models.vae.vae import decode_latents, load_vae
from data.loader import load_training_data
from models.diffusion.diffusion_multimodal import load_diffusion


def _make_patient_folder_name(key: str) -> str:
    """
    Utility to convert a condition key into a folder name:
      - If `key` is purely numeric (e.g. '4'), create folder 'patient_4'.
      - Otherwise, take the basename of `key` (e.g. '/some/path/sub-NACC474680' -> 'sub-NACC474680').
    """
    if key.isdigit():
        return f"patient_{key}"
    return os.path.basename(key)

def save_generated_data(
    result_bucket,
    cond_mapping: Dict[str, List[int]],
    input_bucket,
    save_path: str
) -> None:
    """
    Saves generated images (from `result_bucket`) and their corresponding
    input tabular data (from `input_bucket`) on disk based on `cond_mapping`.
    Each sample is saved in its own pair of files: tab_data_{i}.json and img_data_{i}.npy.

    Args:
        result_bucket: DataBucket containing generated images (DataLabel.IMAGE),
                       typically a list (or tensor) of Tensors, each shaped [C,H,W].
        cond_mapping: Dict mapping condition_key -> list of generated sample indices.
        input_bucket: DataBucket containing the tabular data (DataLabel.TAB)
                      that was actually used for generation.
        save_path: Root directory where data will be saved. If it doesn't exist,
                   this function will create it.
    """
    os.makedirs(save_path, exist_ok=True)

    # We'll assume `result_bucket.data_source` and `input_bucket.data_source` are lists
    # (or similar) of length = total #samples. The i-th item in each corresponds
    # to sample i.
    all_images = result_bucket.data_source
    all_tab_data = input_bucket.data_source

    # For each condition key and all its associated sample indices
    for key, indices in cond_mapping.items():
        folder_name = f"patient_{str(key)}" if str(key).isdigit() else os.path.basename(str(key))
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # For each sample index for this condition key, save tab_data_{i}.json and img_data_{i}.npy
        for i, idx in enumerate(indices, start=1):
            # 1) Save the tabular row to tab_data_{i}.json
            tab_data_path = os.path.join(folder_path, f"tab_data_{i}.json")

            # Grab the i-th row for that key (assuming torch.Tensor or np.array)
            tab_entry = all_tab_data[idx]
            if isinstance(tab_entry, torch.Tensor):
                tab_list = tab_entry.detach().cpu().numpy().tolist()
            elif isinstance(tab_entry, np.ndarray):
                tab_list = tab_entry.tolist()
            else:
                # If it's already a Python list or some other structure
                tab_list = tab_entry

            with open(tab_data_path, "w") as f:
                json.dump(tab_list, f, indent=2)

            # 2) Save the corresponding generated image to img_data_{i}.npy
            img_npy_path = os.path.join(folder_path, f"img_data_{i}.npy")
            image_entry = all_images[idx]
            if isinstance(image_entry, torch.Tensor):
                image_np = image_entry.detach().cpu().numpy()
            else:
                image_np = image_entry
            np.save(img_npy_path, image_np)

if __name__ == "__main__":
    from utils.configurations import set_project_root
    set_project_root()

    # 1) Load config(s)
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        dit_cfg = compose(config_name="base_dit_training")
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        diffusion_cfg = compose(config_name="diffusion")

    # 2) Load the underlying DiT model
    dit_model = load_dit(dit_cfg)

    # 3) Build the diffusion model
    diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

    # 4) Load VAE
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        vae_cfg = compose(config_name="vae")
    vae = load_vae(vae_cfg)
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cuda")

    # 5) Load train dataloader
    train_dataloader = load_training_data(dit_cfg)

    # --------------------------
    # EXAMPLE 1: Using a DataLoader
    # --------------------------
    tab_bucket_dataloader = DataBucket(data_source=train_dataloader, label=DataLabel.TAB)

    # Generate 5 images from this DataLoader
    gen_bucket_dl, cond_map_dl, source_data_bucket = diffusion_model.generate_samples(
        n_samples=9,
        data_bucket=tab_bucket_dataloader,
        vae=vae,
        batch_size=2,
        cfg=1.5,
        steps=6,
        device='cuda'
    )

    save_generated_data(
        result_bucket=gen_bucket_dl,
        cond_mapping=cond_map_dl,
        input_bucket=source_data_bucket,
        save_path="/mnt/dataset_storage/data/nacc_dataset/generated_data"
    )