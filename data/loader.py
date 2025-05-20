import os
import json
import numpy as np
import torch.distributed as dist
import torch
import glob

from torch.utils.data import DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler
from typing import Union
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from data.nacc_dataset import NaccDataset
from enums.data import DatasetType, ImageRange
from torch.utils.data import Subset
import random


def load_training_data(cfg: DictConfig) -> DataLoader:
    """
    Create a DataLoader for your dataset.
    1) Possibly compute stats if needed.
    2) Initialize NaccDataset with those stats.
    3) Return DataLoader with (optionally) a DistributedSampler.
    """

    if cfg.data.dataset_type.lower() == DatasetType.NACC.value:
        # Step 1: Check if we need to precompute stats
        data_dir = cfg.data.data_dir
        stats_dir = cfg.data.get("stats_file") or Path(Path(__file__).resolve().parent, "computations")
        stats_path = Path(stats_dir, "nacc_stats.json")
        compute_dataset_stats(data_dir, stats_path)

        # Step 2: Create the dataset (the dataset will load stats from stats_path)
        dataset = NaccDataset(
            data_dir=data_dir,
            image_height=cfg.data.image_height,
            image_width=cfg.data.image_width,
            domain=cfg.data.domain,
            do_augment=cfg.data.do_augment,
            do_image_normalize=cfg.data.do_image_normalize,
            do_tabular_normalize=cfg.data.do_tabular_normalize,
            target_channels=cfg.data.target_channels,
            final_image_range=ImageRange(cfg.data.final_image_range) ,
            debug=cfg.data.debug,
            stats_file=stats_path
        )
    else:
        # to implement for other datasets
        raise NotImplementedError

    # if "random_subset" in cfg.data and cfg.data.random_subset is not None:
    #     full_indices = list(range(len(dataset)))
    #     random.shuffle(full_indices)
    #     chosen = full_indices[:cfg.data.random_subset]
    #     dataset = Subset(dataset, chosen)

    # Step 3: Build distributed sampler if needed
    world_size, rank = 1, 0
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        shuffle=sampler is None and cfg.data.shuffle
    )

    return loader


def compute_dataset_stats(data_dir: Union[Path, str], stats_path: Union[Path, str]):
    """
    Checks if 'stats_path' already exists.
    - If it does NOT exist, compute stats from 'data_dir' and write them to stats_path.
    - If it exists, do nothing (we assume it's already computed).
    """
    if os.path.exists(stats_path):
        print(f"[INFO] Stats file '{stats_path}' already exists; skipping computation.")
        return

    print(f"[INFO] Stats file '{stats_path}' not found. Computing stats...")

    patient_dirs = [
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    if not patient_dirs:
        raise ValueError(f"No patient directories found in {data_dir}")

    all_image_pixels = []
    tabular_rows = []
    tabular_scaler = StandardScaler()

    for pdir in patient_dirs:
        # 1) Load the first .npy file
        npy_files = glob.glob(os.path.join(pdir, "*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {pdir}")
        image = np.load(npy_files[0]).astype(np.float32)
        all_image_pixels.append(image.flatten())

        # 2) Load the first .json
        json_files = glob.glob(os.path.join(pdir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json file found in {pdir}")
        with open(json_files[0], 'r') as f:
            jdata = json.load(f)
        tab = jdata.get("patient_id", list(jdata.values())[0])
        tab = np.array(tab, dtype=np.float32)
        tab = np.where(tab > 9999, -1, tab)  # sentinel replacement
        tabular_rows.append(tab.reshape(1, -1))

    # Compute global image mean/std
    all_pixels = np.concatenate(all_image_pixels, axis=0)
    image_mean = float(all_pixels.mean())
    image_std = float(all_pixels.std(ddof=1))  # unbiased

    # Fit tabular scaler
    big_tab = np.concatenate(tabular_rows, axis=0)  # shape [N, feats]
    tabular_scaler.fit(big_tab)

    # Save to JSON
    stats_dict = {
        "image_mean": image_mean,
        "image_std": image_std,
        "tabular_scaler_mean_": tabular_scaler.mean_.tolist(),
        "tabular_scaler_scale_": tabular_scaler.scale_.tolist()
    }
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=4)

    print(f"[INFO] Stats file saved to {stats_path}")


if __name__ == "__main__":

    from utils.configurations import set_project_root
    set_project_root()

    def compute_stats(loader, max_batches=20):
        """
        Iterates through `max_batches` of the DataLoader, stores all images in a large tensor,
        and computes:
          - Global mean & variance
          - Per-channel mean & variance
        """
        latents = []

        for i, batch in enumerate(loader):
            images = batch["image"]  # Shape: [B, C, H, W]
            latents.append(images)

            if i + 1 >= max_batches:
                break  # Stop after `max_batches`

        # Stack all collected images into a single tensor
        latents = torch.cat(latents, dim=0)  # Shape: [Total_B, C, H, W]

        # Compute statistics
        global_mean = torch.mean(latents)
        global_var = torch.var(latents, unbiased=True)
        channel_mean = torch.mean(latents, dim=(0, 2, 3))  # Shape: [C]
        channel_var = torch.var(latents, dim=(0, 2, 3), unbiased=True)  # Shape: [C]

        return global_mean, global_var, channel_mean, channel_var


    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        print("[INFO] Loading DataLoader...")
        loader = load_training_data(cfg)

        # Compute statistics over 20 batches
        global_mean, global_var, channel_mean, channel_var = compute_stats(loader, max_batches=2000)

        print(f"[INFO] DataLoader Global Mean: {global_mean.item()}")
        print(f"[INFO] DataLoader Global Variance: {global_var.item()}")
        print(f"[INFO] DataLoader Per-Channel Mean: {channel_mean.tolist()}")
        print(f"[INFO] DataLoader Per-Channel Variance: {channel_var.tolist()}")