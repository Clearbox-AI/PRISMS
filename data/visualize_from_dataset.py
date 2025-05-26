"""
viz_nacc.py – show 50 real + 50 synthetic NACC images as 5×5 montages

The script relies on `load_training_data`, so it respects the exact
normalisation / augmentation parameters defined in your Hydra configs.
"""

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.configurations import set_project_root
from data.loader import load_training_data


# ------------------------------------------------------------------ #
# 1. Hydra → DataLoaders                                             #
# ------------------------------------------------------------------ #
set_project_root()                              # ensures $PROJECT_ROOT

CFG_DIR = Path(os.environ["PROJECT_ROOT"], "configs", "datasets")


def build_loader(cfg_name: str, *, real: bool) -> DataLoader:
    """
    Return a *deterministic* 1-sample-batch loader for the dataset
    described by <cfg_name>.yaml.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name=cfg_name)
    OmegaConf.set_struct(cfg, False)            # allow mutation

    base_loader = load_training_data(cfg, real=real)
    # Re-wrap the underlying dataset so we can iterate sequentially.
    return DataLoader(
        dataset=base_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


ld_nacc  = build_loader("nacc",       real=True)
ld_synth = build_loader("nacc_synth", real=False)

# ------------------------------------------------------------------ #
# 2. utilities                                                        #
# ------------------------------------------------------------------ #
def first_k_images(loader: DataLoader, k: int) -> List[torch.Tensor]:
    """Fetch the first *k* image tensors from *loader* (shape C,H,W)."""
    out = []
    for batch in loader:
        out.append(batch["image"].squeeze(0))   # drop batch dim
        if len(out) == k:
            break
    return out


def make_montage(tensors, nrow=5, ncol=5, pad=2, pad_val=0):
    """
    Convert *nrow*×*ncol* tensors (C,H,W) or (H,W) into a single image.
    """
    assert len(tensors) == nrow * ncol
    imgs = []
    for t in tensors:
        a = t.cpu().numpy()
        if a.ndim == 3 and a.shape[0] == 1:         # (1,H,W) → (H,W)
            a = a[0]
        elif a.ndim == 3:                           # (C,H,W) → (H,W,C)
            a = np.transpose(a, (1, 2, 0))
        imgs.append(a)

    h, w = imgs[0].shape[:2]
    canvas = np.full(
        (nrow * h + pad * (nrow - 1),
         ncol * w + pad * (ncol - 1)) +
        (() if imgs[0].ndim == 2 else (imgs[0].shape[2],)),
        pad_val,
        dtype=imgs[0].dtype
    )

    for idx, img in enumerate(imgs):
        r, c = divmod(idx, ncol)
        top  = r * (h + pad)
        left = c * (w + pad)
        canvas[top:top + h, left:left + w] = img
    return canvas


def show_montages(imgs: List[torch.Tensor], title_prefix: str):
    """Display 50 images as two 25-image figures (one chart each)."""
    for i in range(0, 50, 25):
        montage = make_montage(imgs[i:i + 25])
        plt.figure(figsize=(10, 10))
        if montage.ndim == 2:
            plt.imshow(montage, cmap="gray")
        else:
            plt.imshow(montage)
        plt.axis("off")
        plt.title(f"{title_prefix}: images {i+1}-{i+25}")
        plt.show(block=True)


# ------------------------------------------------------------------ #
# 3. run                                                              #
# ------------------------------------------------------------------ #
imgs_nacc  = first_k_images(ld_nacc,  50)
imgs_synth = first_k_images(ld_synth, 50)

show_montages(imgs_nacc,  "NaccDataset")
show_montages(imgs_synth, "NaccSynthDataset")
