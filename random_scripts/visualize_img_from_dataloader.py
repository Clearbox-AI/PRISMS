import os
import random
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data.loader import load_training_data
from utils.configurations import set_project_root

def get_images_from_dataloader(loader, num_images=30):
    all_images = []

    for batch in loader:
        # If batch is a tuple (image, label) or dict, adjust as needed
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(images, dict) and 'image' in images:
            images = images['image']
        all_images.extend(images)

        if len(all_images) >= num_images:
            break

    return random.sample(all_images, min(num_images, len(all_images)))

def show_images(images, cols=6):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 2.5 * rows))
    for idx, img in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        img_np = img.detach().cpu().numpy()

        # Assume shape [C, H, W]
        if img_np.ndim == 3:
            img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC

        # Clamp to [0, 1] for safety
        img_np = img_np - img_np.min()
        img_np = img_np / (img_np.max() + 1e-5)

        plt.imshow(img_np.squeeze(), cmap='gray' if img_np.shape[-1] == 1 else None)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_project_root()
    config_dir = Path(os.environ["PROJECT_ROOT"], "configs", "datasets")

    with initialize_config_dir(config_dir=str(config_dir)):
        cfg = compose(config_name="nacc")
        OmegaConf.set_struct(cfg, False)

    train_loader = load_training_data(cfg)
    sampled_images = get_images_from_dataloader(train_loader, num_images=30)
    show_images(sampled_images)
