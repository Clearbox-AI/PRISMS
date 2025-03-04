
from models.dit.dit_multimodal import load_dit
from models.diffusion.diffusion_multimodal import load_diffusion

import os
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from models.utils.model_loader import load_model
from data.loader import load_training_data
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from utils.ddp import is_main_process, ddp_sample
from utils.model import save_checkpoint, resume_from_checkpoint
from utils.data import save_images, save_tabulars
from models.vae.vae import  decode_latents
from torch import optim

def main():
    """
        Simple test driver for MultiModalDiffusion with Hydra-based config.
        """

    device = torch.device(f"cuda:{0}")

    vae = load_model(model_type=ModelType.VAE).to(device)
    vae.requires_grad_(False)
    vae.eval()  # Typically we keep the VAE frozen

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        # 2) Load training data
        train_loader = load_training_data(cfg)

    from utils.configurations import set_project_root
    set_project_root()



    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        dit_cfg = compose(config_name="dit")
        dit_model = load_dit(dit_cfg)
        diffusion_cfg = compose(config_name="diffusion")

        mm_diff_model = load_model(
            model_type=ModelType.DIFFUSION,
            model_variant=DiTTrainingVersion.base_dit_training
        ).to(device)

        # optimizer = optim.AdamW(diffusion_model.parameters(), lr=1.5e-4)

        # 7) Optionally resume training from checkpoint
        start_epoch = 0
        global_step = 0
        resume_from_checkpoint(
                resume_dir="/mnt/storage/nacc_sub/dit/2025-03-03_07-45-21_256_dim128_head64_joint50",
                model=mm_diff_model,
                optimizer=None,
                device=device,
                use_ddp=False
            )

    for batch_idx, batch in enumerate(train_loader):

        # 1) Get data
        # images = batch['image'].to(device, non_blocking=True)
        # tab_data = batch['tabular'].to(device, non_blocking=True)

        mm_diff_model.eval()
        with torch.no_grad():
            # Sample latents (conditional on tab_data)
            # sub_tab = tab_data[:4].to(device)
            sampled_latents, cond_tab_out = ddp_sample(
                model=mm_diff_model,
                batch_size=cfg.training.sample_batch_size,
                table_data=None,
                cfg=cfg.training.sample_conditioning,
                steps=cfg.training.sample_steps,
                height=cfg.data.image_size,
                width=cfg.data.image_size,
                device=device,
                save_path=cfg.training.sample_latents_path
            )
            # Now decode latents -> images
            decoded_imgs = decode_latents(vae, sampled_latents, 0.13025)
            save_images("/mnt/storage/nacc_sub/dit/2025-03-03_07-45-21_256_dim128_head64_joint50", decoded_imgs, "aaaaaaaaaaaaaaa")

            if cond_tab_out is not None:
                save_tabulars("/mnt/storage/nacc_sub/dit/2025-03-03_07-45-21_256_dim128_head64_joint50", cond_tab_out, "aaaaaaaaaaaaaaa")

if __name__ == "__main__":
    main()