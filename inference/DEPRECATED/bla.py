# import argparse
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch import optim
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch.distributed as dist
#
# from diffusion.multimodal_diffusion_ddp import MultiModalDiffusion
# from multi_modal_diffusion.model.dit_mm import MultiModalDiT
# from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae
#
#
# def is_main_process():
#     """
#     Utility to check if current process is the global rank 0.
#     """
#     return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
#
# def strip_ddp_prefix(state_dict, keyword):
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith(f"{keyword}."):
#             new_k = k[len(f"{keyword}."):]
#         elif k.startswith(f"{keyword}."):
#             new_k = k[len(f"{keyword}."):]
#         else:
#             new_k = k
#         new_state_dict[new_k] = v
#     return new_state_dict
#
#
# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def main(cfg: DictConfig):
#
#     # ------------------------------------------------------------------
#     # Adjustable user params
#     # ------------------------------------------------------------------
#     checkpoint_path = "/mnt/storage/nacc_sub/mm_dit_con_vae_800/checkpoints/checkpoint_step_24000_final.pt"
#     condition = False
#     batch_size = 4
#     device = "cuda"
#
#     vae = load_stable_diffusion_xl_vae(
#         model_name=cfg.vae.model_name,
#         subfolder=cfg.vae.subfolder,
#         device=device,
#         dtype_str=cfg.vae.dtype
#     )
#     vae.requires_grad_(False)
#     vae.eval()
#
#     dit_model = MultiModalDiT(
#         input_size=cfg.data.image_size,
#         patch_size=4,
#         in_channels=4,  # for SD latents
#         dim=256,
#         depth=16,
#         head_dim=32,
#         num_tab_columns=174,
#         tab_groups=10,
#         out_table_features=174
#     )
#
#     mm_diff_model = MultiModalDiffusion(
#         dit=dit_model,
#         sigma_min=cfg.diffusion.sigma_min,
#         sigma_max=cfg.diffusion.sigma_max,
#         p_mean=cfg.diffusion.p_mean,
#         p_std=cfg.diffusion.p_std,
#         sigma_data=cfg.diffusion.sigma_data,
#         num_steps=cfg.diffusion.num_steps,
#         train_mask_ratio=cfg.diffusion.train_mask_ratio,
#     )
#
#     mm_diff_model.to(device)
#
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     raw_sd = ckpt["model_state_dict"]
#     sd = strip_ddp_prefix(raw_sd, "module")
#     mm_diff_model.load_state_dict(sd, strict=True)
#     mm_diff_model.eval()
#
#     from models.latents.vae_dit_adapter import MicroDiT_Tiny_2
#     from diffusion.multimodal_diffusion_ddp_adapter import LatentsDiffusion
#
#     adapter = MicroDiT_Tiny_2()
#
#     mm_diff_adapter = LatentsDiffusion(
#         dit=adapter,
#         sigma_min=cfg.diffusion.sigma_min,
#         sigma_max=cfg.diffusion.sigma_max,
#         p_mean=cfg.diffusion.p_mean,
#         p_std=cfg.diffusion.p_std,
#         sigma_data=cfg.diffusion.sigma_data,
#         num_steps=cfg.diffusion.num_steps,
#         train_mask_ratio=cfg.diffusion.train_mask_ratio,
#     )
#
#     mm_diff_adapter.to(device)
#
#     adapter_ckpt_path = "/mnt/storage/nacc_sub/mm_dit_con_vae/checkpoints/checkpoint_step_15000_final.pt"
#     adapter_ckpt = torch.load(adapter_ckpt_path, map_location=device)
#     raw_sd = adapter_ckpt["model_state_dict"]
#     sd = strip_ddp_prefix(raw_sd, "module")
#     mm_diff_adapter.load_state_dict(sd, strict=True)
#     print(f"[INFO] Loaded adapter from: {adapter_ckpt_path}")
#     mm_diff_adapter.eval()
#
#     from diffusion_process.dataloaders import load_training_data
#     train_loader = load_training_data(cfg)
#     for batch_idx, batch in enumerate(train_loader):
#
#         # 1) Get data
#         images = batch['image'].to(device, non_blocking=True)  # shape: [B, 3, H, W]
#         tab_data = batch['tabular'].to(device, non_blocking=True)
#
#         with torch.no_grad():
#             latents_dist = vae.encode(images)
#             latents_input = latents_dist.latent_dist.sample()
#
#             latents_dit, _ = mm_diff_model.sample(
#                 batch_size=4,
#                 table_data=tab_data,
#                 cfg=1.0,
#                 steps=None,
#                 height=cfg.data.image_size,  # latents resolution
#                 width=cfg.data.image_size,
#                 device=device,
#                 save_path=None
#             )
#             # Now decode latents -> images
#             # decoded_imgs = vae.decode(sampled_latents / cfg.vae.scaling_factor).sample
#             # decoded_imgs = (decoded_imgs * 0.5 + 0.5).clamp(0, 1)
#
#             latents_adapter = mm_diff_adapter.sample(
#                 batch_size=4,
#                 y=latents_dit,
#                 steps=cfg.diffusion.num_steps,
#                 height=cfg.data.image_size,
#                 width=cfg.data.image_size,
#                 device=device
#             )
#
#             decoded_latents_adapter = vae.decode(latents_adapter / vae.scaling_factor).sample
#             image = (decoded_latents_adapter * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy()
#             from torchvision.utils import save_image
#             image_tensor = torch.from_numpy(image)  # Convert NumPy array to PyTorch tensor
#             if image_tensor.ndim == 4:  # Ensure proper shape: [B, C, H, W]
#                 save_image(image_tensor, "/mnt/storage/nacc_sub/mm_dit_con_vae/tmp/bla.png", nrow=4, normalize=False)
#             else:
#                 raise ValueError(f"Expected 4D tensor, but got shape {image_tensor.shape}")
#         break
#
# if __name__ == "__main__":
#     main()



import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist

from diffusion.multimodal_diffusion_ddp import MultiModalDiffusion
from models.dit.dit_multimodal import MultiModalDiT
from models.vae.loader import load_stable_diffusion_xl_vae
from diffusion.dataloaders import load_training_data




@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------
    # Adjustable user params
    # ------------------------------------------------------------------
    checkpoint_path = "/mnt/storage/nacc_sub/mm_dit_con_vae/checkpoints/checkpoint_step_24000_final.pt"
    condition = False
    batch_size = 4
    device = "cuda"

    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=device,
        dtype_str=cfg.vae.dtype
    )
    vae.requires_grad_(False)
    vae.eval()

    dit_model = MultiModalDiT(
        input_size=cfg.data.image_size,
        patch_size=4,
        in_channels=4,  # for SD latents
        dim=512,
        depth=20,
        head_dim=32,
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174,
        multiple_of = 256,
        patch_mixer_depth=4,
        patch_mixer_dim=512,
        patch_mixer_qkv_ratio=1.0,
        patch_mixer_mlp_ratio=4.0,
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
        latent_reg_weight=cfg.training.latent_reg_weight
    )

    mm_diff_model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_sd = ckpt["model_state_dict"]
    sd = strip_ddp_prefix(raw_sd, "module")
    mm_diff_model.load_state_dict(sd, strict=True)
    mm_diff_model.eval()

    # from models.latents.vae_dit_adapter import MicroDiT_Tiny_2
    # from diffusion.multimodal_diffusion_ddp_adapter import LatentsDiffusion

    # adapter = MicroDiT_Tiny_2()
    #
    # mm_diff_adapter = LatentsDiffusion(
    #     dit=adapter,
    #     sigma_min=cfg.diffusion.sigma_min,
    #     sigma_max=cfg.diffusion.sigma_max,
    #     p_mean=cfg.diffusion.p_mean,
    #     p_std=cfg.diffusion.p_std,
    #     sigma_data=cfg.diffusion.sigma_data,
    #     num_steps=cfg.diffusion.num_steps,
    #     train_mask_ratio=cfg.diffusion.train_mask_ratio,
    # )

    # mm_diff_adapter.to(device)
    #
    # adapter_ckpt_path = "/mnt/storage/nacc_sub/mm_dit_con_vae/checkpoints/checkpoint_step_15000_final.pt"
    # adapter_ckpt = torch.load(adapter_ckpt_path, map_location=device)
    # raw_sd = adapter_ckpt["model_state_dict"]
    # sd = strip_ddp_prefix(raw_sd, "module")
    # mm_diff_adapter.load_state_dict(sd, strict=True)
    # print(f"[INFO] Loaded adapter from: {adapter_ckpt_path}")
    # mm_diff_adapter.eval()


    train_loader = load_training_data(cfg)
    for batch_idx, batch in enumerate(train_loader):

        # 1) Get data
        # src_latents = batch['latents_dit'].to(device, non_blocking=True)
        # tgt_latents = batch['latents_original'].to(device, non_blocking=True)
        images = batch['image'].to(device, non_blocking=True)  # shape: [B, 3, H, W]
        tab_data = batch['tabular'].to(device, non_blocking=True)

        with torch.no_grad():
            latents_dist = vae.encode(images)
            latents_input = latents_dist.latent_dist.sample()

            latents_dit, _ = mm_diff_model.sample(
                batch_size=4,
                table_data=tab_data[:4],
                cfg=1.0,
                steps=None,
                height=cfg.data.image_size,
                width=cfg.data.image_size,
                device=device,
                save_path=None
            )

            # latents_dit2, _ = mm_diff_model.sample(
            #     batch_size=4,
            #     table_data=tab_data,
            #     cfg=1.0,
            #     steps=None,
            #     height=cfg.data.image_size,  # latents resolution
            #     width=cfg.data.image_size,
            #     device=device,
            #     save_path=None
            # )

            # Given values
            # generated_means = torch.tensor([-2.4726, 0.4372, 0.0516, 0.3728]).to(device)
            # original_means = torch.tensor([-2.4637, 0.4328, 0.0259, 0.4130]).to(device)
            #
            # generated_variances = torch.tensor([0.5976, 0.0633, 0.1799, 0.0914]).to(device)
            # original_variances = torch.tensor([0.6803, 0.0937, 0.2500, 0.1329]).to(device)
            #
            # # Compute the mean and variance adjustments
            # delta_mean = generated_means - original_means
            # delta_variance = generated_variances - original_variances
            #
            # # Reshape to (1, C, 1, 1) for broadcasting
            # delta_mean = delta_mean.view(1, -1, 1, 1)
            # delta_variance = delta_variance.view(1, -1, 1, 1)
            #
            # # Apply correction
            # adjusted_tensor = latents_dit + delta_mean  # Shift mean
            # adjusted_tensor = adjusted_tensor * (1 + delta_variance)

            # latents_adapter = mm_diff_adapter.sample(
            #     batch_size=4,
            #     y=src_latents,
            #     steps=cfg.diffusion.num_steps,
            #     height=cfg.data.image_size,
            #     width=cfg.data.image_size,
            #     device=device
            # )
            #
            decoded_latents_adapter = vae.decode(latents_dit / vae.scaling_factor).sample
            image = (decoded_latents_adapter * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy()
            from torchvision.utils import save_image
            image_tensor = torch.from_numpy(image)  # Convert NumPy array to PyTorch tensor
            save_image(image_tensor, "/mnt/storage/nacc_sub/mm_dit_con_vae/tmp/bla.png", nrow=4, normalize=False)

            # decoded_latents_adapter = vae.decode(adjusted_tensor / vae.scaling_factor).sample
            # image = (decoded_latents_adapter * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy()
            # from torchvision.utils import save_image
            # image_tensor = torch.from_numpy(image)  # Convert NumPy array to PyTorch tensor
            # save_image(image_tensor, "/mnt/storage/nacc_sub/mm_dit_con_vae/tmp/blabla.png", nrow=4, normalize=False)


        break

if __name__ == "__main__":
    main()