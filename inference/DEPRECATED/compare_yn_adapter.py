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
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae


def is_main_process():
    """
    Utility to check if current process is the global rank 0.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def strip_ddp_prefix(state_dict, keyword):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(f"{keyword}."):
            new_k = k[len(f"{keyword}."):]
        elif k.startswith(f"{keyword}."):
            new_k = k[len(f"{keyword}."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def get_edm_noised_latents(mm_diff_model, latents, tab_data):
    """
    Return D_xn = c_skip * (latents+noise*sigma) + c_out * F_x,
    i.e. the 'predicted latents' from the diffusion model,
    given real latents + noise.  (One forward pass, no iterative sampler.)
    """
    device = latents.device
    B = latents.shape[0]

    # Sample sigma from lognormal
    rnd_normal = torch.randn([B,1,1,1], device=device)
    sigma = (rnd_normal * mm_diff_model.p_std + mm_diff_model.p_mean).exp()

    # Add noise
    noise = torch.randn_like(latents)
    noised_input = latents + sigma * noise

    # c_in, c_skip, c_out
    c_in = 1.0 / (mm_diff_model.sigma_data**2 + sigma**2).sqrt()
    c_skip = mm_diff_model.sigma_data**2 / (sigma**2 + mm_diff_model.sigma_data**2)
    c_out = sigma * mm_diff_model.sigma_data / (sigma**2 + mm_diff_model.sigma_data**2).sqrt()

    # Time embedding
    t_embed = (sigma.log() / 4.0).reshape(-1)

    # Forward the DiT
    out = mm_diff_model.dit(
        x_img=c_in * noised_input,
        t=t_embed,
        tab=tab_data,
        cfg=1.0,  # no classifier-free guidance here
        mask_ratio=mm_diff_model.train_mask_ratio
    )
    F_x = out['image_sample']

    # Predicted latents
    D_xn = c_skip * noised_input + c_out * F_x  # shape [B,4,H,W]
    return D_xn


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------
    # Adjustable user params
    # ------------------------------------------------------------------
    checkpoint_path = "/mnt/storage/nacc_sub/mm_dit_con_vae_800/checkpoints/checkpoint_step_24000_final.pt"
    condition = False
    batch_size = 4
    device = "cuda"

    # If you want to turn on the adapter pipeline:
    use_adapter = getattr(cfg.exp, "use_adapter", False)

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
    sd = strip_ddp_prefix(raw_sd, "module")
    mm_diff_model.load_state_dict(sd, strict=True)
    mm_diff_model.eval()

    # ------------------------------------------------------------------
    # 3) (Optional) Load the adapter
    # ------------------------------------------------------------------
    if use_adapter:
        # from models.latents.vae_dit_adapter import VaeDitAdapterUNet
        from models.latents.vae_dit_adapter import MicroDiT_Tiny_2
        from diffusion.multimodal_diffusion_ddp_adapter import LatentsDiffusion

        adapter = MicroDiT_Tiny_2()

        mm_diff_adapter = LatentsDiffusion(
            dit=adapter,
            sigma_min=cfg.diffusion.sigma_min,
            sigma_max=cfg.diffusion.sigma_max,
            p_mean=cfg.diffusion.p_mean,
            p_std=cfg.diffusion.p_std,
            sigma_data=cfg.diffusion.sigma_data,
            num_steps=cfg.diffusion.num_steps,
            train_mask_ratio=cfg.diffusion.train_mask_ratio,
        )

        mm_diff_adapter.to(device)

        # adapter = VaeDitAdapterUNet(base_ch=128, attn_heads=4).to(device)

        # If you have a checkpoint for the adapter:
        if hasattr(cfg.adapter, "checkpoint"):
            adapter_ckpt_path = cfg.adapter.checkpoint
            adapter_ckpt = torch.load(adapter_ckpt_path, map_location=device)
            raw_sd = adapter_ckpt["model_state_dict"]
            sd = strip_ddp_prefix(raw_sd, "module")
            mm_diff_adapter.load_state_dict(sd, strict=True)
            print(f"[INFO] Loaded adapter from: {adapter_ckpt_path}")
            mm_diff_adapter.eval()
        else:
            print("[WARN] use_adapter=True but no adapter checkpoint specified. Using random adapter weights.")
    else:
        adapter = None

    # -------------------------------------------------------------------------------
    # ) Get "original" latents from your training loader, for reference
    # -------------------------------------------------------------------------------
    from diffusion_process.dataloaders import load_training_data
    train_loader = load_training_data(cfg)

    for batch_idx, batch in enumerate(train_loader):
        loaded_images = batch['image'].to(device, non_blocking=True)
        condition_tab = batch['tabular'][:batch_size].to(device, non_blocking=True)
        with torch.no_grad():
            encoded_original = vae.encode(loaded_images)
            latents_original = encoded_original.latent_dist.sample() * vae.config.scaling_factor
        # Only process one batch for the comparison
        break

    # ------------------------------------------------------------------
    # 5) Sample from the DiT model -> latents
    # ------------------------------------------------------------------
    with torch.no_grad():
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

        # Decode the latents directly from DiT
        decoded_imgs = vae.decode(latents / vae.scaling_factor).sample
        final_images = (decoded_imgs * 0.5 + 0.5).clamp(0, 1)

        # If using adapter, pass latents through it and decode
        if use_adapter:
            # latents_adapter, _, _, _ = adapter(latents)

            latents_adapter = mm_diff_adapter.sample(
                batch_size=4,
                y=latents,
                steps=cfg.diffusion.num_steps,
                height=cfg.data.image_size,
                width=cfg.data.image_size,
                device=device
            )

            decoded_imgs_adapter = vae.decode(latents_adapter / vae.scaling_factor).sample
            final_images_adapter = (decoded_imgs_adapter * 0.5 + 0.5).clamp(0, 1)

            # ------------------------------------------------------------------
            #  Add the "no_iter" column: Re‐noise the DiT latents & pass to adapter
            # ------------------------------------------------------------------
            D_xn = get_edm_noised_latents(mm_diff_model, latents, condition_tab)
            # latents_no_iter, _, _, _ = adapter(D_xn)
            latents_no_iter = mm_diff_adapter.sample(
                batch_size=4,
                y=D_xn,
                steps=cfg.diffusion.num_steps,
                height=cfg.data.image_size,
                width=cfg.data.image_size,
                device=device
            )
            decoded_imgs_no_iter = vae.decode(latents_no_iter / vae.scaling_factor).sample
            final_images_no_iter = (decoded_imgs_no_iter * 0.5 + 0.5).clamp(0, 1)

        else:
            latents_adapter = None
            final_images_adapter = None
            latents_no_iter = None
            final_images_no_iter = None



    # Decode the original latents
    decoded_original = vae.decode(latents_original / vae.scaling_factor).sample
    final_original = (decoded_original * 0.5 + 0.5).clamp(0, 1)

    # -------------------------------------------------------------------------------
    # 7) Visualization: Compare latents and decoded images
    #    (Generated vs. Original vs. (Optional) Adapter vs. (Optional) No_iter)
    # -------------------------------------------------------------------------------
    out_dir = "/mnt/storage/nacc_sub/mm_dit_con_vae/tmp"
    os.makedirs(out_dir, exist_ok=True)

    # We only have batch_size latents from DiT, and batch_size from the loader.
    # We'll iterate up to the smaller of those two if necessary
    n_to_show = min(batch_size, latents_original.size(0))

    for i in range(n_to_show):
        #  - latents[i], latents_original[i], latents_adapter[i], latents_no_iter[i] ...
        #  - final_images[i], final_original[i], final_images_adapter[i], final_images_no_iter[i] ...

        # ~~~~ Latent arrays (CPU, Numpy) ~~~~
        latent_gen = latents[i].detach().cpu().numpy()  # shape [4, H, W]
        latent_org = latents_original[i].detach().cpu().numpy()
        if use_adapter:
            latent_adapt = latents_adapter[i].detach().cpu().numpy()
            latent_no_iter_ = latents_no_iter[i].detach().cpu().numpy()

        # ~~~~ Decoded images -> [H, W, 3] ~~~~
        # Generated
        image_gen = final_images[i].detach().cpu().numpy()
        if image_gen.shape[0] == 3:
            image_gen = np.transpose(image_gen, (1, 2, 0))
        image_gen = np.clip(image_gen, 0, 1)

        # Original
        image_org = final_original[i].detach().cpu().numpy()
        if image_org.shape[0] == 3:
            image_org = np.transpose(image_org, (1, 2, 0))
        image_org = np.clip(image_org, 0, 1)

        if use_adapter:
            # Adapter
            image_adapt = final_images_adapter[i].detach().cpu().numpy()
            if image_adapt.shape[0] == 3:
                image_adapt = np.transpose(image_adapt, (1, 2, 0))
            image_adapt = np.clip(image_adapt, 0, 1)

            # No_iter
            image_no_iter = final_images_no_iter[i].detach().cpu().numpy()
            if image_no_iter.shape[0] == 3:
                image_no_iter = np.transpose(image_no_iter, (1, 2, 0))
            image_no_iter = np.clip(image_no_iter, 0, 1)

        # ------------------------------------------------------------------
        # Setup subplots
        # 4 channels + 1 row for decoded images -> total 5 rows
        #
        # If adapter is off => 2 columns (Gen / Org).
        # If adapter is on  => 4 columns (Gen / Org / Adapter / No_iter).
        # ------------------------------------------------------------------
        if use_adapter:
            ncols = 4
        else:
            ncols = 2
        nrows = 5  # 4 channels + 1 row for the decoded images

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

        for c in range(4):
            axs[c, 0].imshow(latent_gen[c], cmap="gray")
            axs[c, 0].set_title(f"Gen latent (ch{c})")
            axs[c, 0].axis("off")

            axs[c, 1].imshow(latent_org[c], cmap="gray")
            axs[c, 1].set_title(f"Org latent (ch{c})")
            axs[c, 1].axis("off")

            if use_adapter:
                axs[c, 2].imshow(latent_adapt[c], cmap="gray")
                axs[c, 2].set_title(f"Adapt latent (ch{c})")
                axs[c, 2].axis("off")

                axs[c, 3].imshow(latent_no_iter_[c], cmap="gray")
                axs[c, 3].set_title(f"No_iter latent (ch{c})")
                axs[c, 3].axis("off")

        # Last row => Decoded images
        axs[4, 0].imshow(image_gen)
        axs[4, 0].set_title("Decoded Gen Image")
        axs[4, 0].axis("off")

        axs[4, 1].imshow(image_org)
        axs[4, 1].set_title("Decoded Org Image")
        axs[4, 1].axis("off")

        if use_adapter:
            axs[4, 2].imshow(image_adapt)
            axs[4, 2].set_title("Decoded Adapt Image")
            axs[4, 2].axis("off")

            axs[4, 3].imshow(image_no_iter)
            axs[4, 3].set_title("Decoded No_iter Image")
            axs[4, 3].axis("off")

        # Adjust figure layout
        fig.suptitle(f"Sample {i} Comparison (Condition={condition}, Adapter={use_adapter})")
        plt.tight_layout()

        if is_main_process():
            outpath = os.path.join(out_dir, f"comparison_sample_{i}.png")
            plt.savefig(outpath)
            print(f"Saved {outpath}")

        plt.close(fig)

    # -------------------------------------------------------------------------------
    # 9) Optionally, save the tabular output
    # -------------------------------------------------------------------------------
    if tabular_out is not None:
        tab_out_np = tabular_out.cpu().numpy()  # shape: [batch_size, num_features]
        csv_path = os.path.join(out_dir, "sample_tab_out.csv")
        if is_main_process():
            np.savetxt(csv_path, tab_out_np, delimiter=",", header="Tabular Output", comments="")
            print(f"Saved generated tabular data => {csv_path}")

    if is_main_process():
        print("Sampling completed!")


if __name__ == "__main__":
    main()
