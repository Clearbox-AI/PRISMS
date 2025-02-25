import os
import random
import torch
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from torchvision.utils import save_image


# Reuse your utility functions from your main trainer file or copy them:
from trainers.multimodal_trainer_ddp import is_main_process, setup_distributed, cleanup_distributed
from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae
from diffusion.multimodal_diffusion_ddp import MultiModalDiffusion
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
from diffusion_process.dataloaders import load_training_data
# from models.latents.vae_dit_adapter import VaeDitAdapter
from models.latents.vae_dit_adapter import VaeDitAdapterUNet



def multi_scale_combined_loss(
    preds,    # list/tuple of predicted latents at multiple scales: [out_full, out_1_2, out_1_4, out_1_8]
    targets,  # real latents [B,4,H,W]
    alpha=0.5 # fixed weight for MSE vs. SmoothL1
):
    """
    Compute a mixture of MSE and SmoothL1 across multiple spatial scales.
    For each pred scale, we downsample targets to match, then:

      Loss_scale = alpha*MSE + (1-alpha)*SmoothL1

    Finally we average across all scales.
    """
    total_loss = 0.0
    n_scales = len(preds)

    for p in preds:
        B, C, h, w = p.shape
        # Downsample real latents to this scale
        tgt_down = F.interpolate(targets, size=(h, w), mode='area')
        mse = F.mse_loss(p, tgt_down)
        sl1 = F.smooth_l1_loss(p, tgt_down, beta=1.0)  # typical beta=1
        loss_scale = alpha * mse + (1.0 - alpha) * sl1
        total_loss += loss_scale

    return total_loss / n_scales


def save_samples(global_step, vae, latents, z_hat, cfg):
    """
    Decodes real/predicted latents and saves images to disk,
    including an additional grayscale version using only the first channel.
    """
    if is_main_process() and (global_step % cfg.adapter.sample_interval == 0):

        # Decode latents
        with torch.no_grad():
            real_img = vae.decode(latents / cfg.vae.scaling_factor).sample  # [B, C, H, W]
            pred_img = vae.decode(z_hat / cfg.vae.scaling_factor).sample  # [B, C, H, W]

        # Normalize images
        real_img = (real_img * 0.5 + 0.5).clamp(0, 1)  # [B, C, H, W]
        pred_img = (pred_img * 0.5 + 0.5).clamp(0, 1)

        # Extract only the first channel (assuming it's meaningful as grayscale)
        real_gray = real_img[:, 0:1, :, :]  # Keep dimensions [B, 1, H, W]
        pred_gray = pred_img[:, 0:1, :, :]  # [B, 1, H, W]

        # Select first N samples
        N = min(4, real_img.size(0))  # e.g., 4
        grid_rgb = torch.cat([real_img[:N], pred_img[:N]], dim=0)  # [2N, C, H, W]
        grid_gray = torch.cat([real_gray[:N], pred_gray[:N]], dim=0)  # [2N, 1, H, W]

        out_dir = cfg.adapter.sample_save_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path_rgb = os.path.join(cfg.adapter.sample_save_dir, f"samples_step_{global_step}.png")
        out_path_gray = os.path.join(cfg.adapter.sample_save_dir, f"samples_grey_step_{global_step}.png")

        # Save RGB samples
        save_image(grid_rgb, out_path_rgb, nrow=N, normalize=True, value_range=(0.0, 1.0))
        print(f"[Sample Saved] {out_path_rgb}")

        # Save grayscale samples (first channel only)
        save_image(grid_gray, out_path_gray, nrow=N, normalize=True, value_range=(0.0, 1.0))
        print(f"[Grayscale Sample Saved] {out_path_gray}")

def get_adapter_alpha(adapter, ddp_enabled, use_sigmoid=True):
    """Call adapter.get_alpha or adapter.module.get_alpha if wrapped in DDP."""
    if ddp_enabled:
        return adapter.module.get_alpha(use_sigmoid=use_sigmoid)
    else:
        return adapter.get_alpha(use_sigmoid=use_sigmoid)

def strip_ddp_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def load_frozen_diffusion(cfg, device):
    """
    Load your already-trained DiT model from a checkpoint, and freeze it.
    """
    # Create the DiT model architecture
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

    # Wrap it in MultiModalDiffusion
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

    ckpt = torch.load(cfg.adapter.use_tis_dit, map_location=device)
    raw_sd = ckpt["model_state_dict"]
    sd = strip_ddp_prefix(raw_sd)
    mm_diff_model.load_state_dict(sd, strict=True)

    # Freeze
    mm_diff_model.eval()
    mm_diff_model.requires_grad_(False)

    return mm_diff_model

def get_edm_noised_latents(mm_diff_model, latents, tab_data):
    """
    Return D_xn = c_skip * (latents+noise*sigma) + c_out * F_x,
    i.e. the 'predicted latents' from the diffusion model,
    given real latents + noise.
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
    D_xn = c_skip * noised_input + c_out * F_x  # shape [B,4,64,64]
    return D_xn


def train_adapter_one_epoch(
    epoch,
    adapter,
    adapter_opt,
    mm_diff_model,
    vae,
    train_loader,
    cfg,
    device,
    global_step=0
):
    adapter.train()
    mm_diff_model.eval()

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        latents_dit = batch['latents_dit'].to(device, non_blocking=True)
        latents_original = batch['latents_original'].to(device, non_blocking=True)

        # 3) Adapter forward pass: map D_xn -> z_hat
        out_full, out_1_2, out_1_4, out_1_8 = adapter(latents_dit)

        preds = [out_full, out_1_2, out_1_4, out_1_8]
        loss = multi_scale_combined_loss(preds, latents_original, alpha=0.75)

        adapter_opt.zero_grad()
        loss.backward()
        adapter_opt.step()

        # Optional logging
        if is_main_process() and (global_step % cfg.training.log_interval == 0):
            msg = (
                f"[Epoch {epoch + 1} | Step {global_step}] "
                f"Loss={loss.item():.6f}, "
            )
            print(msg)

        # save samples
        save_samples(
                global_step=global_step,
                vae=vae,
                latents=latents_original,  # real latents
                z_hat=out_full,  # predicted latents
                cfg=cfg
            )

    return global_step


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if is_main_process():
        print("Adapter Training Config:\n", OmegaConf.to_yaml(cfg))

    # 1) Setup DDP
    local_rank = 0
    if cfg.distributed.use_ddp:
        local_rank = setup_distributed(cfg)
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # 2) Load data
    train_loader = load_training_data(cfg)

    # 3) Load SD-XL VAE on the same device if you must encode on the fly
    #    (Alternatively, skip this if you have precomputed latents.)
    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=device,  # same device, no second GPU required
        dtype_str=cfg.vae.dtype
    )
    vae.eval()
    vae.requires_grad_(False)

    # 4) Load your already-trained (and frozen) diffusion model
    mm_diff_model = load_frozen_diffusion(cfg, device)
    mm_diff_model.requires_grad_(False)

    # 5) Build the adapter
    # adapter = VaeDitAdapter(
    #     in_channels=cfg.adapter.in_channels,
    #     hidden_dim=cfg.adapter.hidden_dim,
    #     num_blocks=cfg.adapter.num_blocks
    # ).to(device)

    # 5) Build the adapter
    adapter = VaeDitAdapterUNet(base_ch=128, attn_heads=4).to(device)

    # # Create the projector for PatchNCE
    # projector = PatchProjector(
    #     in_channels=cfg.adapter.in_channels,  # 4 if latents are 4 channels
    #     embed_dim=cfg.adapter.nce_embed_dim,  # e.g. 128
    #     patch_size=cfg.adapter.nce_patch_size  # e.g. 8
    # ).to(device)
    #
    # # Create the PatchNCE criterion
    # patchnce_criterion = PatchNCELoss(
    #     projector=projector,
    #     patch_size=cfg.adapter.nce_patch_size,
    #     num_patches=cfg.adapter.nce_num_patches,  # e.g. 64
    #     temperature=cfg.adapter.nce_temperature  # e.g. 0.07
    # ).to(device)

    # If we want a learnable alpha, set it:
    # if cfg.adapter.learnable_weight:
    #     adapter.set_learnable_weight(True)

    # 6) Optimizer
    adapter_opt = optim.AdamW(adapter.parameters(), lr=cfg.adapter.lr)

    # 7) DDP
    if cfg.distributed.use_ddp:
        adapter = DDP(adapter, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 8) Training loop
    global_step = 0
    for epoch in range(cfg.adapter.epochs):
        if cfg.distributed.use_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        global_step = train_adapter_one_epoch(
            epoch=epoch,
            adapter=adapter,
            adapter_opt=adapter_opt,
            mm_diff_model=mm_diff_model,
            vae=vae,
            train_loader=train_loader,
            cfg=cfg,
            device=device,
            # patchnce_criterion=patchnce_criterion,
            global_step=global_step
        )

        # Optionally: checkpoint the adapter
        if (cfg.adapter.save_model_interval is not None
            and global_step % cfg.adapter.save_model_interval == 0
            and is_main_process()):
            ckpt_path = os.path.join(cfg.adapter.model_save_dir, f"adapter_step_{global_step}.pt")
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'adapter_state_dict': adapter.module.state_dict() if isinstance(adapter, DDP) else adapter.state_dict(),
                'optimizer_state_dict': adapter_opt.state_dict(),
            }, ckpt_path)
            print(f"Saved adapter checkpoint => {ckpt_path}")

    # Final checkpoint
    if is_main_process():
        final_ckpt = os.path.join(cfg.adapter.model_save_dir, f"adapter_step_{global_step}_final.pt")
        torch.save({
            'step': global_step,
            'epoch': cfg.adapter.epochs,
            'adapter_state_dict': adapter.module.state_dict() if isinstance(adapter, DDP) else adapter.state_dict(),
            'optimizer_state_dict': adapter_opt.state_dict(),
        }, final_ckpt)
        print(f"Saved FINAL adapter checkpoint => {final_ckpt}")

    if cfg.distributed.use_ddp:
        cleanup_distributed()

    if is_main_process():
        print("Adapter training complete!")


if __name__ == "__main__":
    main()
