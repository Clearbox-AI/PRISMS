# trainers/multimodal_trainer_ddp.py

import os
import torch
import random
import pandas as pd
from torch import optim
from torchvision.utils import save_image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import DictConfig, OmegaConf

# Import your modules
from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae
from diffusion.multimodal_diffusion_ddp import MultiModalDiffusion
from multi_modal_diffusion.model.dit_mm import MultiModalDiT
from diffusion_process.dataloaders import load_training_data


def setup_distributed(cfg: DictConfig):
    """
    Initialize the torch.distributed process group for DDP.
    We read the environment variables set by torchrun or similar.
    """
    # Often set in environment:
    #   MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE
    # Also ensure you have: backend=nccl or gloo
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method="env://"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process():
    """
    Utility to check if current process is the global rank 0.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def cleanup_distributed():
    """
    Destroy the process group for DDP.
    """
    dist.destroy_process_group()


def ddp_sample(model, *args, **kwargs):
    """Calls 'sample' on the underlying model if wrapped in DDP."""
    if isinstance(model, DDP):
        return model.module.sample(*args, **kwargs)
    else:
        return model.sample(*args, **kwargs)


def train_one_epoch(epoch, model, optimizer, train_loader, cfg, vae, local_rank, global_step=0 ):
    """
    One epoch of training in distributed mode.
    Returns the last global_step and last total_loss for checkpointing.
    """

    model.train()
    device = torch.device(f"cuda:{local_rank}")
    last_total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        # 1) Get data
        images = batch['image'].to(device, non_blocking=True)  # shape: [B, 3, H, W]
        tab_data = batch['tabular'].to(device, non_blocking=True)

        # Possibly drop tab => unconditional
        if random.random() < cfg.training.uncond_prob:
            tab_data = None

        # 2) Encode images -> latents (outside the diffusion model)
        with torch.no_grad():
            latents_dist = vae.encode(images)
            latents = latents_dist.latent_dist.sample() * vae.config.scaling_factor

        # 3) Forward and loss
        total_loss, image_loss, tab_loss = model(latents, tab_data, global_step)
        last_total_loss = total_loss.item()

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # Logging (only rank-0 prints)
        if is_main_process() and (global_step % cfg.training.log_interval == 0):
            msg = (f"[Epoch {epoch + 1} | Step {global_step}] "
                   f"Img Loss: {image_loss.item():.4f}")
            if tab_loss is not None:
                msg += f" | Tab Loss: {tab_loss.item():.4f}"
            msg += f" | Total: {last_total_loss:.4f}"
            if tab_data is None:
                msg += " [UNCOND]"
            print(msg)

        # 5) Sampling (only rank-0)
        if is_main_process() and (global_step % cfg.training.sample_interval == 0):
            model.eval()
            with torch.no_grad():
                # (A) Conditional sample latents
                sub_tab = batch['tabular'][:4].to(device) if tab_data is not None else None
                sampled_latents, cond_tab_out = ddp_sample(
                    model=model,
                    batch_size=4,
                    table_data=sub_tab,
                    cfg=1.0,
                    steps=None,
                    height=cfg.data.image_size,  # latents resolution
                    width=cfg.data.image_size,
                    device=device,
                    save_path=None
                )
                # Now decode latents -> images
                decoded_imgs = vae.decode(sampled_latents / vae.scaling_factor).sample
                decoded_imgs = (decoded_imgs * 0.5 + 0.5).clamp(0, 1)

                cond_file = os.path.join(cfg.training.sample_save_dir, f"samples_step_{global_step}_cond.png")
                save_image(decoded_imgs, cond_file, nrow=2)
                print(f"Saved conditional images => {cond_file}")

                if cond_tab_out is not None:
                    cond_csv = os.path.join(cfg.training.sample_save_dir, f"tabular_step_{global_step}_cond.csv")
                    pd.DataFrame(cond_tab_out.cpu().numpy()).to_csv(cond_csv, index=False)
                    print(f"Saved conditional table => {cond_csv}")

                # (B) Unconditional sample latents
                uncond_latents, uncond_tab_out = ddp_sample(
                    model=model,
                    batch_size=4,
                    table_data=None,
                    cfg=1.0,
                    steps=None,
                    height=cfg.data.image_size,
                    width=cfg.data.image_size,
                    device=device,
                    save_path=None
                )
                uncond_imgs = vae.decode(uncond_latents / vae.scaling_factor).sample
                uncond_imgs = (uncond_imgs * 0.5 + 0.5).clamp(0, 1)

                uncond_file = os.path.join(cfg.training.sample_save_dir, f"samples_step_{global_step}_uncond.png")
                save_image(uncond_imgs, uncond_file, nrow=2)
                print(f"Saved unconditional images => {uncond_file}")

                if uncond_tab_out is not None:
                    uncond_csv = os.path.join(cfg.training.sample_save_dir, f"tabular_step_{global_step}_uncond.csv")
                    pd.DataFrame(uncond_tab_out.cpu().numpy()).to_csv(uncond_csv, index=False)
                    print(f"Saved unconditional table => {uncond_csv}")

            model.train()

        # 6) Save model checkpoints (only rank-0)
        if (cfg.training.save_model_interval is not None and
                global_step % cfg.training.save_model_interval == 0 and
                is_main_process()):
            ckpt_path = os.path.join(cfg.training.model_save_dir, f"checkpoint_step_{global_step}.pt")
            torch.save({
                'step': global_step,
                'epoch': epoch,
                # If wrapped in DDP, need .module to get actual underlying model
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, ckpt_path)
            print(f"Saved checkpoint => {ckpt_path}")

    return global_step, last_total_loss


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    print("Full config:\n", OmegaConf.to_yaml(cfg))

    # 0) Setup output dirs on rank=0 only
    if is_main_process():
        os.makedirs(cfg.training.sample_save_dir, exist_ok=True)
        os.makedirs(cfg.training.model_save_dir, exist_ok=True)

    # 1) Initialize DDP
    local_rank = 0
    if cfg.distributed.use_ddp:
        local_rank = setup_distributed(cfg)
        torch.cuda.set_device(local_rank)

    # 2) Load Data (with distributed sampler if needed)
    train_loader = load_training_data(cfg)  # It should return a DataLoader that uses DistributedSampler if world_size>1

    # 3) Load or create VAE (kept on GPU but *not* wrapped in DDP if frozen)
    device = torch.device(f"cuda:{local_rank}")
    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=device,
        dtype_str=cfg.vae.dtype
    )
    vae.requires_grad_(False)  # Usually we don't train the SD-VAE
    vae.eval()

    # 4) Build your DiT model
    dit_model = MultiModalDiT(
        input_size=cfg.data.image_size,
        patch_size=4,
        in_channels=4,  # for SD latents
        dim=256,
        depth=16,
        head_dim=32,
        num_tab_columns=174,
        tab_groups=10,
        out_table_features=174
        # etc.
    )

    # 5) Create MultiModalDiffusion
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

    # Move model to GPU
    mm_diff_model.to(device)

    # 6) Wrap in DDP (if desired)
    if cfg.distributed.use_ddp:
        mm_diff_model = DDP(mm_diff_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 7) Create optimizer
    optimizer = optim.AdamW(mm_diff_model.parameters(), lr=cfg.training.lr)

    # Optionally: load from existing checkpoint if resuming

    start_epoch = 0
    global_step = 0

    # 8) Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        # If using a DistributedSampler, set epoch for shuffling
        if (cfg.distributed.use_ddp and hasattr(train_loader.sampler, 'set_epoch')):
            train_loader.sampler.set_epoch(epoch)

        global_step, last_loss = train_one_epoch(
            epoch, mm_diff_model, optimizer, train_loader,
            cfg, vae, local_rank, global_step=global_step
        )

    # 9) Final checkpoint (only rank-0)
    if is_main_process() and cfg.training.save_model_interval is not None:
        final_ckpt = os.path.join(cfg.training.model_save_dir, f"checkpoint_step_{global_step}_final.pt")
        torch.save({
            'step': global_step,
            'epoch': cfg.training.epochs,
            'model_state_dict': mm_diff_model.module.state_dict() if cfg.distributed.use_ddp else mm_diff_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': last_loss,
        }, final_ckpt)
        print(f"Saved FINAL checkpoint => {final_ckpt}")

    # 10) Cleanup
    if cfg.distributed.use_ddp:
        cleanup_distributed()

    if is_main_process():
        print("Training complete!")

if __name__ == "__main__":
    main()