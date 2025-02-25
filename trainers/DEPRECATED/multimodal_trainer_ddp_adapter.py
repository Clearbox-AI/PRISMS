import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import hydra
from omegaconf import DictConfig, OmegaConf

# Import your latents diffusion
from diffusion.multimodal_diffusion_ddp_adapter import LatentsDiffusion
from models.latents.vae_dit_adapter import MicroDiT_Tiny_2
from diffusion_process.dataloaders import load_training_data
from models.latents.stability_ai.autoencoder import load_stable_diffusion_xl_vae


def is_main_process():
    """Returns True if this process is the global rank=0."""
    return (
            (not dist.is_available())
            or (not dist.is_initialized())
            or dist.get_rank() == 0
    )


def setup_distributed(cfg: DictConfig):
    """
    Initialize torch.distributed using environment variables:
        MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, WORLD_SIZE
    """
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method="env://"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Destroy the torch.distributed process group."""
    dist.destroy_process_group()


@torch.no_grad()
def save_samples(
        global_step: int,
        vae: nn.Module,
        latents_real: torch.Tensor,
        latents_pred: torch.Tensor,
        cfg: DictConfig,
):
    """
    Decode `latents_real` and `latents_pred` via VAE -> images.
    Save them side by side. 
    (We assume your VAE has a .decode(...) that outputs images in [-1,1] or [0,1].)
    """
    if not is_main_process():
        return

    if (global_step % cfg.training.sample_interval) != 0:
        return

    out_dir = cfg.training.sample_save_dir
    os.makedirs(out_dir, exist_ok=True)

    # latents -> images
    # If your VAE expects scaled latents, do something like:
    scaling_factor = getattr(cfg.vae, "scaling_factor", 1.0)
    latents_real_scaled = latents_real / scaling_factor # TODO: or multiply?
    latents_pred_scaled = latents_pred / scaling_factor

    real_img = vae.decode(latents_real_scaled).sample  # => shape [B, C, H, W]
    pred_img = vae.decode(latents_pred_scaled).sample  # => shape [B, C, H, W]

    # If the VAE output is in [-1,1], convert to [0,1]
    # Here, assume stable diffusion VAE => output ~ [-1,1]
    real_img = (real_img * 0.5 + 0.5).clamp(0, 1)
    pred_img = (pred_img * 0.5 + 0.5).clamp(0, 1)

    # Combine them in a grid: real top row, pred bottom row
    B = min(4, real_img.size(0))  # how many to show
    grid = torch.cat([real_img[:B], pred_img[:B]], dim=0)
    out_path = os.path.join(out_dir, f"samples_step_{global_step}.png")
    save_image(grid, out_path, nrow=B, normalize=False)
    print(f"[Rank=0] Saved latents decode => {out_path}")


def ddp_sample(model, *args, **kwargs):
    """
    Call model.sample(...) if wrapped in DDP.
    """
    if isinstance(model, DDP):
        return model.module.sample(*args, **kwargs)
    return model.sample(*args, **kwargs)


def train_one_epoch(
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader,
        vae: nn.Module,
        local_rank: int,
        cfg: DictConfig,
        global_step: int
):
    """
    One epoch of training for latents->latents diffusion. 
    Returns updated (global_step, last_loss).
    """
    device = torch.device(f"cuda:{local_rank}")
    model.train()

    last_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        # 1) Retrieve latents (source + target)
        src_latents = batch['latents_dit'].to(device, non_blocking=True)
        tgt_latents = batch['latents_original'].to(device, non_blocking=True)

        # 2) Forward pass
        total_loss = model(src_latents, tgt_latents, global_step=global_step)
        last_loss = total_loss.item()

        # 3) Backprop
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # 4) Logging
        if is_main_process() and (global_step % cfg.training.log_interval == 0):
            print(f"[Epoch={epoch + 1} Step={global_step}] Loss={total_loss.item():.4f}")

        # 5) Sampling (only rank-0)
        if is_main_process() and (global_step % cfg.training.sample_interval == 0):
            model.eval()
            with torch.no_grad():
                # a) We run unconditional sampling:
                #    e.g. model.sample(batch_size=4, steps=cfg.diffusion.num_steps, ...)
                latents_gen = ddp_sample(
                    model,
                    batch_size=4,
                    y=src_latents,
                    steps=cfg.diffusion.num_steps,
                    height=cfg.data.image_size,
                    width=cfg.data.image_size,
                    device=device
                )

            save_samples(
                global_step=global_step,
                vae=vae,
                latents_real=tgt_latents,  # real latents
                latents_pred=latents_gen,  # predicted latents
                cfg=cfg
            )

            model.train()

        # 6) Checkpoint
        if (
                cfg.training.save_model_interval is not None
                and global_step % cfg.training.save_model_interval == 0
                and is_main_process()
        ):
            ckpt_path = os.path.join(cfg.training.model_save_dir, f"checkpoint_step_{global_step}.pt")
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save({
                "step": global_step,
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss.item()
            }, ckpt_path)
            print(f"[Rank 0] Saved checkpoint => {ckpt_path}")

    return global_step, last_loss


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Example trainer for latents->latents diffusion with a 
    frozen VAE purely for decoding latents to images.
    """
    print("Full config:\n", OmegaConf.to_yaml(cfg))

    # 0) Setup output dirs on rank=0
    if is_main_process():
        os.makedirs(cfg.training.sample_save_dir, exist_ok=True)
        os.makedirs(cfg.training.model_save_dir, exist_ok=True)

    # 1) Possibly init DDP
    local_rank = 0
    if cfg.distributed.use_ddp:
        local_rank = setup_distributed(cfg)
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # 2) Dataloader (src_latents, tgt_latents) => in your code
    train_loader = load_training_data(cfg)

    # 3) Load your VAE if you want to decode for visualization
    #    We won't train the VAE => freeze it
    vae = load_stable_diffusion_xl_vae(
        model_name=cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        device=device,
        dtype_str=cfg.vae.dtype
    )
    vae.requires_grad_(False)
    vae.eval()

    # 4) Build DiT 
    dit_model = MicroDiT_Tiny_2()

    # 5) Wrap in latents->latents diffusion
    latents_diff_model = LatentsDiffusion(
        dit=dit_model,
        sigma_min=cfg.diffusion.sigma_min,
        sigma_max=cfg.diffusion.sigma_max,
        p_mean=cfg.diffusion.p_mean,
        p_std=cfg.diffusion.p_std,
        sigma_data=cfg.diffusion.sigma_data,
        num_steps=cfg.diffusion.num_steps,
        train_mask_ratio=cfg.diffusion.train_mask_ratio,
    )
    latents_diff_model.to(device)

    # 6) Wrap in DDP if needed
    if cfg.distributed.use_ddp:
        latents_diff_model = DDP(
            latents_diff_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    # 7) Create optimizer
    optimizer = optim.AdamW(latents_diff_model.parameters(), lr=cfg.training.lr)

    start_epoch = 0
    global_step = 0
    last_loss = 0.0

    # 8) Main training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        # If using DistributedSampler
        if (
                cfg.distributed.use_ddp
                and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)

        global_step, last_loss = train_one_epoch(
            epoch,
            latents_diff_model,
            optimizer,
            train_loader,
            vae,
            local_rank,
            cfg,
            global_step=global_step
        )

    # 9) Final checkpoint
    if is_main_process():
        final_ckpt = os.path.join(cfg.training.model_save_dir, f"checkpoint_step_{global_step}_final.pt")
        model_to_save = latents_diff_model.module if isinstance(latents_diff_model, DDP) else latents_diff_model
        torch.save({
            "step": global_step,
            "epoch": cfg.training.epochs,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": last_loss,
        }, final_ckpt)
        print(f"[Rank 0] Saved final checkpoint => {final_ckpt}")

    # 10) Cleanup
    if cfg.distributed.use_ddp:
        cleanup_distributed()

    if is_main_process():
        print("Training complete!")


if __name__ == "__main__":
    main()
