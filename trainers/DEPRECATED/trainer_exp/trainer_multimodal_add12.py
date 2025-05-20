import os
import torch
import random
import pandas as pd
from torch import optim
from torchvision.utils import save_image
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra import compose, initialize_config_dir
import glob

from omegaconf import DictConfig, OmegaConf

# Suppose these come from your code
from data.loader import load_training_data
from models.vae.vae import encode_images, decode_latents
from models.diffusion.diffusion_multimodal_add12 import load_diffusion, EMA
from utils.ddp import is_main_process, setup_distributed, cleanup_distributed
from utils.model12 import save_checkpoint, resume_from_checkpoint
from utils.path_management import setup_storage_directory
from utils.data import save_images, save_tabulars
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,      # This is your MultiModalDiffusion (DDP-wrapped if needed)
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    vae: torch.nn.Module,
    global_step: int = 0,
    base_save_path: Path = None,
    device: torch.device = None,
    ema_obj=None,
) -> (int, float):
    """
    Single epoch training. Now we call 'model(x_img, x_tab)' to compute the diffusion loss.
    """
    model.train()
    last_total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        # 1) Get data
        images = batch['image'].to(device, non_blocking=True)
        tab_data = batch['tabular'].to(device, non_blocking=True)

        # 2) Encode images -> latents
        latents = encode_images(vae, images, cfg.vae.scaling_factor)

        # 3) Forward pass with the diffusion model
        loss_total, loss_img, loss_tab = model(latents, tab_data)
        last_total_loss = loss_total.item()

        # 4) Backprop
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        # 5) EMA update
        if ema_obj is not None:
            ema_obj.update()

        # 6) Logging
        if is_main_process() and (global_step % cfg.training.log_interval == 0):
            msg = (f"[Epoch {epoch + 1} | Step {global_step}] "
                   f"Img Loss: {loss_img.item():.6f} | "
                   f"Tab Loss: {loss_tab.item():.6f} | "
                   f"Total: {last_total_loss:.6f}")
            print(msg)

        # 7) Sampling
        if is_main_process() and (global_step % cfg.training.sample_interval == 0):
            model.eval()
            with torch.no_grad():
                # We'll call a helper that unwraps the model if it's DDP
                x_img_samples, x_tab_samples = ddp_sample(
                    model=model,
                    ema_obj=ema_obj,
                    batch_size=cfg.training.sample_batch_size,
                    x_img_shape=(4, 32, 32),
                    x_tab_shape=(tab_data.shape[1],),
                    num_steps=cfg.training.sample_steps
                )
                # decode from latents
                recon_images = decode_latents(vae, x_img_samples, cfg.vae.scaling_factor)

                # save
                save_images(base_save_path, recon_images, global_step)
                save_tabulars(base_save_path, x_tab_samples, global_step)
            model.train()

        # 8) Save model checkpoints
        if (cfg.training.save_model_interval is not None
            and global_step % cfg.training.save_model_interval == 0
            and is_main_process()):
            save_checkpoint(
                ckpt_dir=base_save_path,
                ckpt_name=f"checkpoint_step_{global_step}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                last_loss=loss_total,
                use_ddp=cfg.distributed.use_ddp,
                ema_obj=ema_obj
            )

    return global_step, loss_total


def ddp_sample(model, ema_obj=None, **kwargs):
    """
    Safely call model.sample(...) from a DDP-wrapped model.
    """
    # 1) If an EMA object is provided, retrieve the actual EMA model:
    if ema_obj is not None:
        sample_model = ema_obj.ema_model_inference()
    else:
        sample_model = None

    # 2) Unwrap DDP if necessary
    if isinstance(model, DDP):
        return model.module.sample(model_ema=sample_model, **kwargs)
    else:
        return model.sample(model_ema=sample_model, **kwargs)


def train_model(cfg: DictConfig) -> None:
    # 0) Possibly set up output dirs
    main_save_dir = get_main_save_directory(cfg)

    # 1) Initialize DDP if needed
    local_rank = 0
    if cfg.distributed.use_ddp:
        local_rank = setup_distributed(cfg)
        torch.cuda.set_device(local_rank)

    # 2) Load data
    train_loader = load_training_data(cfg)
    device = torch.device(f"cuda:{local_rank}") if cfg.training.device == "cuda" else torch.device("cpu")

    # 3) Load or create VAE (frozen)
    from models.utils.model_loader import load_model
    vae = load_model(model_type=ModelType.VAE, **cfg.vae).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # 4) Build your diffusion model
    #    E.g. create a DiT model, then wrap it in MultiModalDiffusion, or use a helper
    mm_diff_model = load_model(
        model_type=ModelType.DIFFUSION,
        model_variant=DiTTrainingVersion.base_dit_training,
        **cfg.diffusion
    )
    mm_diff_model.to(device)

    # 5) Create EMA
    ema = None
    if is_main_process():
        ema = EMA(mm_diff_model, decay=0.9999)

    # 6) Wrap in DDP if needed
    if cfg.distributed.use_ddp:
        mm_diff_model = DDP(mm_diff_model, device_ids=[local_rank], output_device=local_rank,
                            find_unused_parameters=True)

    # 7) Optimizer
    optimizer = optim.AdamW(mm_diff_model.parameters(), lr=cfg.training.lr)

    # 8) Optionally resume
    start_epoch = 0
    global_step = 0
    if cfg.training.resume_training:
        start_epoch, global_step = resume_from_checkpoint(
            resume_dir=main_save_dir,
            model=mm_diff_model,
            optimizer=optimizer,
            device=device,
            use_ddp=cfg.distributed.use_ddp,
            ema_obj=ema
        )

    # 9) Main training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.distributed.use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        global_step, last_loss = train_one_epoch(
            epoch=epoch,
            model=mm_diff_model,
            optimizer=optimizer,
            train_loader=train_loader,
            cfg=cfg,
            vae=vae,
            global_step=global_step,
            base_save_path=main_save_dir,
            device=device,
            ema_obj=ema
        )

    # 10) Final checkpoint
    if cfg.training.save_model_interval is not None and is_main_process():
        save_checkpoint(
            ckpt_dir=main_save_dir,
            ckpt_name=f"checkpoint_step_{global_step}_final.pt",
            model=mm_diff_model,
            optimizer=optimizer,
            epoch=cfg.training.epochs,
            step=global_step,
            last_loss=last_loss,
            use_ddp=cfg.distributed.use_ddp,
            ema_obj=ema
        )

    if cfg.distributed.use_ddp:
        cleanup_distributed()

    if is_main_process():
        print("Training complete!")


def get_main_save_directory(cfg):
    """
    Determines the main save directory for training outputs.

    Args:
        cfg (object): Configuration object with training parameters.
        setup_storage_directory (function): Function to create a new storage directory if needed.

    Returns:
        Path: The path to the main save directory.
    """
    # Determine the base save path
    base_save_path = Path(cfg.training.get("base_save_path", Path(__file__).resolve().parent.parent / "training_outputs"))

    if cfg.training.resume_training:
        main_save_dir = cfg.training.get("resume_checkpoint_dir")

        if not main_save_dir:
            # Find the most recent directory if no checkpoint dir is provided
            try:
                main_save_dir = max(
                    (p for p in base_save_path.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime
                )
            except ValueError:
                raise FileNotFoundError("No existing checkpoint directories found for resuming training.")
    else:
        main_save_dir = setup_storage_directory(base_save_path, label=cfg.training.get("save_label"))

    # Create necessary subdirectories
    for subdir in ["samples", "checkpoints"]:
        os.makedirs(Path(main_save_dir, subdir), exist_ok=True)

    return main_save_dir


def ddp_sample(model, ema_obj=None, **kwargs):
    """
    A helper function to call the diffusion model's 'sample(...)' method in a DDP-safe way.
    - If 'model' is wrapped in DistributedDataParallel (DDP), we unwrap it via 'model.module'.
    - If 'ema_obj' is provided, we call 'ema_obj.ema_model_inference()' to get the EMA model to pass as 'model_ema'.
    - Any additional kwargs are forwarded to the 'sample(...)' method.
    """
    # 1) If an EMA object is provided, retrieve the actual EMA model:
    if ema_obj is not None:
        sample_model = ema_obj.ema_model_inference()  # This is a nn.Module (the EMA copy)
    else:
        sample_model = None  # Means we'll pass None, or use the model itself

    # 2) Unwrap the model if it's in DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # We want to call 'sample(...)' on the actual underlying module
        return model.module.sample(
            model_ema=sample_model,
            **kwargs
        )
    else:
        # Non-DDP or single-GPU usage
        return model.sample(
            model_ema=sample_model,
            **kwargs
        )


if __name__ == "__main__":

    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        train_model(cfg)