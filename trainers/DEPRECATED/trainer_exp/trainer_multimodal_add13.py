import os
from torch import optim
from pathlib import Path
from hydra import compose, initialize_config_dir

from omegaconf import DictConfig, OmegaConf

from data.loader import load_training_data
from models.vae.vae import encode_images, decode_latents
from models.diffusion.diffusion_multimodal_add13 import EMA
from utils.ddp import is_main_process, setup_distributed, cleanup_distributed
from utils.model13 import save_checkpoint, resume_from_checkpoint
from utils.path_management import setup_storage_directory
from utils.data import save_images, save_tabulars
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import math
from torch.cuda.amp import autocast, GradScaler
import torch
from torch import amp
from torch import nn
from typing import List, Tuple
from models.dit.dit_multimodal_add13 import MultiModalDiT
from models.diffusion.diffusion_multimodal_add13 import MultiModalDiffusion

# -----------------------------------------------------------------------------
#  Helper to collect gate parameters for warm‑up
# -----------------------------------------------------------------------------

def collect_gate_params(dit: MultiModalDiT) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for blk in dit.blocks_imgtotab:
        params.append(blk.adaLN_modulation_x[-1].weight)
    for blk in dit.blocks_tabtoimg:
        params.append(blk.adaLN_modulation_t[-1].weight)
    return params

# -----------------------------------------------------------------------------
#  Gate statistics logger
# -----------------------------------------------------------------------------

def log_gate_stats(step: int, gate_params: List[nn.Parameter]):
    if not is_main_process():
        return
    g_mean = torch.stack([p.abs().mean() for p in gate_params]).mean().item()
    print(f"[Gate] step {step}: mean |gate| = {g_mean:.4f}")


# -----------------------------------------------------------------------------
#  Single‑epoch training
# -----------------------------------------------------------------------------

def train_one_epoch(
    epoch: int,
    diffusion: MultiModalDiffusion | DDP,
    optimizer: optim.Optimizer,
    sched_main, sched_gate,
    scaler: GradScaler,
    train_loader,
    cfg: DictConfig,
    vae: nn.Module,
    global_step: int,
    save_dir: Path,
    device: torch.device,
    ema: EMA | None,
    gate_params: List[nn.Parameter]
) -> Tuple[int, float]:

    diffusion.train()
    core = get_inner(diffusion)
    loss_total_last = 0.0
    accum = cfg.training.grad_accum

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device, non_blocking=True)
        tab = batch['tabular'].to(device, non_blocking=True)
        latents = encode_images(vae, images, cfg.vae.scaling_factor)

        with amp.autocast(device_type="cuda"):
            loss_total, loss_img, loss_tab = diffusion(latents, tab)
            loss_total = loss_total / accum  # scale for grad‑accum

        scaler.scale(loss_total).backward()
        loss_total_last = loss_total.item() * accum  # original scale for logging

        # ---- optimisation step every `accum` mini‑batches
        if (batch_idx + 1) % accum == 0:
            global_step += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched_main.step()
            sched_gate.step()
            if ema is not None:
                ema.update(core.dit)

            # ------ logging
            if is_main_process() and global_step % cfg.training.log_interval == 0:
                print(f"[E{epoch + 1} S{global_step}] Img {loss_img.item():.4f}"
                      f"  Tab {loss_tab.item():.4f}  Tot {loss_total_last:.4f}")
                log_gate_stats(global_step, gate_params)

            # ------ sampling
            if is_main_process() and global_step % cfg.training.sample_interval == 0:
                core.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    with torch.no_grad():
                        if ema is not None:
                            model_for_sampling = ema.as_model().to(device)  # safe clone
                        else:
                            model_for_sampling = core.dit.eval()

                    x_img, x_tab = core.sample(
                        model_ema=model_for_sampling,
                        batch_size=cfg.training.sample_batch_size,
                        num_steps=cfg.training.sample_steps,
                    )

                    imgs = decode_latents(vae, x_img, cfg.vae.scaling_factor)
                    save_images(save_dir, imgs, global_step)
                    save_tabulars(save_dir, x_tab, global_step)
                # core.train()

            # -------- checkpoint ----------
            if (cfg.training.save_model_interval and is_main_process()
                    and global_step % cfg.training.save_model_interval == 0):
                save_checkpoint(
                    ckpt_dir=save_dir,
                    ckpt_name=f"ckpt_{global_step}.pt",
                    model=diffusion,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    last_loss=loss_total,
                    use_ddp=cfg.distributed.use_ddp,
                    ema_obj=ema
                )


    return global_step, loss_total_last

# -----------------------------------------------------------------------------
#  Main entry
# -----------------------------------------------------------------------------

def train_model(cfg: DictConfig):
    # 0) Possibly set up output dirs
    save_dir = get_main_save_directory(cfg)

    rank = setup_distributed(cfg) if cfg.distributed.use_ddp else 0
    device = torch.device("cuda", rank) if cfg.training.device == "cuda" else torch.device("cpu")

    # 2) Load data
    train_loader = load_training_data(cfg)
    with open("/home/PRISMS/data/computations/nacc_stats.json", "r") as f:
        stats = json.load(f)
    tab_std = torch.as_tensor(torch.tensor(stats["tabular_scaler_scale_"], dtype=torch.float32), device=device)
    tab_std[tab_std < 1e-12] = 1.0

    # 3) Load or create VAE (frozen)
    from models.utils.model_loader import load_model
    vae = load_model(model_type=ModelType.VAE, **cfg.vae).to(device).eval()
    vae.requires_grad_(False)

    # 4) Build your diffusion model
    #    E.g. create a DiT model, then wrap it in MultiModalDiffusion, or use a helper
    diffusion_core = load_model(
        model_type=ModelType.DIFFUSION,
        model_variant=DiTTrainingVersion.base_dit_training,
        tmp_param=tab_std,
        **cfg.diffusion
    ).to(device)

    if cfg.distributed.use_ddp:
        diffusion = DDP(diffusion_core, device_ids=[rank], output_device=rank,
                        broadcast_buffers=False, find_unused_parameters=True)
    else:
        diffusion = diffusion_core

    core = get_inner(diffusion)

    # ---- EMA (DiT only)
    ema = EMA(core.dit, decay=0.9999) if is_main_process() else None                # T‑1

    # ---- optimiser + schedulers
    gate_params = collect_gate_params(core.dit)
    gate_param_ids = {id(p) for p in gate_params}
    # base group = every parameter whose *identity* is not in gate list
    base_group = {"params": [p for p in diffusion.parameters()
                             if id(p) not in gate_param_ids]}
    gate_group = {"params": gate_params, "lr": cfg.training.lr}
    optimizer = optim.AdamW([base_group, gate_group], lr=cfg.training.lr, weight_decay=0.01)

    # cosine schedule for all params
    total_steps = math.ceil(cfg.training.epochs * len(train_loader) / cfg.training.grad_accum)
    sched_main = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min((s + 1) / (cfg.training.warmup_frac * total_steps), 1.0)
                            * 0.5
                            * (1 + math.cos(math.pi * s / total_steps)),
    )
    sched_gate = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda s: 1.0, lambda s: min(s / cfg.training.gate_warmup_steps, 1.0)],
    )

    scaler = GradScaler()                                                         # T‑7

    # ---- resume
    start_epoch = 0; global_step = 0
    if cfg.training.resume_training:
        start_epoch, global_step = resume_from_checkpoint(
            save_dir, diffusion, optimizer, device,
            use_ddp=cfg.distributed.use_ddp, ema_obj=ema)
        core = diffusion.module if isinstance(diffusion, DDP) else diffusion
        core.global_step = global_step

    # ---- training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.distributed.use_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        global_step, last = train_one_epoch(
            epoch=epoch,
            diffusion=diffusion,
            optimizer=optimizer,
            sched_main=sched_main,
            sched_gate=sched_gate,
            scaler=scaler,
            train_loader=train_loader,
            cfg=cfg,
            vae=vae,
            global_step=global_step,
            save_dir=save_dir,
            device=device,
            ema=ema,
            gate_params=gate_params
        )

    if is_main_process():
        save_checkpoint(
            ckpt_dir=save_dir,
            ckpt_name=f"ckpt_{global_step}_final.pt",
            model=diffusion,
            optimizer=optimizer,
            epoch=cfg.training.epochs,
            step=global_step,
            last_loss=last,
            use_ddp=cfg.distributed.use_ddp,
            ema_obj=ema
        )


    if cfg.distributed.use_ddp:
        cleanup_distributed()
    if is_main_process():
        print("Training complete✓")

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

def get_inner(model: nn.Module) -> nn.Module:
    """
    Return the actual nn.Module, whether `model` is a plain module
    or wrapped in torch.nn.parallel.DistributedDataParallel.
    """
    return model.module if isinstance(model, DDP) else model



if __name__ == "__main__":

    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        train_model(cfg)