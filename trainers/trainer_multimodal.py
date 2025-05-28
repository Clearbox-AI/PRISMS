from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra import compose, initialize_config_dir
import os
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from models.utils.model_loader import load_model
from data.loader import load_training_data
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from utils.ddp import is_main_process, setup_distributed, cleanup_distributed
from utils.model import save_checkpoint, resume_from_checkpoint
from utils.path_management import setup_storage_directory
from utils.data import save_images, save_tabulars
from models.vae.vae import encode_images, decode_latents


def train_one_epoch(
        epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader, cfg: DictConfig, vae, global_step: int = 0,
        base_save_path: Path = None, device: torch.device = None
):
    """
    One epoch of training.
    Returns the updated global_step and last_total_loss for checkpointing.
    """
    model.train()
    last_total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        # 1) Get data
        images = batch['image'].to(device, non_blocking=True)
        tab_data = batch['tabular'].to(device, non_blocking=True)

        # 2) Encode images -> latents (via VAE)
        latents = encode_images(vae, images, cfg.vae.scaling_factor)

        # 3) Forward and loss
        loss_total, loss_img, loss_tab = model(latents, tab_data)
        last_total_loss = loss_total.item()

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prova

        optimizer.step()

        # 4) Logging (only rank-0 prints)
        if is_main_process() and (global_step % cfg.training.log_interval == 0):
            msg = (f"[Epoch {epoch + 1} | Step {global_step}] "
                   f"Img Loss: {loss_img.item():.4f}")
            msg += f" | Tab Loss: {loss_tab.item():.4f}"
            msg += f" | Total: {last_total_loss:.4f}"
            print(msg)

        # 5) Sampling (only rank-0)
        if is_main_process() and (global_step % cfg.training.sample_interval == 0):
            model.eval()
            with torch.no_grad():
                latents_img_out, latents_tab_out = ddp_sample(
                    model=model,
                    batch_size=4,
                )

                # Now decode latents -> images
                # 1) Decode image latents -> actual images
                if latents_img_out is not None:
                    recon_images = decode_latents(vae, latents_img_out, cfg.vae.scaling_factor)
                    save_images(base_save_path, recon_images, global_step)

                # 2) Save tabular data if latents_tab_out is relevant
                if latents_tab_out is not None:
                    save_tabulars(base_save_path, latents_tab_out, global_step)

            model.train()

        # 6) Save model checkpoints (only rank-0)
        if (cfg.training.save_model_interval is not None and
                global_step % cfg.training.save_model_interval == 0 and
                is_main_process()):

            save_checkpoint(
                ckpt_dir=base_save_path,
                ckpt_name=f"checkpoint_step_{global_step}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                last_loss=loss_tab.item(),
                use_ddp=cfg.distributed.use_ddp
            )

    return global_step, last_total_loss


def train_model(cfg: DictConfig) -> None:
    """
    Main training routine that sets up DDP if needed, loads data, models, and
    runs the training loop.
    """

    # 0) Setup output dirs using the date-based subfolder approach
    # paths can be given with cfg or using predefined paths
    main_save_dir = get_main_save_directory(cfg)

    # 1) Initialize DDP if desired
    local_rank = 0
    if cfg.distributed.use_ddp:
        local_rank = setup_distributed(cfg)
        torch.cuda.set_device(local_rank)

    # 2) Load training data
    train_loader, _ = load_training_data(cfg)

    # # 3) Load or create VAE
    # device = torch.device(f"cuda:{local_rank}") if cfg.training.device == "cuda" else torch.device("cpu")
    # vae = load_model(model_type=ModelType.VAE, **cfg.vae).to(device)
    # vae.requires_grad_(False)
    # vae.eval()  # Typically we keep the VAE frozen
    #
    # # 4) Build your diffusion model
    # mm_diff_model = load_model(
    #     model_type=ModelType.DIFFUSION,
    #     model_variant=DiTTrainingVersion.base_dit_training,
    #     **cfg.diffusion
    # )
    # mm_diff_model.to(device)

    # 3) Load or create VAE
    device = torch.device(f"cuda:{local_rank}") if cfg.training.device == "cuda" else torch.device("cpu")
    vae = load_model(ModelType.VAE, cfg=cfg).to(device)
    vae.requires_grad_(False).eval()  # keep VAE frozen

    from models.diffusion.diffusion_multimodal import build_transforms_from_schema
    tab_tf = build_transforms_from_schema("/home/PRISMS/data/computations/nacc_meta.json")
    # fit standardisation stats on one pass over the loader

    # 1) Temporarily disable standardisation
    orig_apply_std = [tf.apply_std for tf in tab_tf]
    for tf in tab_tf:
        tf.apply_std = False
    # 2) One full pass to fit μ/σ in transformed space
    sums = torch.zeros(len(tab_tf), device=device)
    sumsq = torch.zeros_like(sums)
    count = 0
    for batch in train_loader:
        x_raw = batch["tabular"].to(device)
        cols = [tab_tf[i](x_raw[:, i:i + 1]).squeeze(1) for i in range(len(tab_tf))]
        x_lat = torch.stack(cols, dim=1)  # (B, F) after non-linear map
        sums += x_lat.sum(0)
        sumsq += (x_lat ** 2).sum(0)
        count += x_lat.size(0)

    mu = sums / count
    sig = ((sumsq / count) - mu ** 2).sqrt().clamp_min_(1e-9)
    # 3) Register buffers and restore normal behaviour
    for i, tf in enumerate(tab_tf):
        # create *or* overwrite – works both when restarting and resuming
        if 'mu' in dict(tf.named_buffers()):
            tf.mu.data.copy_(mu[i])
            tf.sig.data.copy_(sig[i])
        else:
            tf.register_buffer('mu', mu[i])
            tf.register_buffer('sig', sig[i])

        tf.apply_std = orig_apply_std[i]  # usually True again


    # 4) Build the multi-modal diffusion model
    mm_diff_model = load_model(
        model_type=ModelType.DIFFUSION,
        cfg=cfg,
        tab_transforms=tab_tf
    ).to(device)

    # 5) Wrap the diffusion model in DDP (if desired)
    if cfg.distributed.use_ddp:
        mm_diff_model = DDP(mm_diff_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        monitor_target = mm_diff_model.module
    else:
        monitor_target = mm_diff_model

    THRESHOLD = 1e3
    log_dir = Path(main_save_dir) / "logs"
    monitor = FlowMonitor(
        monitor_target,
        threshold=THRESHOLD,
        log_dir=log_dir,
        rank=local_rank
    )

    # 6) Create optimizer
    optimizer = optim.AdamW(mm_diff_model.parameters(), lr=cfg.training.lr)

    # 7) Optionally resume training from checkpoint
    start_epoch = 0
    global_step = 0
    if cfg.training.resume_training:
        start_epoch, global_step = resume_from_checkpoint(
            resume_dir=main_save_dir,
            model=mm_diff_model,
            optimizer=optimizer,
            device=device,
            use_ddp=cfg.distributed.use_ddp
        )

    # 8) Training loop
    try:
        for epoch in range(start_epoch, cfg.training.epochs):
            # If using a DistributedSampler, set epoch for shuffling
            if cfg.distributed.use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            global_step, last_loss = train_one_epoch(
                epoch=epoch, model=mm_diff_model, optimizer=optimizer, train_loader=train_loader,
                cfg=cfg, vae=vae, global_step=global_step, base_save_path=main_save_dir, device=device,
            )

        # 9) Final checkpoint (only rank-0)
        if cfg.training.save_model_interval is not None and is_main_process():
            save_checkpoint(
                ckpt_dir=main_save_dir,
                ckpt_name=f"checkpoint_step_{global_step}_final.pt",
                model=mm_diff_model,
                optimizer=optimizer,
                epoch=cfg.training.epochs,
                step=global_step,
                last_loss=last_loss,
                use_ddp=cfg.distributed.use_ddp
            )

        # 10) Cleanup
        if cfg.distributed.use_ddp:
            cleanup_distributed()

        if is_main_process():
            print("Training complete!")
    except RuntimeError as e:
        print(f"Training stopped: {e}")
    finally:
        monitor.close()

def ddp_sample(model, *args, **kwargs):
    if isinstance(model, DDP):
        return model.module.sample(*args, **kwargs)
    return model.sample(*args, **kwargs)


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


class FlowMonitor:
    def __init__(self, model: nn.Module, threshold: float, log_dir: Path, rank: int = 0):
        self.threshold = threshold
        self.monitoring = False
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = f"suspicious_rank{rank}.log" if rank else "suspicious.log"
        self.log_path = log_dir / fname
        self.log_file = open(self.log_path, "a", buffering=1)
        self._register_hooks(model)

    def _timestamp(self):
        return datetime.now().isoformat()

    def _log(self, msg: str):
        self.log_file.write(f"{self._timestamp()} {msg}\n")

    def _check_and_log(self, name: str, tensor: torch.Tensor, where: str):
        if not torch.is_tensor(tensor):
            return
        t = tensor.detach()
        # compute stats
        t_min = t.min().item()
        t_max = t.max().item()
        t_mean = t.mean().item()
        # decide if we should start monitoring
        if (t_max > self.threshold or t_min < -self.threshold or
            torch.isnan(t).any() or torch.isinf(t).any() or self.monitoring):
            if not self.monitoring:
                self._log(f"▶▶ Threshold exceeded in `{name}` ({where}): "
                          f"min={t_min:.3e}, max={t_max:.3e}, mean={t_mean:.3e}")
                self.monitoring = True
            else:
                self._log(f"{where} `{name}`: min={t_min:.3e}, max={t_max:.3e}, mean={t_mean:.3e}")
            # if we see a NaN/Inf, immediately stop
            if torch.isnan(t).any() or torch.isinf(t).any():
                self._log(f"‼‼ NaN/Inf detected in `{name}` during {where}. Stopping training.")
                self.log_file.close()
                raise RuntimeError(f"NaN/Inf in `{name}` during {where}")

    def _make_fwd_hook(self, name):
        def hook(module, inp, out):
            # only check the outputs; you could also check inputs if you like
            if isinstance(out, torch.Tensor):
                self._check_and_log(name, out, "forward")
            elif isinstance(out, (tuple, list)):
                for i, o in enumerate(out):
                    self._check_and_log(f"{name}[{i}]", o, "forward")
        return hook

    def _make_grad_hook(self, name):
        def hook(grad):
            self._check_and_log(name, grad, "backward_grad")
            return grad
        return hook

    def _register_hooks(self, model):
        # forward hooks
        for name, module in model.named_modules():
            module.register_forward_hook(self._make_fwd_hook(name))
        # gradient hooks
        for name, param in model.named_parameters():
            param.register_hook(self._make_grad_hook(name))

    def close(self):
        self.log_file.close()

if __name__ == "__main__":

    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        train_model(cfg)