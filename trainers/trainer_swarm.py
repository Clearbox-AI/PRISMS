# trainers/trainer_swarm.py
from __future__ import annotations
import os, torch, torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from datetime import datetime
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# ──────────────────────────────────────────────────────────────────────────────
from data.loader import load_training_data
from data.tabular_transforms import FittedTransforms
from enums.models.model_types import ModelType
from models.utils.model_loader import load_model
from models.swarm_model import SwarmMultiModalModel        # ← path fix
try:
    from swarmlearning.pyt import SwarmCallback
except ImportError:
    from swarmlearning.pytorch import SwarmCallback                  # ← SL import
from utils.ddp import is_main_process, setup_distributed, cleanup_distributed
from utils.model import save_checkpoint, resume_from_checkpoint
from utils.path_management import setup_storage_directory
from utils.data import save_images, save_tabulars
from utils.model import save_checkpoint
from enums.training_versions import DiTTrainingVersion

# --------------------------------------------------------------------------- #
def train_one_epoch(
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        cfg: DictConfig,
        device: torch.device,
        global_step: int,
        base_save_path: Path,
        swarm_cb: SwarmCallback,                    # ← NEW
) -> tuple[int, float]:

    model.train()
    last_total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        # 1) ─── fetch data --------------------------------------------------
        images = batch['image'].to(device, non_blocking=True)
        tabs   = batch['tabular'].to(device, non_blocking=True)
        label  = torch.tensor(
            [0 if g == "CN" else 1 for g in batch['metadata']["GROUP"]],
            device=device
        )

        # 2) ─── forward / loss ---------------------------------------------
        loss_tot, loss_img, loss_tab = model(images, tabs, labels=label)  # ← no manual encode
        last_total_loss = loss_tot.item()

        optimizer.zero_grad(set_to_none=True)
        loss_tot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        swarm_cb.on_batch_end(global_step)            # ← SL hook

        # 3) ─── logging -----------------------------------------------------
        if is_main_process() and global_step % cfg.training.log_interval == 0:
            print(f"[E{epoch+1} | S{global_step}] "
                  f"Img {loss_img.item():.6f}  Tab {loss_tab.item():.6f} "
                  f"Tot {last_total_loss:.6f}")

        # 4) ─── periodic sampling ------------------------------------------
        if is_main_process() and global_step % cfg.training.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                # wrapper.sample already returns *decoded* images + tables
                img_out, tab_out = ddp_sample(model, batch_size=4, guidance_scale=2.5)

                if img_out is not None:
                    save_images(base_save_path, img_out, global_step)
                if tab_out is not None:
                    save_tabulars(base_save_path, tab_out, global_step)
            model.train()

        # 5) ─── checkpoint --------------------------------------------------
        if (cfg.training.save_model_interval and
                global_step % cfg.training.save_model_interval == 0 and
                is_main_process()):
            save_checkpoint(base_save_path,
                            f"checkpoint_step_{global_step}.pt",
                            model, optimizer, epoch,
                            global_step, loss_tab.item(),
                            use_ddp=cfg.distributed.use_ddp)

    return global_step, last_total_loss
# --------------------------------------------------------------------------- #
def train_model(cfg: DictConfig) -> None:

    # 0) ─── output folders ---------------------------------------------------
    main_save_dir = get_main_save_directory(cfg)

    # 1) ─── DDP --------------------------------------------------------------
    local_rank = 0
    if cfg.distributed.use_ddp:
        local_rank = setup_distributed(cfg)
        torch.cuda.set_device(local_rank)

    # 2) ─── data -------------------------------------------------------------
    train_loader, _ = load_training_data(cfg)

    # 3) ─── models -----------------------------------------------------------
    device = torch.device(f"cuda:{local_rank}" if cfg.training.device == "cuda" else "cpu")
    vae   = load_model(ModelType.VAE, cfg).to(device).eval()
    ft    = FittedTransforms.load(Path("/home/PRISMS/data/computations/tab_ft.pkl"))
    diff  = load_model(ModelType.DIFFUSION, cfg=cfg, tab_transforms=ft).to(device)
    model = SwarmMultiModalModel(vae, diff, cfg.vae.scaling_factor).to(device)

    # 4) ─── wrap in DDP (before optimizer!) ----------------------------------
    if cfg.distributed.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # 5) ─── optimiser (diffusion params only) --------------------------------
    diff_ref = model.module.diffusion if isinstance(model, DDP) else model.diffusion
    optimizer = optim.AdamW(param_groups(diff_ref), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

    # 6) ─── resume -----------------------------------------------------------
    start_epoch = global_step = 0
    if cfg.training.resume_training:
        start_epoch, global_step = resume_from_checkpoint(main_save_dir,
                                                          model, optimizer,
                                                          device,
                                                          use_ddp=cfg.distributed.use_ddp)

    # 7) ─── SwarmCallback ----------------------------------------------------
    swSync   = int(os.getenv("SYNC_INTERVAL", "20"))
    minPeers = int(os.getenv("MIN_PEERS", "2"))
    # unwrap for Swarm if DDP is used
    sw_model = model.module if isinstance(model, DDP) else model
    swarm_cb = SwarmCallback(syncFrequency=swSync,
                             minPeers=minPeers,
                             model=sw_model,
                             totalEpochs=cfg.training.epochs)
    swarm_cb.on_train_begin()

    # 8) ─── training loop ----------------------------------------------------
    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.distributed.use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        global_step, last_loss = train_one_epoch(epoch, model, optimizer,
                                                 train_loader, cfg, device,
                                                 global_step, main_save_dir,
                                                 swarm_cb)
        swarm_cb.on_epoch_end(epoch)

    # 9) ─── final checkpoint -------------------------------------------------
    if cfg.training.save_model_interval and is_main_process():
        save_checkpoint(main_save_dir,
                        f"checkpoint_step_{global_step}_final.pt",
                        model, optimizer,
                        cfg.training.epochs, global_step, last_loss,
                        use_ddp=cfg.distributed.use_ddp)
    swarm_cb.on_train_end()

    # 10) ─── cleanup ---------------------------------------------------------
    if cfg.distributed.use_ddp:
        cleanup_distributed()
    if is_main_process():
        print("Training complete!")

# --------------------------------------------------------------------------- #
def ddp_sample(model, *args, **kw):
    return model.module.sample(*args, **kw) if isinstance(model, DDP) else model.sample(*args, **kw)

# … get_main_save_directory(), FlowMonitor, param_groups() stay unchanged …



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

def param_groups(model):
    decay, no_decay = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if isinstance(p, nn.LayerNorm) or 'bias' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': 1e-2},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

if __name__ == "__main__":

    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        train_model(cfg)