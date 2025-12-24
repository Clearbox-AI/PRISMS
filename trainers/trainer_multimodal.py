import warnings
warnings.filterwarnings("ignore", category=FutureWarning,)
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra import compose, initialize_config_dir
import os
import torch
import torch.nn as nn
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from models.utils.model_loader import load_model
from data.loader import load_training_data
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from utils.ddp import is_main_process, setup_distributed, cleanup_distributed, ddp_sample
from utils.model import save_checkpoint, resume_from_checkpoint
from utils.path_management import setup_storage_directory, get_main_save_directory
from utils.data import save_images, save_tabulars, _dynamic_threshold_pixel
from utils.logging import FlowMonitor
from utils.optim_utils import CPUEMA, param_groups
from models.vae.vae import encode_images, decode_latents
import torch.distributed as dist
from tqdm.auto import tqdm
import copy
import math
from torch.cuda.amp import GradScaler
from trainers.trainer_tabsyn import _pretrain_tabsyn_vae


def train_one_epoch(
        epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader, cfg: DictConfig, vae, global_step: int = 0,
        base_save_path: Path = None, device: torch.device = None, ema: Optional[CPUEMA] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
):
    """
    One epoch of training.
    Returns the updated global_step and last_total_loss for checkpointing.
    """
    model.train()
    last_total_loss = 0.0

    # --- AMP setup: FP16 usa GradScaler, BF16 no ---
    amp_dtype_cfg = str(cfg.training.get("amp_dtype", "bf16")).lower()
    use_fp16 = (amp_dtype_cfg == "fp16")
    scaler = GradScaler(enabled=use_fp16)
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        # 1) Get data
        images = batch['image'].to(device, non_blocking=True)
        tab_data = batch['tabular'].to(device, non_blocking=True)
        label = torch.tensor([0 if el == "CN" else 1 for el in batch['metadata']["GROUP"]]).to(device, non_blocking=True)

        # 2) Encode images -> latents (via VAE) + 3) Forward con autocast coerente
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=amp_dtype):
            latents = encode_images(vae, images, cfg.vae.scaling_factor)
            loss_total, loss_img, loss_tab = model(latents, tab_data, labels=label)
        last_total_loss = float(loss_total.detach().item())
        if use_fp16:
            scaler.scale(loss_total).backward()
            # gradient clipping va fatto *dopo* unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None: scheduler.step()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            if scheduler is not None: scheduler.step()

        # update EMA on the unwrapped module
        if ema is not None:
            (ema.update(model.module) if isinstance(model, DDP) else ema.update(model))

        # 4) Logging (only rank-0 prints)
        if is_main_process() and (global_step % cfg.training.log_interval == 0):
            msg = (f"[Epoch {epoch + 1} | Step {global_step}] "
                   f"Img Loss: {loss_img.item():.7f} | "
                   f"Tab Loss: {loss_tab.item():.7f} | "
                   f"Total: {last_total_loss:.7f}")

            # NOTE: TabSyn‑VAE is pretrained in a separate phase; no inline flag here.

            # --- Balancing stats (if the module exposes them) ---
            monitor_t = (model.module if isinstance(model, DDP) else model)
            # traccia sigma_data_img appreso via EMA
            try:
                s_img_now = float(monitor_t.sigma_data_img.item())
                msg += f" | sigma_data_img={s_img_now:.3f}"
            except Exception:
                s_img_now = float('nan')
            w_tab_attr = getattr(monitor_t, "_last_w_tab", None)
            g_img_attr = getattr(monitor_t, "_last_gradnorm_img", None)
            g_tab_attr = getattr(monitor_t, "_last_gradnorm_tab_scaled", None)
            tau_now = (getattr(monitor_t, "_tau_end", None) is not None) and \
                      float(min(1.0, getattr(monitor_t, "step_counter").item() / max(1, getattr(
                      monitor_t, "_tau_wu", 1)))) * \
                      float(getattr(monitor_t, "_tau_end", 0.0))
            edm_on = int(getattr(monitor_t, "step_counter").item() >= getattr(monitor_t, "_tab_edm_wu", 0))

            if (w_tab_attr is not None) and (g_img_attr is not None) and (g_tab_attr is not None):
                msg += (f" | w_tab={w_tab_attr.item():.3f}"
                        f" | ||g_img||={g_img_attr.item():.3e}"
                        f" | ||g_tab||={g_tab_attr.item():.3e}")
                msg += f" | τ={tau_now:.2f} | tabEDM={edm_on}"

            # --- online metrics from forward() (coherence, corrΔ, MMD) ---
            def _to_float(x):
                if x is None: return float('nan')
                if isinstance(x, torch.Tensor):
                    try: return float(x.item())
                    except Exception: return float('nan')
                try: return float(x)
                except Exception: return float('nan')
            coh = _to_float(getattr(monitor_t, "_last_coh_acc1", None))  # retrieval@1 (img↔tab)
            corr = _to_float(getattr(monitor_t, "_last_corr_L1_offdiag", None))  # L1 off-diag corr diff
            mmd = _to_float(getattr(monitor_t, "_last_mmd_num", None))  # MMD on numeric latents
            tab_gate = _to_float(getattr(monitor_t, "_last_tab_gate", None))  # g_tab (soft-TTUR)
            cross_gate = _to_float(getattr(monitor_t, "_last_cross_gate", None))  # g_cross (grad warm-up)
            # (7) nuovi: MSE per bucket di σ (immagine/tab)
            img_mse_s = _to_float(getattr(monitor_t, "_last_img_mse_b_small", None))
            img_mse_m = _to_float(getattr(monitor_t, "_last_img_mse_b_mid", None))
            img_mse_l = _to_float(getattr(monitor_t, "_last_img_mse_b_large", None))
            tab_mse_s = _to_float(getattr(monitor_t, "_last_tab_mse_b_small", None))
            tab_mse_m = _to_float(getattr(monitor_t, "_last_tab_mse_b_mid", None))
            tab_mse_l = _to_float(getattr(monitor_t, "_last_tab_mse_b_large", None))

            if not math.isnan(coh): msg += f" | coh@1={coh:.3f}"
            if not math.isnan(corr): msg += f" | corrΔ_offdiag={corr:.3f}"
            if not math.isnan(mmd): msg += f" | mmd_num={mmd:.3f}"
            if not math.isnan(tab_gate): msg += f" | g_tab={tab_gate:.3f}"
            if not math.isnan(cross_gate): msg += f" | g_cross={cross_gate:.3f}"
            # stampa compatta dei bucket (se disponibili)
            if not math.isnan(
                img_mse_s): msg += f" | imgMSE[σ≈(0.5/1/3)]={img_mse_s:.3e}/{img_mse_m:.3e}/{img_mse_l:.3e}"
            if not math.isnan(
                tab_mse_s): msg += f" | tabMSE[σ≈(0.5/1/3)]={tab_mse_s:.3e}/{tab_mse_m:.3e}/{tab_mse_l:.3e}"

            print(msg)
            # --- Persist metrics to TSV (rank-0 only) ---
            try:
                logs_dir = Path(base_save_path) / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                mpath = logs_dir / "metrics.tsv"
                headers = [
                    "step", "epoch", "loss_total", "loss_img", "loss_tab",
                    "coh_at1", "corr_delta_offdiag", "mmd_num",
                    "w_tab", "grad_img", "grad_tab_scaled",
                    "g_tab", "g_cross",
                    "img_mse_s", "img_mse_m", "img_mse_l",
                    "tab_mse_s", "tab_mse_m", "tab_mse_l"
                ]
                row = [
                    global_step, epoch + 1, last_total_loss, float(loss_img.item()),
                    float(loss_tab.item()),
                    s_img_now,
                    coh, corr, mmd,
                    _to_float(w_tab_attr), _to_float(g_img_attr), _to_float(g_tab_attr),
                    tab_gate, cross_gate,
                    img_mse_s, img_mse_m, img_mse_l,
                    tab_mse_s, tab_mse_m, tab_mse_l
                ]
                if not mpath.exists():
                    with open(mpath, "w") as f:
                        f.write("\t".join(headers) + "\n")
                with open(mpath, "a") as f:
                    f.write("\t".join(str(v) for v in row) + "\n")
            except Exception as e:
                print(f"[WARN] failed to write metrics.tsv: {e}")

        # 5) Sampling (only rank-0)
        if is_main_process() and (global_step % cfg.training.sample_interval == 0):
            model.eval()
            with ((torch.no_grad())):
                # pre-init to avoid UnboundLocalError on early exit/exception
                latents_img_out, latents_tab_out = None, None
                use_ema_now = (ema is not None) and ema.ready()
                if use_ema_now:
                    if isinstance(model, DDP):
                        ema.store(model.module);
                        ema.copy_to(model.module)
                    else:
                        ema.store(model); ema.copy_to(model)

                # Per-modality CFG: image stronger; tab moderate.
                try:
                    _bs = int(getattr(cfg.training, "sample_bs", 4))
                    if 'metadata' in batch and 'GROUP' in batch['metadata']:
                        label = torch.tensor([0 if el == "CN" else 1 for el in batch['metadata']["GROUP"]],device = device)
                        labs = (label[:_bs] if label.numel() >= _bs else label.repeat(((_bs + label.numel() - 1) // label.numel()))[:_bs])
                    else:
                        labs = torch.full((_bs,), (model.module.dit.NULL_ID if isinstance(model, DDP) else model.dit.NULL_ID),
                                                              device = device, dtype = torch.long)
                    latents_img_out, latents_tab_out = ddp_sample(
                        model = model,
                        batch_size = _bs,
                        guidance_scale = {"img": 2.5, "tab": 1.4},
                        labels = labs
                    )
                except Exception as e:
                    print(f"[Sampling] WARN: sampler failed: {e}")

                if use_ema_now:
                    print(f"[Sampling] EMA IS ready.")
                    if isinstance(model, DDP):
                        ema.restore(model.module)
                    else:
                        ema.restore(model)
                else:
                    if is_main_process():
                        print(f"[Sampling] EMA not ready (updates={ema.num_updates}); sampling online weights.")

                # Now decode latents -> images
                # 1) Decode image latents -> actual images
                if latents_img_out is not None:
                    recon_images = decode_latents(vae, latents_img_out, cfg.vae.scaling_factor)
                    # --- Dynamic Thresholding in *pixel space* (post-VAE) ---
                    try:
                        dt_enable = bool(getattr(cfg.training, "dt_pixel_enable", False))
                        if dt_enable:
                            dt_p = float(getattr(cfg.training, "dt_pixel_p", 0.995))
                            dt_rescale = bool(getattr(cfg.training, "dt_pixel_rescale", False))
                            recon_images = _dynamic_threshold_pixel(recon_images, p=dt_p, rescale=dt_rescale)
                    except Exception as e:
                        print(f"[Sampling] pixel-DT skipped (non-fatal): {e}")
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
                use_ddp=cfg.distributed.use_ddp,
                ema_obj = ema
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
    ld = load_training_data(cfg)
    if isinstance(ld, tuple):
        train_loader, val_loader = ld
    else:
        train_loader, val_loader = ld, None

    # 3) Load or create VAE
    device = torch.device(f"cuda:{local_rank}") if cfg.training.device == "cuda" else torch.device("cpu")
    vae = load_model(ModelType.VAE, cfg=cfg).to(device)
    vae.requires_grad_(False).eval()  # keep VAE frozen

    # 4) Estrai *le stesse* FittedTransforms usate dal Dataset (fonte di verità)
    def _get_dataset(obj):
        ds = obj
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        return ds
    train_ds = _get_dataset(train_loader.dataset)
    if not hasattr(train_ds, "ft"):
        raise RuntimeError(
            "Il dataset di training non espone 'ft'. Assicurati che NaccDataset costruisca/carichi FittedTransforms.")
    ft = train_ds.ft

    mm_diff_model = load_model(
        model_type=ModelType.DIFFUSION,
        cfg=cfg,
        tab_transforms=ft
    ).to(device)

    # Collega la VAE al diffusore per la FFT‑loss in pixel‑space
    # e imposta un ramp 2–4k step sul cross‑attn (niente hard‑off).
    try:
        mm_diff_model.img_vae = vae
        mm_diff_model.vae_scaling_factor = float(cfg.vae.scaling_factor)
        mm_diff_model.img_fft_on_decoded = True
        # preferisci ramp morbido al posto dell'hard‑off
        if hasattr(mm_diff_model, "_xattn_hard_off_until"):
            mm_diff_model._xattn_hard_off_until = 0
        # anticipa detach per "cross‑on" graduale
        if hasattr(mm_diff_model, "detach_cross_until"):
            mm_diff_model.detach_cross_until = min(int(mm_diff_model.detach_cross_until), 2000)
        # consenti override da cfg.training, altrimenti default 2–4k
        mm_diff_model.xattn_ramp_from = int(getattr(cfg.training, "xattn_ramp_from", 2000))
        mm_diff_model.xattn_ramp_to = int(getattr(cfg.training, "xattn_ramp_to", 4000))
        mm_diff_model.xattn_cosine_ramp = bool(getattr(cfg.training, "xattn_cosine_ramp", True))
    except Exception as e:
        if is_main_process():
            print(f"[WARN] non sono riuscito a impostare VAE/ramp sul diffusore: {e}")

    if cfg.distributed.use_ddp:
        mm_diff_model = DDP(mm_diff_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        monitor_target = mm_diff_model.module
    else:
        monitor_target = mm_diff_model
    # --- sampler immagine: sigma_max = 5.0 (allineato al training) ---
    monitor_target.edm_img.sigma_max = 5.0
    # --- assicura EMA sbloccata a inizio training ---
    monitor_target._freeze_sigma_data_img = False

    pretrain_needed = False
    has_tabsyn = getattr(monitor_target, "has_tabsyn_vae", lambda: False)()
    if has_tabsyn:
        ckpt_path = cfg.tabsyn.checkpoint_path
        use_ckpt = bool(cfg.tabsyn.use_checkpoint) and os.path.exists(ckpt_path)
        if use_ckpt:
            monitor_target.tabsyn_load(ckpt_path, map_location=device)
            monitor_target.freeze_tab_vae(False)
            if hasattr(monitor_target, "tab_vae") and monitor_target.tab_vae is not None:
                monitor_target.tab_vae.eval()
            if is_main_process():
                print(f"[TabSyn] Loaded pretrained VAE from {ckpt_path}.", flush=True)
        else:
            pretrain_needed = True
            if is_main_process():
                print(f"[TabSyn] No pretrained VAE at {ckpt_path}. Starting pretraining for " 
                      f"P={cfg.tabsyn.tabsyn_pretrain_steps} global steps…", flush=True)


    if pretrain_needed:
        if is_main_process():
            print(f"[Trainer] Pretraining TabSyn VAE (global steps={cfg.tabsyn.tabsyn_pretrain_steps}).",
                               flush=True)
        try:
            vae_loader, val_loader = load_training_data(cfg)
        except Exception:
            vae_loader = load_training_data(cfg)
            val_loader = None
        _pretrain_tabsyn_vae(
            mm_diff_model,
            vae_loader,
            device,
            cfg.tabsyn.tabsyn_pretrain_steps,
            log_every = max(50, cfg.training.log_interval),
            pretrain_epoch = 10_000,
            val_loader=val_loader,
            es_patience=getattr(getattr(cfg.tabsyn, "early_stop", {}), "patience", 5),
            es_min_delta = getattr(getattr(cfg.tabsyn, "early_stop", {}), "min_delta", 1e-3),
            eval_every = getattr(getattr(cfg.tabsyn, "early_stop", {}), "eval_every", 100),
            max_val_batches = getattr(getattr(cfg.tabsyn, "early_stop", {}), "max_val_batches", 32),
        )
        # all ranks finish roughly together now; short sync is fine
        if cfg.distributed.use_ddp and dist.is_initialized():
            dist.barrier()
        # rank‑0 saves; others wait, then everyone loads the same weights

        if is_main_process():
            os.makedirs(os.path.dirname(cfg.tabsyn.checkpoint_path), exist_ok=True)
            monitor_target.tabsyn_save(cfg.tabsyn.checkpoint_path)
            print(f"[TabSyn] Saved VAE checkpoint to {cfg.tabsyn.checkpoint_path}", flush=True)
        if cfg.distributed.use_ddp and dist.is_initialized():
            dist.barrier()  # ensure file is visible
        monitor_target.tabsyn_load(cfg.tabsyn.checkpoint_path, map_location=device)
        monitor_target.freeze_tab_vae(False)  # durante la diffusion resta in inferenza
        monitor_target.to(device)

    ema = CPUEMA(monitor_target, decay=0.9999)

    # -------- Verify σ_data per branch (rank-0 only) ------------------------
    if is_main_process():
        sigma_img = getattr(monitor_target, "sigma_data_img", None)
        sigma_tab_scalar = getattr(monitor_target, "sigma_data_tab", None)
        sigma_tab_vec = getattr(monitor_target, "sigma_data_tab_vec", None)
        if sigma_img is not None:
            s_img = sigma_img.item() if torch.is_tensor(sigma_img) else float(sigma_img)
        else:
            s_img = float('nan')
        if sigma_tab_scalar is not None:
            s_tab = sigma_tab_scalar.item() if torch.is_tensor(sigma_tab_scalar) else float(sigma_tab_scalar)
        elif sigma_tab_vec is not None:
            s_tab = float(sigma_tab_vec.mean().item())
        else:
            s_tab = float('nan')
        print(f"[Verify] sigma_data_img={s_img:.3f} | sigma_data_tab≈{s_tab:.3f}")

    THRESHOLD = 1e3
    log_dir = Path(main_save_dir) / "logs"
    monitor = FlowMonitor(
        monitor_target,
        threshold=THRESHOLD,
        log_dir=log_dir,
        rank=local_rank
    )

    # (4) LR più basso + scheduler warmup+cosine per stabilizzare
    base_lr = float(getattr(cfg.training, "lr", 1e-4))
    optimizer = optim.AdamW(param_groups(mm_diff_model), lr=base_lr, betas=(0.9, 0.999), eps=1e-8)
    # Stima dei passi totali per costruire il lambda (non serve precisione perfetta)
    steps_per_epoch = max(1, len(train_loader))
    total_steps_est = max(steps_per_epoch * int(cfg.training.epochs), 1)
    warmup_steps = int(getattr(cfg.training, "warmup_steps", 2000))
    min_factor = float(getattr(cfg.training, "lr_min_factor", 0.1))  # lr_min = base_lr * min_factor
    import math as _math
    def _warm_cos_lambda(step: int):
        # 0 → 1 su warmup, poi cosine fino a min_factor
        if step < warmup_steps:
            return max(1e-8, float(step) / float(max(1, warmup_steps)))
        t = (step - warmup_steps) / float(max(1, total_steps_est - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        # cosine from 1 → min_factor
        return (min_factor + 0.5 * (1.0 - min_factor) * (1.0 + _math.cos(_math.pi * t)))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=_warm_cos_lambda)

    # 7) Optionally resume training from checkpoint
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

    # 8) Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        # If using a DistributedSampler, set epoch for shuffling
        if cfg.distributed.use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        global_step, last_loss = train_one_epoch(
            epoch=epoch, model=mm_diff_model, optimizer=optimizer, train_loader=train_loader,
            cfg=cfg, vae=vae, global_step=global_step, base_save_path=main_save_dir,
            device = device, ema = ema, scheduler = scheduler
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
            use_ddp=cfg.distributed.use_ddp,
            ema_obj=ema
        )

    # 10) Cleanup
    if cfg.distributed.use_ddp:
        cleanup_distributed()

    if is_main_process():
        print("Training complete!")
    monitor.close()

