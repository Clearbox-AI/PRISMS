import os, json, math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir

from models.utils.model_loader import load_model
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from models.vae.vae import decode_latents
from datetime import datetime
from tqdm import tqdm
from utils.model import resume_from_checkpoint
from sklearn.preprocessing import StandardScaler
from data.tabular_transforms import inverse_transform

import torch.distributed as dist
from utils.ddp import setup_distributed, cleanup_distributed, is_main_process

# ------------------------------------------------------------------------- #
# small helper to broadcast a Python object (e.g., a string path) in DDP
def _bcast_obj(obj, src=0):
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    lst = [obj]
    dist.broadcast_object_list(lst, src=src)
    return lst[0]

# ------------------------------------------------------------------------- #
# directory helpers                                                         #
# ------------------------------------------------------------------------- #
def get_next_run_dir(base_dir: Path, label: Optional[str] = None) -> Path:
    """
    Create a sub-folder named with the current UTC date-time and an optional label,
    e.g.  2025-05-21_14-03-27_v1
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_name  = f"{timestamp}_{label}" if label else timestamp
    run_path  = base_dir / run_name
    run_path.mkdir(exist_ok=True)
    return run_path


# ------------------------------------------------------------------------- #
# main generation routine                                                   #
# ------------------------------------------------------------------------- #
@torch.no_grad()
def generate(cfg: DictConfig):
    # ---------- DDP setup (optional) ------------------------------------- #
    local_rank = 0
    ddp = bool(cfg.distributed.get("use_ddp", False))
    if ddp:
        local_rank = setup_distributed(cfg)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world = dist.get_world_size() if (ddp and dist.is_initialized()) else 1
    rank = dist.get_rank() if (ddp and dist.is_initialized()) else 0

    # 1) ---------- load models ------------------------------------------- #
    vae = load_model(ModelType.VAE, cfg=cfg).to(device).eval()

    from data.tabular_transforms import FittedTransforms
    ft = FittedTransforms.load(Path("/home/PRISMS/data/computations/tab_ft.pkl"))
    diff = load_model(
        model_type=ModelType.DIFFUSION,
        cfg=cfg,
        tab_transforms=ft
    ).to(device)

    _, _ = resume_from_checkpoint(
        resume_dir=cfg.training.resume_checkpoint_dir,
        model=diff,
        optimizer=None,
        device=device,
        use_ddp=False
    )

    # (A) Ensure deterministic inference: disable training-time stochastic gates/dropout
    diff.eval()
    # (4.e) Optional per-run control of noise selection from synth config
    ns = cfg.data_synth.get("noise_select", None)
    if ns is not None:
        try: diff.noise_select = bool(ns)
        except Exception:
            pass

    # 2) ---------- prepare output dir ------------------------------------ #
    synth_base = Path(cfg.data_synth.data_dir)
    if is_main_process():
        run_dir = get_next_run_dir(synth_base, cfg.data_synth.get("save_label"))
    else:
        run_dir = None
    # broadcast path so all ranks write into the SAME directory
    run_dir = Path(_bcast_obj(str(run_dir) if run_dir is not None else "")) if ddp else run_dir
    if ddp:
        # ensure directory exists on all ranks
        os.makedirs(run_dir, exist_ok=True)
        dist.barrier()
    if is_main_process():
        print(f"[INFO] Writing samples to {run_dir}")

    # 4) ── sampling loop ─────────────────────────────────────────────────────
    total_global = int(cfg.data_synth.num_samples)
    bs = int(cfg.data_synth.batch_size)

    # ---- Divide samples across ranks (global → local) -----------------------
    q, r = divmod(total_global, world)
    total_local = q + (1 if rank < r else 0)
    start_index = rank * q + min(rank, r)  # indice globale di partenza per questo rank
    if is_main_process():
        print(f"[DDP] world={world} | total={total_global} | per-rank≈{q} (+1 per i primi {r})")
    if total_local == 0:
        if is_main_process():
            print("[WARN] Too few samples for the number of ranks; some ranks will be idle.")
        if ddp:
            dist.barrier()
            cleanup_distributed()
        return
    n_batches = math.ceil(total_local / bs)

    produced = 0
    iter_range = range(n_batches)
    if is_main_process():
        iter_range = tqdm(iter_range, desc="Generating", unit="batch")
    for _ in iter_range:
        cur_bs = min(bs, total_local - produced)
        if cur_bs <= 0:
            break

        # Choose how to draw labels:
        balance = bool(cfg.data_synth.get("balance_labels", False))
        if balance:
            # 50/50 per batch
            half = cur_bs // 2
            labels = torch.cat([
                torch.zeros(half, dtype=torch.long),
                torch.ones(cur_bs - half, dtype=torch.long)
            ], dim=0)
        else:
            # draw from training priors if available
            if hasattr(diff, "class_counts") and diff.class_counts is not None:
                n0, n1 = diff.class_counts
                p1 = float(n1) / float(n0 + n1 + 1e-9)
            else:
                p1 = 0.5
            n_pos = int(round(cur_bs * p1))
            labels = torch.cat([
                torch.zeros(cur_bs - n_pos, dtype=torch.long),
                torch.ones(n_pos, dtype=torch.long)
            ], dim=0)
        # shuffle and move
        labels = labels[torch.randperm(cur_bs)].to(device, non_blocking=True)

        # 4.1) Sample *latent* image + tabular from diffusion
        # branch‑aware guidance if provided
        g_img = cfg.data_synth.get("guidance_img", None)
        g_tab = cfg.data_synth.get("guidance_tab", None)
        g_scale = {"img": g_img, "tab": g_tab} if (
                g_img is not None and g_tab is not None) else cfg.data_synth.guidance_scale
        # (4.a) reproducibility: optional global seed for this run
        seed = cfg.data_synth.get("seed", None)
        lat_img, lat_tab = diff.sample(
            batch_size=cur_bs,
            guidance_scale=g_scale,
            num_steps=cfg.data_synth.sample_steps,
            labels=labels,
            seed = seed,
            return_latents = True,
        )

        # 4.2) Decode image latents back to pixel space
        recon_imgs = decode_latents(vae, lat_img, cfg.vae.scaling_factor)

        # (4.c) Decode tab *latents* → model-space → inverse_transform a RAW (con categorie non numeriche)
        if hasattr(diff, "tab_vae") and getattr(diff, "has_tabsyn_vae", lambda: False)():
            x_tab_model = diff.tab_vae.decode_flat(lat_tab).cpu().numpy()
        else:
            x_tab_model = lat_tab.cpu().numpy()
        # keep original numeric types; allow object dtype for categoricals
        raw_np = inverse_transform(ft, x_tab_model, return_object=True)

        # 4.4) Write each sample to its own folder
        for b in range(cur_bs):
            global_idx = start_index + produced + b
            sid = f"sample_{global_idx:06d}"
            sd = run_dir / sid
            sd.mkdir()

            # image
            np.save(sd / "image.npy", recon_imgs[b].cpu().numpy())

            # (4.c) tabular RAW, self‑describing (feature → valore, con tipi corretti)
            names = ft.feature_list
            def _to_py(v):
                # JSON-safe: cast numpy scalars, preserve strings, map NaN -> null
                if isinstance(v, np.generic): v = v.item()
                if isinstance(v, float) and np.isnan(v): return None
                return v

            row = {names[i]: _to_py(raw_np[b, i]) for i in range(len(names))}
            with open(sd / "tabular.json", "w") as f:
                json.dump(row, f, ensure_ascii=False)

            # labels
            label_val = labels[b].item()
            group_orig = "CN" if label_val == 0 else "AD"
            metadata = {
                "GROUP": group_orig,
                "GROUP_ENC": label_val
            }
            with open(sd / "metadata.json", "w") as f:
                json.dump(metadata, f)

        produced += cur_bs

    if ddp:
        dist.barrier()
        cleanup_distributed()

    if is_main_process():
        print("[INFO] Generation complete!")

# ------------------------------------------------------------------------- #
# Hydra entry-point                                                        #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")
        OmegaConf.set_struct(cfg, False)

    generate(cfg)
