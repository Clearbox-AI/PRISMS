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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) ---------- load models ------------------------------------------- #
    vae = load_model(ModelType.VAE, cfg=cfg).to(device).eval()

    diff = load_model(
        ModelType.DIFFUSION,
        cfg=cfg
    ).to(device).eval()

    _, _ = resume_from_checkpoint(
        resume_dir=cfg.training.resume_checkpoint_dir,
        model=diff,
        optimizer=None,
        device=device,
        use_ddp=cfg.distributed.use_ddp
    )

    # 2) ── load the *training* StandardScaler ────────────────────────────────
    stats_path = Path(cfg.data_synth.stats_file)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"stats_file not found: {stats_path}\n"
            "Set cfg.data_synth.stats_file to the JSON exported during training."
        )

    with open(stats_path, "r") as f:
        stats = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(stats["tabular_scaler_mean_"], dtype=np.float32)
    scaler.scale_ = np.asarray(stats["tabular_scaler_scale_"], dtype=np.float32)
    scaler.n_features_in_ = len(scaler.mean_)

    # 2) ---------- prepare output dir ------------------------------------ #
    synth_base = Path(cfg.data_synth.data_dir)
    run_dir = get_next_run_dir(synth_base, cfg.data_synth.get("save_label"))
    print(f"[INFO] Writing samples to {run_dir}")

    # 4) ── sampling loop ─────────────────────────────────────────────────────
    total = cfg.data_synth.num_samples
    bs = cfg.data_synth.batch_size
    n_batches = math.ceil(total / bs)

    sample_idx = 0
    for _ in tqdm(range(n_batches), desc="Generating", unit="batch"):
        cur_bs = min(bs, total - sample_idx)

        # 4.1) Sample *latent* image + tabular from diffusion
        lat_img, lat_tab = diff.sample(
            batch_size=cur_bs,
            guidance_scale=cfg.data_synth.guidance_scale,
            num_steps=cfg.data_synth.sample_steps,
        )

        # 4.2) Decode image latents back to pixel space
        recon_imgs = decode_latents(vae, lat_img, cfg.vae.scaling_factor)

        # 4.3) Inverse‑transform tabular Z‑scores → original units
        lat_tab_np = lat_tab.cpu().numpy()  # shape (B, F)
        tab_denorm = scaler.inverse_transform(lat_tab_np)

        # OPTIONAL: restore sentinel *9999* (training converted >9999 → -1)
        # tab_denorm = np.where(tab_denorm < 0, 9999, tab_denorm)

        # 4.4) Write each sample to its own folder
        for b in range(cur_bs):
            sid = f"sample_{sample_idx:06d}"
            sd = run_dir / sid
            sd.mkdir()

            # image
            np.save(sd / "image.npy", recon_imgs[b].cpu().numpy())

            # tabular (now de‑normalised!)
            with open(sd / "tabular.json", "w") as f:
                json.dump(tab_denorm[b].tolist(), f)

            sample_idx += 1

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
