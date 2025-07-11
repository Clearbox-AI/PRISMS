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
        use_ddp=cfg.distributed.use_ddp
    )

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

        half = cur_bs // 2
        labels = torch.cat([
            torch.zeros(half, dtype=torch.long),
            torch.ones(cur_bs - half, dtype=torch.long)
        ], dim=0)
        perm = torch.randperm(cur_bs)
        labels = labels[perm]
        labels = labels.to(device, non_blocking=True)

        # 4.1) Sample *latent* image + tabular from diffusion
        lat_img, lat_tab = diff.sample(
            batch_size=cur_bs,
            guidance_scale=cfg.data_synth.guidance_scale,
            num_steps=cfg.data_synth.sample_steps,
            labels=labels
        )

        # 4.2) Decode image latents back to pixel space
        recon_imgs = decode_latents(vae, lat_img, cfg.vae.scaling_factor)

        lat_tab_np = lat_tab.cpu().numpy()  # shape (B, F)

        # 4.4) Write each sample to its own folder
        for b in range(cur_bs):
            sid = f"sample_{sample_idx:06d}"
            sd = run_dir / sid
            sd.mkdir()

            # image
            np.save(sd / "image.npy", recon_imgs[b].cpu().numpy())

            # tabular (now de‑normalised!)
            with open(sd / "tabular.json", "w") as f:
                json.dump(lat_tab_np[b].tolist(), f)

            # labels
            label_val = labels[b].item()
            group_orig = "CN" if label_val == 0 else "AD"
            metadata = {
                "GROUP": group_orig,
                "GROUP_ENC": label_val
            }
            with open(sd / "metadata.json", "w") as f:
                json.dump(metadata, f)

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
