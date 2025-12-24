"""End-to-end synthetic dataset evaluation entry point.

This script orchestrates:
- Classic tabular and image metrics via `TabularImageMetrics`
- Cross-modal coherence metrics via `Discriminator` and `Similarity`

The original prototype hard-coded paths and config names; this refactor exposes
them as CLI arguments while keeping the default behavior intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a multimodal synthetic dataset.")
    p.add_argument("--real-config", default="nacc", help="Hydra dataset config name for the real dataset.")
    p.add_argument("--synth-config", default="nacc_synth", help="Hydra dataset config name for the synthetic dataset.")
    p.add_argument("--configs-dir", default=None, help="Path to the Hydra dataset configs directory.")
    p.add_argument("--ft-path", default="/home/PRISMS/data/computations/tab_ft.pkl", help="Path to a serialized FittedTransforms (pkl).")
    p.add_argument("--report-dir", default="/home/PRISMS/data/computations/sure_report", help="Output directory for SURE JSON/HTML.")
    p.add_argument("--disc-ckpt", default="/home/PRISMS/metrics/coherence/checkpoints/discriminator_models_100epochs.pth", help="Path to discriminator checkpoint.")
    p.add_argument("--sim-ckpt", default="/home/PRISMS/metrics/coherence/checkpoints/similarity_models_10epochs.pth", help="Path to similarity checkpoint.")
    p.add_argument("--train-coherence", action="store_true", help="Train coherence models instead of loading.")
    p.add_argument("--disc-epochs", type=int, default=100, help="Training epochs for the discriminator.")
    p.add_argument("--sim-epochs", type=int, default=10, help="Training epochs for the similarity model.")
    return p.parse_args()


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args()

    # Local imports to keep module import side effects minimal.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from utils.configurations import set_project_root
    from data.loader import load_training_data
    from data.tabular_transforms import FittedTransforms

    from metrics.tabular_images_metrics import Metrics
    from metrics.coherence.discriminator_score import Discriminator, make_incoherent_loader
    from metrics.coherence.similarity_score import Similarity

    set_project_root()

    # Resolve config directory: default to $PROJECT_ROOT/configs/datasets
    if args.configs_dir is not None:
        cfg_dir = Path(args.configs_dir)
    else:
        import os
        cfg_dir = Path(os.environ["PROJECT_ROOT"], "configs", "datasets")

    with initialize_config_dir(config_dir=str(cfg_dir)):
        cfg_real = compose(config_name=args.real_config)
        OmegaConf.set_struct(cfg_real, False)
    train_loader, val_loader = load_training_data(cfg_real)

    with initialize_config_dir(config_dir=str(cfg_dir)):
        cfg_synth = compose(config_name=args.synth_config)
        OmegaConf.set_struct(cfg_synth, False)
    synth_loader = load_training_data(cfg_synth, real=False)

    # Fit transforms for inverse tabular transform (optional but recommended).
    ft = None
    if args.ft_path is not None:
        ft = FittedTransforms.load(Path(args.ft_path))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build incoherent validation loader.
    val_incoh_loader = make_incoherent_loader(val_loader)

    # Labels for utility metrics (dataset-specific).
    train_label = (np.take(train_loader.dataset.dataset.labels, train_loader.dataset.indices) == "AD").astype(int)
    valid_label = (np.take(val_loader.dataset.dataset.labels, val_loader.dataset.indices) == "AD").astype(int)
    synth_label = np.array([0 if s == "CN" else 1 for s in synth_loader.dataset.labels], dtype=int)

    metrics_manager = Metrics(train_loader, synth_loader, val_loader, ft)
    _ = metrics_manager.images()
    _ = metrics_manager.tabular(train_label, synth_label, valid_label, out_dir=str(report_dir))
    metrics_manager.tab_report(out_dir=str(report_dir), train_label=train_label, synth_label=synth_label, valid_label=valid_label)

    # -------- Coherence metrics -------------------------------------------
    # Discriminator
    discriminator = Discriminator(embed_dim=128, tabular_dim=val_loader.dataset[0]["tabular"].numel())

    disc_train_auc = discriminator.evaluate(train_loader, args.disc_ckpt, train_flag=args.train_coherence, epochs=args.disc_epochs)
    disc_val_coh = discriminator.evaluate(val_loader, args.disc_ckpt, train_flag=False)
    disc_val_incoh = discriminator.evaluate(val_incoh_loader, args.disc_ckpt, train_flag=False)

    print(
        f"\nDiscriminator AUCs\n"
        f"train : {disc_train_auc:.3f}\n"
        f"val-coherent : {disc_val_coh:.3f}\n"
        f"val-incoherent : {disc_val_incoh:.3f}"
    )

    # Similarity
    similarity = Similarity(dim=128, lr=1e-4)

    sim_train = similarity.evaluate(train_loader, args.sim_ckpt, train_flag=args.train_coherence, epochs=args.sim_epochs)
    sim_val = similarity.evaluate(val_loader, args.sim_ckpt, train_flag=False)
    sim_incoh = similarity.evaluate(val_incoh_loader, args.sim_ckpt, train_flag=False)

    print(
        f"\nSimilarity (lower is better)\n"
        f"train : {sim_train:.3f}\n"
        f"val-coherent : {sim_val:.3f}\n"
        f"val-incoherent : {sim_incoh:.3f}"
    )


if __name__ == "__main__":
    main()
