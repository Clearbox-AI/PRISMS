import os
import sys 
import numpy as np

from tabular_images_metrics import Metrics
from data.loader import load_training_data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.configurations import set_project_root
from pathlib import Path

# from coherence.similarity_score import Similarity
from metrics.coherence.discriminator_score import Discriminator, make_incoherent_loader
from metrics.coherence.similarity_score import Similarity

if __name__ == "__main__":

    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc")
        OmegaConf.set_struct(cfg, False)
    train_loader, val_loader = load_training_data(cfg)

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc_synth")
        OmegaConf.set_struct(cfg, False)
    synth_loader = load_training_data(cfg, real=False)

    # with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
    #     # Override 'data.shuffle' prima ancora di restituire cfg
    #     cfg = compose(
    #         config_name="nacc",
    #         overrides=["data.shuffle=true"]
    #     )
    # shuffled_loader, _ = load_training_data(cfg)
    val_incoh_loader = make_incoherent_loader(val_loader)

    class_names = np.array(["CN", "AD"])
    # Classical images and tabular metrics
    train_label = (np.take(train_loader.dataset.dataset.labels, train_loader.dataset.indices) == "AD").astype(int)
    valid_label = (np.take(val_loader.dataset.dataset.labels, val_loader.dataset.indices) == "AD").astype(int)
    synth_label = np.array([0 if s == "CN" else 1 for s in synth_loader.dataset.labels], dtype=int)

    from data.tabular_transforms import FittedTransforms
    ft = FittedTransforms.load(Path("/home/PRISMS/data/computations/tab_ft.pkl"))

    metrics_manager = Metrics(train_loader, synth_loader, val_loader, ft)
    img_metrics = metrics_manager.images()
    tab_metrics = metrics_manager.tabular(train_label, synth_label, valid_label)
    print(f"IMG METRICS:\n{img_metrics}")
    print(f"TAB METRICS:\n{tab_metrics}")
    # metrics_manager.tab_report()

    # Coherence metrics
    train_flag = False
    checkpoint_path = 'PRISMS/metrics/coherence/checkpoints/'
    discriminator_weights = 'discriminator_models_100epochs.pth'
    similarity_weights = 'similarity_models_10epochs.pth'

    checkpoint_discriminator_weights = "/home/PRISMS/metrics/coherence/checkpoints/discriminator_models_100epochs.pth"
    checkpoint_similarity_weights = "/home/PRISMS/metrics/coherence/checkpoints/similarity_models_10epochs.pth"

    # Discriminator score
    discriminator = Discriminator(embed_dim=128, tabular_dim=val_loader.dataset[0]["tabular"].numel())

    disc_train_auc  = discriminator.evaluate(train_loader, checkpoint_discriminator_weights, train_flag=True)
    disc_val_coh = discriminator.evaluate(val_loader, checkpoint_discriminator_weights, train_flag=False)
    disc_val_incoh = discriminator.evaluate(val_incoh_loader, checkpoint_discriminator_weights, train_flag=False)

    print(f"\nDiscriminator AUCs\n"
          f"train : {disc_train_auc:.3f}\n"
          f"val-coherent : {disc_val_coh:.3f}\n"
          f"val-incoherent : {disc_val_incoh:.3f}")

    # Similarity score
    dim    = 128
    epochs = 10

    similarity = Similarity(dim=128, lr=1e-4)

    sim_train = similarity.evaluate(train_loader, checkpoint_similarity_weights, train_flag=True)  # fit
    sim_val = similarity.evaluate(val_loader, checkpoint_similarity_weights, train_flag=False)  # score
    sim_incoh = similarity.evaluate(val_incoh_loader, checkpoint_similarity_weights, train_flag=False)

    print(f"\nSimilarity (lower is better?)\n"
          f"train : {sim_train:.3f}\n"
          f"val-coherent : {sim_val:.3f}\n"
          f"val-incoherent : {sim_incoh:.3f}")