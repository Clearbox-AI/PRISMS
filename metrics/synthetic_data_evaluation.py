import os
import sys 
import numpy as np

from tabular_images_metrics import Metrics
from data.loader import load_training_data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.configurations import set_project_root
from pathlib import Path

# from coherence.discriminator_score import create_shuffled_tabular_loader, Discriminator
# from coherence.similarity_score import Similarity
from metrics.coherence.discriminator_score import create_shuffled_tabular_loader, Discriminator
from metrics.coherence.similarity_score import Similarity

if __name__ == "__main__":

    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc")
        OmegaConf.set_struct(cfg, False)
    train_loader = load_training_data(cfg)
    # val_loader = load_training_data(cfg)
    val_loader = None


    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc_synth")
        OmegaConf.set_struct(cfg, False)
    synth_loader = load_training_data(cfg, real=False)
    # shuffled_loader = create_shuffled_tabular_loader(train_loader)

    # Classincal images and tabular metrics
    # train_label = np.random.randint(0, 2, size=len(next(iter(train_loader))["tabular"]))
    # synth_label = np.random.randint(0, 2, size=len(next(iter(synth_loader))["tabular"]))
    train_label = np.fromiter((sample["metadata"].get("GROUP") == "AD" for sample in train_loader.dataset),
                              dtype=np.int64)
    synth_label = None
    # valid_label = np.random.randint(0, 2, size=len(next(iter(val_loader))["tabular"]))
    valid_label = None

    metrics_manager = Metrics(train_loader, synth_loader, val_loader)
    img_metrics = metrics_manager.images()
    tab_metrics = metrics_manager.tabular(train_label, synth_label, valid_label)

    x=4
    # metrics_manager.tab_report()
    #
    # # Coherence metrics
    # train_flag            = False
    # checkpoint_path       = 'PRISMS/metrics/coherence/checkpoints/'
    # discriminator_weights = 'discriminator_models_100epochs.pth'
    # similarity_weights    = 'similarity_models_10epochs.pth'
    # checkpoint_discriminator_weights = os.path.join(checkpoint_path, discriminator_weights)
    # checkpoint_similarity_weights    = os.path.join(checkpoint_path, similarity_weights)
    #
    # # Discriminator score
    # discriminator = Discriminator()
    # discriminator_scores_train = discriminator.evaluate(train_loader, checkpoint_discriminator_weights, train_flag)
    # discriminator_scores_synth = discriminator.evaluate(shuffled_loader, checkpoint_discriminator_weights, train_flag)
    #
    # print('\nDiscriminator Scores')
    # print(f"Original data:  {discriminator_scores_train}")
    # print(f"Synthetic data: {discriminator_scores_synth}\n")
    #
    # # Similarity score
    # dim    = 128
    # epochs = 10
    #
    # train_loader = load_training_data(cfg)
    # shuffled_loader = create_shuffled_tabular_loader(train_loader)
    #
    # similarity = Similarity(dim=128, lr=1e-4)
    # similarity_score_train = similarity.evaluate(loader = train_loader,
    #                                             checkpoint_path = checkpoint_similarity_weights,
    #                                             train_flag = train_flag)
    #
    # similarity_score_synth = similarity.evaluate(loader = shuffled_loader,
    #                                             checkpoint_path = checkpoint_similarity_weights,
    #                                             train_flag = False)
    #
    # print('\nSimilarity Scores')
    # print(f"Original data:  {similarity_score_train:.3f}")
    # print(f"Synthetic data: {similarity_score_synth:.3f}")