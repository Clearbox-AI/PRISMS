import hydra
from omegaconf import DictConfig, OmegaConf

from trainers.trainer_multimodal import train_model

@hydra.main(version_base=None, config_path="../configs/trainers", config_name="base_dit_training")
def main(cfg: DictConfig) -> None:
    print("Full config:\n", OmegaConf.to_yaml(cfg))
    train_model(cfg)

if __name__ == "__main__":
    main()
