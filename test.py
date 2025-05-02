from data.loader import load_training_data
from hydra import compose, initialize_config_dir
import os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from utils.configurations import set_project_root
set_project_root()

with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
    cfg = compose(config_name="base_dit_training")  # Adjust if needed
    OmegaConf.set_struct(cfg, False)

train_loader = load_training_data(cfg)
for batch_idx, batch in enumerate(train_loader):
    # 1) Get data
    images = batch['image'].to("cpu", non_blocking=True)
    tab_data = batch['tabular'].to("cpu", non_blocking=True)