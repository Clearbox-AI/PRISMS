import torch
import pandas as pd
from torchvision.utils import save_image
from pathlib import Path
from typing import Union


def save_images(base_save_path: Path, samples: torch.Tensor, label: Union[str, int]):
    samples_path = Path(base_save_path, "samples", f"samples_step_{label}.png")
    save_image(samples, samples_path, nrow=2)
    print(f"Saved sample images => {samples_path}")

def save_tabulars(base_save_path: Path, samples: torch.Tensor, label: Union[str, int]):
    samples_csv = Path(base_save_path, "samples", f"tabular_step_{label}.csv")
    pd.DataFrame(samples.cpu().numpy()).to_csv(samples_csv, index=False)
    print(f"Saved sample table => {samples_csv}")