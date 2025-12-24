import torch
import pandas as pd
from torchvision.utils import save_image
from pathlib import Path
from typing import Union, List, Any

from enums.models.diffusion import DataLabel
from torch.utils.data import DataLoader


def save_images(base_save_path: Path, samples: torch.Tensor, label: Union[str, int]):
    samples_path = Path(base_save_path, "samples", f"samples_step_{label}.png")
    save_image(samples, samples_path, nrow=2)
    print(f"Saved sample images => {samples_path}")

def save_tabulars(base_save_path: Path, samples: torch.Tensor, label: Union[str, int]):
    samples_csv = Path(base_save_path, "samples", f"tabular_step_{label}.csv")
    pd.DataFrame(samples.cpu().numpy()).to_csv(samples_csv, index=False)
    print(f"Saved sample table => {samples_csv}")


class DataBucket:
    """
    A container for conditional data or for the final generated samples.
    - data_source can be:
        1) A PyTorch DataLoader,
        2) A list of data items (e.g., images, tabular features, or (img, tab) pairs).
    - label: indicates what type of data is in data_source (image, tab, or both).
    """

    def __init__(self, data_source: Union[DataLoader, List[Any]], label: DataLabel):
        self.data_source = data_source
        self.label = label

    def __len__(self):
        if isinstance(self.data_source, DataLoader):
            # length in terms of #batches (not always exact). For “infinite” iteration, this is less relevant.
            return len(self.data_source)
        return len(self.data_source)


def infinite_loader(dataloader: DataLoader):
    """Create a persistent iterator that cycles through a dataloader indefinitely."""
    while True:
        for batch in dataloader:
            yield batch

# ---------------------------------------------------------------------------
# Dynamic Thresholding in *pixel space* (Imagen-style)
# Applicata DOPO la decodifica VAE, non in latente.
# (Compatibile con immagini in range [-1, 1] o [0, 1].)
# ---------------------------------------------------------------------------
def _dynamic_threshold_pixel(
    imgs: torch.Tensor,
    p: float = 0.995,
    rescale: bool = False
) -> torch.Tensor:
    if imgs.ndim != 4:
        return imgs
    B = imgs.shape[0]
    v = imgs.detach().abs().flatten(1)
    q = torch.quantile(
        v, torch.tensor(float(p), device=imgs.device), dim=1, keepdim=True
    ).clamp(min=1e-3)
    clamped = imgs.clamp(-q.view(B, 1, 1, 1), q.view(B, 1, 1, 1))
    return (clamped / q.view(B, 1, 1, 1)) if rescale else clamped
