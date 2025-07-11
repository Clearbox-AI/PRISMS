import json
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class NaccSynthDataset(Dataset):
    """
    Minimal loader for *synthetically generated* (image, tabular) pairs.

    Each sample lives in its own sub-directory and contains exactly:
      • image.npy      – float32, shape (C, H, W)
      • tabular.json   – list *or* dict of floats (length F)

    No normalisation, augmentation or VAE decoding is done here – the tensors
    are returned exactly as saved by the sampling function.
    """

    _debug_shown_global = False

    def __init__(
        self,
        data_dir: Union[str, Path],
        debug: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.debug = debug

        subdirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
        self.sample_dirs: List[Path] = subdirs
        if not self.sample_dirs:
            raise ValueError(f"No samples found in {self.data_dir}")

        # preload GROUP labels from metadata.json
        self.labels: List[Union[int, None]] = []
        self.labels_enc: List[Union[int, None]] = []
        for pdir in self.sample_dirs:
            meta_path = pdir / "metadata.json"
            if meta_path.is_file():
                with open(meta_path, "r") as mf:
                    meta = json.load(mf)
                # use GROUP numeric value; default to None if missing
                self.labels.append(meta.get("GROUP", "CN"))
                self.labels_enc.append(meta.get("GROUP_ENC", 0))
            else:
                self.labels.append(None)
                self.labels_enc.append(None)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]

        # ---------- image ---------------------------------------- #
        img_path = sample_dir / "image.npy"
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        img = np.load(img_path).astype(np.float32)          # (C,H,W)

        # take the middle slice
        if img.ndim == 3:
            # assume first axis is depth
            mid = img.shape[0] // 2
            img = img[mid]  # now (H,W)
            img = img[np.newaxis, ...]  # add channel dim: (1,H,W)

        img_tensor = torch.from_numpy(img)

        # ---------- tabular -------------------------------------- #
        tab_path = sample_dir / "tabular.json"
        if not tab_path.is_file():
            raise FileNotFoundError(tab_path)
        with open(tab_path, "r") as f:
            j = json.load(f)
        tab = np.array(list(j.values()) if isinstance(j, dict) else j,
                       dtype=np.float32)
        tab_tensor = torch.from_numpy(tab)

        # ---------- metadata ------------------------------------- #
        meta_path = sample_dir / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # ---------- optional first-sample debug ------------------ #
        if self.debug and not type(self)._debug_shown_global:
            self._show_debug(img_tensor)
            type(self)._debug_shown_global = True

        return {
            "image":   img_tensor,          # torch.float32  (C,H,W)
            "tabular": tab_tensor,          # torch.float32  (F,)
            "metadata": metadata,  # dict with GROUP and GROUP_ORIG
            "dir":     str(sample_dir),
        }

    # ------------------------------------------------------------ #
    # utilities                                                    #
    # ------------------------------------------------------------ #
    @staticmethod
    def _show_debug(t: torch.Tensor):
        c, h, w = t.shape
        npimg = t.numpy()
        if c == 1:
            plt.imshow(npimg[0], cmap="gray")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title("NaccSynthDataset – first sample")
        plt.axis("off")
        plt.show(block=True)
