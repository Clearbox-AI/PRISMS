import os
import json
import cv2
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Dict
from monai.transforms import (
    Compose, RandFlip, RandRotate, RandZoom,
    RandGaussianNoise, RandBiasField, RandAdjustContrast
)
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
from pathlib import Path

from data.base_dataset import BaseNaccDataset
from enums.data import ImageRange

import numpy as np
import torch
from monai.transforms import (
    Compose, RandBiasField, RandAdjustContrast,
    RandGaussianNoise, RandRicianNoise
)
import torchio as tio
import pandas as pd


from data.tabular_transforms import FittedTransforms, forward_transform, fit_on_dataframe
class NaccDataset(BaseNaccDataset):
    """
    Specialized dataset for NACC data under the new naming convention:
      Each patient folder contains exactly 3 files:
        - 'image.npy' for the image slice
        - 'tabular.json' for tabular patient data
        - 'metadata.json' for additional metadata
    """

    def __init__(self,
                 data_dir: str,
                 image_height: int = 512,
                 image_width: int = 512,
                 domain: str = "mri",
                 do_augment: bool = False,
                 do_image_normalize: bool = True,
                 do_tabular_normalize: bool = True,
                 target_channels: int = 3,
                 final_image_range: ImageRange = "none",
                 debug: bool = False,
                 stats_file: Optional[Union[Path, str]] = None,  # path to JSON with precomputed stats
                 meta_json: Optional[Union[Path,str]] = None,
                 tab_ft_path: Optional[Union[Path, str]] = None,
                 regen_tab_ft: bool = False,
                 ):
        super().__init__(data_dir=data_dir, debug=debug)

        self.image_h, self.image_w = image_height, image_width
        self.domain = domain.lower()
        self.do_augment = do_augment
        self.do_image_norm = do_image_normalize
        self.target_ch = target_channels
        self.final_range = final_image_range
        self.debug = debug

        # ------------ image global stats -------------------------------
        self.image_mean, self.image_std = self._load_img_stats(stats_file)

        # ------------ tabular transforms -------------------------------
        self.feature_names: List[str]
        self.ft_path = Path(tab_ft_path) if tab_ft_path else None
        if self.ft_path and self.ft_path.is_file() and not regen_tab_ft:
            self.ft = FittedTransforms.load(self.ft_path)
            self.feature_names = self.ft.feature_list
        else:
            self.ft = self._fit_tabular(Path(meta_json))
            if self.ft_path:
                self.ft_path.parent.mkdir(parents=True, exist_ok=True)
                self.ft.dump(self.ft_path)
            self.feature_names = self.ft.feature_list

        # ------------ augmentation pipeline ----------------------------
        self.image_transform = self._make_augment_pipeline() if do_augment else None

        # ------------ preload labels ----------------------------
        self.labels = []
        for pdir in self.patient_dirs:
            meta_path = os.path.join(pdir, "metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.labels.append(meta.get("GROUP", "CN"))
            else:
                self.labels.append(None)

    def _load_img_stats(self, stats_file):
        if stats_file and Path(stats_file).is_file():
            with open(stats_file) as f:
                js = json.load(f)
            return js["image_mean"], js["image_std"]
        return 0.0, 1.0

    # ===================================================================
    #                           TABULAR FIT
    # ===================================================================
    def _parse_meta(self, meta_path: Path) -> List[Dict]:
        """
        Accepts both:
          – a list  [ {...}, {...} ]
          – a dict  {"features": [ {...}, ... ] }
        and always returns the *list of feature entries*.
        """
        raw = json.loads(meta_path.read_text())
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "features" in raw:
            return raw["features"]
        raise ValueError(f"Un-recognised meta json structure in {meta_path}")

    def _fit_tabular(self, meta_path: Path) -> FittedTransforms:
        meta_entries = self._parse_meta(meta_path)
        self.feature_names = [e["feature"] for e in meta_entries]

        # ---- build DataFrame --------------------------------------------
        rows = []
        for pdir in self.patient_dirs:
            with open(Path(pdir) / "tabular.json") as f:
                lst = list(json.load(f).values())[0]
            rows.append(dict(zip(self.feature_names, lst)))
        df = pd.DataFrame(rows)

        ft = fit_on_dataframe(df, meta_entries)

        return ft

    # ===================================================================
    #                           AUGMENTATION
    # ===================================================================
    def _make_augment_pipeline(self):
        intensity_aug = Compose([
            RandBiasField(prob=0.10, coeff_range=(0.0, 0.2), degree=3),
            RandAdjustContrast(prob=0.10, gamma=(0.85, 1.15)),
            RandGaussianNoise(prob=0.05, mean=0., std=0.01),
            RandRicianNoise(prob=0.05, std=0.04, channel_wise=False,
                            relative=False, sample_std=True)
        ])
        artifact_aug = tio.Compose([
            tio.RandomGhosting(p=0.05, num_ghosts=(2, 5)),
            tio.RandomSpike(p=0.05, num_spikes=(1, 2)),
            tio.RandomMotion(p=0.05, degrees=3, translation=3)])

        def _aug(img):
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img).float()
            img = intensity_aug(img)
            img = artifact_aug(tio.ScalarImage(tensor=img.unsqueeze(1))).data
            return img.squeeze(1)

        return _aug

    def __getitem__(self, idx):
        # Patient directory (one subfolder per patient)
        pdir = self._get_patient_dir(idx)

        # -------- image ----------------------------------------------
        img = np.load(os.path.join(pdir, "image.npy")).astype(np.float32)
        img = self._domain_specific_normalization(img)
        img = self._resize_or_pad(img, self.image_h, self.image_w)
        img = self._set_num_channels(img, self.target_ch)
        if self.do_image_norm:
            img = (img - self.image_mean) / (self.image_std + 1e-7)
        img = self._map_final_range(img, self.final_range)
        if self.image_transform:
            img = self.image_transform(img)
        img_tensor = torch.as_tensor(img, dtype=torch.float32)

        # debug display (once per run)
        if self.debug and not type(self)._debug_shown_global:
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                self._debug_show_image(img_tensor)
            type(self)._debug_shown_global = True

        # -------- tabular --------------------------------------------

        with open(os.path.join(pdir, "tabular.json")) as f:
            jdata = json.load(f)
        raw = jdata.get("patient_id", list(jdata.values())[0])
        raw = np.array(raw, dtype=np.float32)
        # raw = np.where(raw >= 9999, -1, raw) # sentinel replacement
        tab = forward_transform(self.ft, raw)
        tab_tensor = torch.from_numpy(tab)

        # -------- metadata -------------------------------------------
        meta = {}
        mpath = os.path.join(pdir, "metadata.json")
        if os.path.isfile(mpath):
            meta = json.loads(Path(mpath).read_text())

        return {"image": img_tensor, "tabular": tab_tensor, "metadata": meta, "dir": pdir}


    def _patient_dirs(self):
        """Yield every sub-folder under data_dir (once)."""
        for root, dirs, _ in os.walk(self.data_dir):
            for d in dirs:
                yield os.path.join(root, d)


    def _domain_specific_normalization(self, image: np.ndarray) -> np.ndarray:
        if self.domain == "ct":
            # For example: image = np.clip(image, 0, 2048) / 2048.0
            pass
        elif self.domain == "mri":
            # For example: _clip_outliers(image)
            pass
        return image

    def _resize_or_pad(self, img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        H, W = img.shape[:2]
        if H > new_h or W > new_w:
            # If bigger, resize down
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Otherwise pad, then ensure final size
            delta_h = new_h - H
            delta_w = new_w - W
            pad_top = delta_h // 2
            pad_bottom = delta_h - pad_top
            pad_left = delta_w // 2
            pad_right = delta_w - pad_left
            img = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
            if img.shape[0] != new_h or img.shape[1] != new_w:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def _set_num_channels(self, img: np.ndarray, target_channels: int) -> np.ndarray:
        if len(img.shape) == 2:  # [H,W]
            img = img[None, ...]  # => [1,H,W]
        C, H, W = img.shape
        if C == target_channels:
            return img
        elif C > target_channels:
            return img[:target_channels, :, :]
        else:
            # Expand channels by repeating
            repeats = target_channels // C
            remainder = target_channels % C
            out = np.concatenate([img] * repeats, axis=0)
            if remainder:
                out = np.concatenate([out, img[:remainder, :, :]], axis=0)
            return out

    def _map_final_range(self, image: np.ndarray, frange: ImageRange) -> np.ndarray:
        if frange == ImageRange.plus0to1:
            return np.clip(image, 0.0, 1.0)
        elif frange == ImageRange.minus1to1:
            mn, mx = image.min(), image.max()
            denom = max(mx - mn, 1e-7)
            image = (image - mn) / denom  # => [0,1]
            image = image * 2.0 - 1.0     # => [-1,1]
            return image
        else:
            return image



# ----- MONAI: intensity & scanner-noise transforms ----------
intensity_aug = Compose([
    # Smooth coil-bias / B1 inhomogeneity  -------------------
    RandBiasField(
        prob=0.30,              # ≤30 % of images
        coeff_range=(0.0, 0.4), # up to 40 % roll-off
        degree=3
    ),  # MONAI docs

    # Gamma/contrast modulation ------------------------------
    RandAdjustContrast(
        prob=0.30,
        gamma=(0.8, 1.25)       # pixel^γ, γ∈[0.8,1.25]
    ),  # MONAI docs

    # Gaussian and Rician noise ------------------------------
    RandGaussianNoise(
        prob=0.25,
        mean=0.0,
        std=0.01                # low SNR slice simulation
    ),  # discussion

    RandRicianNoise(
        prob=0.25,
        std=(0.01, 0.05),       # 1–5 % of max signal
        relative=False
    )   # API reference
])

# ----- TorchIO: k-space artefact transforms -----------------
artifact_aug = tio.Compose([
    # tio.RandomMotion(
    #     p=0.20,
    #     degrees=10,             # random rigid motion ≤10°
    #     translation=10          # translations ≤10 mm
    # ),  # Shaw et al. 2019

    tio.RandomGhosting(
        p=0.15,
        num_ghosts=(2, 6)
    ),  # TorchIO docs

    tio.RandomSpike(
        p=0.10,
        num_spikes=(1, 3)
    )   # TorchIO docs
])

def augment_mri_slice(np_slice: np.ndarray) -> np.ndarray:
    """
    Args
    ----
    np_slice : ndarray, shape [H, W]
        Single axial/orthogonal slice after your atlas normalisation.

    Returns
    -------
    aug_slice : ndarray, shape [H, W]
        Augmented slice, same dtype as input.
    """
    # 1) MONAI pipeline operates on torch tensor [1, H, W]
    tensor = torch.from_numpy(np_slice).float().unsqueeze(0)
    tensor = intensity_aug(tensor)

    # 2) TorchIO expects 3-D volumes (C, X, Y, Z).
    #    Treat the slice as a 1-voxel-deep volume.
    tio_img = tio.ScalarImage(tensor=tensor.unsqueeze(0))  # shape (1,1,H,W)
    tio_img = artifact_aug(tio_img)

    # 3) Back to numpy [H, W]
    aug_slice = tio_img.data.squeeze().numpy()
    return aug_slice