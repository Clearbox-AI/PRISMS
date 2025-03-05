import os
import json
import cv2
import numpy as np
import torch
import torch.distributed as dist
from monai.transforms import (
    Compose, RandFlip, RandRotate, RandZoom,
    RandGaussianNoise, RandBiasField, RandAdjustContrast
)
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
from pathlib import Path

from data.base_dataset import BaseNaccDataset
from enums.data import ImageRange

class NaccDataset(BaseNaccDataset):
    """
    Specialized dataset for NACC data.
    Loads .npy image slices and .json tabular data.
    Reads precomputed normalization stats from a provided stats file.
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
                 stats_file: Optional[Union[Path, str]] = None  # path to JSON with precomputed stats
                 ):
        super().__init__(data_dir=data_dir, debug=debug)

        self.image_height = image_height
        self.image_width = image_width
        self.domain = domain.lower()
        self.do_augment = do_augment
        self.do_image_normalize = do_image_normalize
        self.do_tabular_normalize = do_tabular_normalize
        self.target_channels = target_channels
        self.final_image_range = final_image_range

        # Initialize placeholders for normalization stats
        self.image_mean = 0.0
        self.image_std = 1.0
        self.tabular_scaler = StandardScaler()

        # Load the precomputed stats from JSON (if provided)
        if stats_file is not None and os.path.isfile(stats_file):
            self._load_stats_from_file(stats_file)

        # Set up augmentation pipeline if requested
        if self.do_augment:
            self.image_transform = Compose([
                RandFlip(prob=0.5, spatial_axis=0),
                RandRotate(range_x=15, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                RandGaussianNoise(prob=0.3, std=0.05),
                RandBiasField(degree=4, prob=0.3),
                RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3))
            ])
        else:
            self.image_transform = None

    def __getitem__(self, idx):
        patient_dir = self._get_patient_dir(idx)

        # 1) Load image
        image_path = self._get_first_file(patient_dir, "*.npy")
        image = np.load(image_path).astype(np.float32)  # shape [H, W]

        # 2) Domain-specific normalization (mock)
        image = self._domain_specific_normalization(image)

        # 3) Resize/pad
        image = self._resize_or_pad(image, self.image_height, self.image_width)

        # 4) Convert to correct channels
        image = self._set_num_channels(image, self.target_channels)

        # 5) Global mean/std normalization
        if self.do_image_normalize:
            image = (image - self.image_mean) / (self.image_std + 1e-7)

        # 6) Final range mapping
        image = self._map_final_range(image, self.final_image_range)

        # 7) Augment (if any transforms specified)
        if self.do_augment and self.image_transform is not None:
            image = self.image_transform(image)  # MONAI channel-first array

        # Convert to torch Tensor
        if not isinstance(image, torch.Tensor):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image.float()

        # Debug show
        if self.debug and not type(self)._debug_shown_global:
            rank = 0
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            if rank == 0:
                self._debug_show_image(image_tensor)
            type(self)._debug_shown_global = True

        # 8) Load tabular
        json_path = self._get_first_file(patient_dir, "*.json")
        with open(json_path, 'r') as f:
            jdata = json.load(f)
        tab = jdata.get("patient_id", list(jdata.values())[0])
        tab = np.array(tab, dtype=np.float32)
        tab = np.where(tab > 9999, -1, tab)  # sentinel replacement
        tab = tab.reshape(1, -1)

        if self.do_tabular_normalize:
            tab = self.tabular_scaler.transform(tab)

        tab_tensor = torch.from_numpy(tab).squeeze(0)

        return {
            "image": image_tensor,
            "tabular": tab_tensor,
            "dir": patient_dir
        }

    # -------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------
    def _load_stats_from_file(self, stats_file: str):
        """
        Load the precomputed stats from a JSON file.
        Expecting structure like:
        {
            "image_mean": ...,
            "image_std": ...,
            "tabular_scaler_mean_": [...],
            "tabular_scaler_scale_": [...]
        }
        """
        with open(stats_file, "r") as f:
            stats = json.load(f)

        # Image stats
        self.image_mean = stats.get("image_mean", 0.0)
        self.image_std = stats.get("image_std", 1.0)

        # Tabular scaler stats
        mean_ = stats.get("tabular_scaler_mean_", None)
        scale_ = stats.get("tabular_scaler_scale_", None)
        if mean_ is not None and scale_ is not None:
            # Manually set StandardScaler attributes
            self.tabular_scaler.mean_ = np.array(mean_)
            self.tabular_scaler.scale_ = np.array(scale_)
            self.tabular_scaler.n_features_in_ = len(mean_)
        else:
            pass

    def _domain_specific_normalization(self, image: np.ndarray) -> np.ndarray:
        # Example placeholder
        if self.domain == "ct":
            # image = np.clip(image, 0, 2048) / 2048.0
            pass
        elif self.domain == "mri":
            # e.g., _clip_outliers(image)
            pass
        return image

    def _resize_or_pad(self, img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        H, W = img.shape[:2]
        if H > new_h or W > new_w:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            delta_h = new_h - H
            delta_w = new_w - W
            pad_top = delta_h // 2
            pad_bottom = delta_h - pad_top
            pad_left = delta_w // 2
            pad_right = delta_w - pad_left
            img = np.pad(img,
                         ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=0)
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

        # C < target_channels
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
