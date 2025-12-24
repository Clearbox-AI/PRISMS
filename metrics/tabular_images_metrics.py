"""Tabular and image metrics for synthetic data evaluation.

The implementation follows the original research code semantics:
- Tabular metrics are computed through the `sure` library.
- Image metrics are computed using PIQ (SSIM/MS-SSIM/FID) and a DenseNet121
  feature extractor for FID embeddings.

Notes
-----
This module intentionally keeps the original algorithmic behavior (including
sampling strategy and normalization choices) while improving structure, typing,
and import hygiene.
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from piq import FID, multi_scale_ssim
from monai.networks.nets import densenet121

from sure.privacy import (
    dcr_stats,
    distance_to_closest_record,
    number_of_dcr_equal_to_zero,
    validation_dcr_test,
)
from sure.utility import compute_mutual_info, compute_statistical_metrics, compute_utility_metrics_class

from data.tabular_transforms import FittedTransforms, inverse_transform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for image metric computation."""

    # The original code normalizes images to [0, 1] internally regardless of data_range.
    data_range: float = 1.0
    # Number of (i, j) pairs sampled for SSIM/MS-SSIM estimates.
    num_samples: int = 1000
    # Batch size used to extract DenseNet features for FID.
    fid_batch_size: int = 32


class TabularImageMetrics:
    """Compute metrics for paired image + tabular datasets.

    Parameters
    ----------
    train_loader:
        Dataloader yielding dict batches with keys `image` and `tabular`.
    synth_loader:
        Dataloader yielding synthetic batches with the same structure.
    valid_loader:
        Optional validation dataloader (used for DCR validation test and TSTR).
    ft:
        Optional fitted tabular transforms used to inverse-transform model-space
        tabular tensors back to raw feature space for tabular evaluation.
    device:
        Device used for image metric computation (SSIM/MS-SSIM/FID).
    config:
        Image metric configuration.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        synth_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        ft: Optional[FittedTransforms] = None,
        *,
        device: Optional[torch.device] = None,
        config: Optional[MetricsConfig] = None,
    ) -> None:
        self.ft = ft
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or MetricsConfig()

        # Load data in memory (+ row counts for label alignment).
        self.images_train, self.real_df, self.n_train = self._extract_data_from_loader(train_loader, use_tf=True)
        self.images_synth, self.synth_df, self.n_synth = self._extract_data_from_loader(synth_loader, use_tf=False)

        self.images_valid: Optional[torch.Tensor] = None
        self.valid_df: Optional[pl.DataFrame] = None
        self.n_valid: Optional[int] = None
        if valid_loader is not None:
            self.images_valid, self.valid_df, self.n_valid = self._extract_data_from_loader(valid_loader, use_tf=True)

        # Last computed metrics (kept for backward compatibility).
        self.metrics: dict = {}

    def _extract_data_from_loader(
        self,
        loader: DataLoader,
        use_tf: bool = True,
    ) -> Tuple[torch.Tensor, pl.DataFrame, int]:
        """Materialize a loader into tensors and a Polars DataFrame.

        The returned row count is derived from the final DataFrame height, which is
        used to align labels in downstream metrics.
        """
        images_list = []
        tab_data_list = []

        for batch in loader:
            images_list.append(batch["image"].to("cpu", non_blocking=True))
            if self.ft is not None and use_tf:
                tab_data_list.append(inverse_transform(self.ft, batch["tabular"].to("cpu", non_blocking=True)))
            else:
                tab_data_list.append(batch["tabular"].to("cpu", non_blocking=True))

        images = torch.cat(images_list, dim=0)
        tab_data = np.concatenate(tab_data_list, axis=0)

        columns = [f"col_{i}" for i in range(tab_data.shape[1])]
        df = pl.DataFrame(tab_data, schema=columns)
        return images, df, df.height

    def tabular(
        self,
        train_label: Optional[np.ndarray] = None,
        synth_label: Optional[np.ndarray] = None,
        valid_label: Optional[np.ndarray] = None,
        out_dir: Union[str, Path] = "",
    ) -> dict:
        """Compute tabular metrics between real and synthetic datasets.

        This method preserves the original behavior:
        - statistical feature metrics
        - mutual information / correlation differences
        - privacy metrics via DCR
        - utility metrics via TSTR (only when labels are provided)
        """
        path_to_json = str(out_dir) if out_dir else ""

        # 1) Summary statistics and mutual information.
        features_stats, _, _ = compute_statistical_metrics(self.real_df, self.synth_df, path_to_json=path_to_json)
        corr_real, corr_synth, corr_difference = compute_mutual_info(
            self.real_df, self.synth_df, path_to_json=path_to_json
        )

        # 2) Privacy: distance to closest record (DCR).
        dcr_synth_train = distance_to_closest_record("synth_train", self.synth_df, self.real_df, path_to_json=path_to_json)
        dcr_stats_synth_train = dcr_stats("synth_train", dcr_synth_train, path_to_json=path_to_json)
        dcr_zero_synth_train = number_of_dcr_equal_to_zero("synth_train", dcr_synth_train, path_to_json=path_to_json)

        dcr_stats_synth_valid = None
        dcr_zero_synth_valid = None
        dcr_share = None
        if self.valid_df is not None:
            dcr_synth_valid = distance_to_closest_record("synth_val", self.synth_df, self.valid_df, path_to_json=path_to_json)
            dcr_stats_synth_valid = dcr_stats("synth_val", dcr_synth_valid, path_to_json=path_to_json)
            dcr_zero_synth_valid = number_of_dcr_equal_to_zero("synth_val", dcr_synth_valid, path_to_json=path_to_json)
            dcr_share = validation_dcr_test(dcr_synth_train, dcr_synth_valid, path_to_json=path_to_json)

        # 3) Utility (TSTR): only computed when labels are provided.
        TSTR_train = None
        TSTR_synth = None
        delta = None
        if train_label is not None:
            X_train = self.real_df
            y_train = train_label[: self.n_train]
            X_synth = self.synth_df
            y_synth = synth_label[: self.n_synth] if synth_label is not None else None

            X_test = self.valid_df if self.valid_df is not None else self.real_df
            test_len = self.n_valid if self.valid_df is not None else self.n_train
            y_test = (valid_label if self.valid_df is not None else train_label)[:test_len]

            TSTR_train, TSTR_synth, delta = compute_utility_metrics_class(
                X_train, X_synth, X_test, y_train, y_synth, y_test, path_to_json=path_to_json
            )

        self.metrics = {
            "stats": features_stats,
            "mutual_info": {"real": corr_real, "synth": corr_synth, "diff": corr_difference},
            "dcr": {
                "synth_train": {"stats": dcr_stats_synth_train, "zero_count": dcr_zero_synth_train},
                "synth_valid": (
                    {"stats": dcr_stats_synth_valid, "zero_count": dcr_zero_synth_valid}
                    if self.valid_df is not None
                    else None
                ),
                "share": dcr_share if self.valid_df is not None else None,
            },
            "TSTR": (
                {"train_valid": TSTR_train, "synth_valid": TSTR_synth, "delta": delta}
                if train_label is not None
                else None
            ),
        }
        return self.metrics

    def tab_report(
        self,
        out_dir: Union[str, Path] = "reports/sure_tabular",
        train_label: Optional[np.ndarray] = None,
        synth_label: Optional[np.ndarray] = None,
        valid_label: Optional[np.ndarray] = None,
        use_cached: bool = True,
    ) -> None:
        """Generate a `sure` HTML report for tabular metrics.

        When `use_cached=True`, the report generation is skipped if JSON outputs
        already exist in `out_dir`.
        """
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        have_json = any(out_dir_path.glob("*.json"))
        if not (use_cached and have_json):
            self.tabular(
                train_label=train_label,
                synth_label=synth_label,
                valid_label=valid_label,
                out_dir=out_dir_path,
            )

        from sure.report_generator.report_generator import report as sure_report

        sure_report(self.real_df, self.synth_df, path_to_json=str(out_dir_path))

    def images(self) -> dict:
        """Compute image metrics (SSIM, MS-SSIM, FID) between real and synthetic images."""
        cfg = self.config

        ssim_score = self._ssim_score(self.images_train, self.images_synth, data_range=cfg.data_range, num_samples=cfg.num_samples)
        msssim_train = self._ms_ssim_score(
            self.images_train, self.images_synth, data_range=cfg.data_range, num_samples=cfg.num_samples
        )
        fid_score = self._fid_score(self.images_train, self.images_synth, batch_size=cfg.fid_batch_size)

        self.metrics = {
            "ssim_mean": ssim_score,
            "ms_ssim_mean": msssim_train,
            "fid_mean": fid_score,
        }
        # Preserve the original side effect.
        print(self.metrics)
        return self.metrics

    # ------------------------------------------------------------------
    # Image metric helpers (behavior preserved)
    # ------------------------------------------------------------------

    def _ssim_score(self, x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, num_samples: int = 1000) -> torch.Tensor:
        """Estimate SSIM via sampled (i, j) pairs.

        The original implementation normalizes each *set* to [0, 1] using global
        min/max and then evaluates PIQ's `multi_scale_ssim` per pair.
        """
        N_x = x.size(0)
        N_y = y.size(0)

        x = x.to(self.device)
        y = y.to(self.device)

        # Normalization (global over the set).
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        all_pairs = list(itertools.product(range(N_x), range(N_y)))
        if num_samples > len(all_pairs):
            num_samples = len(all_pairs)

        sampled_pairs = random.sample(all_pairs, num_samples)

        scores = []
        for i, j in sampled_pairs:
            x_img = x[i].unsqueeze(0)
            y_img = y[j].unsqueeze(0)
            score = multi_scale_ssim(x_img, y_img, data_range=data_range, reduction="none")
            scores.append(score.item())

        return torch.tensor(scores).mean()

    def _ms_ssim_score(self, x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, num_samples: int = 1000) -> torch.Tensor:
        """Estimate MS-SSIM via sampled (i, j) pairs."""
        N_x = x.size(0)
        N_y = y.size(0)

        x = x.to(self.device)
        y = y.to(self.device)

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        all_pairs = list(itertools.product(range(N_x), range(N_y)))
        if num_samples > len(all_pairs):
            num_samples = len(all_pairs)

        sampled_pairs = random.sample(all_pairs, num_samples)

        scores = []
        for i, j in sampled_pairs:
            x_img = x[i].unsqueeze(0)
            y_img = y[j].unsqueeze(0)
            score = multi_scale_ssim(x_img, y_img, data_range=data_range, reduction="none")
            scores.append(score.item())

        return torch.tensor(scores).mean()

    def _fid_score(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> float:
        """Compute FID from DenseNet121 embeddings."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Normalize to [0, 1] (global over each set).
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        x_dataset = DenseNetPreprocessedDataset(x)
        y_dataset = DenseNetPreprocessedDataset(y)
        x_dataloader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
        y_dataloader = DataLoader(y_dataset, batch_size=batch_size, shuffle=False)

        feature_extractor = DenseNet121FID(device=self.device)

        x_feats = self._extract_features(x_dataloader, feature_extractor)
        y_feats = self._extract_features(y_dataloader, feature_extractor)

        fid = FID()
        return fid.compute_metric(x_feats, y_feats)

    def _extract_features(self, dataloader: DataLoader, feature_extractor: nn.Module) -> torch.Tensor:
        """Extract DenseNet features for all images in a dataloader."""
        features = []
        feature_extractor = feature_extractor.to(self.device)
        for batch in dataloader:
            batch = batch.to(self.device)
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
            features.append(feats.cpu())
        return torch.cat(features, dim=0).cpu()


# Backward-compatible alias.
Metrics = TabularImageMetrics


class DenseNetPreprocessedDataset(torch.utils.data.Dataset):
    """Dataset that resizes images for DenseNet121 feature extraction."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.tensor[idx]  # (C, H, W)
        img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        return img

    def __len__(self) -> int:
        return int(self.tensor.shape[0])


class DenseNet121FID(nn.Module):
    """DenseNet121-based feature extractor for FID computation."""

    def __init__(self, in_channels: int = 1, device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        model = densenet121(spatial_dims=2, in_channels=in_channels, out_channels=1, pretrained=True)
        model.eval()

        self.features = nn.Sequential(
            model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(device)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)  # (N, 1024)
            return x