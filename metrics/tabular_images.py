from typing import Tuple
import os
import sys 
prisms_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(prisms_path)
from pathlib import Path
import itertools
import random

import polars as pl

import torch
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn
import torch.nn.functional as F

from piq import ssim, multi_scale_ssim, FID

from monai.networks.nets import densenet121
from monai.transforms import Compose, Resize, ScaleIntensity, ToTensor

from data.loader import load_training_data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.configurations import set_project_root

from sure.utility import compute_statistical_metrics, compute_mutual_info, compute_utility_metrics_class
from sure.privacy import distance_to_closest_record, dcr_stats, number_of_dcr_equal_to_zero, validation_dcr_test
from sure import report

class Metrics:
    """
    Class to compute metrics for tabular and image data.
    """
    def __init__(
            self, 
            train_loader: torch.utils.data.dataloader.DataLoader, 
            synth_loader: torch.utils.data.dataloader.DataLoader, 
            valid_loader: torch.utils.data.dataloader.DataLoader = None,
        ):
        """
        Initialize the TabularMetrics class.
        Takes in the training and synthetic data loaders, and an optional preprocessor, and stores the images and the tabular data.

        Args:
            train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for the training data.
            synth_loader (torch.utils.data.dataloader.DataLoader): DataLoader for the synthetic data.
            preprocessor (Preprocessor, optional): Preprocessor instance for data preprocessing. Defaults to None.
        """
        # Check if the device is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the training and synthetic data
        self.images_train, self.real_df = self._extract_data_from_loader(train_loader)
        self.images_synth, self.synth_df = self._extract_data_from_loader(synth_loader)

        if valid_loader is not None:
            images_valid, valid_df = self._extract_data_from_loader(valid_loader)   
            self.images_valid = images_valid      
            self.valid_df = valid_df

    def _extract_data_from_loader(
            self, 
            loader
        ) -> Tuple[torch.Tensor, pl.DataFrame]:
        """
        Returns the images as a torch tensor and tabular data as a polars DataFrames.
        """
        images_list = []
        tab_data_list = []

        # Load the training data
        for batch in loader:
            images_list.append(batch['image'].to("cpu", non_blocking=True))
            tab_data_list.append(batch['tabular'].to("cpu", non_blocking=True))

        images = torch.cat(images_list, dim=0)
        tab_data = torch.cat(tab_data_list, dim=0).numpy()

        columns = [f"col_{i}" for i in range(tab_data.shape[1])]
        df = pl.DataFrame(tab_data, schema=columns)
        
        return images, df
    
    def tabular(self, train_label=None, synth_label=None, valid_label=None):
        """
        Compute the tabular metrics (statistical metrics, mutual information, distance to closest record, TSTR) between the real and synthetic data.

        Args:
            train_label (torch.Tensor, optional): Labels for the training data. Defaults to None.
            synth_label (torch.Tensor, optional): Labels for the synthetic data. Defaults to None.
            valid_label (torch.Tensor, optional): Labels for the validation data. Defaults to None.
        Returns:
            dict: Dictionary containing the computed metrics.
        """
        path_to_json = ""
        # Compute statistical metrics
        features_stats, _, _ = compute_statistical_metrics(self.real_df, self.synth_df, path_to_json=path_to_json)

        # Compute mutual information
        corr_real, corr_synth, corr_difference = compute_mutual_info(self.real_df, self.synth_df, path_to_json=path_to_json)

        # Distance to closest record
        dcr_synth_train       = distance_to_closest_record("synth_train", self.synth_df, self.real_df, path_to_json=path_to_json)
        dcr_stats_synth_train = dcr_stats("synth_train", dcr_synth_train, path_to_json=path_to_json)
        dcr_zero_synth_train  = number_of_dcr_equal_to_zero("synth_train", dcr_synth_train, path_to_json=path_to_json)
        if self.valid_df is not None:
            dcr_synth_valid       = distance_to_closest_record("synth_val", self.synth_df, self.valid_df, path_to_json=path_to_json)
            dcr_stats_synth_valid = dcr_stats("synth_val", dcr_synth_valid, path_to_json=path_to_json)
            dcr_zero_synth_valid  = number_of_dcr_equal_to_zero("synth_val", dcr_synth_valid, path_to_json=path_to_json)
            dcr_share             = validation_dcr_test(dcr_synth_train, dcr_synth_valid, path_to_json=path_to_json)

        # TSTR
        ############
        import numpy as np
        train_label = np.random.randint(0, 2, size=len(self.real_df))
        synth_label = np.random.randint(0, 2, size=len(self.synth_df))
        valid_label = np.random.randint(0, 2, size=len(self.valid_df)) if self.valid_df is not None else None
        ############
        if train_label is not None:
            X_train = self.real_df
            y_train = train_label
            X_synth = self.synth_df
            y_synth = synth_label
            X_test = self.valid_df if self.valid_df is not None else self.real_df
            y_test = valid_label if self.valid_df is not None else train_label
            TSTR_train, TSTR_synth, delta = compute_utility_metrics_class(X_train, X_synth, X_test, y_train, y_synth, y_test, path_to_json=path_to_json)

        # Store the metrics in a dictionary
        self.metrics = {
            "stats": features_stats,
            "mutual_info": {
                "real": corr_real,
                "synth": corr_synth,
                "diff": corr_difference
            },
            "dcr": {
                "synth_train": {
                    "stats": dcr_stats_synth_train,
                    "zero_count": dcr_zero_synth_train
                },
                "synth_valid": {
                    "stats": dcr_stats_synth_valid,
                    "zero_count": dcr_zero_synth_valid
                } if self.valid_df is not None else None,
                "share": dcr_share if self.valid_df is not None else None
            },
            "TSTR": {
                "train_valid": TSTR_train,
                "synth_valid": TSTR_synth,
                "delta": delta
            } if train_label is not None else None
        }
        return self.metrics
    
    def tab_report(self):
        """
        Generate a report of the tabular metrics.
        """
        report(self.real_df, self.synth_df)

    def images(self):
        """
        Compute the image metrics (SSIM, MS-SSIM, FID) between the real and synthetic images.
        """
        data_range = 1.0 # Da mettere in config
        num_samples = 1000 # Da mettere in config
        metrics_batch_size = 32 # Da mettere in config

        # Compute SSIM
        # ssim_score   = self._ssim_score(self.images_valid, self.images_synth, data_range=data_range, num_samples=num_samples)
        # msssim_train = self._ms_ssim_score(self.images_valid, self.images_synth, data_range=data_range, num_samples=num_samples)

        # Compute FID
        fid_score = self._fid_score(self.images_valid, self.images_synth, batch_size=metrics_batch_size)

        # Store the metrics in a dictionary
        self.metrics = {
            # "ssim_mean": ssim_score,
            # "ms_ssim_mean": msssim_train,
            "fid_mean": fid_score
        }
        return self.metrics

    # def _get_nth_batch_norm(self, x, y, n, batch_size):
    #     """
    #     Get the normalized nth batch of data.
    #     """
    #     x_batch = x[n:n+batch_size]
    #     y_batch = y[n:n+batch_size]

    #     if x_batch.shape[0] != y_batch.shape[0]:
    #         min_batch_size = min(x_batch.shape[0], y_batch.shape[0])
    #         x_batch = x_batch[:min_batch_size]
    #         y_batch = y_batch[:min_batch_size]
        
    #     # Normalization [0, 1]
    #     x_batch = (x_batch + 1) / 2
    #     y_batch = (y_batch + 1) / 2

    #     return x_batch, y_batch

    def _ssim_score(self, x, y, data_range=1.0, num_samples=1000):
        """
        Sample unique, random (x[i], y[j]) pairs from two sets and compute SSIM.

        Args:
            x_set, y_set: Tensors of shape (N, C, H, W)
            data_range: Pixel range of images (typically 1.0 if normalized)
            num_samples: Number of unique pairs to evaluate
            batch_size: Number of samples to process at once
        Returns:
            Average SSIM score over sampled unique pairs
        """
        N_x = x.size(0)
        N_y = y.size(0)

        x = x.to(self.device)
        y = y.to(self.device)

        # Normalization
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        # All possible unique (i, j) pairs
        all_pairs = list(itertools.product(range(N_x), range(N_y)))

        if num_samples > len(all_pairs):
            num_samples = len(all_pairs)
        
        sampled_pairs = random.sample(all_pairs, num_samples)

        scores = []
        for i, j in sampled_pairs:
            x_img = x[i].unsqueeze(0)
            y_img = y[j].unsqueeze(0)
            score = ssim(x_img, y_img, data_range=data_range, reduction='none')
            scores.append(score.item())
            
        return torch.tensor(scores).mean()
    
    def _ms_ssim_score(self, x, y, data_range=1.0, num_samples=1000):
        """
        Sample unique, random (x[i], y[j]) pairs from two sets and compute MS-SSIM.

        Args:
            x_set, y_set: Tensors of shape (N, C, H, W)
            data_range: Pixel range of images (typically 1.0 if normalized)
            num_samples: Number of unique pairs to evaluate
            batch_size: Number of samples to process at once
        Returns:
            Average MS-SSIM score over sampled unique pairs
        """
        N_x = x.size(0)
        N_y = y.size(0)

        x = x.to(self.device)
        y = y.to(self.device)

        # Normalization
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        # All possible unique (i, j) pairs
        all_pairs = list(itertools.product(range(N_x), range(N_y)))

        if num_samples > len(all_pairs):
            num_samples = len(all_pairs)
        
        sampled_pairs = random.sample(all_pairs, num_samples)

        scores = []
        for i, j in sampled_pairs:
            x_img = x[i].unsqueeze(0)
            y_img = y[j].unsqueeze(0)
            score = multi_scale_ssim(x_img, y_img, data_range=data_range, reduction='none')
            scores.append(score.item())
            
        return torch.tensor(scores).mean()

    def _fid_score(self, x, y, batch_size=32):
        """
        Compute Multi Scale Structural Similarity Index

        Args:
            x_set, y_set: Tensors of shape (N, C, H, W)
            batch_size: Number of samples to process at once
        Returns:
            Average FID score over sampled unique pairs
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Resize to the minum size between the two tensors
        min_size = min(x.shape[0], y.shape[0])
        x = x[:min_size]
        y = y[:min_size]

        # Normalize [0, 1]
        x = x.float()
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)

        y = y.float()
        y_min = y.amin(dim=(2, 3), keepdim=True)
        y_max = y.amax(dim=(2, 3), keepdim=True)
        y = (y - y_min) / (y_max - y_min + 1e-8)

        # If second dimension is 1, repeat it to make it 3
        # if x.shape[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)
        #     y = y.repeat(1, 3, 1, 1)

        x_dataset = DenseNetPreprocessedDataset(x)
        y_dataset = DenseNetPreprocessedDataset(y)
        x_dataloader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
        y_dataloader = DataLoader(y_dataset, batch_size=batch_size, shuffle=False)

        # # Load DenseNet121FID model and extract features
        # feature_extractor = InceptionFID(device=self.device)
        feature_extractor = DenseNet121FID(device=self.device)

        x_feats = self._extract_features(x_dataloader, feature_extractor)
        y_feats = self._extract_features(y_dataloader, feature_extractor)

        # Compute FID
        fid = FID()
        return fid.compute_metric(x_feats, y_feats)

    def _extract_features(self, dataloader, feature_extractor):
        """
        Extract features from the dataloader using the feature extractor.
        Args:
            dataloader: DataLoader for the dataset.
            feature_extractor: Feature extractor model.
        Returns:
            torch.Tensor: Extracted features.
        """
        features = []
        feature_extractor = feature_extractor.to(self.device)
        for batch in dataloader:
            batch = batch.to(self.device)
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
            features.append(feats.cpu())  # Detach from GPU
        return torch.cat(features, dim=0).cpu()

    
class InceptionFID(nn.Module):
    """
    Inception feature extractor for FID computation.
    """
    def __init__(self, device='cpu'):
        """
        Initialize the InceptionFID class.
        Args:
            device (str): Device to use for computation ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        inception.eval()

        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(device)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the Inception model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Extracted features.
        """
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
            return x


class InceptionPreprocessedDataset(torch.utils.data.Dataset):
    """
    Dataset class for preprocessed images.
    """
    def __init__(self, tensor):
        self.tensor = tensor
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __getitem__(self, idx):
        img = self.tensor[idx]
        img = F.interpolate(img.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)
        img = (img - self.mean.to(img.device)) / self.std.to(img.device)
        return img

    def __len__(self):
        return self.tensor.shape[0]
        
class DenseNetPreprocessedDataset(torch.utils.data.Dataset):
    """
    Dataset that resizes images for DenseNet121 without redundant normalization.
    """
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, idx):
        img = self.tensor[idx]  # shape: (1, H, W)
        img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        return img

    def __len__(self):
        return self.tensor.shape[0]

class DenseNet121FID(nn.Module):
    """
    DenseNet121-based feature extractor for FID computation.
    """
    def __init__(self, in_channels=1, device='cpu'):
        super().__init__()
        model = densenet121(spatial_dims=2, in_channels=in_channels, out_channels=1, pretrained=True)
        model.eval()

        self.features = nn.Sequential(
            model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # output shape (N, 1024, 1, 1)
        ).to(device)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)  # shape (N, 1024)
            return x
##############################################################

if __name__ == "__main__":
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

    train_loader = load_training_data(cfg)
    val_loader   = load_training_data(cfg)
    synth_loader = load_training_data(cfg)

    metrics_manager = Metrics(train_loader, synth_loader, val_loader)
    # tab_metrics = metrics_manager.tabular()
    img_metrics = metrics_manager.images()
    # metrics_manager.tab_report()
    a = 0