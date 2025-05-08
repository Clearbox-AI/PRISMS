from typing import Tuple
import os
from pathlib import Path

import polars as pl

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn
import torch.nn.functional as F

from piq import ssim, multi_scale_ssim, FID

from data.loader import load_training_data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.configurations import set_project_root

from sure.utility import compute_statistical_metrics, compute_mutual_info, compute_utility_metrics_class
from sure.privacy import distance_to_closest_record, dcr_stats, number_of_dcr_equal_to_zero, validation_dcr_test
from sure import report

class Metrics:
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
        # Load the training and synthetic data
        images_train, real_df = self._extract_data_from_loader(train_loader)
        images_synth, synth_df = self._extract_data_from_loader(synth_loader)
        
        # Drop the second channel if it exists
        # if len(images_train.shape) == 4:
        #     images_train = images_train[:, 0]
        #     images_synth = images_synth[:, 0]

        self.images_train = images_train
        self.images_synth = images_synth

        self.real_df = real_df
        self.synth_df = synth_df

        if valid_loader is not None:
            images_valid, valid_df = self._extract_data_from_loader(valid_loader)   
            # if len(images_valid.shape) == 4:
            #     images_valid = images_valid[:, 0]     
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
        data_range = 1.0 # Da mettere in config

        # Compute SSIM
        # ssim_score = self._ssim_score(self.images_valid, self.images_synth, data_range)
        # msssim_train = self._ms_ssim_score(self.images_valid, self.images_synth, data_range=data_range)

        # Compute FID
        fid_score = self._fid_score(self.images_valid, self.images_synth)

        # Store the metrics in a dictionary
        self.metrics = {
            # "ssim_mean": ssim_score,
            # "ms_ssim_mean": msssim_train,
            "fid_mean": fid_score
        }
        return self.metrics

    def _get_nth_batch_norm(self, x, y, n, batch_size):
        """
        Get the normalized nth batch of data.
        """
        x_batch = x[n:n+batch_size]
        y_batch = y[n:n+batch_size]

        if x_batch.shape[0] != y_batch.shape[0]:
            min_batch_size = min(x_batch.shape[0], y_batch.shape[0])
            x_batch = x_batch[:min_batch_size]
            y_batch = y_batch[:min_batch_size]
        
        # Normalization [0, 1]
        x_batch = (x_batch + 1) / 2
        y_batch = (y_batch + 1) / 2

        return x_batch, y_batch

    def _ssim_score(self, x, y, data_range, batch_size=16):
        """
        Compute Structural Similarity Index
        """
        scores = []
        size = min(x.shape[0], y.shape[0])
        for i in range(0, size, batch_size):
            x_batch, y_batch = self._get_nth_batch_norm(x, y, i, batch_size)
            
            ssim_val = ssim(x_batch, y_batch, data_range=data_range)
            scores.append(ssim_val)

        return torch.stack(scores).mean()
    
    def _ms_ssim_score(self, x, y, data_range, batch_size=16):
        """
        Compute Multi Scale Structural Similarity Index
        """
        scores = []
        size = min(x.shape[0], y.shape[0])
        for i in range(0, size, batch_size):
            x_batch, y_batch = self._get_nth_batch_norm(x, y, i, batch_size)

            ms_ssim_val = multi_scale_ssim(x_batch, y_batch, data_range=data_range)
            scores.append(ms_ssim_val)
            
        return torch.stack(scores).mean()

    def _fid_score(self, x, y, batch_size=32):
        """
        Compute Multi Scale Structural Similarity Index
        """
        min_size = min(x.shape[0], y.shape[0])
        x = x[:min_size]
        y = y[:min_size]

        # Normalize [0, 1]
        x = x.float()
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min)

        y = y.float()
        y_min = y.amin(dim=(2, 3), keepdim=True)
        y_max = y.amax(dim=(2, 3), keepdim=True)
        y = (y - y_min) / (y_max - y_min)

        # If second dimension is 1, repeat it to make it 3
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        x_dataset = InceptionPreprocessedDataset(x)
        y_dataset = InceptionPreprocessedDataset(y)
        x_dataloader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
        y_dataloader = DataLoader(y_dataset, batch_size=batch_size, shuffle=False)

        # # Load Inception model and extract features
        feature_extractor = InceptionFID()

        x_feats = self._extract_features(x_dataloader, feature_extractor)
        y_feats = self._extract_features(y_dataloader, feature_extractor)

        # Compute FID
        fid = FID()
        return fid.compute_metric(x_feats, y_feats)

    def _extract_features(self, dataloader, feature_extractor):
        features = []

        for batch in dataloader:
            # batch = batch.cuda()
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1) # Flatten
            # features.append(feats.cpu())
            features.append(feats)#.cpu())

        return torch.cat(features, dim=0).cpu()
    
class InceptionFID(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        inception.eval()

        # Extract only convolutional layers up to AdaptiveAvgPool2d
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
        )

        # Optional: freeze params
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
            return x


class InceptionPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __getitem__(self, idx):
        img = self.tensor[idx]
        img = F.interpolate(img.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)
        img = (img - self.mean) / self.std
        return img

    def __len__(self):
        return self.tensor.shape[0]
        
##############################################################
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
a=1