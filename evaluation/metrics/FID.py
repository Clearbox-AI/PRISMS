# evaluation/metrics/FID.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from scipy import linalg
import numpy as np

from evaluation.metrics.base import BaseMetric

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet Distance between two Gaussians described by
    their means and covariances.
    """
    # mu1, mu2: numpy arrays of shape [2048]
    # sigma1, sigma2: numpy arrays of shape [2048, 2048]
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # fallback to adding small identity
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("FID calculation produced imaginary component.")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = (diff.dot(diff) +
           np.trace(sigma1) +
           np.trace(sigma2) -
           2 * tr_covmean)
    return fid

class InceptionFeatureExtractor(nn.Module):
    """
    Wrap a pretrained Inception network (or any other CNN).
    Exposes a .forward that returns the features from a certain layer.
    For simplicity, assume you have a torch InceptionV3 or another feature net.
    """
    def __init__(self, inception_model):
        super().__init__()
        self.inception_model = inception_model
        self.inception_model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x is a float tensor of shape [B, 3, H, W].
        Return the feature embeddings, e.g. 2048-d.
        """
        with torch.no_grad():
            # The user can adapt this block to extract features from
            # inception_model. Many InceptionV3 wrappers exist,
            # e.g. torchvision.models.inception_v3(pretrained=True).
            features = self.inception_model(x)
            # Suppose we get [B, 2048] from the last pooling
        return features

class FIDMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any], shared_state: Dict[str, Any]):
        super().__init__(config, shared_state)
        # For instance, config might have:
        # config = {
        #   "enabled": True,
        #   "batch_size": 32,
        #   "cache_key_real": "fid_features_real",
        #   "cache_key_synth": "fid_features_synth",
        #   "model_path": "/path/to/inception.pt",
        #   ...
        # }
        self.batch_size = config.get("batch_size", 32)
        self.cache_key_real = config.get("cache_key_real", "fid_features_real")
        self.cache_key_synth = config.get("cache_key_synth", "fid_features_synth")

        # In a real system, you might pass a path to a pretrained model or
        # just use torchvision's standard InceptionV3.
        self.model_path = config.get("model_path", None)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # placeholder for the feature extractor
        self.feature_extractor = None

    def prepare(self):
        """
        Load or prepare anything we need. 
        For FID, we need an Inception-based feature extractor.
        Also check if real/synthetic features are in the shared_state cache
        or if we must compute them.
        """
        if self.feature_extractor is None:
            # If you have a custom pretrained inception:
            # inception = torch.load(self.model_path)  # or something similar
            # Or use torchvision:
            # from torchvision.models import inception_v3
            # inception = inception_v3(pretrained=True, transform_input=False)
            # Here we assume `inception` is already loaded:
            inception = nn.Identity()  # <-- stub, replace with real model
            self.feature_extractor = InceptionFeatureExtractor(inception)
            self.feature_extractor.to(self._device)

    def compute(self) -> Dict[str, Any]:
        """
        Actual FID computation:
          1) Collect or generate real images
          2) Collect or generate synthetic images
          3) Extract features
          4) Compute their Gaussian stats
          5) Compute FID
          6) Return result
        """
        # 1) Check if real features are already cached
        if self.cache_key_real in self.shared_state:
            real_features = self.shared_state[self.cache_key_real]
        else:
            # Otherwise, load/compute real images & get features
            real_images = self._get_real_images()  # You must implement
            real_features = self._extract_features_in_batches(real_images)
            self.shared_state[self.cache_key_real] = real_features

        # 2) Check if synthetic features are already cached
        if self.cache_key_synth in self.shared_state:
            synth_features = self.shared_state[self.cache_key_synth]
        else:
            # Otherwise generate synthetic images & get features
            synthetic_images = self._get_synthetic_images()  # You must implement
            synth_features = self._extract_features_in_batches(synthetic_images)
            self.shared_state[self.cache_key_synth] = synth_features

        # 3) Compute means and covariances
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_synth = np.mean(synth_features, axis=0)
        sigma_synth = np.cov(synth_features, rowvar=False)

        # 4) Compute FID
        fid_value = _calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)

        return {"FID": fid_value}

    def _extract_features_in_batches(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Utility for extracting features from a list of images
        in batches, returning a Numpy array of shape [N, feature_dim].
        """
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i+self.batch_size]
            # stack
            batch_t = torch.stack(batch).to(self._device)
            # shape [B, C, H, W]
            feats = self.feature_extractor(batch_t)  # shape [B, 2048]
            feats = feats.cpu().numpy()
            all_features.append(feats)
        all_features = np.concatenate(all_features, axis=0)
        return all_features

    def _get_real_images(self) -> List[torch.Tensor]:
        """
        Load or retrieve your real images here.
        Return a list of torch.Tensor images, each [3,H,W], for example.
        """
        # TODO: Replace with your actual data loading pipeline:
        # e.g. real_images = [ ... your dataloader ... ]
        # For demonstration, return random data:
        sample_count = 64
        height, width = 299, 299
        random_data = torch.rand(sample_count, 3, height, width)
        # Convert to a list of Tensors
        return list(random_data)

    def _get_synthetic_images(self) -> List[torch.Tensor]:
        """
        Generate or load your synthetic images here.
        Return a list of torch.Tensor images, each [3,H,W], for example.
        """
        # TODO: Replace with your actual synthetic generation step:
        sample_count = 64
        height, width = 299, 299
        random_data = torch.rand(sample_count, 3, height, width)
        # Convert to a list of Tensors
        return list(random_data)
