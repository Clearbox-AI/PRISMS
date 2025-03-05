import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from scipy import linalg

from evaluation.metrics.base import BaseMetric
from evaluation.metrics.artifact_store import ArtifactStore

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """
    Standard FID Frechet distance calculation.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("FID calculation produced imaginary component.")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)

class InceptionFeatureExtractor(nn.Module):
    """
    Wrap a pretrained Inception (or any CNN) to expose feature extraction.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # Suppose that returns a [B, 2048] feature
        return feats

class FIDMetric(BaseMetric):
    """
    FID Metric that uses the ArtifactStore to:
      - Load or generate real images
      - Load or generate synthetic images
      - Load or create a pretrained encoder (Inception)
      - Compute or retrieve FID features
    """
    def __init__(self, config: Dict[str, Any], artifact_store: ArtifactStore):
        super().__init__(config, artifact_store)
        self.batch_size = config.get("batch_size", 32)
        self.force_regenerate = config.get("force_regenerate", False)
        self.force_reload_encoder = config.get("force_reload_encoder", False)

        # Keys for the artifact store
        self.real_images_key = config.get("real_images_key", "real_images")
        self.synth_images_key = config.get("synth_images_key", "synth_images")
        self.encoder_key      = config.get("encoder_key", "inception_encoder")
        self.real_features_key  = config.get("real_features_key", "fid_features_real")
        self.synth_features_key = config.get("synth_features_key", "fid_features_synth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Placeholders
        self.real_features = None
        self.synth_features = None
        self.feature_extractor: nn.Module = None

    def prepare(self):
        """
        - Get or create real images
        - Get or create synthetic images
        - Get or create pretrained encoder
        - Extract features (only if not already cached, unless forced)
        """
        #TODO: move these functions to some utily part
        # 1) Load or create real images
        def load_real_images():
            # TODO: Replace with actual loading from a dataset
            # Example: return list of Tensors [3, H, W]
            sample_count = 32
            return list(torch.rand(sample_count, 3, 299, 299))

        real_images = self.artifact_store.get_or_create_artifact(
            key=self.real_images_key,
            creator_fn=load_real_images,
            force=False  # usually we don't forcibly reload real images
        )

        # 2) Load or create synthetic images
        def generate_synth_images():
            # TODO: Replace with actual generation via your pipeline
            sample_count = 32
            return list(torch.rand(sample_count, 3, 299, 299))

        synth_images = self.artifact_store.get_or_create_artifact(
            key=self.synth_images_key,
            creator_fn=generate_synth_images,
            force=self.force_regenerate
        )

        # 3) Load or create the pretrained encoder (Inception or custom).
        # Example factory:
        def create_inception():
            # e.g. from torchvision.models import inception_v3
            # backbone = inception_v3(pretrained=True, transform_input=False)
            backbone = nn.Identity()  # placeholder
            return InceptionFeatureExtractor(backbone).to(self.device)

        self.feature_extractor = self.artifact_store.get_or_create_pretrained_encoder(
            key=self.encoder_key,
            encoder_factory=create_inception,
            force=self.force_reload_encoder
        )

        # 4) Extract features for real & synthetic images (and store them)
        #    if not already in the store (or if forced).
        def extract_real_features():
            return self._extract_features_in_batches(real_images)

        self.real_features = self.artifact_store.get_or_create_artifact(
            key=self.real_features_key,
            creator_fn=extract_real_features,
            force=self.force_regenerate
        )

        def extract_synth_features():
            return self._extract_features_in_batches(synth_images)

        self.synth_features = self.artifact_store.get_or_create_artifact(
            key=self.synth_features_key,
            creator_fn=extract_synth_features,
            force=self.force_regenerate
        )

    def compute(self) -> Dict[str, Any]:
        mu_real = np.mean(self.real_features, axis=0)
        sigma_real = np.cov(self.real_features, rowvar=False)

        mu_synth = np.mean(self.synth_features, axis=0)
        sigma_synth = np.cov(self.synth_features, rowvar=False)

        fid_value = _calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)
        return {"FID": fid_value}

    def _extract_features_in_batches(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Helper to run images through the feature_extractor in batches.
        """
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            batch_t = torch.stack(batch).to(self.device)  # shape [B, 3, H, W]
            with torch.no_grad():
                feats = self.feature_extractor(batch_t).cpu().numpy()  # [B, 2048] or whatever
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)
