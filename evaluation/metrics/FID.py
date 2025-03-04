import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from scipy import linalg

from evaluation.metrics.base import BaseMetric
from evaluation.metrics.artifact_store import ArtifactStore

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
            # Adapt this block to extract features from
            # inception_model. Many InceptionV3 wrappers exist,
            # e.g. torchvision.models.inception_v3(pretrained=True).
            features = self.inception_model(x)
            # Suppose we get [B, 2048] from the last pooling
        return features

class FIDMetric(BaseMetric):
    def __init__(self, config: Dict[str, Any], artifact_store: ArtifactStore):
        super().__init__(config, artifact_store)
        self.batch_size = config.get("batch_size", 32)
        self.cache_key_real = config.get("cache_key_real", "fid_features_real")
        self.cache_key_synth = config.get("cache_key_synth", "fid_features_synth")

        self.generate_once_key_synth = config.get("generate_once_key_synth", "generated_images")
        self.force_regenerate = config.get("force_regenerate", False)  # example flag

        # placeholders
        self.real_features = None
        self.synth_features = None
        self.feature_extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self):
        """
        1) Prepare the Inception extractor if not already done.
        2) Ensure we have real and synthetic features:
           - If not cached, load/generate images, extract features, store them.
        """
        # 1) Load or create the feature extractor
        if self.feature_extractor is None:
            self.feature_extractor = self._load_inception()
            self.feature_extractor.to(self.device)

        # 2) Real features
        if not self.artifact_store.has(self.cache_key_real):
            # We do not have real features yet, so we load real images and compute
            real_images = self._get_real_images()
            real_feats = self._extract_features_in_batches(real_images)
            self.artifact_store.put(self.cache_key_real, real_feats)
        self.real_features = self.artifact_store.get(self.cache_key_real)

        # 3) Synthetic features
        if not self.artifact_store.has(self.cache_key_synth) or self.force_regenerate:
            # Either we have no synthetic features or we want to force regeneration
            synth_images = self._get_or_generate_synth_images()
            synth_feats = self._extract_features_in_batches(synth_images)
            self.artifact_store.put(self.cache_key_synth, synth_feats)
        self.synth_features = self.artifact_store.get(self.cache_key_synth)

    def compute(self) -> Dict[str, Any]:
        mu_real = np.mean(self.real_features, axis=0)
        sigma_real = np.cov(self.real_features, rowvar=False)

        mu_synth = np.mean(self.synth_features, axis=0)
        sigma_synth = np.cov(self.synth_features, rowvar=False)

        fid_value = _calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)
        return {"FID": fid_value}

    def _load_inception(self) -> nn.Module:
        # Replace with real Inception loading, e.g.:
        # from torchvision.models import inception_v3
        # model = inception_v3(pretrained=True, transform_input=False)
        # or load your custom .pth
        model = nn.Identity()  # stub
        return InceptionFeatureExtractor(model)

    def _extract_features_in_batches(self, images) -> np.ndarray:
        # same approach as before
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i: i + self.batch_size]
            batch_t = torch.stack(batch).to(self.device)
            feats = self.feature_extractor(batch_t)
            feats = feats.cpu().numpy()
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)

    def _get_real_images(self):
        """
        Load or retrieve your real images as a list of [3, H, W] Tensors.
        Here, just stub random data as an example.
        """
        sample_count = 64
        height, width = 299, 299
        random_data = torch.rand(sample_count, 3, height, width)
        return list(random_data)

    def _get_or_generate_synth_images(self):
        """
        If your pipeline already generated synthetic images once,
        you might store them in artifact_store under "generated_images".
        If not, generate them, then store.
        """
        # e.g. check if we have them:
        if self.artifact_store.has(self.generate_once_key_synth) and not self.force_regenerate:
            return self.artifact_store.get(self.generate_once_key_synth)

        # Otherwise, generate them
        sample_count = 64
        height, width = 299, 299
        random_data = torch.rand(sample_count, 3, height, width)
        images_list = list(random_data)

        # store in artifact store for future usage
        self.artifact_store.put(self.generate_once_key_synth, images_list)
        return images_list
