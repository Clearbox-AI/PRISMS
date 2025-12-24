"""Small encoder modules used by coherence metrics.

The encoders are intentionally simple and lightweight; they are not meant to be
state-of-the-art representation learners. The goal is to provide a consistent
and reproducible signal when comparing coherent vs incoherent cross-modal pairs.

Important
---------
The class names and parameter layouts preserve checkpoint compatibility with the
original code (state_dict key structure).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityImageEncoder(nn.Module):
    """CNN image encoder used by the similarity (InfoNCE) metric."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 128×128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64×64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32×32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class SimilarityTabularEncoder(nn.Module):
    """MLP tabular encoder used by the similarity (InfoNCE) metric."""

    def __init__(self, in_features: int, dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class DiscriminatorImageEncoder(nn.Module):
    """CNN image encoder used by the coherence discriminator."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # (B,16,128,128)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (B,32,64,64)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (B,64,32,32)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DiscriminatorTabularEncoder(nn.Module):
    """MLP tabular encoder used by the coherence discriminator."""

    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def info_nce(img_emb: torch.Tensor, tab_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE loss over paired embeddings."""
    img_emb = F.normalize(img_emb, dim=-1)
    tab_emb = F.normalize(tab_emb, dim=-1)
    logits = (img_emb @ tab_emb.T) / temperature
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)
