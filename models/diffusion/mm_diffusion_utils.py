"""
Utilities for the multimodal diffusion wrapper.

This module contains stateless helpers used by `MultiModalDiffusion` to keep the
main implementation focused on training and sampling logic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def log_sigma_to_t(sigma: torch.Tensor) -> torch.Tensor:
    """
    EDM time embedding: t = log(σ) / 4.

    Preserves shape for (B,) tensors and scalar tensors.
    """
    return sigma.log().div(4).float()


def corrcoef(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Correlation matrix of features (columns) using a standardized representation.
    """
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + eps)
    return (x.T @ x) / (x.shape[0] - 1.0)


def coral_loss(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL-style loss: Frobenius distance between correlation matrices,
    emphasizing off-diagonal terms (feature dependencies).
    """
    C_A, C_B = corrcoef(A), corrcoef(B)
    I = torch.eye(C_A.size(0), device=C_A.device, dtype=C_A.dtype)
    return ((C_A - C_B) * (1.0 - I)).pow(2).mean()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE between two embedding batches.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def min_snr_weight(sigma: torch.Tensor, sigma_data: torch.Tensor, gamma: float):
    """
    Min-SNR reweighting factor used in diffusion training.

    Returns a detached tensor factor; for gamma<=0 returns a scalar 1.0 (no-op),
    matching the original behavior.
    """
    if gamma <= 0.0:
        return 1.0
    snr = (sigma_data ** 2) / (sigma ** 2 + 1e-12)
    return (snr / (snr + gamma)).detach()
