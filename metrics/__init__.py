"""Metrics package for evaluating multimodal synthetic datasets.

This package provides:
- Tabular statistics, privacy (DCR), and utility (TSTR) metrics via the `sure` library.
- Image quality/diversity metrics (SSIM/MS-SSIM/FID).
- Cross-modal coherence metrics (discriminator AUC and embedding similarity).
"""

from .tabular_images_metrics import Metrics, TabularImageMetrics, MetricsConfig

__all__ = [
    "Metrics",
    "TabularImageMetrics",
    "MetricsConfig",
]
