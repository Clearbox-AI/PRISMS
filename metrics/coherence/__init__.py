"""Cross-modal coherence metrics.

This subpackage contains lightweight models used to quantify coherence between
paired images and tabular features:
- A contrastive similarity score based on paired encoders (InfoNCE training).
- A discriminator AUC score based on a binary classifier trained to distinguish
  aligned vs randomly shuffled pairs.
"""

from .similarity_score import Similarity
from .discriminator_score import Discriminator, make_incoherent_loader

__all__ = ["Similarity", "Discriminator", "make_incoherent_loader"]
