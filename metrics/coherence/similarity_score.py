"""Embedding similarity coherence metric.

This metric trains (optionally) a pair of lightweight encoders on aligned
image-tabular pairs using a symmetric InfoNCE objective. The final score is the
mean cosine similarity (mapped to [0, 1]) between paired embeddings.

The public `Similarity.evaluate()` API is preserved for backward compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from .encoders import SimilarityImageEncoder, SimilarityTabularEncoder, info_nce


@dataclass(frozen=True)
class SimilarityConfig:
    """Training configuration for the similarity metric."""

    dim: int = 128
    lr: float = 1e-4
    temperature: float = 0.07


class Similarity:
    """Compute a similarity-based coherence score for paired data."""

    def __init__(self, dim: int = 128, lr: float = 1e-4, *, device: Optional[torch.device] = None) -> None:
        self.dim = int(dim)
        self.lr = float(lr)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_enc: Optional[nn.Module] = None
        self.tab_enc: Optional[nn.Module] = None

    def evaluate(
        self,
        loader: DataLoader,
        checkpoint_path: str,
        train_flag: bool = False,
        epochs: int = 10,
    ) -> float:
        """Train (optional) and compute the similarity score.

        Parameters
        ----------
        loader:
            DataLoader yielding dict batches with keys `image` and `tabular`.
        checkpoint_path:
            Path to save or load encoder weights.
        train_flag:
            If True, train encoders and save to `checkpoint_path`.
            If False, load from `checkpoint_path`.
        epochs:
            Number of training epochs (only used when `train_flag=True`).
        """
        if train_flag:
            self._train_similarity(loader, epochs, checkpoint_path)
        else:
            self._load_encoders(checkpoint_path)

        return float(self._similarity_score(loader))

    def _train_similarity(self, loader: DataLoader, epochs: int, save_path: str) -> None:
        """Train encoders with InfoNCE on aligned pairs."""
        # Infer tabular dimensionality from a single batch.
        batch0 = next(iter(loader))
        tab_dim = int(batch0["tabular"].numel() // batch0["tabular"].shape[0])

        self.img_enc = SimilarityImageEncoder(self.dim).to(self.device)
        self.tab_enc = SimilarityTabularEncoder(tab_dim, self.dim).to(self.device)

        optim = torch.optim.AdamW(
            list(self.img_enc.parameters()) + list(self.tab_enc.parameters()),
            lr=self.lr,
        )

        self.img_enc.train()
        self.tab_enc.train()

        for ep in range(int(epochs)):
            total = 0.0
            for batch in tqdm.tqdm(loader, desc=f"Epoch {ep + 1}/{epochs}"):
                img = batch["image"].to(self.device)
                tab = batch["tabular"].to(self.device)

                loss = info_nce(self.img_enc(img), self.tab_enc(tab))
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += float(loss.item())

            print(f"epoch {ep + 1}: loss {total / max(1, len(loader)):.4f}")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {"img_enc": self.img_enc.state_dict(), "tab_enc": self.tab_enc.state_dict()},
            save_path,
        )
        print(f"✓ encoders saved → {os.path.abspath(save_path)}")

    def _load_encoders(self, checkpoint: str) -> None:
        """Load encoder weights from a checkpoint produced by `_train_similarity`."""
        state = torch.load(checkpoint, map_location=self.device)

        self.img_enc = SimilarityImageEncoder(self.dim).to(self.device)

        # Preserve original heuristic to infer tabular dimension from the first Linear layer.
        tab_dim = int(state["tab_enc"]["net.0.weight"].shape[1])
        self.tab_enc = SimilarityTabularEncoder(tab_dim, self.dim).to(self.device)

        self.img_enc.load_state_dict(state["img_enc"])
        self.tab_enc.load_state_dict(state["tab_enc"])
        self.img_enc.eval()
        self.tab_enc.eval()
        print(f"✓ Encoders loaded ← {os.path.abspath(checkpoint)}")

    @torch.no_grad()
    def _similarity_score(self, loader: DataLoader) -> float:
        """Compute the mean cosine similarity mapped to [0, 1]."""
        assert self.img_enc is not None and self.tab_enc is not None, "Encoders are not initialized."

        self.img_enc.eval()
        self.tab_enc.eval()

        sims = []
        for batch in tqdm.tqdm(loader, desc="Evaluating"):
            img = batch["image"].to(self.device)
            tab = batch["tabular"].to(self.device)

            z_img = self.img_enc(img)
            z_tab = self.tab_enc(tab)
            cos = F.cosine_similarity(z_img, z_tab, dim=1)  # [-1, 1]
            sims.extend(((cos + 1) / 2).cpu().tolist())     # [0, 1]

        return float(sum(sims) / max(1, len(sims)))
