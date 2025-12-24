"""Discriminator-based cross-modal coherence metric.

A lightweight discriminator is trained to distinguish:
- *coherent* pairs (image, tabular) sampled from the dataset, and
- *incoherent* pairs created by permuting tabular entries within a batch.

The primary reported metric is ROC-AUC on the training loop, and the final
coherence score is the mean predicted probability for a given loader.

The public `Discriminator.evaluate()` and `make_incoherent_loader()` APIs are
preserved for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .encoders import DiscriminatorImageEncoder, DiscriminatorTabularEncoder


class Discriminator(nn.Module):
    """Coherence discriminator for aligned vs shuffled cross-modal pairs."""

    def __init__(self, embed_dim: int = 128, tabular_dim: int = 158) -> None:
        super().__init__()
        self.image_encoder = DiscriminatorImageEncoder(output_dim=embed_dim)
        self.tabular_encoder = DiscriminatorTabularEncoder(input_dim=tabular_dim, output_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z_img: torch.Tensor, z_tab: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_img, z_tab], dim=1)
        return self.classifier(z)

    def _train_discriminator(self, dataloader: DataLoader, epochs: int, save_path: str, lr: float = 1e-4) -> None:
        """Train the discriminator on aligned vs permuted pairs."""
        self.to(self.device)
        self.image_encoder.train()
        self.tabular_encoder.train()
        super().train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(int(epochs)):
            y_true_all, y_pred_all = [], []
            total_loss = 0.0

            for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x_img = batch["image"].to(self.device)
                x_tab = batch["tabular"].to(self.device)
                B = x_img.size(0)

                # Positive and negative pairs within the batch.
                y_pos = torch.ones(B, 1, device=self.device)
                perm = torch.randperm(B, device=self.device)
                x_tab_neg = x_tab[perm]
                y_neg = torch.zeros(B, 1, device=self.device)

                x_img_all = torch.cat([x_img, x_img], dim=0)
                x_tab_all = torch.cat([x_tab, x_tab_neg], dim=0)
                y_all = torch.cat([y_pos, y_neg], dim=0)

                z_img = self.image_encoder(x_img_all)
                z_tab = self.tabular_encoder(x_tab_all)
                y_pred = self(z_img, z_tab)

                loss = F.binary_cross_entropy(y_pred, y_all)
                total_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_true_all.extend(y_all.detach().cpu().numpy())
                y_pred_all.extend(y_pred.detach().cpu().numpy())

            auc = roc_auc_score(y_true_all, y_pred_all)
            print(f"Epoch {epoch + 1}/{epochs}  |  loss={total_loss:.4f}  |  AUC={auc:.4f}")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"✓ Full model saved → {os.path.abspath(save_path)}")

    def _load_discriminator_models(self, checkpoint_path: str) -> None:
        """Load the full model weights (encoders + classifier)."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(state)
        self.to(self.device)
        self.image_encoder.eval()
        self.tabular_encoder.eval()
        self.eval()
        print(f"✓ Full model loaded ← {os.path.abspath(checkpoint_path)}")

    def evaluate(
        self,
        loader: DataLoader,
        checkpoint_path: str | None = None,
        train_flag: bool = False,
        epochs: int = 10,
        lr: float = 1e-4,
    ) -> float:
        """Train (optional) and compute the coherence score for a loader."""
        if train_flag:
            assert checkpoint_path is not None, "checkpoint_path is required when train_flag=True"
            self._train_discriminator(loader, epochs=epochs, save_path=checkpoint_path, lr=lr)
        else:
            assert checkpoint_path is not None, "checkpoint_path is required when train_flag=False"
            self._load_discriminator_models(checkpoint_path)

        return float(self._discriminator_score(loader))

    @torch.no_grad()
    def _discriminator_score(self, loader: DataLoader) -> float:
        """Return the mean predicted coherence probability for all pairs."""
        self.eval()
        self.image_encoder.eval()
        self.tabular_encoder.eval()

        scores = []
        for batch in tqdm.tqdm(loader, desc="Evaluating"):
            x_img = batch["image"].to(self.device)
            x_tab = batch["tabular"].to(self.device)

            z_img = self.image_encoder(x_img)
            z_tab = self.tabular_encoder(x_tab)
            y_pred = self(z_img, z_tab)  # (B, 1)
            scores.extend(y_pred.squeeze().tolist())

        return float(sum(scores) / max(1, len(scores)))


class IncoherentPairDataset(Dataset):
    """Dataset wrapper that permutes tabular features to break alignment."""

    def __init__(self, src_loader: DataLoader, generator: Optional[torch.Generator] = None) -> None:
        super().__init__()
        all_images = []
        all_tabular = []
        for batch in src_loader:
            all_images.append(batch["image"].detach().cpu())
            all_tabular.append(batch["tabular"].detach().cpu())

        self.images = torch.cat(all_images, dim=0)
        self.tabular = torch.cat(all_tabular, dim=0)

        g = generator
        perm = torch.randperm(self.tabular.size(0), generator=g)
        self.tabular = self.tabular[perm]

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, idx: int):
        return {"image": self.images[idx], "tabular": self.tabular[idx]}


def make_incoherent_loader(src_loader: DataLoader, generator: Optional[torch.Generator] = None) -> DataLoader:
    """Return a DataLoader with the same images but shuffled tabular entries."""
    dataset = IncoherentPairDataset(src_loader, generator)
    return DataLoader(
        dataset,
        batch_size=src_loader.batch_size,
        shuffle=False,
        pin_memory=True,
    )


# Backward-compatible alias kept from earlier experiments.
def create_shuffled_tabular_loader(original_loader: DataLoader) -> DataLoader:
    """Deprecated wrapper; use `make_incoherent_loader` instead."""
    return make_incoherent_loader(original_loader)
