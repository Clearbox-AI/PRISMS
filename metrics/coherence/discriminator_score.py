import os
from pathlib import Path
import sys
prisms_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(prisms_path)

from data.loader import load_training_data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.configurations import set_project_root

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import tqdm

class ImageEncoder(nn.Module):
    """
    Simple CNN for encoding images.
    The input dimension is (1, 256, 256) and the output dimension is 128.
    """
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (B,16,128,128)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (B,32,64,64)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (B,64,32,32)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class TabularEncoder(nn.Module):
    """
    Simple MLP for encoding tabular data.
    The input dimension is 157 (as per the original code) and the output dimension is 128.
    """
    def __init__(self, input_dim: int = 157, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Discriminator(nn.Module):
    """
    Full coherence-discriminator pipeline:
    image_encoder + tabular_encoder + classifier.
    The classifier is a simple MLP that takes the concatenated
    outputs of the image and tabular encoders. 
    The model is trained to distinguish between coherent
    (image, tabular) pairs and incoherent pairs.
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()

        self.image_encoder   = ImageEncoder(output_dim=embed_dim)
        self.tabular_encoder = TabularEncoder(output_dim=embed_dim)
        self.classifier      = nn.Sequential(
            nn.Linear(2 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z_img, z_tab):
        z = torch.cat([z_img, z_tab], dim=1)   # (B, 2*embed_dim)
        return self.classifier(z)              # (B, 1)  prob of “coherent”

    def fit(self,
              dataloader,
              epochs: int = 10,
              save_path: str = "coherence_discriminator.pth",
              lr: float = 1e-4):
        """
        Train the coherence discriminator on the provided dataloader.
        The dataloader should yield batches with 'image' and 'tabular' keys.
        The model is saved to `save_path` after training.
        
        Args:
            dataloader: DataLoader yielding batches as dicts
            epochs: number of training epochs
            save_path: path to save the trained model
            lr: learning rate for the optimizer
        """
        self.to(self.device)          # make sure all submodules are on the same device
        self.image_encoder.train()
        self.tabular_encoder.train()
        super().train()               # sets classifier to train mode

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            y_true_all, y_pred_all = [], []
            total_loss = 0.0

            for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x_img = batch['image'].to(self.device)     # (B,1,256,256)
                x_tab = batch['tabular'].to(self.device)   # (B,157)
                B     = x_img.size(0)

                # build positive / negative pairs
                y_pos  = torch.ones(B, 1, device=self.device)
                perm   = torch.randperm(B, device=self.device)
                x_tab_neg = x_tab[perm]
                y_neg  = torch.zeros(B, 1, device=self.device)

                x_img_all = torch.cat([x_img, x_img], dim=0)          # 2B
                x_tab_all = torch.cat([x_tab, x_tab_neg], dim=0)      # 2B
                y_all     = torch.cat([y_pos, y_neg],  dim=0)         # 2B,1

                # forward
                z_img = self.image_encoder(x_img_all)
                z_tab = self.tabular_encoder(x_tab_all)
                y_pred = self(z_img, z_tab)

                loss = F.binary_cross_entropy(y_pred, y_all)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_true_all.extend(y_all.detach().cpu().numpy())
                y_pred_all.extend(y_pred.detach().cpu().numpy())

            auc = roc_auc_score(y_true_all, y_pred_all)
            print(f"Epoch {epoch+1}/{epochs}  |  loss={total_loss:.4f}  |  AUC={auc:.4f}")

        torch.save(self.state_dict(), save_path)
        print(f"Full model saved to →  {save_path}")

    def _load_discriminator_models(self,
                                   checkpoint_path: str):
        """
        Load the full saved state_dict (encoders + classifier).

        Args:
            checkpoint_path: path to saved model weights
        """
        state = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(state)
        self.to(self.device)
        self.eval()
        self.image_encoder.eval()
        self.tabular_encoder.eval()
        print(f"Full model loaded from ←  {checkpoint_path}")

    @torch.no_grad()
    def evaluate(self,
                 loader,
                 checkpoint_path: str | None = None):
        """
        Evaluate coherence scores (probabilities ∈ [0,1]) for every
        (image, tabular) pair in `loader`.
        If `checkpoint_path` is supplied, the model weights are loaded first.

        Args:
            loader: DataLoader yielding batches as dicts
            checkpoint_path: path to saved model weights (optional)
        Returns:
            List of coherence scores for each batch in the loader.
        """
        if checkpoint_path:
            self._load_discriminator_models(checkpoint_path)

        self.eval()
        self.image_encoder.eval()
        self.tabular_encoder.eval()

        scores = []
        for batch in tqdm.tqdm(loader, desc="Evaluating"):
            x_img = batch['image'].to(self.device)
            x_tab = batch['tabular'].to(self.device)

            z_img = self.image_encoder(x_img)
            z_tab = self.tabular_encoder(x_tab)
            y_pred = self(z_img, z_tab)           # (B,1)
            scores.extend(y_pred.squeeze().tolist())

        return scores

def create_shuffled_tabular_loader(original_loader):
    """
    Given a DataLoader yielding batches with 'image' and 'tabular',
    returns a new DataLoader with tabular data randomly shuffled.

    Args:
        original_loader: DataLoader yielding batches as dicts
        device: device to move data to (if needed)

    Returns:
        DataLoader with same images but shuffled tabular entries.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_images = []
    all_tabular = []

    # Collect entire dataset into memory
    for batch in original_loader:
        all_images.append(batch['image'])
        all_tabular.append(batch['tabular'])

    images_tensor = torch.cat(all_images, dim=0)  # (N, 1, 256, 256)
    tabular_tensor = torch.cat(all_tabular, dim=0)  # (N, 157)

    # Shuffle only the tabular data
    shuffled_indices = torch.randperm(tabular_tensor.size(0))
    shuffled_tabular_tensor = tabular_tensor[shuffled_indices]

    # Wrap in TensorDataset
    dataset = TensorDataset(images_tensor.to(device), shuffled_tabular_tensor.to(device))

    # Return a DataLoader yielding the same keys as original batches
    def collate_fn(batch):
        imgs, tabs = zip(*batch)
        return {
            'image': torch.stack(imgs),
            'tabular': torch.stack(tabs)
        }

    shuffled_loader = DataLoader(dataset, batch_size=original_loader.batch_size,
                                 shuffle=False, collate_fn=collate_fn)
    return shuffled_loader

if __name__ == "__main__":
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_flag = True
    checkpoint_path = 'PRISMS/metrics/coherence/checkpoints/'
    discriminator_weights = 'coherence_discriminator_models_100epochs.pth'
    checkpoint_path_discriminator = os.path.join(checkpoint_path, discriminator_weights)

    train_loader = load_training_data(cfg)
    synth_loader = load_training_data(cfg)
    shuffled_loader = create_shuffled_tabular_loader(train_loader)

    discriminator = Discriminator()
    if train_flag:
        discriminator.fit(train_loader, epochs=100, save_path=checkpoint_path_discriminator)
        coherence_scores = discriminator.evaluate(synth_loader)
        coherence_scores_ = discriminator.evaluate(shuffled_loader)
    else:
        coherence_scores = discriminator.evaluate(synth_loader, checkpoint_path=checkpoint_path_discriminator)
        coherence_scores_ = discriminator.evaluate(shuffled_loader, checkpoint_path=checkpoint_path_discriminator)

    import numpy as np
    print('Coherence Scores')
    print('Original data: ', np.mean(coherence_scores))
    print('Shuffled data: ', np.mean(coherence_scores_))