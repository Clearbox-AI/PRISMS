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

# -----------------------------
# Model Definitions
# -----------------------------

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim=157, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Discriminator(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z_img, z_tab):
        z = torch.cat([z_img, z_tab], dim=1)
        return self.classifier(z)

# Training Function
def train_discriminator(dataloader, epochs=10, device='cuda', save_path=''):
    """
    Train the coherence discriminator on the provided dataloader.
    Args:
        dataloader: DataLoader yielding batches with 'image' and 'tabular'
        epochs: number of training epochs
        device: device to run the training on
        save_path: path to save the trained models
    Returns:
        image_encoder: trained ImageEncoder model
        tabular_encoder: trained TabularEncoder model
        discriminator: trained Discriminator model
    """
    image_encoder = ImageEncoder().to(device)
    tabular_encoder = TabularEncoder().to(device)
    discriminator = Discriminator().to(device)

    all_params = list(image_encoder.parameters()) + list(tabular_encoder.parameters()) + list(discriminator.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)

    for epoch in range(epochs):
        y_true_all = []
        y_pred_all = []
        total_loss = 0.0

        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_img = batch['image'].to(device)
            x_tab = batch['tabular'].to(device)
            B = x_img.size(0)

            # Positive pairs
            y_pos = torch.ones(B, 1).to(device)

            # Negative pairs: shuffled tabular
            perm = torch.randperm(B)
            x_tab_neg = x_tab[perm]
            y_neg = torch.zeros(B, 1).to(device)

            # Combine
            x_img_all = torch.cat([x_img, x_img], dim=0)
            x_tab_all = torch.cat([x_tab, x_tab_neg], dim=0)
            y_all = torch.cat([y_pos, y_neg], dim=0)

            # Forward
            z_img = image_encoder(x_img_all)
            z_tab = tabular_encoder(x_tab_all)
            y_pred = discriminator(z_img, z_tab)

            # Loss
            loss = F.binary_cross_entropy(y_pred, y_all)
            total_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true_all.extend(y_all.detach().cpu().numpy())
            y_pred_all.extend(y_pred.detach().cpu().numpy())

        auc = roc_auc_score(y_true_all, y_pred_all)
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - AUC: {auc:.4f}")

    # Save models
    save_path = os.path.join(save_path, f"coherence_discriminator_models_{epochs}epochs.pth")
    torch.save({
        'image_encoder': image_encoder.state_dict(),
        'tabular_encoder': tabular_encoder.state_dict(),
        'discriminator': discriminator.state_dict()
    }, save_path)

    print(f"Models saved to {save_path}")
    return image_encoder, tabular_encoder, discriminator

def load_discriminator_models(checkpoint_path, device='cuda'):
    """
    Load the trained discriminator models from a checkpoint.
    Args:
        checkpoint_path: path to the checkpoint file
        device: device to load the models on
    Returns:
        image_encoder: ImageEncoder model
        tabular_encoder: TabularEncoder model
        discriminator: Discriminator model
    """
    image_encoder = ImageEncoder().to(device)
    tabular_encoder = TabularEncoder().to(device)
    discriminator = Discriminator().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_encoder.load_state_dict(checkpoint['image_encoder'])
    tabular_encoder.load_state_dict(checkpoint['tabular_encoder'])
    discriminator.load_state_dict(checkpoint['discriminator'])

    image_encoder.eval()
    tabular_encoder.eval()
    discriminator.eval()

    print(f"Models loaded from {checkpoint_path}")
    return image_encoder, tabular_encoder, discriminator


def evaluate_synthetic_coherence(image_encoder, tabular_encoder, discriminator, synthetic_loader, device='cuda'):
    """
    Evaluate the coherence of synthetic data using the trained discriminator.
    Args:
        image_encoder: ImageEncoder model
        tabular_encoder: TabularEncoder model
        discriminator: Discriminator model
        synthetic_loader: DataLoader for synthetic data
        device: device to run the evaluation on
    Returns:
        List of coherence scores for the synthetic data.

    A score close to 0 indicates a mismatch, while a score close to 1 indicates a match.
    """
    image_encoder.eval()
    tabular_encoder.eval()
    discriminator.eval()

    y_preds = []
    with torch.no_grad():
        for batch in tqdm.tqdm(synthetic_loader, desc="Evaluating Synthetic Data"):
            x_img = batch['image'].to(device)
            x_tab = batch['tabular'].to(device)

            z_img = image_encoder(x_img)
            z_tab = tabular_encoder(x_tab)
            y_pred = discriminator(z_img, z_tab)  # (B, 1)

            y_preds.extend(y_pred.squeeze().tolist())

    return y_preds

def create_shuffled_tabular_loader(original_loader, device='cpu'):
    """
    Given a DataLoader yielding batches with 'image' and 'tabular',
    returns a new DataLoader with tabular data randomly shuffled.

    Args:
        original_loader: DataLoader yielding batches as dicts
        device: device to move data to (if needed)

    Returns:
        DataLoader with same images but shuffled tabular entries.
    """
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

#################################################################################

    use_pretrained_discriminator = True
    checkpoint_path_discriminator = 'PRISMS/metrics/coherence/checkpoints/'

    train_loader = load_training_data(cfg)
    synth_loader = load_training_data(cfg)
    shuffled_loader = create_shuffled_tabular_loader(train_loader, device=device)

    if use_pretrained_discriminator:
        image_encoder, tabular_encoder, discriminator = load_discriminator_models(os.path.join(checkpoint_path_discriminator, 'coherence_discriminator_models_100epochs.pth'), device)
    else:
        image_encoder, tabular_encoder, discriminator = train_discriminator(train_loader, epochs=100, device=device, save_path=checkpoint_path_discriminator)

    # Evaluate synthetic data
    coherence_scores = evaluate_synthetic_coherence(
        image_encoder, tabular_encoder, discriminator,
        synth_loader, device=device
    )
    coherence_scores_ = evaluate_synthetic_coherence(
        image_encoder, tabular_encoder, discriminator,
        shuffled_loader, device=device
    )

    import numpy as np
    print('Coherence Scores')
    print('Original data: ', np.mean(coherence_scores))
    print('Shuffled data: ', np.mean(coherence_scores_))