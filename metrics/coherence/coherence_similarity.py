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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

from coherence_discriminator import create_shuffled_tabular_loader

class ImageEncoder(nn.Module):
    """
    Simple CNN for (Bx1x256×256) grayscale image → latent vector. of shape (B, dim).
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 128×128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64×64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 32×32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, dim)
        )

    def forward(self, x):               # (B,1,256,256)
        z = self.net(x)
        return F.normalize(z, dim=1)    # L2-normalise


class TabularEncoder(nn.Module):
    """
    MLP for tabular features → latent vector (dim).
    The input is a (B,in_features) tensor.
    """
    def __init__(self, in_features: int = 157, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):               # (B,157)
        z = self.net(x)
        return F.normalize(z, dim=1)

def info_nce(img_emb, tab_emb, temperature: float = 0.07):
    """
    img_emb, tab_emb : (B, D) already L2-normalised
    returns scalar loss
    """
    logits = (img_emb @ tab_emb.T) / temperature     # (B,B)
    labels = torch.arange(img_emb.size(0), device=img_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)

def train_contrastive(loader: DataLoader,
                      epochs: int = 10,
                      dim: int = 128,
                      lr: float = 1e-4,
                      device: str = "cuda"):

    tab_dim = next(iter(loader))["tabular"].shape[1]

    img_enc = ImageEncoder(dim).to(device)
    tab_enc = TabularEncoder(in_features=tab_dim, dim=dim).to(device)
    opt = torch.optim.AdamW(list(img_enc.parameters()) +
                            list(tab_enc.parameters()), lr=lr)

    for epoch in range(epochs):
        running = 0.0
        for batch in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = batch["image"].to(device)          # (B,1,256,256)
            tab = batch["tabular"].to(device)        # (B,157)

            z_img = img_enc(img)
            z_tab = tab_enc(tab)
            loss  = info_nce(z_img, z_tab)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        print(f"epoch {epoch+1}: loss {running/len(loader):.4f}")

    return img_enc, tab_enc

@torch.no_grad()
def similarity_scores(img_enc, tab_enc, loader: DataLoader,
                      device: str = "cuda"):
    """
    Compute list of cosine similarities (one per pair in loader).
    Args:
    -----------
        img_enc, tab_enc : encoders (already trained)
        loader : DataLoader with (image, tabular) pairs
        device : device to use for computation
    Returns:
    -----------
        list of cosine similarities
    """
    img_enc.eval(); tab_enc.eval()
    sims = []
    for batch in tqdm.tqdm(loader, desc="scoring"):
        img = batch["image"].to(device)
        tab = batch["tabular"].to(device)

        z_img = img_enc(img)           # (B,D)
        z_tab = tab_enc(tab)           # (B,D)

        # cosine of corresponding rows
        c = abs(F.cosine_similarity(z_img, z_tab, dim=1)) # abs(cosine_similarity) for [0,1] range
        sims.extend(c.cpu().tolist())
    return sims

if __name__ == "__main__":
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_flag = True
    checkpoint_path = 'PRISMS/metrics/coherence/checkpoints/'
    contrastive_weights = 'coherence_discriminator_models_100epochs.pth'
    checkpoint_path_contrastive_weights = os.path.join(checkpoint_path, contrastive_weights)

    train_loader = load_training_data(cfg)
    synth_loader = load_training_data(cfg)
    shuffled_loader = create_shuffled_tabular_loader(train_loader, device=device)

    # training on ORIGINAL (coherent) data 
    img_enc, tab_enc = train_contrastive(train_loader,
                                        epochs=5, dim=128, device='cuda')

    # similarity on ORIGINAL  
    real_sim = similarity_scores(img_enc, tab_enc, train_loader)
    # similarity on SYNTHETIC 
    synth_sim = similarity_scores(img_enc, tab_enc, shuffled_loader)
    
    print(f"mean real similarity: {sum(real_sim)/len(real_sim):.3f}")
    print(f"mean synthetic similarity: {sum(synth_sim)/len(synth_sim):.3f}")
