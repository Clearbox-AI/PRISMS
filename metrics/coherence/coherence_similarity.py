import os
from pathlib import Path
from typing import Tuple
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

def train_similarity(loader: DataLoader,
                      epochs: int,
                      dim: int,
                      lr: float,
                      device: str,
                      save_path: str) -> Tuple[nn.Module, nn.Module]:
    """
    Train image and tabular encoders using similarity loss.

    Args:
    -----------
        loader : DataLoader with (image, tabular) pairs
        epochs : number of training epochs
        dim : latent vector size
        lr : learning rate
        device : device to use for computation
        save_path : path to save the trained encoders
    Returns:
    -----------
        img_enc, tab_enc : trained encoders
    """

    tab_dim  = next(iter(loader))["tabular"].shape[1]   # 157 in your case
    img_enc  = ImageEncoder(dim).to(device)
    tab_enc  = TabularEncoder(tab_dim, dim).to(device)

    opt = torch.optim.AdamW(list(img_enc.parameters()) +
                            list(tab_enc.parameters()), lr=lr)

    for ep in range(epochs):
        running = 0.0
        for batch in tqdm.tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"):
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)

            loss = info_nce(img_enc(img), tab_enc(tab))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        print(f"epoch {ep+1}: loss {running/len(loader):.4f}")

    # Save both encoders in one file
    torch.save({"img_enc": img_enc.state_dict(),
                "tab_enc": tab_enc.state_dict()}, save_path)
    print(f"✓ encoders saved → {os.path.abspath(save_path)}")
    return img_enc, tab_enc


def load_encoders(checkpoint: str, dim: int, device: str):
    """
    Load encoders from checkpoint.
    
    Args:
    -----------
        checkpoint : path to saved model weights
        dim : latent vector size
        device : device to use for computation
    Returns:
    -----------
        img_enc, tab_enc : encoders (already trained)
    """
    state   = torch.load(checkpoint, map_location=device)
    img_enc = ImageEncoder(dim).to(device)
    tab_dim = state["tab_enc"]["net.0.weight"].shape[1]  # recover input size
    tab_enc = TabularEncoder(tab_dim, dim).to(device)

    img_enc.load_state_dict(state["img_enc"])
    tab_enc.load_state_dict(state["tab_enc"])
    img_enc.eval(); tab_enc.eval()
    print(f"✓ encoders loaded  ← {os.path.abspath(checkpoint)}")
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

    checkpoint_path = 'PRISMS/metrics/coherence/checkpoints/'
    similarity_weights = 'coherence_similarity_models_10epochs.pth'
    checkpoint_path_similarity_weights = os.path.join(checkpoint_path, similarity_weights)

    train_flag = True
    dim         = 128
    epochs      = 10
    lr          = 1e-4

    train_loader = load_training_data(cfg)
    synth_loader = load_training_data(cfg)
    shuffled_loader = create_shuffled_tabular_loader(train_loader, device=device)

    if train_flag:
        img_enc, tab_enc = train_similarity(train_loader,
                                             epochs=epochs,
                                             dim=dim,
                                             lr=lr,
                                             device=device,
                                             save_path=checkpoint_path_similarity_weights)
    else:
        img_enc, tab_enc = load_encoders(checkpoint_path_similarity_weights, dim, device)

    # example evaluation (real vs synthetic)
    real_sim  = similarity_scores(img_enc, tab_enc, train_loader,  device)
    synth_sim = similarity_scores(img_enc, tab_enc, shuffled_loader, device)
    print(f"mean real   similarity: {sum(real_sim)/len(real_sim):.3f}")
    print(f"mean synthetic similarity: {sum(synth_sim)/len(synth_sim):.3f}")
