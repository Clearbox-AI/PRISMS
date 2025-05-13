import os
from pathlib import Path
from typing import Tuple, List, Optional
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

from discriminator_score import create_shuffled_tabular_loader

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Similarity:
    """
    Wrapper class that can either:
     - train new encoders and save them, or
     - load existing encoders from disk,
    then compute cosine similarities on a DataLoader.

    If the final score is >0.75 the image-tabular pairs are on average correctly matched.
    """
    def __init__(self,
                 dim: int = 128,
                 lr: float = 1e-4):
        self.dim      = dim
        self.lr       = lr
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_enc: Optional[nn.Module] = None
        self.tab_enc: Optional[nn.Module] = None

    def evaluate(self,
                 loader:       DataLoader,
                 checkpoint:   str,
                 train_flag:   bool = False,
                 epochs:       int  = 10) -> float:
        """
        - If `train_flag` is True  → trains encoders with `epochs`
          and saves them to `checkpoint`.
        - If `train_flag` is False → loads encoders from `checkpoint`.

        Args:
            loader: DataLoader yielding batches as dicts
            checkpoint: path to save/load the trained model
            train_flag: whether to train new encoders or load existing ones
            epochs: number of training epochs (only used if `train_flag` is True)
        Returns:
            mean similarity score: average cosine similarity between mapped between 0 and 1.
        """
        if train_flag:
            self._train_similarity(loader, epochs, checkpoint)
        else:
            self._load_encoders(checkpoint)

        return self._similarity_scores(loader)

    def _train_similarity(self,
                          loader: DataLoader,
                          epochs: int,
                          save_path: str):
        """
        Train encoders on the provided dataloader.
        The dataloader should yield batches with 'image' and 'tabular' keys.
        The model is saved to `save_path` after training.

        Args:
            loader: DataLoader yielding batches as dicts
            epochs: number of training epochs
            save_path: path to save the trained model
        """
        tab_dim = next(iter(loader))["tabular"].shape[1]
        self.img_enc = ImageEncoder(self.dim).to(self.device)
        self.tab_enc = TabularEncoder(tab_dim, self.dim).to(self.device)

        optim = torch.optim.AdamW(
            list(self.img_enc.parameters()) + list(self.tab_enc.parameters()),
            lr=self.lr)

        for ep in range(epochs):
            total = 0.0
            for batch in tqdm.tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"):
                img = batch["image"].to(self.device)
                tab = batch["tabular"].to(self.device)

                loss = info_nce(self.img_enc(img), self.tab_enc(tab))
                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()

            print(f"epoch {ep+1}: loss {total/len(loader):.4f}")

        # save weights
        torch.save({"img_enc": self.img_enc.state_dict(),
                    "tab_enc": self.tab_enc.state_dict()},
                   save_path)
        print(f"✓ encoders saved → {os.path.abspath(save_path)}")

    def _load_encoders(self, checkpoint: str):
        """
        Load encoders from the provided checkpoint.
        The checkpoint should contain the state_dicts of the encoders.
        Args:
            checkpoint: path to the checkpoint file
        """
        state = torch.load(checkpoint, map_location=self.device)
        self.img_enc = ImageEncoder(self.dim).to(self.device)
        tab_dim = state["tab_enc"]["net.0.weight"].shape[1]
        self.tab_enc = TabularEncoder(tab_dim, self.dim).to(self.device)

        self.img_enc.load_state_dict(state["img_enc"])
        self.tab_enc.load_state_dict(state["tab_enc"])
        self.img_enc.eval(); self.tab_enc.eval()
        print(f"✓ encoders loaded ← {os.path.abspath(checkpoint)}")

    @torch.no_grad()
    def _similarity_scores(self,
                           loader: DataLoader) -> float:
        """
        Cosine similarities mapped to [0,1].
        Args:
            loader: DataLoader yielding batches as dicts
        Returns:
            Mean cosine similarities for each batch in the loader.
        """
        self.img_enc.eval(); self.tab_enc.eval()
        sims = []
        for batch in tqdm.tqdm(loader, desc="scoring"):
            img = batch["image"].to(self.device)
            tab = batch["tabular"].to(self.device)

            z_img = self.img_enc(img)
            z_tab = self.tab_enc(tab)
            cos   = F.cosine_similarity(z_img, z_tab, dim=1)  # [-1,1]
            sims.extend(((cos + 1) / 2).cpu().tolist())        # cosine similarity normalization → [0,1]
        return sum(sims)/len(sims)

if __name__ == "__main__":
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        cfg = compose(config_name="nacc")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

    # params
    checkpoint_path = 'PRISMS/metrics/coherence/checkpoints/'
    similarity_weights = 'coherence_similarity_models_10epochs.pth'
    checkpoint_path_similarity_weights = os.path.join(checkpoint_path, similarity_weights)
    train_flag = False
    dim         = 128
    epochs      = 10
    lr          = 1e-4

    train_loader = load_training_data(cfg)
    shuffled_loader = create_shuffled_tabular_loader(train_loader)

    similarity = Similarity(dim=128, lr=1e-4)
    real_sim = similarity.evaluate(loader  = train_loader,
                             checkpoint    = checkpoint_path_similarity_weights,
                             train_flag    = train_flag)
    
    synth_sim = similarity.evaluate(loader = shuffled_loader,
                             checkpoint    = checkpoint_path_similarity_weights,
                             train_flag    = False)
    
    print(f"mean real   similarity: {real_sim:.3f}")
    print(f"mean synthetic similarity: {synth_sim:.3f}")
