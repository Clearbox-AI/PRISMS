# prisms_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# sys.path.append(prisms_path)

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
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Discriminator(nn.Module):
    def __init__(self, embed_dim: int = 128, tabular_dim: int = 158):
        super().__init__()
        self.image_encoder   = ImageEncoder(output_dim=embed_dim)
        self.tabular_encoder = TabularEncoder(input_dim=tabular_dim,
                                              output_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2*embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z_img, z_tab):
        z = torch.cat([z_img, z_tab], dim=1)   # (B, 2*embed_dim)
        return self.classifier(z)              # (B, 1)  prob of “coherent”

    def _fit(self,
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
                x_tab = batch['tabular'].to(self.device)   # (B,158)
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
        print(f"✓ Full model saved to →  {save_path}")

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
        print(f"✓ Full model loaded from ←  {checkpoint_path}")

    def evaluate(self,
                 loader,
                 checkpoint_path: str | None = None,
                 train_flag:   bool = False,
                 epochs:       int  = 10
        ) -> float:
        """
        Evaluate coherence scores (probabilities ∈ [0,1]) for every
        (image, tabular) pair in `loader`.
        If `train_flag` is True, the model is trained on the provided
        dataloader for `epochs` epochs and saved to `checkpoint_path`.
        If `train_flag` is False, the model is loaded from `checkpoint_path`.
        The model is set to evaluation mode.
        The dataloader should yield batches with 'image' and 'tabular' keys.
        
        Args:
            loader: DataLoader yielding batches as dicts
            checkpoint_path: path to saved model weights
            train_flag: if True, train the model; if False, load the model
            epochs: number of training epochs (only used if train_flag is True)
        Returns:
            Average coherence score for the entire dataset.
        """
        if train_flag:
            self._fit(loader, epochs, checkpoint_path)
        else:
            self._load_discriminator_models(checkpoint_path)

        with torch.no_grad():
            return self._discriminator_score(loader)

    @torch.no_grad()
    def _discriminator_score(self, loader: DataLoader) -> float:
        """
        Compute the coherence score for the provided dataloader.
        The dataloader should yield batches with 'image' and 'tabular' keys.
        The model is set to evaluation mode.
        
        Args:
            loader: DataLoader yielding batches as dicts
        Returns:
            Average coherence score for the entire dataset.
        """
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

        return sum(scores)/len(scores)

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
    tabular_tensor = torch.cat(all_tabular, dim=0)  # (N, 158)

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






import torch
from torch.utils.data import Dataset, DataLoader

class IncoherentPairDataset(Dataset):
    """
    Wraps an existing dataset that yields dicts with keys
    'image' and 'tabular'.  The tabular branch is shuffled
    once at construction time.
    """
    def __init__(self, src_loader: DataLoader, generator: torch.Generator | None = None):
        # 1. materialise the tensors on *CPU*
        imgs, tabs = [], []
        for batch in src_loader:
            imgs.append(batch["image"].cpu())
            tabs.append(batch["tabular"].cpu())
        self.images  = torch.cat(imgs,  dim=0)         # (N, C, H, W)
        self.tabular = torch.cat(tabs, dim=0)          # (N, F)

        # 2. make a one-time permutation of the tabular rows
        g = generator or torch.Generator()
        perm = torch.randperm(len(self.tabular), generator=g)   # :contentReference[oaicite:2]{index=2}
        self.tabular = self.tabular[perm]

    def __len__(self):  return len(self.images)

    def __getitem__(self, idx: int):
        return {"image": self.images[idx],
                "tabular": self.tabular[idx]}

def make_incoherent_loader(src_loader: DataLoader,
                           generator: torch.Generator | None = None):
    dataset = IncoherentPairDataset(src_loader, generator)
    return DataLoader(dataset,
                      batch_size=src_loader.batch_size,
                      shuffle=False,                   # already incoherent
                      pin_memory=True)                 # faster GPU copy :contentReference[oaicite:3]{index=3}
