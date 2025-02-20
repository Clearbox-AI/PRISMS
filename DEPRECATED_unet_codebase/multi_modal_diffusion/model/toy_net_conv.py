import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1) Basic residual block for images (same as before)
# ------------------------------------------------------------------
class ImageResBlock(nn.Module):
    """
    A simple residual block for image (CNN):
    - 2 convolutional layers with BatchNorm and ReLU
    - Residual (shortcut) connection
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Match residual (shortcut) shape
        residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out

# ------------------------------------------------------------------
# 2) Basic residual block for tabular data (MLP)
# ------------------------------------------------------------------
class TabularResBlock(nn.Module):
    """
    A simple residual block for tabular data (MLP):
    - 2 linear layers with BatchNorm and ReLU
    - Residual (shortcut) connection
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # If dimension changes, use a 1x1 linear for the shortcut
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        # x shape: [batch, in_dim]
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # Match residual shape
        if isinstance(self.shortcut, nn.Linear):
            residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)
        return out

# ------------------------------------------------------------------
# 3) The multi-modal denoising network
# ------------------------------------------------------------------
class ToyConv(nn.Module):
    """
    A multi-modal denoising network that:
      - Processes an image of shape [B, 4, 64, 64] via a CNN stream.
      - Processes tabular data of shape [B, 174] via an MLP stream.
      - Conditions on a diffusion timestep 't' via a small MLP for time embedding.
      - Returns a tuple (image_out, tabular_out) with same shapes.
    """
    def __init__(
        self,
        image_in_channels=3,
        image_out_channels=3,
        image_hidden_dim=32,
        tab_in_dim=174,
        tab_hidden_dim=174,
        time_embed_dim=64
    ):
        super().__init__()

        self.image_size = tuple(int(x) for x in "3,64,64".split(','))
        self.tabular_size = 174
        # ----------------------------------------------------------
        # A) Time Embedding
        # ----------------------------------------------------------
        # We embed the timestep 't' (a scalar or 1D vector) to a feature vector
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # ----------------------------------------------------------
        # B) Image Stream
        # ----------------------------------------------------------

        # 1. Initial convolution to get hidden_dim channels
        self.img_input_conv = nn.Conv2d(
            image_in_channels, image_hidden_dim, kernel_size=3, padding=1
        )

        # 2. Residual blocks for images
        self.img_block1 = ImageResBlock(image_hidden_dim, image_hidden_dim)
        self.img_time_proj1 = nn.Linear(time_embed_dim, image_hidden_dim)

        self.img_block2 = ImageResBlock(image_hidden_dim, image_hidden_dim)
        self.img_time_proj2 = nn.Linear(time_embed_dim, image_hidden_dim)

        self.img_block3 = ImageResBlock(image_hidden_dim, image_hidden_dim)
        self.img_time_proj3 = nn.Linear(time_embed_dim, image_hidden_dim)

        # 3. Final convolution to project back to original image channels
        self.img_output_conv = nn.Conv2d(
            image_hidden_dim, image_out_channels, kernel_size=3, padding=1
        )

        # ----------------------------------------------------------
        # C) Tabular Stream
        # ----------------------------------------------------------

        # 1. Initial FC layer to embed the tabular data
        self.tab_input_fc = nn.Linear(tab_in_dim, tab_hidden_dim)

        # 2. Residual blocks for tabular data
        self.tab_block1 = TabularResBlock(tab_hidden_dim, tab_hidden_dim)
        self.tab_time_proj1 = nn.Linear(time_embed_dim, tab_hidden_dim)

        self.tab_block2 = TabularResBlock(tab_hidden_dim, tab_hidden_dim)
        self.tab_time_proj2 = nn.Linear(time_embed_dim, tab_hidden_dim)

        self.tab_block3 = TabularResBlock(tab_hidden_dim, tab_hidden_dim)
        self.tab_time_proj3 = nn.Linear(time_embed_dim, tab_hidden_dim)

        # 3. Final FC layer to project back to original tab shape
        self.tab_output_fc = nn.Linear(tab_hidden_dim, tab_in_dim)

    def forward(self, x_img, x_tab, t, **kwargs):
        """
        Forward pass:
          x_img: [batch, 4, 64, 64]
          x_tab: [batch, 174]
          t:     [batch] or scalar (scaled timesteps)
        Returns:
          (img_out, tab_out)
        """
        if x_img.dtype != torch.float32:
            x_img = x_img.float()
        if x_tab.dtype != torch.float32:
            x_tab = x_tab.float()
        if t.dtype != torch.float32:
            t = t.float()


        # 1) Embed time => shape: [batch, time_embed_dim]
        #    Ensure t is a float tensor of shape [batch, 1].
        t_emb = self.time_mlp(t.unsqueeze(-1).float())

        # --------------------------------------
        # 2) Image Stream
        # --------------------------------------
        # Initial projection
        h_img = self.img_input_conv(x_img)  # -> [batch, image_hidden_dim, 64, 64]

        # Block 1 + time injection
        time_bias1_img = self.img_time_proj1(t_emb)[:, :, None, None]
        h_img = self.img_block1(h_img + time_bias1_img)

        # Block 2
        time_bias2_img = self.img_time_proj2(t_emb)[:, :, None, None]
        h_img = self.img_block2(h_img + time_bias2_img)

        # Block 3
        time_bias3_img = self.img_time_proj3(t_emb)[:, :, None, None]
        h_img = self.img_block3(h_img + time_bias3_img)

        # Final projection to match original image channels
        img_out = self.img_output_conv(h_img)  # -> [batch, 4, 64, 64]

        # --------------------------------------
        # 3) Tabular Stream
        # --------------------------------------
        # Initial projection
        h_tab = self.tab_input_fc(x_tab)  # -> [batch, tab_hidden_dim]

        # Block 1 + time injection
        time_bias1_tab = self.tab_time_proj1(t_emb)  # shape [batch, tab_hidden_dim]
        h_tab = self.tab_block1(h_tab + time_bias1_tab)

        # Block 2
        time_bias2_tab = self.tab_time_proj2(t_emb)
        h_tab = self.tab_block2(h_tab + time_bias2_tab)

        # Block 3
        time_bias3_tab = self.tab_time_proj3(t_emb)
        h_tab = self.tab_block3(h_tab + time_bias3_tab)

        # Final projection to match original tab shape
        tab_out = self.tab_output_fc(h_tab)  # -> [batch, 174]

        # Return both outputs as a tuple
        return (img_out, tab_out)


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # A) Create random inputs
    batch_size = 2
    x_img = torch.randn(batch_size, 4, 64, 64)   # image
    x_tab = torch.randn(batch_size, 174)         # tabular
    t     = torch.tensor([10.0, 20.0])           # timesteps

    # B) Initialize model
    model = ToyConv()

    # C) Forward pass
    img_out, tab_out = model(x_img, x_tab, t)

    print("img_out shape:", img_out.shape)  # [2, 4, 64, 64]
    print("tab_out shape:", tab_out.shape)  # [2, 174]
