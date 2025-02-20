import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------
#  Time Embedding
# ------------------
class TimeEmbed(nn.Module):
    """
    A simple sinusoidal time embedding followed by a Linear layer.
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.lin = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        """
        t: [batch]
        Produces an embedding of shape [batch, 1, embed_dim].
        """
        half_dim = self.embed_dim // 2
        # Exponential frequency schedule
        freqs = torch.exp(
            torch.linspace(
                math.log(10000) / (half_dim - 1),
                0,
                half_dim
            )
        ).to(t.device)

        # Expand t to [batch, 1], then create the sinusoidal/cosine parts
        t = t.unsqueeze(1)  # [batch, 1]
        sin_part = torch.sin(t * freqs)
        cos_part = torch.cos(t * freqs)

        # Concatenate and project
        emb = torch.cat([sin_part, cos_part], dim=1)  # [batch, embed_dim]
        emb = self.lin(emb)  # [batch, embed_dim]

        # Return as [batch, 1, embed_dim] to act as a single "token"
        return emb.unsqueeze(1)


# ------------------
#  Image Embedding
# ------------------
class PatchEmbed(nn.Module):
    """
    Naive patch embedding using a strided Conv2d.
    Splits the image into (patch_size x patch_size) patches and
    projects each patch into an embedding of size embed_dim.
    """

    def __init__(self, in_channels=3, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: [batch, 4, 64, 64]
        Returns patch tokens of shape [batch, num_patches, embed_dim].
        - If image_wh_size=64 and patch_size=8, num_patches = 8 * 8 = 64.
        """
        # Shape -> [batch, embed_dim, h/patch_size, w/patch_size]
        x = self.proj(x)  # e.g. -> [batch, embed_dim, 8, 8]
        # Flatten spatial dims -> [batch, embed_dim, 64]
        x = x.flatten(2)
        # Transpose to [batch, 64, embed_dim]
        return x.transpose(1, 2)


# ------------------
#  Tabular Embedding
# ------------------
class TabEmbed(nn.Module):
    """
    Single-token embedding for the entire tabular vector.
    """

    def __init__(self, in_dim=174, embed_dim=128):
        super().__init__()
        self.lin = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        """
        x: [batch, 174]
        Returns shape [batch, 1, embed_dim].
        """
        x = self.lin(x)  # [batch, embed_dim]
        return x.unsqueeze(1)  # [batch, 1, embed_dim]


# ------------------
#  Transformer Block
# ------------------
class SimpleTransformer(nn.Module):
    """
    A small TransformerEncoder that operates on the concatenated tokens:
    [time_token, image_patch_tokens, tabular_token].
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # A learnable positional embedding for a fixed maximum sequence length.
        # Here, we assume a maximum of:
        #   1 time token + 64 image patches + 1 tab token = 66 tokens.
        max_seq_len = 66
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

    def forward(self, tokens):
        """
        tokens: [batch, seq_len, embed_dim], where seq_len <= 66 in this example.
        """
        B, L, E = tokens.shape
        # Add learnable positional embeddings (truncate to sequence length L).
        tokens = tokens + self.pos_emb[:, :L, :]

        # Pass through Transformer
        out = self.transformer(tokens)  # [batch, seq_len, embed_dim]
        return out


# ------------------
#  Main Model
# ------------------
class ToyTransf(nn.Module):
    """
    Elementary transformer for noise prediction in a diffusion framework.
    Expects:
      - image_t:  [batch, 4, 64, 64]
      - tabular_t:[batch, 174]
      - t:        [batch] (timestep)
    Returns:
      - image_out:   [batch, 4, 64, 64]
      - tabular_out: [batch, 174]
    """

    def __init__(
            self,
            image_channels=3,
            image_wh_size=64,
            tab_dim=174,
            embed_dim=128,
            patch_size=8,
            num_heads=4,
            num_layers=2
    ):
        super().__init__()

        self.image_size = tuple(int(x) for x in "3,64,64".split(','))
        self.tabular_size = 174

        # Embeddings
        self.patch_embed = PatchEmbed(
            in_channels=image_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        self.tab_embed = TabEmbed(in_dim=tab_dim, embed_dim=embed_dim)
        self.time_embed = TimeEmbed(embed_dim=embed_dim)

        # Transformer
        self.transformer = SimpleTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # Decode / project back to original shapes
        # For image, use a ConvTranspose to "unpatch" the tokens
        self.image_decoder = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=image_channels,
            kernel_size=patch_size,
            stride=patch_size
        )
        # For tabular data, a simple linear layer
        self.tab_decoder = nn.Linear(embed_dim, tab_dim)

        # Remember shapes
        self.patch_size = patch_size
        self.image_wh_size = image_wh_size
        self.tab_dim = tab_dim

        # Number of patches (for splitting back out of the Transformer)
        self.num_patches = (image_wh_size // patch_size) * (image_wh_size // patch_size)

    def forward(self, image_t, tabular_t, t, **kwargs):
        """
        image_t:  [batch, 4, 64, 64]
        tabular_t:[batch, 174]
        t:        [batch]

        Returns:
          image_out:   [batch, 4, 64, 64]
          tabular_out: [batch, 174]
        """

        if image_t.dtype != torch.float32:
            image_t = image_t.float()
        if tabular_t.dtype != torch.float32:
            tabular_t = tabular_t.float()
        if t.dtype != torch.float32:
            t = t.float()

        B = image_t.size(0)

        # 1) Embed inputs
        img_tokens = self.patch_embed(image_t)  # [batch, 64, embed_dim] if patch_size=8
        tab_token = self.tab_embed(tabular_t)  # [batch, 1,  embed_dim]
        time_token = self.time_embed(t)  # [batch, 1,  embed_dim]

        # 2) Concat them as a sequence of tokens
        #    Layout: [time_token, image_tokens, tab_token]
        tokens = torch.cat([time_token, img_tokens, tab_token], dim=1)
        # Expected shape: [batch, 66, embed_dim] if patch_size=8

        # 3) Pass through Transformer
        out = self.transformer(tokens)  # [batch, 66, embed_dim]

        # 4) Split back out
        #    time_out = out[:, 0, :]      (you could use it if you want)
        img_out = out[:, 1: 1 + self.num_patches, :]  # [batch, 64, embed_dim]
        tab_out = out[:, 1 + self.num_patches:, :]  # [batch, 1,  embed_dim]

        # 5) Decode image
        #    First reshape [batch, 64, embed_dim] -> [batch, embed_dim, 8, 8]
        img_out = img_out.transpose(1, 2)  # [batch, embed_dim, 64]
        img_out = img_out.view(
            B,
            -1,
            self.image_wh_size // self.patch_size,
            self.image_wh_size // self.patch_size
        )  # e.g. -> [batch, embed_dim, 8, 8]

        image_out = self.image_decoder(img_out)  # -> [batch, 4, 64, 64]

        # 6) Decode tabular
        tab_out = tab_out.squeeze(1)  # [batch, embed_dim]
        tabular_out = self.tab_decoder(tab_out)  # [batch, 174]

        return image_out, tabular_out

