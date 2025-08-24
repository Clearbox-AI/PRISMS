# PRISMS\trainers\trainer_swarm.py
import torch, torch.nn as nn
from models.vae.vae import encode_images, decode_latents

class SwarmMultiModalModel(nn.Module):
    """
    One composite model that:
      • freezes the VAE
      • keeps MultiModalDiffusion trainable
      • exposes .sample() so inference code stays unchanged
    """
    def __init__(self, vae: nn.Module, diffusion: nn.Module, scaling_factor: float):
        super().__init__()
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.diffusion = diffusion          # all learnable params live here
        self.scaling   = scaling_factor

    # ---------------- training forward ----------------
    def forward(self, pixels, tabular, *, labels):
        latents = encode_images(self.vae, pixels, self.scaling)
        return self.diffusion(latents, tabular, labels=labels)

    # ---------------- inference helper ---------------
    @torch.no_grad()
    def sample(self, *args, **kw):
        z_img, z_tab = self.diffusion.sample(*args, **kw)
        imgs = decode_latents(self.vae, z_img, self.scaling)
        return imgs, z_tab
