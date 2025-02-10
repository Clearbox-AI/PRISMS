# models/autoencoder.py
import torch
from diffusers import AutoencoderKL

def load_stable_diffusion_xl_vae(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder: str = "vae",
    device: str = "cuda",
    dtype_str: str = "float32"
):
    dtype = getattr(torch, dtype_str)
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder=subfolder,
        torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)
    return vae