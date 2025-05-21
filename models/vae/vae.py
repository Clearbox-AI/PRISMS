import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from omegaconf import DictConfig
from typing import Any

# from utils.configurations import apply_overrides
from utils.configurations import _merge_cfg


# def load_vae(cfg: DictConfig, **overrides: Any) -> nn.Module:
#     """
#     Load a VAE model (specifically an AutoencoderKL from diffusers) based on the provided configuration.
#
#     Args:
#         cfg (DictConfig): The Hydra configuration object for the VAE.
#         **overrides (Any): Arbitrary keyword arguments used to override the default configuration.
#
#     Returns:
#         nn.Module: The loaded VAE model.
#     """
#     # Apply any overrides to the config before loading
#     cfg = apply_overrides(cfg, overrides)
#
#     print("[INFO] Loading VAE model with config:", cfg)
#
#     # Instantiate the VAE model from HuggingFace diffusers
#     vae = AutoencoderKL.from_pretrained(
#         cfg.vae.model_name,
#         subfolder=cfg.vae.subfolder,
#         torch_dtype=getattr(torch, cfg.vae.dtype),
#     )
#     print(f"[INFO] Loaded VAE: {cfg.vae.model_name}")
#     return vae

def load_vae(cfg: DictConfig, **overrides):
    """
    Instantiate a VAE.

    Parameters
    ----------
    cfg : DictConfig
        The *vae* subtree from the composed Hydra config.
    **overrides
        Arbitrary key/value pairs that override (or add to) cfg at runtime.
        Nested keys can be written in dot-notation, e.g. ``decoder.dropout=0.0``.
    """
    final_cfg = _merge_cfg(cfg, overrides)

    vae = AutoencoderKL.from_pretrained(
            final_cfg.model_name,
            subfolder=final_cfg.subfolder,
            torch_dtype=getattr(torch, final_cfg.dtype),
        )
    return vae

def encode_images(model: nn.Module, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    with torch.no_grad():
        latents_dist = model.encode(images)
        latents = latents_dist.latent_dist.sample() * scaling_factor

        return latents

def decode_latents(model: nn.Module, latents: torch.Tensor, scaling_factor) -> torch.Tensor:
    with torch.no_grad():
        decoded_imgs = model.decode(latents / scaling_factor).sample
        decoded_imgs = (decoded_imgs * 0.5 + 0.5).clamp(0, 1)

        return decoded_imgs