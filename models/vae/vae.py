import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from omegaconf import DictConfig
from typing import Any

from utils.configurations import apply_overrides

def load_vae(cfg: DictConfig, **overrides: Any) -> nn.Module:
    """
    Load a VAE model (specifically an AutoencoderKL from diffusers) based on the provided configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object for the VAE.
        **overrides (Any): Arbitrary keyword arguments used to override the default configuration.

    Returns:
        nn.Module: The loaded VAE model.
    """
    # Apply any overrides to the config before loading
    cfg = apply_overrides(cfg, overrides)

    print("[INFO] Loading VAE model with config:", cfg)

    # Instantiate the VAE model from HuggingFace diffusers
    vae = AutoencoderKL.from_pretrained(
        cfg.vae.model_name,
        subfolder=cfg.vae.subfolder,
        torch_dtype=getattr(torch, cfg.vae.dtype),
    )
    print(f"[INFO] Loaded VAE: {cfg.vae.model_name}")
    return vae