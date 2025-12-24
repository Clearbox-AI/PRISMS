import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch.nn as nn

from omegaconf import OmegaConf
from omegaconf import DictConfig
from typing import Any, Optional
from pathlib import Path

from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from models.diffusion.diffusion_multimodal import load_diffusion
from models.dit.dit_multimodal import load_dit
from models.vae.vae import load_vae

def load_model(model_type: ModelType, cfg: DictConfig, tab_transforms=None, **overrides: Any,) -> nn.Module:
    """
    Unified entry-point to build VAE, DiT or Diffusion.

    Parameters
    ----------
    model_type : ModelType
        Which model to create.
    cfg : DictConfig
        The **full** Hydra config tree (already composed in the trainer).
    model_variant : Optional[DiTTrainingVersion]
        Kept for backward compatibility; currently only tags the "base" DiT.
    **overrides : Any
        Arbitrary key/value pairs — including dotted keys — that override
        or extend the model’s own config subsection at runtime.

    Notes
    -----
    * The same *overrides* dict is forwarded to the concrete loader.
      Irrelevant keys are silently ignored by that loader’s `_merge_cfg`.
    * Call-sites can therefore do either
        dit = load_model(ModelType.DIT, cfg)
      *or*
        diff = load_model(ModelType.DIFFUSION, cfg, num_tab_features=512, sigma_max=120.0)
    """

    if model_type is ModelType.VAE:
        return load_vae(cfg.vae, **overrides)

    if model_type is ModelType.DIT:
        return load_dit(cfg.dit, **overrides)

    if model_type is ModelType.DIFFUSION:
        # dit_model = load_dit(cfg.dit, **overrides) #TODO: fix, mechanisms to pass overides that could differs between dit and diff
        dit_model = load_dit(cfg.dit)
        return load_diffusion(cfg.diffusion, dit_model=dit_model, tab_transforms=tab_transforms, **overrides)

    raise ValueError(f"Unsupported model type {model_type}")

if __name__ == "__main__":
    """
    Example usage of the load_model function to instantiate a VAE, a DiT, and a Diffusion model.
    """

    # Instantiate a VAE with overrides (e.g., changing the dtype and model_name).
    vae_model = load_model(
        model_type=ModelType.VAE,
        dtype="float16",      # Example override
    )
    print("[INFO] VAE model instantiated successfully!")

    # Instantiate a DiT model with overrides (if any)
    dit_model = load_model(
        model_type=ModelType.DIT,
        hidden_dim=1024  # Example override (assuming 'hidden_dim' exists in your DiT config)
    )
    print("[INFO] DiT model instantiated successfully!")

    # Instantiate a Diffusion model (requires a DiT internally)
    diffusion_model = load_model(
        model_type=ModelType.DIFFUSION,
        # Example override: let's say we want to override something in the diffusion section
        guidance_scale=7.5
    )
    print("[INFO] Diffusion model instantiated successfully!")
