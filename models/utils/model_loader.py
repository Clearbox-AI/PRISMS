import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch.nn as nn

from omegaconf import OmegaConf
from typing import Any, Optional
from pathlib import Path

from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion
from models.diffusion.diffusion_multimodal import load_diffusion
from models.dit.dit_multimodal import load_dit
from models.vae.vae import load_vae


def load_model(
    model_type: ModelType,
    model_variant: Optional[DiTTrainingVersion] = None,
    **overrides: Any
) -> nn.Module:
    """
    Load a specific model type (VAE, DiT, or Diffusion) using Hydra-based configuration management.

    This function initializes Hydra with a config directory derived from the 'PROJECT_ROOT'
    environment variable. It then composes the config for either a 'vae', 'dit', or 'base_dit_training'
    model, loads it, and returns the instantiated model.

    Usage:
        - If model_type == ModelType.VAE:
            Composes and loads a VAE (AutoencoderKL).
        - If model_type == ModelType.DIT:
            Composes and loads a MultiModalDiT model.
        - If model_type == ModelType.DIFFUSION:
            First loads a DiT model (since the Diffusion model depends on DiT),
            then composes and loads a MultiModalDiffusion model using the same config.

    Args:
        model_type (ModelType): An enum value indicating which model to load (VAE, DIT, or DIFFUSION).
        model_variant (Optional[DiTTrainingVersion]): An enum value indicating which variant of the DiT
            training config to load if relevant. For example, `DiTTrainingVersion.base_dit_training`.
        **overrides (Any): Arbitrary keyword arguments to override parts of the configuration.

    Returns:
        nn.Module: The instantiated PyTorch model corresponding to the requested model type.

    Raises:
        ValueError: If the provided model_type is not supported.
    """

    from utils.configurations import set_project_root
    set_project_root()

    # Clear any existing Hydra initialization to avoid conflicts
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    if model_type == ModelType.VAE:
        # Load VAE config and model
        with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
            cfg = compose(config_name="vae")
            OmegaConf.set_struct(cfg, False)
        return load_vae(cfg, **overrides)

    elif model_type == ModelType.DIT or model_type == ModelType.DIFFUSION:
        # Load DiT config from either a 'base_dit_training' config or a standard 'dit' config
        if model_variant == DiTTrainingVersion.base_dit_training:
            with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
                cfg = compose(config_name="base_dit_training")
        else:
            with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
                cfg = compose(config_name="dit")

        dit_model = load_dit(cfg, **overrides)

        if model_type == ModelType.DIFFUSION:
            with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
                cfg = compose(config_name="diffusion")

            diffusion_model = load_diffusion(cfg, dit_model, **overrides)
            return diffusion_model

        return dit_model

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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
