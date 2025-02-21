import utils.project_setup
import os
from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional
from pathlib import Path

from models.dit.dit_multimodal import MultiModalDiT
from diffusion.multimodal_diffusion import MultiModalDiffusion
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion


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


def load_dit(cfg: DictConfig, **overrides: Any) -> nn.Module:
    """
    Load a MultiModalDiT model based on the provided configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object for the DiT model.
        **overrides (Any): Arbitrary keyword arguments used to override the default configuration.

    Returns:
        nn.Module: The loaded MultiModalDiT model.
    """
    # Apply any overrides to the config before loading
    cfg = apply_overrides(cfg, overrides)

    print("[INFO] Loading MultiModalDiT model with config:", cfg)

    # Instantiate the MultiModalDiT model
    model = MultiModalDiT(**cfg.dit)
    print("[INFO] Loaded DiT")
    return model


def load_diffusion(cfg: DictConfig, dit_model: nn.Module, **overrides: Any) -> nn.Module:
    """
    Load a MultiModalDiffusion model based on the provided configuration.

    The config is expected to have a top-level 'diffusion' section containing the parameters
    for the MultiModalDiffusion. The 'dit_model' argument is required because
    MultiModalDiffusion depends on a pre-loaded DiT model.

    Args:
        cfg (DictConfig): The Hydra configuration object (must contain a 'diffusion' section).
        dit_model (nn.Module): The already loaded DiT model, required by the Diffusion model.
        **overrides (Any): Arbitrary keyword arguments used to override the default configuration.

    Returns:
        nn.Module: The loaded MultiModalDiffusion model.
    """
    # Apply any overrides to the config before loading
    cfg = apply_overrides(cfg, overrides)

    print("[INFO] Loading Diffusion model with config:", cfg)

    # Instantiate the Diffusion model, injecting the loaded DiT
    diffusion_model = MultiModalDiffusion(dit=dit_model, **cfg.diffusion)
    print("[INFO] Loaded Diffusion Model")
    return diffusion_model


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


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """
    Apply keyword argument overrides to the first matching top-level section in the Hydra config.

    For example, if cfg has a section 'vae' or 'dit', and you pass an override with a key
    'model_name', it will be applied to 'cfg.vae.model_name' or 'cfg.dit.model_name' if found.

    Args:
        cfg (DictConfig): The original Hydra configuration object.
        overrides (Dict[str, Any]): A dictionary of overrides, where each key-value pair should
            match an existing field in the top-level sections of the config.

    Returns:
        DictConfig: The updated configuration after applying overrides.

    Raises:
        KeyError: If an override key does not exist in any top-level section.
    """
    for key, value in overrides.items():
        for parent_key, section in cfg.items():
            # We only apply overrides to top-level DictConfig sections
            if isinstance(section, DictConfig) and key in section:
                section[key] = value
                break
    return cfg


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
