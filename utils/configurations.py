import os

from omegaconf import DictConfig
from typing import Dict, Any


def get_project_root():
    """Finds the PRISMS root dynamically based on the script's absolute path."""
    script_path = os.path.abspath(__file__)  # Get the absolute path of this file
    parts = script_path.split(os.sep)  # Split path into components

    if "PRISMS" in parts:
        idx = parts.index("PRISMS")  # Find the first occurrence of PRISMS
        return os.sep.join(parts[:idx + 1])  # Join back up to PRISMS
    else:
        raise RuntimeError("PRISMS root not found in the script path.")


def set_project_root():
    # Get the project root and export it as an environment variable for Hydra
    if "PROJECT_ROOT" not in os.environ:
        PROJECT_ROOT = get_project_root()
        os.environ["PROJECT_ROOT"] = PROJECT_ROOT  # Ensure Hydra sees this variable


# def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
#     """
#     Apply keyword argument overrides to the first matching top-level section in the Hydra config.
#
#     For example, if cfg has a section 'vae' or 'dit', and you pass an override with a key
#     'model_name', it will be applied to 'cfg.vae.model_name' or 'cfg.dit.model_name' if found.
#
#     Args:
#         cfg (DictConfig): The original Hydra configuration object.
#         overrides (Dict[str, Any]): A dictionary of overrides, where each key-value pair should
#             match an existing field in the top-level sections of the config.
#
#     Returns:
#         DictConfig: The updated configuration after applying overrides.
#
#     Raises:
#         KeyError: If an override key does not exist in any top-level section.
#     """
#     for key, value in overrides.items():
#         for parent_key, section in cfg.items():
#             # We only apply overrides to top-level DictConfig sections
#             if isinstance(section, DictConfig) and key in section:
#                 section[key] = value
#                 break
#     return cfg

from omegaconf import DictConfig, OmegaConf
from copy import deepcopy


def _merge_cfg(base: DictConfig, overrides: dict) -> DictConfig:
    """
    Return a **new** DictConfig obtained by applying the key/value pairs
    in *overrides* on top of *base*.

    *Overrides* may contain **dot-paths** to reach nested fields, e.g.
    `overrides={"encoder.num_layers": 8}`.
    """
    if not overrides:
        return deepcopy(base)

    # Convert ``{"a.b": 1, "c":2}`` → OmegaConf.dotlist
    dotlist = [f"{k}={v}" for k, v in overrides.items()]
    user_cfg = OmegaConf.from_dotlist(dotlist)
    merged = OmegaConf.merge(base, user_cfg)
    return merged