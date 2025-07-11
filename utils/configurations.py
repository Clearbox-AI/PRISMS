import os

from omegaconf import DictConfig
from typing import Dict, Any, Mapping


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

from copy import deepcopy
from omegaconf import open_dict
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


def _merge_cfg(base: DictConfig, overrides: Mapping[str, Any]) -> dict:
    """
    Merge *overrides* (keys may be dotted paths) onto *base* and return
    a **plain Python dict** that can contain arbitrary objects.

    This avoids OmegaConf's type-checking on unions and its serialization
    limits while keeping the convenience of dotted-key syntax.
    """
    # 1. Convert to a mutable plain dict (still nested)
    cfg: dict = OmegaConf.to_container(base, resolve=False, enum_to_str=False)  # ➜ plain dict :contentReference[oaicite:2]{index=2}
    cfg = deepcopy(cfg)  # preserve immutability of the original Hydra config

    # 2. Helper to set a dotted path inside a nested dict
    def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value

    # 3. Apply all overrides
    for key, val in overrides.items():
        _set_nested(cfg, key, val)

    return cfg


# def _merge_cfg(base: DictConfig, overrides: Mapping[str, Any]) -> DictConfig:
#     """
#     Merge *overrides* onto *base* and return an UN-typed DictConfig that:
#       • still allows dot access (cfg.foo)
#       • can contain arbitrary Python objects (allow_objects=True)
#       • has **no** Union/type checks ⇒ avoids the 'Unions of containers'
#         error that hit you before.
#     """
#     # 1️⃣ flatten base to a plain nested dict
#     data = OmegaConf.to_container(
#         base, resolve=False, enum_to_str=False
#     )               # converts structured config → dict :contentReference[oaicite:0]{index=0}
#     data = deepcopy(data)   # keep original immutable
#
#     # 2️⃣ apply dotted overrides
#     def _set_nested(d: dict, dotted: str, value: Any):
#         parts = dotted.split(".")
#         for p in parts[:-1]:
#             d = d.setdefault(p, {})
#         d[parts[-1]] = value
#
#     for k, v in overrides.items():
#         _set_nested(data, k, v)
#
#     # 3️⃣ re-wrap in an **unstructured** DictConfig
#     cfg = OmegaConf.create(data, flags={"allow_objects": True})  # keeps objects verbatim :contentReference[oaicite:1]{index=1}
#     return cfg