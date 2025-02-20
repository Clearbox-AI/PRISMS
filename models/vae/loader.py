import torch
from diffusers import AutoencoderKL
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../configs/models", config_name="vae")
def load_stable_diffusion_xl_vae(
    cfg: DictConfig,
    model_name: str = None,
    subfolder: str = None,
    device: str = None,
    dtype_str: str = None
):

    model_name = model_name or cfg.model_name
    subfolder = subfolder or cfg.subfolder
    device = device or cfg.device
    dtype_str = dtype_str or cfg.dtype_str

    dtype = getattr(torch, dtype_str)

    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder=subfolder,
        torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)
    return vae

if __name__ == "__main__":
    load_stable_diffusion_xl_vae()
