import os
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from omegaconf import DictConfig

def is_main_process() -> bool:
    """
    Checks if current process is global rank 0 (main) in DDP.
    Returns True if running single-process or if rank==0.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def strip_ddp_prefix(state_dict, keyword: str = "module"):
    """
    Remove a single leading '<keyword>.' from every key in *state_dict*.
    Works for checkpoints saved under DDP as well as non-DDP models.
    """
    return {
        (k[len(keyword) + 1 :] if k.startswith(f"{keyword}.") else k): v
        for k, v in state_dict.items()
    }


def setup_distributed(cfg: DictConfig) -> int:
    """
    Initialize the torch.distributed process group for DDP.
    Returns:
        local_rank (int): The local GPU index this process will use.
    """
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method="env://"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """
    Destroy the process group for DDP.
    """
    dist.destroy_process_group()


def _unwrap(model):
    """Recursively peel {DDP, EMAWrapper/AveragedModel} until custom methods show up."""
    while True:
        if hasattr(model, "sample"):
            return model                               # found the real net
        if isinstance(model, (DDP, AveragedModel)):
            model = model.module                       # step down one layer
        elif hasattr(model, "module"):                 # generic safety-net
            model = model.module
        else:
            break
    raise AttributeError(f"{type(model)} does not expose `.sample()`")


def ddp_sample(model, *args, **kwargs):
    return _unwrap(model).sample(*args, **kwargs)

# def ddp_sample(model, *args, **kwargs):
#     if isinstance(model, DDP):
#         return model.module.sample(*args, **kwargs)
#     return model.sample(*args, **kwargs)