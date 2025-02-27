import os
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig

def is_main_process() -> bool:
    """
    Checks if current process is global rank 0 (main) in DDP.
    Returns True if running single-process or if rank==0.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def strip_ddp_prefix(state_dict, keyword):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(f"{keyword}."):
            new_k = k[len(f"{keyword}."):]
        elif k.startswith(f"{keyword}."):
            new_k = k[len(f"{keyword}."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


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


def ddp_sample(model, *args, **kwargs):
    """
    Calls 'sample' on the underlying model if wrapped in DDP.
    """
    if isinstance(model, DDP):
        return model.module.sample(*args, **kwargs)
    else:
        return model.sample(*args, **kwargs)