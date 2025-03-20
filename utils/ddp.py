# import os
# import torch
# import torch.distributed as dist
#
# from torch.nn.parallel import DistributedDataParallel as DDP
# from omegaconf import DictConfig
#
# def is_main_process() -> bool:
#     """
#     Checks if current process is global rank 0 (main) in DDP.
#     Returns True if running single-process or if rank==0.
#     """
#     return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
#
#
# def strip_ddp_prefix(state_dict, keyword):
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith(f"{keyword}."):
#             new_k = k[len(f"{keyword}."):]
#         elif k.startswith(f"{keyword}."):
#             new_k = k[len(f"{keyword}."):]
#         else:
#             new_k = k
#         new_state_dict[new_k] = v
#     return new_state_dict
#
#
# def setup_distributed(cfg: DictConfig) -> int:
#     """
#     Initialize the torch.distributed process group for DDP.
#     Returns:
#         local_rank (int): The local GPU index this process will use.
#     """
#     dist.init_process_group(
#         backend=cfg.distributed.backend,
#         init_method="env://"
#     )
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     return local_rank
#
#
# def cleanup_distributed():
#     """
#     Destroy the process group for DDP.
#     """
#     dist.destroy_process_group()


import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP


def is_dist_available_and_initialized() -> bool:
    """Check if torch.distributed is both available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current process's rank (global) or 0 if not in distributed mode."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return the world size (total number of processes) or 1 if not in distributed mode."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """
    Checks if current process is global rank 0 (main) in DDP.
    Returns True if running single-process or if rank==0.
    """
    return get_rank() == 0


def setup_distributed(cfg: DictConfig) -> int:
    """
    Initialize the torch.distributed process group for DDP.
    Typical usage: specify `backend` in your config, and rely on the env:// variables
    (LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) to be set by your launcher
    (e.g., torchrun or SLURM).

    Returns:
        local_rank (int): The local GPU index for this process.
    """
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method="env://",
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """
    Destroy the process group for DDP.
    Should be called after all DDP work is complete (typically at script exit).
    """
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def strip_ddp_prefix(state_dict: dict, keyword: str) -> dict:
    """
    Removes a prefix (often 'module.' when using DistributedDataParallel)
    from state_dict keys. This is useful when loading a model trained via DDP
    onto a single GPU or CPU.

    Args:
        state_dict (dict): A PyTorch model state_dict.
        keyword (str): The prefix to remove (e.g. "module").

    Returns:
        new_state_dict (dict): A new dict with the prefix stripped.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(f"{keyword}."):
            new_k = k[len(f"{keyword}."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def barrier():
    """
    A convenience wrapper around dist.barrier() so that we only barrier if
    distributed is available and initialized. Otherwise, do nothing (single process).
    """
    if is_dist_available_and_initialized():
        dist.barrier()


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gathers a tensor from all ranks on every rank and concatenates along the first dimension.
    This is a simple utility for multi-GPU data consolidation:
      - Each rank will have an identical output containing data from all ranks.

    Args:
        tensor (torch.Tensor): The local tensor to be gathered.

    Returns:
        A tensor of shape (sum_of_all_ranks_batch_sizes, ...) containing the gathered data.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # Obtain local tensor shape
    local_size = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]

    # Gather shapes
    dist.all_gather(all_sizes, local_size)

    max_size = max(x.item() for x in all_sizes)
    if local_size < max_size:
        # Pad the tensor along dim=0 (batch dimension)
        pad_shape = (max_size - local_size,) + tensor.shape[1:]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=0)

    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)

    # Trim back to the correct sizes
    results = []
    for rank_idx in range(world_size):
        valid_count = all_sizes[rank_idx].item()
        results.append(gather_list[rank_idx][:valid_count])

    # Combine along dim=0
    return torch.cat(results, dim=0)


def all_gather_object(obj) -> list:
    """
    Gathers an arbitrary picklable Python object from all ranks onto every rank.
    Each rank receives a list of objects [obj_rank0, obj_rank1, ..., obj_rankN].
    """
    gathered_objs = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered_objs, obj)
    return gathered_objs


def broadcast_object(obj, src=0):
    """
    Broadcasts an arbitrary picklable object from the source rank to all ranks.
    Returns the broadcasted object on every rank.
    """
    if get_world_size() == 1:
        return obj  # single process, nothing to do

    # On non-src ranks, create a placeholder object
    if get_rank() != src:
        obj = None
    dist.broadcast_object_list([obj], src=src)
    return obj


def ddp_sample(model, *args, **kwargs):
    """
    Calls 'sample' on the underlying model if wrapped in DDP.
    """
    if isinstance(model, DDP):
        return model.module.generate(*args, **kwargs)
    else:
        return model.generate(*args, **kwargs)