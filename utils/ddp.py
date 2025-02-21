import torch.distributed as dist

def is_main_process():
    """
    Utility to check if current process is the global rank 0.
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