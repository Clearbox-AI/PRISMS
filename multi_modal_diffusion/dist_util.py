# """
# Helpers for distributed training.
# """
#
# import io
# import os
# import socket
# import blobfile as bf
# from mpi4py import MPI
# import torch as th
# import torch.distributed as dist
#
# # Change this to reflect your cluster layout.
# # The GPU for a given rank is (rank % GPUS_PER_NODE).
#
# GPUS_PER_NODE = 8
#
# def setup_dist(devices=None):
#     """
#     Setup a distributed process group.
#     """
#     global GPUS_PER_NODE
#     if dist.is_initialized():
#         return
#
#     if devices.startswith("G"):
#         GPUS_PER_NODE = int(devices[1:])
#         os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
#     else:
#         devices_list=devices.split(',')
#         GPUS_PER_NODE = len(devices_list)
#         os.environ["CUDA_VISIBLE_DEVICES"] =  f"{devices_list[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE]}"
#
#     comm = MPI.COMM_WORLD
#
#     backend = "gloo" if not th.cuda.is_available() else "nccl"
#
#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)
#
#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#
#     dist.init_process_group(backend=backend, init_method="env://")
#
#
#
#
# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#
#     if th.cuda.is_available():
#         return th.device("cuda")
#     return th.device("cpu")
#
# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     with bf.BlobFile(path, "rb") as f:
#         data = f.read()
#
#     return th.load(io.BytesIO(data), **kwargs)
#
# def sync_params(params):
#     """
#     Synchronize a sequence of Tensors across ranks from rank 0.
#     """
#     for p in params:
#         with th.no_grad():
#             dist.broadcast(p, 0)
#
#
#
#
# def _find_free_port():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]
#     finally:
#         s.close()



"""
Helpers for distributed training.
"""


import io
import os
import socket
# import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Initialize global variables
_global_device = None
_global_rank = 0
_global_world_size = 1

def setup_dist(devices=None):
    """
    Setup a distributed process group.
    """
    global _global_device, _global_rank, _global_world_size

    if devices is None or devices.lower() == "cpu":
        # Running on CPU, single process
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _global_device = th.device("cpu")
        _global_rank = 0
        _global_world_size = 1
        return  # Skip initializing the process group

    devices_list = devices.split(',')
    if len(devices_list) == 1 and devices_list[0].lower() == "cpu":
        # Running on CPU, single process
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _global_device = th.device("cpu")
        _global_rank = 0
        _global_world_size = 1
        return

    # Set CUDA_VISIBLE_DEVICES to the specified devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(devices_list)
    num_gpus = len(devices_list)

    # Check if we're running in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # We are in a distributed setting
        _global_rank = int(os.environ['RANK'])
        _global_world_size = int(os.environ['WORLD_SIZE'])
        _global_device = th.device(f"cuda:{_global_rank % num_gpus}")
        th.cuda.set_device(_global_device)

        backend = "nccl" if th.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method='env://')
    else:
        # Single process
        _global_rank = 0
        _global_world_size = 1
        _global_device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if th.cuda.is_available():
            th.cuda.set_device(_global_device)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    global _global_device
    if _global_device is None:
        _global_device = th.device("cuda" if th.cuda.is_available() else "cpu")
    return _global_device

def rank():
    """
    Get the rank of the current process.
    """
    global _global_rank
    return _global_rank

def get_world_size():
    """
    Get the total number of processes.
    """
    global _global_world_size
    return _global_world_size

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    if rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = None
    if get_world_size() > 1:
        data = broadcast_bytes(data)
    return th.load(io.BytesIO(data), **kwargs)

def broadcast_bytes(data):
    """
    Broadcast a sequence of bytes from rank 0 to all other ranks.
    """
    if rank() == 0:
        size = th.tensor([len(data)], dtype=th.long, device=dev())
    else:
        size = th.tensor([0], dtype=th.long, device=dev())
    dist.broadcast(size, src=0)
    buffer = th.zeros(size.item(), dtype=th.uint8, device=dev())
    if rank() == 0:
        buffer[:] = th.tensor(list(data), dtype=th.uint8, device=dev())
    dist.broadcast(buffer, src=0)
    return bytes(buffer.cpu().numpy().tolist())

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.is_initialized() and get_world_size() > 1:
        for p in params:
            with th.no_grad():
                dist.broadcast(p, src=0)

def synchronize():
    """
    Synchronize all processes.
    """
    if dist.is_initialized() and get_world_size() > 1:
        dist.barrier()

def _find_free_port():
    """
    Find a free port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

