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
import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Initialize global variables
GPUS_PER_NODE = 0  # Default to 0 GPUs; will be updated based on devices
_global_device = None
_global_rank = 0
_global_world_size = 1

def setup_dist(devices=None):
    """
    Setup a distributed process group.
    """
    global GPUS_PER_NODE, _global_device, _global_rank, _global_world_size

    if devices is None or devices.lower() == "cpu":
        # Running on CPU, single process
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        GPUS_PER_NODE = 0
        _global_device = th.device("cpu")
        _global_rank = 0
        _global_world_size = 1
        return  # Skip initializing the process group

    devices_list = devices.split(',')
    if len(devices_list) == 1 and devices_list[0].lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        GPUS_PER_NODE = 0
        _global_device = th.device("cpu")
        _global_rank = 0
        _global_world_size = 1
        return

    # Try to import mpi4py and set up MPI
    try:
        comm = MPI.COMM_WORLD
        _global_world_size = comm.Get_size()
        _global_rank = comm.Get_rank()
    except ImportError:
        # mpi4py not available; assume single process
        comm = None
        _global_world_size = 1
        _global_rank = 0

    if _global_world_size == 1:
        # Single process, set CUDA_VISIBLE_DEVICES accordingly
        GPUS_PER_NODE = len(devices_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_list[0]
        _global_device = th.device("cuda" if th.cuda.is_available() else "cpu")
        return  # Skip initializing the process group

    # Distributed environment setup
    GPUS_PER_NODE = len(devices_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices_list[_global_rank % GPUS_PER_NODE]

    backend = "nccl" if th.cuda.is_available() else "gloo"

    if comm is not None:
        # Use MPI to broadcast necessary environment variables
        hostname = socket.gethostname()
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
        os.environ["RANK"] = str(_global_rank)
        os.environ["WORLD_SIZE"] = str(_global_world_size)
        port = comm.bcast(str(_find_free_port()), root=0)
        os.environ["MASTER_PORT"] = port
    else:
        # Single process; set default environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_PORT"] = str(_find_free_port())

    if not dist.is_initialized():
        # Initialize the process group only if not already initialized
        dist.init_process_group(backend=backend, init_method="env://")
        _global_device = th.device("cuda" if th.cuda.is_available() else "cpu")

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
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.is_initialized() and world_size() > 1:
        for p in params:
            with th.no_grad():
                dist.broadcast(p, 0)

def synchronize():
    """
    Synchronize all processes.
    """
    if dist.is_initialized() and world_size() > 1:
        dist.barrier()

def _find_free_port():
    """
    Find a free port on localhost.
    """
    try:
        s = socket.socket()
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port
    except Exception:
        return 12345  # Default port if unable to find one

