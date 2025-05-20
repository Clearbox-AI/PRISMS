import os
import glob
import torch

from pathlib import Path
from typing import Union

from utils.ddp import is_main_process, strip_ddp_prefix


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    """
    Loads the checkpoint into the given model.

    Args:
        model (torch.nn.Module): The model into which the checkpoint should be loaded.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint to.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_sd = ckpt["model_state_dict"]
    model.load_state_dict(raw_sd, strict=True)


def save_checkpoint(
    ckpt_dir: Union[Path, str],
    ckpt_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    last_loss: float,
    use_ddp: bool
) -> None:
    """
    Saves model and optimizer states to the given path.
    """

    ckpt_path = Path(ckpt_dir, "checkpoints", ckpt_name)

    print(f"Saving checkpoint => {ckpt_path}")
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if use_ddp else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': last_loss,
    }, ckpt_path)


def resume_from_checkpoint(
    resume_dir: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_ddp: bool
):
    """
    Finds the latest checkpoint in `resume_dir`, loads it into the model & optimizer.
    Returns (start_epoch, global_step) to continue training.
    """

    resume_dir = Path(resume_dir, "checkpoints")
    ckpt_files = sorted(glob.glob(os.path.join(resume_dir, "*.pt")), key=os.path.getmtime)
    if not ckpt_files:
        print(f"No checkpoint file found in {resume_dir}. Starting fresh.")
        return 0, 0

    resume_path = ckpt_files[-1]
    if is_main_process():
        print(f"Resuming training from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    start_epoch = ckpt.get('epoch', 0)
    global_step = ckpt.get('step', 0)
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    raw_model_state_dict = ckpt["model_state_dict"]
    # If the state_dict was saved from a DDP model, remove prefix
    sd = strip_ddp_prefix(raw_model_state_dict, "module")

    if use_ddp:
        model.module.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(sd, strict=True)

    if is_main_process():
        print(f"Checkpoint loaded, resuming at epoch={start_epoch}, global step={global_step}")
    return start_epoch, global_step
