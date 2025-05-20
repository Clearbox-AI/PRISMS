import os
import glob
import torch
from pathlib import Path
from typing import Union, Optional
from torch.nn.parallel import DistributedDataParallel as DDP


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
    use_ddp: bool,
    ema_obj: Optional["EMA"] = None  # pass your EMA object if you have one
) -> None:
    """
    Saves model (and optionally EMA) + optimizer states to the given path.

    Args:
        ckpt_dir: directory to save checkpoints.
        ckpt_name: checkpoint filename.
        model: the main diffusion model (possibly wrapped in DDP).
        optimizer: the optimizer instance.
        epoch: current epoch number.
        step: global step.
        last_loss: last recorded loss (for logging).
        use_ddp: if True, model is wrapped in DDP => we save model.module's state.
        ema_obj: if provided, we also save the EMA state dict.
    """
    ckpt_path = Path(ckpt_dir, "checkpoints", ckpt_name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoint => {ckpt_path}")

    # If DDP, unwrap the model for state dict
    if use_ddp and isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # If we have an EMA object, also retrieve its state dict
    ema_state_dict = None
    if ema_obj is not None:
        ema_state_dict = ema_obj.model_ema.state_dict()

    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': last_loss,
    }
    if ema_state_dict is not None:
        checkpoint['ema_state_dict'] = ema_state_dict

    torch.save(checkpoint, ckpt_path)


def resume_from_checkpoint(
    resume_dir: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_ddp: bool,
    ema_obj = None
):
    """
    Finds the latest checkpoint in `resume_dir`, loads it into the model & optimizer.
    Also attempts to load EMA weights if `ema_obj` is provided and 'ema_state_dict' is in the checkpoint.

    Returns (start_epoch, global_step) to continue training.
    """
    import os
    import glob

    from utils.ddp import is_main_process, strip_ddp_prefix

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
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    raw_model_state_dict = ckpt["model_state_dict"]
    # If the state_dict was saved from a DDP model, remove prefix
    sd = strip_ddp_prefix(raw_model_state_dict, "module")

    # Load into the model
    if use_ddp and isinstance(model, DDP):
        model.module.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(sd, strict=True)

    # If the checkpoint contains EMA weights and we have an EMA object, load it
    if ema_obj is not None and "ema_state_dict" in ckpt:
        ema_sd = ckpt["ema_state_dict"]
        ema_obj.model_ema.load_state_dict(ema_sd, strict=True)
        if is_main_process():
            print("EMA weights resumed from checkpoint.")

    if is_main_process():
        print(f"Checkpoint loaded, resuming at epoch={start_epoch}, global step={global_step}")
    return start_epoch, global_step
