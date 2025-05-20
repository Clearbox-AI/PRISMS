import os
import glob
import torch
from pathlib import Path
from typing import Union, Optional, List
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
#  Helper serialise / deserialise the lightweight EMA we use everywhere
# -----------------------------------------------------------------------------

def _ema_to_dict(ema_obj) -> dict:
    """Pack the minimal information required to restore an EMA instance."""

    shadow_cpu: List[torch.Tensor] = [p.detach().cpu() for p in ema_obj.shadow_params]
    return {
        "shadow_params": shadow_cpu,
        "num_updates": ema_obj.num_updates,
        "decay": ema_obj.decay,
        "update_after_step": ema_obj.update_after_step,
        "update_every": ema_obj.update_every,
    }

def _dict_to_ema(state: dict, ema_obj, device: torch.device):
    """Load the serialized state **in‑place** into ``ema_obj`` (same class)."""

    for p_shadow, p_saved in zip(ema_obj.shadow_params, state["shadow_params"]):
        p_shadow.data.copy_(p_saved.to(device))

    ema_obj.num_updates = state.get("num_updates", 0)

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
    ema_obj: Optional["EMA"] = None,
) -> None:

    ckpt_path = Path(ckpt_dir, "checkpoints", ckpt_name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint => {ckpt_path}")

    # unwrap DDP if needed
    model_state_dict = model.module.state_dict() if use_ddp and isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": last_loss,
    }

    if ema_obj is not None:
        checkpoint["ema_state_dict"] = _ema_to_dict(ema_obj)

    torch.save(checkpoint, ckpt_path)


def resume_from_checkpoint(
    resume_dir: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_ddp: bool,
    ema_obj=None,
):
    """Resume training from the *latest* ``*.pt`` in ``resume_dir/checkpoints``."""

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

    # ---- load optimiser & step counters ------------------------------------
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("step", 0)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ---- load model ---------------------------------------------------------
    raw_sd = ckpt["model_state_dict"]
    sd = strip_ddp_prefix(raw_sd, "module")
    target = model.module if (use_ddp and isinstance(model, DDP)) else model
    target.load_state_dict(sd, strict=True)

    # ---- load EMA -----------------------------------------------------------
    if ema_obj is not None and "ema_state_dict" in ckpt:
        _dict_to_ema(ckpt["ema_state_dict"], ema_obj, device)
        if is_main_process():
            print("EMA weights resumed from checkpoint.")

    if is_main_process():
        print(f"Checkpoint loaded, resuming at epoch={start_epoch}, global step={global_step}")
    return start_epoch, global_step
