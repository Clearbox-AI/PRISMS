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
    """
    Serializza EMA.
      - AveragedModel: salva .module.state_dict e (opz.) n_averaged
      - Lightweight CPUEMA: salva shadow (su CPU) + metadati
    """
    # AveragedModel
    if hasattr(ema_obj, "module"):
        return {
            "kind": "averaged_model",
            "state_dict": ema_obj.module.state_dict(),
            "num_updates": int(getattr(ema_obj, "n_averaged", 0)),
            # back-compat
            "decay": float(getattr(ema_obj, "decay", 0.0)),
        }

    # Lightweight (CPUEMA)
    if hasattr(ema_obj, "shadow"):
        shadow_cpu = {k: v.detach().cpu() for k, v in ema_obj.shadow.items()}
        return {
            "kind": "lightweight",
            "shadow": shadow_cpu,
            "num_updates": int(getattr(ema_obj, "num_updates", 0)),
            "base_decay": float(getattr(ema_obj, "base_decay", 0.9999)),
            "use_after_updates": int(getattr(ema_obj, "use_after_updates", 300)),
            # back-compat: alcuni vecchi ckpt potrebbero aver salvato "decay"
            "decay": float(getattr(ema_obj, "base_decay", 0.9999)),
        }

    return {"kind": "none"}

def _dict_to_ema(state: dict, ema_obj, device: torch.device) -> None:
    """
    Ripristina l'EMA **in-place** dentro `ema_obj` a partire da `state`.

    Gestisce:
      - kind="averaged_model": ema stile PyTorch AveragedModel (usa .module, opz. n_averaged/decay)
      - kind="lightweight": EMA leggera (shadow dict su CPU + metadati)
    È retro-compatibile con vecchi checkpoint che non salvavano num_updates/base_decay
    e dove "decay" poteva essere 0.0.
    """
    if not state:
        return

    kind = state.get("kind", "averaged_model")

    if kind == "averaged_model":
        # .module è il modello medio (SWA/EMA di PyTorch)
        ema_obj.module.load_state_dict(state["state_dict"])
        # ripristina il contatore se presente
        if "num_updates" in state and hasattr(ema_obj, "n_averaged"):
            # n_averaged in PyTorch è un tensore; copiamo il valore
            val = int(state["num_updates"])
            if isinstance(ema_obj.n_averaged, torch.Tensor):
                ema_obj.n_averaged.copy_(torch.tensor(val, device=device))
            else:
                ema_obj.n_averaged = torch.tensor(val, device=device)
        # ripristina eventuale decay (API AveragedModel)
        if "decay" in state:
            try:
                ema_obj.decay = float(state["decay"])
            except Exception:
                pass
        return

    elif kind == "lightweight":
        # *** Parte fondamentale: le shadow restano su CPU ***
        if hasattr(ema_obj, "shadow") and "shadow" in state:
            ema_obj.shadow.clear()
            for k, v in state["shadow"].items():
                ema_obj.shadow[k] = v.detach().clone().cpu()

        # Ripristina metadati, se presenti (senza sovrascrivere con 0.0 legacy)
        if "num_updates" in state:
            try:
                ema_obj.num_updates = int(state["num_updates"])
            except Exception:
                pass

        # Preferisci "base_decay"; altrimenti usa "decay" solo se > 0 (evita il vecchio 0.0)
        if "base_decay" in state:
            try:
                ema_obj.base_decay = float(state["base_decay"])
            except Exception:
                pass
        else:
            d = float(state.get("decay", -1.0))
            if d > 0.0:
                ema_obj.base_decay = d

        if "use_after_updates" in state:
            try:
                ema_obj.use_after_updates = int(state["use_after_updates"])
            except Exception:
                pass

        return

    # kind "none" o sconosciuto: niente da fare
    return


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
    ema_obj: Optional[object] = None,
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
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    use_ddp: bool,
    ema_obj: Optional[object] = None,
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
        try:
            _dict_to_ema(ckpt["ema_state_dict"], ema_obj, device)
        except Exception as e:
            print(f"Warning: EMA could not be restored ({e}). Continuing without EMA state.")
        if is_main_process():
            print("EMA weights resumed from checkpoint.")

    if is_main_process():
        print(f"Checkpoint loaded, resuming at epoch={start_epoch}, global step={global_step}")
    return start_epoch, global_step
