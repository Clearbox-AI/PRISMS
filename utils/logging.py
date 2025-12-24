from __future__ import annotations

from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

def debug_getitem(func):
    """
    Decorator to optionally visualize the image for debugging
    (only on the first call if self.debug == True).
    """
    def wrapper(self, idx):
        data = func(self, idx)
        if self.debug and not self._debug_shown:
            image = data["image"]  # shape [C,H,W], torch tensor
            self._debug_show_image(image)
            self._debug_shown = True
        return data
    return wrapper

class FlowMonitor:
    def __init__(self, model: nn.Module, threshold: float, log_dir: Path, rank: int = 0):
        self.threshold = threshold
        self.monitoring = False
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = f"suspicious_rank{rank}.log" if rank else "suspicious.log"
        self.log_path = log_dir / fname
        self.log_file = open(self.log_path, "a", buffering=1)
        self._register_hooks(model)

    def _timestamp(self):
        return datetime.now().isoformat()

    def _log(self, msg: str):
        self.log_file.write(f"{self._timestamp()} {msg}\n")

    def _check_and_log(self, name: str, tensor: torch.Tensor, where: str):
        if not torch.is_tensor(tensor):
            return
        t = tensor.detach()
        t_min = t.min().item()
        t_max = t.max().item()
        t_mean = t.mean().item()
        if (t_max > self.threshold or t_min < -self.threshold or
            torch.isnan(t).any() or torch.isinf(t).any() or self.monitoring):
            if not self.monitoring:
                self._log(f"▶▶ Threshold exceeded in `{name}` ({where}): "
                          f"min={t_min:.3e}, max={t_max:.3e}, mean={t_mean:.3e}")
                self.monitoring = True
            else:
                self._log(f"{where} `{name}`: min={t_min:.3e}, max={t_max:.3e}, mean={t_mean:.3e}")
            if torch.isnan(t).any() or torch.isinf(t).any():
                self._log(f"‼‼ NaN/Inf detected in `{name}` during {where}. Stopping training.")
                self.log_file.close()
                raise RuntimeError(f"NaN/Inf in `{name}` during {where}")

    def _make_fwd_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._check_and_log(name, out, "forward")
            elif isinstance(out, (tuple, list)):
                for i, o in enumerate(out):
                    self._check_and_log(f"{name}[{i}]", o, "forward")
        return hook

    def _make_grad_hook(self, name):
        def hook(grad):
            self._check_and_log(name, grad, "backward_grad")
            return grad
        return hook

    def _register_hooks(self, model):
        # Always safe to inspect forward activations.
        for name, module in model.named_modules():
            module.register_forward_hook(self._make_fwd_hook(name))
        # Gradient hooks ONLY on trainable parameters.
        for name, param in model.named_parameters():
            if getattr(param, "requires_grad", False):
                param.register_hook(self._make_grad_hook(name))

    def close(self):
        self.log_file.close()
