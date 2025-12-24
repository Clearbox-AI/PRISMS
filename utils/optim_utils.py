import torch
import torch.nn as nn
from collections import deque
from torch.optim.swa_utils import AveragedModel

class EMAWrapper(AveragedModel):
    """Shadow-EMA with configurable beta (decay)."""
    def __init__(self, model, beta: float = 0.9999):
        dev = next(model.parameters()).device
        super().__init__(
            model,
            device = dev,
            use_buffers = True,
            avg_fn = lambda avg, src, _: beta * avg + (1 - beta) * src,)
        self.decay = beta

    @property
    def module(self):            # to mirror DDP
        return super().module


class AutoClipper:
    """
    Adaptive grad-norm clipper (μ + k·σ).  Call after loss.backward().
    """
    def __init__(self, window: int = 100, k: float = 3.0):
        self.window, self.k = window, k
        self.history = deque(maxlen=window)

    def __call__(self, params):
        # materialise list once — we iterate twice
        params = [p for p in params if p.grad is not None]
        if not params:                         # nothing to clip
            return torch.tensor(0.)

        norms = torch.stack([p.grad.norm() for p in params])
        total_norm = norms.norm()             # L2 over all params
        self.history.append(total_norm.detach())

        # warm-up → fixed 1.0 threshold
        if len(self.history) < 10:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            return total_norm

        hist = torch.stack(tuple(self.history))
        thr = hist.mean() + self.k * hist.std(unbiased=False)
        torch.nn.utils.clip_grad_norm_(params, thr.item())
        return total_norm

class CPUEMA:
    """
    EMA that lives on CPU (zero extra VRAM).
    - Accepts both `decay` and `base_decay` (either name).
    - Stable-Diffusion–style warm-up so EMA tracks early:
        current_decay = min(base_decay, (1 + updates) / (10 + updates))
      => prevents "pixel soup" when sampling early.
    - `ready()` tells you when it’s sensible to sample from EMA.
    """
    def __init__(
        self,
        model: nn.Module,
        base_decay: float = 0.9999,
        use_after_updates: int = 300,
        **kwargs
    ):
        # Back-compat: allow `decay=` alias
        if "decay" in kwargs and kwargs["decay"] is not None:
            base_decay = kwargs.pop("decay")
        if kwargs:
            raise TypeError(f"Unexpected kwargs for CPUEMA: {list(kwargs.keys())}")

        self.base_decay = float(base_decay)
        self.num_updates = 0
        self.use_after_updates = int(use_after_updates)

        self.shadow: dict[str, torch.Tensor] = {}
        self.collected: dict[str, torch.Tensor] | None = None

        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    # initialize shadow exactly on current weights (good for fresh runs)
                    self.shadow[n] = p.detach().cpu().clone()

    def current_decay(self) -> float:
        # SD-like warm-up: fast tracking at the beginning, cap by base_decay
        d = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.base_decay, d)

    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.current_decay()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p_cpu = p.detach().to(device="cpu", dtype=self.shadow[n].dtype)
            self.shadow[n].mul_(d).add_(p_cpu, alpha=1.0 - d)

    @torch.no_grad()
    def bootstrap(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().to(
                    device="cpu",
                    dtype=self.shadow[n].dtype if n in self.shadow else p.dtype
                ).clone()

    def ready(self) -> bool:
        return self.num_updates >= self.use_after_updates

    @torch.no_grad()
    def store(self, model: nn.Module):
        self.collected = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                tmp = self.shadow[n].to(device=p.device, dtype=p.dtype, non_blocking=True)
                p.data.copy_(tmp)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and self.collected is not None:
                p.data.copy_(self.collected[n].to(p.device, non_blocking=True))
        self.collected = None


def param_groups(model):
    """
    Splitta i parametri in due gruppi:
      - weight decay standard su pesi "normali"
      - NO weight decay su bias e layer norm/batch norm
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_l = n.lower()
        is_bias = (".bias" in n) or (n.endswith("bias"))
        is_norm = ("norm" in name_l) or ("layernorm" in name_l) or ("bn" in name_l)
        if is_bias or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": 1e-2},
        {"params": no_decay, "weight_decay": 0.0},
    ]
