import torch
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