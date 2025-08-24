from __future__ import annotations
import torch
from typing import List, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

TAB_PREFIXES: List[str] = [
    "tab_transformer",
    "final_tab",
    "diag_head",
    "tab_map_down",
    "cond_map_down"
]

def freeze_tabular_branch(model: nn.Module) -> None:
    """
    Freezes every tab-only parameter inside the (possibly DDP-wrapped) model.
    """
    # unwrap if we received a DDP wrapper
    target = model.module if isinstance(model, DDP) else model

    # print("questi layers sono freezati")
    # idx = 0
    for name, param in target.named_parameters():
        clean_name = name.split(".", 1)[-1]
        if any(clean_name.startswith(prefix) for prefix in TAB_PREFIXES):
            param.requires_grad = False
            # print(f"{idx:>3}: {name:30}")
            # idx = idx + 1

class TabEarlyStopper:
    """
    Monitors validation *tabular* loss and freezes the tab branch
    when it stops improving for <patience> epochs.
    """
    def __init__(self, model, patience: int = 10, delta: float = 1e-4):
        self.model = model
        self.patience, self.delta = patience, delta
        self.best = float("inf")
        self.bad_epochs = 0
        self.triggered = False

    def step(self, val_tab_loss: float) -> None:
        if self.triggered:
            return
        if val_tab_loss < self.best - self.delta:
            self.best = val_tab_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Tab validation loss plateaued -- freezing tab branch.")
            freeze_tabular_branch(self.model)
            self.triggered = True
