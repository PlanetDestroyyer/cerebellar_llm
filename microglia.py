"""
Microglia-Inspired Pruning

Biology: Microglia eliminate weak synapses during development and maintenance.
They use activity signals — not just connection strength — to decide which
synapses to prune. Inactive synapses (regardless of weight magnitude) are
eliminated.

Our implementation:
  - Track exponential moving average (EMA) of |activation| per weight matrix
  - Prune weights with lowest cumulative activity (not lowest magnitude)
  - Contrast with magnitude pruning: uses |weight| value, ignores activity

Why activity > magnitude:
  A large weight that never activates wastes capacity.
  A small weight on a frequently active pathway is crucial.
  Microglia pruning preserves functional pathways, not just large values.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import GemmaCerebellarSLM


class MicrogliaTracker:
    """
    Tracks activation statistics for all weight matrices via forward hooks.
    Computes EMA of |input * weight| as proxy for synaptic activity.
    """

    def __init__(self, model: "GemmaCerebellarSLM", ema_decay: float = 0.99):
        self.model     = model
        self.ema_decay = ema_decay
        self.activity: dict[str, torch.Tensor] = {}
        self._hooks    = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad:
                hook = module.register_forward_hook(
                    self._make_hook(name, module)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str, module: nn.Linear):
        def hook(mod, inp, out):
            x = inp[0].detach()  # [B, T, in_features]
            w = mod.weight.detach()  # [out, in]

            # Activity = mean |activation| per output neuron
            # Proxy: mean |x| averaged over batch+seq, then broadcast to weight shape
            act_in = x.abs().mean(dim=(0, 1))  # [in_features]
            # Weight activity = |w_ij| * activity_j (input activity weighted by weight)
            weight_activity = w.abs() * act_in.unsqueeze(0)  # [out, in]

            if name not in self.activity:
                self.activity[name] = weight_activity.clone()
            else:
                self.activity[name].mul_(self.ema_decay).add_(
                    weight_activity * (1 - self.ema_decay)
                )
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def prune_microglia(model: "GemmaCerebellarSLM",
                    tracker: MicrogliaTracker,
                    prune_fraction: float = 0.3) -> dict:
    """
    Prune weights with lowest cumulative activity.
    Sets pruned weights to zero (unstructured sparsity).
    Returns pruning stats.
    """
    total_pruned = 0
    total_params = 0
    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not module.weight.requires_grad:
            continue
        if name not in tracker.activity:
            continue

        activity = tracker.activity[name]
        w        = module.weight.data

        n_prune  = int(prune_fraction * w.numel())
        if n_prune == 0:
            continue

        # Find threshold: bottom prune_fraction by activity
        flat_act = activity.view(-1)
        threshold = flat_act.kthvalue(n_prune).values.item()

        mask = activity >= threshold
        w.mul_(mask.float())

        pruned = (~mask).sum().item()
        total_pruned += pruned
        total_params += w.numel()
        stats[name] = {"pruned": pruned, "total": w.numel(),
                       "sparsity": pruned / w.numel()}

    overall_sparsity = total_pruned / total_params if total_params > 0 else 0.0
    return {"overall_sparsity": overall_sparsity, "total_pruned": total_pruned,
            "total_params": total_params, "per_layer": stats}


def prune_magnitude(model: "GemmaCerebellarSLM",
                    prune_fraction: float = 0.3) -> dict:
    """
    Baseline: prune by weight magnitude (smallest |w| → zeroed).
    Standard pruning approach for comparison.
    """
    total_pruned = 0
    total_params = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not module.weight.requires_grad:
            continue

        w       = module.weight.data
        n_prune = int(prune_fraction * w.numel())
        if n_prune == 0:
            continue

        flat     = w.abs().view(-1)
        threshold = flat.kthvalue(n_prune).values.item()
        mask     = w.abs() >= threshold
        w.mul_(mask.float())
        total_pruned += (~mask).sum().item()
        total_params += w.numel()

    return {
        "overall_sparsity": total_pruned / total_params if total_params > 0 else 0.0,
        "total_pruned": total_pruned,
        "total_params": total_params,
    }
