"""
Microglia-Inspired Synaptic Pruning for LLM Generalization.

Blog hypothesis: Biological brains first over-produce connections, then
microglia prune weak/inactive synapses. This creates generalized intelligence.
LLMs scaled without pruning accumulate dead weight that hurts OOD generalization.

Microglia pruning criterion (dual-gate):
  Prune weight W[i,j] if:
    |W[i,j]| < magnitude_threshold   (weak synapse)
    AND
    activity_count[i,j] < activity_threshold  (rarely activated)

vs. standard magnitude-only pruning (control).

Key difference: activity tracking means weights that are small but frequently
used are preserved — just like microglia spare active synapses even if small.
"""
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ActivityTracker:
    """Track activation frequency per weight (proxy for synaptic activity)."""
    shape: tuple
    counts: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.counts = torch.zeros(self.shape)

    def update(self, pre_activation: torch.Tensor, post_activation: torch.Tensor):
        """
        Increment count for weight [i,j] if both pre[j] and post[i] are active.
        Activity = |activation| > 0.1 threshold.
        """
        pre_active = (pre_activation.abs() > 0.1).float()  # (batch, in)
        post_active = (post_activation.abs() > 0.1).float()  # (batch, out)
        # Outer product: weight [out, in] is "active" if both neurons fired
        activity = torch.einsum("bi,bj->ij", post_active, pre_active) / pre_activation.shape[0]
        self.counts += activity.cpu()

    def reset(self):
        self.counts.zero_()


class MicrogliaPruner:
    """
    Prune weights using dual-gate criterion: magnitude AND activity.
    """
    def __init__(
        self,
        magnitude_threshold: float = 0.01,
        activity_threshold: float = 0.05,
        prune_fraction: float = 0.3,  # prune this fraction of weights
    ):
        self.mag_thresh = magnitude_threshold
        self.act_thresh = activity_threshold
        self.prune_fraction = prune_fraction

    def microglia_mask(
        self,
        weight: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return binary mask: 1 = keep, 0 = prune.
        Prune if small AND rarely active.
        """
        mag_weak = weight.abs() < self.mag_thresh
        act_weak = activity < self.act_thresh
        should_prune = mag_weak & act_weak

        # If fewer than prune_fraction pruned, also prune smallest magnitudes
        n_total = weight.numel()
        n_pruned = should_prune.sum().item()
        target_pruned = int(self.prune_fraction * n_total)

        if n_pruned < target_pruned:
            # Add more by magnitude alone (but only from act_weak weights)
            extra_needed = target_pruned - n_pruned
            act_weak_only = act_weak & ~mag_weak
            if act_weak_only.sum() > 0:
                mag_vals = weight.abs().clone()
                mag_vals[~act_weak_only] = float("inf")
                threshold = torch.kthvalue(mag_vals.flatten(), min(extra_needed, act_weak_only.sum().item())).values
                should_prune |= (act_weak_only & (weight.abs() <= threshold))

        return ~should_prune  # keep mask

    def magnitude_only_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Standard magnitude pruning (control)."""
        n_prune = int(self.prune_fraction * weight.numel())
        threshold = torch.kthvalue(weight.abs().flatten(), n_prune).values
        return weight.abs() > threshold

    def apply(
        self,
        model: nn.Module,
        trackers: dict[str, ActivityTracker],
        method: str = "microglia",
    ) -> dict:
        """Apply pruning masks to all Linear layers. Returns sparsity stats."""
        stats = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            W = module.weight.data
            if method == "microglia" and name in trackers:
                mask = self.microglia_mask(W, trackers[name].counts.to(W.device))
            else:
                mask = self.magnitude_only_mask(W)

            module.weight.data *= mask.float()
            sparsity = 1.0 - mask.float().mean().item()
            stats[name] = {"sparsity": sparsity, "method": method}

        return stats


def attach_trackers(model: nn.Module) -> dict[str, ActivityTracker]:
    """Create an ActivityTracker for each Linear layer and register forward hooks."""
    trackers = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        tracker = ActivityTracker(shape=(module.out_features, module.in_features))
        trackers[name] = tracker

        def make_hook(t):
            def hook(mod, inp, out):
                t.update(inp[0].detach().cpu(), out.detach().cpu())
            return hook

        hooks.append(module.register_forward_hook(make_hook(tracker)))

    return trackers, hooks


def prune_and_evaluate(
    model: nn.Module,
    train_loader,
    test_id_loader,
    test_ood_loader,
    method: str,
    prune_fraction: float = 0.3,
    n_activity_epochs: int = 3,
    device: str = "cpu",
) -> dict:
    """
    Full pipeline: track activity → prune → evaluate ID+OOD.
    Returns accuracy and sparsity stats.
    """
    import copy
    from train import evaluate_model

    model_copy = copy.deepcopy(model).to(device)
    trackers, hooks = attach_trackers(model_copy)

    # Activity tracking phase
    model_copy.eval()
    with torch.no_grad():
        for epoch in range(n_activity_epochs):
            for xb, yb in train_loader:
                model_copy(xb.to(device))

    for hook in hooks:
        hook.remove()

    # Prune
    pruner = MicrogliaPruner(prune_fraction=prune_fraction)
    sparsity_stats = pruner.apply(model_copy, trackers, method=method)

    avg_sparsity = float(np.mean([v["sparsity"] for v in sparsity_stats.values()]))

    # Evaluate
    id_eval = evaluate_model(model_copy, test_id_loader, device)
    ood_eval = evaluate_model(model_copy, test_ood_loader, device)

    return {
        "method": method,
        "avg_sparsity": avg_sparsity,
        "id_accuracy": id_eval["accuracy"],
        "ood_accuracy": ood_eval["accuracy"],
        "ood_gap": ood_eval["perplexity"] - id_eval["perplexity"],
    }
