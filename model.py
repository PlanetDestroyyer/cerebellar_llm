"""
Cerebellar wrapper for any HuggingFace CausalLM.

Works with: SmolLM, SmolLM2, Qwen2.5, Qwen3.x — any decoder-only HF model.

Attaches CerebellarModule to hidden states via forward hooks at every N layers.
No modification to base model weights — fully additive.

Cerebellar module:
  1. Granule layer: fixed random sparse expansion (hidden_dim -> granule_dim)
  2. Purkinje cells: adaptive linear (granule_dim -> hidden_dim)
     Updated via Hebbian rule — NOT backpropagation.
  Correction = scale * purkinje(sparse_granule(hidden_state))
  Hebbian update: dW proportional to error * granule_activation
  where error = -hidden_state (climbing fiber approximation)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelEntry, CerebellarConfig


class CerebellarModule(nn.Module):
    def __init__(self, hidden_dim: int, cfg: CerebellarConfig):
        super().__init__()
        g = hidden_dim * cfg.granule_expansion
        self.k     = max(1, int(cfg.cerebellar_sparsity * g))
        self.lr    = cfg.cerebellar_lr
        self.scale = cfg.correction_scale

        # Fixed random granule layer (no grad)
        self.granule = nn.Linear(hidden_dim, g, bias=False)
        nn.init.normal_(self.granule.weight, std=1.0 / math.sqrt(hidden_dim))
        self.granule.weight.requires_grad_(False)

        # Adaptive Purkinje cells (Hebbian only, no backprop)
        self.purkinje = nn.Linear(g, hidden_dim, bias=False)
        nn.init.zeros_(self.purkinje.weight)
        self.purkinje.weight.requires_grad_(False)

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @torch.no_grad()
    def _hebbian_update(self, g_act: torch.Tensor, error: torch.Tensor) -> None:
        dW = torch.einsum('btd,btg->dg', error, g_act) / (error.shape[0] * error.shape[1])
        self.purkinje.weight.add_(self.lr * dW)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g_pre = self.granule(x)
        topk_v, topk_i = g_pre.topk(self.k, dim=-1)
        g_act = torch.zeros_like(g_pre)
        g_act.scatter_(-1, topk_i, F.relu(topk_v))
        correction = self.proj(self.purkinje(g_act))
        if self.training:
            self._hebbian_update(g_act.detach(), -x.detach())
        return x + self.scale * correction


class CerebellarHFWrapper(nn.Module):
    """
    Wraps any HuggingFace CausalLM and injects cerebellar modules
    into transformer hidden states via forward hooks at every N layers.
    Base model weights are frozen — only cerebellar proj weights are trainable.
    """

    def __init__(self, base_model: nn.Module, entry: ModelEntry, cfg: CerebellarConfig):
        super().__init__()
        self.base   = base_model
        self.entry  = entry
        self.cfg    = cfg
        self._hooks = []

        # Build cerebellar modules
        self.cerebellar_modules = nn.ModuleDict()
        if cfg.use_cerebellar:
            for i in range(entry.n_layers):
                if i % cfg.cerebellar_every == 0:
                    self.cerebellar_modules[str(i)] = CerebellarModule(
                        entry.hidden_dim, cfg
                    )
            self._register_hooks()

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad_(False)

    def _get_layers(self) -> nn.ModuleList:
        """Return transformer layer list for any HF architecture."""
        for attr in ["model.layers", "transformer.h", "model.decoder.layers",
                     "gpt_neox.layers"]:
            obj = self.base
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, nn.ModuleList):
                    return obj
            except AttributeError:
                continue
        raise AttributeError(f"Cannot find transformer layers in {type(self.base).__name__}")

    def _register_hooks(self):
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            if str(i) not in self.cerebellar_modules:
                continue
            cereb = self.cerebellar_modules[str(i)]

            def make_hook(cereb_mod):
                def hook(module, inputs, output):
                    # HF layers return (hidden_state,) or (hidden_state, cache, ...)
                    if isinstance(output, tuple):
                        hidden    = output[0]
                        corrected = cereb_mod(hidden)
                        return (corrected,) + output[1:]
                    return cereb_mod(output)
                return hook

            h = layer.register_forward_hook(make_hook(cereb))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None, **kwargs):
        return self.base(input_ids=input_ids, labels=labels, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

    def num_params(self) -> dict:
        base  = sum(p.numel() for p in self.base.parameters())
        cereb = sum(p.numel() for p in self.cerebellar_modules.parameters())
        return {"base_M": base/1e6, "cerebellar_M": cereb/1e6,
                "total_M": (base+cereb)/1e6}
