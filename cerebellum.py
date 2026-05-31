"""
Cerebellar Error Correction Module for LLM adaptation.

Biological basis:
- Cerebellum learns to predict sensory consequences of actions
- Purkinje cells receive error signals (climbing fibers) and adapt
- Learning rule: delta_w = -eta * error * eligibility_trace

Applied to LLMs:
- "Mossy fibers" = current context representation (LLM hidden state)
- "Climbing fiber error" = prediction error (|predicted - actual| for next token)
- "Purkinje output" = correction signal applied to LLM generation
- Granule cell expansion = sparse, high-dim representation of context

This implements a lightweight, trainable correction module that wraps
around any LLM's output distribution and adjusts it based on learned
error patterns — analogous to how the cerebellum fine-tunes motor output.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GranuleLayer(nn.Module):
    """
    Sparse expansion: low-dim context → high-dim granule representation.
    Analogous to granule cells in cerebellum (huge expansion ratio).
    """

    def __init__(self, input_dim: int, granule_dim: int, sparsity: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, granule_dim, bias=True)
        self.sparsity = sparsity
        # Fixed random weights (granule cells don't learn — cerebellum biology)
        nn.init.normal_(self.linear.weight, 0, 1.0 / np.sqrt(input_dim))
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.linear(x))
        # k-winner-take-all sparsification
        k = max(1, int(h.shape[-1] * self.sparsity))
        topk_vals, _ = h.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        return h * (h >= threshold).float()


class PurkinjeCell(nn.Module):
    """
    Purkinje-like output cell: learns to predict and correct errors.
    Receives granule representation, outputs correction to LLM logits.
    """

    def __init__(self, granule_dim: int, output_dim: int):
        super().__init__()
        self.correction = nn.Linear(granule_dim, output_dim, bias=True)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, granule: torch.Tensor) -> torch.Tensor:
        return self.correction(granule)


class EligibilityTrace:
    """
    Maintains eligibility trace for cerebellar learning rule.
    e_{t+1} = gamma * e_t + granule_t
    """

    def __init__(self, granule_dim: int, gamma: float = 0.9):
        self.gamma = gamma
        self.trace = None
        self.granule_dim = granule_dim

    def update(self, granule: torch.Tensor) -> torch.Tensor:
        if self.trace is None:
            self.trace = granule.detach().clone()
        else:
            self.trace = self.gamma * self.trace + granule.detach()
        return self.trace

    def reset(self) -> None:
        self.trace = None


class CerebellarModule(nn.Module):
    """
    Full cerebellar error-correction module.

    Input:  context_embedding [B, D] from LLM hidden state
    Output: correction_logits [B, V] to add to LLM output logits

    Training signal: next-token prediction error (climbing fiber error).
    """

    def __init__(
        self,
        context_dim: int,
        vocab_size: int,
        granule_dim: int = 512,
        sparsity: float = 0.05,
        gamma_trace: float = 0.9,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.granule = GranuleLayer(context_dim, granule_dim, sparsity)
        self.purkinje = PurkinjeCell(granule_dim, vocab_size)
        self.eligibility = EligibilityTrace(granule_dim, gamma_trace)
        self.lr = learning_rate
        self.context_dim = context_dim
        self.vocab_size = vocab_size

        # Error history for analysis
        self._error_history: list[float] = []

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (correction_logits, granule_representation).
        correction_logits should be added to LLM output logits.
        """
        granule = self.granule(context)
        correction = self.purkinje(granule)
        return correction, granule

    def apply_climbing_fiber(
        self,
        granule: torch.Tensor,
        error_signal: torch.Tensor,
    ) -> None:
        """
        Online learning rule (bypasses autograd for biological fidelity):
        delta_w = -lr * error * eligibility_trace

        error_signal: [B, V] — prediction error (actual - predicted logits)
        """
        trace = self.eligibility.update(granule)

        with torch.no_grad():
            # Hebbian-like update: error × trace
            delta_w = self.lr * (error_signal.T @ trace) / trace.shape[0]
            self.purkinje.correction.weight.data += delta_w

        self._error_history.append(float(error_signal.abs().mean().item()))

    def reset_trace(self) -> None:
        self.eligibility.reset()

    @property
    def mean_error(self) -> float:
        if not self._error_history:
            return 0.0
        return float(np.mean(self._error_history[-100:]))
