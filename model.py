"""
LLM + Cerebellar module wrapper.

For experiments without a real LLM, uses a small transformer (GPT-like)
trained on a language modeling task. The cerebellar module wraps it.

For real LLM experiments via API, uses the correction module on embeddings.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from cerebellum import CerebellarModule


class MiniGPT(nn.Module):
    """Small transformer for language modeling experiments."""

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._last_hidden: Optional[torch.Tensor] = None

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: [B, T]. Returns (logits [B, T, V], hidden [B, T, D])."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(pos)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        hidden = self.transformer(x, mask=mask, is_causal=True)
        self._last_hidden = hidden.detach()

        logits = self.lm_head(hidden)
        return logits, hidden

    def get_hidden(self) -> Optional[torch.Tensor]:
        return self._last_hidden


from typing import Optional


class CerebellarLLM(nn.Module):
    """LLM + cerebellar correction module."""

    def __init__(
        self,
        base_model: MiniGPT,
        cerebellar: CerebellarModule,
        correction_scale: float = 0.1,
    ):
        super().__init__()
        self.base = base_model
        self.cerebellum = cerebellar
        self.correction_scale = correction_scale

    def forward(
        self,
        tokens: torch.Tensor,
        apply_correction: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (corrected_logits, base_logits, hidden).
        """
        base_logits, hidden = self.base(tokens)

        if apply_correction:
            B, T, D = hidden.shape
            ctx = hidden.reshape(B * T, D)
            correction, granule = self.cerebellum(ctx)
            correction = correction.reshape(B, T, -1)
            corrected_logits = base_logits + self.correction_scale * correction
        else:
            corrected_logits = base_logits
            granule = None

        return corrected_logits, base_logits, hidden
