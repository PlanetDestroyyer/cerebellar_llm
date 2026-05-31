"""
GemmaCerebellar SLM — Architecture

Incorporates 2025-2026 SOTA:
  - RMSNorm (Gemma3, Qwen3, Llama3)
  - RoPE with high theta=500000 (Phi-4 style)
  - SwiGLU FFN (Gemma3, Qwen3)
  - Grouped Query Attention / GQA (Gemma3, Qwen3)
  - QK-Norm (Qwen3) — normalize Q,K before dot product for stability
  - PyTorch SDPA / Flash Attention backend
  - Tied input-output embeddings

Our additions:
  - CerebellarModule: Purkinje cell Hebbian error correction (no backprop)
  - Applied every N layers as a correction signal
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ── Rotary Position Embeddings (RoPE) ─────────────────────────────────────────

def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t     = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos   = freqs.cos().to(dtype)
    sin   = freqs.sin().to(dtype)
    return cos, sin  # [seq_len, half]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # [1, 1, T, half]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


# ── Grouped Query Attention + QK-Norm ────────────────────────────────────────

class GQAttention(nn.Module):
    """
    Grouped Query Attention with optional QK-Norm (Qwen3 style).
    Uses PyTorch SDPA for flash attention backend on CUDA.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.use_flash  = cfg.use_flash_attn

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model,    bias=False)

        # QK-Norm (Qwen3) — normalize Q and K independently before attention
        if cfg.use_qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self,
                x:   torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm applied per head
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (flash backend on CUDA)
        if self.use_flash and x.is_cuda:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            else:
                causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
                scores = scores + causal
            out = F.softmax(scores, dim=-1) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Gemma3, Qwen3, Llama3 standard).
    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate  = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up    = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down  = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn  = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn   = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Cerebellar Module ─────────────────────────────────────────────────────────

class CerebellarModule(nn.Module):
    """
    Biologically-inspired cerebellar error correction module.

    Architecture:
      1. Granule layer: fixed random sparse expansion (d_model → granule_dim)
         Analogous to cerebellar granule cells — high-dimensional sparse codes.
      2. Purkinje cells: adaptive linear layer (granule_dim → d_model)
         Updated via Hebbian learning rule, NOT backpropagation.
         Analogous to Purkinje cells receiving climbing fiber error signals.

    Learning rule (online Hebbian):
      dW_purkinje ∝ error * granule_activation
      where error = target - current output (approximated as -current hidden state)

    This module adds a correction signal to the transformer's hidden states,
    theoretically improving OOD generalization by providing a fast-adapting
    error-correction pathway alongside slow gradient-descent learning.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d   = cfg.d_model
        g   = cfg.granule_dim
        sp  = cfg.cerebellar_sparsity
        self.lr    = cfg.cerebellar_lr
        self.scale = cfg.correction_scale

        # Fixed random granule layer (no gradient)
        self.granule = nn.Linear(d, g, bias=False)
        nn.init.normal_(self.granule.weight, std=1.0 / math.sqrt(d))
        self.granule.weight.requires_grad_(False)

        # Sparse mask — only top-k granule neurons active per sample
        self.k = max(1, int(sp * g))

        # Adaptive Purkinje cells (Hebbian, not backprop)
        self.purkinje = nn.Linear(g, d, bias=False)
        nn.init.zeros_(self.purkinje.weight)
        self.purkinje.weight.requires_grad_(False)

        # Projection to mix correction into residual stream
        self.proj = nn.Linear(d, d, bias=False)

    @torch.no_grad()
    def hebbian_update(self, granule_act: torch.Tensor, error: torch.Tensor) -> None:
        """Online Hebbian update: dW = lr * mean(error^T * granule_act)."""
        B, T, G = granule_act.shape
        B, T, D = error.shape
        # Average over batch and sequence
        dW = torch.einsum('btd,btg->dg', error, granule_act) / (B * T)
        self.purkinje.weight.add_(self.lr * dW)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Granule layer: sparse activation (top-k)
        g_pre = self.granule(x)                         # [B, T, G]
        topk_vals, topk_idx = g_pre.topk(self.k, dim=-1)
        g_act = torch.zeros_like(g_pre)
        g_act.scatter_(-1, topk_idx, F.relu(topk_vals))  # sparse

        # Purkinje output: correction signal
        correction = self.purkinje(g_act)               # [B, T, D]
        correction = self.proj(correction)

        # Hebbian update: treat -x as the "climbing fiber" error signal
        # (negative gradient = error correction direction)
        if self.training:
            self.hebbian_update(g_act.detach(), -x.detach())

        return x + self.scale * correction


# ── Main Model ────────────────────────────────────────────────────────────────

class GemmaCerebellarSLM(nn.Module):
    """
    ~100M parameter SLM with 2025-2026 SOTA architecture + cerebellar extension.

    Architecture:
      - Token embedding (tied with LM head)
      - N transformer blocks: RMSNorm + GQA(QK-Norm) + RMSNorm + SwiGLU
      - Optional CerebellarModule every K layers
      - Final RMSNorm + LM head (tied)

    Benchmarks on WikiText-103 (target):
      Baseline (no cerebellar): comparable to published SLMs at 100M scale
      + Cerebellar: hypothesis — lower perplexity on OOD / domain-shifted splits
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed     = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out  = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie embeddings (standard SOTA)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Cerebellar modules (every cerebellar_every layers)
        if cfg.use_cerebellar:
            self.cerebellar_modules = nn.ModuleDict({
                str(i): CerebellarModule(cfg)
                for i in range(cfg.n_layers)
                if i % cfg.cerebellar_every == 0
            })
        else:
            self.cerebellar_modules = {}

        # RoPE cache (computed lazily)
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'embed' in name or 'lm_head' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'weight' in name and p.dim() == 1:
                nn.init.ones_(p)

    def _get_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            self._rope_cos, self._rope_sin = build_rope_cache(
                max(seq_len, self.cfg.max_seq_len),
                self.cfg.head_dim, self.cfg.rope_theta, device, dtype,
            )
        return (self._rope_cos[:seq_len].to(device),
                self._rope_sin[:seq_len].to(device))

    def forward(self,
                input_ids: torch.Tensor,
                labels:    Optional[torch.Tensor] = None) -> dict:
        B, T = input_ids.shape
        device, dtype = input_ids.device, self.embed.weight.dtype

        x   = self.embed(input_ids)
        cos, sin = self._get_rope(T, device, x.dtype)

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if self.cfg.use_cerebellar and str(i) in self.cerebellar_modules:
                x = self.cerebellar_modules[str(i)](x)

        x      = self.norm_out(x)
        logits = self.lm_head(x)   # [B, T, V]

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(ctx)["logits"][:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
