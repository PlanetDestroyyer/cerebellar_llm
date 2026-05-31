"""
Model configurations for GemmaCerebellar SLM experiments.

Architectures grounded in:
  Gemma 3 (Google, 2025)    — RoPE, GQA, RMSNorm, SwiGLU
  Qwen3 (Alibaba, 2025)     — QK-Norm for training stability
  DeepSeek-V3 (2024)        — MLA concept (simplified here as GQA)
  Phi-4 Mini (Microsoft)    — High RoPE base, tied embeddings

Our additions:
  Cerebellar module         — Purkinje cell Hebbian error correction
  Microglia pruning         — Activity-based synaptic pruning
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ── Dimensions ───────────────────────────────────────────────────────────
    vocab_size:     int   = 50257    # GPT-2 BPE tokenizer
    d_model:        int   = 768      # hidden dim
    n_heads:        int   = 12       # query heads
    n_kv_heads:     int   = 4        # key/value heads (GQA, Gemma3/Qwen3 style)
    n_layers:       int   = 12       # transformer layers
    ffn_multiplier: float = 8/3      # SwiGLU intermediate = ffn_multiplier * d_model
    max_seq_len:    int   = 1024

    # ── Architectural flags (SOTA 2025-2026) ─────────────────────────────────
    use_qk_norm:    bool  = True     # QK-Norm (Qwen3) — normalize Q,K before attention
    use_rope:       bool  = True     # Rotary position embeddings
    rope_theta:     float = 500000.0 # High base freq (Phi-4 style, better long-ctx)
    tie_embeddings: bool  = True     # Tie input/output embeddings (standard SOTA)
    use_flash_attn: bool  = True     # Use PyTorch SDPA (flash attention backend)

    # ── Regularization ───────────────────────────────────────────────────────
    dropout:        float = 0.0      # 0 for pretraining (standard)
    rms_norm_eps:   float = 1e-6

    # ── Cerebellar module ─────────────────────────────────────────────────────
    use_cerebellar: bool  = False
    granule_dim:    int   = 2048     # sparse expansion layer size
    cerebellar_sparsity: float = 0.05  # fraction of granule neurons active
    cerebellar_lr:  float = 0.01     # Hebbian learning rate (not gradient)
    correction_scale: float = 0.1   # scale of cerebellar correction signal
    cerebellar_every: int = 4        # apply cerebellar module every N layers

    # ── Microglia pruning ─────────────────────────────────────────────────────
    use_microglia:  bool  = False
    prune_fraction: float = 0.3      # fraction of weights to prune
    prune_every:    int   = 1000     # prune every N steps
    activity_ema:   float = 0.99     # EMA decay for activity tracking

    @property
    def ffn_dim(self) -> int:
        return int(self.ffn_multiplier * self.d_model)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ── Preset configs ────────────────────────────────────────────────────────────

def config_100m() -> ModelConfig:
    """~100M params: d=768, 12L, 12H, GQA-4. Standard Gemma3/Qwen3 style."""
    return ModelConfig(
        vocab_size=50257, d_model=768, n_heads=12, n_kv_heads=4,
        n_layers=12, max_seq_len=1024,
        use_qk_norm=True, rope_theta=500000.0,
    )


def config_100m_cerebellar() -> ModelConfig:
    """100M + cerebellar correction module."""
    cfg = config_100m()
    cfg.use_cerebellar = True
    return cfg


def config_100m_microglia() -> ModelConfig:
    """100M + microglia activity-based pruning."""
    cfg = config_100m()
    cfg.use_microglia = True
    return cfg


def config_100m_full() -> ModelConfig:
    """100M + both cerebellar + microglia."""
    cfg = config_100m()
    cfg.use_cerebellar = True
    cfg.use_microglia  = True
    return cfg


def config_quick() -> ModelConfig:
    """Small config for quick testing (smoke test)."""
    return ModelConfig(
        vocab_size=50257, d_model=256, n_heads=4, n_kv_heads=2,
        n_layers=4, max_seq_len=256,
    )


def count_params(cfg: ModelConfig) -> dict:
    """Estimate parameter count without building the model."""
    head_dim   = cfg.d_model // cfg.n_heads
    kv_dim     = cfg.n_kv_heads * head_dim

    embed      = cfg.vocab_size * cfg.d_model
    per_layer  = (
        cfg.d_model * cfg.d_model +          # Q
        cfg.d_model * kv_dim +               # K
        cfg.d_model * kv_dim +               # V
        cfg.d_model * cfg.d_model +          # O
        cfg.d_model * cfg.ffn_dim +          # gate
        cfg.d_model * cfg.ffn_dim +          # up
        cfg.ffn_dim * cfg.d_model +          # down
        cfg.d_model * 2                      # 2× RMSNorm (negligible)
    )
    if cfg.use_qk_norm:
        per_layer += head_dim + kv_dim       # QK-Norm scale params

    lm_head  = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    total    = embed + cfg.n_layers * per_layer + cfg.d_model + lm_head

    return {
        "embedding_M":   embed / 1e6,
        "per_layer_M":   per_layer / 1e6,
        "total_M":       total / 1e6,
    }
