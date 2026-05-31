"""
Model registry and experiment configuration.

Baselines pulled directly from HuggingFace — no training needed.
Cerebellar + microglia wrappers applied on top.

Models:
  SmolLM-135M    HuggingFaceTB/SmolLM-135M        135M  Llama-style
  SmolLM2-135M   HuggingFaceTB/SmolLM2-135M       135M  Llama-style v2
  Qwen2.5-0.5B   Qwen/Qwen2.5-0.5B                494M  Qwen arch
  Qwen2.5-1.5B   Qwen/Qwen2.5-1.5B               1500M  Qwen arch (large)

Note on Qwen3.5: update hf_id in qwen3.5 entry when smallest Qwen3 is released.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelEntry:
    hf_id:      str    # HuggingFace model ID
    short_name: str    # used in filenames / plots
    n_params_M: float  # approx params in millions
    arch:       str    # architecture family
    hidden_dim: int    # d_model (for cerebellar sizing)
    n_layers:   int    # number of transformer layers


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "smollm": ModelEntry(
        hf_id      = "HuggingFaceTB/SmolLM-135M",
        short_name = "SmolLM-135M",
        n_params_M = 135,
        arch       = "llama",
        hidden_dim = 576,
        n_layers   = 30,
    ),
    "smollm2": ModelEntry(
        hf_id      = "HuggingFaceTB/SmolLM2-135M",
        short_name = "SmolLM2-135M",
        n_params_M = 135,
        arch       = "llama",
        hidden_dim = 576,
        n_layers   = 30,
    ),
    "qwen2.5": ModelEntry(
        hf_id      = "Qwen/Qwen2.5-0.5B",
        short_name = "Qwen2.5-0.5B",
        n_params_M = 494,
        arch       = "qwen2",
        hidden_dim = 896,
        n_layers   = 24,
    ),
    "qwen3.5": ModelEntry(
        hf_id      = "Qwen/Qwen3.5-0.8B",
        short_name = "Qwen3.5-0.8B",
        n_params_M = 800,
        arch       = "qwen3",
        hidden_dim = 1024,
        n_layers   = 24,
    ),
}

ALL_MODELS = list(MODEL_REGISTRY.keys())


@dataclass
class CerebellarConfig:
    """Config for cerebellar + microglia wrappers on any base model."""
    # Cerebellar module
    use_cerebellar:      bool  = True
    granule_expansion:   int   = 4       # granule_dim = expansion * hidden_dim
    cerebellar_sparsity: float = 0.05
    cerebellar_lr:       float = 0.005   # Hebbian lr
    correction_scale:    float = 0.05
    cerebellar_every:    int   = 4       # attach every N layers

    # Microglia pruning
    use_microglia:  bool  = False
    prune_fraction: float = 0.30
    prune_every:    int   = 500
    activity_ema:   float = 0.99

    # Short fine-tuning after attaching cerebellar (0 = eval-only)
    finetune_steps: int   = 2_000
    finetune_lr:    float = 1e-4
    finetune_batch: int   = 8
    seq_len:        int   = 512
