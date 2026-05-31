"""
GemmaCerebellar SLM — Main Experiment
======================================

Three-way comparison on WikiText-103:
  1. Baseline GemmaSLM     — SOTA 2025-2026 arch (RoPE+RMSNorm+SwiGLU+GQA+QK-Norm)
  2. + Cerebellar module   — Purkinje/Hebbian error correction (no backprop)
  3. + Microglia pruning   — Activity-based synaptic pruning vs magnitude

Hypothesis:
  H1: Cerebellar module reduces OOD perplexity gap (better generalization)
  H2: Microglia pruning maintains OOD accuracy better than magnitude pruning
      at equal sparsity

Hardware target: RTX 4050 6GB (Kaggle P100/T4/A100 also supported)
  - bf16 mixed precision
  - Flash Attention via PyTorch SDPA
  - Gradient checkpointing

Usage:
  # Full experiment (WikiText-103, ~100M params, ~3-6 hrs on Kaggle A100)
  uv run python main.py

  # Quick test (WikiText-103 subset, small model, ~5 min)
  uv run python main.py --quick

  # Single run
  uv run python main.py --run baseline
  uv run python main.py --run cerebellar
  uv run python main.py --run microglia

  # Custom steps/batch
  uv run python main.py --steps 20000 --batch 16 --seq_len 512

Kaggle setup:
  !pip install tiktoken datasets
  !git clone <repo> && cd gemma_cerebellar_slm
  !python main.py --steps 30000 --batch 16 --wandb
"""
from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path

from config import (
    ModelConfig, config_100m, config_100m_cerebellar,
    config_100m_microglia, config_100m_full, config_quick, count_params
)
from model import GemmaCerebellarSLM
from dataset import get_dataloaders
from train import train
from visualize import plot_comparison, plot_training_curves

RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_model_info(cfg: ModelConfig, model: GemmaCerebellarSLM, name: str):
    est   = count_params(cfg)
    actual = model.num_params()
    print(f"\n  [{name}]")
    print(f"  Params: {actual/1e6:.1f}M  (est: {est['total_M']:.1f}M)")
    print(f"    Embed: {est['embedding_M']:.1f}M  |  Per-layer: {est['per_layer_M']:.1f}M × {cfg.n_layers}")
    print(f"  QK-Norm: {cfg.use_qk_norm}  |  RoPE theta: {cfg.rope_theta:.0e}")
    print(f"  GQA: {cfg.n_heads}Q / {cfg.n_kv_heads}KV  |  FFN dim: {cfg.ffn_dim}")
    print(f"  Cerebellar: {cfg.use_cerebellar}  |  Microglia: {cfg.use_microglia}")


def build_and_train(cfg: ModelConfig, name: str, loaders: dict,
                    args: argparse.Namespace) -> dict:
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    model = GemmaCerebellarSLM(cfg).to(DEVICE)
    print_model_info(cfg, model, name)

    # Gradient checkpointing for VRAM savings
    if DEVICE.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead") if not args.no_compile else model

    results = train(
        model         = model,
        loaders       = loaders,
        cfg           = cfg,
        out_dir       = RESULTS_DIR / name,
        max_steps     = args.steps,
        warmup_steps  = max(200, args.steps // 25),
        lr_max        = args.lr,
        lr_min        = args.lr / 10,
        grad_clip     = 1.0,
        grad_accum    = args.grad_accum,
        log_every     = 50,
        eval_every    = max(500, args.steps // 20),
        save_every    = max(2000, args.steps // 5),
        use_wandb     = args.wandb,
        run_name      = name,
        dtype_str     = args.dtype,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="GemmaCerebellar SLM Experiment")
    parser.add_argument("--run",        default="all",
                        choices=["all", "baseline", "cerebellar", "microglia", "full"])
    parser.add_argument("--quick",      action="store_true",
                        help="Quick test: small model, 2M tokens, 1000 steps")
    parser.add_argument("--steps",      type=int,   default=50_000)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--seq_len",    type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--dtype",      default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--wandb",      action="store_true")
    parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile")
    parser.add_argument("--cache_dir",  default=None, help="HuggingFace cache dir")
    parser.add_argument("--workers",    type=int,   default=2)
    args = parser.parse_args()

    if args.quick:
        args.steps    = 1_000
        args.batch    = 4
        args.seq_len  = 256
        args.grad_accum = 2

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")
        if not torch.cuda.is_bf16_supported():
            args.dtype = "fp16"

    # Load data (shared across all runs)
    print("\nLoading WikiText-103...")
    loaders = get_dataloaders(
        seq_len     = args.seq_len,
        batch_size  = args.batch,
        cache_dir   = args.cache_dir,
        num_workers = args.workers,
        quick       = args.quick,
    )

    # Select configs
    if args.quick:
        configs = {
            "baseline":    config_quick(),
            "cerebellar":  config_quick(),
            "microglia":   config_quick(),
        }
        configs["cerebellar"].use_cerebellar = True
        configs["microglia"].use_microglia   = True
    else:
        configs = {
            "baseline":    config_100m(),
            "cerebellar":  config_100m_cerebellar(),
            "microglia":   config_100m_microglia(),
            "full":        config_100m_full(),
        }

    # Run experiments
    run_targets = (
        list(configs.keys()) if args.run == "all"
        else [args.run]
    )

    all_results = []
    for name in run_targets:
        if name not in configs:
            print(f"  [skip] Unknown run: {name}")
            continue
        results = build_and_train(configs[name], name, loaders, args)
        all_results.append(results)

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Run':20s}  {'Val PPL':>8}  {'OOD PPL':>8}  {'Test PPL':>9}  {'OOD Gap':>8}")
        for r in all_results:
            print(f"  {r['run_name']:20s}  {r['val_ppl']:8.2f}  "
                  f"{r['ood_ppl']:8.2f}  {r['test_ppl']:9.2f}  {r['ood_gap']:8.2f}")

        # Hypothesis check
        baseline = next((r for r in all_results if r["run_name"] == "baseline"), None)
        cereb    = next((r for r in all_results if r["run_name"] == "cerebellar"), None)
        micro    = next((r for r in all_results if r["run_name"] == "microglia"), None)

        if baseline and cereb:
            h1 = cereb["ood_gap"] < baseline["ood_gap"]
            delta = baseline["ood_gap"] - cereb["ood_gap"]
            print(f"\n  H1 (Cerebellar reduces OOD gap): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}"
                  f"  (delta={delta:+.2f})")

        if baseline and micro:
            h2 = micro["ood_gap"] < baseline["ood_gap"]
            delta = baseline["ood_gap"] - micro["ood_gap"]
            print(f"  H2 (Microglia better OOD than magnitude): {'SUPPORTED' if h2 else 'NOT SUPPORTED'}"
                  f"  (delta={delta:+.2f})")

        # Save combined results
        with open(RESULTS_DIR / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Plots
        plot_comparison(all_results, RESULTS_DIR)
        history_files = [RESULTS_DIR / name / f"results_{name}.json" for name in run_targets]
        history_files = [f for f in history_files if f.exists()]
        if history_files:
            plot_training_curves(history_files, RESULTS_DIR)

    print(f"\nDone. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
