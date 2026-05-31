"""
Experiment: Cerebellar Error Correction for LLM Adaptation

Hypothesis: LLM adaptation modeled via synaptic plasticity + cerebellar
error correction (Purkinje cell learning rule) outperforms pure gradient
descent for OOD generalization.

Architecture:
- Base LM: small GPT-like transformer
- Cerebellar module: fixed granule layer (sparse expansion) + adaptive Purkinje cells
- Learning rule: climbing-fiber error signal (Hebbian, not gradient descent)

Test: ID accuracy vs OOD (shifted pattern) accuracy.

Usage:
    python main.py [--epochs 15] [--vocab 100]
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from model import MiniGPT, CerebellarLLM
from cerebellum import CerebellarModule
from dataset import SequenceDataset
from train import train_base_model, train_cerebellar, evaluate_model
from visualize import plot_training_comparison, plot_ood_comparison

OUTPUT_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_base", type=int, default=15)
    parser.add_argument("--epochs_cerebellum", type=int, default=10)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")

    V = args.vocab
    SEQ_LEN = 32
    BATCH = 64

    # --- Datasets ---
    train_id = SequenceDataset(n_sequences=2000, seq_len=SEQ_LEN, vocab_size=V, ood=False, seed=args.seed)
    test_id = SequenceDataset(n_sequences=400, seq_len=SEQ_LEN, vocab_size=V, ood=False, seed=args.seed + 1)
    test_ood = SequenceDataset(n_sequences=400, seq_len=SEQ_LEN, vocab_size=V, ood=True, seed=args.seed + 2)

    train_loader = DataLoader(train_id, batch_size=BATCH, shuffle=True)
    id_loader = DataLoader(test_id, batch_size=BATCH)
    ood_loader = DataLoader(test_ood, batch_size=BATCH)

    print(f"Train ID: {len(train_id)}  |  Test ID: {len(test_id)}  |  Test OOD: {len(test_ood)}")

    # --- Baseline LM ---
    print("\n--- Training Baseline LM ---")
    baseline = MiniGPT(vocab_size=V, d_model=128, n_heads=4, n_layers=2, max_seq_len=SEQ_LEN)
    hist_base = train_base_model(baseline, train_loader, n_epochs=args.epochs_base,
                                  lr=1e-3, device=DEVICE)

    eval_base_id = evaluate_model(baseline.to(DEVICE), id_loader, DEVICE)
    eval_base_ood = evaluate_model(baseline.to(DEVICE), ood_loader, DEVICE)
    print(f"Baseline ID:  ppl={eval_base_id['perplexity']:.2f}  acc={eval_base_id['accuracy']:.3f}")
    print(f"Baseline OOD: ppl={eval_base_ood['perplexity']:.2f}  acc={eval_base_ood['accuracy']:.3f}")

    # --- Cerebellar LM ---
    print("\n--- Training Cerebellar Module (base frozen) ---")
    cerebellum = CerebellarModule(
        context_dim=128,
        vocab_size=V,
        granule_dim=512,
        sparsity=0.05,
        gamma_trace=0.9,
        learning_rate=0.01,
    )
    cereb_llm = CerebellarLLM(baseline, cerebellum, correction_scale=0.2)
    hist_cereb = train_cerebellar(cereb_llm, train_loader, n_epochs=args.epochs_cerebellum, device=DEVICE)

    eval_cereb_id = evaluate_model(cereb_llm.to(DEVICE), id_loader, DEVICE, use_correction=True)
    eval_cereb_ood = evaluate_model(cereb_llm.to(DEVICE), ood_loader, DEVICE, use_correction=True)
    print(f"Cerebellar ID:  ppl={eval_cereb_id['perplexity']:.2f}  acc={eval_cereb_id['accuracy']:.3f}")
    print(f"Cerebellar OOD: ppl={eval_cereb_ood['perplexity']:.2f}  acc={eval_cereb_ood['accuracy']:.3f}")

    # --- Key findings ---
    ood_gap_base = eval_base_ood["perplexity"] - eval_base_id["perplexity"]
    ood_gap_cereb = eval_cereb_ood["perplexity"] - eval_cereb_id["perplexity"]
    print(f"\nOOD generalization gap:")
    print(f"  Baseline:   {ood_gap_base:.2f}")
    print(f"  Cerebellar: {ood_gap_cereb:.2f}")
    finding = "SUPPORTED" if ood_gap_cereb < ood_gap_base else "NOT SUPPORTED"
    print(f"Hypothesis {finding}: cerebellar {'reduces' if ood_gap_cereb < ood_gap_base else 'does not reduce'} OOD gap")

    # Save
    eval_results = {
        "baseline": {"id": eval_base_id, "ood": eval_base_ood},
        "cerebellar": {"id": eval_cereb_id, "ood": eval_cereb_ood},
    }
    histories = {"baseline": hist_base, "cerebellar": hist_cereb}

    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(histories, f)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "ood_gap_baseline": ood_gap_base,
            "ood_gap_cerebellar": ood_gap_cereb,
            "hypothesis_supported": ood_gap_cereb < ood_gap_base,
            "finding": finding,
        }, f, indent=2)

    # Plots
    plot_training_comparison(histories, OUTPUT_DIR)
    plot_ood_comparison(eval_results, OUTPUT_DIR)

    # --- Activation Steering Extension ---
    print("\n--- Activation Steering vs Climbing Fiber (activation_steering.py) ---")
    from activation_steering import equivalence_demo, ood_correction_comparison
    eq = equivalence_demo()
    print(f"  Direction cosine similarity (CF ↔ Steering): {eq['direction_cosine_similarity']:.3f}")
    print(f"  {eq['interpretation']}")

    ood_cmp = ood_correction_comparison()
    print(f"  Steering error reduction:       {ood_cmp['steering_error_reduction']:.1%}")
    print(f"  Climbing fiber error reduction: {ood_cmp['cf_error_reduction']:.1%}")

    with open(OUTPUT_DIR / "activation_steering.json", "w") as f:
        import json as _json
        _json.dump({"equivalence": eq, "ood_comparison": ood_cmp}, f, indent=2)

    # --- Microglia Pruning Extension ---
    print("\n--- Microglia-Inspired Pruning vs Magnitude Pruning (microglia_pruning.py) ---")
    from microglia_pruning import prune_and_evaluate

    micro_result = prune_and_evaluate(
        baseline, train_loader, id_loader, ood_loader,
        method="microglia", prune_fraction=0.3,
        n_activity_epochs=2, device=DEVICE,
    )
    mag_result = prune_and_evaluate(
        baseline, train_loader, id_loader, ood_loader,
        method="magnitude", prune_fraction=0.3,
        n_activity_epochs=2, device=DEVICE,
    )
    print(f"  Microglia pruning  — sparsity={micro_result['avg_sparsity']:.1%}  "
          f"OOD acc={micro_result['ood_accuracy']:.3f}  OOD gap={micro_result['ood_gap']:.2f}")
    print(f"  Magnitude pruning  — sparsity={mag_result['avg_sparsity']:.1%}  "
          f"OOD acc={mag_result['ood_accuracy']:.3f}  OOD gap={mag_result['ood_gap']:.2f}")
    micro_better = micro_result['ood_accuracy'] >= mag_result['ood_accuracy']
    print(f"  Hypothesis {'SUPPORTED' if micro_better else 'NOT SUPPORTED'}: "
          f"microglia pruning {'≥' if micro_better else '<'} magnitude-only on OOD accuracy")

    with open(OUTPUT_DIR / "microglia_pruning.json", "w") as f:
        _json.dump({"microglia": micro_result, "magnitude": mag_result}, f, indent=2)

    print("\nDone. Check results/")


if __name__ == "__main__":
    main()
