"""Visualization for cerebellar LLM experiment."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_comparison(histories: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if "base_loss" in histories.get("cerebellar", {}):
        ax.plot(histories["cerebellar"]["base_loss"], "r--", label="Base LM (frozen)", linewidth=2)
        ax.plot(histories["cerebellar"]["corrected_loss"], "g-", label="+ Cerebellar correction", linewidth=2)
    if "loss" in histories.get("baseline", {}):
        ax.plot(histories["baseline"]["loss"], "b-", label="Baseline (no cerebellum)", linewidth=2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy Loss")
    ax.set_title("Training Loss: Cerebellar Correction vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if "cerebellum_error" in histories.get("cerebellar", {}):
        ax2.plot(histories["cerebellar"]["cerebellum_error"], "m-", linewidth=2, label="Climbing-fiber error")
        ax2.set_ylabel("Mean |Error Signal|")
        ax2.set_title("Cerebellar Learning: Error Signal (Climbing Fiber)\nDecaying error = cerebellum learned corrections")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Hypothesis: LLM adaptation via synaptic plasticity + cerebellar error correction\n"
        "surpasses pure gradient descent for OOD generalization",
        fontsize=10
    )
    fig.tight_layout()
    out = output_dir / "training_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Training comparison: {out}")
    plt.close(fig)


def plot_ood_comparison(eval_results: dict, output_dir: Path) -> None:
    conditions = ["ID (in-dist)", "OOD (out-of-dist)"]
    models = list(eval_results.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["perplexity", "accuracy"]
    ylabels = ["Perplexity (lower=better)", "Accuracy (higher=better)"]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        x = np.arange(len(conditions))
        width = 0.8 / len(models)
        for i, (model_name, color) in enumerate(zip(models, colors)):
            vals = [eval_results[model_name].get(split, {}).get(metric, 0)
                    for split in ["id", "ood"]]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model_name, color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Condition")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "OOD Generalization: Cerebellar Error Correction vs Baselines\n"
        "Key metric: OOD perplexity gap (smaller gap = better generalization)",
        fontsize=10
    )
    fig.tight_layout()
    out = output_dir / "ood_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"OOD comparison: {out}")
    plt.close(fig)
