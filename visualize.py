"""Plots for Cerebellar SLM experiment."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


MODELS     = ["smollm", "smollm2", "qwen2.5", "qwen3.5"]
CONDITIONS = ["baseline", "cerebellar", "microglia"]
COND_COLORS = {"baseline": "#3498db", "cerebellar": "#2ecc71", "microglia": "#e74c3c"}


def plot_comparison(results: list[dict], out_dir: Path) -> None:
    """4-panel: one per model, showing baseline vs cerebellar vs microglia."""
    model_keys = list(dict.fromkeys(r["model_key"] for r in results))
    n_models   = len(model_keys)

    fig, axes = plt.subplots(2, max(2, n_models), figsize=(5 * n_models, 10))
    fig.suptitle(
        "Cerebellar SLM — WikiText-103 (Salesforce)\n"
        "Baselines: SmolLM, SmolLM2, Qwen2.5-0.5B, Qwen3.5-0.8B",
        fontsize=12, fontweight="bold",
    )

    for col, mkey in enumerate(model_keys):
        mres = {r["condition"]: r for r in results if r["model_key"] == mkey}
        conds = [c for c in CONDITIONS if c in mres]

        # Row 0: Val vs OOD PPL
        ax = axes[0][col]
        x = np.arange(len(conds))
        w = 0.35
        val_ppls = [mres[c]["val_ppl"]  for c in conds]
        ood_ppls = [mres[c]["ood_ppl"]  for c in conds]
        ax.bar(x - w/2, val_ppls, w, label="Val PPL",  color="#3498db", alpha=0.8)
        ax.bar(x + w/2, ood_ppls, w, label="OOD PPL",  color="#e74c3c", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=10)
        ax.set_title(f"{mres[conds[0]]['model']}\nVal vs OOD PPL")
        ax.set_ylabel("Perplexity"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

        # Row 1: OOD gap
        ax = axes[1][col]
        gaps   = [mres[c]["ood_gap"] for c in conds]
        colors = [COND_COLORS.get(c, "#999") for c in conds]
        bars   = ax.bar(conds, gaps, color=colors, alpha=0.8, width=0.5)
        for bar, val in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", fontsize=9)
        ax.set_title("OOD Gap\n(lower = better generalization)")
        ax.set_ylabel("OOD PPL - Val PPL")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = out_dir / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_training_curves(result_files: list[Path], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for path in result_files:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        losses = data.get("ft_losses", [])
        if not losses:
            continue
        label = f"{data.get('model','')} {data.get('condition','')}"
        ax.plot(losses, label=label, alpha=0.8)

    ax.set_xlabel("Fine-tune step"); ax.set_ylabel("Loss")
    ax.set_title("Fine-tuning Loss (cerebellar proj weights)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    out = out_dir / "finetune_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
