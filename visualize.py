"""Plots for GemmaCerebellar SLM comparison."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_comparison(results: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GemmaCerebellar SLM — WikiText-103 Comparison\n"
                 "(100M params, RTX 4050, SOTA 2025-2026 architecture)",
                 fontsize=11, fontweight="bold")

    names  = [r["run_name"] for r in results]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"][:len(results)]

    # Panel 1: Val vs OOD perplexity
    ax = axes[0]
    x  = np.arange(len(names))
    w  = 0.35
    val_ppls = [r["val_ppl"]  for r in results]
    ood_ppls = [r["ood_ppl"]  for r in results]
    ax.bar(x - w/2, val_ppls, w, label="Val PPL",  color="#3498db", alpha=0.8)
    ax.bar(x + w/2, ood_ppls, w, label="OOD PPL",  color="#e74c3c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Validation vs OOD Perplexity")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: OOD gap (proxy for generalization)
    ax = axes[1]
    gaps = [r["ood_gap"] for r in results]
    bars = ax.bar(names, gaps, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("OOD PPL − Val PPL (lower = better generalization)")
    ax.set_title("OOD Generalization Gap\n(cerebellar hypothesis: reduces gap)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Test perplexity
    ax = axes[2]
    test_ppls = [r["test_ppl"] for r in results]
    bars = ax.bar(names, test_ppls, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, test_ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Test Perplexity")
    ax.set_title("WikiText-103 Test Perplexity\n(standard benchmark)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = out_dir / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_training_curves(history_files: list[Path], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, path in enumerate(history_files):
        with open(path) as f:
            data = json.load(f)
        hist = data.get("history", {})
        name = data.get("run_name", path.stem)
        c    = colors[i % len(colors)]

        if "step" in hist and "train_loss" in hist:
            axes[0].plot(hist["step"], hist["train_loss"], color=c, label=name, alpha=0.8)
        if "val_ppl" in hist:
            n = len(hist["val_ppl"])
            steps = list(range(0, n * 1000, 1000))[:n]
            axes[1].plot(steps, hist["val_ppl"], color=c, label=name, linewidth=2)

    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title("Validation PPL"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
