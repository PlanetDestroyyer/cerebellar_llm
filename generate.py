"""
Generation evaluation for the cerebellar LLM experiment.

Three analyses:

1. Side-by-side generation
   For each prompt: generate 200 tokens with baseline vs cerebellar model.
   Save text side by side for human + automated comparison.

2. Error recovery visualization
   Track the cerebellar correction signal magnitude (L2 norm of correction)
   at each token position during generation.
   Spike in correction = model was about to drift, cerebellar intervened.
   Correlate spikes with semantically weak/incoherent tokens.

3. Coherence scoring
   Use a simple n-gram repetition + sentence completion heuristic as
   automated proxy for generation quality (no external model needed).
   Real evaluation: human reading.
"""
from __future__ import annotations

import json
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import Optional

from prompts import ALL_PROMPTS, ID_PROMPTS, OOD_PROMPTS, STRESS_PROMPTS


# ── Coherence heuristics ──────────────────────────────────────────────────────

def repetition_score(tokens: list[int], window: int = 20) -> float:
    """Fraction of tokens that are repeats within a rolling window. Lower = better."""
    if len(tokens) < window:
        return 0.0
    repeats = 0
    for i in range(window, len(tokens)):
        if tokens[i] in tokens[max(0, i-window):i]:
            repeats += 1
    return repeats / (len(tokens) - window)


def distinct_n(tokens: list[int], n: int = 2) -> float:
    """Distinct n-gram ratio. Higher = more diverse/coherent."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ngrams)) / len(ngrams)


# ── Generation with correction tracking ──────────────────────────────────────

class CerebellarGenerationTracker:
    """
    Hooks into cerebellar modules during generation to track
    correction signal magnitude at each token position.
    """
    def __init__(self, model):
        self.corrections: list[float] = []
        self._hooks = []
        self._current_step_corrections: list[float] = []
        self._register(model)

    def _register(self, model):
        # Import here to avoid circular
        from model import CerebellarHFWrapper
        if not isinstance(model, CerebellarHFWrapper):
            return
        for name, mod in model.cerebellar_modules.items():
            def make_hook(n):
                def hook(m, inp, out):
                    # correction = out - inp[0]  (magnitude = how much was changed)
                    if isinstance(inp, tuple) and len(inp) > 0:
                        corr_magnitude = (out - inp[0]).norm(dim=-1).mean().item()
                        self._current_step_corrections.append(corr_magnitude)
                return hook
            self._hooks.append(mod.proj.register_forward_hook(make_hook(name)))

    def step_done(self):
        """Call after each token generation step."""
        if self._current_step_corrections:
            self.corrections.append(np.mean(self._current_step_corrections))
        else:
            self.corrections.append(0.0)
        self._current_step_corrections = []

    def remove(self):
        for h in self._hooks:
            h.remove()


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt:       str,
    max_new:      int = 200,
    temperature:  float = 0.8,
    top_p:        float = 0.9,
    device:       str = "cuda",
    track_corrections: bool = False,
) -> dict:
    model.eval()
    ids    = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen    = ids.clone()

    tracker = CerebellarGenerationTracker(model) if track_corrections else None
    generated_tokens = []

    for _ in range(max_new):
        out    = model(gen[:, -512:])  # context window limit
        logits = out.logits[:, -1, :] if hasattr(out, "logits") else out["logits"][:, -1, :]

        # Temperature + top-p sampling
        if temperature != 1.0:
            logits = logits / temperature
        probs  = F.softmax(logits, dim=-1)

        # Top-p nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask   = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)].unsqueeze(0).unsqueeze(0)

        gen = torch.cat([gen, next_id], dim=1)
        generated_tokens.append(next_id.item())

        if tracker:
            tracker.step_done()

        if next_id.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    gen_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if tracker:
        tracker.remove()

    return {
        "full_text":        text,
        "generated_only":   gen_only,
        "tokens":           generated_tokens,
        "repetition_score": repetition_score(generated_tokens),
        "distinct_2":       distinct_n(generated_tokens, 2),
        "distinct_3":       distinct_n(generated_tokens, 3),
        "corrections":      tracker.corrections if tracker else [],
    }


# ── Side-by-side comparison ───────────────────────────────────────────────────

def run_generation_comparison(
    models:    dict,     # {"baseline": model, "cerebellar": model, ...}
    tokenizer,
    out_dir:   Path,
    device:    str = "cuda",
    max_new:   int = 200,
    prompt_set: str = "all",  # "id", "ood", "stress", "all"
) -> dict:
    """Generate from all prompts with all models and save comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = {"id": ID_PROMPTS, "ood": OOD_PROMPTS,
                  "stress": STRESS_PROMPTS, "all": ALL_PROMPTS}
    prompts = prompt_map.get(prompt_set, ALL_PROMPTS)

    all_results = {}

    for prompt_info in prompts:
        pid    = prompt_info["id"]
        prompt = prompt_info["text"]
        domain = prompt_info["domain"]
        print(f"\n  Prompt [{pid}] ({domain})")
        print(f"  '{prompt[:60]}...'")

        results = {}
        for model_name, model in models.items():
            track = "cerebellar" in model_name
            out = generate_text(model, tokenizer, prompt,
                                max_new=max_new, device=device,
                                track_corrections=track)
            results[model_name] = out
            rep = out["repetition_score"]
            d2  = out["distinct_2"]
            print(f"    [{model_name:15s}] rep={rep:.3f}  dist2={d2:.3f}")
            print(f"      {out['generated_only'][:120]}...")

        all_results[pid] = {
            "prompt_info": prompt_info,
            "results":     {k: {
                "generated_only":   v["generated_only"],
                "repetition_score": v["repetition_score"],
                "distinct_2":       v["distinct_2"],
                "distinct_3":       v["distinct_3"],
            } for k, v in results.items()},
        }

    # Save text comparison
    with open(out_dir / "generation_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot coherence metrics
    _plot_coherence(all_results, models.keys(), out_dir)

    return all_results


# ── Error recovery visualization ──────────────────────────────────────────────

def plot_error_recovery(
    cerebellar_model,
    tokenizer,
    prompt:   str,
    out_dir:  Path,
    device:   str = "cuda",
    max_new:  int = 150,
):
    """
    Visualize cerebellar correction signal over token positions.
    Spikes = model was drifting, cerebellar intervened.
    """
    out = generate_text(cerebellar_model, tokenizer, prompt,
                        max_new=max_new, device=device,
                        track_corrections=True)

    corrections = out["corrections"]
    tokens      = out["tokens"]
    token_strs  = [tokenizer.decode([t]) for t in tokens]

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Panel 1: Correction magnitude over tokens
    ax1 = fig.add_subplot(gs[0])
    x   = np.arange(len(corrections))
    ax1.fill_between(x, corrections, alpha=0.4, color="#e74c3c")
    ax1.plot(x, corrections, color="#e74c3c", linewidth=1.5)

    # Annotate top-5 correction spikes with token text
    if corrections:
        top5 = sorted(range(len(corrections)), key=lambda i: corrections[i], reverse=True)[:5]
        for idx in top5:
            if idx < len(token_strs):
                ax1.annotate(
                    repr(token_strs[idx]),
                    xy=(idx, corrections[idx]),
                    xytext=(idx, corrections[idx] * 1.15),
                    fontsize=7, ha="center", color="#c0392b",
                    arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
                )

    ax1.set_ylabel("Cerebellar correction magnitude (L2)")
    ax1.set_title(
        "Cerebellar Error Correction Signal During Generation\n"
        f"Prompt: '{prompt[:70]}...'\n"
        "Spikes = Purkinje cells intervening to correct hidden state drift"
    )
    ax1.axhline(np.mean(corrections) if corrections else 0,
                color="gray", linestyle="--", linewidth=1, label="mean")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Token text (abbreviated)
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    # Show first 50 generated tokens
    display_tokens = token_strs[:50]
    token_display  = " ".join(
        f"[{t.strip()}]" if i in (top5 if corrections else []) else t.strip()
        for i, t in enumerate(display_tokens)
    )
    ax2.text(0.01, 0.7, "Generated: " + token_display[:200] + "...",
             transform=ax2.transAxes, fontsize=8, wrap=True,
             verticalalignment="top", family="monospace")

    plt.tight_layout()
    out_path = out_dir / "error_recovery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return out


# ── Coherence plot ────────────────────────────────────────────────────────────

def _plot_coherence(all_results: dict, model_names, out_dir: Path):
    prompts = list(all_results.keys())
    metrics = ["repetition_score", "distinct_2"]
    colors  = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Generation Quality — Baseline vs Cerebellar\n"
                 "Repetition (lower=better)  |  Distinct-2 (higher=better)",
                 fontsize=11)

    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i]
        x  = np.arange(len(prompts))
        w  = 0.8 / len(model_names)

        for j, mname in enumerate(model_names):
            vals = []
            for pid in prompts:
                r = all_results[pid]["results"]
                vals.append(r.get(mname, {}).get(metric, 0.0))
            offset = (j - len(model_names)/2 + 0.5) * w
            ax.bar(x + offset, vals, w * 0.9,
                   label=mname, color=colors[j % len(colors)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in prompts], fontsize=7)
        label = "Repetition Score (lower=better)" if metric == "repetition_score" \
                else "Distinct-2 (higher=better)"
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = out_dir / "generation_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
