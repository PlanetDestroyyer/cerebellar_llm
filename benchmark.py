"""
Standard NLP Benchmarks for SLM evaluation.

Benchmarks:
  MMLU          57-subject knowledge test (5-shot, MCQ)
  HellaSwag     Commonsense sentence completion (0-shot, log-likelihood)
  ARC-Challenge Science reasoning (0-shot, MCQ log-likelihood)
  WinoGrande    Winograd schemas / pronoun resolution (0-shot)
  TruthfulQA    Hallucination / truthfulness (0-shot MCQ)
  BoolQ         Reading comprehension boolean (0-shot)

Method: log-likelihood ranking (standard for MCQ benchmarks).
  For each question, compute log P(choice) for all answer choices.
  Predict the choice with highest log P. No generation needed.
  This is identical to how lm-evaluation-harness evaluates MCQ tasks.

Works for:
  - Our custom GemmaSLM (via model forward pass)
  - Any HuggingFace CausalLM

Usage:
  uv run python benchmark.py                    # all benchmarks, all models
  uv run python benchmark.py --models smollm    # single HF model
  uv run python benchmark.py --our_ckpt results/our_models/baseline/ckpt_final.pt
  uv run python benchmark.py --quick            # 100 samples per benchmark
"""
from __future__ import annotations

import argparse
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Log-likelihood scorer ────────────────────────────────────────────────────

@torch.no_grad()
def log_likelihood(model, input_ids: torch.Tensor,
                   target_ids: torch.Tensor) -> float:
    """
    Compute sum of log P(target_ids | input_ids) using teacher forcing.
    Standard method for MCQ evaluation.
    """
    ids = torch.cat([input_ids, target_ids], dim=-1).unsqueeze(0).to(DEVICE)
    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    logits = logits[0]  # [T, V]

    # We want log P of target tokens given context
    ctx_len = input_ids.shape[-1]
    tgt_len = target_ids.shape[-1]

    # Shift: logits[ctx_len-1 : ctx_len+tgt_len-1] predict target_ids
    tgt_logits = logits[ctx_len - 1: ctx_len + tgt_len - 1]
    log_probs  = F.log_softmax(tgt_logits, dim=-1)

    ll = 0.0
    for i, tok in enumerate(target_ids):
        ll += log_probs[i, tok.item()].item()
    return ll


@torch.no_grad()
def score_choices(model, tokenizer, context: str,
                  choices: list[str]) -> int:
    """
    Score each answer choice by log-likelihood given context.
    Return index of best choice.
    """
    ctx_ids = torch.tensor(tokenizer.encode(context), dtype=torch.long)
    scores  = []
    for choice in choices:
        tgt_ids = torch.tensor(tokenizer.encode(" " + choice.strip()),
                               dtype=torch.long)
        if len(tgt_ids) == 0:
            scores.append(float("-inf"))
            continue
        ll = log_likelihood(model, ctx_ids, tgt_ids)
        # Normalize by length (length-normalized log-likelihood)
        scores.append(ll / len(tgt_ids))
    return int(np.argmax(scores))


# ── MMLU ─────────────────────────────────────────────────────────────────────

def eval_mmlu(model, tokenizer, n_samples: Optional[int] = None,
              n_shot: int = 5, cache_dir: Optional[str] = None) -> dict:
    """
    MMLU: 57 subjects, 5-shot evaluation.
    HF dataset: cais/mmlu, all subjects.
    Method: 5-shot prompt + log-likelihood ranking over A/B/C/D.
    """
    from datasets import load_dataset

    CHOICES = ["A", "B", "C", "D"]

    # Load all subjects
    all_correct, all_total = 0, 0
    subject_results = {}

    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]

    if n_samples:
        subjects = subjects[:max(1, n_samples // 10)]

    for subject in tqdm(subjects, desc="MMLU subjects"):
        try:
            ds_test = load_dataset("cais/mmlu", subject, split="test",
                                   cache_dir=cache_dir)
            ds_dev  = load_dataset("cais/mmlu", subject, split="dev",
                                   cache_dir=cache_dir)
        except Exception:
            continue

        # Build 5-shot examples from dev
        few_shot = ""
        for ex in list(ds_dev)[:n_shot]:
            q = ex["question"]
            opts = ex["choices"]
            ans  = CHOICES[ex["answer"]]
            few_shot += (f"Question: {q}\n"
                         f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
                         f"Answer: {ans}\n\n")

        correct, total = 0, 0
        items = list(ds_test)
        if n_samples:
            items = items[:max(1, n_samples // len(subjects))]

        for ex in items:
            q    = ex["question"]
            opts = ex["choices"]
            ans  = ex["answer"]
            ctx  = (few_shot +
                    f"Question: {q}\n"
                    f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
                    f"Answer:")
            pred = score_choices(model, tokenizer, ctx, CHOICES)
            if pred == ans:
                correct += 1
            total += 1

        subject_results[subject] = {"accuracy": correct / total if total else 0,
                                     "n": total}
        all_correct += correct
        all_total   += total

    overall = all_correct / all_total if all_total > 0 else 0.0
    return {"accuracy": overall, "n": all_total, "subjects": subject_results}


# ── HellaSwag ─────────────────────────────────────────────────────────────────

def eval_hellaswag(model, tokenizer, n_samples: Optional[int] = None,
                   cache_dir: Optional[str] = None) -> dict:
    """
    HellaSwag: sentence completion, 0-shot log-likelihood ranking.
    Pick the most likely completion from 4 choices.
    """
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir=cache_dir)
    items = list(ds)[:n_samples] if n_samples else list(ds)

    correct, total = 0, 0
    for ex in tqdm(items, desc="HellaSwag"):
        ctx     = ex["activity_label"] + ": " + ex["ctx"]
        endings = ex["endings"]
        label   = int(ex["label"])
        pred    = score_choices(model, tokenizer, ctx, endings)
        if pred == label:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total else 0, "n": total}


# ── ARC-Challenge ─────────────────────────────────────────────────────────────

def eval_arc(model, tokenizer, n_samples: Optional[int] = None,
             cache_dir: Optional[str] = None) -> dict:
    """
    ARC-Challenge: science reasoning MCQ, 0-shot log-likelihood.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge",
                      split="test", cache_dir=cache_dir)
    items = list(ds)[:n_samples] if n_samples else list(ds)

    correct, total = 0, 0
    for ex in tqdm(items, desc="ARC-Challenge"):
        q       = ex["question"]
        choices = ex["choices"]["text"]
        labels  = ex["choices"]["label"]
        ans_key = ex["answerKey"]

        ctx  = f"Question: {q}\nAnswer:"
        pred = score_choices(model, tokenizer, ctx, choices)

        # Map predicted index to label
        pred_label = labels[pred] if pred < len(labels) else "X"
        if pred_label == ans_key:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total else 0, "n": total}


# ── WinoGrande ────────────────────────────────────────────────────────────────

def eval_winogrande(model, tokenizer, n_samples: Optional[int] = None,
                    cache_dir: Optional[str] = None) -> dict:
    """
    WinoGrande: fill-in-the-blank pronoun resolution, 0-shot.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl",
                      split="validation", cache_dir=cache_dir,
                      trust_remote_code=True)
    items = list(ds)[:n_samples] if n_samples else list(ds)

    correct, total = 0, 0
    for ex in tqdm(items, desc="WinoGrande"):
        sentence = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        answer = int(ex["answer"])  # 1 or 2

        # Create two versions: fill blank with option1 vs option2
        s1 = sentence.replace("_", opt1)
        s2 = sentence.replace("_", opt2)

        # Score each filled sentence
        ctx_ids  = torch.tensor([], dtype=torch.long)
        ids1 = torch.tensor(tokenizer.encode(s1), dtype=torch.long)
        ids2 = torch.tensor(tokenizer.encode(s2), dtype=torch.long)

        # Log-likelihood of full sentence
        dummy = torch.tensor(tokenizer.encode("The"), dtype=torch.long)

        def sent_ll(ids):
            if len(ids) < 2:
                return 0.0
            return log_likelihood(model, ids[:-1], ids[1:])

        score1 = sent_ll(ids1) / len(ids1)
        score2 = sent_ll(ids2) / len(ids2)

        pred = 1 if score1 > score2 else 2
        if pred == answer:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total else 0, "n": total}


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def eval_truthfulqa(model, tokenizer, n_samples: Optional[int] = None,
                    cache_dir: Optional[str] = None) -> dict:
    """
    TruthfulQA MC1: single correct answer from ~4-7 choices.
    0-shot log-likelihood ranking.
    """
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation", cache_dir=cache_dir)
    items = list(ds)[:n_samples] if n_samples else list(ds)

    correct, total = 0, 0
    for ex in tqdm(items, desc="TruthfulQA"):
        q       = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels  = ex["mc1_targets"]["labels"]  # 1 = correct, 0 = wrong

        ctx  = f"Q: {q}\nA:"
        pred = score_choices(model, tokenizer, ctx, choices)

        if labels[pred] == 1:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total else 0, "n": total}


# ── BoolQ ────────────────────────────────────────────────────────────────────

def eval_boolq(model, tokenizer, n_samples: Optional[int] = None,
               cache_dir: Optional[str] = None) -> dict:
    """
    BoolQ: yes/no question answering given a passage. 0-shot.
    """
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation", cache_dir=cache_dir)
    items = list(ds)[:n_samples] if n_samples else list(ds)

    correct, total = 0, 0
    for ex in tqdm(items, desc="BoolQ"):
        passage  = ex["passage"][:512]  # truncate long passages
        question = ex["question"]
        answer   = ex["answer"]  # True / False

        ctx  = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        pred = score_choices(model, tokenizer, ctx, ["yes", "no"])
        pred_bool = (pred == 0)  # 0 = "yes" = True

        if pred_bool == answer:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total else 0, "n": total}


# ── Run all benchmarks ────────────────────────────────────────────────────────

BENCHMARK_FNS = {
    "mmlu":        eval_mmlu,
    "hellaswag":   eval_hellaswag,
    "arc":         eval_arc,
    "winogrande":  eval_winogrande,
    "truthfulqa":  eval_truthfulqa,
    "boolq":       eval_boolq,
}

# Published reference scores for comparison in paper
PUBLISHED_SCORES = {
    "SmolLM-135M":  {"mmlu": 26.5, "hellaswag": 44.4, "arc": 33.9},
    "SmolLM2-135M": {"mmlu": 27.2, "hellaswag": 45.1, "arc": 35.1},
    "Qwen2.5-0.5B": {"mmlu": 47.4, "hellaswag": 52.8, "arc": 36.1},
    "Qwen3.5-0.8B": {"mmlu": 55.2, "hellaswag": 58.0, "arc": 43.0},
}


def run_benchmarks(model, tokenizer, model_name: str,
                   benchmarks: list[str], n_samples: Optional[int],
                   cache_dir: Optional[str], out_dir: Path) -> dict:
    model.eval()
    results = {"model": model_name}

    for bname in benchmarks:
        if bname not in BENCHMARK_FNS:
            print(f"  Unknown benchmark: {bname}")
            continue
        print(f"\n  Running {bname}...")
        try:
            r = BENCHMARK_FNS[bname](model, tokenizer,
                                     n_samples=n_samples,
                                     cache_dir=cache_dir)
            results[bname] = r
            print(f"  {bname}: {r['accuracy']*100:.1f}% (n={r['n']})")
        except Exception as e:
            print(f"  {bname} failed: {e}")
            results[bname] = {"accuracy": 0.0, "error": str(e)}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"benchmarks_{model_name.replace('/', '_')}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_benchmark_results(all_results: list[dict], out_dir: Path,
                            benchmarks: list[str]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, len(benchmarks),
                              figsize=(4 * len(benchmarks), 6))
    if len(benchmarks) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results) + len(PUBLISHED_SCORES)))

    for ax_i, bname in enumerate(benchmarks):
        ax = axes[ax_i]
        names, scores = [], []

        # Our models
        for i, r in enumerate(all_results):
            acc = r.get(bname, {}).get("accuracy", 0) * 100
            names.append(r["model"])
            scores.append(acc)

        # Published reference scores
        for mname, pub in PUBLISHED_SCORES.items():
            if bname in pub:
                names.append(f"{mname}*")
                scores.append(pub[bname])

        bar_colors = [colors[i] for i in range(len(names))]
        bars = ax.bar(range(len(names)), scores, color=bar_colors, alpha=0.85)
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=8)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(bname.upper())
        ax.set_ylim(0, 100)
        ax.axhline(25, color="gray", linestyle="--", linewidth=0.8,
                   label="random (4-choice)")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Benchmark Results — Our Models vs Published SLMs\n"
                 "* = published reference scores", fontsize=11)
    plt.tight_layout()
    out = out_dir / "benchmark_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--our_ckpt", default=None,
                        help="Path to our trained GemmaSLM checkpoint")
    parser.add_argument("--models",   nargs="+", default=[],
                        help="HF model IDs to also benchmark")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["mmlu", "hellaswag", "arc", "truthfulqa"],
                        choices=list(BENCHMARK_FNS.keys()))
    parser.add_argument("--quick",    action="store_true",
                        help="100 samples per benchmark (fast)")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    n_samples = 100 if args.quick else None
    out_dir   = RESULTS_DIR / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Our trained models
    our_variants = ["baseline", "cerebellar", "microglia"]
    for variant in our_variants:
        ckpt = RESULTS_DIR / "our_models" / variant / "ckpt_final.pt"
        if args.our_ckpt:
            ckpt = Path(args.our_ckpt)
        if not ckpt.exists():
            continue

        print(f"\n{'='*50}")
        print(f"  Benchmarking: GemmaSLM-100M [{variant}]")

        from model import GemmaCerebellarSLM
        from config import config_100m, config_100m_cerebellar, config_100m_microglia

        cfg_map = {"baseline": config_100m(),
                   "cerebellar": config_100m_cerebellar(),
                   "microglia": config_100m_microglia()}
        cfg   = cfg_map.get(variant, config_100m())
        model = GemmaCerebellarSLM(cfg).to(DEVICE)
        ckpt_data = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt_data["model"])

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        class TikWrap:
            def __init__(self, e): self._e = e
            def encode(self, t, **kw): return self._e.encode(t)
        tokenizer = TikWrap(enc)

        r = run_benchmarks(model, tokenizer, f"GemmaSLM_{variant}",
                           args.benchmarks, n_samples, args.cache_dir, out_dir)
        all_results.append(r)

    # HF models
    for hf_id in args.models:
        print(f"\n{'='*50}")
        print(f"  Benchmarking: {hf_id}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok   = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        ).to(DEVICE)
        r = run_benchmarks(model, tok, hf_id.split("/")[-1],
                           args.benchmarks, n_samples, args.cache_dir, out_dir)
        all_results.append(r)

    # Summary table
    if all_results:
        print(f"\n{'='*65}")
        header = f"{'Model':30s}" + "".join(f"{b.upper():>12s}" for b in args.benchmarks)
        print(header)
        print("="*65)
        for r in all_results:
            row = f"{r['model']:30s}"
            for b in args.benchmarks:
                acc = r.get(b, {}).get("accuracy", 0) * 100
                row += f"{acc:12.1f}"
            print(row)

        # Published reference
        print("\n  * Published reference scores:")
        for mname, scores in PUBLISHED_SCORES.items():
            row = f"  {mname:28s}"
            for b in args.benchmarks:
                row += f"{scores.get(b, 0):12.1f}"
            print(row)

        with open(out_dir / "all_benchmarks.json", "w") as f:
            json.dump(all_results, f, indent=2)

        plot_benchmark_results(all_results, out_dir, args.benchmarks)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
