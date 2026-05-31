"""
Cerebellar LLM Experiment — Full Pipeline
==========================================

Thesis: LLMs are all-cerebellum. Adding a biological cerebellar
error-correction module + microglia pruning during training produces:
  1. Better OOD generalization (PPL on held-out domains)
  2. Higher quality text generation on unfamiliar topics
  3. Visible error-correction signal during generation (Purkinje spikes)
  4. More efficient representations (microglia pruning)

Pipeline:
  Phase 1 — Train our GemmaSLM-100M on WikiText-103
             Variant A: standard training (baseline)
             Variant B: + microglia concurrent pruning during training

  Phase 2 — Attach cerebellar module to trained baseline
             Short adaptation (2K steps), base weights frozen

  Phase 3 — Quantitative eval (PPL)
             WikiText-103 test + OOD split for all 3 variants

  Phase 4 — Generation eval (THE KEY CONTRIBUTION)
             ID + OOD + stress prompts
             Side-by-side: baseline vs cerebellar
             Error recovery visualization: Purkinje correction signal

  Phase 5 — HF baseline comparison (sanity check)
             SmolLM / SmolLM2 / Qwen2.5 / Qwen3.5
             Eval PPL on same test/OOD splits only

Usage (Kaggle):
  uv run python main.py                          # full pipeline
  uv run python main.py --quick                  # smoke test (~10 min)
  uv run python main.py --skip_train             # skip training, load checkpoints
  uv run python main.py --phase gen              # generation eval only
  uv run python main.py --phase hf               # HF comparison only
"""
from __future__ import annotations

import argparse
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from config import MODEL_REGISTRY, CerebellarConfig
from model import CerebellarHFWrapper
from dataset import get_dataloaders
from microglia import MicrogliaTracker, prune_microglia
from generate import run_generation_comparison, plot_error_recovery
from visualize import plot_comparison, plot_training_curves

RESULTS_DIR = Path("results")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 100) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids    = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out    = model(ids, labels=labels)
        loss   = out.loss if hasattr(out, "loss") else out["loss"]
        n      = (labels != -100).sum().item()
        total_loss   += loss.item() * n
        total_tokens += n
    avg = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return {"loss": avg, "perplexity": math.exp(min(avg, 20))}


def load_our_model(ckpt_path: Optional[Path], cfg_name: str, cfg, quick: bool):
    """Load or init our GemmaSLM."""
    # Import here to avoid circular
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from model import GemmaCerebellarSLM
    from config import config_100m, config_100m_cerebellar, config_100m_microglia, config_quick

    cfg_map = {
        "baseline":   config_quick() if quick else config_100m(),
        "cerebellar": config_quick() if quick else config_100m_cerebellar(),
        "microglia":  config_quick() if quick else config_100m_microglia(),
    }
    model_cfg = cfg_map[cfg_name]
    model = GemmaCerebellarSLM(model_cfg).to(DEVICE)

    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  Initialized fresh model ({model.num_params()/1e6:.1f}M params)")

    return model, model_cfg


# ── Phase 1+2: Train our models ───────────────────────────────────────────────

def phase_train(args, loaders):
    from train import train as train_fn
    from config import config_100m, config_100m_microglia, config_quick

    results = []
    for variant in ["baseline", "microglia"]:
        print(f"\n{'='*60}")
        print(f"  TRAINING: GemmaSLM-100M [{variant}]")
        print(f"{'='*60}")

        out_dir = RESULTS_DIR / "our_models" / variant
        ckpt    = out_dir / f"ckpt_final.pt"
        if ckpt.exists() and not args.force_retrain:
            print(f"  Checkpoint exists, skipping. Use --force_retrain to override.")
            continue

        cfg = config_quick() if args.quick else (
            config_100m() if variant == "baseline" else config_100m_microglia()
        )
        from model import GemmaCerebellarSLM
        model = GemmaCerebellarSLM(cfg).to(DEVICE)
        print(f"  Params: {model.num_params()/1e6:.1f}M")

        r = train_fn(
            model       = model,
            loaders     = loaders,
            cfg         = cfg,
            out_dir     = out_dir,
            max_steps   = 500 if args.quick else args.steps,
            run_name    = f"gemma_{variant}",
            dtype_str   = args.dtype,
            use_wandb   = args.wandb,
        )
        # Save final checkpoint
        torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt)
        results.append(r)

    # Phase 2: attach cerebellar to trained baseline
    baseline_ckpt = RESULTS_DIR / "our_models" / "baseline" / "ckpt_final.pt"
    if baseline_ckpt.exists():
        print(f"\n{'='*60}")
        print(f"  ATTACHING CEREBELLAR MODULE to trained baseline")
        print(f"{'='*60}")

        model, _ = load_our_model(baseline_ckpt, "cerebellar", None, args.quick)

        from config import config_100m_cerebellar, config_quick
        cfg = config_quick() if args.quick else config_100m_cerebellar()
        cfg_cereb = CerebellarConfig(
            use_cerebellar  = True,
            finetune_steps  = 200 if args.quick else 2000,
            finetune_lr     = 1e-4,
            finetune_batch  = args.batch,
            seq_len         = args.seq_len,
        )

        # Short fine-tune of cerebellar proj weights
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            opt = torch.optim.AdamW(trainable, lr=cfg_cereb.finetune_lr)
            model.train()
            train_iter = iter(loaders["train"])
            from tqdm import tqdm
            for step in tqdm(range(cfg_cereb.finetune_steps), desc="  cerebellar finetune"):
                try: batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(loaders["train"])
                    batch = next(train_iter)
                ids = batch["input_ids"].to(DEVICE)
                lbl = batch["labels"].to(DEVICE)
                out = model(ids, labels=lbl)
                loss = out["loss"] if isinstance(out, dict) else out.loss
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad()

        torch.save({"model": model.state_dict()},
                   RESULTS_DIR / "our_models" / "cerebellar" / "ckpt_final.pt")

    return results


# ── Phase 3: Quantitative eval ────────────────────────────────────────────────

def phase_eval_our(args, loaders, tokenizer) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Quantitative PPL Eval — Our Models")
    print(f"{'='*60}")

    results = []
    for variant in ["baseline", "cerebellar", "microglia"]:
        ckpt = RESULTS_DIR / "our_models" / variant / "ckpt_final.pt"
        if not ckpt.exists():
            print(f"  [{variant}] No checkpoint found, skipping.")
            continue

        model, _ = load_our_model(ckpt, variant, None, args.quick)
        val_m    = evaluate(model, loaders["val"],  DEVICE)
        test_m   = evaluate(model, loaders["test"], DEVICE)
        ood_m    = evaluate(model, loaders["ood"],  DEVICE)

        r = {
            "model": f"GemmaSLM-100M", "variant": variant,
            "source": "ours",
            "val_ppl": val_m["perplexity"], "test_ppl": test_m["perplexity"],
            "ood_ppl": ood_m["perplexity"],
            "ood_gap": ood_m["perplexity"] - val_m["perplexity"],
        }
        print(f"  [{variant}] val={r['val_ppl']:.2f}  test={r['test_ppl']:.2f}  "
              f"ood={r['ood_ppl']:.2f}  gap={r['ood_gap']:.2f}")
        results.append(r)

    return results


# ── Phase 4: Generation eval ─────────────────────────────────────────────────

def phase_generation(args, tokenizer) -> dict:
    print(f"\n{'='*60}")
    print(f"  PHASE 4: Generation Eval — Baseline vs Cerebellar")
    print(f"{'='*60}")

    models_to_compare = {}
    for variant in ["baseline", "cerebellar"]:
        ckpt = RESULTS_DIR / "our_models" / variant / "ckpt_final.pt"
        if ckpt.exists():
            model, _ = load_our_model(ckpt, variant, None, args.quick)
            models_to_compare[f"gemma_{variant}"] = model

    if len(models_to_compare) < 2:
        print("  Need both baseline and cerebellar checkpoints. Skipping generation eval.")
        return {}

    gen_dir = RESULTS_DIR / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Side-by-side comparison on all prompt tiers
    results = run_generation_comparison(
        models    = models_to_compare,
        tokenizer = tokenizer,
        out_dir   = gen_dir,
        device    = str(DEVICE),
        max_new   = 100 if args.quick else 200,
        prompt_set = "all",
    )

    # Error recovery visualization (OOD prompt, cerebellar model)
    cereb_model = models_to_compare.get("gemma_cerebellar")
    if cereb_model:
        from prompts import OOD_PROMPTS
        plot_error_recovery(
            cerebellar_model = cereb_model,
            tokenizer        = tokenizer,
            prompt           = OOD_PROMPTS[0]["text"],
            out_dir          = gen_dir,
            device           = str(DEVICE),
            max_new          = 100 if args.quick else 150,
        )

    return results


# ── Phase 5: HF baseline comparison ─────────────────────────────────────────

def phase_hf_baselines(args, loaders) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFTok

    print(f"\n{'='*60}")
    print(f"  PHASE 5: HF Baseline Comparison (eval only)")
    print(f"{'='*60}")

    results = []
    for key, entry in MODEL_REGISTRY.items():
        print(f"\n  Loading {entry.short_name} ({entry.hf_id})...")
        try:
            hf_tok = HFTok.from_pretrained(entry.hf_id)
            if hf_tok.pad_token is None:
                hf_tok.pad_token = hf_tok.eos_token

            # Reload loaders with HF tokenizer for fair eval
            hf_loaders = get_dataloaders(
                tokenizer   = hf_tok,
                seq_len     = args.seq_len,
                batch_size  = args.batch,
                cache_dir   = args.cache_dir,
                num_workers = args.workers,
                quick       = True,  # HF baselines: quick eval only
            )

            model = AutoModelForCausalLM.from_pretrained(
                entry.hf_id,
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            ).to(DEVICE)

            val_m  = evaluate(model, hf_loaders["val"],  DEVICE, max_batches=50)
            test_m = evaluate(model, hf_loaders["test"], DEVICE, max_batches=50)
            ood_m  = evaluate(model, hf_loaders["ood"],  DEVICE, max_batches=50)

            r = {
                "model":    entry.short_name,
                "variant":  "pretrained",
                "source":   "huggingface",
                "params_M": entry.n_params_M,
                "val_ppl":  val_m["perplexity"],
                "test_ppl": test_m["perplexity"],
                "ood_ppl":  ood_m["perplexity"],
                "ood_gap":  ood_m["perplexity"] - val_m["perplexity"],
            }
            print(f"  [{entry.short_name}] val={r['val_ppl']:.2f}  "
                  f"test={r['test_ppl']:.2f}  ood={r['ood_ppl']:.2f}")
            results.append(r)

            del model; torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{key}] Failed: {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "train", "eval", "gen", "hf"])
    parser.add_argument("--quick",        action="store_true")
    parser.add_argument("--skip_train",   action="store_true")
    parser.add_argument("--force_retrain",action="store_true")
    parser.add_argument("--steps",        type=int,   default=50_000)
    parser.add_argument("--batch",        type=int,   default=8)
    parser.add_argument("--seq_len",      type=int,   default=512)
    parser.add_argument("--dtype",        default="bf16")
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--cache_dir",    default=None)
    parser.add_argument("--workers",      type=int,   default=2)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Load dataset + tokenizer (GPT-2 BPE for our models)
    import tiktoken
    tokenizer_gpt2 = tiktoken.get_encoding("gpt2")

    # Wrap tiktoken to match HF tokenizer interface
    class TiktokenWrapper:
        def __init__(self, enc):
            self._enc = enc
            self.eos_token_id = enc.eot_token
            self.pad_token_id = enc.eot_token
        def encode(self, text, add_special_tokens=False):
            return self._enc.encode(text)
        def decode(self, ids, skip_special_tokens=True):
            return self._enc.decode(ids)

    tokenizer = TiktokenWrapper(tokenizer_gpt2)

    print("\nLoading WikiText-103 (Salesforce)...")
    loaders = get_dataloaders(
        tokenizer   = tokenizer,
        seq_len     = args.seq_len,
        batch_size  = args.batch,
        cache_dir   = args.cache_dir,
        num_workers = args.workers,
        quick       = args.quick,
    )

    all_results = []

    # Phase 1+2: Train
    if args.phase in ("all", "train") and not args.skip_train:
        phase_train(args, loaders)

    # Phase 3: Quantitative eval
    if args.phase in ("all", "eval"):
        our_results = phase_eval_our(args, loaders, tokenizer)
        all_results.extend(our_results)

    # Phase 4: Generation eval
    if args.phase in ("all", "gen"):
        phase_generation(args, tokenizer)

    # Phase 5: HF baselines
    if args.phase in ("all", "hf"):
        hf_results = phase_hf_baselines(args, loaders)
        all_results.extend(hf_results)

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Model':25s} {'Variant':12s} {'Val PPL':>8} {'Test PPL':>9} {'OOD Gap':>8}")
        print(f"{'='*70}")
        for r in sorted(all_results, key=lambda x: x.get("val_ppl", 999)):
            print(f"{r['model']:25s} {r.get('variant',''):12s} "
                  f"{r.get('val_ppl',0):8.2f} {r.get('test_ppl',0):9.2f} "
                  f"{r.get('ood_gap',0):8.2f}")

        with open(RESULTS_DIR / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        plot_comparison(all_results, RESULTS_DIR)

    print(f"\nDone. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
