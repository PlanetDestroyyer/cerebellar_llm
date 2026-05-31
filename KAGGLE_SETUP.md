## Kaggle Setup

### 1. Install uv
```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source $HOME/.local/bin/env
```

### 2. Smoke test (~10 min)
```bash
!uv run python main.py --quick
```

### 3. Full experiment (~3-6 hrs on A100)
```bash
# Phase 1+2: Train GemmaSLM baseline + microglia, attach cerebellar
!uv run python main.py --phase train --steps 50000 --batch 16

# Phase 3: PPL eval on WikiText-103 test + OOD
!uv run python main.py --phase eval

# Phase 4: Generation eval (ID/OOD/stress prompts + error recovery viz)
!uv run python main.py --phase gen

# Phase 5: HF baseline comparison (SmolLM/Qwen eval only)
!uv run python main.py --phase hf
```

### 4. Standard benchmarks (MMLU / HellaSwag / ARC / TruthfulQA)
```bash
# Our trained models
!uv run python benchmark.py --benchmarks mmlu hellaswag arc truthfulqa

# Quick test (100 samples each)
!uv run python benchmark.py --quick

# Also benchmark HF baselines
!uv run python benchmark.py \
  --models HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B Qwen/Qwen3.5-0.8B \
  --benchmarks mmlu hellaswag arc truthfulqa
```

### Results in results/
- our_models/{baseline,cerebellar,microglia}/   training checkpoints + PPL
- generation/   side-by-side text + error_recovery.png + generation_quality.png
- benchmarks/   MMLU/HellaSwag/ARC scores + benchmark_results.png
- all_results.json, comparison.png

### Published reference scores (for paper table)
| Model          | MMLU  | HellaSwag | ARC   |
|----------------|-------|-----------|-------|
| SmolLM-135M    | 26.5  | 44.4      | 33.9  |
| SmolLM2-135M   | 27.2  | 45.1      | 35.1  |
| Qwen2.5-0.5B   | 47.4  | 52.8      | 36.1  |
| Qwen3.5-0.8B   | 55.2  | 58.0      | 43.0  |
