## Kaggle Setup (uv everywhere)

### 1. Install uv
```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source $HOME/.cargo/env
```

### 2. Quick smoke test (verify code works, ~5 min)
```bash
!uv run python main.py --quick
```

### 3. Full experiment (WikiText-103, ~3-6 hrs on A100)
```bash
!uv run python main.py --steps 50000 --batch 16 --seq_len 1024 --grad_accum 4
```

### 4. Single runs
```bash
!uv run python main.py --run baseline   --steps 50000 --batch 16
!uv run python main.py --run cerebellar --steps 50000 --batch 16
!uv run python main.py --run microglia  --steps 50000 --batch 16
```

### 5. With WandB logging
```bash
!uv run python main.py --steps 50000 --batch 16 --wandb
```

### Expected results (A100, ~4 hrs):
- Baseline val PPL: ~25-35 (WikiText-103 standard)
- Cerebellar OOD gap: hypothesis — lower than baseline
- Microglia OOD gap: hypothesis — lower than magnitude pruning

### Files produced in results/:
- all_results.json
- comparison.png
- training_curves.png
- baseline/results_baseline.json
- cerebellar/results_cerebellar.json
- microglia/results_microglia.json
