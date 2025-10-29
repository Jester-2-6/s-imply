# s-imply

Back implication prediction using attention for netlists.

## Reconvergent Path Justification System

A hybrid RL + supervised learning system for learning to justify reconvergent path structures in digital circuits.

### Quick Start

```bash
# Activate environment
conda activate torch

# Build dataset + train model
python -m src.ml.train_reconv both \
  --bench-dir data/bench/ISCAS85 \
  --dataset data/datasets/reconv_dataset.pkl \
  --checkpoint-dir checkpoints/reconv_rl \
  --epochs 50 \
  --amp \
  --verbose \
  --include-hard-negatives

# Evaluate trained model
python -m src.ml.evaluate_reconv \
  --checkpoint checkpoints/reconv_rl/best_model.pth \
  --dataset data/datasets/reconv_dataset.pkl \
  --verbose
```

### Documentation

See **[GUIDE.md](GUIDE.md)** for complete training and evaluation documentation.

### Key Components

- **Dataset Builder**: `src/atpg/reconv_podem.py` - Finds and justifies reconvergent structures
- **Model**: `src/ml/reconv_lib.py` - Multi-path transformer with attention
- **Trainer**: `src/ml/reconv_rl_trainer.py` - Hybrid supervised + RL training
- **CLI Tools**: `src/ml/train_reconv.py`, `src/ml/evaluate_reconv.py`

## Minimal Trainer (Supervised Only)

For a lightweight baseline without RL or auto-batch features, use the minimal trainer:

```zsh
conda activate torch

# Train
python -m src.ml.train_reconv train \
  --dataset data/datasets/reconv_dataset.pkl \
  --output checkpoints/reconv_minimal \
  --epochs 5 \
  --batch-size 8 \
  --embedding-dim 128

# Evaluate
python -m src.ml.evaluate_reconv \
  --checkpoint checkpoints/reconv_minimal/best_model.pth \
  --dataset data/datasets/reconv_dataset.pkl \
  --embedding-dim 128
```

Notes:
- Defaults to embedding_dim=128 to match the dummy embedding path (when DeepGate is not available).
- Expects dataset samples with `info.paths` and optional `justification_1/0` dicts.
