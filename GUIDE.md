# Reconvergent Path Justification - Training & Evaluation Guide

Complete guide for building datasets, training models, and evaluating performance for the reconvergent path justification system.

## Quick Start

```zsh
# Activate environment
conda activate torch

# Build dataset + train in one command
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

---

## Dataset Building

### Basic Build

```zsh
python -m src.ml.train_reconv build \
  --bench-dir data/bench/ISCAS85 \
  --output data/datasets/reconv_dataset.pkl \
  --max-samples 1000 \
  --include-hard-negatives
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--bench-dir` | `data/bench/ISCAS85` | Directory with `.bench` files |
| `--output` | `data/datasets/reconv_dataset.pkl` | Output dataset path |
| `--max-samples` | `1000` | Max samples per file |
| `--include-hard-negatives` | off | Include unjustifiable paths as hard negatives |

### What It Does

1. **Finds reconvergent structures**: Exhaustively enumerates all reconvergent path pairs in each circuit
2. **Attempts justification**: Tries to justify both 0 and 1 values at the reconvergent node
3. **Creates samples**:
   - **Both-justifiable**: Positive samples where both values can be justified
   - **Hard negatives** (if `--include-hard-negatives`): Samples where one or both justifications fail
4. **Shuffles**: Randomizes sample order for better training dynamics
5. **Saves**: Pickles to specified output path

### Output

```
Dataset statistics:
  Total samples: 1523
  Samples per file:
    c17.bench: 45
    c432.bench: 312
    ...

Dataset composition (by justifiability):
  Both-justifiable: 1234
  Only-1-justifiable: 123
  Only-0-justifiable: 104
  None-justifiable: 62

Shuffling dataset...
Dataset saved to data/datasets/reconv_dataset.pkl (1523 entries)
```

---

## Model Training

### Basic Training

```zsh
python -m src.ml.train_reconv train \
  --dataset data/datasets/reconv_dataset.pkl \
  --checkpoint-dir checkpoints/reconv_rl \
  --epochs 50 \
  --amp \
  --verbose
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `data/datasets/reconv_dataset.pkl` | Path to dataset |
| `--checkpoint-dir` | `checkpoints/reconv_rl` | Checkpoint directory |
| `--epochs` | `50` | Number of epochs |
| `--batch-size` | `8192` | Initial batch size (auto-scaled) |
| `--lr` | `1e-4` | Learning rate |
| `--supervised-weight` | `0.7` | Supervised loss weight |
| `--rl-weight` | `0.3` | RL loss weight |
| `--embedding-dim` | `512` | Embedding dimension |
| `--nhead` | `8` | Attention heads |
| `--num-encoder-layers` | `8` | Encoder layers |
| `--num-interaction-layers` | `4` | Interaction layers |
| `--amp` | off | Enable mixed precision training |
| `--verbose` | off | Detailed training logs |

### Training Process

1. **Auto-batch sizing**: Probes GPU memory to find max safe batch size
2. **Hybrid training**: Combines supervised learning (on known justifications) with RL (for constraint satisfaction)
3. **Per-epoch metrics**: Loss, accuracy, and rewards
4. **Checkpointing**: Saves every 10 epochs + best model by reward
5. **OOM recovery**: Auto-halves batch size and retries on CUDA OOM

### Training Output

```
[TRAIN] dataset_len=1523, effective_batch_size=4096, total_batches=1

Epoch progress [##################################################] Batch 1/1

Metrics:
  supervised_loss: 0.2345
  accuracy: 0.8542
  accuracy_both: 0.9123
  accuracy_only1: 0.6234
  policy_loss: 0.1234
  avg_reward: 2.4567

Checkpoint saved to checkpoints/reconv_rl/checkpoint_epoch_10.pth
New best model saved! Reward: 2.4567
```

---

## Model Evaluation

### Basic Evaluation

```zsh
python -m src.ml.evaluate_reconv \
  --checkpoint checkpoints/reconv_rl/best_model.pth \
  --dataset data/datasets/reconv_dataset.pkl \
  --verbose
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Path to `.pth` checkpoint |
| `--dataset` | `data/datasets/reconv_dataset.pkl` | Evaluation dataset |
| `--batch-size` | `16` | Batch size |
| `--embedding-dim` | `512` | Must match training |
| `--nhead` | `8` | Must match training |
| `--num-encoder-layers` | `8` | Must match training |
| `--num-interaction-layers` | `4` | Must match training |
| `--verbose` | off | Per-batch logs |
| `--cpu` | off | Force CPU (default: CUDA if available) |

### Evaluation Output

```
EVALUATION RESULTS
============================================================
Average Loss: 0.234567
Overall Accuracy: 0.8542 (85.42%)

Per-Category Accuracy:
  both      : 0.9123 (91.23%) on 12845 positions
  only1     : 0.6234 (62.34%) on 1203 positions
  only0     : 0.5987 (59.87%) on 1104 positions
  none      : 0.4521 (45.21%) on 456 positions
============================================================
```

### Understanding Metrics

- **Average Loss**: Cross-entropy loss averaged across batches
- **Overall Accuracy**: % of gate values correctly predicted
- **Per-Category Accuracy**:
  - `both`: Both 0 and 1 justifiable (positive samples)
  - `only1`/`only0`: Hard negatives (one value unjustifiable)
  - `none`: Hard negatives (neither justifiable)
- **Positions**: Each gate in each path is a "position" with a target value

---

## Advanced Workflows

### Compare Checkpoints

```zsh
for ckpt in checkpoints/reconv_rl/checkpoint_epoch_*.pth; do
  echo "=== $ckpt ==="
  python -m src.ml.evaluate_reconv --checkpoint "$ckpt"
done
```

### Train on Different Circuits

```zsh
# Build from ISCAS89 instead
python -m src.ml.train_reconv build \
  --bench-dir data/bench/iscas89 \
  --output data/datasets/iscas89_dataset.pkl \
  --include-hard-negatives

# Train on new dataset
python -m src.ml.train_reconv train \
  --dataset data/datasets/iscas89_dataset.pkl \
  --checkpoint-dir checkpoints/iscas89_rl \
  --epochs 50 \
  --amp
```

### Larger Model

```zsh
python -m src.ml.train_reconv train \
  --dataset data/datasets/reconv_dataset.pkl \
  --embedding-dim 1024 \
  --num-encoder-layers 12 \
  --num-interaction-layers 6 \
  --checkpoint-dir checkpoints/large_model \
  --epochs 50 \
  --amp
```

**Important**: Evaluation architecture must match training:

```zsh
python -m src.ml.evaluate_reconv \
  --checkpoint checkpoints/large_model/best_model.pth \
  --embedding-dim 1024 \
  --num-encoder-layers 12 \
  --num-interaction-layers 6
```

---

## Architecture

### Model Components

- **Multi-Path Transformer** (`src/ml/reconv_lib.py`): Processes multiple reconvergent paths with attention
- **Embedding Extractor** (`src/ml/embedding_extractor.py`): Circuit gate embeddings
- **Hybrid Trainer** (`src/ml/reconv_rl_trainer.py`): Supervised + RL training

### Training Approach

1. **Supervised Learning**: Train on known justifications from ATPG
2. **Reinforcement Learning**: Learn policy to satisfy FANIN constraints
3. **Rewards**:
   - Positive: Satisfying FANIN_LUT requirements
   - Penalty: Violating FANIN_LUT or inconsistent start/reconv values

### Dataset Structure

Each sample contains:
```python
{
    'file': 'path/to/circuit.bench',
    'info': {
        'start': <node_id>,
        'reconv': <node_id>,
        'branches': [<branch1>, <branch2>],
        'paths': [[node_ids...], [node_ids...]]
    },
    'justifiable_1': True/False,
    'justifiable_0': True/False,
    'justification_1': {gate_name: LogicValue, ...},
    'justification_0': {gate_name: LogicValue, ...}
}
```

---

## Troubleshooting

### CUDA Out of Memory

The trainer auto-detects and handles OOM:
- Probes maximum safe batch size
- Applies 60% safety margin
- Auto-halves and retries on OOM during training

If issues persist:
```zsh
# Reduce initial batch size
python -m src.ml.train_reconv train --batch-size 2048 ...

# Or reduce model size
python -m src.ml.train_reconv train \
  --embedding-dim 256 \
  --num-encoder-layers 4 \
  --num-interaction-layers 2 ...
```

### Architecture Mismatch

Error: `RuntimeError: Error(s) in loading state_dict`

**Solution**: Ensure eval flags match training:
```zsh
# Check training config in checkpoint
python -c "import torch; c=torch.load('checkpoints/reconv_rl/best_model.pth'); print(c.keys())"

# Match eval to training
python -m src.ml.evaluate_reconv \
  --checkpoint <path> \
  --embedding-dim <match_training> \
  --num-encoder-layers <match_training> \
  ...
```

### No Hard Negatives in Dataset

If you see only `accuracy` (no per-category), rebuild with:
```zsh
python -m src.ml.train_reconv build \
  --bench-dir data/bench/ISCAS85 \
  --output data/datasets/reconv_dataset.pkl \
  --include-hard-negatives  # <-- Add this flag
```

---

## File Locations

- **Training/Eval Scripts**: `src/ml/train_reconv.py`, `src/ml/evaluate_reconv.py`
- **Trainer**: `src/ml/reconv_rl_trainer.py`
- **Model**: `src/ml/reconv_lib.py`
- **Dataset Builder**: `src/atpg/reconv_podem.py`
- **Checkpoints**: `checkpoints/reconv_rl/*.pth`
- **Datasets**: `data/datasets/*.pkl`
