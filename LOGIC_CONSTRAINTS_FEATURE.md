# Logic Value Constraints Feature

## Overview
This document describes the logic value constraints feature added to the reconvergent path justification training system. This feature allows the model to learn specific logic requirements at certain gates, with reinforcement learning penalties for constraint violations.

## Changes Made

### 1. Logic Value as Embedding Feature
**File**: `src/ml/reconv_rl_trainer.py`
**Method**: `create_simple_gate_embedding()`

- Added logic value (`gate.val`) as a feature in gate embeddings
- Normalized values to [0, 1] range:
  - `LogicValue.ZERO (0)` → 0.0
  - `LogicValue.ONE (1)` → 0.5
  - `LogicValue.XD (don't care/2)` → 1.0
- Embedding structure now: `gate_type (11) + structural (2) + logic_value (1) = 14 base features`

### 2. Gate Initialization to Don't Care
**File**: `src/ml/reconv_rl_trainer.py`
**Method**: `prepare_batch()`

- All gates in paths are initialized to `LogicValue.XD` (don't care) before embedding extraction
- Ensures consistent starting state for training
- Allows the model to learn to assign specific values during forward pass

### 3. Logic Constraints in Dataset
**File**: `src/atpg/reconv_podem.py`
**Function**: `build_dataset()`

- Added `constraint_probability` parameter (default: 0.0)
- When enabled, randomly adds logic constraints to a percentage of dataset entries
- Constraints stored as dict: `{'gate_name': required_value (0 or 1), ...}`
- Example: `{'G14': 1, 'G22': 0}` means G14 must be 1, G22 must be 0
- Constraints are randomly generated for 1-3 gates per sample when enabled

### 4. Supervised Training with Constraints
**File**: `src/ml/reconv_rl_trainer.py`
**Method**: `train_step_supervised()`

- Checks for `logic_constraints` in dataset entry
- When constraints exist, uses them to create target tensor instead of justification dict
- Falls back to justification dict when no constraints present
- Maintains backward compatibility with existing datasets

### 5. Constraint Violation Penalty in RL
**File**: `src/ml/reconv_rl_trainer.py`
**New Method**: `compute_constraint_penalty()`

- Computes penalty when predicted values don't match required logic constraints
- Checks each constrained gate across all paths in predictions
- Returns negative penalty per violation

**Modified Method**: `train_step_rl()`
- Integrated constraint penalty into reward computation
- Penalty applied when `logic_constraints` exist in entry
- Tracks and reports constraint penalty in metrics

### 6. Reward Weights Configuration
**File**: `src/ml/reconv_rl_trainer.py`
**Property**: `reward_weights`

Added new weight for constraint violations:
```python
reward_weights = {
    'fanin_correct': 1.0,           # Reward for satisfying FANIN_LUT
    'fanin_wrong': -0.2,            # Small penalty for FANIN violations
    'consistency': -5.0,            # Large penalty for inconsistency
    'constraint_violation': -2.0    # Penalty for logic constraint violations
}
```

## Usage

### Building Dataset with Constraints
```bash
conda activate torch
python -m src.ml.train_reconv build \
    --circuit data/bench/ISCAS85/c432.bench \
    --output data/datasets/c432_with_constraints.pkl \
    --constraint_probability 0.3  # 30% of samples will have constraints
```

### Training with Constrained Dataset
```bash
python -m src.ml.train_reconv train \
    --dataset data/datasets/c432_with_constraints.pkl \
    --output checkpoints/constrained_model/ \
    --epochs 50
```

The model will:
1. Learn to predict justifications (supervised loss)
2. Learn to satisfy FANIN_LUT rules (RL reward)
3. Learn to respect logic value constraints (RL penalty)
4. Incur penalty when predicted values violate constraints

### Build and Train in One Command
```bash
python -m src.ml.train_reconv both \
    --circuit data/bench/ISCAS85/c432.bench \
    --dataset data/datasets/c432_constrained.pkl \
    --output checkpoints/constrained_model/ \
    --epochs 50 \
    --constraint_probability 0.3
```

## Benefits

1. **More Constrained Learning**: Model learns to satisfy specific logic requirements, not just justifiability
2. **Test Pattern Specificity**: Useful for ATPG scenarios where certain observable points need specific values
3. **Backward Compatible**: Existing datasets without constraints still work (constraint_probability=0.0)
4. **Flexible Constraints**: Can control what percentage of training samples have constraints
5. **RL Integration**: Constraints naturally integrated as penalties in policy gradient training

## Technical Details

### Logic Value Encoding
- Don't care (XD) is the neutral state: neither 0 nor 1
- During training, gates start as don't care
- Model learns to assign specific values based on constraints
- Normalized encoding allows smooth gradient flow

### Constraint Format
Dataset entries with constraints have an additional field:
```python
{
    'circuit_name': 'c432.bench',
    'paths': [[1, 5, 10, 15], [2, 7, 10, 15]],
    'start_node': 1,
    'reconv_node': 10,
    'justification_1': {'G10': 1, 'G5': 0, ...},
    'logic_constraints': {'G10': 1, 'G15': 0}  # NEW FIELD
}
```

### Reward Computation Flow
1. FANIN reward: check if predicted values satisfy gate logic rules
2. Consistency reward: check if start/reconv nodes match across paths
3. **Constraint penalty**: check if predicted values match required constraints
4. Total reward = fanin_reward + consistency_reward + constraint_penalty
5. Policy gradient: update policy to maximize total reward

## Testing

To test the constraint feature:
1. Build a small dataset with constraints (e.g., c17.bench with constraint_probability=0.5)
2. Train for a few epochs
3. Check that constraint_penalty appears in logs
4. Verify that penalty decreases over training (model learning to satisfy constraints)

Example:
```bash
# Build test dataset
python -m src.ml.train_reconv build \
    --circuit data/bench/ISCAS85/c17.bench \
    --output data/datasets/c17_test_constraints.pkl \
    --constraint_probability 0.5

# Train with verbose output
python -m src.ml.train_reconv train \
    --dataset data/datasets/c17_test_constraints.pkl \
    --output checkpoints/test_constraints/ \
    --epochs 5 \
    --batch_size 32 \
    --verbose
```

Expected output should show constraint penalties in RL training steps:
```
    [RL ] policy_loss=X.XXXXXX, avg_reward=X.XXXXXX, constraint_penalty=-X.XXXXXX
```

## Future Enhancements

Potential improvements:
1. **Variable constraint weights**: Different penalties for different gates
2. **Soft constraints**: Probabilistic constraints instead of hard requirements
3. **Constraint generation strategies**: Smart selection of which gates to constrain
4. **Multi-value constraints**: Support for don't care constraints (neither 0 nor 1)
5. **Constraint-aware evaluation**: Separate accuracy metrics for constrained samples
