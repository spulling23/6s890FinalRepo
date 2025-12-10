# Chess Behavioral Cloning with Game-Theoretic Rationality

This directory contains experiments for the 6.S890 final project on leveraging game-theoretic rationality to reduce sample complexity in behavioral cloning for chess.

## Project Structure
```
experiments/
├── configs/          # Configuration files for experimental conditions
├── data/            # Data preparation scripts and utilities
├── models/          # Custom model implementations and loss functions
├── results/         # Training logs, checkpoints, and evaluation results
├── scripts/         # Training and evaluation scripts
└── README.md        # This file
```

## Experimental Conditions

### 1. Random (Baseline)
- **Config**: `configs/random_config.py`
- **Dataset**: Elo 1000+ (268,062 games)
- **Loss**: Cross-entropy
- **Purpose**: Baseline representing naive behavioral cloning on mixed-skill data

### 2. Expert
- **Config**: `configs/expert_config.py`
- **Dataset**: At least one player Elo 2000+ (47,066 games)
- **Loss**: Cross-entropy
- **Hypothesis (H1)**: Lower variance in expert play reduces sample complexity

### 3. Grandmaster
- **Config**: `configs/grandmaster_config.py`
- **Dataset**: At least one player Elo 2400+ (274,794 games)
- **Loss**: Cross-entropy
- **Purpose**: Elite-level play approximating equilibrium strategies

### 4. Game-Theoretic Regularization
- **Config**: `configs/game_theoretic_config.py`
- **Dataset**: Grandmaster data
- **Loss**: Cross-entropy − λH(p) (entropy regularization)
- **Hypothesis (H2)**: Regularization toward equilibrium further reduces sample complexity
- **Note**: Original KL-divergence approach with Stockfish proved computationally infeasible

### Additional Configs
- `expert_entropy_2k.py`: Expert data with entropy regularization
- `expert_standard_2k.py`: Expert data with standard cross-entropy
- `medium_config.py`: Intermediate configuration for testing

## Evaluation Metrics

1. **Training Dynamics**: Steps to reach accuracy thresholds (10%, 20%, 30%)
2. **Top-K Accuracy**: Top-1 and top-3 move prediction with legal move masking
3. **Stockfish Alignment**: Agreement with engine moves at Elo 1500, 2000, 2500

## Quick Start
```bash
# Activate environment
source ../venv/bin/activate

# Train Random baseline
python scripts/train.py --config configs/random_config.py

# Train Expert model
python scripts/train.py --config configs/expert_config.py

# Train Grandmaster model
python scripts/train.py --config configs/grandmaster_config.py

# View training logs
tensorboard --logdir results/
```

## Evaluation
```bash
# Calculate Stockfish alignment
python scripts/stock_agreement_new.py

# Evaluate checkpoint
python scripts/extract_all_metrics.py
```

## Dependencies

- PyTorch 2.0+
- python-chess
- tensorboard
- h5py / tables
- Stockfish (for evaluation)
