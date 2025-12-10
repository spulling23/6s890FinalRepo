# Game-Theoretic Behavioral Cloning for Chess

**Course**: 6.S890 Multi-Agent Learning  
**Team**: Isaac (Zack) Duitz, Charikleia (Hara) Moraitaki, Skyler Pulling

## Project Overview

This project investigates whether training on near-equilibrium data can reduce sample complexity in behavioral cloning for chess. We train transformer models on datasets stratified by player skill and measure learning efficiency, validation accuracy, and alignment with engine play.

**Research Questions**:
- **H1 (Expert Variance)**: Does training exclusively on expert gameplay (Elo 2000+) require fewer samples than mixed-skill data?
- **H2 (Game-Theoretic Regularization)**: Does adding KL-divergence regularization toward Stockfish further reduce sample complexity?

**Key Finding**: Data quality dramatically outweighs data quantity. The Expert model (47K games) achieves 2.7× higher accuracy than the Random model (268K games).

## Experimental Conditions

| Condition | Training Data | Games | Loss Function |
|-----------|--------------|-------|---------------|
| Random | Elo 1000+ | 268,062 | Cross-entropy |
| Expert | At least one player Elo 2000+ | 47,066 | Cross-entropy |
| Grandmaster | At least one player Elo 2400+ | 274,794 | Cross-entropy |
| GT-regularized | Grandmaster + guidance | 274,794 | CE − λH(p) |

## Results Summary

- **Expert model**: 46% top-1 validation accuracy, 25–33% Stockfish agreement
- **Random model**: 15% top-1 validation accuracy, 7–11% Stockfish agreement
- Expert reaches 30% accuracy threshold in ~3,500 steps; Random never reaches it

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/haramor/6s890-finalproject.git
cd 6s890-finalproject
```

### 2. Set Up Environment
```bash
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio scipy h5py tqdm tensorboard python-chess tables

# Clone base model architecture
git clone https://github.com/sgrvinod/chess-transformers.git
```

### 3. Data Preparation
See `full_pipeline_setup.md` for the full reproducible pipeline, including:
- Environment setup from a fresh container
- Installing dependencies (PyTorch, chess-transformers, pgn-extract)
- Downloading and processing Lichess PGN archives
- Chunked processing for large-scale data
- Combining H5 files and generating statistics

  
### 4. Run Training
```bash
cd experiments

# Random baseline
python scripts/train.py --config configs/random_config.py

# Expert-only (H1)
python scripts/train.py --config configs/expert_config.py

# Grandmaster
python scripts/train.py --config configs/grandmaster_config.py
```

### 5. View Logs
```bash
tensorboard --logdir results/
```

## Project Structure
```
experiments/
├── configs/                    # Experimental configurations
├── models/                     # Model and loss implementations
├── scripts/                    # Training and evaluation scripts
├── data/                       # Data preparation utilities
└── results/                    # Training logs and checkpoints

chess-transformers/             # Clone separately (not in repo)
```

## Model Architecture

We use the `chess-transformers` encoder-only architecture (CT-E-20) with ~20M parameters:
- 512-d embeddings, 6 layers, 8 attention heads
- Input: 69 tokens (turn + castling rights + 64 board squares)
- Output: distribution over 1,971 UCI moves

## Evaluation

- **Training dynamics**: Steps to reach accuracy thresholds (10%, 20%, 30%)
- **Validation accuracy**: Top-1 and top-3 on held-out positions with legal move masking
- **Stockfish alignment**: Agreement with engine moves at Elo 1500, 2000, 2500

## Documentation

- **README.md** — Project overview and quick start
- **full_pipeline_setup.md** — Full reproducible pipeline: environment setup, PGN processing, and dataset creation from scratch
- **experiments/README.md** — Experiment configurations and training details
  
## Requirements

- Python 3.8+
- PyTorch 2.0+
- python-chess, h5py, tensorboard
- Stockfish (for evaluation only)

## Citation

If you use this code, please cite our project report:
```
Duitz, Moraitaki, and Pulling. "Leveraging Game-Theoretic Rationality to Reduce 
Sample Complexity in Behavioral Cloning." 6.S890 Final Project, MIT, 2025.
```

## License

Academic research project for 6.S890.
