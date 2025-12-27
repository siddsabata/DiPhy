#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --output=train_dev_%j.out
#SBATCH --error=train_dev_%j.err

# ==============================================================================
# DiPhy DEV Training Script - For quick testing
# ==============================================================================
# This uses the dev.yaml config with reduced parameters for fast iteration.
# Update PROJECT_DIR before running.
# ==============================================================================

PROJECT_DIR="/path/to/DiPhy"

cd "${PROJECT_DIR}"
source .venv/bin/activate

# Disable W&B for dev runs (already disabled in dev.yaml, but just in case)
export WANDB_MODE=disabled

python model/src/train.py \
    --config-name=dev \
    dataset.data_path=model/data/dev.pkl
