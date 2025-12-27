#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# ==============================================================================
# DiPhy Training Script
# ==============================================================================
# Update the following before running:
#   1. SBATCH -A <your_account>  -> your allocation account
#   2. PROJECT_DIR               -> absolute path to DiPhy repo
#   3. WANDB_API_KEY             -> your W&B key (or set wandb: disabled in config)
# ==============================================================================

# --- Configuration ---
PROJECT_DIR="/path/to/DiPhy"
CONFIG_NAME="config"  # Use "dev" for testing, "config" for production

# --- Environment Setup ---
cd "${PROJECT_DIR}"

# Option 1: Use uv (recommended)
source .venv/bin/activate

# Option 2: If uv venv not available, create it
# module load python/3.11
# uv sync
# source .venv/bin/activate

# --- Weights & Biases ---
# Set your API key, or disable W&B in config with: general.wandb=disabled
export WANDB_API_KEY="${WANDB_API_KEY:-<your_wandb_api_key>}"

# --- Run Training ---
python model/src/train.py \
    --config-name="${CONFIG_NAME}" \
    general.gpus=4
