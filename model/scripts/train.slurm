#!/bin/bash
#SBATCH -A cis250063p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:4
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --signal=SIGUSR1@120  # Signal 120 seconds before time limit for graceful shutdown

# ==============================================================================
# DiPhy Training Script
# ==============================================================================
# Update the following before running:
#   1. SBATCH -A <your_account>  -> your allocation account
#   2. PROJECT_DIR               -> absolute path to DiPhy repo
#   3. WANDB_API_KEY             -> your W&B key (or set wandb: disabled in config)
# ==============================================================================

# --- Configuration ---
PROJECT_DIR="/ocean/projects/cis250063p/ssabata/DiPhy/"
CONFIG_NAME="phylo_mini"  # Use "dev" for testing, "config" for production

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
# Fresh run:
#   sbatch model/scripts/train.slurm
#
# Resume from checkpoint (pass CKPT env var):
#   sbatch --export=ALL,CKPT=/path/to/checkpoints/last.ckpt model/scripts/train.slurm

if [ -n "${CKPT}" ]; then
    # Resume mode: derive run_dir from checkpoint path
    RUN_DIR=$(dirname $(dirname "${CKPT}"))
    echo "Resuming from: ${CKPT}"
    echo "Output dir: ${RUN_DIR}"
    python model/src/train.py \
        --config-name="${CONFIG_NAME}" \
        train.resume_from_checkpoint="${CKPT}" \
        hydra.run.dir="${RUN_DIR}" \
        general.gpus=4
else
    # Fresh start
    python model/src/train.py \
        --config-name="${CONFIG_NAME}" \
        general.gpus=4
fi
