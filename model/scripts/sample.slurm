#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --output=sample_%j.out
#SBATCH --error=sample_%j.err

# ==============================================================================
# DiPhy Sampling Script
# ==============================================================================
# Update the following before running:
#   1. SBATCH -A <your_account>  -> your allocation account
#   2. PROJECT_DIR               -> absolute path to DiPhy repo
#   3. CHECKPOINT_PATH           -> path to trained model checkpoint
# ==============================================================================

PROJECT_DIR="/path/to/DiPhy"
CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints/phylo/best.ckpt"
OUTPUT_DIR="${PROJECT_DIR}/samples"

cd "${PROJECT_DIR}"
source .venv/bin/activate

python model/src/sample.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --num_samples 1000 \
    --batch_size 32 \
    --output_dir "${OUTPUT_DIR}" \
    --output_format pickle
