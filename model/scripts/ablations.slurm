#!/bin/bash
#SBATCH -A cis250063p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=44:00:00
#SBATCH --mem=48G
#SBATCH --array=0-8
#SBATCH --output=ablation_%A_%a.out
#SBATCH --error=ablation_%A_%a.err

# ==============================================================================
# DiPhy Ablation Study - 9 Jobs (3 datasets x 3 model sizes)
# ==============================================================================
# Before submitting, pre-process all datasets:
#   python model/src/preprocess.py model/data/subset_25.pkl model/data/subset_50.pkl model/data/main_filtered.pkl
#
# Submit with:
#   sbatch model/scripts/ablations.slurm
# ==============================================================================

PROJECT_DIR="/ocean/projects/cis250063p/ssabata/DiPhy/"
CONFIG_NAME="phylo"

cd "${PROJECT_DIR}"
source .venv/bin/activate
export WANDB_API_KEY="${WANDB_API_KEY:-<your_key>}"

# ==============================================================================
# ABLATION CONFIGURATIONS
# Edit the variables below for each job
# ==============================================================================

case $SLURM_ARRAY_TASK_ID in
    # --- Dataset: Small ---
    0)
        NAME="data_small_model_small"
        DATA="model/data/subset_25.pkl"
        LAYERS=6
        ;;
    1)
        NAME="data_small_model_medium"
        DATA="model/data/subset_25.pkl"
        LAYERS=9
        ;;
    2)
        NAME="data_small_model_large"
        DATA="model/data/subset_25.pkl"
        LAYERS=12
        ;;

    # --- Dataset: Medium ---
    3)
        NAME="data_medium_model_small"
        DATA="model/data/subset_50.pkl"
        LAYERS=6
        ;;
    4)
        NAME="data_medium_model_medium"
        DATA="model/data/subset_50.pkl"
        LAYERS=9
        ;;
    5)
        NAME="data_medium_model_large"
        DATA="model/data/subset_50.pkl"
        LAYERS=12
        ;;

    # --- Dataset: Large ---
    6)
        NAME="data_large_model_small"
        DATA="model/data/main_filtered.pkl"
        LAYERS=6
        ;;
    7)
        NAME="data_large_model_medium"
        DATA="model/data/main_filtered.pkl"
        LAYERS=9
        ;;
    8)
        NAME="data_large_model_large"
        DATA="model/data/main_filtered.pkl"
        LAYERS=12
        ;;
esac

echo "=============================================="
echo "Job $SLURM_ARRAY_TASK_ID: $NAME"
echo "  Dataset: $DATA"
echo "  Layers: $LAYERS, dx: $DX"
echo "=============================================="

# ==============================================================================
# TRAINING
# ==============================================================================

OUTPUT=$(python model/src/train.py \
    --config-name="${CONFIG_NAME}" \
    general.name="${NAME}" \
    dataset.data_path="${DATA}" \
    model.n_layers="${LAYERS}" \
    2>&1 | tee /dev/stderr)

# ==============================================================================
# SAMPLING
# ==============================================================================

RUN_DIR=$(echo "$OUTPUT" | grep -oP '(?<=Output directory: ).*' | head -1)
CHECKPOINT="${RUN_DIR}/checkpoints/best.ckpt"

echo "Run directory: $RUN_DIR"
echo "Checkpoint: $CHECKPOINT"

if [ -f "$CHECKPOINT" ]; then
    echo "Running sampling..."
    python model/src/sample.py --checkpoint "$CHECKPOINT"
else
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Job $SLURM_ARRAY_TASK_ID complete!"
