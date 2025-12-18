#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Load environment
module load anaconda3
conda activate <your_conda_env>

# Set wandb API key (or disable with: export WANDB_MODE=disabled)
export WANDB_API_KEY=<your_wandb_api_key>

# Run training
# Update paths below to match your setup
python /path/to/DiPhy/model/src/train.py \
    dataset.data_path=/path/to/phylo.pkl \
    general.gpus=4 \
    train.n_epochs=500
