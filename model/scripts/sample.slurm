#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --output=sample_%j.out
#SBATCH --error=sample_%j.err

# Load environment
module load anaconda3
conda activate <your_conda_env>

# Run sampling
# Update paths below to match your setup
python /path/to/DiPhy/model/src/sample.py \
    --checkpoint /path/to/checkpoints/best.ckpt \
    --num_samples 1000 \
    --batch_size 32 \
    --output_dir /path/to/output/samples \
    --output_format pickle
