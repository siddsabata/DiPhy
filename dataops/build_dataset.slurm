#!/bin/bash
#SBATCH --job-name=sistem_regimes
#SBATCH --output=logs/build.log
#SBATCH --error=logs/build.err
#SBATCH --time=72:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH -A bio230007p

source /Users/siddharthsabata/dev/research/DiPhy/.venv/bin/activate

python dataops/build_dataset.py \
    --input-root datagen/output/sistem_regimes_v1 \
    --output model/data/sistem_regimes_v1.pkl