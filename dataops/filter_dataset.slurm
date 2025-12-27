#!/bin/bash
#SBATCH --job-name=sistem_regimes
#SBATCH --output=logs/sub.log
#SBATCH --error=logs/sub.err
#SBATCH --time=72:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH -A bio230007p

source /ocean/projects/cis250063p/ssabata/DiPhy/.venv/bin/activate

python dataops/filter_dataset.py \
      --input model/data/dev.pkl \
      --output model/data/dev_filtered.pkl \
      --max-nodes 200

python dataops/filter_dataset.py \
      --input model/data/sistem_regimes_v1.pkl \
      --output model/data/main_filtered.pkl \
      --max-nodes 200