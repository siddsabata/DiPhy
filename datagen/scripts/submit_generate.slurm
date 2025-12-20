#!/bin/bash
#SBATCH --job-name=sistem_regimes
#SBATCH --output=logs/sistem_regimes_%A_%a.out
#SBATCH --error=logs/sistem_regimes_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3GB
#SBATCH -A bio230007p

set -euo pipefail

JOBS_PATH=${JOBS_PATH:-"$PWD/datagen/output/sistem_regimes_v1/jobs.jsonl"}
TASK_ID=${SLURM_ARRAY_TASK_ID}

module load anaconda3

conda activate /ocean/projects/cis250063p/ssabata/DiPhy/datagen/sistem

python datagen/run_job.py --jobs "$JOBS_PATH" --task-id "$TASK_ID"