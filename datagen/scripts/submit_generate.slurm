#!/bin/bash
#SBATCH --job-name=sistem_regimes
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=72:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3GB
#SBATCH -A bio230007p

set -euo pipefail

JOBS_PATH=${JOBS_PATH:-"$PWD/datagen/output/sistem_regimes_v1/jobs.jsonl"}
OFFSET=${OFFSET:-0}
TASK_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /ocean/projects/cis250063p/ssabata/DiPhy/datagen/sistem

python - <<'PY'
import sys
try:
    import sistem
    from sistem import GrowthSimulator
except Exception as exc:
    print("SISTEM import failed:", exc, file=sys.stderr)
    raise
print("Using python:", sys.executable)
print("SISTEM module:", sistem.__file__)
PY

python datagen/run_job.py --jobs "$JOBS_PATH" --task-id "$TASK_ID"
