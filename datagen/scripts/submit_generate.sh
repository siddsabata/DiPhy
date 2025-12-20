#!/bin/bash
set -euo pipefail

CONFIG_PATH=${1:-"datagen/configs/regimes.yaml"}
python datagen/build_jobs.py --config "$CONFIG_PATH"

OUTPUT_ROOT=$(python - <<PY
import json
from pathlib import Path
import yaml

config_path = Path("$CONFIG_PATH")
config = yaml.safe_load(config_path.read_text())
output_root = config["experiment"]["output_root"]
print(output_root)
PY
)

JOBS_PATH="datagen/${OUTPUT_ROOT}/jobs.jsonl"
JOB_COUNT=$(python - <<PY
from pathlib import Path
path = Path("$JOBS_PATH")
print(sum(1 for _ in path.open("r", encoding="utf-8")))
PY
)

if [ "$JOB_COUNT" -le 0 ]; then
  echo "No jobs found in $JOBS_PATH" >&2
  exit 1
fi

mkdir -p logs

CHUNK_SIZE=${CHUNK_SIZE:-1000}
MAX_CONCURRENT=${MAX_CONCURRENT:-200}

if [ "$CHUNK_SIZE" -gt 0 ]; then
  START=0
  while [ "$START" -lt "$JOB_COUNT" ]; do
    END=$((START + CHUNK_SIZE - 1))
    if [ "$END" -ge "$JOB_COUNT" ]; then
      END=$((JOB_COUNT - 1))
    fi
    sbatch --array=${START}-${END}%${MAX_CONCURRENT} \
      --export=JOBS_PATH="$JOBS_PATH" \
      datagen/scripts/submit_generate.slurm
    START=$((END + 1))
  done
else
  sbatch --array=0-$((JOB_COUNT - 1))%${MAX_CONCURRENT} \
    --export=JOBS_PATH="$JOBS_PATH" \
    datagen/scripts/submit_generate.slurm
fi
