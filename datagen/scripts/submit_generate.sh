#!/bin/bash
set -euo pipefail

CONFIG_PATH=${1:-"datagen/configs/regimes.yaml"}
python datagen/build_jobs.py --config "$CONFIG_PATH"

OUTPUT_ROOT=$(python - <<PY
from pathlib import Path
import yaml

config_path = Path("$CONFIG_PATH")
config = yaml.safe_load(config_path.read_text())
print(config["experiment"]["output_root"])
PY
)

JOBS_PATH="datagen/${OUTPUT_ROOT}/jobs.jsonl"
STATE_PATH="datagen/${OUTPUT_ROOT}/submit_state.json"
JOB_COUNT=$(wc -l < "$JOBS_PATH" | tr -d ' ')

if [ "$JOB_COUNT" -le 0 ]; then
  echo "No jobs found in $JOBS_PATH" >&2
  exit 1
fi

mkdir -p logs

CHUNK_SIZE=${CHUNK_SIZE:-1000}
MAX_CONCURRENT=${MAX_CONCURRENT:-200}
MAX_BATCHES=${MAX_BATCHES:-0}

if [ "$CHUNK_SIZE" -gt 0 ]; then
  if [ -f "$STATE_PATH" ]; then
    START=$(cat "$STATE_PATH" | tr -d ' ')
  else
    START=0
  fi
  if [ "$START" -ge "$JOB_COUNT" ]; then
    echo "All jobs already submitted (next_offset=$START)" >&2
    exit 0
  fi
  BATCHES=0
  while [ "$START" -lt "$JOB_COUNT" ]; do
    END=$((START + CHUNK_SIZE - 1))
    if [ "$END" -ge "$JOB_COUNT" ]; then
      END=$((JOB_COUNT - 1))
    fi
    COUNT=$((END - START + 1))
    sbatch --array=0-$((COUNT - 1))%${MAX_CONCURRENT} \
      --export=JOBS_PATH="$JOBS_PATH",OFFSET="$START" \
      datagen/scripts/submit_generate.slurm
    START=$((END + 1))
    mkdir -p "$(dirname "$STATE_PATH")"
    printf "%s\n" "$START" > "$STATE_PATH"
    if [ "$MAX_BATCHES" -gt 0 ]; then
      BATCHES=$((BATCHES + 1))
      if [ "$BATCHES" -ge "$MAX_BATCHES" ]; then
        break
      fi
    fi
  done
else
  sbatch --array=0-$((JOB_COUNT - 1))%${MAX_CONCURRENT} \
    --export=JOBS_PATH="$JOBS_PATH",OFFSET=0 \
    datagen/scripts/submit_generate.slurm
fi
