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

mkdir -p logs

sbatch --array=0-$((JOB_COUNT - 1))%200 \
  --export=JOBS_PATH="$JOBS_PATH" \
  datagen/scripts/submit_generate.slurm
