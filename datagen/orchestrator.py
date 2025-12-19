#!/usr/bin/env python3
"""Simplified orchestrator for large SISTEM simulation batches.

The script now owns all configuration so long-running VM jobs can be started
with a single command.  Parameters are loaded from ``configs/config.yaml``
and workers receive them via command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from sistem import Parameters

# Resolve key paths once so subprocess invocations remain simple and robust.
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'configs' / 'config.yaml'

# Load configuration from YAML file
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Extract experiment parameters
PARAM_SET_ID: str = CONFIG['experiment']['param_set_id']
TOTAL_TUMORS: int = CONFIG['experiment']['total_tumors']
RESAMPLES_PER_TUMOR: int = CONFIG['experiment']['resamples_per_tumor']
SISTEM_PARAMS: Dict[str, Any] = CONFIG['sistem_parameters']

# JSON-encoded SISTEM params for passing to workers
SISTEM_PARAMS_JSON: str = json.dumps(SISTEM_PARAMS)

# Output directory structure
OUTPUT_ROOT = ROOT / 'output' / PARAM_SET_ID
TUMOR_DIR = OUTPUT_ROOT / 'tumors'
RESAMPLE_DIR = OUTPUT_ROOT / 'resamples'
LOG_FILE = OUTPUT_ROOT / 'pipeline.log'


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for running individual pipeline stages.

    The pipeline can now be invoked one stage at a time to simplify debugging
    and recovery.  Each sub-command exposes a ``--workers`` flag that controls
    concurrency for that specific stage.
    """

    cpu_total = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description='Run SISTEM pipeline stages')
    subparsers = parser.add_subparsers(dest='command', required=True)

    generate_parser = subparsers.add_parser(
        'generate',
        help='Run tumor generation (Stage 1)',
    )
    generate_parser.add_argument(
        '--workers',
        type=int,
        default=max(1, min(4, cpu_total // 2)),
        help='Concurrent tumor simulations',
    )

    resample_parser = subparsers.add_parser(
        'resample',
        help='Run tumor resampling (Stage 2)',
    )
    resample_parser.add_argument(
        '--workers',
        type=int,
        default=max(1, cpu_total),
        help='Concurrent resample jobs',
    )

    return parser.parse_args()


def build_parameters(out_dir: Path | str) -> Parameters:
    """Return a fresh SISTEM Parameters instance tied to ``out_dir``.

    The SISTEM API requires a Parameters object, so we create a new one for
    each worker. This keeps the core configuration immutable while allowing
    every job to emit files into its own directory.
    """
    params = Parameters(out_dir=str(out_dir), **SISTEM_PARAMS)
    return params


def main() -> None:
    """Coordinate the selected pipeline stage."""

    args = parse_args()

    # Ensure output directories and logging are ready before launching workers.
    setup_directories()
    setup_logging()

    logging.info(
        'Starting SISTEM pipeline stage | stage=%s | param_set=%s | tumors=%d | resamples=%d',
        args.command,
        PARAM_SET_ID,
        TOTAL_TUMORS,
        RESAMPLES_PER_TUMOR,
    )
    logging.info('Log file: %s', LOG_FILE)

    if args.command == 'generate':
        run_stage_one(args.workers)
    elif args.command == 'resample':
        run_stage_two(args.workers)
    else:
        # ``argparse`` guarantees command is one of the subcommands, but keep a
        # fallback guard in case new commands are added without updates here.
        logging.error('Unknown stage requested: %s', args.command)
        sys.exit(1)

    logging.info('Stage complete | outputs stored in %s', OUTPUT_ROOT)


def setup_directories() -> None:
    """Create the output directory structure required by all stages."""

    TUMOR_DIR.mkdir(parents=True, exist_ok=True)
    RESAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)


def setup_logging() -> None:
    """Configure console + file logging."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_stage_one(max_workers: int) -> None:
    """Generate all tumors with bounded concurrency.

    The implementation now reports individual failures but keeps the overall
    batch moving so a single bad simulation no longer aborts the run.
    """

    logging.info('Stage 1: generating %d tumors (workers=%d)', TOTAL_TUMORS, max_workers)
    tumor_ids = range(1, TOTAL_TUMORS + 1)

    failures: List[int] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_tid: Dict = {
            executor.submit(run_generate_tumor, tid): tid for tid in tumor_ids
        }
        for future in as_completed(future_to_tid):
            tumor_id = future_to_tid[future]
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                failures.append(tumor_id)
                logging.error(
                    'Tumor %03d failed | returncode=%s | command=%s',
                    tumor_id,
                    exc.returncode,
                    exc.cmd,
                )
            except Exception as exc:  # noqa: BLE001 - surface unexpected issues
                failures.append(tumor_id)
                logging.exception('Tumor %03d failed with unexpected error: %s', tumor_id, exc)
            else:
                logging.info('Tumor %03d ready', tumor_id)

    if failures:
        logging.warning('Stage 1 complete with %d failures: %s', len(failures), ', '.join(f'{tid:03d}' for tid in failures))
    else:
        logging.info('Stage 1 complete without failures')


def run_stage_two(max_workers: int) -> None:
    """Resample every available tumor ``RESAMPLES_PER_TUMOR`` times."""

    # Assemble tasks only for tumors that produced checkpoints.  This keeps the
    # stage tolerant of missing upstream outputs and avoids noisy stack traces.
    tasks: List[Tuple[int, int]] = []
    missing: List[int] = []
    for tid in range(1, TOTAL_TUMORS + 1):
        checkpoint = TUMOR_DIR / f'tumor_{tid:03d}' / 'gs.pkl'
        if not checkpoint.exists():
            missing.append(tid)
            continue
        for rid in range(1, RESAMPLES_PER_TUMOR + 1):
            tasks.append((tid, rid))

    if missing:
        logging.warning(
            'Skipping %d tumors without checkpoints: %s',
            len(missing),
            ', '.join(f'{tid:03d}' for tid in missing),
        )

    if not tasks:
        logging.warning('No resample tasks scheduled; ensure tumors are generated first')
        return

    total_resamples = len(tasks)
    logging.info(
        'Stage 2: resampling %d tumors × %d (workers=%d) | scheduled=%d',
        TOTAL_TUMORS,
        RESAMPLES_PER_TUMOR,
        max_workers,
        total_resamples,
    )

    completed = 0
    failures: List[Tuple[int, int]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task: Dict = {
            executor.submit(run_resample, tid, rid): (tid, rid) for tid, rid in tasks
        }
        for future in as_completed(future_to_task):
            tid, rid = future_to_task[future]
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                failures.append((tid, rid))
                logging.error(
                    'Resample failed | tumor=%03d rep=%03d | returncode=%s | command=%s',
                    tid,
                    rid,
                    exc.returncode,
                    exc.cmd,
                )
            except Exception as exc:  # noqa: BLE001 - surface unexpected issues
                failures.append((tid, rid))
                logging.exception(
                    'Resample failed with unexpected error | tumor=%03d rep=%03d | error=%s',
                    tid,
                    rid,
                    exc,
                )
            else:
                completed += 1
                if completed % 100 == 0 or completed == total_resamples:
                    logging.info(
                        'Resample progress: %d/%d (latest tumor %03d rep %03d)',
                        completed,
                        total_resamples,
                        tid,
                        rid,
                    )

    if failures:
        logging.warning(
            'Stage 2 complete with %d failures: %s',
            len(failures),
            ', '.join(f't{tid:03d}-r{rid:03d}' for tid, rid in failures),
        )
    else:
        logging.info('Stage 2 complete without failures')


def run_generate_tumor(tumor_id: int) -> int:
    """Spawn ``generate_tumor.py`` and stream its output into the log file."""

    cmd = [
        sys.executable,
        str(ROOT / 'generate_tumor.py'),
        '--tumor-id',
        str(tumor_id),
        '--param-set-id',
        PARAM_SET_ID,
        '--sistem-params',
        SISTEM_PARAMS_JSON,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return tumor_id


def run_resample(tumor_id: int, replicate_id: int) -> Tuple[int, int]:
    """Spawn ``resample_tumor.py`` and stream its output into the log file."""

    cmd = [
        sys.executable,
        str(ROOT / 'resample_tumor.py'),
        '--tumor-id',
        str(tumor_id),
        '--replicate-id',
        str(replicate_id),
        '--param-set-id',
        PARAM_SET_ID,
        '--sistem-params',
        SISTEM_PARAMS_JSON,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return tumor_id, replicate_id


if __name__ == '__main__':
    main()

