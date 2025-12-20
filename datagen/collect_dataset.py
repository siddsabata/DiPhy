#!/usr/bin/env python3
"""Collect successful runs and assign dataset splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dataset outputs")
    parser.add_argument("--run-id", type=str, required=True, help="Run id, e.g. sistem_regimes_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "output" / args.run_id

    jobs_path = output_dir / "jobs.jsonl"
    config_path = output_dir / "summary.json"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs manifest: {jobs_path}")

    summary = json.loads(config_path.read_text()) if config_path.exists() else {}
    rng = np.random.default_rng(20251219)

    successes: Dict[str, List[Dict[str, str]]] = {}
    failures: List[Dict[str, str]] = []

    with jobs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            job = json.loads(line)
            regime_id = job["regime_id"]
            tumor_index = int(job["tumor_index"])
            tumor_dir = output_dir / "regimes" / regime_id / "tumors" / f"tumor_{tumor_index:05d}"
            success_path = tumor_dir / "success.json"
            if not success_path.exists():
                failures.append({"regime_id": regime_id, "tumor_index": str(tumor_index)})
                continue
            success = json.loads(success_path.read_text())
            attempt_dir = tumor_dir / success["attempt_dir"]
            row = {
                "run_id": job["run_id"],
                "regime_id": regime_id,
                "tumor_index": str(tumor_index),
                "attempt_index": str(success["attempt_index"]),
                "path": str(attempt_dir),
            }
            successes.setdefault(regime_id, []).append(row)

    dataset_rows: List[Dict[str, str]] = []
    split_ratios = summary.get("splits", {"train": 0.9, "val": 0.05, "test": 0.05})

    for regime_id, rows in successes.items():
        rng.shuffle(rows)
        total = len(rows)
        n_train = int(total * split_ratios.get("train", 0.9))
        n_val = int(total * split_ratios.get("val", 0.05))
        split_points = [n_train, n_train + n_val]

        for idx, row in enumerate(rows):
            if idx < split_points[0]:
                split = "train"
            elif idx < split_points[1]:
                split = "val"
            else:
                split = "test"
            row["split"] = split
            dataset_rows.append(row)

    dataset_path = output_dir / "dataset.csv"
    failures_path = output_dir / "failures.csv"

    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "regime_id", "tumor_index", "attempt_index", "split", "path"],
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime_id", "tumor_index"])
        writer.writeheader()
        writer.writerows(failures)


if __name__ == "__main__":
    main()
