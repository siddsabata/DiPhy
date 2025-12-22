#!/usr/bin/env python3
"""Build a training dataset pickle from datagen regime outputs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

DATAOPS_ROOT = Path(__file__).resolve().parent
if str(DATAOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATAOPS_ROOT))

from phylogeny import Phylogeny  # noqa: E402

REQUIRED_FILES = ("clone_tree.nwk", "SNV_events.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset pickle from datagen outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to datagen output root, e.g. datagen/output/sistem_regimes_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the combined .pkl file",
    )
    return parser.parse_args()


def build_tree_dict(tree_id: str, newick_file: Path, snv_file: Path) -> Dict[str, List]:
    phylo = Phylogeny(str(newick_file), str(snv_file))
    X, E, L = phylo.transform()
    return {
        "tree_id": tree_id,
        "X": X.tolist(),
        "E": E.tolist(),
        "L": L.tolist(),
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_path = args.output.resolve()

    regimes_root = input_root / "regimes"
    if not regimes_root.is_dir():
        raise FileNotFoundError(f"Missing regimes directory: {regimes_root}")

    dataset: List[Dict[str, List]] = []
    counts: Dict[str, int] = defaultdict(int)
    skipped = 0

    for regime_dir in sorted(regimes_root.iterdir()):
        if not regime_dir.is_dir():
            continue
        regime_id = regime_dir.name
        tumors_root = regime_dir / "tumors"
        if not tumors_root.is_dir():
            continue

        for tumor_dir in sorted(tumors_root.iterdir()):
            if not tumor_dir.is_dir():
                continue
            success_path = tumor_dir / "success.json"
            if not success_path.exists():
                skipped += 1
                continue
            try:
                success = json.loads(success_path.read_text())
            except json.JSONDecodeError:
                skipped += 1
                continue

            attempt_dir = tumor_dir / success.get("attempt_dir", "")
            if not attempt_dir.is_dir():
                skipped += 1
                continue

            missing = [name for name in REQUIRED_FILES if not (attempt_dir / name).exists()]
            if missing:
                skipped += 1
                continue

            tree_id = f"{regime_id}/{tumor_dir.name}/{attempt_dir.name}"
            try:
                tree_dict = build_tree_dict(
                    tree_id,
                    attempt_dir / "clone_tree.nwk",
                    attempt_dir / "SNV_events.tsv",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {tree_id}: {exc}")
                skipped += 1
                continue

            dataset.append(tree_dict)
            counts[regime_id] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(dataset, handle)

    total = sum(counts.values())
    print(f"Wrote {total} trees to {output_path}")
    print("Counts by regime:")
    for regime_id in sorted(counts):
        print(f"  {regime_id}: {counts[regime_id]}")
    if skipped:
        print(f"Skipped: {skipped} entries with missing or invalid outputs")


if __name__ == "__main__":
    main()
