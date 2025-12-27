#!/usr/bin/env python3
"""Filter an existing dataset pickle by node count.

Use this to filter already-built pickle files. For new builds, prefer using
the --max-nodes flag in build_dataset.py.

Usage:
    python filter_dataset.py --input data.pkl --output filtered.pkl --max-nodes 200
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a dataset pickle by maximum node count."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input pickle file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write filtered pickle file",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Maximum number of nodes allowed per graph (default: 200)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't write output file",
    )
    return parser.parse_args()


def get_regime_from_tree_id(tree_id: str) -> str:
    """Extract regime ID from tree_id (format: regime_id/tumor_id/attempt_id)."""
    parts = tree_id.split("/")
    return parts[0] if parts else "unknown"


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    max_nodes = args.max_nodes

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load dataset
    print(f"Loading {input_path}...")
    with input_path.open("rb") as handle:
        dataset: List[Dict] = pickle.load(handle)

    print(f"Loaded {len(dataset)} graphs")

    # Analyze node counts
    node_counts = [len(tree["X"]) for tree in dataset]
    print(f"\nNode count statistics:")
    print(f"  Min: {min(node_counts)}")
    print(f"  Max: {max(node_counts)}")
    print(f"  Mean: {sum(node_counts)/len(node_counts):.1f}")

    # Filter and track by regime
    kept: List[Dict] = []
    kept_counts: Dict[str, int] = defaultdict(int)
    filtered_counts: Dict[str, int] = defaultdict(int)

    for tree in dataset:
        num_nodes = len(tree["X"])
        regime = get_regime_from_tree_id(tree.get("tree_id", "unknown"))

        if num_nodes > max_nodes:
            filtered_counts[regime] += 1
        else:
            kept.append(tree)
            kept_counts[regime] += 1

    total_kept = len(kept)
    total_filtered = len(dataset) - total_kept

    print(f"\nFiltering with max_nodes={max_nodes}:")
    print("\nCounts by regime (kept / filtered):")
    all_regimes = sorted(set(kept_counts.keys()) | set(filtered_counts.keys()))
    for regime in all_regimes:
        k = kept_counts[regime]
        f = filtered_counts[regime]
        print(f"  {regime}: {k} kept / {f} filtered")

    pct = total_filtered / len(dataset) * 100 if dataset else 0
    print(f"\nTotal: {total_kept} kept / {total_filtered} filtered ({pct:.1f}% removed)")

    if args.stats_only:
        print("\n--stats-only: No output file written")
        return

    # Write filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(kept, handle)

    print(f"\nWrote {total_kept} graphs to {output_path}")


if __name__ == "__main__":
    main()
