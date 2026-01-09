#!/usr/bin/env python3
"""Filter an existing dataset pickle by node count and tree validity.

Use this to filter already-built pickle files. For new builds, prefer using
the --max-nodes flag in build_dataset.py.

Filters:
- Node count: Trees with more than --max-nodes are removed.
- Validity: Trees that fail structural validity checks are removed (enabled by default).
  Validity checks include: no cycles in clone subgraph, root has exactly one clone
  neighbor, clone edges only connect root/clone nodes, mutation edges only connect
  clone-mutation pairs.

Usage:
    python filter_dataset.py --input data.pkl --output filtered.pkl --max-nodes 200
    python filter_dataset.py --input data.pkl --output filtered.pkl --no-filter-invalid
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tree_metrics import compute_tree_metrics


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
    parser.add_argument(
        "--filter-invalid",
        action="store_true",
        default=True,
        help="Filter out trees that fail validity checks (default: True)",
    )
    parser.add_argument(
        "--no-filter-invalid",
        action="store_false",
        dest="filter_invalid",
        help="Disable validity filtering",
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
    invalid_counts: Dict[str, int] = defaultdict(int)

    for tree in dataset:
        num_nodes = len(tree["X"])
        regime = get_regime_from_tree_id(tree.get("tree_id", "unknown"))

        # Check max nodes
        if num_nodes > max_nodes:
            filtered_counts[regime] += 1
            continue

        # Check validity if enabled
        if args.filter_invalid:
            metrics = compute_tree_metrics(tree)
            if not metrics.validity.is_valid:
                invalid_counts[regime] += 1
                continue

        kept.append(tree)
        kept_counts[regime] += 1

    total_kept = len(kept)
    total_size_filtered = sum(filtered_counts.values())
    total_invalid = sum(invalid_counts.values())

    print(f"\nFiltering with max_nodes={max_nodes}, filter_invalid={args.filter_invalid}:")
    all_regimes = sorted(
        set(kept_counts.keys()) | set(filtered_counts.keys()) | set(invalid_counts.keys())
    )

    if args.filter_invalid:
        print("\nCounts by regime (kept / size-filtered / invalid):")
        for regime in all_regimes:
            k = kept_counts[regime]
            f = filtered_counts[regime]
            inv = invalid_counts[regime]
            print(f"  {regime}: {k} kept / {f} size-filtered / {inv} invalid")
    else:
        print("\nCounts by regime (kept / size-filtered):")
        for regime in all_regimes:
            k = kept_counts[regime]
            f = filtered_counts[regime]
            print(f"  {regime}: {k} kept / {f} size-filtered")

    pct_size = total_size_filtered / len(dataset) * 100 if dataset else 0
    if args.filter_invalid:
        pct_invalid = total_invalid / len(dataset) * 100 if dataset else 0
        print(
            f"\nTotal: {total_kept} kept / {total_size_filtered} size-filtered ({pct_size:.1f}%) / "
            f"{total_invalid} invalid ({pct_invalid:.1f}%)"
        )
    else:
        print(f"\nTotal: {total_kept} kept / {total_size_filtered} size-filtered ({pct_size:.1f}%)")

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
