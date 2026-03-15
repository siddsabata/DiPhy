#!/usr/bin/env python3
"""Sample a dataset uniformly by regime.

Creates balanced subsets by uniformly sampling regimes, then sampling trees
within each regime. Supports including/excluding specific regimes.

Sampling is done WITHOUT replacement - no tree can be sampled twice.
If any regime runs out of samples before reaching the target count, an error is raised.

Usage:
    # Sample from all regimes
    python sample_by_regime.py --input data.pkl --output sampled.pkl --num-samples 1000

    # Exclude specific regime
    python sample_by_regime.py --input data.pkl --output sampled.pkl \
        --num-samples 1000 --exclude-regimes R12_small_trees_early_detection_low_sampling

    # Include only specific regime
    python sample_by_regime.py --input data.pkl --output sampled.pkl \
        --num-samples 1000 --include-regimes R01_single_site_arm_neutral
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a dataset uniformly by regime."
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
        help="Path to write sampled pickle file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--include-regimes",
        type=str,
        default=None,
        help="Comma-separated list of regimes to include (default: all)",
    )
    parser.add_argument(
        "--exclude-regimes",
        type=str,
        default=None,
        help="Comma-separated list of regimes to exclude (default: none)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
    num_samples = args.num_samples
    seed = args.seed

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Set random seed
    random.seed(seed)

    # Load dataset
    print(f"Loading {input_path}...")
    with input_path.open("rb") as handle:
        dataset: List[Dict] = pickle.load(handle)

    print(f"Loaded {len(dataset)} graphs")

    # Group indices by regime
    regime_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, tree in enumerate(dataset):
        regime = get_regime_from_tree_id(tree.get("tree_id", "unknown"))
        regime_indices[regime].append(idx)

    all_regimes = sorted(regime_indices.keys())
    print(f"\nFound {len(all_regimes)} regimes:")
    for regime in all_regimes:
        print(f"  {regime}: {len(regime_indices[regime])} trees")

    # Determine which regimes to sample from
    if args.include_regimes:
        include_set = set(r.strip() for r in args.include_regimes.split(","))
        # Validate all specified regimes exist
        missing = include_set - set(all_regimes)
        if missing:
            raise ValueError(f"Specified regimes not found in dataset: {missing}")
        target_regimes = sorted(include_set)
    else:
        target_regimes = all_regimes.copy()

    if args.exclude_regimes:
        exclude_set = set(r.strip() for r in args.exclude_regimes.split(","))
        target_regimes = [r for r in target_regimes if r not in exclude_set]

    if not target_regimes:
        raise ValueError("No regimes to sample from after applying include/exclude filters")

    print(f"\nSampling from {len(target_regimes)} regimes: {target_regimes}")

    # Calculate target samples per regime
    base_per_regime = num_samples // len(target_regimes)
    print(f"Target: ~{base_per_regime} samples per regime")

    # Check total available samples
    total_available = sum(len(regime_indices[r]) for r in target_regimes)
    if total_available < num_samples:
        raise ValueError(
            f"Total available samples ({total_available}) is less than requested ({num_samples})"
        )

    # Sample uniformly by regime using round-robin to ensure balance
    used_indices: Set[int] = set()
    sampled_trees: List[Dict] = []
    regime_sample_counts: Dict[str, int] = defaultdict(int)

    # Create a copy of available indices per regime (shuffled for randomness)
    available_per_regime: Dict[str, List[int]] = {}
    for regime in target_regimes:
        indices = regime_indices[regime].copy()
        random.shuffle(indices)
        available_per_regime[regime] = indices

    # Calculate samples per regime: base amount plus remainder distribution
    base_per_regime = num_samples // len(target_regimes)
    remainder = num_samples % len(target_regimes)

    # Shuffle regimes to randomize which ones get the extra sample
    shuffled_regimes = target_regimes.copy()
    random.shuffle(shuffled_regimes)

    samples_needed: Dict[str, int] = {}
    for i, regime in enumerate(shuffled_regimes):
        samples_needed[regime] = base_per_regime + (1 if i < remainder else 0)

    print(f"\nSampling {num_samples} trees ({base_per_regime} per regime, {remainder} extra distributed)...")

    # Sample from each regime
    for regime in target_regimes:
        count_needed = samples_needed[regime]
        available = available_per_regime[regime]

        for _ in range(count_needed):
            if not available:
                raise RuntimeError(
                    f"Regime '{regime}' exhausted. "
                    f"This should not happen if validation passed."
                )

            global_idx = available.pop()
            used_indices.add(global_idx)
            sampled_trees.append(dataset[global_idx])
            regime_sample_counts[regime] += 1

    # Shuffle final list
    random.shuffle(sampled_trees)

    # Print statistics
    print(f"\nSampled {len(sampled_trees)} trees:")
    for regime in target_regimes:
        count = regime_sample_counts[regime]
        pct = count / len(sampled_trees) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    # Verify no duplicates
    tree_ids = [t.get("tree_id", f"idx_{i}") for i, t in enumerate(sampled_trees)]
    if len(tree_ids) != len(set(tree_ids)):
        raise RuntimeError("Duplicate tree_ids found in sampled dataset!")

    if args.stats_only:
        print("\n--stats-only: No output file written")
        return

    # Write sampled dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(sampled_trees, handle)

    print(f"\nWrote {len(sampled_trees)} graphs to {output_path}")


if __name__ == "__main__":
    main()
