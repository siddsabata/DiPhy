#!/usr/bin/env python3
"""Create a uniformly subsampled dataset pickle."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample a dataset pickle by fraction or absolute size."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source .pkl file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the subsampled .pkl file",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help="Fraction of the dataset to keep (0 < f <= 1).",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Absolute number of samples to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20251219,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_target_size(total: int, fraction: float | None, size: int | None) -> int:
    if fraction is None and size is None:
        raise ValueError("Provide either --fraction or --size.")
    if fraction is not None and size is not None:
        raise ValueError("Provide only one of --fraction or --size.")
    if fraction is not None:
        if fraction <= 0 or fraction > 1:
            raise ValueError("--fraction must be in (0, 1].")
        return max(1, int(round(total * fraction)))
    if size is None or size <= 0:
        raise ValueError("--size must be a positive integer.")
    if size > total:
        raise ValueError("--size cannot exceed dataset size.")
    return size


def main() -> None:
    args = parse_args()
    data_path = args.input.resolve()
    output_path = args.output.resolve()

    with data_path.open("rb") as handle:
        dataset: List[Any] = pickle.load(handle)

    total = len(dataset)
    target_size = resolve_target_size(total, args.fraction, args.size)

    rng = random.Random(args.seed)
    subset = rng.sample(dataset, target_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(subset, handle)

    print(f"Loaded {total} samples from {data_path}")
    print(f"Wrote {len(subset)} samples to {output_path}")


if __name__ == "__main__":
    main()
