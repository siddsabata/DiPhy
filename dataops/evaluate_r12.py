#!/usr/bin/env python3
"""R12 Generalizability evaluation for phylogenetic tree generation.

Computes distributional metrics comparing generated samples against R12 reference data:
- 1D Wasserstein distances for clone fraction, depth, leaves, mutations
- Direct MMD^2 (generated vs reference)

Usage:
    python evaluate_r12.py \
        --generated samples.pkl \
        --reference data_700_R12_only.pkl \
        --output results_r12.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

from evaluate import (
    EvalMetrics,
    compute_eval_metrics,
    metrics_to_feature_matrix,
    standardize_features,
    subsample_features,
)


@dataclass
class R12EvaluationResults:
    """R12 generalizability evaluation results."""

    # Wasserstein distances
    wasserstein_clone_fraction: float
    wasserstein_clone_depth: float
    wasserstein_clone_leaves: float
    wasserstein_avg_mutations: float

    # MMD (direct, no ratio)
    mmd_gen_ref: float

    # Counts
    num_generated: int
    num_reference: int


def compute_wasserstein_distances(
    gen_metrics: Sequence[EvalMetrics],
    ref_metrics: Sequence[EvalMetrics],
) -> Dict[str, float]:
    """Compute 1D Wasserstein distance for each metric."""
    metric_names = [
        "clone_fraction",
        "clone_tree_depth",
        "num_clone_leaves",
        "avg_mutations_per_clone",
    ]
    results: Dict[str, float] = {}

    for metric_name in metric_names:
        gen_values = np.array([getattr(m, metric_name) for m in gen_metrics])
        ref_values = np.array([getattr(m, metric_name) for m in ref_metrics])

        if len(gen_values) == 0 or len(ref_values) == 0:
            results[metric_name] = float("inf")
        else:
            results[metric_name] = float(wasserstein_distance(gen_values, ref_values))

    return results


def rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute RBF kernel matrix between X and Y."""
    sq_dists = (
        np.sum(X**2, axis=1, keepdims=True)
        + np.sum(Y**2, axis=1)
        - 2 * X @ Y.T
    )
    return np.exp(-sq_dists / (2 * bandwidth**2))


def median_heuristic_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute bandwidth using median heuristic on combined data."""
    combined = np.vstack([X, Y])
    if combined.shape[0] < 2:
        return 1.0
    dists = pdist(combined, metric="euclidean")
    return float(np.median(dists)) if len(dists) > 0 else 1.0


def compute_mmd_squared(
    X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None
) -> float:
    """Compute squared MMD between distributions X and Y."""
    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(X, Y)

    K_xx = rbf_kernel(X, X, bandwidth)
    K_yy = rbf_kernel(Y, Y, bandwidth)
    K_xy = rbf_kernel(X, Y, bandwidth)

    n, m = X.shape[0], Y.shape[0]

    # Unbiased estimator (exclude diagonal for K_xx and K_yy)
    if n > 1:
        term1 = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    else:
        term1 = 0.0

    if m > 1:
        term2 = (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    else:
        term2 = 0.0

    term3 = 2 * K_xy.mean()

    return max(0.0, float(term1 + term2 - term3))


def load_pickle(path: Path) -> List[Dict[str, Any]]:
    """Load a pickle file containing list of tree dicts."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def print_results(results: R12EvaluationResults) -> None:
    """Print evaluation results to console."""
    print("=" * 60)
    print("R12 GENERALIZABILITY EVALUATION RESULTS")
    print("=" * 60)

    print("\nSample Counts:")
    print(f"  Generated:  {results.num_generated}")
    print(f"  Reference:  {results.num_reference}")

    print("\nWasserstein Distances (Generated vs R12):")
    print(f"  Clone Fraction:     {results.wasserstein_clone_fraction:.4f}")
    print(f"  Clone Tree Depth:   {results.wasserstein_clone_depth:.4f}")
    print(f"  Clone Leaves:       {results.wasserstein_clone_leaves:.4f}")
    print(f"  Avg Muts/Clone:     {results.wasserstein_avg_mutations:.4f}")

    print("\nMMD Analysis:")
    print(f"  MMD^2(gen, ref):    {results.mmd_gen_ref:.6f}")

    print("=" * 60)


def save_results(results: R12EvaluationResults, output_path: Path) -> None:
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(results), handle, indent=2)
    print(f"\nResults saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated phylogenetic trees against R12 reference data."
    )
    parser.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="Path to generated samples pickle file",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to R12 reference dataset pickle file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results (optional)",
    )
    parser.add_argument(
        "--mmd-subsample",
        type=int,
        default=1000,
        help="Max samples per set for MMD computation (default: 1000, 0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate paths
    if not args.generated.exists():
        raise FileNotFoundError(f"Generated samples not found: {args.generated}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference data not found: {args.reference}")

    # 1. Load generated samples
    print(f"Loading generated samples from {args.generated}...")
    generated_trees = load_pickle(args.generated)
    print(f"  Loaded {len(generated_trees)} generated samples")

    # 2. Load reference data
    print(f"Loading reference data from {args.reference}...")
    reference_trees = load_pickle(args.reference)
    print(f"  Loaded {len(reference_trees)} reference samples")

    # 3. Compute evaluation metrics for each set
    print("Computing evaluation metrics...")
    gen_eval_metrics = [compute_eval_metrics(t) for t in generated_trees]
    ref_eval_metrics = [compute_eval_metrics(t) for t in reference_trees]

    # 4. Compute Wasserstein distances
    print("Computing Wasserstein distances...")
    wasserstein_results = compute_wasserstein_distances(gen_eval_metrics, ref_eval_metrics)

    # 5. Compute direct MMD
    print("Computing MMD...")
    gen_feats = metrics_to_feature_matrix(gen_eval_metrics)
    ref_feats = metrics_to_feature_matrix(ref_eval_metrics)

    # Subsample for memory efficiency
    if args.mmd_subsample > 0:
        rng = np.random.default_rng(seed=42)
        orig_sizes = (gen_feats.shape[0], ref_feats.shape[0])
        gen_feats = subsample_features(gen_feats, args.mmd_subsample, rng)
        ref_feats = subsample_features(ref_feats, args.mmd_subsample, rng)
        new_sizes = (gen_feats.shape[0], ref_feats.shape[0])
        if orig_sizes != new_sizes:
            print(f"  Subsampled for MMD: gen={new_sizes[0]}, ref={new_sizes[1]}")

    # Standardize features before MMD computation
    gen_feats, ref_feats = standardize_features(gen_feats, ref_feats)

    mmd_gen_ref = compute_mmd_squared(gen_feats, ref_feats)

    # 6. Assemble results
    results = R12EvaluationResults(
        wasserstein_clone_fraction=wasserstein_results["clone_fraction"],
        wasserstein_clone_depth=wasserstein_results["clone_tree_depth"],
        wasserstein_clone_leaves=wasserstein_results["num_clone_leaves"],
        wasserstein_avg_mutations=wasserstein_results["avg_mutations_per_clone"],
        mmd_gen_ref=mmd_gen_ref,
        num_generated=len(generated_trees),
        num_reference=len(reference_trees),
    )

    # 7. Output results
    print_results(results)
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
