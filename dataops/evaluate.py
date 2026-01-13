#!/usr/bin/env python3
"""Evaluation pipeline for phylogenetic tree generation.

Computes distributional metrics comparing generated samples against test set:
- 1D Wasserstein distances for clone fraction, depth, leaves, mutations
- Validity pass rates (overall and per-test)
- MMD ratio: MMD^2(gen, test) / MMD^2(train, test)

Usage:
    python evaluate.py \
        --generated samples.pkl \
        --original-data phylo.pkl \
        --train-split train_index.json \
        --test-split test_index.json \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

from tree_metrics import TreeMetrics, compute_tree_metrics


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """Per-tree metrics for evaluation."""

    tree_id: str
    clone_fraction: float
    clone_tree_depth: int
    num_clone_leaves: int
    avg_mutations_per_clone: float


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    # Wasserstein distances
    wasserstein_clone_fraction: float
    wasserstein_clone_depth: float
    wasserstein_clone_leaves: float
    wasserstein_avg_mutations: float

    # Validity metrics - generated
    gen_validity_overall_pct: float
    gen_no_cycle_pct: float
    gen_root_degree_pct: float
    gen_clone_edge_types_pct: float
    gen_mutation_edge_types_pct: float

    # Validity metrics - test
    test_validity_overall_pct: float
    test_no_cycle_pct: float
    test_root_degree_pct: float
    test_clone_edge_types_pct: float
    test_mutation_edge_types_pct: float

    # MMD
    mmd_ratio: float
    mmd_gen_test: float
    mmd_train_test: float

    # Counts
    num_generated: int
    num_train: int
    num_test: int


# ---------------------------------------------------------------------------
# Per-tree metric computation
# ---------------------------------------------------------------------------


def _to_numpy(tree_dict: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert tree dict to numpy arrays."""
    node_types = np.asarray(tree_dict["X"], dtype=int)
    adjacency = np.asarray(tree_dict["E"], dtype=int)
    return node_types, adjacency


def compute_clone_tree_depth(adjacency: np.ndarray, node_types: np.ndarray) -> int:
    """Compute maximum depth of clone tree using BFS.

    The clone tree consists of:
    - Root node (type 0)
    - Clone nodes (type 1)
    - Connected via clone edges (type 1)

    Returns: max depth (root = depth 0)
    """
    root_indices = np.flatnonzero(node_types == 0)
    if root_indices.size != 1:
        return 0

    root_idx = int(root_indices[0])

    visited: set[int] = set()
    queue: deque[tuple[int, int]] = deque([(root_idx, 0)])
    max_depth = 0

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        max_depth = max(max_depth, depth)

        # Find neighbors via clone edges (type 1)
        clone_neighbors = np.flatnonzero(adjacency[node] == 1)
        for neighbor in clone_neighbors:
            # Only traverse to clone nodes (type 1), not back to root
            if neighbor not in visited and node_types[neighbor] == 1:
                queue.append((neighbor, depth + 1))

    return max_depth


def count_clone_leaves(adjacency: np.ndarray, node_types: np.ndarray) -> int:
    """Count clone nodes that have no clone-edge children.

    A clone is a leaf if:
    - It is type 1 (clone)
    - No clone edges connect it to other clones that are "children"

    Since the tree is undirected, we use BFS from root to determine
    parent-child relationships.
    """
    root_indices = np.flatnonzero(node_types == 0)
    if root_indices.size != 1:
        return 0

    root_idx = int(root_indices[0])
    clone_indices = np.flatnonzero(node_types == 1)

    if clone_indices.size == 0:
        return 0

    # Build parent-child relationships via BFS from root
    visited: set[int] = set()
    children: Dict[int, List[int]] = {i: [] for i in range(len(node_types))}
    queue: deque[int] = deque([root_idx])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        # Clone neighbors (edges of type 1)
        clone_neighbors = np.flatnonzero(adjacency[node] == 1)
        for neighbor in clone_neighbors:
            if neighbor not in visited and node_types[neighbor] in (0, 1):
                children[node].append(neighbor)
                queue.append(neighbor)

    # Count clones with no clone children
    leaf_count = 0
    for clone_idx in clone_indices:
        clone_children = [c for c in children[clone_idx] if node_types[c] == 1]
        if len(clone_children) == 0:
            leaf_count += 1

    return leaf_count


def compute_eval_metrics(tree_dict: Mapping[str, Any]) -> EvalMetrics:
    """Compute evaluation metrics for a single tree."""
    node_types, adjacency = _to_numpy(tree_dict)

    total_nodes = int(node_types.size)
    clone_nodes = int(np.count_nonzero(node_types == 1))

    # Clone fraction
    clone_fraction = clone_nodes / total_nodes if total_nodes > 0 else 0.0

    # Clone tree depth
    clone_tree_depth = compute_clone_tree_depth(adjacency, node_types)

    # Number of clone leaves
    num_clone_leaves = count_clone_leaves(adjacency, node_types)

    # Average mutations per clone
    # Count mutation edges (type 2)
    upper_r, upper_c = np.triu_indices_from(adjacency, k=1)
    upper_values = adjacency[upper_r, upper_c]
    mutation_edges = int(np.count_nonzero(upper_values == 2))
    avg_mutations_per_clone = mutation_edges / clone_nodes if clone_nodes > 0 else 0.0

    return EvalMetrics(
        tree_id=str(tree_dict.get("tree_id", "unknown")),
        clone_fraction=clone_fraction,
        clone_tree_depth=clone_tree_depth,
        num_clone_leaves=num_clone_leaves,
        avg_mutations_per_clone=avg_mutations_per_clone,
    )


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------


def compute_wasserstein_distances(
    gen_metrics: Sequence[EvalMetrics],
    test_metrics: Sequence[EvalMetrics],
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
        test_values = np.array([getattr(m, metric_name) for m in test_metrics])

        if len(gen_values) == 0 or len(test_values) == 0:
            results[metric_name] = float("inf")
        else:
            results[metric_name] = float(wasserstein_distance(gen_values, test_values))

    return results


# ---------------------------------------------------------------------------
# MMD with RBF kernel
# ---------------------------------------------------------------------------


def rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute RBF kernel matrix between X and Y."""
    # X: (n, d), Y: (m, d)
    # Returns: (n, m) kernel matrix
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
    """Compute squared MMD between distributions X and Y.

    Args:
        X: (n, d) feature matrix for first distribution
        Y: (m, d) feature matrix for second distribution
        bandwidth: RBF kernel bandwidth (uses median heuristic if None)

    Returns:
        MMD^2 value
    """
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

    return float(term1 + term2 - term3)


def metrics_to_feature_matrix(metrics: Sequence[EvalMetrics]) -> np.ndarray:
    """Convert list of EvalMetrics to feature matrix for MMD."""
    if not metrics:
        return np.zeros((0, 4))

    features = np.array(
        [
            [
                m.clone_fraction,
                m.clone_tree_depth,
                m.num_clone_leaves,
                m.avg_mutations_per_clone,
            ]
            for m in metrics
        ]
    )
    return features


def subsample_features(
    feats: np.ndarray, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Randomly subsample feature matrix if it exceeds max_samples."""
    if max_samples <= 0 or feats.shape[0] <= max_samples:
        return feats
    indices = rng.choice(feats.shape[0], size=max_samples, replace=False)
    return feats[indices]


def compute_mmd_ratio(
    gen_feats: np.ndarray,
    train_feats: np.ndarray,
    test_feats: np.ndarray,
) -> tuple[float, float, float]:
    """Compute MMD ratio: MMD^2(gen, test) / MMD^2(train, test).

    Returns:
        (mmd_ratio, mmd_gen_test, mmd_train_test)
    """
    # Compute bandwidth from train + test only for stable baseline
    # This ensures MMD²(train, test) is identical across runs with same dataset
    baseline_feats = np.vstack([train_feats, test_feats])
    bandwidth = median_heuristic_bandwidth(baseline_feats, baseline_feats)

    mmd_gen_test = compute_mmd_squared(gen_feats, test_feats, bandwidth)
    mmd_train_test = compute_mmd_squared(train_feats, test_feats, bandwidth)

    # Clamp to non-negative (unbiased estimator can be slightly negative when
    # distributions are nearly identical due to sampling variance)
    mmd_gen_test = max(0.0, mmd_gen_test)
    mmd_train_test = max(0.0, mmd_train_test)

    # Compute ratio, handling zero denominator
    if mmd_train_test < 1e-10:
        ratio = float("inf") if mmd_gen_test > 1e-10 else 1.0
    else:
        ratio = mmd_gen_test / mmd_train_test

    return ratio, mmd_gen_test, mmd_train_test


# ---------------------------------------------------------------------------
# Validity aggregation
# ---------------------------------------------------------------------------


def aggregate_validity_stats(metrics: Sequence[TreeMetrics]) -> Dict[str, float]:
    """Compute overall and per-test validity pass percentages."""
    if not metrics:
        return {
            "overall_pass_pct": 0.0,
            "no_cycle_pass_pct": 0.0,
            "root_degree_pass_pct": 0.0,
            "clone_edge_types_pass_pct": 0.0,
            "mutation_edge_types_pass_pct": 0.0,
        }

    n = len(metrics)

    return {
        "overall_pass_pct": 100.0 * sum(1 for m in metrics if m.validity.is_valid) / n,
        "no_cycle_pass_pct": 100.0
        * sum(1 for m in metrics if not m.validity.has_cycle)
        / n,
        "root_degree_pass_pct": 100.0
        * sum(1 for m in metrics if m.validity.root_has_single_clone_neighbor)
        / n,
        "clone_edge_types_pass_pct": 100.0
        * sum(1 for m in metrics if m.validity.clone_edge_types_valid)
        / n,
        "mutation_edge_types_pass_pct": 100.0
        * sum(1 for m in metrics if m.validity.mutation_edge_types_valid)
        / n,
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_pickle(path: Path) -> List[Dict[str, Any]]:
    """Load a pickle file containing list of tree dicts."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_split_tree_ids(json_path: Path) -> set[str]:
    """Extract tree_ids from a split JSON file."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    tree_ids = set()
    for entry in data.get("entries", []):
        tree_id = entry.get("tree_id")
        if tree_id:
            tree_ids.add(tree_id)

    return tree_ids


def filter_trees_by_ids(
    trees: List[Dict[str, Any]], tree_ids: set[str]
) -> List[Dict[str, Any]]:
    """Filter trees to only include those with matching tree_ids."""
    return [t for t in trees if t.get("tree_id") in tree_ids]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results(results: EvaluationResults, verbose: bool = False) -> None:
    """Print evaluation results to console."""
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\nSample Counts:")
    print(f"  Generated: {results.num_generated}")
    print(f"  Train: {results.num_train}")
    print(f"  Test: {results.num_test}")

    print("\nWasserstein Distances (Generated vs Test):")
    print(f"  Clone Fraction:     {results.wasserstein_clone_fraction:.4f}")
    print(f"  Clone Tree Depth:   {results.wasserstein_clone_depth:.4f}")
    print(f"  Clone Leaves:       {results.wasserstein_clone_leaves:.4f}")
    print(f"  Avg Muts/Clone:     {results.wasserstein_avg_mutations:.4f}")

    print("\nValidity Pass Rates:")
    print(f"{'':22s} {'Generated':>12s} {'Test':>12s}")
    print(
        f"  {'Overall:':<20s} {results.gen_validity_overall_pct:>10.1f}% "
        f"{results.test_validity_overall_pct:>10.1f}%"
    )
    print(
        f"  {'No Cycle:':<20s} {results.gen_no_cycle_pct:>10.1f}% "
        f"{results.test_no_cycle_pct:>10.1f}%"
    )
    print(
        f"  {'Root Degree:':<20s} {results.gen_root_degree_pct:>10.1f}% "
        f"{results.test_root_degree_pct:>10.1f}%"
    )
    print(
        f"  {'Clone Edge Types:':<20s} {results.gen_clone_edge_types_pct:>10.1f}% "
        f"{results.test_clone_edge_types_pct:>10.1f}%"
    )
    print(
        f"  {'Mutation Edge Types:':<20s} {results.gen_mutation_edge_types_pct:>10.1f}% "
        f"{results.test_mutation_edge_types_pct:>10.1f}%"
    )

    print("\nMMD Analysis:")
    print(f"  MMD^2(gen, test):   {results.mmd_gen_test:.6f}")
    print(f"  MMD^2(train, test): {results.mmd_train_test:.6f}")
    print(f"  MMD Ratio:          {results.mmd_ratio:.4f}")

    print("=" * 80)


def save_results(results: EvaluationResults, output_path: Path) -> None:
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(results), handle, indent=2)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated phylogenetic trees against test distribution."
    )
    parser.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="Path to generated samples pickle file (~1000 samples)",
    )
    parser.add_argument(
        "--original-data",
        type=Path,
        required=True,
        help="Path to original dataset pickle file (pre-split)",
    )
    parser.add_argument(
        "--train-split",
        type=Path,
        required=True,
        help="Path to train split JSON file (contains tree_ids)",
    )
    parser.add_argument(
        "--val-split",
        type=Path,
        default=None,
        help="Path to val split JSON file (optional, not used)",
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        required=True,
        help="Path to test split JSON file (contains tree_ids)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results (optional, prints to stdout if not provided)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-metric statistics",
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
    if not args.original_data.exists():
        raise FileNotFoundError(f"Original data not found: {args.original_data}")
    if not args.train_split.exists():
        raise FileNotFoundError(f"Train split not found: {args.train_split}")
    if not args.test_split.exists():
        raise FileNotFoundError(f"Test split not found: {args.test_split}")

    # 1. Load generated samples
    print(f"Loading generated samples from {args.generated}...")
    generated_trees = load_pickle(args.generated)
    print(f"  Loaded {len(generated_trees)} generated samples")

    # 2. Load original data and split by tree_ids
    print(f"Loading original data from {args.original_data}...")
    original_trees = load_pickle(args.original_data)
    print(f"  Loaded {len(original_trees)} original samples")

    print(f"Loading split files...")
    train_ids = load_split_tree_ids(args.train_split)
    test_ids = load_split_tree_ids(args.test_split)
    print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

    train_trees = filter_trees_by_ids(original_trees, train_ids)
    test_trees = filter_trees_by_ids(original_trees, test_ids)
    print(f"  Filtered: Train={len(train_trees)}, Test={len(test_trees)}")

    # 3. Compute evaluation metrics for each set
    print("Computing evaluation metrics...")
    gen_eval_metrics = [compute_eval_metrics(t) for t in generated_trees]
    test_eval_metrics = [compute_eval_metrics(t) for t in test_trees]
    train_eval_metrics = [compute_eval_metrics(t) for t in train_trees]

    # 4. Compute tree metrics for validity
    print("Computing validity metrics...")
    gen_tree_metrics = [compute_tree_metrics(t) for t in generated_trees]
    test_tree_metrics = [compute_tree_metrics(t) for t in test_trees]

    # 5. Compute Wasserstein distances
    print("Computing Wasserstein distances...")
    wasserstein_results = compute_wasserstein_distances(gen_eval_metrics, test_eval_metrics)

    # 6. Compute validity stats
    print("Aggregating validity statistics...")
    gen_validity = aggregate_validity_stats(gen_tree_metrics)
    test_validity = aggregate_validity_stats(test_tree_metrics)

    # 7. Compute MMD ratio
    print("Computing MMD ratio...")
    gen_feats = metrics_to_feature_matrix(gen_eval_metrics)
    train_feats = metrics_to_feature_matrix(train_eval_metrics)
    test_feats = metrics_to_feature_matrix(test_eval_metrics)

    # Subsample for memory efficiency (kernel matrices are O(n²))
    if args.mmd_subsample > 0:
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        orig_sizes = (gen_feats.shape[0], train_feats.shape[0], test_feats.shape[0])
        gen_feats = subsample_features(gen_feats, args.mmd_subsample, rng)
        train_feats = subsample_features(train_feats, args.mmd_subsample, rng)
        test_feats = subsample_features(test_feats, args.mmd_subsample, rng)
        new_sizes = (gen_feats.shape[0], train_feats.shape[0], test_feats.shape[0])
        if orig_sizes != new_sizes:
            print(f"  Subsampled for MMD: gen={new_sizes[0]}, train={new_sizes[1]}, test={new_sizes[2]}")

    mmd_ratio, mmd_gen_test, mmd_train_test = compute_mmd_ratio(
        gen_feats, train_feats, test_feats
    )

    # 8. Assemble results
    results = EvaluationResults(
        wasserstein_clone_fraction=wasserstein_results["clone_fraction"],
        wasserstein_clone_depth=wasserstein_results["clone_tree_depth"],
        wasserstein_clone_leaves=wasserstein_results["num_clone_leaves"],
        wasserstein_avg_mutations=wasserstein_results["avg_mutations_per_clone"],
        gen_validity_overall_pct=gen_validity["overall_pass_pct"],
        gen_no_cycle_pct=gen_validity["no_cycle_pass_pct"],
        gen_root_degree_pct=gen_validity["root_degree_pass_pct"],
        gen_clone_edge_types_pct=gen_validity["clone_edge_types_pass_pct"],
        gen_mutation_edge_types_pct=gen_validity["mutation_edge_types_pass_pct"],
        test_validity_overall_pct=test_validity["overall_pass_pct"],
        test_no_cycle_pct=test_validity["no_cycle_pass_pct"],
        test_root_degree_pct=test_validity["root_degree_pass_pct"],
        test_clone_edge_types_pct=test_validity["clone_edge_types_pass_pct"],
        test_mutation_edge_types_pct=test_validity["mutation_edge_types_pass_pct"],
        mmd_ratio=mmd_ratio,
        mmd_gen_test=mmd_gen_test,
        mmd_train_test=mmd_train_test,
        num_generated=len(generated_trees),
        num_train=len(train_trees),
        num_test=len(test_trees),
    )

    # 9. Output results
    print_results(results, verbose=args.verbose)
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
