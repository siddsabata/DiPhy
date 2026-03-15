#!/usr/bin/env python3
"""Generate dataset statistics and figures for the DiPhy paper.

This script analyzes the training dataset and produces:
1. Overall dataset statistics table (LaTeX format)
2. Per-regime breakdown table (LaTeX format)
3. Node count distribution histograms
4. Clone fraction vs mutation fraction scatter plots
5. Tree depth distribution by regime family

Usage:
    python dataops/paper_statistics.py --data data/data_100.pkl --output paper/figures/

The output directory will contain:
    - dataset_stats.tex: Overall statistics table
    - regime_stats.tex: Per-regime breakdown table
    - node_count_histogram.pdf: Node count distributions
    - clone_mutation_scatter.pdf: Clone vs mutation counts
    - depth_by_regime.pdf: Tree depth distributions
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Regime family mappings
REGIME_FAMILIES = {
    "Single-Site": ["R01", "R02", "R03", "R04", "R05"],
    "Three-Site Metastasis": ["R06", "R07"],
    "Five+-Site Metastasis": ["R08", "R09", "R10", "R11"],
    "Early Detection": ["R12"],
}

REGIME_NAMES = {
    "R01": "Single-site, near-neutral",
    "R02": "Single-site, strong selection",
    "R03": "Single-site, region-driven",
    "R04": "Single-site, hybrid CNA/SNV",
    "R05": "Single-site, high CIN + WGD",
    "R06": "3-site, shared landscape",
    "R07": "3-site, distance-based",
    "R08": "5-site, high migration",
    "R09": "5-site, genotype migration",
    "R10": "5-site, organotropism",
    "R11": "7-site, sparse sampling",
    "R12": "Small trees, early detection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics for DiPhy paper."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/data_100.pkl"),
        help="Path to the dataset pickle file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/"),
        help="Output directory for figures and tables",
    )
    return parser.parse_args()


def extract_regime(tree_id: str) -> str:
    """Extract regime identifier from tree_id (e.g., 'R01/tumor_001/...' -> 'R01')."""
    match = re.match(r"(R\d+)", tree_id)
    return match.group(1) if match else "Unknown"


def compute_tree_statistics(tree: Dict[str, Any]) -> Dict[str, float]:
    """Compute statistics for a single tree."""
    X = np.array(tree["X"])
    E = np.array(tree["E"])

    n_nodes = len(X)
    n_root = int(np.sum(X == 0))
    n_clones = int(np.sum(X == 1))
    n_mutations = int(np.sum(X == 2))

    # Count edges (upper triangle only since symmetric)
    upper = np.triu(E, k=1)
    n_clone_edges = int(np.sum(upper == 1))
    n_mutation_edges = int(np.sum(upper == 2))

    # Clone fraction
    clone_fraction = n_clones / n_nodes if n_nodes > 0 else 0

    # Mutations per clone
    mutations_per_clone = n_mutations / n_clones if n_clones > 0 else 0

    # Compute tree depth (max distance from root in clone backbone)
    depth = compute_clone_depth(X, E)

    # Clone branching factor (average children per non-leaf clone)
    branching = compute_branching_factor(X, E)

    return {
        "n_nodes": n_nodes,
        "n_clones": n_clones,
        "n_mutations": n_mutations,
        "n_clone_edges": n_clone_edges,
        "n_mutation_edges": n_mutation_edges,
        "clone_fraction": clone_fraction,
        "mutations_per_clone": mutations_per_clone,
        "depth": depth,
        "branching_factor": branching,
    }


def compute_clone_depth(X: np.ndarray, E: np.ndarray) -> int:
    """Compute maximum depth of clone tree from root."""
    clone_mask = (X == 0) | (X == 1)  # Root and clones
    clone_indices = np.where(clone_mask)[0]

    if len(clone_indices) == 0:
        return 0

    # Find root
    root_indices = np.where(X == 0)[0]
    if len(root_indices) != 1:
        return 0
    root = root_indices[0]

    # Build adjacency for clone subgraph
    n = len(X)
    clone_adj = defaultdict(list)
    for i in range(n):
        if not clone_mask[i]:
            continue
        for j in range(n):
            if not clone_mask[j]:
                continue
            if E[i, j] == 1:  # Clone edge
                clone_adj[i].append(j)

    # BFS from root
    visited = {root}
    queue = [(root, 0)]
    max_depth = 0

    while queue:
        node, depth = queue.pop(0)
        max_depth = max(max_depth, depth)
        for neighbor in clone_adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return max_depth


def compute_branching_factor(X: np.ndarray, E: np.ndarray) -> float:
    """Compute average branching factor of clone nodes."""
    clone_mask = (X == 0) | (X == 1)  # Root and clones
    clone_indices = np.where(clone_mask)[0]

    if len(clone_indices) <= 1:
        return 0.0

    # Count clone neighbors for each clone
    degrees = []
    for i in clone_indices:
        clone_neighbors = 0
        for j in clone_indices:
            if i != j and E[i, j] == 1:
                clone_neighbors += 1
        if clone_neighbors > 0:  # Only non-leaf nodes
            degrees.append(clone_neighbors)

    return np.mean(degrees) if degrees else 0.0


def generate_overall_stats_table(
    all_stats: List[Dict[str, float]], output_path: Path
) -> None:
    """Generate LaTeX table with overall dataset statistics."""

    n_graphs = len(all_stats)

    # Extract arrays
    nodes = [s["n_nodes"] for s in all_stats]
    clones = [s["n_clones"] for s in all_stats]
    mutations = [s["n_mutations"] for s in all_stats]
    depths = [s["depth"] for s in all_stats]
    clone_fracs = [s["clone_fraction"] for s in all_stats]
    mut_per_clone = [s["mutations_per_clone"] for s in all_stats]

    latex = r"""\begin{table}[h]
\centering
\caption{Overall dataset statistics across all regimes.}
\label{tab:dataset_overall}
\begin{tabular}{lcccc}
\toprule
\textbf{Statistic} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
"""

    for name, values in [
        ("Total Nodes", nodes),
        ("Clone Nodes", clones),
        ("Mutation Nodes", mutations),
        ("Tree Depth", depths),
        ("Clone Fraction", clone_fracs),
        ("Mutations/Clone", mut_per_clone),
    ]:
        arr = np.array(values)
        latex += f"{name} & {np.mean(arr):.1f} & {np.std(arr):.1f} & {int(np.min(arr))} & {int(np.max(arr))} \\\\\n"

    latex += r"""\midrule
Total Graphs & \multicolumn{4}{c}{""" + str(n_graphs) + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""

    output_path.write_text(latex)
    print(f"Wrote overall stats table to {output_path}")


def generate_regime_stats_table(
    stats_by_regime: Dict[str, List[Dict[str, float]]], output_path: Path
) -> None:
    """Generate LaTeX table with per-regime statistics."""

    latex = r"""\begin{table}[h]
\centering
\small
\caption{Per-regime dataset statistics.}
\label{tab:dataset_regimes}
\begin{tabular}{lcccccc}
\toprule
\textbf{Regime} & \textbf{Count} & \textbf{Nodes} & \textbf{Clones} & \textbf{Mutations} & \textbf{Depth} & \textbf{Clone Frac} \\
\midrule
"""

    for regime in sorted(stats_by_regime.keys()):
        stats = stats_by_regime[regime]
        n = len(stats)

        nodes = np.mean([s["n_nodes"] for s in stats])
        clones = np.mean([s["n_clones"] for s in stats])
        mutations = np.mean([s["n_mutations"] for s in stats])
        depth = np.mean([s["depth"] for s in stats])
        clone_frac = np.mean([s["clone_fraction"] for s in stats])

        latex += f"{regime} & {n} & {nodes:.1f} & {clones:.1f} & {mutations:.1f} & {depth:.1f} & {clone_frac:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path.write_text(latex)
    print(f"Wrote regime stats table to {output_path}")


def plot_node_count_histogram(
    all_stats: List[Dict[str, float]],
    stats_by_regime: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    """Create node count distribution histograms."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall distribution
    nodes = [s["n_nodes"] for s in all_stats]
    axes[0].hist(nodes, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(np.median(nodes), color="red", linestyle="--", label=f"Median: {np.median(nodes):.0f}")
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Node Count Distribution")
    axes[0].legend()

    # By regime family
    family_colors = plt.cm.tab10(np.linspace(0, 1, len(REGIME_FAMILIES)))
    for idx, (family, regimes) in enumerate(REGIME_FAMILIES.items()):
        family_nodes = []
        for regime in regimes:
            if regime in stats_by_regime:
                family_nodes.extend([s["n_nodes"] for s in stats_by_regime[regime]])
        if family_nodes:
            axes[1].hist(
                family_nodes, bins=30, alpha=0.5, label=f"{family} (n={len(family_nodes)})",
                color=family_colors[idx]
            )

    axes[1].set_xlabel("Number of Nodes")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Node Count by Regime Family")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved node count histogram to {output_path}")


def plot_clone_mutation_scatter(
    stats_by_regime: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    """Create clone count vs mutation count scatter plot."""

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, 12))

    for idx, regime in enumerate(sorted(stats_by_regime.keys())):
        stats = stats_by_regime[regime]
        clones = [s["n_clones"] for s in stats]
        mutations = [s["n_mutations"] for s in stats]

        ax.scatter(
            clones, mutations,
            alpha=0.3,
            label=regime,
            color=colors[idx],
            s=20
        )

    ax.set_xlabel("Number of Clones")
    ax.set_ylabel("Number of Mutations")
    ax.set_title("Clone vs Mutation Counts by Regime")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved clone-mutation scatter to {output_path}")


def plot_depth_by_regime(
    stats_by_regime: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> None:
    """Create tree depth distribution by regime."""

    fig, ax = plt.subplots(figsize=(10, 5))

    regimes = sorted(stats_by_regime.keys())
    depths_data = []

    for regime in regimes:
        depths = [s["depth"] for s in stats_by_regime[regime]]
        depths_data.append(depths)

    bp = ax.boxplot(depths_data, labels=regimes, patch_artist=True)

    # Color by family
    regime_to_family = {}
    for family, regs in REGIME_FAMILIES.items():
        for r in regs:
            regime_to_family[r] = family

    family_colors = {
        "Single-Site": "#1f77b4",
        "Three-Site Metastasis": "#ff7f0e",
        "Five+-Site Metastasis": "#2ca02c",
        "Early Detection": "#d62728",
    }

    for idx, regime in enumerate(regimes):
        family = regime_to_family.get(regime, "Unknown")
        color = family_colors.get(family, "gray")
        bp["boxes"][idx].set_facecolor(color)
        bp["boxes"][idx].set_alpha(0.6)

    ax.set_xlabel("Regime")
    ax.set_ylabel("Tree Depth")
    ax.set_title("Tree Depth Distribution by Regime")
    ax.tick_params(axis="x", rotation=45)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, alpha=0.6, label=family)
        for family, color in family_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved depth by regime plot to {output_path}")


def main() -> None:
    args = parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data}")
    with open(args.data, "rb") as f:
        trees = pickle.load(f)

    print(f"Loaded {len(trees)} trees")

    # Compute statistics
    print("Computing statistics...")
    all_stats = []
    stats_by_regime = defaultdict(list)

    for tree in trees:
        stats = compute_tree_statistics(tree)
        regime = extract_regime(tree.get("tree_id", ""))
        stats["regime"] = regime
        all_stats.append(stats)
        stats_by_regime[regime].append(stats)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    generate_overall_stats_table(all_stats, args.output / "dataset_stats.tex")
    generate_regime_stats_table(stats_by_regime, args.output / "regime_stats.tex")
    plot_node_count_histogram(all_stats, stats_by_regime, args.output / "node_count_histogram.pdf")
    plot_clone_mutation_scatter(stats_by_regime, args.output / "clone_mutation_scatter.pdf")
    plot_depth_by_regime(stats_by_regime, args.output / "depth_by_regime.pdf")

    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total graphs: {len(all_stats)}")
    print(f"Regimes: {len(stats_by_regime)}")
    print("\nPer-regime counts:")
    for regime in sorted(stats_by_regime.keys()):
        print(f"  {regime}: {len(stats_by_regime[regime])}")

    print("\nOverall statistics:")
    nodes = [s["n_nodes"] for s in all_stats]
    clones = [s["n_clones"] for s in all_stats]
    depths = [s["depth"] for s in all_stats]
    print(f"  Nodes: mean={np.mean(nodes):.1f}, median={np.median(nodes):.0f}, range=[{min(nodes)}, {max(nodes)}]")
    print(f"  Clones: mean={np.mean(clones):.1f}, median={np.median(clones):.0f}")
    print(f"  Depth: mean={np.mean(depths):.1f}, max={max(depths)}")


if __name__ == "__main__":
    main()
