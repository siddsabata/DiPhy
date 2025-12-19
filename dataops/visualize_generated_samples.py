"""Visualise random valid and invalid trees from a generated samples pickle.

The aggregated pickle produced by ``aggregate_generated_samples.py`` stores
lists of dictionaries with ``X`` node types, ``E`` adjacency matrices, and a
``validity`` mapping mirroring ``tree_metrics.ValidityResult``.  This helper
selects ``n`` valid and ``n`` invalid trees at random, renders them via the
existing ``visualize.plot_phylogeny`` helper, and either saves the figures or
shows them interactively.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from visualize import plot_phylogeny


def _load_trees(pickle_path: Path) -> List[Dict[str, Any]]:
    """Load the list of tree dictionaries from ``pickle_path``."""

    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, list):
        raise TypeError(f"Expected a list of trees in {pickle_path}, found {type(data)!r}.")

    return data


def _filter_trees(trees: Iterable[Dict[str, Any]], valid: bool) -> List[Dict[str, Any]]:
    """Select trees whose validity flag matches ``valid``."""

    filtered: List[Dict[str, Any]] = []
    for tree in trees:
        validity = tree.get("validity", {})
        if not isinstance(validity, dict):
            continue
        if bool(validity.get("is_valid")) == valid:
            filtered.append(tree)
    return filtered


def _prepare_labels(tree: Dict[str, Any], node_count: int) -> Sequence[Any]:
    """Return labels for plotting, defaulting to simple indices when missing."""

    labels = tree.get("L")
    if labels is None or len(labels) != node_count:
        return np.arange(node_count)
    return labels


def _slugify(value: str) -> str:
    """Transform ``value`` into a filesystem-friendly slug."""

    slug = value.replace("/", "-").replace(" ", "_")
    return "".join(ch for ch in slug if ch.isalnum() or ch in {"-", "_"}) or "tree"


def _tree_identifier(tree: Dict[str, Any], fallback_index: int) -> str:
    """Return a readable identifier for logging and filenames."""

    tree_id = tree.get("tree_id")
    if isinstance(tree_id, str) and tree_id:
        return tree_id
    return f"index-{fallback_index}"


def _select_random(trees: Sequence[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample ``count`` unique trees from ``trees`` using ``rng``."""

    if len(trees) < count:
        raise ValueError(f"Requested {count} trees but only found {len(trees)} candidates.")
    return rng.sample(list(trees), count)


def _visualise_group(
    trees: Sequence[Dict[str, Any]],
    label: str,
    output_dir: Path | None,
    show_plots: bool,
) -> None:
    """Render ``trees`` and optionally write them to ``output_dir``."""

    for idx, tree in enumerate(trees, start=1):
        X = np.asarray(tree["X"], dtype=int)
        E = np.asarray(tree["E"], dtype=int)
        labels = _prepare_labels(tree, len(X))

        tree_name = _tree_identifier(tree, idx)
        title = f"{label.capitalize()} tree - {tree_name}"
        save_path = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            slug = _slugify(f"{label}_{idx:03d}_{tree_name}")
            save_path = output_dir / f"{slug}.png"

        # Use the existing plotting helper for consistency across the project.
        plot_phylogeny(X, E, labels, save_path=save_path, title=title, show=show_plots)


def _build_cli() -> argparse.ArgumentParser:
    """Create the command-line parser so the module doubles as a script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pickle_path",
        type=Path,
        help="Path to the aggregated tree pickle containing X, E, and validity entries.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of valid and invalid trees to visualise (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to save the generated figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic random sampling.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots interactively. Without this flag, only files are saved.",
    )
    return parser


def main() -> None:
    """CLI entry point: load data, sample trees, and render them."""

    parser = _build_cli()
    args = parser.parse_args()

    if not args.show and args.output_dir is None:
        raise ValueError("Provide --output-dir or enable --show to see the plots.")

    pickle_path: Path = args.pickle_path.resolve()
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Could not find pickle at {pickle_path}.")

    trees = _load_trees(pickle_path)

    valid_trees = _filter_trees(trees, valid=True)
    invalid_trees = _filter_trees(trees, valid=False)

    rng = random.Random(args.seed)
    selected_valid = _select_random(valid_trees, args.count, rng)
    selected_invalid = _select_random(invalid_trees, args.count, rng)

    _visualise_group(
        selected_valid,
        label="valid",
        output_dir=args.output_dir,
        show_plots=args.show,
    )

    _visualise_group(
        selected_invalid,
        label="invalid",
        output_dir=args.output_dir,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()


