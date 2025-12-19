"""Utility for parsing training logs and generating diagnostic plots.

This script follows a simple pipeline:

1. Parse the raw training log to extract training losses, validation losses,
   and per-validation generated statistics.
2. Persist the parsed data to JSON so it can be reused without re-reading the
   large text file.
3. Load the JSON into pandas DataFrames and generate the requested plots.

Run the module directly to reproduce the JSON file and the figures.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


DATASET_CONFIGS = {
    "marginal": {
        "label": "Marginal sampling",
        "color": "#1f77b4",  # Matplotlib blue
        "log_path": Path(
            "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/marginal/"
            "phylo_full_marginal_updated_viz.out"
        ),
        "json_path": Path(
            "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/marginal/"
            "phylo_full_marginal_updated_viz_metrics.json"
        ),
        "plot_dir": Path(
            "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/marginal/plots"
        ),
    },
    "uniform": {
        "label": "Uniform sampling",
        "color": "#ff7f0e",  # Matplotlib orange
        "log_path": Path(
            "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/uniform/"
            "phylo_full_uniform_updated_viz.out"
        ),
        "json_path": Path(
            "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/uniform/"
            "phylo_full_uniform_updated_viz_metrics.json"
        ),
        "plot_dir": Path(
            "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/uniform/plots"
        ),
    },
}

COMPARISON_PLOT_DIR = Path(
    "/Users/siddharthsabata/dev/research/sistem-transform/training_logs/comparison/plots"
)

DATASET_ORDER = ("marginal", "uniform")


TRAIN_RE = re.compile(
    r"^Epoch (?P<epoch>\d+): X_CE: (?P<x_ce>-?[0-9.]+) -- E_CE: (?P<e_ce>-?[0-9.]+) -- "
    r"y_CE: (?P<y_ce>-?[0-9.]+) -- (?P<duration>-?[0-9.]+)s$"
)
VAL_HEADER_RE = re.compile(r"^Epoch (?P<epoch>\d+): Val NLL")
VAL_LOSS_RE = re.compile(r"^Val loss: (?P<loss>-?[0-9.]+)")
GENERATED_STATS_PREFIX = "[PhyloSamplingMetrics] Generated stats: "


def parse_training_log(path: Path, *, max_epoch: int) -> Dict[str, List[Dict[str, float]]]:
    """Stream the training log and collect the metrics we care about."""

    training_rows: List[Dict[str, float]] = []
    validation_rows: List[Dict[str, float]] = []
    generated_rows: List[Dict[str, float]] = []

    current_val_epoch: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            train_match = TRAIN_RE.match(line)
            if train_match:
                epoch = int(train_match.group("epoch"))
                if epoch > max_epoch:
                    break

                training_rows.append(
                    {
                        "epoch": epoch,
                        "x_ce": float(train_match.group("x_ce")),
                        "e_ce": float(train_match.group("e_ce")),
                        "y_ce": float(train_match.group("y_ce")),
                    }
                )
                continue

            val_header_match = VAL_HEADER_RE.match(line)
            if val_header_match:
                current_val_epoch = int(val_header_match.group("epoch"))
                if current_val_epoch > max_epoch:
                    break
                continue

            val_loss_match = VAL_LOSS_RE.match(line)
            if val_loss_match and current_val_epoch is not None:
                validation_rows.append(
                    {
                        "epoch": current_val_epoch,
                        "val_loss": float(val_loss_match.group("loss")),
                    }
                )
                continue

            if line.startswith(GENERATED_STATS_PREFIX) and current_val_epoch is not None:
                payload = line[len(GENERATED_STATS_PREFIX) :]
                try:
                    stats = ast.literal_eval(payload)
                except (SyntaxError, ValueError) as error:
                    raise ValueError(
                        f"Failed to parse generated stats at epoch {current_val_epoch}: {payload}"
                    ) from error

                stats_row = {"epoch": current_val_epoch}
                for key in (
                    "mean_nodes",
                    "mean_edges",
                    "mean_clone_fraction",
                    "mean_mutation_fraction",
                    "validity_pass_pct",
                ):
                    stats_row[key] = float(stats[key])
                generated_rows.append(stats_row)

    return {
        "training": training_rows,
        "validation": validation_rows,
        "generated_stats": generated_rows,
    }


def ensure_plot_directory(path: Path) -> None:
    """Create the directory that will hold the figures."""

    path.mkdir(parents=True, exist_ok=True)


def dump_metrics_to_json(metrics: Dict[str, List[Dict[str, float]]], path: Path) -> None:
    """Persist the parsed metrics so future runs can reload them quickly."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_metrics(path: Path) -> Dict[str, List[Dict[str, float]]]:
    """Load the metrics JSON file created earlier."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return rows sorted by epoch for consistent plotting order."""

    return sorted(rows, key=lambda row: row["epoch"])


def lighten_color(color: str, amount: float = 0.35) -> tuple[float, float, float]:
    """Blend ``color`` towards white so we can show related series together."""

    base_r, base_g, base_b = mcolors.to_rgb(color)
    amount = float(max(0.0, min(amount, 1.0)))

    def _towards_white(component: float) -> float:
        return component + (1.0 - component) * amount

    return (_towards_white(base_r), _towards_white(base_g), _towards_white(base_b))


def plot_training_losses_comparison(
    metrics_by_dataset: Dict[str, Dict[str, List[Dict[str, float]]]], output: Path
) -> None:
    """Render marginal vs uniform training losses side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, dataset_key in zip(axes, DATASET_ORDER):
        config = DATASET_CONFIGS[dataset_key]
        label = config["label"]
        base_color = config["color"]

        rows = sort_rows(metrics_by_dataset[dataset_key]["training"])
        epochs = [row["epoch"] for row in rows]
        x_ce = [row["x_ce"] for row in rows]
        e_ce = [row["e_ce"] for row in rows]

        ax.plot(epochs, x_ce, color=base_color, linewidth=2, label="X_CE")
        ax.plot(
            epochs,
            e_ce,
            color=lighten_color(base_color, amount=0.55),
            linewidth=2,
            linestyle="--",
            label="E_CE",
        )

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Cross-entropy")
    fig.suptitle("Training losses per epoch", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.0)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_validation_loss_comparison(
    metrics_by_dataset: Dict[str, Dict[str, List[Dict[str, float]]]], output: Path
) -> None:
    """Render validation losses with one subplot per dataset."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, dataset_key in zip(axes, DATASET_ORDER):
        config = DATASET_CONFIGS[dataset_key]
        label = config["label"]
        color = config["color"]

        rows = sort_rows(metrics_by_dataset[dataset_key]["validation"])
        epochs = [row["epoch"] for row in rows]
        losses = [row["val_loss"] for row in rows]

        ax.plot(epochs, losses, marker="o", linewidth=2, color=color)
        ax.set_title(label)
        ax.set_xlabel("Validation epoch")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Validation loss")
    fig.suptitle("Validation loss across epochs", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.0)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_generated_stats_comparison(
    metrics_by_dataset: Dict[str, Dict[str, List[Dict[str, float]]]], output: Path
) -> None:
    """Render generated stats and validation loss in a 3x2 grid with both datasets overlaid."""

    # Include validation loss as the first metric, then the generated stats
    metrics = [
        ("validation", "val_loss", "Validation loss"),
        ("generated_stats", "mean_nodes", "Mean nodes"),
        ("generated_stats", "mean_edges", "Mean edges"),
        ("generated_stats", "mean_clone_fraction", "Mean clone fraction"),
        ("generated_stats", "mean_mutation_fraction", "Mean mutation fraction"),
        ("generated_stats", "validity_pass_pct", "Validity pass %"),
    ]

    # Create a 3x2 grid (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for ax, (source_key, column, ylabel) in zip(axes_flat, metrics):
        # Plot both marginal and uniform on the same subplot
        for dataset_key in DATASET_ORDER:
            config = DATASET_CONFIGS[dataset_key]
            color = config["color"]

            rows = sort_rows(metrics_by_dataset[dataset_key][source_key])
            epochs = [row["epoch"] for row in rows]
            values = [row[column] for row in rows]

            ax.plot(epochs, values, marker="o", linewidth=2, color=color)

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Set xlabel only on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("Validation epoch")

    fig.suptitle("Validation metrics and generated stats comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5, w_pad=1.2)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    """Execute the parse -> JSON -> plot pipeline."""

    metrics_by_dataset: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for dataset_key in DATASET_ORDER:
        config = DATASET_CONFIGS[dataset_key]
        ensure_plot_directory(config["plot_dir"])

        metrics = parse_training_log(config["log_path"], max_epoch=499)
        dump_metrics_to_json(metrics, config["json_path"])
        metrics_by_dataset[dataset_key] = load_metrics(config["json_path"])

    ensure_plot_directory(COMPARISON_PLOT_DIR)

    plot_training_losses_comparison(
        metrics_by_dataset, COMPARISON_PLOT_DIR / "training_losses_comparison.png"
    )
    plot_validation_loss_comparison(
        metrics_by_dataset, COMPARISON_PLOT_DIR / "validation_loss_comparison.png"
    )
    plot_generated_stats_comparison(
        metrics_by_dataset, COMPARISON_PLOT_DIR / "generated_stats_comparison.png"
    )


if __name__ == "__main__":
    main()


