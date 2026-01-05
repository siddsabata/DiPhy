"""
sample.py - Generate phylogenetic graphs from a trained DiPhy model.

Usage:
    python model/src/sample.py --checkpoint /path/to/checkpoint.ckpt --num_samples 100
    python model/src/sample.py --checkpoint /path/to/checkpoint.ckpt --num_samples 500 --output_dir ./samples
    python model/src/sample.py --checkpoint /path/to/checkpoint.ckpt --num_samples 100 --batch_size 16

For SLURM:
    sbatch model/scripts/sample.slurm
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple

# Add model/ directory to path so 'src' is importable
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

import torch

from src.datasets.phylo_dataset import PhyloGraphDataModule, PhyloDatasetInfos
from src.analysis.visualization import NonMolecularVisualization
from src.analysis.phylo_utils import PhyloSamplingMetrics
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.diffusion.extra_features import DummyExtraFeatures
from src.models.diffusion_model import DiscreteDenoisingDiffusion


# Type alias for generated graphs
GeneratedGraph = Tuple[torch.Tensor, torch.Tensor]  # (node_types, edge_types)


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded DiscreteDenoisingDiffusion model
        cfg: Configuration used during training
        dataset_infos: Dataset information including node distribution
    """
    print(f"[sample.py] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint to get config
    # weights_only=False needed because checkpoint contains OmegaConf DictConfig objects
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['hyper_parameters']['cfg']

    # Rebuild model components with dataset stats for node distribution
    datamodule = PhyloGraphDataModule(cfg)
    dataset_infos = PhyloDatasetInfos(datamodule, cfg.dataset)

    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features
    )

    model_kwargs = {
        'dataset_infos': dataset_infos,
        'train_metrics': TrainAbstractMetricsDiscrete(),
        'sampling_metrics': PhyloSamplingMetrics(datamodule),
        'visualization_tools': NonMolecularVisualization(),
        'extra_features': extra_features,
        'domain_features': domain_features,
    }

    # Load the model from checkpoint
    # weights_only=False needed because checkpoint contains OmegaConf DictConfig objects
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        checkpoint_path,
        weights_only=False,
        **model_kwargs
    )
    model.to(device)
    model.eval()

    return model, cfg, dataset_infos


def generate_samples(
    model: DiscreteDenoisingDiffusion,
    num_samples: int,
    batch_size: int = 16,
    number_chain_steps: int = 50,
) -> List[GeneratedGraph]:
    """Generate graphs using the trained model.

    Uses the learned node count distribution from the training data.

    Args:
        model: Trained diffusion model
        num_samples: Total number of graphs to generate
        batch_size: Batch size for generation
        number_chain_steps: Number of chain steps to save for visualization

    Returns:
        List of (node_types, edge_types) tuples
    """
    all_samples = []
    samples_remaining = num_samples
    batch_id = 0

    print(f"[sample.py] Generating {num_samples} samples...")

    with torch.no_grad():
        while samples_remaining > 0:
            current_batch_size = min(batch_size, samples_remaining)

            # sample_batch uses model.node_dist to sample node counts
            # This distribution was learned from the training data
            batch_samples = model.sample_batch(
                batch_id=batch_id,
                batch_size=current_batch_size,
                num_nodes=None,  # Use learned distribution
                save_final=0,    # Don't save visualizations
                keep_chain=0,    # Don't keep chains
                number_chain_steps=number_chain_steps,
            )
            all_samples.extend(batch_samples)

            samples_remaining -= current_batch_size
            batch_id += current_batch_size
            print(f"  Generated {len(all_samples)}/{num_samples}", end='\r')

    print(f"\n[sample.py] Generated {len(all_samples)} samples")
    return all_samples


def save_samples(
    samples: List[GeneratedGraph],
    output_path: str,
    output_format: str = 'pickle'
) -> None:
    """Save generated samples to disk.

    Args:
        samples: List of (node_types, edge_types) tuples
        output_path: Path to save the samples
        output_format: 'pickle' or 'txt'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'pickle':
        # Save as list of dicts matching input format
        output_data = []
        for i, (node_types, edge_types) in enumerate(samples):
            output_data.append({
                'tree_id': f'generated_{i}',
                'X': node_types.cpu().numpy().tolist(),
                'E': edge_types.cpu().numpy().tolist(),
            })

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"[sample.py] Saved {len(samples)} samples to {output_path}")

    elif output_format == 'txt':
        with open(output_path, 'w') as f:
            for i, (node_types, edge_types) in enumerate(samples):
                f.write(f"# Sample {i}\n")
                f.write(f"N={node_types.shape[0]}\n")
                f.write("X: " + " ".join(map(str, node_types.cpu().tolist())) + "\n")
                f.write("E:\n")
                for row in edge_types.cpu().tolist():
                    f.write(" ".join(map(str, row)) + "\n")
                f.write("\n")
        print(f"[sample.py] Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate phylogenetic graphs from a trained DiPhy model'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Number of graphs to generate (default: 100)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for sampling (default: 16)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./generated_samples',
        help='Directory to save generated samples (default: ./generated_samples)'
    )
    parser.add_argument(
        '--output_format', type=str, default='pickle',
        choices=['pickle', 'txt'],
        help='Output format for samples (default: pickle)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda/cpu, default: cuda)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[sample.py] Using random seed: {args.seed}")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[sample.py] CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load model
    model, cfg, dataset_infos = load_model(args.checkpoint, device=args.device)

    # Print dataset statistics being used
    print(f"[sample.py] Using node count distribution from training data")
    print(f"[sample.py] Max nodes in distribution: {dataset_infos.max_n_nodes}")

    # Generate samples
    samples = generate_samples(
        model=model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # Save samples
    ext = 'pkl' if args.output_format == 'pickle' else 'txt'
    output_filename = f'samples_{args.num_samples}.{ext}'
    output_path = Path(args.output_dir) / output_filename
    save_samples(samples, output_path, output_format=args.output_format)

    print(f"[sample.py] Done!")


if __name__ == '__main__':
    main()
