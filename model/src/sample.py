"""
sample.py - Run test phase on a trained DiPhy model.

Loads a checkpoint, rebuilds the model from the run's Hydra config,
and runs trainer.test() which computes test metrics and generates samples.

Usage:
    python model/src/sample.py --checkpoint /path/to/checkpoints/best.ckpt
    python model/src/sample.py --checkpoint /path/to/checkpoints/best.ckpt --device cpu

Output is saved to the run's directory (alongside the checkpoint):
    /path/to/generated_samples/samples.pkl

Sample counts are controlled by config values:
    - general.final_model_samples_to_generate
    - general.final_model_samples_to_save
    - general.final_model_chains_to_save

For SLURM:
    sbatch model/scripts/sample.slurm
"""

import argparse
import os
import sys
from pathlib import Path

# Add model/ directory to path so 'src' is importable
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from src.datasets.phylo_dataset import PhyloGraphDataModule, PhyloDatasetInfos
from src.analysis.visualization import NonMolecularVisualization
from src.analysis.phylo_utils import PhyloSamplingMetrics
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.diffusion.extra_features import DummyExtraFeatures
from src.models.diffusion_model import DiscreteDenoisingDiffusion


def main():
    parser = argparse.ArgumentParser(
        description='Run test phase on a trained DiPhy model'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda/cpu, default: cuda)'
    )
    args = parser.parse_args()

    # Derive paths from checkpoint
    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent.parent  # .../checkpoints/best.ckpt -> .../
    config_path = run_dir / '.hydra' / 'config.yaml'

    print(f"[sample.py] Checkpoint: {checkpoint_path}")
    print(f"[sample.py] Run directory: {run_dir}")
    print(f"[sample.py] Config: {config_path}")

    # Load Hydra config from run directory
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Validate dataset exists
    data_path = cfg.dataset.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"This checkpoint was trained with this dataset path. "
            f"Ensure the dataset is accessible."
        )
    print(f"[sample.py] Dataset: {data_path}")

    # Check device
    use_gpu = args.device == 'cuda' and torch.cuda.is_available()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[sample.py] CUDA not available, falling back to CPU")
        use_gpu = False

    # Build datamodule and model components
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

    # Load model from checkpoint
    print(f"[sample.py] Loading model from checkpoint...")
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        str(checkpoint_path),
        weights_only=False,
        **model_kwargs
    )
    model.output_dir = str(run_dir)  # For saving generated samples

    # Build trainer
    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1,
        logger=False,  # No logging during test
        enable_progress_bar=True,
    )

    # Run test phase
    print(f"[sample.py] Running test phase...")
    print(f"[sample.py] Will generate {cfg.general.final_model_samples_to_generate} samples")
    trainer.test(model, datamodule=datamodule)

    print(f"[sample.py] Done! Samples saved to: {run_dir / 'generated_samples'}")


if __name__ == '__main__':
    main()
