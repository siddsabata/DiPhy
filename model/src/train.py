"""
train.py - Training script for DiPhy discrete diffusion model.

Usage:
    python model/src/train.py --config-name=dev
    python model/src/train.py dataset.data_path=/path/to/data.pkl
    python model/src/train.py train.n_epochs=1000 general.gpus=2

For SLURM:
    sbatch model/scripts/train.slurm
"""

import os
import sys
import warnings
from typing import Optional

# Add model/ directory to path so 'src' is importable
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.warnings import PossibleUserWarning


@rank_zero_only
def print_rank_zero(msg):
    """Print only on rank 0 (for DDP compatibility)."""
    print(msg)

from src import utils
from src.datasets.phylo_dataset import PhyloGraphDataModule, PhyloDatasetInfos
from src.analysis.visualization import NonMolecularVisualization
from src.analysis.phylo_utils import PhyloSamplingMetrics
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.diffusion.extra_features import DummyExtraFeatures
from src.models.diffusion_model import DiscreteDenoisingDiffusion


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def setup_model(cfg: DictConfig):
    """Initialize datamodule, model, and all required components."""
    # Initialize data module
    datamodule = PhyloGraphDataModule(cfg)
    dataset_infos = PhyloDatasetInfos(datamodule, cfg.dataset)

    # Initialize components
    train_metrics = TrainAbstractMetricsDiscrete()
    visualization_tools = NonMolecularVisualization()
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    # Compute input/output dimensions
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features
    )

    # Build model kwargs
    model_kwargs = {
        'dataset_infos': dataset_infos,
        'train_metrics': train_metrics,
        'sampling_metrics': PhyloSamplingMetrics(datamodule),
        'visualization_tools': visualization_tools,
        'extra_features': extra_features,
        'domain_features': domain_features,
    }

    # Create model
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    return datamodule, model


def build_callbacks(cfg: DictConfig, output_dir: str):
    """Create checkpoint callbacks for saving best models."""
    callbacks = []
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if cfg.train.save_model:
        # 1. Best checkpoint by validation NLL (always keeps the single best)
        best_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best',
            monitor='val/epoch_NLL',
            save_top_k=1,
            mode='min',
        )
        callbacks.append(best_checkpoint)

        # 2. Last checkpoint for resuming (overwrites each epoch)
        last_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='last',
            save_top_k=1,
            every_n_epochs=1,
        )
        callbacks.append(last_checkpoint)

        # 3. Periodic checkpoints (e.g., every 100 epochs for 500 total)
        # Default: 5 checkpoints spread across training
        checkpoint_every_n = getattr(cfg.train, 'checkpoint_every_n_epochs', cfg.train.n_epochs // 5)
        if checkpoint_every_n > 0:
            periodic_checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='epoch{epoch:03d}',
                auto_insert_metric_name=False,  # Prevents "epoch=" prefix
                save_top_k=-1,  # Keep all periodic checkpoints
                every_n_epochs=checkpoint_every_n,
                save_on_train_epoch_end=True,  # Save based on training epochs, not validation
            )
            callbacks.append(periodic_checkpoint)

    # Optional EMA callback
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    return callbacks


def get_resume_checkpoint(cfg: DictConfig, output_dir: str) -> Optional[str]:
    """Determine checkpoint path for resumption.

    Args:
        cfg: Hydra config with train.resume_from_checkpoint option
        output_dir: Current run's output directory

    Returns:
        Path to checkpoint file if resuming, None otherwise
    """
    resume_path = getattr(cfg.train, 'resume_from_checkpoint', None)

    if resume_path is None:
        return None

    if resume_path == "auto":
        # Look for last.ckpt in the current output directory
        last_ckpt = os.path.join(output_dir, "checkpoints", "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt
        print_rank_zero(f"[train.py] No last.ckpt found at {last_ckpt}, starting fresh")
        return None

    # Explicit path provided
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    # Derive expected run directory from checkpoint path
    # /path/run/checkpoints/last.ckpt → /path/run/
    expected_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(resume_path)))

    if os.path.realpath(output_dir) != os.path.realpath(expected_run_dir):
        print_rank_zero(f"[WARNING] Output directory mismatch when resuming!")
        print_rank_zero(f"  Checkpoint's run dir: {expected_run_dir}")
        print_rank_zero(f"  Current output dir:   {output_dir}")
        print_rank_zero(f"  To keep outputs together, add:")
        print_rank_zero(f"    hydra.run.dir={expected_run_dir}")

    return resume_path


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    """Main training entry point."""
    # Get Hydra output directory
    output_dir = HydraConfig.get().runtime.output_dir
    print_rank_zero(f"[train.py] Output directory: {output_dir}")

    # Create output directories inside Hydra's output dir
    utils.create_folders(output_dir)

    # Debug mode warning
    if cfg.general.name == 'debug':
        print_rank_zero("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run.")

    # Setup model and data (pass output_dir for graphs/chains)
    datamodule, model = setup_model(cfg)
    model.output_dir = output_dir  # Store for use in visualization

    # Setup W&B logger (only on rank 0, handled by Lightning)
    wandb_logger = None
    if cfg.general.wandb != 'disabled':
        wandb_logger = WandbLogger(
            project="graph_ddm_phylo",
            name=cfg.general.name,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Build trainer
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else "auto",
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=cfg.train.progress_bar,
        callbacks=build_callbacks(cfg, output_dir),
        log_every_n_steps=cfg.general.log_every_steps,
        logger=wandb_logger if wandb_logger else [],
    )

    # Check for checkpoint to resume from
    ckpt_path = get_resume_checkpoint(cfg, output_dir)
    if ckpt_path:
        print_rank_zero(f"[train.py] Resuming from checkpoint: {ckpt_path}")

    # Train (no automatic test phase)
    # weights_only=False needed for PyTorch 2.6+ to load omegaconf objects in checkpoint
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    print_rank_zero(f"\n[train.py] Training complete!")
    print_rank_zero(f"[train.py] Checkpoints saved to: {os.path.join(output_dir, 'checkpoints')}")


if __name__ == '__main__':
    main()
