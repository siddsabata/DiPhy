"""
train.py - Training script for DiPhy discrete diffusion model.

Usage:
    python train.py
    python train.py dataset.data_path=/path/to/data.pkl
    python train.py train.n_epochs=1000 general.gpus=2

For SLURM:
    sbatch scripts/train.slurm
"""

import os
import warnings

import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

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


def build_callbacks(cfg: DictConfig):
    """Create checkpoint callbacks for saving best models."""
    callbacks = []

    if cfg.train.save_model:
        # Save top-k best models by validation NLL
        best_checkpoint = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='best-{epoch}-{val_epoch_NLL:.4f}',
            monitor='val/epoch_NLL',
            save_top_k=3,
            mode='min',
            every_n_epochs=1
        )
        callbacks.append(best_checkpoint)

        # Save last checkpoint for potential resume
        last_checkpoint = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='last',
            every_n_epochs=1
        )
        callbacks.append(last_checkpoint)

    # Optional EMA callback
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    return callbacks


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    """Main training entry point."""
    # Create output directories
    utils.create_folders(cfg)

    # Debug mode warning
    if cfg.general.name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run.")

    # Setup model and data
    datamodule, model = setup_model(cfg)

    # Build trainer
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=cfg.train.progress_bar,
        callbacks=build_callbacks(cfg),
        log_every_n_steps=cfg.general.log_every_steps,
        logger=[]
    )

    # Train (no automatic test phase)
    trainer.fit(model, datamodule=datamodule)

    print(f"\n[train.py] Training complete!")
    print(f"[train.py] Checkpoints saved to: checkpoints/{cfg.general.name}/")


if __name__ == '__main__':
    main()
