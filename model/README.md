# Discrete Diffusion Model

Graph transformer with discrete denoising diffusion for phylogenetic tree generation.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Training

```bash
# Quick test (5 epochs, 4 layers)
python model/src/train.py --config-name=dev

# Full training
python model/src/train.py --config-name=phylo

# Override parameters
python model/src/train.py train.n_epochs=1000 general.gpus=2 dataset.data_path=/path/to/data.pkl
```

## Sampling

```bash
python model/src/sample.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --num_samples 100 \
    --batch_size 32 \
    --output_dir ./samples
```

## Configuration

Edit `model/configs/phylo.yaml` or override from command line:

| Parameter | Description |
|-----------|-------------|
| `dataset.data_path` | Path to pickle file |
| `general.gpus` | Number of GPUs |
| `general.wandb` | Logging: "online", "offline", "disabled" |
| `train.n_epochs` | Training epochs |
| `train.batch_size` | Batch size |
| `model.diffusion_steps` | Diffusion timesteps (default 1000) |
| `model.n_layers` | Transformer layers |

## SLURM (HPC)

```bash
sbatch model/scripts/train.slurm
sbatch model/scripts/sample.slurm
```

Update paths in SLURM scripts before running.

## Architecture

- **Diffusion**: Discrete denoising diffusion on graph structures
- **Backbone**: Graph transformer with XEyTransformerLayer
- **Features**: Separate distributions for nodes (X), edges (E), and global (y)
- **Noise schedule**: Cosine or custom (configurable)

## Output

Checkpoints saved to `checkpoints/<general.name>/`:
- `best.ckpt` - Best validation NLL
- `last.ckpt` - Most recent
- `epoch_*.ckpt` - Periodic snapshots
