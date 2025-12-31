# Data Operations

Data processing pipeline: transforms SISTEM outputs into model-ready datasets.

## Scripts

| Script | Purpose |
|--------|---------|
| build_dataset.py | Aggregate SISTEM outputs into pickle |
| filter_dataset.py | Filter datasets by node count |
| subsample_dataset.py | Create smaller subsets |
| phylogeny.py | Transform Newick+TSV to graph format |
| tree_metrics.py | Compute tree statistics and validity checks |
| visualize_generated_samples.py | Visualize generated trees |

## Usage

### Build dataset from SISTEM output

```bash
python dataops/build_dataset.py \
    --input-root datagen/output/sistem_regimes_v1 \
    --output model/data/dataset.pkl \
    --max-nodes 200
```

### Filter existing dataset

```bash
python dataops/filter_dataset.py \
    --input model/data/dataset.pkl \
    --output model/data/filtered.pkl \
    --max-nodes 200

# Stats only (no output file)
python dataops/filter_dataset.py --input model/data/dataset.pkl --stats-only
```

### Create development subset

```bash
python dataops/subsample_dataset.py \
    --input model/data/dataset.pkl \
    --output model/data/dev.pkl \
    --fraction 0.05 \
    --seed 42
```

### Visualize samples

```bash
python dataops/visualize_generated_samples.py aggregated.pkl \
    --count 10 \
    --output-dir ./figures/
```

## Data Format

**Input** (SISTEM outputs):
```
datagen/output/<run_id>/regimes/<regime>/tumors/<tumor>/
├── clone_tree.nwk     # Newick tree
└── SNV_events.tsv     # Mutation events
```

**Output** (pickle):
```python
[
    {
        "tree_id": "regime/tumor/attempt",
        "X": [0, 1, 1, 2, 2, ...],     # Node types: 0=root, 1=clone, 2=mutation
        "E": [[0, 1, 0, ...], ...],    # Edge matrix: 0=none, 1=clone, 2=mutation
        "L": ["root", "P1", ...]       # Node labels
    },
    ...
]
```

## SLURM (HPC)

```bash
sbatch dataops/build_dataset.slurm
sbatch dataops/filter_dataset.slurm
sbatch dataops/subsample_dataset.slurm
```
