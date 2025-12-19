# SISTEM Pipeline

Minimal tumor simulation pipeline built around SISTEM.

## Build

```bash
docker build -t sistem-pipeline .
```

## Configure

Edit `configs/config.yaml` to modify experiment parameters. The orchestrator loads this file and passes configuration to all worker scripts.

**Example configuration:**
```yaml
experiment:
  param_set_id: "exp_004"        # Experiment identifier (output directory name)
  total_tumors: 100              # Number of tumors to generate
  resamples_per_tumor: 3         # Resamples per tumor

sistem_parameters:
  nsites: 3                      # Number of tumor sites
  epsilon: 1.0e-9                # Numerical precision
  min_detectable: 500000         # Minimum detectable tumor size
  capacities: 10000000           # Maximum tumor capacity
  focal_driver_rate: 0.0005      # Focal driver mutation rate
  SNV_pass_rate: 0.01            # SNV passenger mutation rate
  alter_prop: 0.3                # Proportion of genome alterable
  ncells_prim: 10000             # Cells sampled from primary tumor
  ncells_meta: 5000              # Cells sampled from metastases
  ncells_normal: 2000            # Normal cells sampled
  min_mut_fraction: 0.05         # Minimum mutation frequency threshold
  coverage: 100                  # Sequencing coverage depth
```

**Note:** Legacy JSON configs in `configs/` are kept for reference but are not used by the pipeline.

## Run (Docker)

```bash
# Stage 1: generate tumors
docker run --rm \
  -v $(pwd)/output:/app/output \
  sistem-pipeline \
  conda run -n sistem python orchestrator.py generate --workers 2

# Stage 2: resample available tumors
docker run --rm \
  -v $(pwd)/output:/app/output \
  sistem-pipeline \
  conda run -n sistem python orchestrator.py resample --workers 8
```

## Run (Conda)

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate sistem

# Run the pipeline stages
python orchestrator.py generate --workers 2
python orchestrator.py resample --workers 8
```

**Dependencies:**
- Python 3.11
- SISTEM (simulation engine)
- PyYAML (configuration loading)

## Monitor

Outputs land in `output/<PARAM_SET_ID>/`. Follow progress with:

```bash
tail -f output/<PARAM_SET_ID>/pipeline.log
```

Check tumor checkpoints: `ls output/<PARAM_SET_ID>/tumors/`

Check resamples: `ls output/<PARAM_SET_ID>/resamples/`

