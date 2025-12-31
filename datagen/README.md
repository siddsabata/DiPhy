# Data Generation

Tumor simulation pipeline using SISTEM.

## Scripts

| Script | Purpose |
|--------|---------|
| orchestrator.py | Coordinates 2-stage pipeline (multiprocessing) |
| generate_tumor.py | Single tumor worker |
| resample_tumor.py | Single resample worker |
| build_jobs.py | Expand regimes into job manifest for SLURM |
| run_job.py | SLURM job executor with retries |
| collect_dataset.py | Collect successful tumors, assign splits |

## Build (Docker)

```bash
docker build -t sistem-pipeline .
```

## Configure

Edit `configs/config.yaml`:

```yaml
experiment:
  param_set_id: "exp_004"
  total_tumors: 100
  resamples_per_tumor: 3

sistem_parameters:
  nsites: 3
  epsilon: 1.0e-9
  min_detectable: 500000
  capacities: 10000000
  focal_driver_rate: 0.0005
  SNV_pass_rate: 0.01
  alter_prop: 0.3
  ncells_prim: 10000
  ncells_meta: 5000
  ncells_normal: 2000
  min_mut_fraction: 0.05
  coverage: 100
```

## Run (Simple)

### Docker

```bash
docker run --rm -v $(pwd)/output:/app/output sistem-pipeline \
  conda run -n sistem python orchestrator.py generate --workers 2

docker run --rm -v $(pwd)/output:/app/output sistem-pipeline \
  conda run -n sistem python orchestrator.py resample --workers 8
```

### Conda

```bash
conda env create -f environment.yml
conda activate sistem
python orchestrator.py generate --workers 2
python orchestrator.py resample --workers 8
```

## Run (SLURM - Large Scale)

```bash
# 1. Build job manifest
python datagen/build_jobs.py --config datagen/configs/regimes.yaml

# 2. Submit array
datagen/scripts/submit_generate.sh datagen/configs/regimes.yaml

# 3. Collect results
python datagen/collect_dataset.py --run-id sistem_regimes_v1
```

## Monitor

```bash
tail -f output/<PARAM_SET_ID>/pipeline.log
ls output/<PARAM_SET_ID>/tumors/
ls output/<PARAM_SET_ID>/resamples/
```

## Success Criteria

A tumor is successful if these exist: `gs.pkl`, `SNV_events.tsv`, `clone_tree.nwk`
