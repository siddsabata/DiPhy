# DiPhy

This work is an adaptation of [DiGress](https://github.com/cvignac/DiGress/tree/main), using [SISTEM](https://github.com/samsonweiner/sistem?tab=readme-ov-file) for synthetic data. 

Unconditional discrete diffusion for generating tumor phylogenies.

## Overview

DiPhy generates synthetic phylogenetic trees using discrete denoising diffusion. The pipeline has three stages:

1. **datagen/** - Simulate tumors with SISTEM to generate phylogenetic data
2. **dataops/** - Process raw outputs into model-ready datasets
3. **model/** - Train and sample from the diffusion model

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Quick Start

See each directory for detailed instructions:

- [datagen/README.md](datagen/README.md) - Data generation
- [dataops/README.md](dataops/README.md) - Data processing
- [model/README.md](model/README.md) - Model training & sampling

## Data Format

Phylogenetic trees are represented as graphs:

- **X**: Node types (0=root, 1=clone, 2=mutation)
- **E**: Edge types (0=none, 1=clone edge, 2=mutation edge)
- **L**: Node labels

## References

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```

```
@article{Weiner2025SISTEMSO,
  title={SISTEM: simulation of tumor evolution, metastasis, and DNA-seq data under genotype-driven selection},
  author={Samson Weiner and Mukul S. Bansal},
  journal={Bioinformatics},
  year={2025},
  volume={41},
  url={https://api.semanticscholar.org/CorpusID:283240234}
}
```