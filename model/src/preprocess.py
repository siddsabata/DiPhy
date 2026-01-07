"""
preprocess.py - Pre-process datasets for parallel training.

Pre-processes dataset pickle files into sharded format before running
parallel training jobs. This prevents race conditions when multiple
jobs with the same dataset start simultaneously.

Usage:
    python model/src/preprocess.py model/data/main_filtered.pkl
    python model/src/preprocess.py model/data/*.pkl
"""

import argparse
import os
import sys
from pathlib import Path

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from src.datasets.phylo_dataset import PhyloGraphDataset


def main():
    parser = argparse.ArgumentParser(
        description='Pre-process datasets for parallel training'
    )
    parser.add_argument(
        'data', type=str, nargs='+',
        help='Path(s) to pickle files to process'
    )
    args = parser.parse_args()

    for data_path in args.data:
        path = Path(data_path).resolve()

        if not path.exists():
            print(f"Skipping {path.name}: file not found")
            continue

        print(f"Processing: {path.name}")

        for split in ('train', 'val', 'test'):
            print(f"  {split}...", end=' ', flush=True)
            PhyloGraphDataset(data_path=str(path), split=split)
            print("done")

        cache_dir = path.parent / path.stem
        print(f"  -> {cache_dir}/")


if __name__ == '__main__':
    main()
