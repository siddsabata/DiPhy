#!/usr/bin/env python3
"""
Resample a tumor from its checkpoint.

This script:
1. Loads a previously generated tumor checkpoint (gs.pkl)
2. Samples cells from the tumor
3. Simulates single-cell lineage reconstruction
4. Generates output files (tree, CNP profiles, SNV profiles, etc.)

Each resample represents a different random sampling of cells from
the same tumor, simulating biological/technical variability in sampling.
"""
import argparse
import json
from pathlib import Path

try:
    from sistem import Parameters, load_gs
except ImportError:  # pragma: no cover
    from sistem.parameters import Parameters
    from sistem.growth_sim import load_gs


def main():
    """Resample a tumor from its checkpoint and generate outputs."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Resample tumor')
    parser.add_argument('--tumor-id', type=int, required=True, help='Tumor ID')
    parser.add_argument('--replicate-id', type=int, required=True, help='Replicate ID')
    parser.add_argument('--param-set-id', type=str, required=True, help='Parameter set ID')
    parser.add_argument('--sistem-params', type=str, required=True, help='JSON-encoded SISTEM parameters')
    args = parser.parse_args()

    # Parse SISTEM parameters from JSON
    sistem_params = json.loads(args.sistem_params)

    # Setup paths
    # We need to locate the checkpoint file and create the output directory
    base_dir = Path('output') / args.param_set_id

    # Checkpoint location: output/{param_set_id}/tumors/tumor_{id:03d}/gs.pkl
    tumor_dir = base_dir / 'tumors' / f"tumor_{args.tumor_id:03d}"

    # Output location: output/{param_set_id}/resamples/tumor_{id:03d}/rep_{rid:03d}/
    output_dir = base_dir / 'resamples' / f"tumor_{args.tumor_id:03d}" / f"rep_{args.replicate_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the checkpoint file
    # This contains the complete tumor state from the growth simulation
    checkpoint_path = tumor_dir / 'gs.pkl'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    gs = load_gs(str(checkpoint_path))

    # Recreate Parameters object with new output directory
    # This ensures outputs are written to the replicate-specific directory
    params = Parameters(out_dir=str(output_dir), **sistem_params)
    
    # Sample cells from the tumor
    # This randomly selects cells according to the sampling parameters
    gs.sample_cells(params=params)
    
    # Simulate clonal lineage reconstruction because we only need clone-level
    # summaries for downstream analysis.  This mirrors the behaviour in
    # sistem_t3.py and keeps runtimes and SNV counts aligned with expectations.
    # SISTEM automatically writes output files to the specified out_dir:
    # - tree.newick: phylogenetic tree capturing clone structure
    # - observed_CNPs.tsv: copy number profiles per clone
    # - SNV_profiles.tsv: SNV genotypes annotated at the clone level
    # - and other SISTEM-generated files needed by the pipeline
    gs.simulate_clonal_lineage(params=params, out_dir=str(output_dir))
    
    # Note: We don't print completion messages to avoid overwhelming output
    # when running thousands of resamples in parallel


if __name__ == '__main__':
    main()
