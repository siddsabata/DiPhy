#!/usr/bin/env python3
"""
Generate a single tumor checkpoint using SISTEM.

This script:
1. Loads configuration parameters from command-line arguments
2. Initializes SISTEM components (library, anatomy, simulator)
3. Runs agent-based tumor growth simulation
4. Saves the resulting checkpoint for later resampling
"""
import argparse
import json
from pathlib import Path

from sistem import GrowthSimulator, Parameters
from sistem.anatomy import SimpleAnatomy
from sistem.selection import RandomRegionLibrary


def main():
    """Generate a single tumor and save its checkpoint."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate tumor checkpoint')
    parser.add_argument('--tumor-id', type=int, required=True, help='Tumor ID')
    parser.add_argument('--param-set-id', type=str, required=True, help='Parameter set ID')
    parser.add_argument('--sistem-params', type=str, required=True, help='JSON-encoded SISTEM parameters')
    args = parser.parse_args()

    # Parse SISTEM parameters from JSON
    sistem_params = json.loads(args.sistem_params)

    # Setup output directory for this tumor
    # Directory structure: output/{param_set_id}/tumors/tumor_{id:03d}/
    tumor_dir = Path('output') / args.param_set_id / 'tumors' / f"tumor_{args.tumor_id:03d}"
    tumor_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tumor {args.tumor_id:03d}] Starting generation...")

    # Initialize SISTEM Parameters object
    # The out_dir parameter tells SISTEM where to write any outputs
    params = Parameters(out_dir=str(tumor_dir), **sistem_params)
    
    # Initialize the random region library
    # This library defines driver genes and their selection coefficients
    library = RandomRegionLibrary(params=params)
    library.initialize(params=params)
    
    # Initialize the anatomical structure
    # SimpleAnatomy represents a simple tumor site structure
    anatomy = SimpleAnatomy(library, params=params)
    
    # Create the growth simulator
    # This is the main simulation engine
    gs = GrowthSimulator(anatomy)
    
    # Run the agent-based tumor growth simulation
    # This simulates cell division, mutation, and selection over time
    # This is typically the longest-running step (can take 30min-1hr per tumor)
    gs.simulate_agents(params=params)
    
    # Save the checkpoint file
    # This checkpoint contains the complete tumor state and can be
    # loaded later for resampling without re-running the simulation
    checkpoint_path = tumor_dir / 'gs.pkl'
    gs.save_checkpoint(str(checkpoint_path))
    
    print(f"[Tumor {args.tumor_id:03d}] Complete! Saved to {checkpoint_path}")


if __name__ == '__main__':
    main()

