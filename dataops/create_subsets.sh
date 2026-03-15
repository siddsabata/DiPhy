#!/bin/bash
# Create 3 regime-balanced subsets from the main dataset
# All subsets have 700 samples for equal dataset sizes.
#
# Subset 1 (regular): 700 samples uniformly from all 12 regimes (~58 each)
# Subset 2 (no_R12): 700 samples uniformly from 11 regimes (~64 each)
# Subset 3 (R1_only): 700 samples from R01 only
#
# Usage: ./create_subsets.sh [input_file] [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-data/data_100.pkl}"
OUTPUT_DIR="${2:-data}"

echo "Creating regime-balanced subsets from: $INPUT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Subset 1: Regular (all 12 regimes) - 700 samples, ~58 per regime
echo "=== Creating regular subset (all 12 regimes) ==="
python "$SCRIPT_DIR/sample_by_regime.py" \
    --input "$INPUT" \
    --output "$OUTPUT_DIR/data_700_regular.pkl" \
    --num-samples 700 \
    --seed 42

echo ""

# Subset 2: No R12 (11 regimes) - 700 samples, ~64 per regime
echo "=== Creating no_R12 subset (11 regimes, exclude R12) ==="
python "$SCRIPT_DIR/sample_by_regime.py" \
    --input "$INPUT" \
    --output "$OUTPUT_DIR/data_700_no_R12.pkl" \
    --num-samples 700 \
    --exclude-regimes R12_small_trees_early_detection_low_sampling \
    --seed 42

echo ""

# Subset 3: R1 only - 700 samples from R01
echo "=== Creating R1_only subset ==="
python "$SCRIPT_DIR/sample_by_regime.py" \
    --input "$INPUT" \
    --output "$OUTPUT_DIR/data_700_R1_only.pkl" \
    --num-samples 700 \
    --include-regimes R01_single_site_arm_neutral \
    --seed 42

echo ""

# Subset 4: R12 only - 700 samples from R12 (for later analyses)
echo "=== Creating R12_only subset ==="
python "$SCRIPT_DIR/sample_by_regime.py" \
    --input "$INPUT" \
    --output "$OUTPUT_DIR/data_700_R12_only.pkl" \
    --num-samples 700 \
    --include-regimes R12_small_trees_early_detection_low_sampling \
    --seed 42

echo ""

# Subset 5: No R1 (11 regimes) - 700 samples, ~64 per regime
echo "=== Creating no_R1 subset (11 regimes, exclude R01) ==="
python "$SCRIPT_DIR/sample_by_regime.py" \
    --input "$INPUT" \
    --output "$OUTPUT_DIR/data_700_no_R1.pkl" \
    --num-samples 700 \
    --exclude-regimes R01_single_site_arm_neutral \
    --seed 42

echo ""
echo "Done! Created 5 subsets of 700 samples each in $OUTPUT_DIR"
