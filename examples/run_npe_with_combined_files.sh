#!/bin/bash
# Example: Using combined L1 norm files for NPE inference with Halofit selection
#
# This script shows how to use the combined files from l1_norm_processing.py
# for NPE inference with the Halofit selection criteria.

# Configuration
SCRIPT_DIR="/home/tersenov/software/bar_impact/scripts"
OUTPUT_DIR="/home/tersenov/software/bar_impact/outputs"

# ============================================================================
# Example 1: Training with combined files (Halofit selection)
# ============================================================================
echo "Example 1: Training NPE with combined files (Halofit selection)"
echo "================================================================"

python3 ${SCRIPT_DIR}/run_npe_inference_halofit.py \
    --training-dataset halofit \
    --fiducial-dataset fiducial \
    --simulation-type nobaryons \
    --bins 1,2,3,4 \
    --noisy \
    --noise-level 0.26 \
    --use-combined \
    --train \
    --epochs 2000 \
    --batch-size 40 \
    --learning-rate 1e-4 \
    --save-feature-mask \
    --variance-threshold 1e-10 \
    --output-dir ${OUTPUT_DIR}/plots \
    --samples-dir ${OUTPUT_DIR}/samples

# ============================================================================
# Example 2: Inference only (load existing model)
# ============================================================================
echo ""
echo "Example 2: Inference with existing model"
echo "=========================================="

python3 ${SCRIPT_DIR}/run_npe_inference_halofit.py \
    --training-dataset halofit \
    --fiducial-dataset fiducial \
    --simulation-type nobaryons \
    --bins 1,2,3,4 \
    --noisy \
    --noise-level 0.26 \
    --use-combined \
    --num-samples 5000 \
    --random-seed 42 \
    --output-dir ${OUTPUT_DIR}/plots \
    --samples-dir ${OUTPUT_DIR}/samples

# ============================================================================
# Example 3: Coverage test with combined files
# ============================================================================
echo ""
echo "Example 3: TARP coverage test"
echo "=============================="

python3 ${SCRIPT_DIR}/run_npe_inference_halofit.py \
    --training-dataset halofit \
    --simulation-type nobaryons \
    --bins 1,2,3,4 \
    --noisy \
    --noise-level 0.26 \
    --use-combined \
    --run-coverage-test \
    --coverage-num-sims 200 \
    --coverage-num-samples 1000 \
    --coverage-bootstrap \
    --coverage-num-bootstrap 100 \
    --output-dir ${OUTPUT_DIR}/plots

# ============================================================================
# Example 4: Training with baryonified simulations
# ============================================================================
echo ""
echo "Example 4: Training with baryonified maps"
echo "=========================================="

python3 ${SCRIPT_DIR}/run_npe_inference_halofit.py \
    --training-dataset halofit \
    --fiducial-dataset fiducial \
    --simulation-type baryonified \
    --bins 1,2,3,4 \
    --noisy \
    --noise-level 0.26 \
    --use-combined \
    --train \
    --epochs 2000 \
    --save-feature-mask \
    --output-dir ${OUTPUT_DIR}/plots

echo ""
echo "All examples defined. Uncomment the desired example to run."
