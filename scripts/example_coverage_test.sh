#!/bin/bash
#
# Example script demonstrating how to run TARP coverage tests
# for NPE posterior quality assessment
#

# Basic inference with coverage testing
# This will:
# 1. Load or train the NPE model
# 2. Run TARP coverage test on 100 simulations from the training set
# 3. Generate coverage diagnostic plots
# 4. Sample from the posterior for the fiducial observation

python run_npe_inference_auto_cross_ps.py \
    --simulation-type nobaryons \
    --fiducial-type baryonified \
    --bins 1,2,3,4 \
    --lower-cut 30 \
    --upper-cut 1024 \
    --rebin 20 \
    --noisy \
    --noise-level 0.26 \
    --run-coverage-test \
    --coverage-num-sims 100 \
    --coverage-num-samples 1000 \
    --coverage-bootstrap \
    --coverage-num-bootstrap 100 \
    --num-samples 3000 \
    --gpu 3

# Example with fewer bootstrap iterations for faster testing
# python run_npe_inference_auto_cross_ps.py \
#     --simulation-type nobaryons \
#     --fiducial-type baryonified \
#     --bins 1,2,3,4 \
#     --lower-cut 30 \
#     --upper-cut 1024 \
#     --rebin 20 \
#     --noisy \
#     --noise-level 0.26 \
#     --run-coverage-test \
#     --coverage-num-sims 50 \
#     --coverage-num-samples 500 \
#     --num-samples 3000 \
#     --gpu 0

# Example without bootstrap (faster, no uncertainty estimates)
# python run_npe_inference_auto_cross_ps.py \
#     --simulation-type nobaryons \
#     --fiducial-type baryonified \
#     --bins 1,2,3,4 \
#     --lower-cut 30 \
#     --upper-cut 1024 \
#     --rebin 20 \
#     --noisy \
#     --noise-level 0.26 \
#     --run-coverage-test \
#     --coverage-num-sims 100 \
#     --coverage-num-samples 1000 \
#     --num-samples 3000 \
#     --gpu 0
