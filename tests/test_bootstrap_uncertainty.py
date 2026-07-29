#!/usr/bin/env python3
"""
Test script to verify that bootstrap uncertainties are reasonable.
This script compares bootstrap variance before and after the fix.
"""

import os
import sys
import numpy as np

# Add tarp package to path
tarp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tarp', 'src')
if tarp_path not in sys.path:
    sys.path.insert(0, tarp_path)

from tarp import get_tarp_coverage

# Generate test data
print("Generating test data...")
np.random.seed(42)

num_samples = 200
num_sims = 100
num_dims = 5

# True parameter values
theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))

# Posterior samples (Gaussian around true values with varying uncertainty)
log_sigma = np.random.uniform(low=-2, high=-0.5, size=(num_sims, num_dims))
sigma = np.exp(log_sigma)
samples = np.random.normal(
    loc=theta[np.newaxis, :, :], 
    scale=sigma[np.newaxis, :, :], 
    size=(num_samples, num_sims, num_dims)
)

print(f"  Samples shape: {samples.shape}")
print(f"  Theta shape: {theta.shape}")

# Run TARP with bootstrap
print("\nRunning TARP coverage test with bootstrap...")
ecp_boot, alpha = get_tarp_coverage(
    samples=samples,
    theta=theta,
    references="random",
    metric="euclidean",
    norm=True,
    bootstrap=True,
    num_bootstrap=50,
    seed=42
)

print(f"  Bootstrap ECP shape: {ecp_boot.shape}")

# Compute statistics
ecp_mean = np.mean(ecp_boot, axis=0)
ecp_std = np.std(ecp_boot, axis=0)

print("\nBootstrap uncertainty statistics:")
print(f"  Mean std across credibility levels: {np.mean(ecp_std):.4f}")
print(f"  Max std: {np.max(ecp_std):.4f}")
print(f"  Min std: {np.min(ecp_std):.4f}")

# Check if uncertainties are reasonable (should be > 0.01 typically)
if np.mean(ecp_std) > 0.01:
    print("\n✓ Bootstrap uncertainties look reasonable!")
    print(f"  Average uncertainty: {np.mean(ecp_std):.4f} (good range: 0.01-0.05)")
elif np.mean(ecp_std) > 0.005:
    print("\n⚠ Bootstrap uncertainties are moderate")
    print(f"  Average uncertainty: {np.mean(ecp_std):.4f}")
    print("  Consider increasing num_bootstrap or num_sims for more reliable estimates")
else:
    print("\n⚠ Bootstrap uncertainties seem too small!")
    print(f"  Average uncertainty: {np.mean(ecp_std):.4f}")
    print("  This might indicate an issue with the bootstrap implementation")

# Show some example values
print("\nExample uncertainties at different credibility levels:")
for i in range(len(alpha)):
    print(f"  α = {alpha[i]:.2f}: mean ECP = {ecp_mean[i]:.3f} ± {ecp_std[i]:.3f}")

# Compare variation across bootstrap samples
print("\nVariation across bootstrap samples:")
variation = np.max(ecp_boot, axis=0) - np.min(ecp_boot, axis=0)
print(f"  Mean range (max-min): {np.mean(variation):.4f}")
print(f"  Max range: {np.max(variation):.4f}")

if np.mean(variation) > 0.05:
    print("\n✓ Good variation across bootstrap samples!")
else:
    print("\n⚠ Limited variation - uncertainties might be underestimated")

print("\n" + "="*60)
print("Bootstrap uncertainty test complete!")
print("="*60)
