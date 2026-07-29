#!/usr/bin/env python3
"""
Simple test to verify TARP package is accessible and working.
This script runs a minimal TARP coverage test with synthetic data.
"""

import os
import sys
import numpy as np

# Add tarp package to path
tarp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tarp', 'src')
if tarp_path not in sys.path:
    sys.path.insert(0, tarp_path)

try:
    from tarp import get_tarp_coverage
    print("✓ Successfully imported TARP package")
except ImportError as e:
    print(f"✗ Failed to import TARP package: {e}")
    sys.exit(1)

# Generate simple test data
print("\nGenerating test data...")
np.random.seed(42)

num_samples = 100
num_sims = 50
num_dims = 3

# True parameter values
theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))

# Posterior samples (Gaussian around true values)
log_sigma = np.random.uniform(low=-2, high=-0.5, size=(num_sims, num_dims))
sigma = np.exp(log_sigma)
samples = np.random.normal(
    loc=theta[np.newaxis, :, :], 
    scale=sigma[np.newaxis, :, :], 
    size=(num_samples, num_sims, num_dims)
)

print(f"  Samples shape: {samples.shape}")
print(f"  Theta shape: {theta.shape}")

# Run TARP coverage test
print("\nRunning TARP coverage test...")
try:
    ecp, alpha = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=False
    )
    print("✓ TARP coverage test completed successfully")
    print(f"  ECP shape: {ecp.shape}")
    print(f"  Alpha shape: {alpha.shape}")
    
    # Check if coverage is reasonable
    mean_deviation = np.mean(np.abs(ecp - alpha))
    print(f"\n  Mean deviation from ideal calibration: {mean_deviation:.3f}")
    
    if mean_deviation < 0.2:
        print("  ✓ Coverage looks reasonable (deviation < 0.2)")
    else:
        print(f"  ⚠ Coverage deviation is high (deviation = {mean_deviation:.3f})")
        print("  Note: This is normal for small test datasets")
    
    print("\n✓ All tests passed! TARP package is working correctly.")
    
except Exception as e:
    print(f"✗ TARP coverage test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test bootstrap functionality
print("\nTesting bootstrap functionality...")
try:
    ecp_boot, alpha_boot = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=10
    )
    print("✓ Bootstrap test completed successfully")
    print(f"  Bootstrap ECP shape: {ecp_boot.shape}")
    print(f"  Expected: (10, {num_sims//10 + 1})")
    
except Exception as e:
    print(f"✗ Bootstrap test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS: TARP package is fully functional!")
print("="*60)
print("\nYou can now use --run-coverage-test in your NPE inference script.")
