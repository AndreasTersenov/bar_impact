"""
Basic example: Processing a convergence map with BAR_IMPACT

This example demonstrates the core functionality of loading,
processing, and analyzing a cosmological convergence map.
"""

import numpy as np
import healpy as hp
from bar_impact.utils import add_shape_noise
from bar_impact.processing import apply_bnt_transform

# Example 1: Load and add noise to a map
print("Example 1: Loading and adding shape noise")
print("=" * 50)

# Generate a simple test map (in practice, load from file)
nside = 512
npix = hp.nside2npix(nside)
test_map = np.random.randn(npix) * 0.01  # Simulated convergence map

print(f"Created test map with nside={nside}, npix={npix}")
print(f"Map statistics: mean={test_map.mean():.6f}, std={test_map.std():.6f}")

# Add realistic shape noise
noisy_map = add_shape_noise(
    test_map,
    sigma_e=0.26,  # Intrinsic ellipticity dispersion
    nside=nside,
    ngal_arcmin2=30.0,  # Galaxy density
    seed=42
)

print(f"Noisy map statistics: mean={noisy_map.mean():.6f}, std={noisy_map.std():.6f}")
print(f"Noise contribution: {(noisy_map.std()**2 - test_map.std()**2)**0.5:.6f}")
print()

# Example 2: Apply BNT transform
print("Example 2: Applying Band-limited Nulling Transform")
print("=" * 50)

# Create 4 redshift bin maps
n_bins = 4
maps = np.random.randn(n_bins, npix) * 0.01

print(f"Created {n_bins} maps with shape {maps.shape}")
print("Original map standard deviations:")
for i in range(n_bins):
    print(f"  Bin {i+1}: {maps[i].std():.6f}")

# Apply BNT transform
bnt_maps = apply_bnt_transform(maps)

print("\nBNT-transformed map standard deviations:")
for i in range(n_bins):
    print(f"  BNT {i+1}: {bnt_maps[i].std():.6f}")
print()

# Example 3: Complete processing pipeline
print("Example 3: Complete processing pipeline")
print("=" * 50)

def process_map(map_data, add_noise=True):
    """Process a single map through the analysis pipeline."""
    # Step 1: Add noise if requested
    if add_noise:
        map_data = add_shape_noise(map_data, sigma_e=0.26, nside=512)
    
    # Step 2: Would compute wavelet transform here
    # (requires pycs to be installed)
    # l1_norms = compute_l1_norms(map_data)
    
    # Step 3: Would compute power spectrum
    # cls = hp.anafast(map_data)
    
    return map_data

# Process the map
processed = process_map(test_map, add_noise=True)
print(f"Processed map shape: {processed.shape}")
print(f"Processed map statistics: mean={processed.mean():.6f}, std={processed.std():.6f}")
print()

print("✅ All examples completed successfully!")
print("\nNext steps:")
print("- See notebooks/ for interactive examples")
print("- Check scripts/ for production processing pipelines")
print("- Read docs/workflows/ for complete analysis workflows")
