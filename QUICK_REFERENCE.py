#!/usr/bin/env python3
"""
Quick Reference: Using bar_impact Modules

This file shows examples of how to use the refactored bar_impact modules
in your scripts and notebooks.
"""

import numpy as np
import healpy as hp

# =============================================================================
# 1. CONSTANTS
# =============================================================================

from bar_impact.constants import (
    DEFAULT_NSIDE,                # 512
    DEFAULT_LMAX,                 # 1024
    DEFAULT_SIGMA_E,              # 0.26
    DEFAULT_GALAXY_DENSITY,       # 6.75
    DEFAULT_MASK_AREA_SQDEG,      # 14000.0
    DEFAULT_MASK_CENTER,          # (0.0, 90.0)
    DEFAULT_NUM_SCALES,           # 5
    DEFAULT_NOISE_STD,            # 0.0146
    BNT_MATRIX_DEFAULT,           # 4x4 BNT matrix
    COSMOGRID_MAP_KEY_TEMPLATE,   # "kg/stage3_lensing{bin_number}"
)

# Get BNT matrix (with validation)
from bar_impact.constants import get_bnt_matrix
bnt_matrix = get_bnt_matrix(n_bins=4)

# =============================================================================
# 2. SHAPE NOISE
# =============================================================================

from bar_impact.utils.noise import add_shape_noise

# Add shape noise to a map
kappa = np.random.randn(hp.nside2npix(512)) * 0.01
kappa_noisy = add_shape_noise(
    kappa,
    sigma_e=0.26,
    galaxy_density=6.75,  # or ngal_arcmin2=6.75
    nside=512,
    seed=42  # For reproducibility
)

# =============================================================================
# 3. SURVEY MASKS
# =============================================================================

from bar_impact.core.masks import SurveyMask

# Create binary disk mask
mask = SurveyMask.create_disk_mask(
    nside=512,
    target_area_sqdeg=14000.0,
    center_coords=(0.0, 90.0),
    use_cache=True  # Caches mask for reuse
)

# Access mask properties
print(f"f_sky = {mask.f_sky:.3f}")
print(f"Area = {mask.area_sqdeg:.1f} sq deg")
mask_array = mask.data  # numpy array for masking

# Apply mask to map
masked_kappa = kappa * mask_array

# Create apodized mask (for power spectrum MASTER correction)
mask_apod = SurveyMask.create_apodized_disk_mask(
    nside=512,
    target_area_sqdeg=14000.0,
    center_coords=(0.0, 90.0),
    apodization_deg=2.0,  # Smooth transition width
    use_cache=True
)

# =============================================================================
# 4. L1 NORMS
# =============================================================================

from bar_impact.processing.l1_norms import (
    L1NormProcessor, 
    L1NormConfig,
    compute_l1_norms  # Low-level function
)

# Method 1: Using processor class (recommended)
config = L1NormConfig(
    nscales=5,
    nbins=40,
    noise_std=0.0146,
    min_snr=-13.0,
    max_snr=13.0,
    min_snr_coarse=100.0,
    max_snr_coarse=200.0,
)
processor = L1NormProcessor(config=config)
l1_norms = processor.process_single(kappa_noisy, mask=mask_array)

# Method 2: Using low-level function
l1_norms = compute_l1_norms(
    kappa_noisy,
    nscales=5,
    nbins=40,
    mask=mask_array,
    noise_std=0.0146,
    min_snr=-13.0,
    max_snr=13.0,
)

# =============================================================================
# 5. PEAK COUNTS
# =============================================================================

from bar_impact.processing.peak_counts import (
    PeakCountProcessor,
    PeakCountConfig,
    compute_peak_counts
)

# Using processor
config = PeakCountConfig(
    nscales=5,
    nbins=31,
    noise_std=0.0146,
    min_val=-2.0,
    max_val=6.0,
)
processor = PeakCountProcessor(config=config)
peaks = processor.process_single(kappa_noisy)

# =============================================================================
# 6. POWER SPECTRA
# =============================================================================

from bar_impact.processing.power_spectrum import (
    PowerSpectrumProcessor,
    compute_power_spectrum,
    compute_cross_power_spectrum,
)

# Auto power spectrum
cls = compute_power_spectrum(kappa_noisy, lmax=1024)

# Cross power spectrum
kappa2 = np.random.randn(hp.nside2npix(512)) * 0.01
cls_cross = compute_cross_power_spectrum(kappa_noisy, kappa2, lmax=1024)

# Using processor (with ell selection)
processor = PowerSpectrumProcessor(lmax=1024, ell_min=100, ell_max=500)
cls = processor.process_single(kappa_noisy, return_ell=False)

# =============================================================================
# 7. BNT TRANSFORM
# =============================================================================

from bar_impact.processing.bnt_transforms import apply_bnt_transform

# Load 4 tomographic bins
maps = np.array([
    kappa_bin1,  # shape: (npix,)
    kappa_bin2,
    kappa_bin3,
    kappa_bin4,
])  # shape: (4, npix)

# Apply BNT transform
bnt_maps = apply_bnt_transform(maps, bnt_matrix=BNT_MATRIX_DEFAULT)
# bnt_maps shape: (4, npix)

# Process individual BNT bin
bnt_bin_0 = bnt_maps[0]  # First BNT bin
l1_norms_bnt0 = processor.process_single(bnt_bin_0)

# =============================================================================
# 8. LOADING DATA
# =============================================================================

from bar_impact.utils.io import load_healpy_map, save_results
import h5py

# Load from HEALPix FITS
kappa = load_healpy_map('map.fits', field=0)

# Load from HDF5 (CosmoGRID format)
with h5py.File('cosmology_001/nobaryons_lensing_maps.h5', 'r') as f:
    kappa_bin1 = f['kg/stage3_lensing1'][()]
    kappa_bin2 = f['kg/stage3_lensing2'][()]
    # ... etc

# Save results
save_results(l1_norms, 'l1_norms.npy', format='npy')
save_results(
    {'l1_norms': l1_norms, 'peaks': peaks},
    'results.npz',
    format='npz'
)

# =============================================================================
# 9. BATCH PROCESSING
# =============================================================================

from bar_impact.processing.base import ProcessingConfig

# Configuration for batch processing
config = ProcessingConfig(
    add_noise=True,
    noise_level=0.26,
    galaxy_density=6.75,
    apply_mask=True,
    mask_area_sqdeg=14000.0,
    mask_center=(0.0, 90.0),
    n_workers=70,
    verbose=True,
    force_overwrite=False,
)

# Use in processor
from bar_impact.processing.l1_norms import L1NormProcessor
processor = L1NormProcessor(config=config)

# Process with automatic preprocessing
result = processor.process(
    kappa,
    apply_preprocessing=True,  # Adds noise and applies mask automatically
)

# =============================================================================
# 10. TYPICAL WORKFLOW EXAMPLE
# =============================================================================

def process_cosmology_file(h5_file, bin_number, output_dir):
    """Complete workflow for processing one cosmology."""
    
    # 1. Load data
    with h5py.File(h5_file, 'r') as f:
        kappa = f[f'kg/stage3_lensing{bin_number}'][()]
    
    # 2. Add noise
    kappa_noisy = add_shape_noise(
        kappa,
        sigma_e=DEFAULT_SIGMA_E,
        galaxy_density=DEFAULT_GALAXY_DENSITY,
        nside=DEFAULT_NSIDE,
    )
    
    # 3. Create mask
    mask = SurveyMask.create_disk_mask(
        nside=DEFAULT_NSIDE,
        target_area_sqdeg=14000.0,
        center_coords=DEFAULT_MASK_CENTER,
        use_cache=True,
    )
    
    # 4. Apply mask
    kappa_masked = kappa_noisy * mask.data
    
    # 5. Compute statistics
    l1_processor = L1NormProcessor()
    l1_norms = l1_processor.process_single(kappa_masked, mask=mask.data)
    
    peak_processor = PeakCountProcessor()
    peaks = peak_processor.process_single(kappa_masked)
    
    ps_processor = PowerSpectrumProcessor(lmax=1024)
    power_spectrum = ps_processor.process_single(kappa_masked)
    
    # 6. Save results
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / f'l1_norms_bin{bin_number}.npy', l1_norms)
    np.save(output_path / f'peaks_bin{bin_number}.npy', peaks)
    np.save(output_path / f'cls_bin{bin_number}.npy', power_spectrum)
    
    return {
        'l1_norms': l1_norms,
        'peaks': peaks,
        'power_spectrum': power_spectrum,
    }


# Usage
if __name__ == '__main__':
    results = process_cosmology_file(
        h5_file='/data/cosmogrid/cosmology_001/nobaryons_lensing_maps.h5',
        bin_number=2,
        output_dir='outputs/cosmology_001'
    )
    print(f"L1 norms shape: {results['l1_norms'].shape}")
    print(f"Peaks shape: {results['peaks'].shape}")
    print(f"Power spectrum shape: {results['power_spectrum'].shape}")
