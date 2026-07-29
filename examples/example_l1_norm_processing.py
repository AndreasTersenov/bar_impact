#!/usr/bin/env python3
"""
Example: L1 Norm Processing using the bar_impact package

This example shows how to process convergence maps to compute L1 norms
using the new package structure.

IMPORTANT: This example requires the pycs library which is not part of
the standard bar_impact installation. The pycs library must be installed
in a separate environment.

To run this example:
    # If pycs is in your current environment:
    python examples/example_l1_norm_processing.py
    
    # If pycs is in a conda environment called 'pycs':
    conda run -n pycs python examples/example_l1_norm_processing.py

Equivalent to running:
    python scripts/l1_norm_processing.py \
        --bins 1,2,3,4 --noise-level 0.26 --apply-mask \
        --mask-area-sqdeg 35000.0 --num-workers 4 \
        --save-combined --fiducial
"""

import os
import numpy as np
import h5py
import glob
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Import from the new package
from bar_impact.core import ConvergenceMap, SurveyMask

# Use pycs directly for L1 norms (like original script)
from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere


# ============================================================================
# Configuration (equivalent to argparse arguments)
# ============================================================================

# CosmoGRID directory structure:
# - Grid cosmologies: BASE_DIR/cosmo_XXXX/perm_XXXX/projected_probes_maps_*.h5
# - Fiducial cosmology: FIDUCIAL_DIR/perm_XXXX/projected_probes_maps_*.h5

BASE_DIR_GRID = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid"
BASE_DIR_FIDUCIAL = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial"

BINS = [1, 2, 3, 4]
NOISE_LEVEL = 0.26
APPLY_MASK = True
MASK_AREA_SQDEG = 14002.0  # NEW TEST: using a mask area that hasn't been processed before
NUM_WORKERS = 40
SAVE_COMBINED = True
PROCESS_FIDUCIAL = True      # Process fiducial cosmology (nobaryons reference)
PROCESS_GRID = True          # Process grid cosmologies
USE_BARYONIFIED = False      # Use nobaryons maps for training
OUTPUT_DIR = "./outputs/l1_norms"
NSIDE = 512

# Number of permutations
N_PERMS_FIDUCIAL = 200  # perm_0000 to perm_0199
N_PERMS_GRID = 7        # perm_0000 to perm_0006


# ============================================================================
# Processing Functions
# ============================================================================

def get_file_list_fiducial(base_dir: str, baryonified: bool = True) -> list:
    """
    Get list of fiducial cosmology simulation files.
    
    Structure: base_dir/perm_XXXX/projected_probes_maps_*.h5
    """
    filename = "projected_probes_maps_baryonified512.h5" if baryonified else "projected_probes_maps_nobaryons512.h5"
    perm_dirs = [f"perm_{i:04d}" for i in range(N_PERMS_FIDUCIAL)]
    
    file_paths = []
    for perm in perm_dirs:
        file_path = os.path.join(base_dir, perm, filename)
        if os.path.exists(file_path):
            file_paths.append(file_path)
    
    return sorted(file_paths)


def get_file_list_grid(base_dir: str, baryonified: bool = True) -> list:
    """
    Get list of grid cosmology simulation files.
    
    Structure: base_dir/cosmo_XXXX/perm_XXXX/projected_probes_maps_*.h5
    """
    filename = "projected_probes_maps_baryonified512.h5" if baryonified else "projected_probes_maps_nobaryons512.h5"
    
    # Find all cosmology directories
    if not os.path.exists(base_dir):
        return []
    
    cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
    perm_dirs = [f"perm_{i:04d}" for i in range(N_PERMS_GRID)]
    
    file_paths = []
    for cosmo in cosmo_dirs:
        for perm in perm_dirs:
            file_path = os.path.join(base_dir, cosmo, perm, filename)
            if os.path.exists(file_path):
                file_paths.append(file_path)
    
    return sorted(file_paths)


def process_single_file(
    file_path: str,
    bin_number: int,
    mask: SurveyMask | None,
    noise_level: float,
    output_dir: str,
    mask_area_sqdeg: float = 0.0,  # Pass target mask area for filename
) -> str | None:
    """
    Process a single file to compute L1 norms.
    
    Uses pycs directly like the original script.
    """
    # Build output filename using TARGET mask area (like original script)
    # Original uses: area_tag = int(round(mask_area_sqdeg)) from the requested area
    mask_suffix = f"_masked_{int(round(mask_area_sqdeg))}sqdeg" if mask else ""
    noise_suffix = f"_noisy_s{noise_level:.2f}" if noise_level > 0 else ""
    output_name = Path(file_path).stem + f"_l1_norms_bin{bin_number}{mask_suffix}{noise_suffix}.npy"
    output_path = os.path.join(output_dir, output_name)
    
    # Skip if already exists
    if os.path.exists(output_path):
        return output_path
    
    try:
        # Load convergence map
        with h5py.File(file_path, "r") as f:
            kappa_data = np.array(f[f"kg/stage3_lensing{bin_number}"])
        
        kappa_map = ConvergenceMap(
            data=kappa_data,
            nside=NSIDE,
            bin_number=bin_number,
        )
        
        # Add shape noise
        if noise_level > 0:
            kappa_map = kappa_map.add_shape_noise(sigma_e=noise_level)
        
        # Apply mask BEFORE computing L1 norms (like original script)
        # Original script multiplies kg by mask before calling get_wtl1_sphere
        kappa_data = kappa_map.data
        if mask is not None:
            kappa_data = kappa_data * mask.data
        
        # Compute L1 norms using pycs directly (like original script)
        # This returns shape (nscales, nbins) = (5, 40)
        # 
        # CRITICAL: Match original script exactly:
        # 1. Do NOT pass Mask parameter to get_wtl1_sphere (apply mask to data first)
        # 2. Do NOT use min_snr_coarse/max_snr_coarse parameters (causes ~1400x larger values)
        # 3. Use only: nscales=5, nbins=40, min_snr=-13, max_snr=13, noise_std=0.0146
        _, l1_norms = get_wtl1_sphere(
            kappa_data,
            nscales=5,
            nbins=40,
            min_snr=-13,
            max_snr=13,
            noise_std=0.0146
        )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_path, l1_norms)
        
        return output_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_bin(
    bin_number: int,
    files: list,
    mask: SurveyMask | None,
    noise_level: float,
    output_dir: str,
    num_workers: int,
    mask_area_sqdeg: float = 0.0,  # Pass target mask area for filename
) -> list:
    """Process all files for a single redshift bin."""
    
    # Create partial function for parallel processing
    process_func = partial(
        process_single_file,
        bin_number=bin_number,
        mask=mask,
        noise_level=noise_level,
        output_dir=output_dir,
        mask_area_sqdeg=mask_area_sqdeg,
    )
    
    # Process files in parallel
    results = []
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, files),
                total=len(files),
                desc=f"Bin {bin_number}",
            ))
    else:
        for f in tqdm(files, desc=f"Bin {bin_number}"):
            results.append(process_func(f))
    
    return [r for r in results if r is not None]


def combine_results(
    output_files: list,
    output_path: str,
) -> np.ndarray:
    """
    Combine individual results into a single array.
    
    Uses np.stack to create shape (n_samples, nscales, nbins) from
    individual files of shape (nscales, nbins).
    """
    arrays = []
    for f in output_files:
        data = np.load(f)
        arrays.append(data)
    
    # Stack to create (n_samples, nscales, nbins) shape
    combined = np.stack(arrays, axis=0)
    np.save(output_path, combined)
    print(f"Saved combined results: {combined.shape} to {output_path}")
    return combined


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def main():
    """Main processing pipeline."""
    
    print("=" * 60)
    print("L1 Norm Processing with bar_impact package")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create mask if requested
    mask = None
    if APPLY_MASK:
        mask = SurveyMask.create_disk_mask(
            nside=NSIDE,
            target_area_sqdeg=MASK_AREA_SQDEG,
            center_coords=(0.0, 90.0),
        )
        print(f"Created mask: area={mask.area_sqdeg:.1f} sq.deg, f_sky={mask.f_sky:.4f}")
    
    # Process grid cosmologies
    if PROCESS_GRID:
        map_type = "baryonified" if USE_BARYONIFIED else "nobaryons"
        print(f"\n--- Processing grid cosmologies ({map_type}) ---")
        files = get_file_list_grid(BASE_DIR_GRID, baryonified=USE_BARYONIFIED)
        print(f"Found {len(files)} files in {BASE_DIR_GRID}")
        
        if not files:
            print(f"  WARNING: No files found!")
            print(f"  Expected structure: {BASE_DIR_GRID}/cosmo_XXXX/perm_XXXX/projected_probes_maps_{map_type}512.h5")
        else:
            for bin_num in BINS:
                output_files = process_bin(
                    bin_number=bin_num,
                    files=files,
                    mask=mask,
                    noise_level=NOISE_LEVEL,
                    output_dir=os.path.join(OUTPUT_DIR, f"grid_{map_type}"),
                    num_workers=NUM_WORKERS,
                    mask_area_sqdeg=MASK_AREA_SQDEG,
                )
                
                if SAVE_COMBINED and output_files:
                    mask_tag = f"_masked_{int(MASK_AREA_SQDEG)}sqdeg" if mask else ""
                    combined_path = os.path.join(
                        OUTPUT_DIR,
                        f"combined_l1_norms_grid_{map_type}_bin{bin_num}{mask_tag}_noisy_s{NOISE_LEVEL:.2f}.npy"
                    )
                    combine_results(output_files, combined_path)
    
    # Process fiducial cosmology
    if PROCESS_FIDUCIAL:
        # Fiducial typically uses nobaryons for reference
        print(f"\n--- Processing fiducial cosmology (nobaryons) ---")
        files = get_file_list_fiducial(BASE_DIR_FIDUCIAL, baryonified=False)
        print(f"Found {len(files)} files in {BASE_DIR_FIDUCIAL}")
        
        if not files:
            print(f"  WARNING: No files found!")
            print(f"  Expected structure: {BASE_DIR_FIDUCIAL}/perm_XXXX/projected_probes_maps_nobaryons512.h5")
        else:
            for bin_num in BINS:
                output_files = process_bin(
                    bin_number=bin_num,
                    files=files,
                    mask=mask,
                    noise_level=NOISE_LEVEL,
                    output_dir=os.path.join(OUTPUT_DIR, "fiducial_nobaryons"),
                    num_workers=NUM_WORKERS,
                    mask_area_sqdeg=MASK_AREA_SQDEG,
                )
                
                if SAVE_COMBINED and output_files:
                    mask_tag = f"_masked_{int(MASK_AREA_SQDEG)}sqdeg" if mask else ""
                    combined_path = os.path.join(
                        OUTPUT_DIR,
                        f"combined_l1_norms_fiducial_nobaryons_bin{bin_num}{mask_tag}_noisy_s{NOISE_LEVEL:.2f}.npy"
                    )
                    combine_results(output_files, combined_path)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
