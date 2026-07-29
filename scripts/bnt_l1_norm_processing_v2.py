w#!/usr/bin/env python3
"""
BNT L1 Norm Processing Script (Refactored Version)

This script applies Band-limited Nulling Transform (BNT) to cosmological maps
and computes L1 norms using the modular bar_impact package.

Key improvements over original version:
- Uses centralized BNT transform and constants
- Leverages L1NormProcessor class for cleaner code
- Imports from bar_impact.* modules instead of duplicating code
- Maintains same command-line interface for backward compatibility
"""

import os
import sys
import argparse
import contextlib
import io
import glob
import h5py
import numpy as np
import healpy as hp
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

# Import from bar_impact package
from bar_impact.constants import (
    DEFAULT_NSIDE,
    DEFAULT_MASK_AREA_SQDEG,
    DEFAULT_MASK_CENTER,
    DEFAULT_SIGMA_E,
    DEFAULT_GALAXY_DENSITY,
    DEFAULT_NUM_SCALES,
    DEFAULT_NOISE_STD,
    BNT_MATRIX_DEFAULT,
)
from bar_impact.core.masks import SurveyMask
from bar_impact.utils.noise import add_shape_noise
from bar_impact.utils.reproducibility import seed_worker
from bar_impact.utils.paths import get_data_file_paths
from bar_impact.processing.l1_norms import L1NormProcessor, L1NormConfig
from bar_impact.processing.bnt_transforms import apply_bnt_transform


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout


def process_file(file_path, bnt_bin=3, noise_level=0.26, add_noise=True,
                min_snr=-13, max_snr=13, noise_std=0.0146, verbose=False,
                apply_mask=False, mask_area_sqdeg=14000.0, mask_center=(0.0, 90.0),
                force_overwrite=False, min_snr_coarse=100, max_snr_coarse=200):
    """
    Process a single file: extract kappa maps for all bins, apply BNT transform,
    compute L1 norms for the specified BNT bin, and save results.
    """
    # Define output filename based on BNT bin number, mask, and noise level
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
    
    # Match original filename format: _bnt_l1_norms_bin{1-indexed}
    if add_noise:
        suffix = f"_bnt_l1_norms_bin{bnt_bin+1}{mask_suffix}_noisy_s{noise_level:.2f}_new_normalization.npy"
    else:
        suffix = f"_bnt_l1_norms_bin{bnt_bin+1}{mask_suffix}_new_normalization.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if file already exists (unless force_overwrite is set)
    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)} - output already exists")
        return save_path
    
    try:
        # Load kappa maps for all 4 bins
        with h5py.File(file_path, "r") as f:
            kg_maps = np.array([
                f["kg/stage3_lensing1"][()],
                f["kg/stage3_lensing2"][()],
                f["kg/stage3_lensing3"][()],
                f["kg/stage3_lensing4"][()],
            ])
        
        # Add shape noise if requested
        if add_noise:
            for i in range(4):
                kg_maps[i] = add_shape_noise(
                    kg_maps[i],
                    sigma_e=noise_level,
                    galaxy_density=DEFAULT_GALAXY_DENSITY,
                    nside=DEFAULT_NSIDE
                )
        
        # Apply mask if requested (before BNT transform)
        if apply_mask:
            mask = SurveyMask.create_disk_mask(
                nside=DEFAULT_NSIDE,
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
                use_cache=True
            )
            mask_array = mask.data
            # Apply mask to all bins before BNT transform
            for i in range(4):
                kg_maps[i] = kg_maps[i] * mask_array
        
        # Apply BNT transform
        bnt_maps = apply_bnt_transform(kg_maps, bnt_matrix=BNT_MATRIX_DEFAULT)
        
        # Extract the specified BNT bin
        kg_bnt = bnt_maps[bnt_bin]
        
        # Create L1 norm processor configuration
        config = L1NormConfig(
            nscales=DEFAULT_NUM_SCALES,
            nbins=40,
            noise_std=noise_std,
            min_snr=min_snr,
            max_snr=max_snr,
            min_snr_coarse=min_snr_coarse,
            max_snr_coarse=max_snr_coarse,
        )
        
        # Compute L1 norms
        processor = L1NormProcessor(config=config)
        
        with suppress_stdout():
            l1_norms = processor.process_single(
                kg_bnt,
                mask=None,
            )
        
        # Save results
        np.save(save_path, l1_norms)
        
        if verbose:
            print(f"Processed {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        
        return save_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Process HEALPix maps with BNT transform and compute L1 norms."
    )
    
    # Main processing options
    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", 
                        help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")
    
    # BNT bin selection (mutually exclusive)
    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument("--bnt-bin", type=int, default=None,
                        help="Single BNT bin to analyze (0-3)")
    bin_group.add_argument("--bnt-bins", type=str, default=None,
                        help="Comma-separated list of BNT bins to process (e.g., '0,1,2,3')")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=DEFAULT_SIGMA_E, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before BNT transform.")
    parser.add_argument("--mask-area-sqdeg", type=float, default=DEFAULT_MASK_AREA_SQDEG,
                        help="Area of the Euclid-like mask in square degrees.")
    parser.add_argument("--mask-center", type=float, nargs=2, metavar=("LON", "LAT"),
                        default=DEFAULT_MASK_CENTER,
                        help="Mask centre in Galactic-like (lon, lat) degrees.")
    
    # Algorithm parameters
    parser.add_argument("--min-snr", type=float, default=-13, 
                        help="Minimum SNR value.")
    parser.add_argument("--max-snr", type=float, default=13, 
                        help="Maximum SNR value.")
    parser.add_argument("--min-snr-coarse", type=str, default="10,40,100,150",
                        help="Comma-separated min SNR values for coarse scale per BNT bin (default: '10,40,100,150' for bins 0-3).")
    parser.add_argument("--max-snr-coarse", type=str, default="50,100,200,300",
                        help="Comma-separated max SNR values for coarse scale per BNT bin (default: '50,100,200,300' for bins 0-3).")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD,
                        help="Noise standard deviation for wavelet normalization.")
    
    # Execution options
    parser.add_argument("--num-workers", type=int, default=70,
                        help="Number of worker processes.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    # Output options
    parser.add_argument("--save-combined", action="store_true",
                        help="Save combined L1 norms to a single file.")
    parser.add_argument("--combined-output", 
                        help="Path for combined output file.")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Force reprocessing of files even if output already exists.")
    
    args = parser.parse_args()
    
    # Parse BNT bins
    if args.bnt_bins:
        bnt_bin_numbers = [int(b.strip()) for b in args.bnt_bins.split(',')]
    elif args.bnt_bin is not None:
        bnt_bin_numbers = [args.bnt_bin]
    else:
        print("Error: Must specify --bnt-bin or --bnt-bins")
        sys.exit(1)
    
    # Validate BNT bin numbers
    for bnt_bin in bnt_bin_numbers:
        if bnt_bin not in [0, 1, 2, 3]:
            print(f"Error: BNT bin {bnt_bin} is invalid. Must be 0, 1, 2, or 3.")
            sys.exit(1)
    
    # Parse per-bin coarse SNR ranges
    min_snr_coarse_list = [float(x.strip()) for x in args.min_snr_coarse.split(',')]
    max_snr_coarse_list = [float(x.strip()) for x in args.max_snr_coarse.split(',')]
    coarse_snr_min = {i: min_snr_coarse_list[i] for i in range(len(min_snr_coarse_list))}
    coarse_snr_max = {i: max_snr_coarse_list[i] for i in range(len(max_snr_coarse_list))}
    
    print(f"Coarse scale SNR ranges per BNT bin:")
    for b in bnt_bin_numbers:
        min_snr_c = coarse_snr_min.get(b, 100)
        max_snr_c = coarse_snr_max.get(b, 200)
        print(f"  BNT bin {b}: [{min_snr_c}, {max_snr_c}]")
    
    # Convert mask_center to tuple
    mask_center = tuple(args.mask_center)
    
    # Set the base directory based on fiducial flag or override
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/new_grid/"
    
    # Get file paths using utility function
    base_dir, file_paths = get_data_file_paths(
        base_dir=args.base_dir,
        fiducial=args.fiducial,
        baryonified=args.baryonified,
    )
    
    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"\n{'='*60}")
    print(f"BNT L1 Norm Processing (Refactored Version)")
    print(f"{'='*60}")
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Computing BNT L1 norms for BNT bins: {[b for b in bnt_bin_numbers]}")
    
    # Print mask information
    if args.apply_mask:
        print(f"\nMask configuration:")
        print(f"  Area: {args.mask_area_sqdeg:.1f} sq deg")
        print(f"  Center: ({args.mask_center[0]:.1f}, {args.mask_center[1]:.1f}) deg")
        f_sky_approx = args.mask_area_sqdeg / 41252.96125
        print(f"  f_sky ≈ {f_sky_approx:.3f}")
    
    print(f"\nNoise configuration:")
    print(f"  Add noise: {not args.no_noise}")
    if not args.no_noise:
        print(f"  Noise level (sigma_e): {args.noise_level:.2f}")
    print(f"{'='*60}\n")
    
    # Track results per bin
    all_bin_results = {bnt_bin: [] for bnt_bin in bnt_bin_numbers}
    
    # Process each BNT bin
    for bnt_bin in bnt_bin_numbers:
        print(f"\n{'='*60}")
        print(f"Processing BNT Bin {bnt_bin}")
        print(f"{'='*60}")
        
        # Get coarse SNR range for this bin
        min_snr_coarse = coarse_snr_min.get(bnt_bin, 100)
        max_snr_coarse = coarse_snr_max.get(bnt_bin, 200)
        
        # Create partial function with fixed parameters
        process_func = partial(
            process_file,
            bnt_bin=bnt_bin,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            noise_std=args.noise_std,
            verbose=args.verbose,
            apply_mask=args.apply_mask,
            mask_area_sqdeg=args.mask_area_sqdeg,
            mask_center=mask_center,
            force_overwrite=args.force_overwrite,
            min_snr_coarse=min_snr_coarse,
            max_snr_coarse=max_snr_coarse,
        )
        
        # Process files in parallel
        print(f"Processing {len(file_paths)} files with {args.num_workers} workers...")
        with Pool(processes=args.num_workers, initializer=seed_worker) as pool:
            results = list(tqdm(
                pool.imap(process_func, file_paths),
                total=len(file_paths),
                desc=f"BNT bin {bnt_bin}"
            ))
        
        # Filter out None results
        successful_results = [r for r in results if r is not None]
        all_bin_results[bnt_bin] = successful_results
        
        print(f"\nBNT bin {bnt_bin} complete:")
        print(f"  Successfully processed: {len(successful_results)}/{len(file_paths)}")
        print(f"{'='*60}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Processing Summary")
    print(f"{'='*60}")
    for bnt_bin in bnt_bin_numbers:
        successful = len(all_bin_results[bnt_bin])
        print(f"BNT bin {bnt_bin}: {successful}/{len(file_paths)} files processed")
    print(f"{'='*60}\n")
    
    # Optionally save combined results for each bin
    if args.save_combined:
        print("Saving combined results...")
        for bnt_bin in bnt_bin_numbers:
            results = all_bin_results[bnt_bin]
            if len(results) > 0:
                all_l1 = []
                for result_path in results:
                    l1 = np.load(result_path)
                    all_l1.append(l1)
                all_l1 = np.array(all_l1)
                
                if args.combined_output:
                    combined_path = args.combined_output
                else:
                    mask_suffix = f"_masked_{int(args.mask_area_sqdeg)}sqdeg" if args.apply_mask else ""
                    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if not args.no_noise else ""
                    combined_path = os.path.join(
                        base_dir,
                        f"all_bnt_l1_norms_bin{bnt_bin+1}{mask_suffix}{noise_suffix}_new_normalization.npy"
                    )
                
                np.save(combined_path, all_l1)
                print(f"Saved combined results for BNT bin {bnt_bin} to {combined_path}")
                print(f"  Shape: {all_l1.shape}")
    
    print("\nAll processing complete!")


if __name__ == "__main__":
    main()
