#!/usr/bin/env python3
"""
L1 Norm Processing Script (Refactored Version)

This script uses the modular organization from bar_impact package.
Processes cosmological data files to compute L1 norms using spherical wavelets.

Key improvements over original version:
- Uses centralized constants and utility functions
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
from multiprocessing import Pool, cpu_count
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
    DEFAULT_MIN_SNR,
    DEFAULT_MAX_SNR,
)
from bar_impact.core.masks import SurveyMask
from bar_impact.utils.noise import add_shape_noise
from bar_impact.processing.l1_norms import L1NormProcessor, L1NormConfig


# Context manager to suppress stdout
@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout


def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))


def process_file(file_path, bin_number=2, noise_level=0.26, add_noise=True,
                min_snr=-13, max_snr=13, noise_std=0.0146, verbose=False,
                apply_mask=False, mask_area_sqdeg=14000.0, mask_center=(0.0, 90.0),
                force_overwrite=False, min_snr_coarse=100, max_snr_coarse=200):
    """
    Process a single file: extract kappa map, apply optional mask, compute L1 norms, save results.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing kappa maps
    bin_number : int
        Bin number to process (1-4)
    noise_level : float
        Shape noise level (sigma_e)
    add_noise : bool
        Whether to add shape noise
    min_snr : float
        Minimum SNR for fine scale bins
    max_snr : float
        Maximum SNR for fine scale bins
    noise_std : float
        Noise standard deviation for wavelet normalization
    verbose : bool
        Whether to print detailed output
    apply_mask : bool
        Whether to apply survey mask
    mask_area_sqdeg : float
        Mask area in square degrees
    mask_center : tuple
        Mask center (lon, lat) in degrees
    force_overwrite : bool
        Whether to overwrite existing output files
    min_snr_coarse : float
        Minimum SNR for coarse scale
    max_snr_coarse : float
        Maximum SNR for coarse scale
        
    Returns
    -------
    str or None
        Path to saved output file, or None if processing failed
    """
    # Define output filename
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
    
    if add_noise:
        suffix = f"_l1_norms_bin{bin_number}{mask_suffix}_noisy_s{noise_level:.2f}_new_normalization.npy"
    else:
        suffix = f"_l1_norms_bin{bin_number}{mask_suffix}_new_normalization.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Map key based on bin number
    map_key = f"kg/stage3_lensing{bin_number}"
    
    # Skip if file already exists (unless force_overwrite is set)
    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)} - output already exists")
        return save_path
    
    try:
        # Load kappa map
        with h5py.File(file_path, "r") as f:
            kg = f[map_key][()]
        
        # Add shape noise if requested
        if add_noise:
            kg = add_shape_noise(
                kg, 
                sigma_e=noise_level, 
                galaxy_density=DEFAULT_GALAXY_DENSITY, 
                nside=DEFAULT_NSIDE
            )
        
        # Apply mask if requested
        mask_array = None
        if apply_mask:
            # Create or retrieve cached mask
            mask = SurveyMask.create_disk_mask(
                nside=DEFAULT_NSIDE,
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
                use_cache=True
            )
            mask_array = mask.data
            # Note: We pass the mask to L1NormProcessor (which passes to get_wtl1_sphere)
            # so it only computes L1 norms on valid (unmasked) pixels. We do NOT multiply
            # kg*mask because that would create edge artifacts in the wavelet transform.
        
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
        
        # Suppress pycs output
        with suppress_stdout():
            l1_norms = processor.process_single(
                kg,
                mask=mask_array,
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
        description="Process HEALPix maps to compute L1 norms using modular bar_impact package."
    )
    
    # Main processing options
    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", 
                        help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")
    
    # Bin selection
    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument("--bin-number", type=int, default=1, 
                        help="Single bin number to process (used for map_key 'kg/stage3_lensing{bin_number}' and suffix)")
    bin_group.add_argument("--bins", type=str,
                        help="Comma-separated list of bin numbers to process (e.g., '1,2,3,4')")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=DEFAULT_SIGMA_E, 
                        help=f"Shape noise level (sigma_e) (default: {DEFAULT_SIGMA_E})")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before computing L1 norms.")
    parser.add_argument("--mask-area-sqdeg", type=float, default=DEFAULT_MASK_AREA_SQDEG,
                        help=f"Area of the Euclid-like mask in square degrees (default: {DEFAULT_MASK_AREA_SQDEG}).")
    parser.add_argument("--mask-center", type=float, nargs=2, metavar=("LON", "LAT"),
                        default=DEFAULT_MASK_CENTER,
                        help=f"Mask centre in Galactic-like (lon, lat) degrees (default: {DEFAULT_MASK_CENTER}).")
    
    # Algorithm parameters
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR, 
                        help="Minimum SNR value.")
    parser.add_argument("--max-snr", type=float, default=DEFAULT_MAX_SNR, 
                        help="Maximum SNR value.")
    parser.add_argument("--min-snr-coarse", type=str, default="10,40,100,150",
                        help="Comma-separated min SNR values for coarse scale per bin (default: '10,40,100,150' for bins 1-4).")
    parser.add_argument("--max-snr-coarse", type=str, default="50,100,200,300",
                        help="Comma-separated max SNR values for coarse scale per bin (default: '50,100,200,300' for bins 1-4).")
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
    
    # Set the base directory based on fiducial flag or override
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    
    # Set the filename based on the baryonified flag
    if args.baryonified:
        filename = "projected_probes_maps_baryonified512.h5"
    else:
        filename = "projected_probes_maps_nobaryons512.h5"
    
    # Set permutation directories based on fiducial flag
    if args.fiducial:
        # Fiducial cosmology: perm_0000 to perm_0199 directly under base_dir
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
    else:
        # Grid cosmologies: cosmo_XXXX/perm_0000 to perm_0006
        cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
        perm_dirs = [f"perm_{i:04d}" for i in range(7)]
        file_paths = [
            os.path.join(base_dir, cosmo, perm, filename)
            for cosmo in cosmo_dirs
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
        ]
    
    # Normalize mask center to tuple
    mask_center = tuple(args.mask_center)
    
    # Parse bin numbers
    if args.bins:
        bin_numbers = [int(b.strip()) for b in args.bins.split(',')]
    else:
        bin_numbers = [args.bin_number]
    
    # Parse per-bin coarse SNR ranges
    min_snr_coarse_list = [float(x.strip()) for x in args.min_snr_coarse.split(',')]
    max_snr_coarse_list = [float(x.strip()) for x in args.max_snr_coarse.split(',')]
    
    # Create dictionaries mapping bin number to coarse SNR range
    coarse_snr_min = {i+1: min_snr_coarse_list[i] for i in range(len(min_snr_coarse_list))}
    coarse_snr_max = {i+1: max_snr_coarse_list[i] for i in range(len(max_snr_coarse_list))}
    
    print(f"Coarse scale SNR ranges per bin:")
    for b in bin_numbers:
        min_snr_c = coarse_snr_min.get(b, 100)
        max_snr_c = coarse_snr_max.get(b, 200)
        print(f"  Bin {b}: [{min_snr_c}, {max_snr_c}]")
    
    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"\n{'='*60}")
    print(f"L1 Norm Processing (Refactored Version)")
    print(f"{'='*60}")
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    if len(bin_numbers) == 1:
        print(f"Computing L1 norms for bin {bin_numbers[0]}")
    else:
        print(f"Computing L1 norms for bins: {bin_numbers}")
    
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
    
    # Process each bin
    all_bin_results = {}
    for bin_number in bin_numbers:
        print(f"\n{'='*60}")
        print(f"Processing Bin {bin_number}")
        print(f"{'='*60}")
        
        # Get coarse SNR range for this bin
        min_snr_coarse = coarse_snr_min.get(bin_number, 100)
        max_snr_coarse = coarse_snr_max.get(bin_number, 200)
        
        # Create partial function with fixed parameters
        process_func = partial(
            process_file,
            bin_number=bin_number,
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
                desc=f"Bin {bin_number}"
            ))
        
        # Filter out None results (failed processing)
        successful_results = [r for r in results if r is not None]
        all_bin_results[bin_number] = successful_results
        
        print(f"\nBin {bin_number} complete:")
        print(f"  Successfully processed: {len(successful_results)}/{len(file_paths)}")
        print(f"{'='*60}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Processing Summary")
    print(f"{'='*60}")
    for bin_number in bin_numbers:
        successful = len(all_bin_results[bin_number])
        print(f"Bin {bin_number}: {successful}/{len(file_paths)} files processed")
    print(f"{'='*60}\n")
    
    # Optionally save combined results
    if args.save_combined:
        print("Saving combined results...")
        for bin_number in bin_numbers:
            results = all_bin_results[bin_number]
            if len(results) > 0:
                # Load all L1 norms
                all_l1 = []
                for result_path in results:
                    l1 = np.load(result_path)
                    all_l1.append(l1)
                all_l1 = np.array(all_l1)
                
                # Determine output filename
                if args.combined_output:
                    combined_path = args.combined_output
                else:
                    mask_suffix = f"_masked_{int(args.mask_area_sqdeg)}sqdeg" if args.apply_mask else ""
                    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if not args.no_noise else ""
                    combined_path = os.path.join(
                        base_dir,
                        f"all_l1_norms_bin{bin_number}{mask_suffix}{noise_suffix}_new_normalization.npy"
                    )
                
                np.save(combined_path, all_l1)
                print(f"Saved combined results for bin {bin_number} to {combined_path}")
                print(f"  Shape: {all_l1.shape}")
    
    print("\nAll processing complete!")


if __name__ == "__main__":
    main()
