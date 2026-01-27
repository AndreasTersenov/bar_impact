#!/usr/bin/env python3
"""
BNT Peak Counts Processing Script (Refactored Version)

This script applies Band-limited Nulling Transform (BNT) to cosmological maps
and computes peak counts using the modular bar_impact package.

Key improvements over original version:
- Uses centralized BNT transform and constants
- Leverages PeakCountProcessor class for cleaner code
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
from bar_impact.processing.peak_counts import PeakCountProcessor, PeakCountConfig
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


def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))


def process_file(file_path, bnt_bin=3, noise_level=0.26, add_noise=True, 
                noise_std=0.0146, nbins=31, min_val=-2, max_val=6, verbose=False,
                apply_mask=False, mask_area_sqdeg=14000.0, mask_center=(0.0, 90.0),
                force_overwrite=False, min_snr_coarse=100, max_snr_coarse=200):
    """
    Process a single file: extract kappa maps for all bins, apply BNT transform,
    compute peak counts for the specified BNT bin, and save results.
    """
    # Define output filename based on BNT bin number, mask, and noise level
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
    
    if add_noise:
        suffix = f"_peak_counts_bnt{bnt_bin}{mask_suffix}_noisy_s{noise_level:.2f}.npy"
    else:
        suffix = f"_peak_counts_bnt{bnt_bin}{mask_suffix}.npy"
    
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
        mask_array = None
        if apply_mask:
            mask = SurveyMask.create_disk_mask(
                nside=DEFAULT_NSIDE,
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
                use_cache=True
            )
            mask_array = mask.data
            # Apply mask to all bins
            for i in range(4):
                kg_maps[i] = kg_maps[i] * mask_array
        
        # Apply BNT transform
        bnt_maps = apply_bnt_transform(kg_maps, bnt_matrix=BNT_MATRIX_DEFAULT)
        
        # Extract the specified BNT bin
        kg_bnt = bnt_maps[bnt_bin]
        
        # Create peak count processor configuration
        config = PeakCountConfig(
            nscales=DEFAULT_NUM_SCALES,
            nbins=nbins,
            noise_std=noise_std,
            min_val=min_val,
            max_val=max_val,
        )
        
        # Compute peak counts
        processor = PeakCountProcessor(config=config)
        
        with suppress_stdout():
            peak_counts = processor.process_single(kg_bnt)
        
        # Save results
        np.save(save_path, peak_counts)
        
        if verbose:
            print(f"Processed {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        
        return save_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Process HEALPix maps with BNT transform to compute multiscale peak counts."
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
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD,
                        help="Noise standard deviation for wavelet normalization.")
    parser.add_argument("--nbins", type=int, default=31,
                        help="Number of bins for the histogram.")
    parser.add_argument("--min-val", type=float, default=-2,
                        help="Minimum value for the histogram bins.")
    parser.add_argument("--max-val", type=float, default=10,
                        help="Maximum value for the histogram bins.")
    parser.add_argument("--min-snr-coarse", type=str, default="10,40,100,150",
                        help="Comma-separated min SNR values for coarse scale per BNT bin.")
    parser.add_argument("--max-snr-coarse", type=str, default="50,100,200,300",
                        help="Comma-separated max SNR values for coarse scale per BNT bin.")
    
    # Execution options
    parser.add_argument("--num-workers", type=int, default=70,
                        help="Number of worker processes.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    # Output options
    parser.add_argument("--save-combined", action="store_true",
                        help="Save combined peak counts to a single file.")
    parser.add_argument("--combined-output", 
                        help="Path for combined output file.")
    parser.add_argument("--cleanup-empty", action="store_true",
                        help="Remove empty output files before processing.")
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
    
    # Set the base directory
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    
    # Set the filename
    if args.baryonified:
        filename = "projected_probes_maps_baryonified512.h5"
    else:
        filename = "projected_probes_maps_nobaryons512.h5"
    
    # Set permutation directories
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
    
    # Print configuration
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"\n{'='*60}")
    print(f"BNT Peak Counts Processing (Refactored Version)")
    print(f"{'='*60}")
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Computing BNT peak counts for BNT bins: {[b for b in bnt_bin_numbers]}")
    
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
        
        # Create partial function
        process_func = partial(
            process_file,
            bnt_bin=bnt_bin,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            noise_std=args.noise_std,
            nbins=args.nbins,
            min_val=args.min_val,
            max_val=args.max_val,
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
    
    # Save combined results
    if args.save_combined:
        print("Saving combined results...")
        for bnt_bin in bnt_bin_numbers:
            results = all_bin_results[bnt_bin]
            if len(results) > 0:
                all_peaks = []
                for result_path in results:
                    peaks = np.load(result_path)
                    all_peaks.append(peaks)
                all_peaks = np.array(all_peaks)
                
                if args.combined_output:
                    combined_path = args.combined_output
                else:
                    mask_suffix = f"_masked_{int(args.mask_area_sqdeg)}sqdeg" if args.apply_mask else ""
                    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if not args.no_noise else ""
                    combined_path = os.path.join(
                        base_dir,
                        f"all_peak_counts_bnt{bnt_bin}{mask_suffix}{noise_suffix}.npy"
                    )
                
                np.save(combined_path, all_peaks)
                print(f"Saved combined results for BNT bin {bnt_bin} to {combined_path}")
                print(f"  Shape: {all_peaks.shape}")
    
    print("\nAll processing complete!")


if __name__ == "__main__":
    main()
