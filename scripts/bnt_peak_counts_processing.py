#!/usr/bin/env python3
# filepath: /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/scripts/bnt_peak_counts_processing_new.py
"""
BNT Peak Counts Processing Script - Applies BNT transform to cosmological maps and computes peak counts.
"""

import os
import h5py
import healpy as hp
import numpy as np
import argparse
import tempfile
import multiprocessing as mp
import contextlib
import sys
import io
from tqdm import tqdm
from functools import partial
from pycs.sparsity.mrs.mrs_starlet import CMRStarlet
from pycs.astro.wl.hos_peaks_l1 import get_wtpeaks_sphere

# Speed up the pycs spherical starlet transform (map2alm iter, neighbour cache).
# See scripts/pycs_speedups.py. Must run before the multiprocessing Pool is
# created so forked workers inherit the patches.
import pycs_speedups
pycs_speedups.enable(starlet_iter=1)

# BNT transformation matrix
BNT_MATRIX = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [-1.        ,  1.        ,  0.        ,  0.        ],
                       [ 0.4521097 , -1.4521097 ,  1.        ,  0.        ],
                       [ 0.        ,  0.25127807, -1.251278  ,  1.        ]])

MASK_CACHE = {}

# Add this context manager to suppress stdout
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
    # Use a source of entropy from the OS to seed the worker
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """
    Adds shape noise to a full-sky Healpix convergence (kappa) map.
    
    Parameters:
    - kg: np.ndarray, the input kappa map
    - sigma_e: float, intrinsic ellipticity dispersion per galaxy
    - galaxy_density: float, galaxy number density per arcmin²
    - nside: int, Healpix resolution parameter
    
    Returns:
    - noisy_kg: np.ndarray, kappa map with added shape noise
    """
    npix = hp.nside2npix(nside)  # Total number of pixels
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600  # Convert to arcmin²
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)  # Compute pixel noise
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)  # Generate noise
    return kg + noise  # Add noise to kappa map


def create_euclid_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0)):
    """
    Create a contiguous Euclid-like disk mask of a specific sky area.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the input maps.
    target_area_sqdeg : float
        Desired unmasked area in square degrees.
    center_coords : tuple(float, float)
        (lon, lat) in degrees for the disk centre; defaults to North Pole.

    Returns
    -------
    mask : np.ndarray
        Binary mask array (1 inside mask, 0 outside) with dtype float32.
    f_sky : float
        Fraction of sky retained by the mask.
    angular_radius_deg : float
        Angular radius of the resulting spherical cap in degrees.
    """
    total_area_sqdeg = 41252.96125  # 4 * pi * (180/pi)^2
    angular_radius_rad = np.arccos(1 - (target_area_sqdeg / total_area_sqdeg) * 2)
    angular_radius_deg = np.rad2deg(angular_radius_rad)

    theta_center = np.deg2rad(90.0 - center_coords[1])
    phi_center = np.deg2rad(center_coords[0])
    center_vec = hp.ang2vec(theta_center, phi_center)

    disc_pixels = hp.query_disc(nside, center_vec, angular_radius_rad)

    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float32)
    mask[disc_pixels] = 1.0

    f_sky = mask.mean()
    return mask, f_sky, angular_radius_deg


def get_cached_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0)):
    """Return a cached Euclid-like mask to avoid recomputation in each worker."""
    key = (int(nside), float(target_area_sqdeg), float(center_coords[0]), float(center_coords[1]))
    if key not in MASK_CACHE:
        mask, f_sky, angular_radius_deg = create_euclid_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
        )
        MASK_CACHE[key] = (mask, f_sky, angular_radius_deg)
    return MASK_CACHE[key]


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
        area_tag = int(round(mask_area_sqdeg)) if mask_area_sqdeg else "mask"
        mask_suffix = f"_masked_{area_tag}sqdeg"
    
    if add_noise:
        suffix = f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}_noisy_s{noise_level:.2f}_new_normalization.npy"
    else:
        suffix = f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}_new_normalization.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if file already exists (unless force_overwrite is set)
    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, BNT peak counts file already exists.")
        return save_path
    
    try:
        # Load all 4 kappa maps
        kgs = []
        with h5py.File(file_path, "r") as f:
            for i in range(4):
                map_key = f"kg/stage3_lensing{i+1}"
                kgs.append(np.array(f[map_key]))
        
        # Apply mask if requested (before noise and BNT transform)
        if apply_mask:
            mask, f_sky, angular_radius_deg = get_cached_mask(
                nside=512,
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center
            )
            # Apply mask to all bins
            kgs = [kg * mask for kg in kgs]
        
        # Add shape noise if requested (after mask, before BNT transform)
        if add_noise:
            kgs = [add_shape_noise(kg, sigma_e=noise_level) for kg in kgs]
        
        # Apply BNT transform
        kgs = np.array(kgs)  # Convert to numpy array for matrix multiplication
        kgs_bnt = BNT_MATRIX @ kgs  # This is the key step from the notebook
        
        peak_counts, _ = get_wtpeaks_sphere(
            kgs_bnt[bnt_bin], nscales=5, noise_std=noise_std, nbins=nbins, Min=min_val, Max=max_val, verbose=False
        )
        
        # Convert to numpy array for consistent saving
        peak_counts = np.array(peak_counts)
        
        # Validate the result before saving
        if peak_counts.size == 0:
            if verbose:
                print(f"Warning: Empty peak counts for {os.path.basename(file_path)}")
            return None
        
        # Save results with error handling
        try:
            np.save(save_path, peak_counts)
            # Verify the file was saved correctly
            if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                if verbose:
                    print(f"Warning: Failed to save or empty file created for {os.path.basename(file_path)}")
                return None
        except Exception as save_error:
            if verbose:
                print(f"Error saving {os.path.basename(file_path)}: {save_error}")
            return None
        if verbose:
            print(f"Processed: {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(description="Process HEALPix maps with BNT transform to compute multiscale peak counts.")
    
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
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before BNT transform.")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the Euclid-like mask in square degrees (default: 14000).")
    parser.add_argument("--mask-center", type=float, nargs=2, metavar=("LON", "LAT"),
                        default=(0.0, 90.0),
                        help="Mask centre in Galactic-like (lon, lat) degrees (default: 0 90).")
    
    # Algorithm parameters
    parser.add_argument("--noise-std", type=float, default=0.0146,
                        help="Noise standard deviation for wavelet normalization.")
    parser.add_argument("--nbins", type=int, default=31,
                        help="Number of bins for the histogram.")
    parser.add_argument("--min-val", type=float, default=-2,
                        help="Minimum value for the histogram bins.")
    parser.add_argument("--max-val", type=float, default=10,
                        help="Maximum value for the histogram bins.")
    parser.add_argument("--min-snr-coarse", type=str, default="10,40,100,150",
                        help="Comma-separated min SNR values for coarse scale per BNT bin (default: '10,40,100,150' for bins 0-3).")
    parser.add_argument("--max-snr-coarse", type=str, default="50,100,200,300",
                        help="Comma-separated max SNR values for coarse scale per BNT bin (default: '50,100,200,300' for bins 0-3).")
    
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
        # Default to bin 3 (4th bin)
        bnt_bin_numbers = [3]
    
    # Validate BNT bin numbers
    for bnt_bin in bnt_bin_numbers:
        if bnt_bin < 0 or bnt_bin > 3:
            print(f"Error: BNT bin {bnt_bin} is out of range [0, 3].")
            return
    
    # Parse per-bin coarse SNR ranges
    min_snr_coarse_list = [float(x.strip()) for x in args.min_snr_coarse.split(',')]
    max_snr_coarse_list = [float(x.strip()) for x in args.max_snr_coarse.split(',')]
    coarse_snr_min = {i: min_snr_coarse_list[i] for i in range(len(min_snr_coarse_list))}
    coarse_snr_max = {i: max_snr_coarse_list[i] for i in range(len(max_snr_coarse_list))}
    
    print(f"Coarse scale SNR ranges per BNT bin:")
    for b in bnt_bin_numbers:
        if b in coarse_snr_min and b in coarse_snr_max:
            print(f"  BNT bin {b}: [{coarse_snr_min[b]}, {coarse_snr_max[b]}]")
        else:
            print(f"  BNT bin {b}: using defaults (no custom range specified)")
    
    # Convert mask_center to tuple
    mask_center = tuple(args.mask_center)
    args.mask_center = mask_center
    
    # Set the base directory based on fiducial flag or override
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/new_grid/"
    
    # Set the filename based on the baryonified flag
    if args.baryonified:
        filename = "projected_probes_maps_baryonified512.h5"
    else:
        filename = "projected_probes_maps_nobaryons512.h5"
    
    # Set permutation directories based on fiducial flag
    if args.fiducial:
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]  # "perm_0000" to "perm_0199"
        # Generate all file paths for fiducial cosmology (direct in perm dirs)
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
    else:
        # Find all cosmology directories
        cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
        perm_dirs = [f"perm_{i:04d}" for i in range(7)]  # "perm_0000" to "perm_0006"
        
        # Generate all file paths for grid cosmologies
        file_paths = [
            os.path.join(base_dir, cosmo, perm, filename)
            for cosmo in cosmo_dirs
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
        ]
    
    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Computing BNT peak counts for BNT bins: {[b+1 for b in bnt_bin_numbers]} (displaying as 1-indexed)")
    
    # Print mask information
    if args.apply_mask:
        mask, f_sky, angular_radius_deg = create_euclid_mask(
            nside=512,
            target_area_sqdeg=args.mask_area_sqdeg,
            center_coords=args.mask_center
        )
        print(f"\n{'='*60}")
        print(f"Euclid-like mask applied:")
        print(f"  Target area: {args.mask_area_sqdeg:.1f} sq. deg.")
        print(f"  f_sky: {f_sky:.6f}")
        print(f"  Angular radius: {angular_radius_deg:.2f} degrees")
        print(f"  Center: (lon={args.mask_center[0]:.1f}°, lat={args.mask_center[1]:.1f}°)")
        print(f"{'='*60}\n")
    
    # Determine suffix for output files
    mask_suffix = ""
    if args.apply_mask:
        area_tag = int(round(args.mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
    
    # Clean up empty files if requested (for all bins)
    if args.cleanup_empty:
        print("Cleaning up empty files...")
        empty_count = 0
        for bnt_bin in bnt_bin_numbers:
            for file_path in file_paths:
                # Construct expected output path
                if not args.no_noise:
                    expected_output = file_path.replace(".h5", f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}_noisy_s{args.noise_level:.2f}_new_normalization.npy")
                else:
                    expected_output = file_path.replace(".h5", f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}_new_normalization.npy")
                
                if os.path.exists(expected_output) and os.path.getsize(expected_output) == 0:
                    os.remove(expected_output)
                    empty_count += 1
                    if args.verbose:
                        print(f"Removed empty file: {expected_output}")
        
        print(f"Removed {empty_count} empty files")
    
    # Track results per bin
    all_bin_results = {bnt_bin: [] for bnt_bin in bnt_bin_numbers}
    
    # Process each BNT bin
    for bnt_bin in bnt_bin_numbers:
        print(f"\n{'='*60}")
        print(f"Processing BNT bin {bnt_bin+1} (0-indexed: {bnt_bin})")
        print(f"{'='*60}")
        
        if args.no_noise:
            suffix = f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}_new_normalization.npy"
        else:
            suffix = f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}_noisy_s{args.noise_level:.2f}_new_normalization.npy"
        print(f"Output suffix: {suffix}")
        
        # Get coarse SNR range for this bin (with fallback defaults)
        bin_min_snr_coarse = coarse_snr_min.get(bnt_bin, 100)
        bin_max_snr_coarse = coarse_snr_max.get(bnt_bin, 200)
        print(f"Using coarse scale SNR range: [{bin_min_snr_coarse}, {bin_max_snr_coarse}]")
        
        # Process files in parallel with progress bar
        with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
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
                mask_center=args.mask_center,
                force_overwrite=args.force_overwrite,
                min_snr_coarse=bin_min_snr_coarse,
                max_snr_coarse=bin_max_snr_coarse,
            )
            results = list(tqdm(
                pool.imap(process_func, file_paths),
                total=len(file_paths),
                desc=f"Processing BNT bin {bnt_bin+1}"
            ))
        
        # Count successful files for this bin
        successful = [r for r in results if r is not None]
        processed = len([r for r in successful if os.path.exists(r)])
        all_bin_results[bnt_bin] = successful
        print(f"BNT bin {bnt_bin+1} complete: {processed}/{len(file_paths)} files processed")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Processing Summary")
    print(f"{'='*60}")
    for bnt_bin in bnt_bin_numbers:
        successful = all_bin_results[bnt_bin]
        processed = len([r for r in successful if os.path.exists(r)])
        print(f"  BNT bin {bnt_bin+1}: {processed}/{len(file_paths)} files")
    print(f"{'='*60}\n")
    
    # Optionally save combined results for each bin
    if args.save_combined:
        dataset_name = "fiducial" if args.fiducial else "grid"
        map_suffix = "baryonified" if args.baryonified else "nobaryons"
        mask_suffix_combined = ""
        if args.apply_mask:
            area_tag = int(round(args.mask_area_sqdeg))
            mask_suffix_combined = f"_masked_{area_tag}sqdeg"
        
        # Process combined output for each BNT bin
        for bnt_bin in bnt_bin_numbers:
            print(f"\n{'='*60}")
            print(f"Creating combined output for BNT bin {bnt_bin+1}")
            print(f"{'='*60}")
            
            successful = all_bin_results[bnt_bin]
            
            # Generate output path for this bin
            if args.no_noise:
                combined_output = os.path.join(base_dir, f"all_bnt_peak_counts_{dataset_name}_{map_suffix}_bin{bnt_bin+1}{mask_suffix_combined}_new_normalization.npy")
            else:
                combined_output = os.path.join(base_dir, f"all_bnt_peak_counts_{dataset_name}_{map_suffix}_bin{bnt_bin+1}{mask_suffix_combined}_noisy_s{args.noise_level:.2f}_new_normalization.npy")
            
            print(f"Loading and combining {len(successful)} result files...")
            
            # Load all successful outputs for this bin
            all_peak_counts = []
            skipped_files = 0
            empty_files = []
            corrupted_files = []
            
            for file_path in tqdm(successful, desc=f"Loading BNT bin {bnt_bin+1}"):
                try:
                    # Check file size first
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        skipped_files += 1
                        empty_files.append(file_path)
                        if args.verbose:
                            print(f"Skipping empty file: {file_path}")
                        else:
                            print(f"Skipping empty file: {file_path}")
                        continue
                    
                    data = np.load(file_path, allow_pickle=True)
                    if len(data.shape) == 2:  # Validate shape
                        all_peak_counts.append(data)
                    else:
                        skipped_files += 1
                        corrupted_files.append(file_path)
                        if args.verbose:
                            print(f"Skipping {file_path} due to unexpected shape {data.shape}")
                except Exception as e:
                    skipped_files += 1
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else "File not found"
                    corrupted_files.append(file_path)
                    print(f"Error loading {file_path} (size: {file_size} bytes): {e}")
            
            # Convert list to numpy array
            if all_peak_counts:
                all_peak_counts = np.stack(all_peak_counts, axis=0)
                print(f"Combined shape: {all_peak_counts.shape}")
                
                # Save combined array
                np.save(combined_output, all_peak_counts)
                print(f"Saved combined BNT peak counts to: {os.path.basename(combined_output)}")
                
                if skipped_files > 0 and args.verbose:
                    print(f"Note: {skipped_files} files were skipped during combination.")
                
                # Print summary of problematic files
                if empty_files:
                    print(f"\nEmpty files ({len(empty_files)}):")
                    for empty_file in empty_files:
                        print(f"  {empty_file}")
                
                if corrupted_files:
                    print(f"\nCorrupted files ({len(corrupted_files)}):")
                    for corrupted_file in corrupted_files:
                        print(f"  {corrupted_file}")
                        
            else:
                print(f"No valid BNT peak counts files found for bin {bnt_bin+1} combined output!")


if __name__ == "__main__":
    main()
