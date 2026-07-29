#!/usr/bin/env python3
# filepath: /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/scripts/peak_counts_processing_new.py
"""
Peak Counts Processing Script - Processes cosmological data files to compute multiscale peak counts.
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

# Global mask cache
MASK_CACHE = {}


def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    # Use a source of entropy from the OS to seed the worker
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))


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


def process_file(file_path, bin_number=2, noise_level=0.26, add_noise=True,
                noise_std=0.0146, nbins=31, min_val=-2, max_val=6, verbose=False,
                apply_mask=False, mask_area_sqdeg=14000.0, mask_center=(0.0, 90.0),
                force_overwrite=False, min_snr_coarse=100, max_snr_coarse=200,
                submean=False):
    """Process a single file: extract kappa map, apply optional mask, compute peak counts, save results.

    With ``submean`` (only meaningful for masked runs), the footprint mean is subtracted from the
    masked field before the starlet transform. The masked field steps from <kappa> down to 0 at the
    boundary, and that (cosmology-dependent) step leaks into the detail-scale peaks, spuriously
    tightening masked-peak constraints; subtracting the mean removes the step at the source. Mirrors
    the masked-PS submean. Outputs carry a ``_submean`` tag so they do not collide with the originals.
    """

    # Define output filename based on bin number, noise level, and mask
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
    submean_suffix = "_submean" if (apply_mask and submean) else ""

    if add_noise:
        suffix = f"_peak_counts_bin{bin_number}{mask_suffix}{submean_suffix}_noisy_s{noise_level:.2f}_new_normalization.npy"
    else:
        suffix = f"_peak_counts_bin{bin_number}{mask_suffix}{submean_suffix}_new_normalization.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Map key based on bin number
    map_key = f"kg/stage3_lensing{bin_number}"
    
    # Skip if file already exists (unless force_overwrite is set)
    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, peak counts file already exists.")
        return save_path
    
    try:
        # Load kappa map
        with h5py.File(file_path, "r") as f:
            kg = np.array(f[map_key])
        
        # Add shape noise if requested
        if add_noise:
            kg = add_shape_noise(kg, sigma_e=noise_level)
        
        # Apply mask if requested
        if apply_mask:
            nside = hp.get_nside(kg)
            mask, f_sky, _ = get_cached_mask(
                nside=nside,
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
            )
            kg = kg * mask
            if submean:
                # subtract the footprint mean, re-zero the exterior (kills the boundary step)
                kg = (kg - kg[mask != 0].mean()) * mask

        peak_counts, _ = get_wtpeaks_sphere(
            kg, nscales=5, noise_std=noise_std, nbins=nbins, Min=min_val, Max=max_val, verbose=False
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
    parser = argparse.ArgumentParser(description="Process HEALPix maps to compute multiscale peak counts.")
    
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
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before computing peak counts.")
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
                        help="Comma-separated min SNR values for coarse scale per bin (default: '10,40,100,150' for bins 1-4).")
    parser.add_argument("--max-snr-coarse", type=str, default="50,100,200,300",
                        help="Comma-separated max SNR values for coarse scale per bin (default: '50,100,200,300' for bins 1-4).")
    
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
    parser.add_argument("--submean", action="store_true",
                        help="Subtract the footprint mean from the masked field before the starlet "
                             "transform (fixes the spurious masked-peak tightness; mirrors the PS "
                             "submean). Requires --apply-mask; tags outputs '_submean'.")

    args = parser.parse_args()

    if args.submean and not args.apply_mask:
        print("Warning: --submean has no effect without --apply-mask; ignoring.")
        args.submean = False
    
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
    
    # Normalize mask center to tuple
    mask_center = tuple(args.mask_center)
    
    # Parse bin numbers
    if args.bins:
        bin_numbers = [int(b.strip()) for b in args.bins.split(',')]
        print(f"Processing multiple bins: {bin_numbers}")
    else:
        bin_numbers = [args.bin_number]
        print(f"Processing single bin: {args.bin_number}")

    # Parse per-bin coarse SNR ranges
    min_snr_coarse_list = [float(x.strip()) for x in args.min_snr_coarse.split(',')]
    max_snr_coarse_list = [float(x.strip()) for x in args.max_snr_coarse.split(',')]
    coarse_snr_min = {i+1: min_snr_coarse_list[i] for i in range(len(min_snr_coarse_list))}
    coarse_snr_max = {i+1: max_snr_coarse_list[i] for i in range(len(max_snr_coarse_list))}
    print(f"Coarse scale SNR ranges per bin:")
    for b in bin_numbers:
        if b in coarse_snr_min and b in coarse_snr_max:
            print(f"  Bin {b}: [{coarse_snr_min[b]}, {coarse_snr_max[b]}]")
        else:
            print(f"  Bin {b}: using defaults (no custom range specified)")

    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    if len(bin_numbers) == 1:
        print(f"Map key: kg/stage3_lensing{bin_numbers[0]}")
    else:
        print(f"Map keys: {', '.join([f'kg/stage3_lensing{b}' for b in bin_numbers])}")
    
    if args.apply_mask:
        mask_info = get_cached_mask(
            nside=512,
            target_area_sqdeg=args.mask_area_sqdeg,
            center_coords=mask_center,
        )
        _, mask_f_sky, mask_radius = mask_info
        print(
            f"Applying Euclid-like mask: "
            f"{args.mask_area_sqdeg:.0f} sq deg (f_sky≈{mask_f_sky:.3f}, radius≈{mask_radius:.2f}°) "
            f"centered at lon={mask_center[0]:.1f}°, lat={mask_center[1]:.1f}°"
        )
    
    # Clean up empty files if requested
    if args.cleanup_empty:
        for bin_number in bin_numbers:
            mask_suffix = ""
            if args.apply_mask:
                area_tag = int(round(args.mask_area_sqdeg))
                mask_suffix = f"_masked_{area_tag}sqdeg"
            
            print(f"Cleaning up empty files for bin {bin_number}...")
            empty_count = 0
            for file_path in file_paths:
                # Construct expected output path
                if not args.no_noise:
                    expected_output = file_path.replace(".h5", f"_peak_counts_bin{bin_number}{mask_suffix}_noisy_s{args.noise_level:.2f}_new_normalization.npy")
                else:
                    expected_output = file_path.replace(".h5", f"_peak_counts_bin{bin_number}{mask_suffix}_new_normalization.npy")
                
                if os.path.exists(expected_output) and os.path.getsize(expected_output) == 0:
                    os.remove(expected_output)
                    empty_count += 1
                    if args.verbose:
                        print(f"Removed empty file: {expected_output}")
            
            print(f"Removed {empty_count} empty files for bin {bin_number}")
    
    # Process each bin
    all_bin_results = {}
    for bin_number in bin_numbers:
        print(f"\n{'='*60}")
        print(f"Processing bin {bin_number}")
        print(f"{'='*60}")
        
        # Determine suffix for output files
        mask_suffix = ""
        if args.apply_mask:
            area_tag = int(round(args.mask_area_sqdeg))
            mask_suffix = f"_masked_{area_tag}sqdeg"
        
        if args.no_noise:
            suffix = f"_peak_counts_bin{bin_number}{mask_suffix}_new_normalization.npy"
        else:
            suffix = f"_peak_counts_bin{bin_number}{mask_suffix}_noisy_s{args.noise_level:.2f}_new_normalization.npy"
        print(f"Output suffix: {suffix}")
        
        # Get coarse SNR range for this bin (with fallback defaults)
        bin_min_snr_coarse = coarse_snr_min.get(bin_number, 100)
        bin_max_snr_coarse = coarse_snr_max.get(bin_number, 200)
        print(f"Using coarse scale SNR range: [{bin_min_snr_coarse}, {bin_max_snr_coarse}]")
        
        # Process files in parallel with progress bar
        with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
            process_func = partial(
                process_file,
                bin_number=bin_number,
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
                min_snr_coarse=bin_min_snr_coarse,
                max_snr_coarse=bin_max_snr_coarse,
                submean=args.submean,
            )
            results = list(tqdm(
                pool.imap(process_func, file_paths),
                total=len(file_paths),
                desc=f"Processing bin {bin_number}"
            ))
        
        # Count successful files
        successful = [r for r in results if r is not None]
        processed = len([r for r in successful if os.path.exists(r)])
        print(f"Bin {bin_number} processing complete: {processed}/{len(file_paths)} files processed")
        
        all_bin_results[bin_number] = successful
    
    print(f"\n{'='*60}")
    print(f"All bins processing complete")
    print(f"{'='*60}")
    
    # Optionally save combined results
    if args.save_combined:
        for bin_number in bin_numbers:
            successful = all_bin_results[bin_number]
            if not successful:
                print(f"No successful files for bin {bin_number}, skipping combined output")
                continue
            
            # Generate default output path if not specified
            combined_output = args.combined_output
            if not combined_output:
                dataset_name = "fiducial" if args.fiducial else "grid"
                map_suffix = "baryonified" if args.baryonified else "nobaryons"
                
                mask_suffix = ""
                if args.apply_mask:
                    area_tag = int(round(args.mask_area_sqdeg))
                    mask_suffix = f"_masked_{area_tag}sqdeg"
                submean_suffix = "_submean" if (args.apply_mask and args.submean) else ""

                if args.no_noise:
                    combined_output = os.path.join(base_dir, f"all_peak_counts_{dataset_name}_{map_suffix}_bin{bin_number}{mask_suffix}{submean_suffix}_new_normalization.npy")
                else:
                    combined_output = os.path.join(base_dir, f"all_peak_counts_{dataset_name}_{map_suffix}_bin{bin_number}{mask_suffix}{submean_suffix}_noisy_s{args.noise_level:.2f}_new_normalization.npy")
            elif len(bin_numbers) > 1:
                # If custom output is specified for multiple bins, add bin number to filename
                base, ext = os.path.splitext(combined_output)
                combined_output = f"{base}_bin{bin_number}{ext}"
            
            print(f"\nBin {bin_number}: Loading and combining {len(successful)} result files...")
            
            # Load all successful outputs
            all_peak_counts = []
            skipped_files = 0
            empty_files = []
            corrupted_files = []
            
            for file_path in tqdm(successful, desc=f"Loading bin {bin_number} results"):
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
                print(f"Bin {bin_number} combined shape: {all_peak_counts.shape}")
                
                # Save combined array
                np.save(combined_output, all_peak_counts)
                print(f"Saved combined peak counts to: {os.path.basename(combined_output)}")
                
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
                print(f"No valid peak counts files found for bin {bin_number} combined output!")


if __name__ == "__main__":
    main()
