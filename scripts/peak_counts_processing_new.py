#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/peak_counts_processing_new.py
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


def process_file(file_path, bin_number=2, noise_level=0.26, add_noise=True, 
                noise_std=0.0146, nbins=31, min_val=-2, max_val=6, verbose=False):
    """Process a single file: extract kappa map, compute peak counts, save results."""
    
    # Define output filename based on bin number and noise level
    if add_noise:
        suffix = f"_peak_counts_bin{bin_number}_noisy_s{noise_level:.2f}_new_normalization.npy"
    else:
        suffix = f"_peak_counts_bin{bin_number}_new_normalization.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Map key based on bin number
    map_key = f"kg/stage3_lensing{bin_number}"
    
    # Skip if file already exists
    if os.path.exists(save_path):
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
        
        peak_counts, _ = get_wtpeaks_sphere(
            kg, nscales=5, noise_std=noise_std, nbins=nbins, Min=min_val, Max=max_val, verbose=False
        )
        
        # Convert to numpy array for consistent saving
        peak_counts = np.array(peak_counts)
        
        # Save results
        np.save(save_path, peak_counts)
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
    parser.add_argument("--bin-number", type=int, default=1, 
                        help="Bin number (used for map_key 'kg/stage3_lensing{bin_number}' and suffix)")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Algorithm parameters
    parser.add_argument("--noise-std", type=float, default=0.0146,
                        help="Noise standard deviation for wavelet normalization.")
    parser.add_argument("--nbins", type=int, default=31,
                        help="Number of bins for the histogram.")
    parser.add_argument("--min-val", type=float, default=-2,
                        help="Minimum value for the histogram bins.")
    parser.add_argument("--max-val", type=float, default=6,
                        help="Maximum value for the histogram bins.")
    
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
    print(f"Map key: kg/stage3_lensing{args.bin_number}")
    
    # Determine suffix for output files
    if args.no_noise:
        suffix = f"_peak_counts_bin{args.bin_number}_new_normalization.npy"
    else:
        suffix = f"_peak_counts_bin{args.bin_number}_noisy_s{args.noise_level:.2f}_new_normalization.npy"
    print(f"Output suffix: {suffix}")
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=args.num_workers) as pool:
        process_func = partial(
            process_file,
            bin_number=args.bin_number,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            noise_std=args.noise_std,
            nbins=args.nbins,
            min_val=args.min_val,
            max_val=args.max_val,
            verbose=args.verbose
        )
        results = list(tqdm(
            pool.imap(process_func, file_paths),
            total=len(file_paths),
            desc="Processing files"
        ))
    
    # Count successful files
    successful = [r for r in results if r is not None]
    processed = len([r for r in successful if os.path.exists(r)])
    print(f"Processing complete: {processed}/{len(file_paths)} files processed")
    
    # Optionally save combined results
    if args.save_combined and successful:
        # Generate default output path if not specified
        combined_output = args.combined_output
        if not combined_output:
            dataset_name = "fiducial" if args.fiducial else "grid"
            map_suffix = "baryonified" if args.baryonified else "nobaryons"
            if args.no_noise:
                combined_output = os.path.join(base_dir, f"all_peak_counts_{dataset_name}_{map_suffix}_bin{args.bin_number}_new_normalization.npy")
            else:
                combined_output = os.path.join(base_dir, f"all_peak_counts_{dataset_name}_{map_suffix}_bin{args.bin_number}_noisy_s{args.noise_level:.2f}_new_normalization.npy")
        
        print(f"Loading and combining {len(successful)} result files...")
        
        # Load all successful outputs
        all_peak_counts = []
        skipped_files = 0
        
        for file_path in tqdm(successful, desc="Loading results"):
            try:
                data = np.load(file_path, allow_pickle=True)
                if len(data.shape) == 2:  # Validate shape
                    all_peak_counts.append(data)
                else:
                    skipped_files += 1
                    if args.verbose:
                        print(f"Skipping {os.path.basename(file_path)} due to unexpected shape {data.shape}")
            except Exception as e:
                skipped_files += 1
                if args.verbose:
                    print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        # Convert list to numpy array
        if all_peak_counts:
            all_peak_counts = np.stack(all_peak_counts, axis=0)
            print(f"Combined shape: {all_peak_counts.shape}")
            
            # Save combined array
            np.save(combined_output, all_peak_counts)
            print(f"Saved combined peak counts to: {os.path.basename(combined_output)}")
            
            if skipped_files > 0 and args.verbose:
                print(f"Note: {skipped_files} files were skipped during combination.")
        else:
            print("No valid peak counts files found for combined output!")


if __name__ == "__main__":
    main()
