#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/bnt_l1_norm_processing.py
"""
BNT L1 Norm Processing Script - Applies BNT transform to cosmological maps and computes L1 norms.
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
from pycs.sparsity.mrs.mrs_starlet import mrs_uwttrans

# BNT transformation matrix
BNT_MATRIX = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [-1.        ,  1.        ,  0.        ,  0.        ],
                       [ 0.4521097 , -1.4521097 ,  1.        ,  0.        ],
                       [ 0.        ,  0.25127807, -1.251278  ,  1.        ]])

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


def get_norm_wtl1_sphere(Map, nscales, nbins=None, Mask=None, min_snr=None, max_snr=None, path="/."):
    """
    Computes L1 norms of wavelet transform coefficients at different scales.
    """
    # Set default for nbins if not provided
    if nbins is None:
        nbins = 40

    # Perform undecimated wavelet transform on the spherical map
    # Use suppress_stdout to hide the "setting output map dtype" messages
    with suppress_stdout():
        WT = mrs_uwttrans(Map, nscale=nscales, verbose=False, path=path)

    # Initialize lists to collect the l1 norm and bins
    l1norm_coll = []
    bins_coll = []

    # Loop through each scale of the wavelet transform
    for i in range(nscales):
        ScaleCoeffs = WT[i]  # Accessing the coefficients for the i-th scale

        # Apply the mask if provided
        if Mask is not None:
            ScaleCoeffs = ScaleCoeffs[Mask != 0]

        # Normalize the wavelet scale to the same energy level
        energy = np.sum(ScaleCoeffs**2)
        normalization_factor = np.sqrt(energy)
        if normalization_factor > 0:
            ScaleCoeffs_normalized = ScaleCoeffs / normalization_factor 
        
        # Set the minimum and maximum values based on inputs or defaults
        min_val = min_snr if min_snr is not None else np.min(ScaleCoeffs_normalized)
        max_val = max_snr if max_snr is not None else np.max(ScaleCoeffs_normalized)

        # Define thresholds and bins
        thresholds = np.linspace(min_val, max_val, nbins + 1)
        bins = 0.5 * (thresholds[:-1] + thresholds[1:])

        # Digitize the values into bins
        digitized = np.digitize(ScaleCoeffs_normalized, thresholds)

        # Calculate the l1 norm for each bin
        bin_l1_norm = [
            np.sum(np.abs(ScaleCoeffs_normalized[digitized == j]))
            for j in range(1, len(thresholds))
        ]

        # Store the bins and l1 norms for this scale
        bins_coll.append(bins)
        l1norm_coll.append(bin_l1_norm)

    # Return the bins and l1 norms for each scale
    return np.array(bins_coll), np.array(l1norm_coll)


def process_file(file_path, bnt_bin=3, noise_level=0.26, add_noise=True,
                min_snr=-0.003, max_snr=0.003, temp_dir=None, verbose=False):
    """
    Process a single file: extract kappa maps for all bins, apply BNT transform, 
    compute L1 norms for the specified BNT bin, and save results.
    """
    
    # Define output filename based on BNT bin number and noise level
    if add_noise:
        suffix = f"_bnt_l1_norms_bin{bnt_bin+1}_noisy_s{noise_level:.2f}.npy"
    else:
        suffix = f"_bnt_l1_norms_bin{bnt_bin+1}.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if file already exists
    if os.path.exists(save_path):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, BNT L1 norm file already exists.")
        return save_path
    
    try:
        # Load all 4 kappa maps
        kgs = []
        with h5py.File(file_path, "r") as f:
            for i in range(4):
                map_key = f"kg/stage3_lensing{i+1}"
                kgs.append(np.array(f[map_key]))
        
        # Add shape noise if requested (before BNT transform)
        if add_noise:
            kgs = [add_shape_noise(kg, sigma_e=noise_level) for kg in kgs]
        
        # Apply BNT transform
        kgs = np.array(kgs)  # Convert to numpy array for matrix multiplication
        kgs_bnt = BNT_MATRIX @ kgs  # This is the key step from the notebook
        
        # Use shared temp directory with unique filenames or create temporary directory
        if temp_dir:
            unique_path = os.path.join(temp_dir, f"proc_{os.getpid()}_{os.path.basename(file_path)}")
            _, l1norms = get_norm_wtl1_sphere(
                kgs_bnt[bnt_bin], nscales=5, nbins=40, min_snr=min_snr, max_snr=max_snr, path=unique_path
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                _, l1norms = get_norm_wtl1_sphere(
                    kgs_bnt[bnt_bin], nscales=5, nbins=40, min_snr=min_snr, max_snr=max_snr, path=tmp_dir
                )
        
        # Save results
        np.save(save_path, l1norms)
        if verbose:
            print(f"Processed: {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
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
    parser.add_argument("--bnt-bin", type=int, default=3, 
                        help="BNT bin to analyze (0-3, default=3 which is the 4th bin)")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Algorithm parameters
    parser.add_argument("--min-snr", type=float, default=-0.003, 
                        help="Minimum SNR value.")
    parser.add_argument("--max-snr", type=float, default=0.003, 
                        help="Maximum SNR value.")
    
    # Execution options
    parser.add_argument("--num-workers", type=int, default=70,
                        help="Number of worker processes.")
    parser.add_argument("--shared-temp", action="store_true",
                        help="Use a shared temporary directory.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    # Output options
    parser.add_argument("--save-combined", action="store_true",
                        help="Save combined L1 norms to a single file.")
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
    print(f"Computing BNT L1 norms for BNT bin {args.bnt_bin+1}")
    
    # Determine suffix for output files
    if args.no_noise:
        suffix = f"_bnt_l1_norms_bin{args.bnt_bin+1}.npy"
    else:
        suffix = f"_bnt_l1_norms_bin{args.bnt_bin+1}_noisy_s{args.noise_level:.2f}.npy"
    print(f"Output suffix: {suffix}")
    
    # Create shared temp directory if requested
    temp_dir = None
    if args.shared_temp:
        temp_dir = "/tmp/mrs_bnt_temp/"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Using shared temporary directory: {temp_dir}")
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=args.num_workers) as pool:
        process_func = partial(
            process_file,
            bnt_bin=args.bnt_bin,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            temp_dir=temp_dir,
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
                combined_output = os.path.join(
                    base_dir, 
                    f"all_bnt_l1_norms_{dataset_name}_{map_suffix}_bin{args.bnt_bin+1}.npy"
                )
            else:
                combined_output = os.path.join(
                    base_dir, 
                    f"all_bnt_l1_norms_{dataset_name}_{map_suffix}_bin{args.bnt_bin+1}_noisy_s{args.noise_level:.2f}.npy"
                )
        
        print(f"Loading and combining {len(successful)} result files...")
        
        # Load all successful outputs
        all_l1_norms = []
        skipped_files = 0
        
        for file_path in tqdm(successful, desc="Loading results"):
            try:
                data = np.load(file_path, allow_pickle=True)
                if len(data.shape) == 2:  # Validate shape
                    all_l1_norms.append(data)
                else:
                    skipped_files += 1
                    if args.verbose:
                        print(f"Skipping {os.path.basename(file_path)} due to unexpected shape {data.shape}")
            except Exception as e:
                skipped_files += 1
                if args.verbose:
                    print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        # Convert list to numpy array
        if all_l1_norms:
            all_l1_norms = np.stack(all_l1_norms, axis=0)
            print(f"Combined shape: {all_l1_norms.shape}")
            
            # Save combined array
            np.save(combined_output, all_l1_norms)
            print(f"Saved combined BNT L1 norms to: {os.path.basename(combined_output)}")
            
            if skipped_files > 0 and args.verbose:
                print(f"Note: {skipped_files} files were skipped during combination.")
        else:
            print("No valid BNT L1 norm files found for combined output!")


if __name__ == "__main__":
    main()