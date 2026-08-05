#!/usr/bin/env python3
"""
Power Spectrum Processing Script - Processes cosmological data files to compute Power Spectra (Cls).
"""

import os
import h5py
import healpy as hp
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


# BNT transformation matrix
BNT_MATRIX = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [-1.        ,  1.        ,  0.        ,  0.        ],
                       [ 0.4521097 , -1.4521097 ,  1.        ,  0.        ],
                       [ 0.        ,  0.25127807, -1.251278  ,  1.        ]])


def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    # Use a source of entropy from the OS to seed the worker
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """
    Adds shape noise to a full-sky Healpix convergence (kappa) map.
    """
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise


def get_power_spectrum(Map, lmax=1024):
    """
    Computes the angular power spectrum (Cl) of a HEALPix map.
    
    Parameters:
    - Map: np.ndarray, the input HEALPix map
    - lmax: int, the maximum multipole to compute
    
    Returns:
    - Cls: np.ndarray, the angular power spectrum
    """
    alm = hp.map2alm(Map, lmax=lmax)
    Cls = hp.alm2cl(alm)
    return Cls


def process_file(file_path, bnt_bin=3, noise_level=0.26, add_noise=True, 
                 lmax=1024, verbose=False, out_dir=None):
    """
    Process a single file: extract kappa maps for all bins, apply BNT transform, 
    compute power spectrum for the specified BNT bin, and save results.
    """
    
    # Define output filename based on BNT bin number and noise level
    if add_noise:
        suffix = f"_bnt_cls_bin{bnt_bin+1}_noisy_s{noise_level:.2f}.npy"
    else:
        suffix = f"_bnt_cls_bin{bnt_bin+1}.npy"
    
    if out_dir:
        # The maps live in the READ-ONLY shared dataset, so outputs cannot go beside them.
        # The perm directory name is what makes the output unique -- every realisation shares
        # the same basename, so dropping it would collapse 200 outputs onto one file.
        perm = os.path.basename(os.path.dirname(file_path))
        base = os.path.basename(file_path).replace(".h5", suffix)
        save_path = os.path.join(out_dir, f"{perm}_{base}")
    else:
        save_path = file_path.replace(".h5", suffix)
    
    # Skip if file already exists
    if os.path.exists(save_path):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, BNT Cls file already exists.")
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
        kgs_bnt = BNT_MATRIX @ kgs
        
        # Compute power spectrum for the specified BNT bin
        cls = get_power_spectrum(kgs_bnt[bnt_bin], lmax=lmax)
        
        # Save results
        np.save(save_path, cls)
        if verbose:
            print(f"Processed: {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(description="Process HEALPix maps with BNT transform to compute Power Spectra (Cls).")
    
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
    parser.add_argument("--lmax", type=int, default=1024, 
                        help="Maximum multipole (l) for power spectrum calculation.")
    
    # Execution options
    parser.add_argument("--num-workers", type=int, default=70,
                        help="Number of worker processes.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    # Output options
    parser.add_argument("--save-combined", action="store_true",
                        help="Save combined Cls to a single file.")
    parser.add_argument("--out-dir", default=None,
                        help="write per-realisation outputs here instead of beside the input "
                             "map. Required when the maps are read-only, e.g. the shared "
                             "CosmoGridV1 dataset.")
    parser.add_argument("--combined-output", 
                        help="Path for combined output file.")
    
    args = parser.parse_args()
    
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
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
    else:
        cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
        perm_dirs = [f"perm_{i:04d}" for i in range(7)]
        file_paths = [
            os.path.join(base_dir, cosmo, perm, filename)
            for cosmo in cosmo_dirs
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
        ]
    
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Computing BNT Power Spectrum for BNT bin {args.bnt_bin+1}")
    
    # Determine suffix for output files
    if args.no_noise:
        suffix = f"_bnt_cls_bin{args.bnt_bin+1}.npy"
    else:
        suffix = f"_bnt_cls_bin{args.bnt_bin+1}_noisy_s{args.noise_level:.2f}.npy"
    print(f"Output suffix: {suffix}")
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
        process_func = partial(
            process_file,
            bnt_bin=args.bnt_bin,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            verbose=args.verbose,
            out_dir=args.out_dir
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
        combined_output = args.combined_output
        if not combined_output and args.out_dir:
            base_dir = args.out_dir      # never default the combined file into a read-only tree
        if not combined_output:
            dataset_name = "fiducial" if args.fiducial else "grid"
            map_suffix = "baryonified" if args.baryonified else "nobaryons"
            if args.no_noise:
                combined_output = os.path.join(base_dir, f"all_bnt_cls_{dataset_name}_{map_suffix}_bin{args.bnt_bin+1}.npy")
            else:
                combined_output = os.path.join(base_dir, f"all_bnt_cls_{dataset_name}_{map_suffix}_bin{args.bnt_bin+1}_noisy_s{args.noise_level:.2f}.npy")
        
        print(f"Loading and combining {len(successful)} result files...")
        
        all_cls = []
        skipped_files = 0
        
        for file_path in tqdm(successful, desc="Loading results"):
            try:
                data = np.load(file_path, allow_pickle=True)
                if len(data.shape) == 1:  # Validate shape for 1D Cl array
                    all_cls.append(data)
                else:
                    skipped_files += 1
                    if args.verbose:
                        print(f"Skipping {os.path.basename(file_path)} due to unexpected shape {data.shape}")
            except Exception as e:
                skipped_files += 1
                if args.verbose:
                    print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        if all_cls:
            all_cls = np.stack(all_cls, axis=0)
            print(f"Combined shape: {all_cls.shape}")
            
            np.save(combined_output, all_cls)
            print(f"Saved combined BNT Cls to: {os.path.basename(combined_output)}")
            
            if skipped_files > 0 and args.verbose:
                print(f"Note: {skipped_files} files were skipped during combination.")
        else:
            print("No valid BNT Cls files found for combined output!")


if __name__ == "__main__":
    main()