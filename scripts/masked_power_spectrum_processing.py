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


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """
    Adds shape noise to a full-sky Healpix convergence (kappa) map.
    """
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise

def create_fixed_disk_mask(nside=512, target_area_sqdeg=14300.0, center_coords=(0, 90)):
    """
    Creates a reproducible contiguous disk mask of a specific area.
    
    Parameters:
    - nside: int, the HEALPix resolution.
    - target_area_sqdeg: float, the desired unmasked area in square degrees.
    - center_coords: tuple (lon, lat) in degrees for the disk center. (0, 90) is the North Pole.
    
    Returns:
    - mask: np.ndarray, the binary HEALPix mask (1=unmasked, 0=masked)
    - f_sky: float, the fraction of the sky that is unmasked.
    - angular_radius_deg: float, the calculated angular radius of the patch in degrees.
    """
    import healpy as hp
    import numpy as np

    # 1. Calculate the required angular radius (theta) in degrees
    total_area_sqdeg = 41252.96125 # 4*pi*(180/pi)^2
    
    # Calculate angular radius in radians: theta = arccos(1 - A / (2*pi*R^2))
    # Area of cap in steradians = A_deg * (pi/180)^2
    angular_radius_rad = np.arccos(1 - (target_area_sqdeg / total_area_sqdeg) * 2)
    angular_radius_deg = np.rad2deg(angular_radius_rad) # ~44.86 degrees
    
    # 2. Get the pixels within that angular radius of the center
    # center is (lon, lat) in degrees, converted to HEALPix's (theta, phi)
    theta_center = np.deg2rad(90 - center_coords[1])  # 90-latitude gives theta (colatitude)
    phi_center = np.deg2rad(center_coords[0])         # longitude gives phi
    
    center_vec = hp.ang2vec(theta_center, phi_center)
    
    # Pixels inside the disk (radius in radians)
    disc_pixels = hp.query_disc(nside, center_vec, angular_radius_rad)
    
    # 3. Create the mask
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.int8)
    mask[disc_pixels] = 1 # Set pixels inside the disc to 1 (unmasked)
    
    # 4. Calculate the actual f_sky (for correction)
    f_sky = mask.sum() / npix

    return mask, f_sky, angular_radius_deg


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


def process_file(file_path, bin_number=2, noise_level=0.26, add_noise=True, 
                 lmax=1024, verbose=False):
    """Process a single file: extract kappa map, apply mask, compute power spectrum, save results."""
    
    # Define output filename based on bin number, noise level, and mask area
    mask_suffix = "_masked_14300sqdeg"
    if add_noise:
        suffix = f"_cls_bin{bin_number}{mask_suffix}_noisy_s{noise_level:.2f}.npy"
    else:
        suffix = f"_cls_bin{bin_number}{mask_suffix}.npy"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Map key based on bin number
    map_key = f"kg/stage3_lensing{bin_number}"
    
    # Skip if file already exists
    if os.path.exists(save_path):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, Cls file already exists.")
        return save_path
    
    try:
        # Load kappa map
        with h5py.File(file_path, "r") as f:
            kg = np.array(f[map_key])
        
        # Add shape noise if requested
        if add_noise:
            kg = add_shape_noise(kg, sigma_e=noise_level)

        # Generate the fixed mask (14300 sq deg at North Pole)
        mask, f_sky, radius_deg = create_fixed_disk_mask(nside=512, 
                                                         target_area_sqdeg=14300.0,
                                                         center_coords=(0.0, 90.0))
        # Apply the mask
        kg_masked = kg * mask
        
        # Compute power spectrum of masked map
        cls = get_power_spectrum(kg_masked, lmax=lmax)

        # Save results
        np.save(save_path, cls)
        if verbose:
            print(f"Processed (masked f_sky={f_sky:.4f}): {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(description="Process HEALPix maps to compute Power Spectra (Cls).")
    
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
    
    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Map key: kg/stage3_lensing{args.bin_number}")
    print(f"Applying disk mask: 14300 sq deg at North Pole")
    
    # Determine suffix for output files (including mask info)
    mask_suffix = "_masked_14300sqdeg"
    if args.no_noise:
        suffix = f"_cls_bin{args.bin_number}{mask_suffix}.npy"
    else:
        suffix = f"_cls_bin{args.bin_number}{mask_suffix}_noisy_s{args.noise_level:.2f}.npy"
    print(f"Output suffix: {suffix}")
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=args.num_workers) as pool:
        process_func = partial(
            process_file,
            bin_number=args.bin_number,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
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
        combined_output = args.combined_output
        if not combined_output:
            dataset_name = "fiducial" if args.fiducial else "grid"
            map_suffix = "baryonified" if args.baryonified else "nobaryons"
            mask_info = "_masked_14300sqdeg"
            if args.no_noise:
                combined_output = os.path.join(base_dir, f"all_cls_{dataset_name}_{map_suffix}_bin{args.bin_number}{mask_info}.npy")
            else:
                combined_output = os.path.join(base_dir, f"all_cls_{dataset_name}_{map_suffix}_bin{args.bin_number}{mask_info}_noisy_s{args.noise_level:.2f}.npy")
        
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
            print(f"Saved combined masked Cls to: {os.path.basename(combined_output)}")
            
            if skipped_files > 0 and args.verbose:
                print(f"Note: {skipped_files} files were skipped during combination.")
        else:
            print("No valid Cls files found for combined output!")


if __name__ == "__main__":
    main()