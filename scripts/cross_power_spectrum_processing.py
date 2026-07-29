#!/usr/bin/env python3
"""
Cross Power Spectrum Processing Script - Processes cosmological data files to compute cross power spectra between different redshift bins.
"""

import os
import h5py
import healpy as hp
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import combinations


MASK_CACHE = {}


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


def get_cached_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0), apodization_deg=0.0):
    """Return a cached Euclid-like mask to avoid recomputation in each worker.

    Added apodization support to reduce mode-coupling from sharp mask edges.
    The cache key includes nside, target area, centre coordinates and apodization width.
    """
    # Normalize inputs for a stable cache key
    lon = float(center_coords[0])
    lat = float(center_coords[1])
    key = (int(nside), float(target_area_sqdeg), round(lon, 6), round(lat, 6), float(apodization_deg))

    if key not in MASK_CACHE:
        mask, f_sky, angular_radius_deg = create_euclid_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
        )

        # Apply apodization if requested (smooth the mask edges)
        if apodization_deg and apodization_deg > 0.0:
            mask = apodize_mask(mask, nside, center_coords, angular_radius_deg, apodization_deg)
            f_sky = float(mask.mean())

        MASK_CACHE[key] = (mask, f_sky, angular_radius_deg)

    return MASK_CACHE[key]


def apodize_mask(mask, nside, center_coords, angular_radius_deg, apodization_deg):
    """Apply a cosine apodization to the binary disk mask.

    The apodization is applied over an inner boundary of width `apodization_deg` (degrees).
    Pixels with angular distance <= (angular_radius_deg - apodization_deg) are set to 1.
    Pixels with angular distance >= angular_radius_deg are set to 0.
    In between we apply a cosine taper.
    """
    if apodization_deg <= 0.0:
        return mask

    npix = hp.nside2npix(nside)
    # get pixel vectors and compute angular distance to centre
    pix_vecs = np.vstack(hp.pix2vec(nside, np.arange(npix)))  # shape (3, npix)
    theta_center = np.deg2rad(90.0 - center_coords[1])
    phi_center = np.deg2rad(center_coords[0])
    center_vec = hp.ang2vec(theta_center, phi_center)

    dots = np.dot(center_vec, pix_vecs)
    dots = np.clip(dots, -1.0, 1.0)
    ang_rad = np.arccos(dots)
    ang_deg = np.rad2deg(ang_rad)

    inner = angular_radius_deg - apodization_deg
    inner = max(0.0, inner)

    # Create new mask float array
    new_mask = np.zeros_like(mask, dtype=np.float32)
    inside_inner = ang_deg <= inner
    outside = ang_deg >= angular_radius_deg
    transition = (~inside_inner) & (~outside)

    new_mask[inside_inner] = 1.0

    # cosine taper between inner and angular_radius_deg
    if np.any(transition):
        x = (ang_deg[transition] - inner) / (angular_radius_deg - inner)
        # x in [0,1], taper = 0.5*(1+cos(pi*x)) from 1 -> 0
        taper = 0.5 * (1.0 + np.cos(np.pi * x))
        new_mask[transition] = taper.astype(np.float32)

    return new_mask


def get_cross_power_spectra(maps_dict, lmax=1024):
    """
    Computes auto and cross power spectra for multiple redshift bins.
    
    Parameters:
    - maps_dict: dict, dictionary with bin numbers as keys and HEALPix maps as values
    - lmax: int, the maximum multipole to compute
    
    Returns:
    - cls_dict: dict, dictionary with (bin_i, bin_j) tuples as keys and Cls as values
    """
    # Convert all maps to alms first
    alms_dict = {}
    for bin_num, map_data in maps_dict.items():
        alms_dict[bin_num] = hp.map2alm(map_data, lmax=lmax)
    
    # Compute all auto and cross power spectra
    cls_dict = {}
    bin_numbers = sorted(maps_dict.keys())
    
    # Auto power spectra (diagonal elements)
    for bin_num in bin_numbers:
        cls_dict[(bin_num, bin_num)] = hp.alm2cl(alms_dict[bin_num])
    
    # Cross power spectra (off-diagonal elements)
    for bin_i, bin_j in combinations(bin_numbers, 2):
        cls_dict[(bin_i, bin_j)] = hp.alm2cl(alms_dict[bin_i], alms_dict[bin_j])
    
    return cls_dict


def aggregate_for_inference(processed_files, output_dir, bin_range=[1, 2, 3, 4], 
                           dataset_type="grid", map_type="nobaryons", 
                           noise_level=0.26, add_noise=True, lmax=1024,
                           verbose=False, apply_mask=False, mask_area_sqdeg=None):
    """
    Aggregate processed .npz files into inference-ready format.
    
    Creates separate .npy files for:
    - Each auto power spectrum (all_cls_<dataset>_<map>_bin<N>.npy)
    - Combined cross power spectra (all_cross_cls_<dataset>_<map>_bins<1234>.npy)
    
    Parameters:
    - processed_files: list of paths to processed .npz files
    - output_dir: directory to save aggregated files
    - bin_range: list of bin numbers
    - dataset_type: "grid" or "fiducial"
    - map_type: "nobaryons" or "baryonified"
    - noise_level: noise level used in processing
    - add_noise: whether noise was added
    - apply_mask: whether a sky mask was applied
    - mask_area_sqdeg: area of the sky mask in square degrees
    - verbose: print detailed information
    """
    from itertools import combinations
    
    print(f"\n{'='*60}")
    print(f"Aggregating {len(processed_files)} files for inference...")
    print(f"{'='*60}")
    
    # Initialize storage for each spectrum type
    auto_spectra = {bin_num: [] for bin_num in bin_range}
    cross_spectra = {(i, j): [] for i, j in combinations(bin_range, 2)}
    
    # Track issues for detailed reporting
    failed_files = []
    incomplete_files = {}  # file_path -> list of missing keys
    
    # Load all files and extract spectra
    for file_path in tqdm(processed_files, desc="Loading files"):
        try:
            data = np.load(file_path, allow_pickle=True)
            
            missing_keys = []
            file_is_complete = True
            
            # Extract auto spectra
            for bin_num in bin_range:
                key = f"cls_{bin_num}_{bin_num}"
                if key in data.files:
                    auto_spectra[bin_num].append(data[key])
                else:
                    missing_keys.append(key)
                    file_is_complete = False
            
            # Extract cross spectra
            for i, j in combinations(bin_range, 2):
                key = f"cls_{i}_{j}"
                if key in data.files:
                    cross_spectra[(i, j)].append(data[key])
                else:
                    missing_keys.append(key)
                    file_is_complete = False
            
            # Track incomplete files
            if not file_is_complete:
                incomplete_files[file_path] = missing_keys
            
        except Exception as e:
            failed_files.append((file_path, str(e)))
    
    # Detailed error reporting
    if failed_files or incomplete_files:
        print(f"\n{'='*60}")
        print("⚠️  ISSUES DETECTED")
        print(f"{'='*60}")
        
        if failed_files:
            print(f"\n❌ Failed to load {len(failed_files)} files:")
            for file_path, error in failed_files:
                print(f"  • {file_path}")
                print(f"    Error: {error}")
        
        if incomplete_files:
            print(f"\n⚠️  {len(incomplete_files)} files with missing keys:")
            for file_path, missing in incomplete_files.items():
                print(f"  • {file_path}")
                print(f"    Missing: {', '.join(missing)}")
        
        print(f"\n{'='*60}")
        
        # Save detailed log
        log_file = os.path.join(output_dir, "aggregation_issues.log")
        with open(log_file, 'w') as f:
            f.write("Aggregation Issues Report\n")
            f.write("="*60 + "\n\n")
            
            if failed_files:
                f.write(f"Failed to load {len(failed_files)} files:\n")
                for file_path, error in failed_files:
                    f.write(f"  {file_path}\n")
                    f.write(f"    Error: {error}\n\n")
            
            if incomplete_files:
                f.write(f"\n{len(incomplete_files)} files with missing keys:\n")
                for file_path, missing in incomplete_files.items():
                    f.write(f"  {file_path}\n")
                    f.write(f"    Missing: {', '.join(missing)}\n\n")
        
        print(f"Detailed log saved to: {log_file}")
    
    # Determine noise and lmax suffixes
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg)) if mask_area_sqdeg else "mask"
        mask_suffix = f"_masked_{area_tag}sqdeg"
    noise_suffix = f"_noisy_s{noise_level:.2f}" if add_noise else ""
    lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    # Save individual auto power spectra
    print("\nSaving auto power spectra...")
    for bin_num in bin_range:
        if auto_spectra[bin_num]:
            # Stack into array: shape (n_files, n_multipoles)
            auto_array = np.array(auto_spectra[bin_num])
            
            # Create filename: all_cls_<dataset>_<map>_bin<N>.npy (include noise/lmax suffixes)
            filename = f"all_cls_{dataset_type}_{map_type}_bin{bin_num}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
            output_path = os.path.join(output_dir, filename)
            
            np.save(output_path, auto_array)
            created_files.append(output_path)
            
            print(f"  ✓ Bin {bin_num}: {filename} - shape {auto_array.shape}")
        else:
            print(f"  ✗ Bin {bin_num}: No data found")
    
    # Save combined cross power spectra
    print("\nSaving cross power spectra...")
    cross_data_parts = []
    cross_pairs_found = []
    
    for i, j in combinations(sorted(bin_range), 2):
        if cross_spectra[(i, j)]:
            cross_array = np.array(cross_spectra[(i, j)])
            cross_data_parts.append(cross_array)
            cross_pairs_found.append((i, j))
            print(f"  ✓ Cross ({i},{j}): shape {cross_array.shape}")
        else:
            print(f"  ✗ Cross ({i},{j}): No data found")
    
    if cross_data_parts:
        # Concatenate all cross spectra along spectrum dimension (axis=1)
        # Shape: (n_files, total_cross_spectrum_length)
        cross_combined = np.concatenate(cross_data_parts, axis=1)
        
        # Create filename: all_cross_cls_<dataset>_<map>_bins<1234>.npy (include noise/lmax suffixes)
        bin_str = "".join(map(str, bin_range))
        filename = f"all_cross_cls_{dataset_type}_{map_type}_bins{bin_str}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        output_path = os.path.join(output_dir, filename)
        
        np.save(output_path, cross_combined)
        created_files.append(output_path)
        
        print(f"\n  ✓ Combined cross spectra: {filename}")
        print(f"    Shape: {cross_combined.shape}")
        print(f"    Pairs included: {cross_pairs_found}")
    else:
        print("\n  ✗ No cross spectra data found")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Aggregation complete!")
    print(f"  Files created: {len(created_files)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return created_files


def process_file(file_path, bin_range=[1, 2, 3, 4], noise_level=0.26, add_noise=True, 
                 lmax=1024, cross_only=False, verbose=False, apply_mask=False,
                 mask_area_sqdeg=14000.0, mask_center=(0.0, 90.0)):
    """Process a single file: extract kappa maps, apply optional mask, compute spectra."""
    
    # Define output filename
    bin_str = "".join(map(str, bin_range))
    # include lmax in per-file suffix when different from default
    lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg)) if mask_area_sqdeg else "mask"
        mask_suffix = f"_masked_{area_tag}sqdeg"
    if add_noise:
        if cross_only:
            suffix = f"_cross_cls_bins{bin_str}{mask_suffix}_noisy_s{noise_level:.2f}{lmax_suffix}.npz"
        else:
            suffix = f"_all_cls_bins{bin_str}{mask_suffix}_noisy_s{noise_level:.2f}{lmax_suffix}.npz"
    else:
        if cross_only:
            suffix = f"_cross_cls_bins{bin_str}{mask_suffix}{lmax_suffix}.npz"
        else:
            suffix = f"_all_cls_bins{bin_str}{mask_suffix}{lmax_suffix}.npz"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if file already exists
    if os.path.exists(save_path):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, cross-Cls file already exists.")
        return save_path
    
    try:
        # Load multiple kappa maps
        maps_dict = {}
        missing_bins = []
        
        mask_tuple = None
        with h5py.File(file_path, "r") as f:
            for bin_num in bin_range:
                map_key = f"kg/stage3_lensing{bin_num}"
                if map_key in f:
                    kg = np.array(f[map_key])
                    
                    # Add shape noise if requested
                    if add_noise:
                        kg = add_shape_noise(kg, sigma_e=noise_level)

                    if apply_mask:
                        if mask_tuple is None:
                            nside = hp.get_nside(kg)
                            mask_tuple = get_cached_mask(
                                nside=nside,
                                target_area_sqdeg=mask_area_sqdeg,
                                center_coords=mask_center,
                            )
                        mask = mask_tuple[0]
                        kg = kg * mask
                    
                    maps_dict[bin_num] = kg
                else:
                    missing_bins.append(bin_num)
                    if verbose:
                        print(f"Warning: {map_key} not found in {os.path.basename(file_path)}")
        
        if not maps_dict:
            if verbose:
                print(f"No valid maps found in {os.path.basename(file_path)}")
            return None
        
        if missing_bins and verbose:
            print(f"Missing bins {missing_bins} in {os.path.basename(file_path)}, proceeding with available bins")
        
        # Compute cross power spectra
        cls_dict = get_cross_power_spectra(maps_dict, lmax=lmax)
        
        # Filter to cross-correlations only if requested
        if cross_only:
            cls_dict = {(i, j): cls for (i, j), cls in cls_dict.items() if i != j}
        
        # Prepare data for saving
        save_dict = {}
        for (i, j), cls in cls_dict.items():
            save_dict[f"cls_{i}_{j}"] = cls
        
        # Add metadata
        save_dict['bin_range'] = np.array(list(maps_dict.keys()))
        save_dict['lmax'] = lmax
        if apply_mask and mask_tuple is not None:
            mask, f_sky, mask_radius = mask_tuple
            save_dict['mask_area_sqdeg'] = float(mask_area_sqdeg)
            save_dict['mask_f_sky'] = float(f_sky)
            save_dict['mask_center_lon_lat_deg'] = np.array(mask_center, dtype=np.float64)
            save_dict['mask_angular_radius_deg'] = float(mask_radius)
        if missing_bins:
            save_dict['missing_bins'] = np.array(missing_bins)
        
        # Save results as compressed npz file
        np.savez_compressed(save_path, **save_dict)
        
        if verbose:
            spectra_type = "cross" if cross_only else "auto+cross"
            print(f"Processed: {os.path.basename(file_path)} -> {os.path.basename(save_path)} ({len(cls_dict)} {spectra_type} spectra)")
        
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(description="Process HEALPix maps to compute cross power spectra between redshift bins.")
    
    # Main processing options
    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", 
                        help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")
    parser.add_argument("--bin-range", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="List of bins to include in cross-correlation analysis (default: 1 2 3 4).")
    parser.add_argument("--cross-only", action="store_true",
                        help="Only compute cross power spectra, exclude auto power spectra.")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")

    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before computing spectra.")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the Euclid-like mask in square degrees (default: 14000).")
    parser.add_argument("--mask-center", type=float, nargs=2, metavar=("LON", "LAT"),
                        default=(0.0, 90.0),
                        help="Mask centre in Galactic-like (lon, lat) degrees (default: 0 90).")
    
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
                        help="Create a summary file listing all processed files.")
    parser.add_argument("--combined-output", 
                        help="Path for combined summary file.")
    
    # Aggregation for inference
    parser.add_argument("--aggregate-for-inference", action="store_true",
                        help="Aggregate processed files into inference-ready format (separate auto and combined cross .npy files).")
    parser.add_argument("--inference-output-dir", type=str,
                        help="Output directory for inference-ready files. Defaults to base_dir/new_grid or base_dir for fiducial.")
    
    args = parser.parse_args()

    mask_center = tuple(args.mask_center)
    args.mask_center = mask_center
    
    # Validate bin range
    if len(args.bin_range) < 2 and args.cross_only:
        print("Error: Need at least 2 bins for cross-correlation analysis.")
        return
    
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
    print(f"Computing cross power spectra for bins: {args.bin_range}")

    mask_info = None
    if args.apply_mask:
        mask_info = get_cached_mask(
            nside=512,
            target_area_sqdeg=args.mask_area_sqdeg,
            center_coords=mask_center,
        )
        _, mask_f_sky, mask_radius = mask_info
        print(
            "Applying Euclid-like mask: "
            f"{args.mask_area_sqdeg:.0f} sq deg (f_sky≈{mask_f_sky:.3f}, radius≈{mask_radius:.2f}°) "
            f"centered at lon={mask_center[0]:.1f}°, lat={mask_center[1]:.1f}°"
        )
    
    if args.cross_only:
        cross_pairs = list(combinations(args.bin_range, 2))
        print(f"Cross-correlation pairs: {cross_pairs}")
    else:
        auto_pairs = [(b, b) for b in args.bin_range]
        cross_pairs = list(combinations(args.bin_range, 2))
        print(f"Auto power spectra: {auto_pairs}")
        print(f"Cross power spectra: {cross_pairs}")
    
    # Determine suffix for output files
    bin_str = "".join(map(str, args.bin_range))
    mask_suffix = ""
    if args.apply_mask:
        area_tag = int(round(args.mask_area_sqdeg)) if args.mask_area_sqdeg else "mask"
        mask_suffix = f"_masked_{area_tag}sqdeg"
    lmax_suffix = f"_lmax{args.lmax}" if args.lmax != 1024 else ""
    if args.no_noise:
        if args.cross_only:
            suffix = f"_cross_cls_bins{bin_str}{mask_suffix}{lmax_suffix}.npz"
        else:
            suffix = f"_all_cls_bins{bin_str}{mask_suffix}{lmax_suffix}.npz"
    else:
        if args.cross_only:
            suffix = f"_cross_cls_bins{bin_str}{mask_suffix}_noisy_s{args.noise_level:.2f}{lmax_suffix}.npz"
        else:
            suffix = f"_all_cls_bins{bin_str}{mask_suffix}_noisy_s{args.noise_level:.2f}{lmax_suffix}.npz"
    
    print(f"Output suffix: {suffix}")
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
        process_func = partial(
            process_file,
            bin_range=args.bin_range,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            cross_only=args.cross_only,
            verbose=args.verbose,
            apply_mask=args.apply_mask,
            mask_area_sqdeg=args.mask_area_sqdeg,
            mask_center=mask_center,
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
    
    # Optionally save combined summary
    if args.save_combined and successful:
        combined_output = args.combined_output
        if not combined_output:
            dataset_name = "fiducial" if args.fiducial else "grid"
            map_suffix = "baryonified" if args.baryonified else "nobaryons"
            
            bin_str = "".join(map(str, args.bin_range))
            spectra_type = "cross" if args.cross_only else "all"
            
            if args.no_noise:
                combined_output = os.path.join(
                    base_dir,
                    f"{spectra_type}_cls_summary_{dataset_name}_{map_suffix}_bins{bin_str}{mask_suffix}{lmax_suffix}.txt"
                )
            else:
                combined_output = os.path.join(
                    base_dir,
                    f"{spectra_type}_cls_summary_{dataset_name}_{map_suffix}_bins{bin_str}{mask_suffix}_noisy_s{args.noise_level:.2f}{lmax_suffix}.txt"
                )
        
        print(f"Creating summary file: {os.path.basename(combined_output)}")
        
        with open(combined_output, 'w') as f:
            f.write("# Cross Power Spectrum Processing Summary\n")
            f.write(f"# Dataset: {dataset_type} {map_type}\n")
            f.write(f"# Bins: {args.bin_range}\n")
            f.write(f"# Cross-only: {args.cross_only}\n")
            f.write(f"# Noise level: {args.noise_level if not args.no_noise else 'None'}\n")
            f.write(f"# Lmax: {args.lmax}\n")
            f.write(f"# Mask applied: {args.apply_mask}\n")
            if args.apply_mask and mask_info is not None:
                _, mask_f_sky, mask_radius = mask_info
                f.write(f"# Mask area (sq deg): {args.mask_area_sqdeg}\n")
                f.write(f"# Mask f_sky: {mask_f_sky}\n")
                f.write(f"# Mask angular radius (deg): {mask_radius}\n")
                f.write(f"# Mask centre (lon, lat): {mask_center}\n")
            f.write(f"# Total files processed: {processed}/{len(file_paths)}\n")
            f.write("# Processed files:\n")
            
            for file_path in successful:
                if os.path.exists(file_path):
                    f.write(f"{file_path}\n")
        
        print(f"Summary saved to: {os.path.basename(combined_output)}")
        
        # Also save an example of how to load the data
        example_file = combined_output.replace('.txt', '_example_load.py')
        with open(example_file, 'w') as f:
            f.write('''# Example of how to load and use the cross power spectra files
import numpy as np
import glob

# Load a single file
filename = "your_file_cross_cls_bins1234_masked_14000sqdeg_noisy_s0.26_lmax1024.npz"  # remove the masked/lmax parts if not applicable
data = np.load(filename)

# Print available power spectra
print("Available power spectra:")
for key in data.files:
    if key.startswith('cls_'):
        print(f"  {key}: {data[key].shape}")

# Access specific cross power spectra
''')
            if args.cross_only:
                cross_pairs = list(combinations(args.bin_range, 2))
                for i, j in cross_pairs:
                    f.write(f"cross_{i}_{j} = data['cls_{i}_{j}']  # Cross power spectrum between bins {i} and {j}\n")
            else:
                f.write("# Auto power spectra:\n")
                for b in args.bin_range:
                    f.write(f"auto_{b}_{b} = data['cls_{b}_{b}']  # Auto power spectrum of bin {b}\n")
                f.write("\n# Cross power spectra:\n")
                cross_pairs = list(combinations(args.bin_range, 2))
                for i, j in cross_pairs:
                    f.write(f"cross_{i}_{j} = data['cls_{i}_{j}']  # Cross power spectrum between bins {i} and {j}\n")
            
            f.write('''
# Load metadata
bin_range = data['bin_range']
lmax = data['lmax']
if 'missing_bins' in data.files:
    missing_bins = data['missing_bins']

# Load multiple files
pattern = "path/to/files/*cross_cls_bins1234*.npz"
files = glob.glob(pattern)
all_data = []
for file in files:
    data = np.load(file)
    all_data.append(data)
''')
        
        print(f"Example loading script saved to: {os.path.basename(example_file)}")
    
    # Aggregate for inference if requested
    if args.aggregate_for_inference and successful:
        print(f"\n{'='*60}")
        print("Starting aggregation for inference pipeline...")
        print(f"{'='*60}")
        
        # Determine output directory
        if args.inference_output_dir:
            inference_output_dir = args.inference_output_dir
        elif args.fiducial:
            inference_output_dir = base_dir  # For fiducial, save in cosmo_fiducial directory
        else:
            inference_output_dir = base_dir  # For grid, save in new_grid directory
        
        # Call aggregation function
        created_files = aggregate_for_inference(
            processed_files=successful,
            output_dir=inference_output_dir,
            bin_range=args.bin_range,
            dataset_type=dataset_type,
            map_type=map_type,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            verbose=args.verbose,
            apply_mask=args.apply_mask,
            mask_area_sqdeg=args.mask_area_sqdeg,
        )
        
        if created_files:
            print("\n✓ Inference-ready files created:")
            for f in created_files:
                print(f"  - {os.path.basename(f)}")
            print(f"\nThese files are ready to use with run_npe_inference_auto_cross_ps.py")
        else:
            print("\n✗ No inference files were created (check for errors above)")


if __name__ == "__main__":
    main()
