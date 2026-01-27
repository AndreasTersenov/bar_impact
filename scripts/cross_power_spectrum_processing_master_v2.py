#!/usr/bin/env python3
"""
Cross Power Spectrum Processing Script with MASTER Correction (Refactored Version)

This script uses the modular bar_impact package for computing cross power spectra
with proper MASTER mode-coupling correction for masked data.

Key features:
- Uses bar_impact.processing.master_correction for MASTER algorithm
- Leverages bar_impact.core.masks for C1/C2 apodization
- Deterministic random seeds for reproducibility
- Per-bin galaxy densities
- Modular, maintainable code

Requirements:
    pip install pymaster  # For MASTER correction

Key improvements over original version:
- Uses centralized MASTER correction utilities
- Leverages apodized mask creation from package
- Imports from bar_impact.* modules instead of duplicating code
- Maintains same command-line interface for backward compatibility
"""

import os
import sys
import h5py
import argparse
import hashlib
import numpy as np
import healpy as hp
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import combinations

# Import from bar_impact package
from bar_impact.constants import (
    DEFAULT_NSIDE,
    DEFAULT_MASK_AREA_SQDEG,
    DEFAULT_MASK_CENTER,
    DEFAULT_SIGMA_E,
    DEFAULT_GALAXY_DENSITY,
    DEFAULT_LMAX,
)
from bar_impact.core.masks import SurveyMask
from bar_impact.utils.noise import add_shape_noise
from bar_impact.processing.master_correction import (
    HAS_NAMASTER,
    compute_power_spectra_master,
    compute_pseudo_cls_simple,
)


def get_deterministic_seed(file_path, global_seed=42):
    """
    Generate a deterministic seed from file path and global seed.
    Ensures reproducibility across runs.
    """
    hash_input = f"{file_path}_{global_seed}".encode('utf-8')
    hash_digest = hashlib.sha256(hash_input).digest()
    seed = int.from_bytes(hash_digest[:4], byteorder='big')
    return seed % (2**32)


def seed_worker(global_seed):
    """Initializer for multiprocessing pool with deterministic seeds."""
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    np.random.seed((global_seed + worker_id) % (2**32))


def process_file(
    file_path,
    bin_range=[1, 2, 3, 4],
    noise_level=0.26,
    galaxy_densities=None,
    add_noise=True,
    lmax=1024,
    cross_only=False,
    verbose=False,
    apply_mask=False,
    mask_area_sqdeg=14000.0,
    mask_center=(0.0, 90.0),
    apodization_type='C2',
    apodization_scale_deg=2.0,
    use_namaster=True,
    bin_edges=None,
    global_seed=42,
    force_overwrite=False,
):
    """
    Process a single file: extract kappa maps, apply mask, compute MASTER-corrected spectra.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file containing kappa maps
    bin_range : list, optional
        Bins to process (1-4)
    noise_level : float, optional
        Shape noise level (sigma_e)
    galaxy_densities : list, optional
        Per-bin galaxy densities in arcmin^-2
    add_noise : bool, optional
        Whether to add shape noise
    lmax : int, optional
        Maximum multipole
    cross_only : bool, optional
        Only compute cross spectra (not auto)
    verbose : bool, optional
        Print detailed output
    apply_mask : bool, optional
        Apply survey mask
    mask_area_sqdeg : float, optional
        Mask area in square degrees
    mask_center : tuple, optional
        Mask center (lon, lat) in degrees
    apodization_type : str, optional
        'C1' or 'C2' apodization
    apodization_scale_deg : float, optional
        Apodization width in degrees
    use_namaster : bool, optional
        Use NaMaster for MASTER correction
    bin_edges : np.ndarray, optional
        Custom multipole bin edges
    global_seed : int, optional
        Global random seed
    force_overwrite : bool, optional
        Overwrite existing outputs
        
    Returns
    -------
    str or None
        Path to output file, or None if failed
    """
    # Generate deterministic seed for this file
    file_seed = get_deterministic_seed(file_path, global_seed)
    rng = np.random.default_rng(file_seed)
    
    # Default galaxy densities
    if galaxy_densities is None:
        galaxy_densities = [DEFAULT_GALAXY_DENSITY] * len(bin_range)
    
    # Construct output filename
    bin_str = "".join(map(str, bin_range))
    lmax_suffix = f"_lmax{lmax}" if lmax != DEFAULT_LMAX else ""
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        apod_tag = f"apod{apodization_scale_deg:.1f}" if apodization_scale_deg > 0 else "noapod"
        method_tag = "master" if use_namaster and HAS_NAMASTER else "pseudo"
        mask_suffix = f"_masked_{area_tag}sqdeg_{apod_tag}_{method_tag}"
    else:
        mask_suffix = "_master"
    
    noise_suffix = f"_noisy_s{noise_level:.2f}" if add_noise else ""
    spectra_type = "cross_cls" if cross_only else "all_cls"
    suffix = f"_{spectra_type}_bins{bin_str}{mask_suffix}{noise_suffix}{lmax_suffix}.npz"
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if already processed
    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)} - already processed")
        return save_path
    
    try:
        # Load kappa maps for specified bins
        maps_dict = {}
        with h5py.File(file_path, "r") as f:
            for bin_num in bin_range:
                map_key = f"kg/stage3_lensing{bin_num}"
                kg = f[map_key][()]
                
                # Add shape noise if requested
                if add_noise:
                    galaxy_density = galaxy_densities[bin_range.index(bin_num)]
                    kg = add_shape_noise(
                        kg,
                        sigma_e=noise_level,
                        galaxy_density=galaxy_density,
                        nside=DEFAULT_NSIDE,
                        rng=rng,
                    )
                
                maps_dict[bin_num] = kg
        
        # Apply mask if requested
        if apply_mask:
            mask = SurveyMask.create_apodized_disk_mask(
                nside=DEFAULT_NSIDE,
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
                apodization_deg=apodization_scale_deg,
                apodization_type=apodization_type,
                use_cache=True,
            )
            mask_array = mask.data
            f_sky = mask.f_sky
            
            # NOTE: Do NOT multiply maps by mask! NaMaster handles this internally
            # Pre-multiplying would result in double-masking (mask² effect)
        else:
            mask_array = np.ones(hp.nside2npix(DEFAULT_NSIDE), dtype=np.float32)
            f_sky = 1.0
        
        # Compute power spectra with MASTER correction
        if use_namaster and HAS_NAMASTER and apply_mask:
            cls_dict, ells = compute_power_spectra_master(
                maps_dict=maps_dict,
                mask=mask_array,
                lmax=lmax,
                bin_edges=bin_edges,
                include_auto=not cross_only,
                verbose=verbose,
            )
        else:
            # Fallback: pseudo-Cls without correction
            if apply_mask:
                # Apply mask to maps for pseudo-Cl computation
                for bin_num in maps_dict:
                    maps_dict[bin_num] = maps_dict[bin_num] * mask_array
            
            cls_dict = compute_pseudo_cls_simple(
                maps_dict,
                lmax=lmax,
                include_auto=not cross_only,
            )
            ells = np.arange(lmax + 1)
        
        # Prepare data for saving
        save_dict = {}
        for (i, j), cls in cls_dict.items():
            key = f"cl_{i}_{j}"
            save_dict[key] = cls
        
        # Add metadata
        save_dict['bin_range'] = np.array(bin_range)
        save_dict['lmax'] = lmax
        save_dict['ells'] = ells
        save_dict['file_seed'] = file_seed
        save_dict['global_seed'] = global_seed
        
        if add_noise:
            save_dict['noise_level'] = noise_level
            save_dict['galaxy_densities'] = np.array(galaxy_densities)
        
        if apply_mask:
            save_dict['mask_area_sqdeg'] = mask_area_sqdeg
            save_dict['mask_center'] = np.array(mask_center)
            save_dict['apodization_type'] = apodization_type
            save_dict['apodization_scale_deg'] = apodization_scale_deg
            save_dict['f_sky'] = f_sky
            save_dict['use_namaster'] = use_namaster and HAS_NAMASTER
        
        # Save results
        np.savez_compressed(save_path, **save_dict)
        
        if verbose:
            n_spectra = len(cls_dict)
            print(f"Processed {os.path.basename(file_path)}: {n_spectra} spectra")
        
        return save_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    """Main function with CLI for cross power spectrum processing."""
    parser = argparse.ArgumentParser(
        description="Process HEALPix maps to compute cross power spectra with MASTER correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data source options
    parser.add_argument("--fiducial", action="store_true",
                       help="Process fiducial cosmology instead of grid.")
    parser.add_argument("--base-dir", type=str,
                       help="Override default base directory.")
    parser.add_argument("--baryonified", action="store_true",
                       help="Use baryonified maps instead of nobaryons.")
    parser.add_argument("--bin-range", type=int, nargs="+", default=[1, 2, 3, 4],
                       help="Bins to process (1-4).")
    parser.add_argument("--cross-only", action="store_true",
                       help="Only compute cross power spectra.")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=DEFAULT_SIGMA_E,
                       help="Shape noise sigma_e.")
    parser.add_argument("--no-noise", action="store_true",
                       help="Don't add shape noise.")
    parser.add_argument("--galaxy-densities", type=float, nargs="+",
                       help=f"Per-bin galaxy densities in arcmin^-2 (default: {DEFAULT_GALAXY_DENSITY} for all).")
    
    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                       help="Apply sky mask (enables MASTER correction).")
    parser.add_argument("--mask-area-sqdeg", type=float, default=DEFAULT_MASK_AREA_SQDEG,
                       help="Mask area in square degrees.")
    parser.add_argument("--mask-center", type=float, nargs=2, 
                       default=DEFAULT_MASK_CENTER,
                       metavar=("LON", "LAT"),
                       help="Mask center (lon, lat) in degrees.")
    parser.add_argument("--apodization-type", type=str, default='C2',
                       choices=['C1', 'C2'],
                       help="Apodization smoothness (C2 recommended for lensing).")
    parser.add_argument("--apodization-scale-deg", type=float, default=2.0,
                       help="Apodization width in degrees (0 = no apodization).")
    parser.add_argument("--no-namaster", action="store_true",
                       help="Disable NaMaster (use naive pseudo-Cls).")
    
    # Algorithm options
    parser.add_argument("--lmax", type=int, default=DEFAULT_LMAX,
                       help="Maximum multipole.")
    parser.add_argument("--bin-edges", type=str,
                       help="Comma-separated multipole bin edges (e.g., '30,100,200,500,1024').")
    
    # Reproducibility
    parser.add_argument("--global-seed", type=int, default=42,
                       help="Global random seed for reproducibility.")
    
    # Execution
    parser.add_argument("--num-workers", type=int, default=70,
                       help="Number of parallel workers.")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress.")
    parser.add_argument("--force-overwrite", action="store_true",
                       help="Overwrite existing output files.")
    
    args = parser.parse_args()
    
    # Validate bins
    for bin_num in args.bin_range:
        if bin_num < 1 or bin_num > 4:
            print(f"Error: Bin {bin_num} is invalid. Must be 1-4.")
            sys.exit(1)
    
    if len(args.bin_range) < 2 and args.cross_only:
        print("Error: Need at least 2 bins for cross-correlation analysis.")
        sys.exit(1)
    
    # Validate NaMaster
    if args.apply_mask and not args.no_namaster and not HAS_NAMASTER:
        print("\n" + "="*60)
        print("ERROR: NaMaster not found but required for masked runs!")
        print("="*60)
        print("\nInstall with:")
        print("  pip install pymaster")
        print("\nOr use --no-namaster to proceed with uncorrected pseudo-Cls")
        print("(not recommended - will have mode-coupling artifacts)\n")
        sys.exit(1)
    
    # Parse bin edges
    bin_edges = None
    if args.bin_edges:
        bin_edges = np.array([int(x) for x in args.bin_edges.split(',')])
        print(f"Using custom bin edges: {bin_edges}")
    
    # Set base directory
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    
    # Set filename
    filename = "projected_probes_maps_baryonified512.h5" if args.baryonified else "projected_probes_maps_nobaryons512.h5"
    
    # Build file list
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
    
    # Print configuration
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    use_namaster = args.apply_mask and not args.no_namaster and HAS_NAMASTER
    
    print(f"\n{'='*60}")
    print(f"Cross Power Spectrum Processing (Refactored Version)")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_type} {map_type}")
    print(f"Files: {len(file_paths)}")
    print(f"Bins: {args.bin_range}")
    print(f"lmax: {args.lmax}")
    print(f"Global seed: {args.global_seed}")
    
    if args.apply_mask:
        print(f"\nMask configuration:")
        print(f"  Area: {args.mask_area_sqdeg:.0f} sq deg")
        print(f"  Center: {args.mask_center}")
        print(f"  Apodization: {args.apodization_type}, scale={args.apodization_scale_deg}°")
        print(f"  Mode-coupling correction: {'MASTER (NaMaster)' if use_namaster else 'None (pseudo-Cls)'}")
        if not use_namaster:
            print("  ⚠️  WARNING: No mode-coupling correction - results may be biased!")
    else:
        print(f"\nFull-sky mode (no mask)")
    
    print(f"{'='*60}\n")
    
    # Process files
    with mp.Pool(processes=args.num_workers,
                initializer=seed_worker,
                initargs=(args.global_seed,)) as pool:
        process_func = partial(
            process_file,
            bin_range=args.bin_range,
            noise_level=args.noise_level,
            galaxy_densities=args.galaxy_densities,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            cross_only=args.cross_only,
            verbose=args.verbose,
            apply_mask=args.apply_mask,
            mask_area_sqdeg=args.mask_area_sqdeg,
            mask_center=tuple(args.mask_center),
            apodization_type=args.apodization_type,
            apodization_scale_deg=args.apodization_scale_deg,
            use_namaster=use_namaster,
            bin_edges=bin_edges,
            global_seed=args.global_seed,
            force_overwrite=args.force_overwrite,
        )
        
        results = list(tqdm(
            pool.imap(process_func, file_paths),
            total=len(file_paths),
            desc="Processing"
        ))
    
    successful = [r for r in results if r is not None and os.path.exists(r)]
    print(f"\n✓ Processed {len(successful)}/{len(file_paths)} files")
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
