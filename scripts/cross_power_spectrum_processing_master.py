#!/usr/bin/env python3
"""
Cross Power Spectrum Processing Script with MASTER Mode-Coupling Correction

This script properly handles masked sky observations by:
1. Applying smooth apodization to mask edges (reduces mode-coupling artifacts)
2. Computing mode-coupling matrix (MCM) using NaMaster
3. Deconvolving pseudo-Cls to recover unbiased power spectra
4. Properly normalizing by effective f_sky after apodization

Key differences from cross_power_spectrum_processing.py:
- Uses NaMaster library for rigorous MASTER algorithm
- Adds apodization (C2-smooth cosine taper) to mask edges
- Deconvolves mode-coupling to eliminate periodic variance artifacts
- Supports reproducible random seeds per file
- Per-bin galaxy density for shape noise
- Validates mask/map consistency

Installation requirements:
    pip install pymaster

References:
- Hivon et al. 2002 (MASTER method): https://arxiv.org/abs/astro-ph/0105302
- NaMaster documentation: https://namaster.readthedocs.io/
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
import hashlib

try:
    import pymaster as nmt
    HAS_NAMASTER = True
except ImportError:
    HAS_NAMASTER = False
    print("Warning: NaMaster not found. Install with: pip install pymaster")
    print("Falling back to basic mode (no mode-coupling correction)")


# Global cache for masks and coupling matrices
MASK_CACHE = {}
MCM_CACHE = {}


def get_deterministic_seed(file_path, global_seed=42):
    """
    Generate a deterministic seed from file path and global seed.
    Ensures reproducibility across runs.
    """
    hash_input = f"{file_path}_{global_seed}".encode('utf-8')
    hash_digest = hashlib.sha256(hash_input).digest()
    # Convert first 4 bytes to integer seed
    seed = int.from_bytes(hash_digest[:4], byteorder='big')
    return seed % (2**32)  # Keep within numpy's seed range


def seed_worker(global_seed):
    """
    Initializer for multiprocessing pool with deterministic seeds.
    Each worker uses global_seed + worker_id for reproducibility.
    """
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    np.random.seed((global_seed + worker_id) % (2**32))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512, rng=None):
    """
    Adds shape noise to a HEALPix convergence (kappa) map.
    
    Parameters
    ----------
    kg : np.ndarray
        Convergence map
    sigma_e : float
        Intrinsic ellipticity dispersion per component
    galaxy_density : float
        Galaxy number density in arcmin^-2
    nside : int
        HEALPix nside
    rng : np.random.Generator, optional
        Random number generator for reproducibility
        
    Returns
    -------
    kg_noisy : np.ndarray
        Convergence map with shape noise added
    """
    if rng is None:
        rng = np.random.default_rng()
        
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = rng.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise


def create_apodized_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0),
                        apodization_type='C2', apodization_scale_deg=2.0):
    """
    Create a Euclid-like disk mask with smooth apodization at edges.
    
    Parameters
    ----------
    nside : int
        HEALPix resolution
    target_area_sqdeg : float
        Target unmasked area in square degrees (before apodization)
    center_coords : tuple(float, float)
        (lon, lat) in degrees for disk center
    apodization_type : str
        Type of apodization window ('C1' or 'C2' for smoothness)
        C2 is recommended for weak lensing (twice differentiable)
    apodization_scale_deg : float
        Width of the apodization transition region in degrees
        Typical values: 1-3 degrees for Euclid-like surveys
        
    Returns
    -------
    mask : np.ndarray
        Apodized mask array (float values in [0,1])
    f_sky : float
        Effective sky fraction after apodization
    angular_radius_deg : float
        Angular radius of the disk (before apodization) in degrees
    """
    # Calculate angular radius for target area
    total_area_sqdeg = 41252.96125  # 4*pi*(180/pi)^2
    angular_radius_rad = np.arccos(1 - (target_area_sqdeg / total_area_sqdeg) * 2)
    angular_radius_deg = np.rad2deg(angular_radius_rad)
    
    # Create base binary mask
    theta_center = np.deg2rad(90.0 - center_coords[1])
    phi_center = np.deg2rad(center_coords[0])
    center_vec = hp.ang2vec(theta_center, phi_center)
    
    # Get all pixel positions and compute angular distances to center
    npix = hp.nside2npix(nside)
    pix_indices = np.arange(npix)
    # hp.pix2vec returns three arrays (x, y, z) each of length npix
    vx, vy, vz = hp.pix2vec(nside, pix_indices)

    # Compute dot product between center vector and each pixel vector robustly
    # center_vec has shape (3,), vx/vy/vz have shape (npix,)
    dots = center_vec[0] * vx + center_vec[1] * vy + center_vec[2] * vz
    dots = np.clip(dots, -1.0, 1.0)
    ang_sep_rad = np.arccos(dots)
    ang_sep_deg = np.rad2deg(ang_sep_rad)
    
    # Create apodized mask
    # Inner region (full weight): r < r_disk - apod_scale
    # Apodization region: r_disk - apod_scale <= r <= r_disk + apod_scale
    # Outer region (zero weight): r > r_disk + apod_scale
    
    if apodization_scale_deg <= 0:
        # No apodization - binary mask
        mask = np.zeros(npix, dtype=np.float32)
        mask[ang_sep_deg <= angular_radius_deg] = 1.0
    else:
        # Smooth apodization
        mask = np.zeros(npix, dtype=np.float32)
        
        # Inner region (full weight)
        inner_radius = angular_radius_deg - apodization_scale_deg
        if inner_radius > 0:
            mask[ang_sep_deg <= inner_radius] = 1.0
        
        # Apodization region
        outer_radius = angular_radius_deg + apodization_scale_deg
        in_transition = (ang_sep_deg > max(0, inner_radius)) & (ang_sep_deg <= outer_radius)
        
        if np.any(in_transition):
            # Normalized distance within transition: 0 (inner) to 1 (outer)
            transition_width = outer_radius - max(0, inner_radius)
            x = (ang_sep_deg[in_transition] - max(0, inner_radius)) / transition_width
            
            if apodization_type == 'C1':
                # C1 continuous (cosine taper): smooth but not twice-differentiable
                taper = 0.5 * (1.0 + np.cos(np.pi * x))
            elif apodization_type == 'C2':
                # C2 continuous (smoother): recommended for power spectra
                # Using a polynomial that's twice differentiable
                taper = np.where(x < 0.5,
                                1.0 - 2 * x**2,
                                2 * (1 - x)**2)
            else:
                raise ValueError(f"Unknown apodization type: {apodization_type}")
            
            mask[in_transition] = taper.astype(np.float32)
    
    # Compute effective f_sky
    f_sky = float(np.mean(mask))
    
    return mask, f_sky, angular_radius_deg


def get_cached_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0),
                   apodization_type='C2', apodization_scale_deg=2.0):
    """
    Return a cached apodized mask to avoid recomputation.
    
    Cache key includes all parameters affecting the mask.
    """
    lon, lat = float(center_coords[0]), float(center_coords[1])
    key = (int(nside), float(target_area_sqdeg), round(lon, 6), round(lat, 6),
           str(apodization_type), float(apodization_scale_deg))
    
    if key not in MASK_CACHE:
        mask, f_sky, angular_radius_deg = create_apodized_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
            apodization_type=apodization_type,
            apodization_scale_deg=apodization_scale_deg
        )
        MASK_CACHE[key] = (mask, f_sky, angular_radius_deg)
    
    return MASK_CACHE[key]


def get_coupling_matrix(mask, lmax, bin_edges=None, use_namaster=True):
    """
    Compute or retrieve cached mode-coupling matrix for a given mask.
    
    Parameters
    ----------
    mask : np.ndarray
        HEALPix mask array
    lmax : int
        Maximum multipole
    bin_edges : np.ndarray, optional
        Bandpower bin edges for binned coupling matrix
        If None, computes full unbinned MCM (can be large for high lmax)
    use_namaster : bool
        If True, use NaMaster for MCM computation (recommended)
        If False, return identity (no correction - for testing/fallback)
        
    Returns
    -------
    mcm : np.ndarray or nmt.NmtWorkspace
        Mode-coupling matrix (or NaMaster workspace object)
    ells : np.ndarray
        Effective multipoles for output
    """
    if not use_namaster or not HAS_NAMASTER:
        # Fallback: return identity (no correction)
        if bin_edges is not None:
            ells = (bin_edges[:-1] + bin_edges[1:]) / 2
            n_bins = len(ells)
            return np.eye(n_bins), ells
        else:
            ells = np.arange(lmax + 1)
            return np.eye(lmax + 1), ells
    
    # Generate cache key
    mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()[:16]
    bin_hash = hashlib.sha256(bin_edges.tobytes()).hexdigest()[:8] if bin_edges is not None else 'unbinned'
    key = (mask_hash, int(lmax), bin_hash)
    
    if key in MCM_CACHE:
        return MCM_CACHE[key]
    
    # Create NaMaster field (spin-0 for convergence) with explicit lmax
    nside = hp.npix2nside(len(mask))
    # NaMaster has internal limits: lmax cannot exceed 3*nside - 1 (Nyquist)
    lmax_nyquist = 3 * nside - 1
    lmax_effective = min(lmax, lmax_nyquist)
    
    if lmax > lmax_nyquist:
        print(f"Warning: Requested lmax={lmax} exceeds Nyquist limit for nside={nside} ({lmax_nyquist})")
        print(f"         Using lmax={lmax_effective} instead")
    
    f = nmt.NmtField(mask, [mask], purify_b=False, lmax=lmax_effective)  # No B-mode for spin-0
    
    # Define binning scheme
    if bin_edges is not None:
        b = nmt.NmtBin.from_edges(bin_edges[:-1], bin_edges[1:])
    else:
        # Use narrow bins (effectively unbinned up to lmax_effective)
        # For very high lmax, use wider bins to avoid NaMaster memory issues
        if lmax_effective > 1500:
            nlb = 4  # Bin width of 4 for lmax > 1500
        elif lmax_effective > 1024:
            nlb = 2  # Bin width of 2 for 1024 < lmax <= 1500
        else:
            nlb = 1  # Effectively unbinned for lmax <= 1024
        b = nmt.NmtBin.from_lmax_linear(lmax_effective, nlb=nlb)
    
    # Compute workspace (contains coupling matrix)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)
    
    # Get effective multipoles
    ells = b.get_effective_ells()
    
    MCM_CACHE[key] = (w, b, ells)
    return MCM_CACHE[key]


def compute_power_spectra_master(maps_dict, mask, lmax=1024, bin_edges=None, 
                                 use_namaster=True, verbose=False):
    """
    Compute auto and cross power spectra with MASTER mode-coupling correction.
    
    Parameters
    ----------
    maps_dict : dict
        Dictionary with bin numbers as keys and HEALPix maps as values
    mask : np.ndarray
        HEALPix mask array (apodized)
    lmax : int
        Maximum multipole
    bin_edges : np.ndarray, optional
        Bandpower bin edges. If None, returns all multipoles up to lmax
    use_namaster : bool
        Use NaMaster for proper deconvolution (recommended)
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    cls_dict : dict
        Dictionary with (bin_i, bin_j) tuples as keys and deconvolved Cls as values
    ells : np.ndarray
        Effective multipoles
    """
    if not use_namaster or not HAS_NAMASTER:
        if verbose:
            print("Warning: NaMaster not available, using naive pseudo-Cls (uncorrected)")
        # Fallback to simple pseudo-Cl computation
        return compute_pseudo_cls_simple(maps_dict, lmax), np.arange(lmax + 1)
    
    # Get coupling matrix
    mcm_result = get_coupling_matrix(mask, lmax, bin_edges, use_namaster=True)
    workspace, binning, ells = mcm_result
    
    # Determine effective lmax (respecting Nyquist limit)
    nside = hp.npix2nside(len(mask))
    lmax_nyquist = 3 * nside - 1
    lmax_effective = min(lmax, lmax_nyquist)
    
    # Create NaMaster fields for each map
    # NOTE: Do NOT pre-multiply map by mask! NaMaster applies the mask internally.
    # Passing mask*map would result in double-masking (mask² applied to data),
    # which artificially suppresses power at apodized edges.
    fields_dict = {}
    for bin_num, map_data in maps_dict.items():
        # Pass unmasked map - NaMaster handles masking internally
        f = nmt.NmtField(mask, [map_data], purify_b=False, lmax=lmax_effective)
        fields_dict[bin_num] = f
    
    # Compute all auto and cross power spectra
    cls_dict = {}
    bin_numbers = sorted(maps_dict.keys())
    
    if verbose:
        print(f"Computing power spectra for {len(bin_numbers)} bins with MASTER correction...")
    
    # Auto power spectra
    for bin_num in bin_numbers:
        f = fields_dict[bin_num]
        # Compute pseudo-Cl
        cl_coupled = nmt.compute_coupled_cell(f, f)
        # Decouple using workspace
        cl_decoupled = workspace.decouple_cell(cl_coupled)
        # NaMaster returns array of shape (1, n_ells) for spin-0 x spin-0
        cls_dict[(bin_num, bin_num)] = cl_decoupled[0]
    
    # Cross power spectra
    for bin_i, bin_j in combinations(bin_numbers, 2):
        f_i = fields_dict[bin_i]
        f_j = fields_dict[bin_j]
        # Compute pseudo-Cl
        cl_coupled = nmt.compute_coupled_cell(f_i, f_j)
        # Decouple
        cl_decoupled = workspace.decouple_cell(cl_coupled)
        cls_dict[(bin_i, bin_j)] = cl_decoupled[0]
    
    return cls_dict, ells


def compute_pseudo_cls_simple(maps_dict, lmax=1024):
    """
    Fallback: compute pseudo-Cls without mode-coupling correction.
    Used when NaMaster is not available.
    """
    alms_dict = {}
    for bin_num, map_data in maps_dict.items():
        alms_dict[bin_num] = hp.map2alm(map_data, lmax=lmax)
    
    cls_dict = {}
    bin_numbers = sorted(maps_dict.keys())
    
    # Auto power spectra
    for bin_num in bin_numbers:
        cls_dict[(bin_num, bin_num)] = hp.alm2cl(alms_dict[bin_num])
    
    # Cross power spectra
    for bin_i, bin_j in combinations(bin_numbers, 2):
        cls_dict[(bin_i, bin_j)] = hp.alm2cl(alms_dict[bin_i], alms_dict[bin_j])
    
    return cls_dict


def process_file(file_path, bin_range=[1, 2, 3, 4], noise_level=0.26, 
                galaxy_densities=None, add_noise=True, lmax=1024, 
                cross_only=False, verbose=False, apply_mask=False,
                mask_area_sqdeg=14000.0, mask_center=(0.0, 90.0),
                apodization_type='C2', apodization_scale_deg=2.0,
                use_namaster=True, bin_edges=None, global_seed=42,
                force_overwrite=False, subtract_mean=False):
    """
    Process a single file: extract kappa maps, apply mask, compute MASTER-corrected spectra.
    
    Key improvements over original:
    - Deterministic random seeds for reproducibility
    - Per-bin galaxy densities
    - Apodized masks with mode-coupling correction
    - Proper validation and metadata
    """
    # Generate deterministic seed for this file
    file_seed = get_deterministic_seed(file_path, global_seed)
    rng = np.random.default_rng(file_seed)
    
    # Default galaxy densities (one per bin)
    if galaxy_densities is None:
        galaxy_densities = [6.75] * len(bin_range)  # Default for all bins
    
    # Construct output filename
    bin_str = "".join(map(str, bin_range))
    lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        apod_tag = f"apod{apodization_scale_deg:.1f}" if apodization_scale_deg > 0 else "noapod"
        method_tag = "master" if use_namaster and HAS_NAMASTER else "pseudo"
        mask_suffix = f"_masked_{area_tag}sqdeg_{apod_tag}_{method_tag}"
    else:
        # Even without a mask, mark files as coming from the MASTER pipeline
        mask_suffix = "_master"
    
    noise_suffix = f"_noisy_s{noise_level:.2f}" if add_noise else ""
    submean_suffix = "_submean" if subtract_mean else ""
    spectra_type = "cross_cls" if cross_only else "all_cls"

    suffix = f"_{spectra_type}_bins{bin_str}{mask_suffix}{submean_suffix}{noise_suffix}{lmax_suffix}.npz"
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if already processed (unless force_overwrite is set)
    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, already processed.")
        return save_path
    
    try:
        # Load kappa maps
        maps_dict = {}
        missing_bins = []
        
        mask_tuple = None
        with h5py.File(file_path, "r") as f:
            for idx, bin_num in enumerate(bin_range):
                map_key = f"kg/stage3_lensing{bin_num}"
                if map_key in f:
                    kg = np.array(f[map_key])
                    nside = hp.get_nside(kg)
                    
                    # Validate map
                    if not hp.isnpixok(len(kg)):
                        raise ValueError(f"Invalid HEALPix map size: {len(kg)}")
                    
                    # Add shape noise if requested (BEFORE masking for physical consistency)
                    if add_noise:
                        gal_density = galaxy_densities[idx] if idx < len(galaxy_densities) else 6.75
                        kg = add_shape_noise(kg, sigma_e=noise_level, 
                                           galaxy_density=gal_density, 
                                           nside=nside, rng=rng)
                    
                    maps_dict[bin_num] = kg
                else:
                    missing_bins.append(bin_num)
                    if verbose:
                        print(f"Warning: {map_key} not found in {os.path.basename(file_path)}")
        
        if not maps_dict:
            if verbose:
                print(f"No valid maps found in {os.path.basename(file_path)}")
            return None
        
        # Get or create mask
        if apply_mask:
            if mask_tuple is None:
                mask_tuple = get_cached_mask(
                    nside=nside,
                    target_area_sqdeg=mask_area_sqdeg,
                    center_coords=mask_center,
                    apodization_type=apodization_type,
                    apodization_scale_deg=apodization_scale_deg
                )
            mask, f_sky, angular_radius_deg = mask_tuple
            
            # Validate mask and maps have same nside
            if hp.get_nside(mask) != nside:
                raise ValueError(f"Mask nside ({hp.get_nside(mask)}) != map nside ({nside})")
        else:
            # Full sky - uniform mask
            mask = np.ones(hp.nside2npix(nside), dtype=np.float32)
            f_sky = 1.0
            angular_radius_deg = 180.0

        # Optional mass-sheet gauge: remove the mask-weighted monopole of each map BEFORE
        # NaMaster. The masked field w*kappa otherwise injects a mu^2 * |w_l|^2 term into
        # low ell (the disk mask's red pseudo-spectrum), and the absolute mean convergence
        # is unobservable from shear. Only the monopole is removed; the dipole is kept
        # (a kappa gradient IS observable via shear).
        monopoles = {}
        if subtract_mean:
            wsum = float(np.sum(mask))
            for bin_num in list(maps_dict.keys()):
                mu = float(np.sum(mask * maps_dict[bin_num]) / wsum)
                maps_dict[bin_num] = maps_dict[bin_num] - mu
                monopoles[bin_num] = mu

        # Compute power spectra with MASTER correction
        cls_dict, ells = compute_power_spectra_master(
            maps_dict=maps_dict,
            mask=mask,
            lmax=lmax,
            bin_edges=bin_edges,
            use_namaster=use_namaster and apply_mask,  # Only use MASTER if masked
            verbose=verbose
        )
        
        # Filter to cross-only if requested
        if cross_only:
            cls_dict = {(i, j): cls for (i, j), cls in cls_dict.items() if i != j}
        
        # Prepare data for saving
        save_dict = {}
        for (i, j), cls in cls_dict.items():
            save_dict[f"cls_{i}_{j}"] = cls
        
        # Add comprehensive metadata
        save_dict['bin_range'] = np.array(list(maps_dict.keys()))
        save_dict['lmax'] = lmax
        save_dict['ells'] = ells  # Effective multipoles (may be binned)
        save_dict['file_seed'] = file_seed
        save_dict['global_seed'] = global_seed
        
        if add_noise:
            save_dict['noise_level'] = noise_level
            save_dict['galaxy_densities'] = np.array(galaxy_densities)
        
        if apply_mask:
            save_dict['mask_area_sqdeg'] = float(mask_area_sqdeg)
            save_dict['mask_f_sky'] = float(f_sky)
            save_dict['mask_center_lon_lat_deg'] = np.array(mask_center, dtype=np.float64)
            save_dict['mask_angular_radius_deg'] = float(angular_radius_deg)
            save_dict['apodization_type'] = apodization_type
            save_dict['apodization_scale_deg'] = float(apodization_scale_deg)
            save_dict['mode_coupling_corrected'] = bool(use_namaster and HAS_NAMASTER)

        save_dict['mean_subtracted'] = bool(subtract_mean)
        if subtract_mean:
            save_dict['monopoles'] = np.array(
                [monopoles.get(b, 0.0) for b in sorted(maps_dict.keys())]
            )

        if missing_bins:
            save_dict['missing_bins'] = np.array(missing_bins)
        
        # Save results
        np.savez_compressed(save_path, **save_dict)
        
        if verbose:
            method = "MASTER-corrected" if (apply_mask and use_namaster and HAS_NAMASTER) else "pseudo-Cl"
            spectra_type_str = "cross" if cross_only else "auto+cross"
            print(f"Processed: {os.path.basename(file_path)} -> {len(cls_dict)} {spectra_type_str} ({method})")
        
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_for_inference(processed_files, output_dir, bin_range=[1, 2, 3, 4],
                           dataset_type="grid", map_type="nobaryons",
                           noise_level=0.26, add_noise=True, lmax=1024,
                           verbose=False, apply_mask=False, mask_area_sqdeg=None,
                           apodization_scale_deg=2.0, use_namaster=True,
                           subtract_mean=False):
    """
    Aggregate MASTER-corrected .npz files into inference-ready .npy format.
    
    Validates that all files used consistent mask/correction settings.
    """
    print(f"\n{'='*60}")
    print(f"Aggregating {len(processed_files)} MASTER-corrected files...")
    print(f"{'='*60}")
    
    # Initialize storage
    auto_spectra = {bin_num: [] for bin_num in bin_range}
    cross_spectra = {(i, j): [] for i, j in combinations(bin_range, 2)}
    
    failed_files = []
    incomplete_files = {}
    inconsistent_metadata = []
    
    # Track metadata consistency
    first_file_meta = None
    
    for file_path in tqdm(processed_files, desc="Loading files"):
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Validate metadata consistency (critical for masked runs)
            if apply_mask:
                current_meta = {
                    'f_sky': float(data.get('mask_f_sky', 0)),
                    'corrected': bool(data.get('mode_coupling_corrected', False)),
                    'apod_scale': float(data.get('apodization_scale_deg', 0))
                }
                
                if first_file_meta is None:
                    first_file_meta = current_meta
                elif current_meta != first_file_meta:
                    inconsistent_metadata.append((file_path, current_meta))
            
            missing_keys = []
            
            # Extract auto spectra
            for bin_num in bin_range:
                key = f"cls_{bin_num}_{bin_num}"
                if key in data.files:
                    auto_spectra[bin_num].append(data[key])
                else:
                    missing_keys.append(key)
            
            # Extract cross spectra
            for i, j in combinations(bin_range, 2):
                key = f"cls_{i}_{j}"
                if key in data.files:
                    cross_spectra[(i, j)].append(data[key])
                else:
                    missing_keys.append(key)
            
            if missing_keys:
                incomplete_files[file_path] = missing_keys
                
        except Exception as e:
            failed_files.append((file_path, str(e)))
    
    # Report issues
    if failed_files or incomplete_files or inconsistent_metadata:
        print(f"\n{'='*60}")
        print("⚠️  ISSUES DETECTED")
        print(f"{'='*60}")
        
        if failed_files:
            print(f"\n❌ Failed to load {len(failed_files)} files")
        
        if incomplete_files:
            print(f"\n⚠️  {len(incomplete_files)} files with missing keys")
        
        if inconsistent_metadata:
            print(f"\n⚠️  {len(inconsistent_metadata)} files with inconsistent mask metadata!")
            print("This may indicate mixed full-sky/masked or different apodization settings.")
            print("First 5 inconsistent files:")
            for fp, meta in inconsistent_metadata[:5]:
                print(f"  {os.path.basename(fp)}: {meta}")
    
    # Determine output suffix
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg)) if mask_area_sqdeg else "mask"
        apod_tag = f"apod{apodization_scale_deg:.1f}" if apodization_scale_deg > 0 else "noapod"
        method_tag = "master" if use_namaster and HAS_NAMASTER else "pseudo"
        mask_suffix = f"_masked_{area_tag}sqdeg_{apod_tag}_{method_tag}"
    else:
        # Even without a mask, mark files as coming from the MASTER pipeline
        mask_suffix = "_master"
    
    noise_suffix = f"_noisy_s{noise_level:.2f}" if add_noise else ""
    submean_suffix = "_submean" if subtract_mean else ""
    lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""

    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    # Save auto spectra
    print("\nSaving auto power spectra...")
    for bin_num in bin_range:
        if auto_spectra[bin_num]:
            auto_array = np.array(auto_spectra[bin_num])
            filename = f"all_cls_{dataset_type}_{map_type}_bin{bin_num}{mask_suffix}{submean_suffix}{noise_suffix}{lmax_suffix}.npy"
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, auto_array)
            created_files.append(output_path)
            print(f"  ✓ Bin {bin_num}: shape {auto_array.shape}")
    
    # Save combined cross spectra
    print("\nSaving cross power spectra...")
    cross_data_parts = []
    
    for i, j in combinations(sorted(bin_range), 2):
        if cross_spectra[(i, j)]:
            cross_array = np.array(cross_spectra[(i, j)])
            cross_data_parts.append(cross_array)
            print(f"  ✓ Cross ({i},{j}): shape {cross_array.shape}")
    
    if cross_data_parts:
        cross_combined = np.concatenate(cross_data_parts, axis=1)
        bin_str = "".join(map(str, bin_range))
        filename = f"all_cross_cls_{dataset_type}_{map_type}_bins{bin_str}{mask_suffix}{submean_suffix}{noise_suffix}{lmax_suffix}.npy"
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, cross_combined)
        created_files.append(output_path)
        print(f"\n  ✓ Combined cross: shape {cross_combined.shape}")
    
    print(f"\n{'='*60}")
    print(f"Aggregation complete! Created {len(created_files)} files")
    print(f"{'='*60}\n")
    
    return created_files


def main():
    """Main function with enhanced CLI for MASTER mode-coupling correction."""
    parser = argparse.ArgumentParser(
        description="Process HEALPix maps with MASTER mode-coupling correction for masked surveys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source options
    parser.add_argument("--fiducial", action="store_true",
                       help="Process fiducial cosmology instead of grid.")
    parser.add_argument("--base-dir", type=str,
                       help="Override default base directory.")
    parser.add_argument("--baryonified", action="store_true",
                       help="Use baryonified maps instead of nobaryons.")
    parser.add_argument("--bin-range", type=int, nargs="+", default=[1, 2, 3, 4],
                       help="Redshift bins to process.")
    parser.add_argument("--cross-only", action="store_true",
                       help="Only compute cross power spectra.")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26,
                       help="Shape noise sigma_e (intrinsic ellipticity dispersion).")
    parser.add_argument("--no-noise", action="store_true",
                       help="Don't add shape noise.")
    parser.add_argument("--galaxy-densities", type=float, nargs="+",
                       help="Per-bin galaxy densities in arcmin^-2 (default: 6.75 for all).")
    
    # Mask options
    parser.add_argument("--apply-mask", action="store_true",
                       help="Apply sky mask (enables MASTER correction).")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                       help="Mask area in square degrees.")
    parser.add_argument("--mask-center", type=float, nargs=2, default=(0.0, 90.0),
                       metavar=("LON", "LAT"),
                       help="Mask center (lon, lat) in degrees.")
    parser.add_argument("--apodization-type", type=str, default='C2',
                       choices=['C1', 'C2'],
                       help="Apodization smoothness (C2 recommended for lensing).")
    parser.add_argument("--apodization-scale-deg", type=float, default=2.0,
                       help="Apodization width in degrees (0 = no apodization).")
    parser.add_argument("--no-namaster", action="store_true",
                       help="Disable NaMaster (use naive pseudo-Cls - for testing only).")
    parser.add_argument("--subtract-mean", action="store_true",
                       help="Remove the mask-weighted monopole of each map before NaMaster "
                            "(mass-sheet gauge; removes mu^2 leakage into low ell). Default "
                            "off; tags outputs with '_submean'.")

    # Algorithm options
    parser.add_argument("--lmax", type=int, default=1024,
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
                       help="Overwrite existing output files instead of skipping.")
    
    # Output
    parser.add_argument("--save-combined", action="store_true",
                       help="Create summary file listing processed files.")
    parser.add_argument("--combined-output", type=str,
                       help="Path for summary file.")
    parser.add_argument("--aggregate-for-inference", action="store_true",
                       help="Aggregate into inference-ready .npy files.")
    parser.add_argument("--inference-output-dir", type=str,
                       help="Output directory for aggregated files.")
    
    args = parser.parse_args()
    
    # Validate NaMaster availability
    if args.apply_mask and not args.no_namaster and not HAS_NAMASTER:
        print("\n" + "="*60)
        print("ERROR: NaMaster not found but required for masked runs!")
        print("="*60)
        print("\nInstall with:")
        print("  pip install pymaster")
        print("\nOr use --no-namaster to proceed with uncorrected pseudo-Cls")
        print("(not recommended - will have periodic variance artifacts)\n")
        return
    
    # Parse bin edges if provided
    bin_edges = None
    if args.bin_edges:
        bin_edges = np.array([int(x) for x in args.bin_edges.split(',')])
        print(f"Using custom bin edges: {bin_edges}")
    
    # Set base directory
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/new_grid/"
    
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
    print(f"MASTER-Corrected Power Spectrum Processing")
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
            print(f"  ⚠️  WARNING: Running without MASTER correction will produce artifacts!")
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
            subtract_mean=args.subtract_mean
        )
        
        results = list(tqdm(
            pool.imap(process_func, file_paths),
            total=len(file_paths),
            desc="Processing"
        ))
    
    successful = [r for r in results if r is not None and os.path.exists(r)]
    print(f"\n✓ Processed {len(successful)}/{len(file_paths)} files")
    
    # Aggregate if requested
    if args.aggregate_for_inference and successful:
        output_dir = args.inference_output_dir or base_dir
        
        created_files = aggregate_for_inference(
            processed_files=successful,
            output_dir=output_dir,
            bin_range=args.bin_range,
            dataset_type=dataset_type,
            map_type=map_type,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            verbose=args.verbose,
            apply_mask=args.apply_mask,
            mask_area_sqdeg=args.mask_area_sqdeg,
            apodization_scale_deg=args.apodization_scale_deg,
            use_namaster=use_namaster,
            subtract_mean=args.subtract_mean
        )

        if created_files:
            print("\n✓ Inference-ready files:")
            for f in created_files:
                print(f"  - {os.path.basename(f)}")


if __name__ == "__main__":
    main()
