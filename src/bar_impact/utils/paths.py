"""
Path and file discovery utilities for processing scripts.

This module provides utilities for finding data files and constructing
output paths with consistent naming conventions.
"""

import os
from typing import List, Tuple, Optional


__all__ = [
    "get_data_file_paths",
    "build_output_suffix",
]


def get_data_file_paths(
    base_dir: Optional[str] = None,
    fiducial: bool = False,
    baryonified: bool = False,
) -> Tuple[str, List[str]]:
    """
    Get list of data file paths for processing.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory override. If None, uses default paths.
    fiducial : bool, optional
        Process fiducial cosmology (200 permutations) vs grid (N cosmo × 7 perms)
    baryonified : bool, optional
        Use baryonified maps vs nobaryons
        
    Returns
    -------
    base_dir : str
        Resolved base directory path
    file_paths : List[str]
        List of full paths to data files that exist
        
    Examples
    --------
    >>> base_dir, files = get_data_file_paths(fiducial=True, baryonified=False)
    >>> len(files)  # Should be ~200 for fiducial
    200
    >>> base_dir, files = get_data_file_paths(fiducial=False, baryonified=False)
    >>> len(files)  # Should be N_cosmo * 7 for grid
    1400
    """
    # Determine base directory
    if base_dir is None:
        if fiducial:
            base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
        else:
            base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    
    # Determine filename
    filename = "projected_probes_maps_baryonified512.h5" if baryonified else "projected_probes_maps_nobaryons512.h5"
    
    # Build file list
    if fiducial:
        # Fiducial: perm_0000 to perm_0199 directly under base_dir
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
    else:
        # Grid: cosmo_XXXX/perm_0000 to perm_0006
        cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
        perm_dirs = [f"perm_{i:04d}" for i in range(7)]
        file_paths = [
            os.path.join(base_dir, cosmo, perm, filename)
            for cosmo in cosmo_dirs
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
        ]
    
    return base_dir, file_paths


def build_output_suffix(
    statistic_type: str,
    bin_number: Optional[int] = None,
    bin_range: Optional[List[int]] = None,
    bnt_bin: Optional[int] = None,
    bnt_bin_range: Optional[List[int]] = None,
    apply_mask: bool = False,
    mask_area_sqdeg: float = 14000.0,
    apodization_scale_deg: Optional[float] = None,
    apodization_type: Optional[str] = None,
    use_namaster: bool = False,
    add_noise: bool = False,
    noise_level: float = 0.26,
    lmax: Optional[int] = None,
    cross_only: bool = False,
    new_normalization: bool = True,
) -> str:
    """
    Build consistent output file suffix.
    
    Parameters
    ----------
    statistic_type : str
        Type of statistic: 'l1_norms', 'peak_counts', 'cls', etc.
    bin_number : int, optional
        Single bin number (for single-bin statistics)
    bin_range : List[int], optional
        Multiple bins (for multi-bin statistics)
    bnt_bin : int, optional
        Single BNT bin (0-indexed)
    bnt_bin_range : List[int], optional
        Multiple BNT bins (0-indexed)
    apply_mask : bool, optional
        Whether mask is applied
    mask_area_sqdeg : float, optional
        Mask area in square degrees
    apodization_scale_deg : float, optional
        Apodization width (for power spectrum)
    apodization_type : str, optional
        Apodization type: 'C1' or 'C2' (for power spectrum)
    use_namaster : bool, optional
        Whether MASTER correction is used (for power spectrum)
    add_noise : bool, optional
        Whether noise is added
    noise_level : float, optional
        Shape noise level
    lmax : int, optional
        Maximum multipole (for power spectrum)
    cross_only : bool, optional
        Cross-only flag (for power spectrum)
    new_normalization : bool, optional
        Use new normalization tag
        
    Returns
    -------
    str
        Output filename suffix (including extension)
        
    Examples
    --------
    >>> suffix = build_output_suffix('l1_norms', bin_number=1, add_noise=True, noise_level=0.26)
    >>> suffix
    '_l1_norms_bin1_noisy_s0.26_new_normalization.npy'
    
    >>> suffix = build_output_suffix('peak_counts', bnt_bin=2, apply_mask=True, mask_area_sqdeg=14000)
    >>> suffix
    '_peak_counts_bnt3_masked_14000sqdeg.npy'
    """
    parts = [f"_{statistic_type}"]
    
    # Handle bin specification
    if bnt_bin is not None:
        # BNT bins are 0-indexed, display as 1-indexed
        parts.append(f"_bnt{bnt_bin + 1}")
    elif bnt_bin_range is not None:
        # Multiple BNT bins
        bin_str = "".join([str(b + 1) for b in bnt_bin_range])
        if cross_only:
            parts.insert(0, "_bnt_cross")
            parts[1] = f"_cls"
        else:
            parts.insert(0, "_bnt_all")
            parts[1] = f"_cls"
        parts.append(f"_bins{bin_str}")
    elif bin_number is not None:
        # Single regular bin
        parts.append(f"_bin{bin_number}")
    elif bin_range is not None:
        # Multiple regular bins
        bin_str = "".join(map(str, bin_range))
        if cross_only:
            parts[0] = "_cross_cls"
        elif statistic_type == "cls":
            parts[0] = "_all_cls"
        parts.append(f"_bins{bin_str}")
    
    # Mask information
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        parts.append(f"_masked_{area_tag}sqdeg")
        
        # Apodization info (for power spectra)
        if apodization_scale_deg is not None:
            if apodization_scale_deg > 0:
                apod_tag = f"apod{apodization_scale_deg:.1f}"
                parts.append(f"_{apod_tag}")
            else:
                parts.append("_noapod")
        
        # MASTER correction (for power spectra)
        if statistic_type == "cls" or "cls" in statistic_type:
            method_tag = "master" if use_namaster else "pseudo"
            parts.append(f"_{method_tag}")
    elif statistic_type == "cls" or "cls" in statistic_type:
        # Power spectra without mask still get _master tag
        parts.append("_master")
    
    # Noise information
    if add_noise:
        parts.append(f"_noisy_s{noise_level:.2f}")
    
    # Lmax (for power spectra)
    if lmax is not None and lmax != 1024:
        parts.append(f"_lmax{lmax}")
    
    # Normalization flag
    if new_normalization and statistic_type in ['l1_norms', 'peak_counts']:
        parts.append("_new_normalization")
    
    # Extension
    if statistic_type == "cls" or "cls" in statistic_type:
        parts.append(".npz")
    else:
        parts.append(".npy")
    
    return "".join(parts)
