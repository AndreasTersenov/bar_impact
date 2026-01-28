"""
MASTER mode-coupling correction for power spectrum estimation from masked data.

This module provides utilities for computing unbiased power spectra from masked
HEALPix maps using the MASTER (Monte carlo Apodised Spherical Transform Estimator)
algorithm via the NaMaster library.

Key features:
- Mode-coupling matrix (MCM) computation and caching
- Proper deconvolution of pseudo-Cls
- Nyquist limit validation
- Support for binned and unbinned spectra

References:
- Hivon et al. 2002 (MASTER method): https://arxiv.org/abs/astro-ph/0105302
- NaMaster documentation: https://namaster.readthedocs.io/

Installation:
    pip install pymaster
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Dict, Optional, Tuple

import healpy as hp
import numpy as np

try:
    import pymaster as nmt

    HAS_NAMASTER = True
except ImportError:
    HAS_NAMASTER = False


__all__ = [
    "HAS_NAMASTER",
    "compute_coupling_matrix",
    "compute_power_spectra_master",
    "compute_pseudo_cls_simple",
    "MCM_CACHE",
]


# Global cache for mode-coupling matrices
MCM_CACHE: Dict = {}


def compute_coupling_matrix(
    mask: np.ndarray,
    lmax: int,
    bin_edges: Optional[np.ndarray] = None,
    use_cache: bool = True,
) -> Tuple:
    """
    Compute or retrieve cached mode-coupling matrix for a given mask.

    Parameters
    ----------
    mask : np.ndarray
        HEALPix mask array (can be binary or apodized)
    lmax : int
        Maximum multipole
    bin_edges : np.ndarray, optional
        Bandpower bin edges for binned coupling matrix.
        If None, computes full unbinned MCM.
    use_cache : bool, optional
        Whether to use cached MCM (default: True).

    Returns
    -------
    workspace : nmt.NmtWorkspace
        NaMaster workspace containing the coupling matrix
    binning : nmt.NmtBin
        Binning scheme used
    ells : np.ndarray
        Effective multipoles for output

    Raises
    ------
    ImportError
        If NaMaster is not installed.
    ValueError
        If lmax is invalid.

    Notes
    -----
    The mode-coupling matrix encodes how the mask affects power spectrum
    estimation. For a mask M(n), the observed pseudo-Cl is related to the
    true Cl by: Cl_pseudo = sum_l' M_ll' Cl_true, where M_ll' is the MCM.

    NaMaster computes M_ll' and performs the deconvolution to recover Cl_true.
    """
    if not HAS_NAMASTER:
        raise ImportError(
            "NaMaster is required for MASTER correction. "
            "Install with: pip install pymaster"
        )

    # Validate lmax
    nside = hp.npix2nside(len(mask))
    lmax_nyquist = 3 * nside - 1

    if lmax > lmax_nyquist:
        print(
            f"Warning: Requested lmax={lmax} exceeds Nyquist limit "
            f"for nside={nside} ({lmax_nyquist})"
        )
        print(f"         Using lmax={lmax_nyquist} instead")
        lmax = lmax_nyquist

    # Generate cache key
    if use_cache:
        mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()[:16]
        bin_hash = (
            hashlib.sha256(bin_edges.tobytes()).hexdigest()[:8]
            if bin_edges is not None
            else "unbinned"
        )
        cache_key = (mask_hash, int(lmax), bin_hash)

        if cache_key in MCM_CACHE:
            return MCM_CACHE[cache_key]

    # Create NaMaster field (spin-0 for convergence)
    f = nmt.NmtField(mask, [mask], purify_b=False, lmax=lmax)

    # Define binning scheme
    if bin_edges is not None:
        b = nmt.NmtBin.from_edges(bin_edges[:-1], bin_edges[1:])
    else:
        # Use adaptive binning based on lmax
        if lmax > 1500:
            nlb = 4  # Bin width of 4 for high lmax
        elif lmax > 1024:
            nlb = 2  # Bin width of 2 for medium lmax
        else:
            nlb = 1  # Effectively unbinned for low lmax
        b = nmt.NmtBin.from_lmax_linear(lmax, nlb=nlb)

    # Compute workspace (contains coupling matrix)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)

    # Get effective multipoles
    ells = b.get_effective_ells()

    if use_cache:
        MCM_CACHE[cache_key] = (w, b, ells)

    return w, b, ells


def compute_power_spectra_master(
    maps_dict: Dict[int, np.ndarray],
    mask: np.ndarray,
    lmax: int = 1024,
    bin_edges: Optional[np.ndarray] = None,
    include_auto: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray]:
    """
    Compute auto and cross power spectra with MASTER mode-coupling correction.

    Parameters
    ----------
    maps_dict : Dict[int, np.ndarray]
        Dictionary with bin numbers as keys and HEALPix maps as values.
    mask : np.ndarray
        HEALPix mask array (should be apodized for best results).
    lmax : int, optional
        Maximum multipole (default: 1024).
    bin_edges : np.ndarray, optional
        Bandpower bin edges. If None, returns all multipoles up to lmax.
    include_auto : bool, optional
        Whether to compute auto power spectra (default: True).
    verbose : bool, optional
        Print diagnostic information (default: False).

    Returns
    -------
    cls_dict : Dict[Tuple[int, int], np.ndarray]
        Dictionary with (bin_i, bin_j) tuples as keys and deconvolved Cls as values.
        Auto-spectra have bin_i == bin_j, cross-spectra have bin_i < bin_j.
    ells : np.ndarray
        Effective multipoles.

    Raises
    ------
    ImportError
        If NaMaster is not installed.

    Notes
    -----
    This function:
    1. Creates NaMaster fields for each map (with mask applied internally)
    2. Computes pseudo-Cls (biased by mode-coupling)
    3. Deconvolves using the mode-coupling matrix to recover unbiased Cls

    **IMPORTANT:** Do NOT pre-multiply maps by the mask! NaMaster applies
    the mask internally. Pre-multiplying would result in double-masking
    (mask² applied), artificially suppressing power.
    """
    if not HAS_NAMASTER:
        raise ImportError(
            "NaMaster is required for MASTER correction. "
            "Install with: pip install pymaster"
        )

    # Get coupling matrix
    workspace, binning, ells = compute_coupling_matrix(
        mask, lmax, bin_edges, use_cache=True
    )

    # Validate lmax
    nside = hp.npix2nside(len(mask))
    lmax_nyquist = 3 * nside - 1
    lmax_effective = min(lmax, lmax_nyquist)

    # Create NaMaster fields for each map
    # NOTE: Pass unmasked maps - NaMaster handles masking internally!
    fields_dict = {}
    for bin_num, map_data in maps_dict.items():
        f = nmt.NmtField(mask, [map_data], purify_b=False, lmax=lmax_effective)
        fields_dict[bin_num] = f

    if verbose:
        print(
            f"Computing power spectra for {len(maps_dict)} bins with MASTER correction..."
        )
        print(f"  lmax: {lmax_effective}")
        print(f"  n_ells: {len(ells)}")

    # Compute all auto and cross power spectra
    cls_dict = {}
    bin_numbers = sorted(maps_dict.keys())

    # Auto power spectra
    if include_auto:
        for bin_num in bin_numbers:
            f = fields_dict[bin_num]
            # Compute pseudo-Cl (coupled)
            cl_coupled = nmt.compute_coupled_cell(f, f)
            # Decouple using workspace
            cl_decoupled = workspace.decouple_cell(cl_coupled)
            # NaMaster returns array of shape (1, n_ells) for spin-0 x spin-0
            cls_dict[(bin_num, bin_num)] = cl_decoupled[0]

    # Cross power spectra
    for bin_i, bin_j in combinations(bin_numbers, 2):
        f_i = fields_dict[bin_i]
        f_j = fields_dict[bin_j]
        # Compute pseudo-Cl (coupled)
        cl_coupled = nmt.compute_coupled_cell(f_i, f_j)
        # Decouple
        cl_decoupled = workspace.decouple_cell(cl_coupled)
        cls_dict[(bin_i, bin_j)] = cl_decoupled[0]

    if verbose:
        print(f"  Computed {len(cls_dict)} power spectra")

    return cls_dict, ells


def compute_pseudo_cls_simple(
    maps_dict: Dict[int, np.ndarray],
    lmax: int = 1024,
    include_auto: bool = True,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Fallback: compute pseudo-Cls without mode-coupling correction.

    This function is used when NaMaster is not available or when running
    on full-sky data (no mask). The resulting power spectra are biased
    if the data is masked.

    Parameters
    ----------
    maps_dict : Dict[int, np.ndarray]
        Dictionary with bin numbers as keys and HEALPix maps as values.
    lmax : int, optional
        Maximum multipole (default: 1024).
    include_auto : bool, optional
        Whether to compute auto power spectra (default: True).

    Returns
    -------
    cls_dict : Dict[Tuple[int, int], np.ndarray]
        Dictionary with (bin_i, bin_j) tuples as keys and pseudo-Cls as values.
    """
    # Compute all alms first (more efficient)
    alms_dict = {}
    for bin_num, map_data in maps_dict.items():
        alms_dict[bin_num] = hp.map2alm(map_data, lmax=lmax)

    cls_dict = {}
    bin_numbers = sorted(maps_dict.keys())

    # Auto power spectra
    if include_auto:
        for bin_num in bin_numbers:
            cls_dict[(bin_num, bin_num)] = hp.alm2cl(alms_dict[bin_num])

    # Cross power spectra
    for bin_i, bin_j in combinations(bin_numbers, 2):
        cls_dict[(bin_i, bin_j)] = hp.alm2cl(alms_dict[bin_i], alms_dict[bin_j])

    return cls_dict


def clear_mcm_cache():
    """Clear the mode-coupling matrix cache."""
    global MCM_CACHE
    MCM_CACHE.clear()


def get_mcm_cache_info() -> Dict:
    """
    Get information about the current MCM cache state.

    Returns
    -------
    dict
        Dictionary with cache statistics.
    """
    return {
        "n_cached": len(MCM_CACHE),
        "cache_keys": list(MCM_CACHE.keys()),
    }
