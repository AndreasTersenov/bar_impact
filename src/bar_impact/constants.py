"""
Physical constants and default parameters for BAR_IMPACT.

This module centralizes all physical constants, transformation matrices,
and default parameter values used throughout the package.
"""

import numpy as np
from typing import Dict, List, Tuple

__all__ = [
    # BNT Transform
    "BNT_MATRIX",
    "BNT_MATRIX_DEFAULT",
    "get_bnt_matrix",
    # HEALPix defaults
    "DEFAULT_NSIDE",
    "DEFAULT_LMAX",
    # Shape noise defaults
    "DEFAULT_SIGMA_E",
    "DEFAULT_GALAXY_DENSITY",
    # L1 norm defaults
    "DEFAULT_NUM_SCALES",
    "DEFAULT_MIN_SNR",
    "DEFAULT_MAX_SNR",
    "DEFAULT_NOISE_STD",
    "DEFAULT_NBINS",
    # Survey defaults
    "DEFAULT_MASK_AREA_SQDEG",
    "FULL_SKY_AREA_SQDEG",
    # Cosmological parameter names
    "COSMO_PARAM_NAMES",
    "COSMO_PARAM_LABELS",
    "COSMO_PARAM_NAMES_SUBSET",
    "COSMO_PARAM_LABELS_SUBSET",
    # Redshift bins
    "NUM_REDSHIFT_BINS",
    "REDSHIFT_BIN_NAMES",
]


# =============================================================================
# BNT (Bernardeau-Nishimichi-Taruya) Transform Matrix
# =============================================================================

# Default BNT transformation matrix for 4 redshift bins
# This is a lower-triangular matrix that nulls correlations between bins
# Derived for the specific redshift distribution of the survey
BNT_MATRIX_DEFAULT: np.ndarray = np.array([
    [1.0,        0.0,         0.0,        0.0],
    [-1.0,       1.0,         0.0,        0.0],
    [0.4521097, -1.4521097,   1.0,        0.0],
    [0.0,        0.25127807, -1.251278,   1.0],
], dtype=np.float64)

# Alias for convenience
BNT_MATRIX = BNT_MATRIX_DEFAULT


def get_bnt_matrix(n_bins: int = 4, custom_matrix: np.ndarray = None) -> np.ndarray:
    """
    Get the BNT transformation matrix.
    
    Parameters
    ----------
    n_bins : int, optional
        Number of redshift bins. Default is 4.
    custom_matrix : np.ndarray, optional
        Custom BNT matrix to use instead of the default.
        Must be shape (n_bins, n_bins).
        
    Returns
    -------
    np.ndarray
        BNT transformation matrix of shape (n_bins, n_bins).
        
    Raises
    ------
    ValueError
        If n_bins != 4 and no custom_matrix is provided, or if
        custom_matrix has wrong shape.
        
    Notes
    -----
    The BNT transform is designed to null cross-correlations between
    different redshift bins, making the resulting maps statistically
    independent. This is useful for tomographic weak lensing analysis.
    
    The default matrix is derived for a Stage-III-like survey with
    4 tomographic bins.
    
    Examples
    --------
    >>> bnt = get_bnt_matrix()
    >>> bnt.shape
    (4, 4)
    >>> bnt[0, 0]
    1.0
    
    >>> # Using a custom matrix
    >>> custom = np.eye(3)
    >>> bnt = get_bnt_matrix(n_bins=3, custom_matrix=custom)
    """
    if custom_matrix is not None:
        custom_matrix = np.asarray(custom_matrix, dtype=np.float64)
        if custom_matrix.shape != (n_bins, n_bins):
            raise ValueError(
                f"Custom matrix shape {custom_matrix.shape} does not match "
                f"n_bins={n_bins}. Expected shape ({n_bins}, {n_bins})."
            )
        return custom_matrix.copy()
    
    if n_bins != 4:
        raise ValueError(
            f"Default BNT matrix only available for n_bins=4, got {n_bins}. "
            "Please provide a custom_matrix for other bin configurations."
        )
    
    return BNT_MATRIX_DEFAULT.copy()


# =============================================================================
# HEALPix Parameters
# =============================================================================

# Default HEALPix resolution parameter
# NSIDE=512 gives ~12 million pixels, ~7 arcmin resolution
DEFAULT_NSIDE: int = 512

# Default maximum multipole for power spectrum calculations
DEFAULT_LMAX: int = 1024


# =============================================================================
# Shape Noise Parameters
# =============================================================================

# Default intrinsic ellipticity dispersion (per component)
DEFAULT_SIGMA_E: float = 0.26

# Default galaxy number density in arcmin^-2
# This is typical for Stage-III surveys
DEFAULT_GALAXY_DENSITY: float = 6.75


# =============================================================================
# L1 Norm / Wavelet Analysis Parameters
# =============================================================================

# Default number of wavelet scales
DEFAULT_NUM_SCALES: int = 5

# Default number of histogram bins for L1 norm calculation
DEFAULT_NBINS: int = 40

# Default SNR range for L1 norm histograms
DEFAULT_MIN_SNR: float = -13.0
DEFAULT_MAX_SNR: float = 13.0

# Default noise standard deviation for normalization
DEFAULT_NOISE_STD: float = 0.0146


# =============================================================================
# Survey Mask Parameters
# =============================================================================

# Default survey mask area in square degrees (Euclid-like)
DEFAULT_MASK_AREA_SQDEG: float = 14000.0

# Full sky area in square degrees
# 4 * pi * (180/pi)^2
FULL_SKY_AREA_SQDEG: float = 41252.96125

# Default mask center coordinates (longitude, latitude) in degrees
# North pole for Euclid-like survey
DEFAULT_MASK_CENTER: Tuple[float, float] = (0.0, 90.0)


# =============================================================================
# Cosmological Parameters
# =============================================================================

# Full set of cosmological parameter names (as used in CosmoGRID)
COSMO_PARAM_NAMES: Tuple[str, ...] = (
    "Omega_m",
    "S_8",
    "w_0",
    "H_0",
    "n_s",
    "Omega_b",
)

# LaTeX labels for plotting
COSMO_PARAM_LABELS: Tuple[str, ...] = (
    r"$\Omega_{\rm m}$",
    r"$S_8$",
    r"$w_0$",
    r"$H_0$",
    r"$n_s$",
    r"$\Omega_{\rm b}$",
)

# Subset of parameters commonly used for tension analysis
COSMO_PARAM_NAMES_SUBSET: Tuple[str, ...] = (
    "Omega_m",
    "S_8",
    "w_0",
)

COSMO_PARAM_LABELS_SUBSET: Tuple[str, ...] = (
    r"$\Omega_{\rm m}$",
    r"$S_8$",
    r"$w_0$",
)

# Indices of subset parameters in the full parameter list
COSMO_PARAM_SUBSET_INDICES: Tuple[int, ...] = (0, 1, 2)


# =============================================================================
# Redshift Bins
# =============================================================================

# Number of tomographic redshift bins
NUM_REDSHIFT_BINS: int = 4

# Names for redshift bins (1-indexed as commonly used in cosmology)
REDSHIFT_BIN_NAMES: Tuple[str, ...] = ("bin1", "bin2", "bin3", "bin4")

# HDF5 key template for loading maps from CosmoGRID files
COSMOGRID_MAP_KEY_TEMPLATE: str = "kg/stage3_lensing{bin_number}"


# =============================================================================
# File naming conventions
# =============================================================================

# Supported simulation types
SIMULATION_TYPES: Tuple[str, ...] = ("baryonified", "nobaryons")

# Default noise level suffix format
NOISE_SUFFIX_FORMAT: str = "_noisy_s{noise_level:.2f}"
