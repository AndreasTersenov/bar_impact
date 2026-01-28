"""
Band-limited Nulling Transform (BNT) for cosmological analysis.

This module implements BNT transformations to separate signals
across different redshift bins.
"""

import numpy as np
from typing import Optional

from bar_impact.constants import BNT_MATRIX_DEFAULT, get_bnt_matrix


def apply_bnt_transform(
    maps: np.ndarray,
    bnt_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply Band-limited Nulling Transform to a set of maps.
    
    Parameters
    ----------
    maps : np.ndarray
        Input maps with shape (n_bins, n_pixels) or (n_bins, ...)
    bnt_matrix : np.ndarray, optional
        BNT transformation matrix. If None, uses default 4-bin matrix.
        
    Returns
    -------
    np.ndarray
        Transformed maps with same shape as input
        
    Notes
    -----
    The BNT transform decorrelates signals across different redshift bins,
    making it useful for separating cosmological signals.
    
    Examples
    --------
    >>> maps = np.random.randn(4, 12*512**2)  # 4 redshift bins
    >>> bnt_maps = apply_bnt_transform(maps)
    >>> bnt_maps.shape
    (4, 3145728)
    """
    if bnt_matrix is None:
        bnt_matrix = BNT_MATRIX_DEFAULT
    
    # Apply transformation: BNT @ maps
    if maps.ndim == 2:
        return bnt_matrix @ maps
    else:
        # Handle higher dimensional arrays
        original_shape = maps.shape
        n_bins = original_shape[0]
        maps_flat = maps.reshape(n_bins, -1)
        transformed = bnt_matrix @ maps_flat
        return transformed.reshape(original_shape)


# Re-export get_bnt_matrix from constants for backward compatibility
__all__ = ["apply_bnt_transform", "get_bnt_matrix", "BNT_MATRIX_DEFAULT"]
