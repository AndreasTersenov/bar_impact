"""
Processing module for cosmological map analysis.

This module contains functions for processing HEALPix convergence maps,
including L1 norm calculations, power spectrum analysis, and peak counting.
"""

from bar_impact.processing.bnt_transforms import (
    apply_bnt_transform,
    get_bnt_matrix,
    BNT_MATRIX_DEFAULT,
)

# These will be implemented in Phase 2
# from bar_impact.processing.l1_norms import process_l1_norms
# from bar_impact.processing.power_spectrum import process_power_spectrum
# from bar_impact.processing.peak_counts import process_peak_counts

__all__ = [
    "apply_bnt_transform",
    "get_bnt_matrix",
    "BNT_MATRIX_DEFAULT",
    # "process_l1_norms",
    # "process_power_spectrum",
    # "process_peak_counts",
]
