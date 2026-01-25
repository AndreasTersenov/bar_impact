"""
Processing module for cosmological map analysis.

This module provides processors for computing summary statistics from
HEALPix convergence maps, including power spectra, L1 norms, and peak counts.

Classes
-------
BaseProcessor
    Abstract base class for all map processors.
ProcessingConfig
    Configuration dataclass for processing options.
PowerSpectrumProcessor
    Processor for computing angular power spectra.
L1NormProcessor
    Processor for computing L1 norms of wavelet coefficients.
PeakCountProcessor
    Processor for computing wavelet peak counts.

Functions
---------
apply_bnt_transform
    Apply BNT transform to a set of tomographic convergence maps.
compute_power_spectrum
    Compute power spectrum from a single map.
compute_l1_norms
    Compute L1 norms from a single map.
compute_peak_counts
    Compute peak counts from a single map.
"""

# Base classes
from bar_impact.processing.base import (
    BaseProcessor,
    ProcessingConfig,
)

# BNT transforms
from bar_impact.processing.bnt_transforms import (
    apply_bnt_transform,
    get_bnt_matrix,
    BNT_MATRIX_DEFAULT,
)

# Power spectrum processing
from bar_impact.processing.power_spectrum import (
    PowerSpectrumProcessor,
    PowerSpectrumConfig,
    compute_power_spectrum,
    compute_cross_power_spectrum,
)

# L1 norm processing
from bar_impact.processing.l1_norms import (
    L1NormProcessor,
    L1NormConfig,
    compute_l1_norms,
)

# Peak count processing
from bar_impact.processing.peak_counts import (
    PeakCountProcessor,
    PeakCountConfig,
    compute_peak_counts,
)


__all__ = [
    # Base classes
    "BaseProcessor",
    "ProcessingConfig",
    # BNT
    "apply_bnt_transform",
    "get_bnt_matrix",
    "BNT_MATRIX_DEFAULT",
    # Power spectrum
    "PowerSpectrumProcessor",
    "PowerSpectrumConfig",
    "compute_power_spectrum",
    "compute_cross_power_spectrum",
    # L1 norms
    "L1NormProcessor",
    "L1NormConfig",
    "compute_l1_norms",
    # Peak counts
    "PeakCountProcessor",
    "PeakCountConfig",
    "compute_peak_counts",
]
