"""
BAR_IMPACT: Baryon Impact Analysis for Cosmological Maps

This package provides tools to analyze the impact of baryons on cosmological 
weak lensing maps through wavelet-based L1 norm calculations and 
simulation-based inference.
"""

__version__ = "0.1.0"
__author__ = "Andreas Tersenov"

# Import key functions for convenient access
from bar_impact.processing import (
    process_l1_norms,
    process_power_spectrum,
    process_peak_counts,
)
from bar_impact.inference import run_npe_inference
from bar_impact.analysis import aggregate_results

__all__ = [
    "process_l1_norms",
    "process_power_spectrum", 
    "process_peak_counts",
    "run_npe_inference",
    "aggregate_results",
    "__version__",
]
