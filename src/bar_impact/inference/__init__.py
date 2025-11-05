"""
Inference module for Neural Posterior Estimation.

This module provides tools for performing simulation-based inference
using Neural Posterior Estimation (NPE) on cosmological data.
"""

from bar_impact.inference.npe import run_npe_inference
from bar_impact.inference.fisher import run_fisher_forecast

__all__ = [
    "run_npe_inference",
    "run_fisher_forecast",
]
