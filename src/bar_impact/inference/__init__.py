"""
Inference module for simulation-based inference.

This module provides tools for performing simulation-based inference
using Neural Posterior Estimation (NPE) and posterior validation
using TARP coverage tests.

Classes
-------
NPEInference
    Main class for NPE training and sampling.
NPEConfig
    Configuration for NPE inference.
NPEResult
    Container for NPE results.
CoverageTester
    Coverage testing using TARP method.
CoverageConfig
    Configuration for coverage testing.
CoverageResult
    Container for coverage test results.

Functions
---------
run_npe_inference
    Convenience function to train and sample from NPE.
compute_tarp_coverage
    Compute TARP coverage from posterior samples.
"""

# NPE classes and functions
from bar_impact.inference.npe import (
    NPEInference,
    NPEConfig,
    NPEResult,
    run_npe_inference,
    train_npe_model,
    sample_posterior,
)

# Coverage testing
from bar_impact.inference.coverage import (
    CoverageTester,
    CoverageConfig,
    CoverageResult,
    compute_tarp_coverage,
)

# Fisher forecast (placeholder)
from bar_impact.inference.fisher import run_fisher_forecast


__all__ = [
    # NPE
    "NPEInference",
    "NPEConfig",
    "NPEResult",
    "run_npe_inference",
    "train_npe_model",
    "sample_posterior",
    # Coverage
    "CoverageTester",
    "CoverageConfig",
    "CoverageResult",
    "compute_tarp_coverage",
    # Fisher
    "run_fisher_forecast",
]
