"""
Utility functions for the BAR_IMPACT package.

This module contains common utilities for I/O operations,
noise generation, reproducibility, data manipulation, and inference workflows.
"""

from bar_impact.utils.io import load_healpy_map, save_results
from bar_impact.utils.noise import add_shape_noise
from bar_impact.utils.reproducibility import (
    get_deterministic_seed,
    seed_worker,
    create_seed_worker_initializer,
)
from bar_impact.utils.paths import (
    get_data_file_paths,
    build_output_suffix,
)
from bar_impact.utils.inference import (
    run_tarp_coverage_test,
    plot_tarp_coverage,
    train_npe_with_nan_retry,
)
from bar_impact.utils.npe_workflow import (
    STANDARD_COSMO_PARAMS,
    initialize_npe,
    train_or_load_npe,
    create_triangle_plot,
    sample_and_save_posterior,
    setup_jax_environment,
    print_analysis_summary,
    print_completion_summary,
)

__all__ = [
    "load_healpy_map",
    "save_results",
    "add_shape_noise",
    "get_deterministic_seed",
    "seed_worker",
    "create_seed_worker_initializer",
    "get_data_file_paths",
    "build_output_suffix",
    "run_tarp_coverage_test",
    "plot_tarp_coverage",
    "train_npe_with_nan_retry",
    "STANDARD_COSMO_PARAMS",
    "initialize_npe",
    "train_or_load_npe",
    "create_triangle_plot",
    "sample_and_save_posterior",
    "setup_jax_environment",
    "print_analysis_summary",
    "print_completion_summary",
]
