"""
Utility functions for the BAR_IMPACT package.

This module contains common utilities for I/O operations,
noise generation, reproducibility, data manipulation, inference workflows,
and logging.

Note: Some utilities (NPE workflow, inference) require optional dependencies
(JAX, jaxili, getdist, TARP) and are only available when those packages
are installed.
"""

from bar_impact.utils.io import load_healpy_map, save_results
from bar_impact.utils.logging import (
    LoggingContext,
    configure_logging,
    disable_logging,
    enable_logging,
    get_logger,
    set_log_level,
)
from bar_impact.utils.noise import add_shape_noise
from bar_impact.utils.paths import (
    build_output_suffix,
    get_data_file_paths,
)
from bar_impact.utils.reproducibility import (
    create_seed_worker_initializer,
    get_deterministic_seed,
    seed_worker,
)

# Conditionally import inference utilities (requires JAX, jaxili, tarp)
try:
    from bar_impact.utils.inference import (
        plot_tarp_coverage,
        run_tarp_coverage_test,
        train_npe_with_nan_retry,
    )

    _HAS_INFERENCE_UTILS = True
except ImportError:
    _HAS_INFERENCE_UTILS = False
    run_tarp_coverage_test = None
    plot_tarp_coverage = None
    train_npe_with_nan_retry = None

# Conditionally import NPE workflow utilities (requires JAX, jaxili, getdist)
try:
    from bar_impact.utils.npe_workflow import (
        STANDARD_COSMO_PARAMS,
        create_triangle_plot,
        initialize_npe,
        print_analysis_summary,
        print_completion_summary,
        sample_and_save_posterior,
        setup_jax_environment,
        train_or_load_npe,
    )

    _HAS_NPE_WORKFLOW = True
except ImportError:
    _HAS_NPE_WORKFLOW = False
    STANDARD_COSMO_PARAMS = None
    initialize_npe = None
    train_or_load_npe = None
    create_triangle_plot = None
    sample_and_save_posterior = None
    setup_jax_environment = None
    print_analysis_summary = None
    print_completion_summary = None

__all__ = [
    # I/O
    "load_healpy_map",
    "save_results",
    # Noise
    "add_shape_noise",
    # Reproducibility
    "get_deterministic_seed",
    "seed_worker",
    "create_seed_worker_initializer",
    # Paths
    "get_data_file_paths",
    "build_output_suffix",
    # Logging
    "get_logger",
    "configure_logging",
    "set_log_level",
    "disable_logging",
    "enable_logging",
    "LoggingContext",
]

# Add inference utilities if available
if _HAS_INFERENCE_UTILS:
    __all__.extend(
        [
            "run_tarp_coverage_test",
            "plot_tarp_coverage",
            "train_npe_with_nan_retry",
        ]
    )

# Add NPE workflow utilities if available
if _HAS_NPE_WORKFLOW:
    __all__.extend(
        [
            "STANDARD_COSMO_PARAMS",
            "initialize_npe",
            "train_or_load_npe",
            "create_triangle_plot",
            "sample_and_save_posterior",
            "setup_jax_environment",
            "print_analysis_summary",
            "print_completion_summary",
        ]
    )
