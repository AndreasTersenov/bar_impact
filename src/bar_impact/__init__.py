"""
BAR_IMPACT: Baryon Impact Analysis for Cosmological Maps

This package provides tools to analyze the impact of baryons on cosmological
weak lensing maps through wavelet-based L1 norm calculations and
simulation-based inference.

Main Features
-------------
- Convergence map processing with shape noise and survey masks
- Multiple summary statistics: L1 norms, power spectra, peak counts
- BNT (Bernardeau-Nishimichi-Taruya) transform for tomographic analysis
- Neural Posterior Estimation (NPE) for simulation-based inference
- Coverage testing with TARP

Quick Start
-----------
>>> from bar_impact.core import ConvergenceMap, SurveyMask
>>> from bar_impact.processing import PowerSpectrumProcessor, L1NormProcessor
>>> from bar_impact.constants import BNT_MATRIX_DEFAULT
>>>
>>> # Load a map
>>> kappa = ConvergenceMap.from_h5("map.h5", bin_number=1)
>>>
>>> # Add noise and apply mask
>>> mask = SurveyMask.create_disk_mask(nside=512, target_area_sqdeg=14000)
>>> kappa_noisy = kappa.add_shape_noise(sigma_e=0.26)
>>> kappa_masked = kappa_noisy.apply_mask(mask)
>>>
>>> # Compute statistics
>>> processor = PowerSpectrumProcessor(lmax=1024)
>>> cls = processor.process_single(kappa_masked.data)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bar_impact")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without pip install -e)
    __version__ = "0.1.0.dev0"

__author__ = "Andreas Tersenov"

# Core data structures
from bar_impact.core import (
    ConvergenceMap,
    ConvergenceMapCollection,
    SurveyMask,
    DataVector,
    DataVectorCollection,
)

# Constants
from bar_impact.constants import (
    BNT_MATRIX_DEFAULT,
    get_bnt_matrix,
    DEFAULT_NSIDE,
    DEFAULT_LMAX,
    DEFAULT_SIGMA_E,
    DEFAULT_GALAXY_DENSITY,
    COSMO_PARAM_NAMES,
)

# Processing classes and functions
from bar_impact.processing import (
    # Base classes
    BaseProcessor,
    ProcessingConfig,
    # BNT
    apply_bnt_transform,
    # Power spectrum
    PowerSpectrumProcessor,
    PowerSpectrumConfig,
    compute_power_spectrum,
    compute_cross_power_spectrum,
    # L1 norms
    L1NormProcessor,
    L1NormConfig,
    compute_l1_norms,
    # Peak counts
    PeakCountProcessor,
    PeakCountConfig,
    compute_peak_counts,
)

# Inference classes and functions
from bar_impact.inference import (
    NPEInference,
    NPEConfig,
    NPEResult,
    run_npe_inference,
    CoverageTester,
    CoverageConfig,
    CoverageResult,
    compute_tarp_coverage,
)

# Custom exceptions
from bar_impact.exceptions import (
    BarImpactError,
    ConfigurationError,
    DataLoadError,
    ProcessingError,
    MaskError,
    TransformError,
    InferenceError,
    TrainingError,
    SamplingError,
)

# Logging utilities
from bar_impact.utils.logging import get_logger, configure_logging

# Analysis functions (will be implemented in Phase 4)
# from bar_impact.analysis import aggregate_results

__all__ = [
    # Core classes
    "ConvergenceMap",
    "ConvergenceMapCollection", 
    "SurveyMask",
    "DataVector",
    "DataVectorCollection",
    # Constants
    "BNT_MATRIX_DEFAULT",
    "get_bnt_matrix",
    "DEFAULT_NSIDE",
    "DEFAULT_LMAX",
    "DEFAULT_SIGMA_E",
    "DEFAULT_GALAXY_DENSITY",
    "COSMO_PARAM_NAMES",
    # Processing - Base
    "BaseProcessor",
    "ProcessingConfig",
    "apply_bnt_transform",
    # Processing - Power spectrum
    "PowerSpectrumProcessor",
    "PowerSpectrumConfig",
    "compute_power_spectrum",
    "compute_cross_power_spectrum",
    # Processing - L1 norms
    "L1NormProcessor",
    "L1NormConfig",
    "compute_l1_norms",
    # Processing - Peak counts
    "PeakCountProcessor",
    "PeakCountConfig",
    "compute_peak_counts",
    # Inference - NPE
    "NPEInference",
    "NPEConfig",
    "NPEResult",
    "run_npe_inference",
    # Inference - Coverage
    "CoverageTester",
    "CoverageConfig",
    "CoverageResult",
    "compute_tarp_coverage",
    # Exceptions
    "BarImpactError",
    "ConfigurationError",
    "DataLoadError",
    "ProcessingError",
    "MaskError",
    "TransformError",
    "InferenceError",
    "TrainingError",
    "SamplingError",
    # Logging
    "get_logger",
    "configure_logging",
    # Version
    "__version__",
]
