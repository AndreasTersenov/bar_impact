"""
Core data structures for BAR_IMPACT.

This subpackage provides the fundamental data classes used throughout
the package for representing convergence maps, survey masks, and 
data vectors.

Classes
-------
ConvergenceMap
    Representation of a HEALPix convergence (kappa) map.
ConvergenceMapCollection
    Collection of maps across tomographic redshift bins.
SurveyMask
    Survey footprint mask for partial-sky analysis.
DataVector
    Container for summary statistics used in inference.
DataVectorCollection
    Collection of data vectors from multiple simulations.
"""

from bar_impact.core.maps import ConvergenceMap, ConvergenceMapCollection
from bar_impact.core.masks import SurveyMask, get_cached_mask, clear_mask_cache
from bar_impact.core.datavectors import DataVector, DataVectorCollection

__all__ = [
    "ConvergenceMap",
    "ConvergenceMapCollection",
    "SurveyMask",
    "get_cached_mask",
    "clear_mask_cache",
    "DataVector",
    "DataVectorCollection",
]
