"""
Analysis module for data aggregation and visualization.

This module provides utilities for aggregating processed data,
performing statistical analysis, and creating visualizations.
"""

from bar_impact.analysis.aggregation import (
    AggregationConfig,
    ResultsAggregator,
    aggregate_l1_norms,
    aggregate_power_spectra,
    aggregate_results,
    load_datavectors,
)
from bar_impact.analysis.visualization import (
    CoveragePlotter,
    PlotConfig,
    PosteriorPlotter,
    PowerSpectrumPlotter,
    plot_power_spectrum,
    plot_triangle,
    visualize_coverage,
)

__all__ = [
    # Aggregation
    "ResultsAggregator",
    "AggregationConfig",
    "aggregate_results",
    "aggregate_l1_norms",
    "aggregate_power_spectra",
    "load_datavectors",
    # Visualization
    "PosteriorPlotter",
    "CoveragePlotter",
    "PowerSpectrumPlotter",
    "PlotConfig",
    "visualize_coverage",
    "plot_power_spectrum",
    "plot_triangle",
]
