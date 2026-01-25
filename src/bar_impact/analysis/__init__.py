"""
Analysis module for data aggregation and visualization.

This module provides utilities for aggregating processed data,
performing statistical analysis, and creating visualizations.
"""

from bar_impact.analysis.aggregation import (
    ResultsAggregator,
    AggregationConfig,
    aggregate_results,
    aggregate_l1_norms,
    aggregate_power_spectra,
    load_datavectors,
)
from bar_impact.analysis.visualization import (
    PosteriorPlotter,
    CoveragePlotter,
    PowerSpectrumPlotter,
    PlotConfig,
    visualize_coverage,
    plot_power_spectrum,
    plot_triangle,
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
