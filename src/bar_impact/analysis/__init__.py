"""
Analysis module for data aggregation and visualization.

This module provides utilities for aggregating processed data,
performing statistical analysis, and creating visualizations.
"""

from bar_impact.analysis.aggregation import aggregate_results
from bar_impact.analysis.visualization import visualize_coverage

__all__ = [
    "aggregate_results",
    "visualize_coverage",
]
