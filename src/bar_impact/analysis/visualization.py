"""
Visualization utilities for cosmological analysis results.

This module provides classes for creating publication-quality plots
of posterior distributions, coverage tests, and summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    from getdist import MCSamples
    from getdist import plots as gdplots

    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    MCSamples = None
    gdplots = None


__all__ = [
    "PosteriorPlotter",
    "CoveragePlotter",
    "PowerSpectrumPlotter",
    "PlotConfig",
    "visualize_coverage",
    "plot_power_spectrum",
    "plot_triangle",
]


@dataclass
class PlotConfig:
    """
    Configuration for plot styling.

    Parameters
    ----------
    figsize : tuple of float
        Figure size in inches.
    dpi : int
        Resolution for saved figures.
    fontsize : int
        Base font size.
    linewidth : float
        Default line width.
    colormap : str
        Default colormap name.
    style : str
        Matplotlib style to use.
    """

    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 150
    fontsize: int = 12
    linewidth: float = 1.5
    colormap: str = "viridis"
    style: Optional[str] = None


class PosteriorPlotter:
    """
    Plotter for posterior distributions using getdist.

    This class wraps getdist functionality to create triangle plots
    and 1D/2D marginal distributions.

    Parameters
    ----------
    config : PlotConfig, optional
        Plot configuration.

    Examples
    --------
    >>> from bar_impact.analysis import PosteriorPlotter
    >>>
    >>> plotter = PosteriorPlotter()
    >>>
    >>> # Create triangle plot from samples
    >>> fig = plotter.triangle_plot(
    ...     samples,  # shape (n_samples, n_params)
    ...     param_names=["Om", "S8"],
    ...     param_labels=[r"$\\Omega_m$", r"$S_8$"],
    ... )
    >>> fig.savefig("posterior.png")
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        if not HAS_GETDIST:
            raise ImportError(
                "getdist is required for PosteriorPlotter. "
                "Install with: pip install getdist"
            )

        self.config = config if config is not None else PlotConfig()

        if self.config.style:
            plt.style.use(self.config.style)

    def create_samples(
        self,
        chains: np.ndarray,
        param_names: Optional[List[str]] = None,
        param_labels: Optional[List[str]] = None,
        label: str = "posterior",
        weights: Optional[np.ndarray] = None,
    ) -> MCSamples:
        """
        Create a getdist MCSamples object.

        Parameters
        ----------
        chains : np.ndarray
            Posterior samples, shape (n_samples, n_params).
        param_names : list of str, optional
            Parameter names (used internally).
        param_labels : list of str, optional
            Parameter labels (for plots, can include LaTeX).
        label : str
            Label for this set of samples.
        weights : np.ndarray, optional
            Sample weights.

        Returns
        -------
        MCSamples
            getdist samples object.
        """
        n_params = chains.shape[1]

        if param_names is None:
            param_names = [f"p{i}" for i in range(n_params)]

        if param_labels is None:
            param_labels = param_names

        return MCSamples(
            samples=chains,
            names=param_names,
            labels=param_labels,
            label=label,
            weights=weights,
        )

    def triangle_plot(
        self,
        samples: Union[np.ndarray, MCSamples, List[MCSamples]],
        param_names: Optional[List[str]] = None,
        param_labels: Optional[List[str]] = None,
        filled: bool = True,
        legend_labels: Optional[List[str]] = None,
        contour_colors: Optional[List[str]] = None,
        truth_values: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create a triangle plot (corner plot) of posteriors.

        Parameters
        ----------
        samples : array, MCSamples, or list of MCSamples
            Posterior samples. If array, converted to MCSamples.
        param_names : list of str, optional
            Parameter names.
        param_labels : list of str, optional
            Parameter labels with LaTeX.
        filled : bool
            Whether to fill contours.
        legend_labels : list of str, optional
            Labels for multiple sample sets.
        contour_colors : list of str, optional
            Colors for contours.
        truth_values : dict, optional
            True parameter values to mark.
        **kwargs
            Additional arguments to triangle_plot.

        Returns
        -------
        matplotlib.figure.Figure
            Triangle plot figure.
        """
        # Convert to MCSamples if needed
        if isinstance(samples, np.ndarray):
            samples = [self.create_samples(samples, param_names, param_labels)]
        elif isinstance(samples, MCSamples):
            samples = [samples]

        # Create plotter
        g = gdplots.get_subplot_plotter()
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add = 0.4

        # Apply config
        g.settings.axes_fontsize = self.config.fontsize
        g.settings.lab_fontsize = self.config.fontsize + 2
        g.settings.legend_fontsize = self.config.fontsize

        # Set up contour args
        if contour_colors is not None:
            kwargs.setdefault("contour_colors", contour_colors)

        if legend_labels is not None:
            kwargs.setdefault("legend_labels", legend_labels)

        # Create plot
        g.triangle_plot(
            samples,
            filled=filled,
            **kwargs,
        )

        # Add truth values if provided
        if truth_values is not None:
            for name, value in truth_values.items():
                g.add_x_marker(value, label=name)

        return g.fig

    def plot_1d(
        self,
        samples: Union[np.ndarray, MCSamples, List[MCSamples]],
        param_names: Optional[List[str]] = None,
        param_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create 1D marginal distribution plots.

        Parameters
        ----------
        samples : array, MCSamples, or list
            Posterior samples.
        param_names : list of str, optional
            Parameters to plot.
        param_labels : list of str, optional
            Parameter labels.
        **kwargs
            Additional plot options.

        Returns
        -------
        matplotlib.figure.Figure
            1D distributions figure.
        """
        if isinstance(samples, np.ndarray):
            samples = [self.create_samples(samples, param_names, param_labels)]
        elif isinstance(samples, MCSamples):
            samples = [samples]

        g = gdplots.get_subplot_plotter()
        g.settings.axes_fontsize = self.config.fontsize

        g.plots_1d(samples, **kwargs)

        return g.fig

    def plot_2d(
        self,
        samples: Union[np.ndarray, MCSamples, List[MCSamples]],
        param1: str,
        param2: str,
        param_names: Optional[List[str]] = None,
        param_labels: Optional[List[str]] = None,
        filled: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """
        Create a 2D marginal distribution plot.

        Parameters
        ----------
        samples : array, MCSamples, or list
            Posterior samples.
        param1 : str
            First parameter name.
        param2 : str
            Second parameter name.
        param_names : list of str, optional
            All parameter names.
        param_labels : list of str, optional
            Parameter labels.
        filled : bool
            Whether to fill contours.
        **kwargs
            Additional plot options.

        Returns
        -------
        matplotlib.figure.Figure
            2D distribution figure.
        """
        if isinstance(samples, np.ndarray):
            samples = [self.create_samples(samples, param_names, param_labels)]
        elif isinstance(samples, MCSamples):
            samples = [samples]

        g = gdplots.get_subplot_plotter()
        g.settings.axes_fontsize = self.config.fontsize

        g.plot_2d(samples, param1, param2, filled=filled, **kwargs)

        return g.fig


class CoveragePlotter:
    """
    Plotter for TARP coverage test results.

    This class creates coverage plots showing expected vs observed
    coverage probabilities with confidence bands.

    Parameters
    ----------
    config : PlotConfig, optional
        Plot configuration.

    Examples
    --------
    >>> from bar_impact.analysis import CoveragePlotter
    >>>
    >>> plotter = CoveragePlotter()
    >>> fig = plotter.plot_coverage(
    ...     ecp=coverage_result.ecp,
    ...     alpha=coverage_result.alpha,
    ...     ecp_std=coverage_result.ecp_std,
    ... )
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config if config is not None else PlotConfig()

        if self.config.style:
            plt.style.use(self.config.style)

    def plot_coverage(
        self,
        ecp: np.ndarray,
        alpha: np.ndarray,
        ecp_std: Optional[np.ndarray] = None,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        label: Optional[str] = None,
        color: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        show_diagonal: bool = True,
        show_uncertainty: bool = True,
    ) -> plt.Figure:
        """
        Plot TARP coverage results.

        Parameters
        ----------
        ecp : np.ndarray
            Expected Coverage Probability values.
        alpha : np.ndarray
            Credibility levels.
        ecp_std : np.ndarray, optional
            Standard deviation of ECP from bootstrapping.
        n_bootstrap : int
            Number of bootstrap samples (for uncertainty bands).
        confidence_level : float
            Confidence level for uncertainty bands.
        label : str, optional
            Label for the coverage line.
        color : str, optional
            Line color.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        show_diagonal : bool
            Whether to show perfect calibration line.
        show_uncertainty : bool
            Whether to show uncertainty bands.

        Returns
        -------
        matplotlib.figure.Figure
            Coverage plot figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.figure

        # Default color
        if color is None:
            color = "C0"

        # Plot diagonal (perfect calibration)
        if show_diagonal:
            ax.plot(
                [0, 1],
                [0, 1],
                "k--",
                linewidth=self.config.linewidth,
                label="Perfect calibration",
                alpha=0.7,
            )

        # Plot coverage
        ax.plot(
            alpha,
            ecp,
            color=color,
            linewidth=self.config.linewidth * 1.5,
            label=label or "Coverage",
        )

        # Plot uncertainty bands
        if show_uncertainty and ecp_std is not None:
            z = 1.96 if confidence_level == 0.95 else 1.0  # Approx z-score
            ax.fill_between(
                alpha,
                ecp - z * ecp_std,
                ecp + z * ecp_std,
                color=color,
                alpha=0.3,
                label=f"{int(confidence_level * 100)}% CI",
            )

        # Styling
        ax.set_xlabel(r"Credibility level $\alpha$", fontsize=self.config.fontsize)
        ax.set_ylabel("Expected Coverage Probability", fontsize=self.config.fontsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=self.config.fontsize - 2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def plot_multi_coverage(
        self,
        coverage_results: List[Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot multiple coverage results for comparison.

        Parameters
        ----------
        coverage_results : list of dict
            List of dicts with 'ecp', 'alpha', and optional 'ecp_std'.
        labels : list of str, optional
            Labels for each result.
        colors : list of str, optional
            Colors for each result.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.figure.Figure
            Multi-coverage plot figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.figure

        n_results = len(coverage_results)

        if labels is None:
            labels = [f"Result {i + 1}" for i in range(n_results)]

        if colors is None:
            colors = [f"C{i}" for i in range(n_results)]

        # Plot diagonal once
        ax.plot([0, 1], [0, 1], "k--", linewidth=self.config.linewidth, alpha=0.7)

        # Plot each result
        for i, result in enumerate(coverage_results):
            ax.plot(
                result["alpha"],
                result["ecp"],
                color=colors[i],
                linewidth=self.config.linewidth * 1.5,
                label=labels[i],
            )

            if "ecp_std" in result:
                ax.fill_between(
                    result["alpha"],
                    result["ecp"] - 1.96 * result["ecp_std"],
                    result["ecp"] + 1.96 * result["ecp_std"],
                    color=colors[i],
                    alpha=0.2,
                )

        ax.set_xlabel(r"Credibility level $\alpha$", fontsize=self.config.fontsize)
        ax.set_ylabel("Expected Coverage Probability", fontsize=self.config.fontsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=self.config.fontsize - 2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig


class PowerSpectrumPlotter:
    """
    Plotter for angular power spectra.

    Parameters
    ----------
    config : PlotConfig, optional
        Plot configuration.

    Examples
    --------
    >>> from bar_impact.analysis import PowerSpectrumPlotter
    >>>
    >>> plotter = PowerSpectrumPlotter()
    >>> fig = plotter.plot_cls(
    ...     ells=np.arange(100, 1000),
    ...     cls=power_spectrum_data,
    ...     yerr=power_spectrum_errors,
    ... )
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config if config is not None else PlotConfig()

        if self.config.style:
            plt.style.use(self.config.style)

    def plot_cls(
        self,
        ells: np.ndarray,
        cls: np.ndarray,
        yerr: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        logx: bool = True,
        logy: bool = True,
        multiply_ell: bool = True,
        ell_power: int = 2,
    ) -> plt.Figure:
        """
        Plot angular power spectrum C_ell.

        Parameters
        ----------
        ells : np.ndarray
            Multipole moments.
        cls : np.ndarray
            Power spectrum values.
        yerr : np.ndarray, optional
            Error bars.
        label : str, optional
            Plot label.
        color : str, optional
            Line color.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        logx : bool
            Use log scale for x-axis.
        logy : bool
            Use log scale for y-axis.
        multiply_ell : bool
            Whether to plot ell^power * C_ell.
        ell_power : int
            Power for ell multiplication.

        Returns
        -------
        matplotlib.figure.Figure
            Power spectrum plot figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.figure

        # Prepare y values
        if multiply_ell:
            ell_factor = ells**ell_power
            y = cls * ell_factor
            if yerr is not None:
                yerr = yerr * ell_factor
            ylabel = rf"$\ell^{ell_power} C_\ell$"
        else:
            y = cls
            ylabel = r"$C_\ell$"

        # Plot
        if yerr is not None:
            ax.errorbar(
                ells,
                y,
                yerr=yerr,
                color=color,
                linewidth=self.config.linewidth,
                label=label,
                capsize=2,
            )
        else:
            ax.plot(
                ells,
                y,
                color=color,
                linewidth=self.config.linewidth,
                label=label,
            )

        # Scaling
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        ax.set_xlabel(r"$\ell$", fontsize=self.config.fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.fontsize)

        if label:
            ax.legend(fontsize=self.config.fontsize - 2)

        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        return fig

    def plot_ratio(
        self,
        ells: np.ndarray,
        cls1: np.ndarray,
        cls2: np.ndarray,
        label: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot ratio of two power spectra.

        Parameters
        ----------
        ells : np.ndarray
            Multipole moments.
        cls1 : np.ndarray
            First power spectrum (numerator).
        cls2 : np.ndarray
            Second power spectrum (denominator).
        label : str, optional
            Plot label.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        matplotlib.figure.Figure
            Ratio plot figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.figure

        ratio = cls1 / cls2

        ax.plot(
            ells,
            ratio,
            linewidth=self.config.linewidth,
            label=label,
        )
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$", fontsize=self.config.fontsize)
        ax.set_ylabel(r"$C_\ell^{(1)} / C_\ell^{(2)}$", fontsize=self.config.fontsize)
        ax.grid(True, alpha=0.3)

        if label:
            ax.legend(fontsize=self.config.fontsize - 2)

        plt.tight_layout()

        return fig


# Functional interface for backwards compatibility


def visualize_coverage(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualize TARP coverage test results.

    Parameters
    ----------
    results : dict
        TARP coverage test results with 'ecp' and 'alpha' keys.
    output_path : str, optional
        Path to save figure.
    **kwargs
        Plotting options.

    Returns
    -------
    matplotlib.figure.Figure
        Coverage plot figure.
    """
    plotter = CoveragePlotter()

    fig = plotter.plot_coverage(
        ecp=results.get("ecp", results.get("expected_coverage")),
        alpha=results.get("alpha", results.get("credibility_levels")),
        ecp_std=results.get("ecp_std"),
        **kwargs,
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_power_spectrum(
    ells: np.ndarray,
    cls: np.ndarray,
    output_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot angular power spectrum.

    Parameters
    ----------
    ells : np.ndarray
        Multipole moments.
    cls : np.ndarray
        Power spectrum values.
    output_path : str, optional
        Path to save figure.
    **kwargs
        Plotting options.

    Returns
    -------
    matplotlib.figure.Figure
        Power spectrum plot.
    """
    plotter = PowerSpectrumPlotter()

    fig = plotter.plot_cls(ells, cls, **kwargs)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_triangle(
    samples: np.ndarray,
    param_names: Optional[List[str]] = None,
    param_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Create a triangle plot of posterior samples.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (n_samples, n_params).
    param_names : list of str, optional
        Parameter names.
    param_labels : list of str, optional
        Parameter labels for display.
    output_path : str, optional
        Path to save figure.
    **kwargs
        Additional plotting options.

    Returns
    -------
    matplotlib.figure.Figure
        Triangle plot figure.
    """
    plotter = PosteriorPlotter()

    fig = plotter.triangle_plot(
        samples,
        param_names=param_names,
        param_labels=param_labels,
        **kwargs,
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
