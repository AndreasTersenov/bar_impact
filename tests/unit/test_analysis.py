"""
Unit tests for bar_impact.analysis module.
"""

import os

import numpy as np
import pytest

# Import availability flag from conftest (pytest auto-imports conftest.py)
from conftest import HAS_GETDIST

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
    PowerSpectrumPlotter,
    plot_power_spectrum,
    visualize_coverage,
)

# =============================================================================
# AggregationConfig Tests
# =============================================================================


class TestAggregationConfig:
    """Tests for AggregationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AggregationConfig()

        assert config.filter_nans is True
        assert config.filter_infs is True
        assert config.filter_zeros is False
        assert config.compute_statistics is True
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AggregationConfig(
            filter_nans=False,
            filter_infs=False,
            filter_zeros=True,
            compute_statistics=False,
            verbose=False,
        )

        assert config.filter_nans is False
        assert config.filter_infs is False
        assert config.filter_zeros is True
        assert config.compute_statistics is False
        assert config.verbose is False


# =============================================================================
# ResultsAggregator Tests
# =============================================================================


class TestResultsAggregator:
    """Tests for ResultsAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator with quiet config."""
        config = AggregationConfig(verbose=False)
        return ResultsAggregator(config=config)

    @pytest.fixture
    def temp_data_files(self, tmp_path):
        """Create temporary data files for testing."""
        # Create test data
        data1 = np.random.randn(10, 5)
        data2 = np.random.randn(10, 5)
        data3 = np.random.randn(10, 5)

        # Save to files
        f1 = tmp_path / "data_001.npy"
        f2 = tmp_path / "data_002.npy"
        f3 = tmp_path / "data_003.npy"

        np.save(f1, data1)
        np.save(f2, data2)
        np.save(f3, data3)

        return [str(f1), str(f2), str(f3)], [data1, data2, data3]

    def test_init_default_config(self):
        """Test initialization with default config."""
        aggregator = ResultsAggregator()

        assert aggregator.config is not None
        assert isinstance(aggregator.config, AggregationConfig)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = AggregationConfig(verbose=False)
        aggregator = ResultsAggregator(config=config)

        assert aggregator.config.verbose is False

    def test_load_from_files(self, aggregator, temp_data_files):
        """Test loading and concatenating data from files."""
        file_paths, original_data = temp_data_files

        result = aggregator.load_from_files(file_paths)

        expected = np.concatenate(original_data, axis=0)
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == (30, 5)

    def test_load_from_pattern(self, aggregator, temp_data_files, tmp_path):
        """Test loading data using glob pattern."""
        file_paths, _ = temp_data_files

        pattern = str(tmp_path / "data_*.npy")
        result = aggregator.load_from_pattern(pattern)

        assert result.shape == (30, 5)

    def test_load_from_pattern_no_files(self, aggregator, tmp_path):
        """Test error when no files match pattern."""
        pattern = str(tmp_path / "nonexistent_*.npy")

        with pytest.raises(FileNotFoundError):
            aggregator.load_from_pattern(pattern)

    def test_load_multi_bin(self, aggregator, tmp_path):
        """Test loading and concatenating multi-bin data."""
        # Create bin data
        bin1 = np.random.randn(20, 10)
        bin2 = np.random.randn(20, 10)

        f1 = tmp_path / "bin1.npy"
        f2 = tmp_path / "bin2.npy"
        np.save(f1, bin1)
        np.save(f2, bin2)

        result = aggregator.load_multi_bin([str(f1), str(f2)], axis=1)

        assert result.shape == (20, 20)

    def test_load_with_parameters(self, aggregator, tmp_path):
        """Test loading data with corresponding parameters."""
        data = np.random.randn(100, 10)
        params = np.random.randn(100, 3)

        data_path = tmp_path / "data.npy"
        params_path = tmp_path / "params.npy"
        np.save(data_path, data)
        np.save(params_path, params)

        loaded_data, loaded_params = aggregator.load_with_parameters(
            data_path, params_path
        )

        np.testing.assert_array_equal(loaded_data, data)
        np.testing.assert_array_equal(loaded_params, params)

    def test_load_with_parameters_mismatch(self, aggregator, tmp_path):
        """Test error when data and params have different lengths."""
        data = np.random.randn(100, 10)
        params = np.random.randn(50, 3)  # Different length

        data_path = tmp_path / "data.npy"
        params_path = tmp_path / "params.npy"
        np.save(data_path, data)
        np.save(params_path, params)

        with pytest.raises(ValueError, match="different lengths"):
            aggregator.load_with_parameters(data_path, params_path)

    def test_filter_by_mask_with_indices(self, aggregator):
        """Test filtering data by index array."""
        data = np.arange(20).reshape(10, 2)
        params = np.arange(10)
        indices = np.array([0, 2, 4, 6, 8])

        filtered_data, filtered_params = aggregator.filter_by_mask(
            data, params, valid_indices=indices
        )

        assert filtered_data.shape == (5, 2)
        assert len(filtered_params) == 5
        np.testing.assert_array_equal(filtered_params, indices)

    def test_filter_by_mask_with_boolean(self, aggregator):
        """Test filtering data by boolean mask."""
        data = np.arange(20).reshape(10, 2)
        params = np.arange(10)
        mask = np.array(
            [True, False, True, False, True, False, True, False, True, False]
        )

        filtered_data, filtered_params = aggregator.filter_by_mask(
            data, params, mask=mask
        )

        assert filtered_data.shape == (5, 2)
        assert len(filtered_params) == 5

    def test_compute_statistics(self, aggregator):
        """Test computing summary statistics."""
        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        stats = aggregator.compute_statistics(data)

        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats

        np.testing.assert_array_almost_equal(stats["mean"], [4, 5, 6])
        np.testing.assert_array_almost_equal(stats["min"], [1, 2, 3])
        np.testing.assert_array_almost_equal(stats["max"], [7, 8, 9])

    def test_select_scales(self, aggregator):
        """Test selecting specific wavelet scales."""
        # Create data: 100 samples, 4 scales, 40 bins each
        data = np.random.randn(100, 4, 40)

        # Select scales 1 and 3
        result = aggregator.select_scales(data, [1, 3])

        assert result.shape == (100, 80)  # 2 scales * 40 bins

    def test_select_scales_flattened_input(self, aggregator):
        """Test selecting scales from already-flattened data."""
        # Create flattened data: 100 samples, 160 features (4 scales * 40 bins)
        data = np.random.randn(100, 160)

        result = aggregator.select_scales(data, [0, 2], nbins_per_scale=40)

        assert result.shape == (100, 80)

    def test_nan_filtering(self, tmp_path):
        """Test that NaN values are filtered when enabled."""
        config = AggregationConfig(filter_nans=True, verbose=False)
        aggregator = ResultsAggregator(config=config)

        # Create data with NaN
        data = np.array(
            [
                [1, 2, 3],
                [np.nan, 5, 6],
                [7, 8, 9],
            ]
        )

        f = tmp_path / "data_with_nan.npy"
        np.save(f, data)

        result = aggregator.load_from_files([str(f)])

        # Row with NaN should be filtered
        assert result.shape[0] == 2


# =============================================================================
# Functional Interface Tests
# =============================================================================


class TestAggregationFunctions:
    """Tests for functional aggregation interface."""

    def test_aggregate_results(self, tmp_path):
        """Test aggregate_results function."""
        # Create test files
        for i in range(3):
            data = np.random.randn(10, 5)
            np.save(tmp_path / f"result_{i:03d}.npy", data)

        pattern = str(tmp_path / "result_*.npy")
        result = aggregate_results(pattern, verbose=False)

        assert "data" in result
        assert result["data"].shape == (30, 5)
        assert "mean" in result
        assert "std" in result

    def test_aggregate_l1_norms(self, tmp_path):
        """Test aggregate_l1_norms function."""
        # Create test L1 norm files
        files = []
        for i in range(2):
            data = np.random.randn(50, 160)  # 4 scales * 40 bins
            f = tmp_path / f"l1_{i}.npy"
            np.save(f, data)
            files.append(str(f))

        result = aggregate_l1_norms(files, verbose=False)

        assert result.shape == (100, 160)

    def test_aggregate_l1_norms_with_scale_selection(self, tmp_path):
        """Test aggregate_l1_norms with scale selection."""
        files = []
        for i in range(2):
            data = np.random.randn(50, 160)
            f = tmp_path / f"l1_{i}.npy"
            np.save(f, data)
            files.append(str(f))

        result = aggregate_l1_norms(files, scale_indices=[0, 2], verbose=False)

        assert result.shape == (100, 80)  # 2 scales * 40 bins

    def test_aggregate_power_spectra(self, tmp_path):
        """Test aggregate_power_spectra function."""
        files = []
        for i in range(2):
            cls = np.random.randn(50, 100)
            f = tmp_path / f"cls_{i}.npy"
            np.save(f, cls)
            files.append(str(f))

        result = aggregate_power_spectra(files, verbose=False)

        assert "cls" in result
        assert "ell" in result
        assert "cls_mean" in result
        assert "cls_std" in result
        assert result["cls"].shape == (100, 100)

    def test_aggregate_power_spectra_ell_range(self, tmp_path):
        """Test aggregate_power_spectra with ell range."""
        files = []
        for i in range(2):
            cls = np.random.randn(50, 500)
            f = tmp_path / f"cls_{i}.npy"
            np.save(f, cls)
            files.append(str(f))

        result = aggregate_power_spectra(files, ell_range=(100, 400), verbose=False)

        assert result["cls"].shape[-1] == 301  # 400 - 100 + 1
        assert result["ell"][0] == 100
        assert result["ell"][-1] == 400

    def test_load_datavectors(self, tmp_path):
        """Test load_datavectors function."""
        data = np.random.randn(100, 50)
        params = np.random.randn(100, 3)

        np.save(tmp_path / "data.npy", data)
        np.save(tmp_path / "params.npy", params)

        loaded_data, loaded_params = load_datavectors(
            tmp_path / "data.npy",
            tmp_path / "params.npy",
        )

        np.testing.assert_array_equal(loaded_data, data)
        np.testing.assert_array_equal(loaded_params, params)


# =============================================================================
# PlotConfig Tests
# =============================================================================


class TestPlotConfig:
    """Tests for PlotConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlotConfig()

        assert config.figsize == (8, 6)
        assert config.dpi == 150
        assert config.fontsize == 12
        assert config.linewidth == 1.5
        assert config.colormap == "viridis"
        assert config.style is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PlotConfig(
            figsize=(10, 8),
            dpi=300,
            fontsize=14,
            linewidth=2.0,
            colormap="plasma",
            style="seaborn",
        )

        assert config.figsize == (10, 8)
        assert config.dpi == 300
        assert config.fontsize == 14


# =============================================================================
# CoveragePlotter Tests
# =============================================================================


class TestCoveragePlotter:
    """Tests for CoveragePlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance."""
        return CoveragePlotter()

    @pytest.fixture
    def coverage_data(self):
        """Generate mock coverage data."""
        alpha = np.linspace(0, 1, 50)
        ecp = alpha + 0.05 * np.random.randn(50)
        ecp = np.clip(ecp, 0, 1)
        ecp_std = 0.02 * np.ones_like(alpha)
        return alpha, ecp, ecp_std

    def test_init(self, plotter):
        """Test plotter initialization."""
        assert plotter.config is not None
        assert isinstance(plotter.config, PlotConfig)

    def test_plot_coverage(self, plotter, coverage_data):
        """Test basic coverage plot."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        alpha, ecp, ecp_std = coverage_data

        fig = plotter.plot_coverage(ecp=ecp, alpha=alpha)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_coverage_with_uncertainty(self, plotter, coverage_data):
        """Test coverage plot with uncertainty bands."""
        import matplotlib

        matplotlib.use("Agg")

        alpha, ecp, ecp_std = coverage_data

        fig = plotter.plot_coverage(
            ecp=ecp,
            alpha=alpha,
            ecp_std=ecp_std,
            show_uncertainty=True,
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_coverage_no_diagonal(self, plotter, coverage_data):
        """Test coverage plot without diagonal line."""
        import matplotlib

        matplotlib.use("Agg")

        alpha, ecp, _ = coverage_data

        fig = plotter.plot_coverage(
            ecp=ecp,
            alpha=alpha,
            show_diagonal=False,
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_multi_coverage(self, plotter, coverage_data):
        """Test plotting multiple coverage results."""
        import matplotlib

        matplotlib.use("Agg")

        alpha, ecp, ecp_std = coverage_data

        results = [
            {"alpha": alpha, "ecp": ecp, "ecp_std": ecp_std},
            {"alpha": alpha, "ecp": ecp + 0.05},
        ]

        fig = plotter.plot_multi_coverage(results, labels=["Method A", "Method B"])

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# =============================================================================
# PowerSpectrumPlotter Tests
# =============================================================================


class TestPowerSpectrumPlotter:
    """Tests for PowerSpectrumPlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance."""
        return PowerSpectrumPlotter()

    @pytest.fixture
    def power_spectrum_data(self):
        """Generate mock power spectrum data."""
        ells = np.arange(100, 1000)
        cls = 1e-5 * (ells / 100.0) ** (-2.5)
        cls_err = 0.1 * cls
        return ells, cls, cls_err

    def test_init(self, plotter):
        """Test plotter initialization."""
        assert plotter.config is not None

    def test_plot_cls(self, plotter, power_spectrum_data):
        """Test basic power spectrum plot."""
        import matplotlib

        matplotlib.use("Agg")

        ells, cls, _ = power_spectrum_data

        fig = plotter.plot_cls(ells, cls)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cls_with_errors(self, plotter, power_spectrum_data):
        """Test power spectrum plot with error bars."""
        import matplotlib

        matplotlib.use("Agg")

        ells, cls, cls_err = power_spectrum_data

        fig = plotter.plot_cls(ells, cls, yerr=cls_err)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cls_linear_scale(self, plotter, power_spectrum_data):
        """Test power spectrum with linear scales."""
        import matplotlib

        matplotlib.use("Agg")

        ells, cls, _ = power_spectrum_data

        fig = plotter.plot_cls(ells, cls, logx=False, logy=False)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cls_no_ell_multiply(self, plotter, power_spectrum_data):
        """Test power spectrum without ell multiplication."""
        import matplotlib

        matplotlib.use("Agg")

        ells, cls, _ = power_spectrum_data

        fig = plotter.plot_cls(ells, cls, multiply_ell=False)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_ratio(self, plotter, power_spectrum_data):
        """Test power spectrum ratio plot."""
        import matplotlib

        matplotlib.use("Agg")

        ells, cls, _ = power_spectrum_data
        cls2 = cls * 1.1  # Slightly different

        fig = plotter.plot_ratio(ells, cls, cls2)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# =============================================================================
# Functional Visualization Interface Tests
# =============================================================================


class TestVisualizationFunctions:
    """Tests for functional visualization interface."""

    def test_visualize_coverage(self):
        """Test visualize_coverage function."""
        import matplotlib

        matplotlib.use("Agg")

        alpha = np.linspace(0, 1, 50)
        ecp = alpha + 0.05 * np.random.randn(50)
        ecp = np.clip(ecp, 0, 1)

        results = {"alpha": alpha, "ecp": ecp}

        fig = visualize_coverage(results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visualize_coverage_alternative_keys(self):
        """Test visualize_coverage with alternative key names."""
        import matplotlib

        matplotlib.use("Agg")

        alpha = np.linspace(0, 1, 50)
        ecp = alpha

        results = {
            "credibility_levels": alpha,
            "expected_coverage": ecp,
        }

        fig = visualize_coverage(results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_power_spectrum_function(self):
        """Test plot_power_spectrum function."""
        import matplotlib

        matplotlib.use("Agg")

        ells = np.arange(100, 500)
        cls = 1e-5 * (ells / 100.0) ** (-2)

        fig = plot_power_spectrum(ells, cls)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visualize_coverage_save_file(self, tmp_path):
        """Test saving coverage plot to file."""
        import matplotlib

        matplotlib.use("Agg")

        alpha = np.linspace(0, 1, 50)
        ecp = alpha

        results = {"alpha": alpha, "ecp": ecp}
        output_path = str(tmp_path / "coverage.png")

        fig = visualize_coverage(results, output_path=output_path)

        assert os.path.exists(output_path)
        import matplotlib.pyplot as plt

        plt.close(fig)


# =============================================================================
# PosteriorPlotter Tests (with mock getdist)
# =============================================================================


class TestPosteriorPlotter:
    """Tests for PosteriorPlotter class (requires getdist)."""

    def test_import_check(self):
        """Test that PosteriorPlotter checks for getdist."""
        from bar_impact.analysis.visualization import HAS_GETDIST

        # This test just verifies the import checking exists
        assert isinstance(HAS_GETDIST, bool)

    @pytest.mark.skipif(
        not HAS_GETDIST,
        reason="getdist not installed",
    )
    def test_create_samples(self):
        """Test creating MCSamples from array."""
        try:
            from bar_impact.analysis.visualization import PosteriorPlotter

            plotter = PosteriorPlotter()

            chains = np.random.randn(1000, 2)
            samples = plotter.create_samples(
                chains,
                param_names=["Om", "S8"],
                param_labels=[r"$\Omega_m$", r"$S_8$"],
            )

            assert samples is not None
        except ImportError:
            pytest.skip("getdist not installed")
