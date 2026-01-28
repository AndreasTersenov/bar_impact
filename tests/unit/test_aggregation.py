"""
Unit tests for analysis.aggregation module - ResultsAggregator enhancements.

Tests cover:
- Scale selection methods
- Bin range selection
- Zero-variance filtering
- Per-bin operations
"""


import numpy as np


class TestResultsAggregatorScaleSelection:
    """Tests for scale selection methods."""

    def test_select_scales_basic(self):
        """Test basic scale selection."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create test data: 10 samples, 4 scales, 40 bins per scale
        n_samples, n_scales, nbins = 10, 4, 40
        data = np.random.randn(n_samples, n_scales, nbins)

        # Select scales 0 and 2
        selected = aggregator.select_scales(data, [0, 2], nbins_per_scale=40)

        # Should have 2 scales * 40 bins = 80 features
        assert selected.shape == (n_samples, 2 * nbins)

    def test_select_scales_flattened_input(self):
        """Test scale selection with already flattened input."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Flattened: 10 samples, 160 features (4 scales * 40 bins)
        n_samples = 10
        data = np.random.randn(n_samples, 160)

        # Should reshape and select
        selected = aggregator.select_scales(data, [1, 3], nbins_per_scale=40)

        assert selected.shape == (n_samples, 80)  # 2 scales * 40 bins

    def test_select_scales_single_scale(self):
        """Test selecting a single scale."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        data = np.random.randn(20, 5, 40)
        selected = aggregator.select_scales(data, [2], nbins_per_scale=40)

        assert selected.shape == (20, 40)

    def test_select_scales_all_scales(self):
        """Test selecting all scales."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        n_samples, n_scales, nbins = 15, 4, 40
        data = np.random.randn(n_samples, n_scales, nbins)

        selected = aggregator.select_scales(data, [0, 1, 2, 3], nbins_per_scale=40)

        assert selected.shape == (n_samples, n_scales * nbins)

    def test_select_scales_preserves_values(self):
        """Test that scale selection preserves correct values."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create identifiable data
        data = np.arange(400).reshape(10, 4, 10)  # 10 samples, 4 scales, 10 bins

        # Select scale 1
        selected = aggregator.select_scales(data, [1], nbins_per_scale=10)

        # First sample, scale 1 should be [10, 11, ..., 19]
        expected_first = np.arange(10, 20)
        np.testing.assert_array_equal(selected[0], expected_first)


class TestResultsAggregatorScalesPerBin:
    """Tests for per-bin scale selection."""

    def test_select_scales_per_bin_basic(self):
        """Test basic per-bin scale selection."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # 2 redshift bins, each with 4 scales, 40 bins per scale
        bin1 = np.random.randn(10, 4, 40)
        bin2 = np.random.randn(10, 4, 40)

        # Select different scales per bin
        scales_per_bin = [[0, 1, 2], [1, 2, 3]]

        result = aggregator.select_scales_per_bin(
            [bin1, bin2], scales_per_bin, nbins_per_scale=40
        )

        # Bin1: 3 scales, Bin2: 3 scales = 240 features total
        assert result.shape == (10, 240)

    def test_select_scales_per_bin_different_scales(self):
        """Test per-bin selection with very different scale choices."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        bin1 = np.random.randn(5, 4, 20)
        bin2 = np.random.randn(5, 4, 20)
        bin3 = np.random.randn(5, 4, 20)

        # Different number of scales per bin
        scales_per_bin = [[0], [0, 1, 2], [1, 2, 3]]  # 1 scale  # 3 scales  # 3 scales

        result = aggregator.select_scales_per_bin(
            [bin1, bin2, bin3], scales_per_bin, nbins_per_scale=20
        )

        # Total: (1 + 3 + 3) * 20 = 140 features
        assert result.shape == (5, 140)

    def test_select_scales_per_bin_flattened_input(self):
        """Test per-bin selection with flattened inputs."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Flattened data
        bin1 = np.random.randn(8, 160)  # 4 scales * 40 bins
        bin2 = np.random.randn(8, 160)

        scales_per_bin = [[0, 2], [1, 3]]

        result = aggregator.select_scales_per_bin(
            [bin1, bin2], scales_per_bin, nbins_per_scale=40
        )

        # 2 scales per bin * 2 bins * 40 bins = 160 features
        assert result.shape == (8, 160)

    def test_select_scales_per_bin_preserves_order(self):
        """Test that per-bin selection preserves concatenation order."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create identifiable data
        bin1 = np.ones((3, 2, 10)) * 100  # Values: 100
        bin2 = np.ones((3, 2, 10)) * 200  # Values: 200

        scales_per_bin = [[0], [1]]

        result = aggregator.select_scales_per_bin(
            [bin1, bin2], scales_per_bin, nbins_per_scale=10
        )

        # First 10 features should be from bin1, next 10 from bin2
        assert np.all(result[:, :10] == 100)
        assert np.all(result[:, 10:20] == 200)


class TestResultsAggregatorBinRangeSelection:
    """Tests for bin range selection."""

    def test_select_bin_range_basic(self):
        """Test basic bin range selection."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        data = np.random.randn(10, 100)

        # Select bins 20-29 (inclusive)
        selected = aggregator.select_bin_range(data, start_idx=20, end_idx=29)

        assert selected.shape == (10, 10)

    def test_select_bin_range_single_bin(self):
        """Test selecting a single bin."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        data = np.random.randn(15, 50)
        selected = aggregator.select_bin_range(data, start_idx=25, end_idx=25)

        assert selected.shape == (15, 1)

    def test_select_bin_range_preserves_values(self):
        """Test that bin range selection preserves correct values."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create data with known values
        data = np.arange(500).reshape(5, 100)

        # Select bins 10-14 from first sample
        selected = aggregator.select_bin_range(data, start_idx=10, end_idx=14)

        expected_first = np.arange(10, 15)
        np.testing.assert_array_equal(selected[0], expected_first)

    def test_select_bin_ranges_per_bin(self):
        """Test selecting different bin ranges for each redshift bin."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        bin1 = np.random.randn(10, 50)
        bin2 = np.random.randn(10, 60)
        bin3 = np.random.randn(10, 40)

        bin_ranges = [(10, 29), (20, 39), (5, 24)]  # Each selects 20 bins

        result = aggregator.select_bin_ranges_per_bin([bin1, bin2, bin3], bin_ranges)

        # Total: 20 + 20 + 20 = 60 features
        assert result.shape == (10, 60)

    def test_select_bin_ranges_per_bin_different_sizes(self):
        """Test per-bin ranges with different range sizes."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        bin1 = np.random.randn(8, 100)
        bin2 = np.random.randn(8, 100)

        bin_ranges = [(0, 9), (50, 79)]  # 10 bins, 30 bins

        result = aggregator.select_bin_ranges_per_bin([bin1, bin2], bin_ranges)

        assert result.shape == (8, 40)  # 10 + 30 = 40


class TestResultsAggregatorZeroVarianceFilter:
    """Tests for zero-variance filtering."""

    def test_filter_zero_variance_basic(self):
        """Test basic zero-variance filtering."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create data with some zero-variance features
        data = np.random.randn(20, 10)
        data[:, 3] = 5.0  # Zero variance
        data[:, 7] = 0.0  # Zero variance

        filtered = aggregator.filter_zero_variance(data, min_variance=1e-10)

        # Should remove 2 features
        assert filtered.shape == (20, 8)

    def test_filter_zero_variance_return_mask(self):
        """Test zero-variance filtering with mask return."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        data = np.random.randn(15, 8)
        data[:, 2] = 1.0
        data[:, 5] = 2.0

        filtered, mask = aggregator.filter_zero_variance(
            data, min_variance=1e-10, return_mask=True
        )

        assert filtered.shape == (15, 6)
        assert mask.shape == (8,)
        assert np.sum(mask) == 6
        assert not mask[2]
        assert not mask[5]

    def test_filter_zero_variance_threshold(self):
        """Test zero-variance filtering with custom threshold."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create data with very low but non-zero variance
        np.random.seed(42)  # For reproducibility
        data = np.random.randn(50, 5) * 10  # Ensure good variance
        data[:, 1] = np.random.randn(50) * 1e-11  # Very low variance

        # With high threshold, should filter the low-variance column
        filtered_high = aggregator.filter_zero_variance(data, min_variance=1e-9)
        assert filtered_high.shape[1] == 4

        # With very low threshold, should keep all
        filtered_low = aggregator.filter_zero_variance(data, min_variance=1e-25)
        assert filtered_low.shape[1] == 5

    def test_filter_zero_variance_no_filtering_needed(self):
        """Test when no filtering is needed."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # All features have good variance
        data = np.random.randn(10, 20)

        filtered = aggregator.filter_zero_variance(data)

        assert filtered.shape == data.shape

    def test_filter_zero_variance_all_filtered(self):
        """Test when all features would be filtered."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # All features have zero variance
        data = np.ones((10, 5))

        filtered = aggregator.filter_zero_variance(data)

        assert filtered.shape == (10, 0)

    def test_filter_zero_variance_preserves_data(self):
        """Test that filtering preserves non-filtered data correctly."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Create identifiable data
        data = np.arange(60).reshape(10, 6)
        data[:, 2] = 100  # Zero variance

        filtered, mask = aggregator.filter_zero_variance(data, return_mask=True)

        # Check that remaining columns match original
        original_valid_cols = data[:, mask]
        np.testing.assert_array_equal(filtered, original_valid_cols)


class TestResultsAggregatorIntegration:
    """Integration tests combining multiple aggregation operations."""

    def test_scale_selection_then_bin_range(self):
        """Test combining scale selection and bin range selection."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Start with multi-scale data
        data = np.random.randn(10, 4, 40)

        # Select scales 0 and 2
        data = aggregator.select_scales(data, [0, 2], nbins_per_scale=40)

        # Then select bin range 10:29
        data = aggregator.select_bin_range(data, start_idx=10, end_idx=29)

        # Should have 20 bins from 2 scales = 40 features
        assert data.shape == (10, 20)

    def test_per_bin_scales_then_filter_zeros(self):
        """Test per-bin scale selection followed by zero-variance filtering."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        bin1 = np.random.randn(15, 4, 30)
        bin2 = np.random.randn(15, 4, 30)

        # Select scales
        data = aggregator.select_scales_per_bin(
            [bin1, bin2], [[0, 1], [2, 3]], nbins_per_scale=30
        )

        # Add some zero-variance features
        data[:, 10] = 1.0
        data[:, 50] = 2.0

        # Filter zeros
        data = aggregator.filter_zero_variance(data)

        # Should have removed 2 features from 120
        assert data.shape == (15, 118)

    def test_full_pipeline(self):
        """Test complete aggregation pipeline."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Simulate 3 redshift bins
        bin1 = np.random.randn(20, 5, 40)
        bin2 = np.random.randn(20, 5, 40)
        bin3 = np.random.randn(20, 5, 40)

        # Step 1: Select scales per bin
        data_list = [bin1, bin2, bin3]
        scales_per_bin = [[1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        data = aggregator.select_scales_per_bin(
            data_list, scales_per_bin, nbins_per_scale=40
        )

        # Step 2: Add zero-variance features (simulating real data)
        data[:, 50] = 1.0
        data[:, 100] = 2.0

        # Step 3: Filter zero-variance
        data, mask = aggregator.filter_zero_variance(data, return_mask=True)

        # Verify final shape
        # Total scales: 3 + 4 + 4 = 11 scales
        # Total features before filtering: 11 * 40 = 440
        # After filtering: 440 - 2 = 438
        assert data.shape == (20, 438)
        assert np.sum(mask) == 438


class TestResultsAggregatorConfigVerbosity:
    """Test verbosity control in ResultsAggregator."""

    def test_verbose_false_suppresses_output(self, capsys):
        """Test that verbose=False suppresses output."""
        from bar_impact.analysis.aggregation import AggregationConfig, ResultsAggregator

        config = AggregationConfig(verbose=False)
        aggregator = ResultsAggregator(config=config)

        data = np.random.randn(10, 100)
        data[:, 5] = 1.0  # Zero variance

        aggregator.filter_zero_variance(data)

        captured = capsys.readouterr()
        assert "Filtered" not in captured.out

    def test_verbose_true_shows_output(self, capsys):
        """Test that verbose=True shows output."""
        from bar_impact.analysis.aggregation import AggregationConfig, ResultsAggregator

        config = AggregationConfig(verbose=True)
        aggregator = ResultsAggregator(config=config)

        data = np.random.randn(10, 100)
        data[:, 5] = 1.0
        data[:, 10] = 2.0

        aggregator.filter_zero_variance(data)

        captured = capsys.readouterr()
        assert "Filtered" in captured.out or "zero-variance" in captured.out.lower()


class TestResultsAggregatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_select_scales_empty_list(self):
        """Test scale selection with empty scale list."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        data = np.random.randn(10, 4, 40)
        selected = aggregator.select_scales(data, [], nbins_per_scale=40)

        # Should return empty feature dimension
        assert selected.shape == (10, 0)

    def test_select_bin_range_full_range(self):
        """Test selecting full range."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        data = np.random.randn(8, 50)
        selected = aggregator.select_bin_range(data, start_idx=0, end_idx=49)

        # Should be identical to input
        np.testing.assert_array_equal(selected, data)

    def test_filter_zero_variance_single_sample(self):
        """Test zero-variance filtering with single sample."""
        from bar_impact.analysis.aggregation import ResultsAggregator

        aggregator = ResultsAggregator()

        # Single sample - all features will have zero variance
        data = np.random.randn(1, 20)

        filtered = aggregator.filter_zero_variance(data)

        # All features should be filtered
        assert filtered.shape == (1, 0)
