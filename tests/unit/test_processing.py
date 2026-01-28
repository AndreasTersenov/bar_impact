"""
Unit tests for the processing module.

Tests cover:
- PowerSpectrumProcessor
- L1NormProcessor (with pycs mock)
- PeakCountProcessor (with pycs mock)
- Base processor functionality
"""

import numpy as np
import pytest
import healpy as hp
from unittest.mock import patch, MagicMock


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""
    
    def test_default_values(self):
        """Test ProcessingConfig has sensible defaults."""
        from bar_impact.processing.base import ProcessingConfig
        
        config = ProcessingConfig()
        # Default is add_noise=True (standard for cosmological analysis)
        assert config.add_noise is True
        assert config.noise_level == 0.26
        assert config.apply_mask is False
    
    def test_custom_values(self):
        """Test ProcessingConfig with custom values."""
        from bar_impact.processing.base import ProcessingConfig
        
        config = ProcessingConfig(
            add_noise=True,
            noise_level=0.30,
            apply_mask=True,
            mask_area_sqdeg=10000.0,
        )
        assert config.add_noise is True
        assert config.noise_level == 0.30
        assert config.apply_mask is True
        assert config.mask_area_sqdeg == 10000.0


class TestPowerSpectrumProcessor:
    """Tests for PowerSpectrumProcessor."""
    
    @pytest.fixture
    def simple_map(self):
        """Create a simple test map."""
        nside = 64  # Small for fast testing
        npix = hp.nside2npix(nside)
        return np.random.randn(npix) * 0.01
    
    @pytest.fixture
    def processor(self):
        """Create a default processor."""
        from bar_impact.processing import PowerSpectrumProcessor
        return PowerSpectrumProcessor(lmax=100)
    
    def test_init_default(self):
        """Test processor initialization with defaults."""
        from bar_impact.processing import PowerSpectrumProcessor
        from bar_impact.constants import DEFAULT_LMAX
        
        processor = PowerSpectrumProcessor()
        assert processor.lmax == DEFAULT_LMAX
        assert processor.ell_min is None
        assert processor.ell_max is None
    
    def test_init_custom_lmax(self):
        """Test processor initialization with custom lmax."""
        from bar_impact.processing import PowerSpectrumProcessor
        
        processor = PowerSpectrumProcessor(lmax=512)
        assert processor.lmax == 512
    
    def test_init_with_ell_range(self):
        """Test processor initialization with ell range."""
        from bar_impact.processing import PowerSpectrumProcessor
        
        processor = PowerSpectrumProcessor(lmax=1024, ell_min=100, ell_max=500)
        assert processor.lmax == 1024
        assert processor.ell_min == 100
        assert processor.ell_max == 500
    
    def test_process_single(self, simple_map, processor):
        """Test processing a single map."""
        cls = processor.process_single(simple_map)
        
        assert isinstance(cls, np.ndarray)
        assert len(cls) == 101  # lmax + 1
        assert np.all(np.isfinite(cls))
    
    def test_process_single_with_return_ell(self, simple_map, processor):
        """Test processing with return_ell=True."""
        cls, ell = processor.process_single(simple_map, return_ell=True)
        
        assert len(cls) == len(ell)
        assert ell[0] == 0
        assert ell[-1] == 100
    
    def test_process_single_with_ell_selection(self, simple_map):
        """Test processing with ell range selection."""
        from bar_impact.processing import PowerSpectrumProcessor
        
        processor = PowerSpectrumProcessor(lmax=100, ell_min=20, ell_max=80)
        cls = processor.process_single(simple_map)
        
        assert len(cls) == 61  # 80 - 20 + 1
    
    def test_process_cross(self, simple_map):
        """Test cross power spectrum computation."""
        from bar_impact.processing import PowerSpectrumProcessor
        
        processor = PowerSpectrumProcessor(lmax=100)
        map2 = np.random.randn(len(simple_map)) * 0.01
        
        cls_cross = processor.process_cross(simple_map, map2)
        
        assert isinstance(cls_cross, np.ndarray)
        assert len(cls_cross) == 101
    
    def test_process_all_cross_spectra(self, simple_map):
        """Test computing all cross spectra."""
        from bar_impact.processing import PowerSpectrumProcessor
        
        processor = PowerSpectrumProcessor(lmax=100)
        maps = [simple_map, np.random.randn(len(simple_map)) * 0.01]
        
        cls_dict = processor.process_all_cross_spectra(maps)
        
        # Should have 2 auto + 1 cross = 3 spectra
        assert len(cls_dict) == 3
        assert (0, 0) in cls_dict
        assert (1, 1) in cls_dict
        assert (0, 1) in cls_dict
    
    def test_statistic_type(self, processor):
        """Test statistic type attribute."""
        assert processor.statistic_type == "power_spectrum"


class TestComputeFunctions:
    """Tests for standalone compute functions."""
    
    @pytest.fixture
    def simple_map(self):
        """Create a simple test map."""
        nside = 64
        npix = hp.nside2npix(nside)
        return np.random.randn(npix) * 0.01
    
    def test_compute_power_spectrum(self, simple_map):
        """Test compute_power_spectrum function."""
        from bar_impact.processing import compute_power_spectrum
        
        cls = compute_power_spectrum(simple_map, lmax=100)
        
        assert isinstance(cls, np.ndarray)
        assert len(cls) == 101
        assert np.all(np.isfinite(cls))
    
    def test_compute_cross_power_spectrum(self, simple_map):
        """Test compute_cross_power_spectrum function."""
        from bar_impact.processing import compute_cross_power_spectrum
        
        map2 = np.random.randn(len(simple_map)) * 0.01
        cls = compute_cross_power_spectrum(simple_map, map2, lmax=100)
        
        assert isinstance(cls, np.ndarray)
        assert len(cls) == 101


class TestL1NormProcessor:
    """Tests for L1NormProcessor."""
    
    @pytest.fixture
    def simple_map(self):
        """Create a simple test map."""
        nside = 64
        npix = hp.nside2npix(nside)
        return np.random.randn(npix) * 0.01
    
    def test_init_default(self):
        """Test processor initialization with defaults."""
        from bar_impact.processing import L1NormProcessor
        
        processor = L1NormProcessor()
        assert processor.nscales == 5
        assert processor.nbins == 40
    
    def test_init_custom(self):
        """Test processor initialization with custom values."""
        from bar_impact.processing import L1NormProcessor
        
        processor = L1NormProcessor(nscales=4, nbins=30)
        assert processor.nscales == 4
        assert processor.nbins == 30
    
    def test_statistic_type(self):
        """Test statistic type attribute."""
        from bar_impact.processing import L1NormProcessor
        
        processor = L1NormProcessor()
        assert processor.statistic_type == "l1_norm"
    
    def test_get_output_shape(self):
        """Test get_output_shape method."""
        from bar_impact.processing import L1NormProcessor
        
        processor = L1NormProcessor(nscales=5, nbins=40)
        assert processor.get_output_shape() == (200,)
    
    def test_pycs_availability_flag(self):
        """Test pycs availability is checked."""
        from bar_impact.processing import L1NormProcessor
        
        processor = L1NormProcessor()
        # pycs_available should be a boolean
        assert isinstance(processor.pycs_available, bool)
    
    @patch('bar_impact.processing.l1_norms._check_pycs_available', return_value=False)
    def test_process_single_without_pycs(self, mock_check, simple_map):
        """Test that process_single raises ImportError without pycs."""
        from bar_impact.processing import L1NormProcessor
        
        processor = L1NormProcessor()
        processor.pycs_available = False
        
        with pytest.raises(ImportError, match="pycs library is required"):
            processor.process_single(simple_map)
    
    @patch('bar_impact.processing.l1_norms.compute_l1_norms')
    def test_process_single_with_mock_pycs(self, mock_compute, simple_map):
        """Test process_single with mocked pycs."""
        from bar_impact.processing import L1NormProcessor
        
        # Set up mock return value
        mock_compute.return_value = np.random.randn(200)
        
        processor = L1NormProcessor(nscales=5, nbins=40)
        processor.pycs_available = True
        
        result = processor.process_single(simple_map)
        
        mock_compute.assert_called_once()
        assert result.shape == (200,)


class TestPeakCountProcessor:
    """Tests for PeakCountProcessor."""
    
    @pytest.fixture
    def simple_map(self):
        """Create a simple test map."""
        nside = 64
        npix = hp.nside2npix(nside)
        return np.random.randn(npix) * 0.01
    
    def test_init_default(self):
        """Test processor initialization with defaults."""
        from bar_impact.processing import PeakCountProcessor
        
        processor = PeakCountProcessor()
        assert processor.nscales == 5
        assert processor.nbins == 40
        assert processor.min_val == -4.0
        assert processor.max_val == 4.0
    
    def test_init_custom(self):
        """Test processor initialization with custom values."""
        from bar_impact.processing import PeakCountProcessor
        
        processor = PeakCountProcessor(nscales=4, nbins=30, min_val=-3.0, max_val=3.0)
        assert processor.nscales == 4
        assert processor.nbins == 30
        assert processor.min_val == -3.0
        assert processor.max_val == 3.0
    
    def test_statistic_type(self):
        """Test statistic type attribute."""
        from bar_impact.processing import PeakCountProcessor
        
        processor = PeakCountProcessor()
        assert processor.statistic_type == "peak_counts"
    
    def test_get_output_shape(self):
        """Test get_output_shape method."""
        from bar_impact.processing import PeakCountProcessor
        
        processor = PeakCountProcessor(nscales=5, nbins=40)
        assert processor.get_output_shape() == (200,)
    
    def test_pycs_availability_flag(self):
        """Test pycs availability is checked."""
        from bar_impact.processing import PeakCountProcessor
        
        processor = PeakCountProcessor()
        assert isinstance(processor.pycs_available, bool)
    
    @patch('bar_impact.processing.peak_counts._check_pycs_available', return_value=False)
    def test_process_single_without_pycs(self, mock_check, simple_map):
        """Test that process_single raises ImportError without pycs."""
        from bar_impact.processing import PeakCountProcessor
        
        processor = PeakCountProcessor()
        processor.pycs_available = False
        
        with pytest.raises(ImportError, match="pycs library is required"):
            processor.process_single(simple_map)


class TestIdentifyPeaks:
    """Tests for the identify_peaks function."""
    
    def test_identify_peaks_simple(self):
        """Test peak identification on a simple map."""
        from bar_impact.processing.peak_counts import identify_peaks
        
        # Create a map with a known peak
        nside = 8
        npix = hp.nside2npix(nside)
        map_data = np.zeros(npix)
        
        # Set one pixel high (center of disk should be a peak)
        map_data[100] = 1.0
        
        peaks = identify_peaks(map_data, threshold=0.5)
        
        # Should find at least one peak
        assert len(peaks) > 0
        # The peak we set should be found
        peak_indices = [p[0] for p in peaks]
        assert 100 in peak_indices
    
    def test_identify_peaks_with_threshold(self):
        """Test peak identification with threshold."""
        from bar_impact.processing.peak_counts import identify_peaks
        
        nside = 8
        npix = hp.nside2npix(nside)
        map_data = np.zeros(npix)
        map_data[100] = 0.5
        
        # High threshold should find no peaks
        peaks_high = identify_peaks(map_data, threshold=1.0)
        assert len(peaks_high) == 0
        
        # Low threshold should find the peak
        peaks_low = identify_peaks(map_data, threshold=0.1)
        peak_indices = [p[0] for p in peaks_low]
        assert 100 in peak_indices


class TestPowerSpectrumConfig:
    """Tests for PowerSpectrumConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from bar_impact.processing.power_spectrum import PowerSpectrumConfig
        from bar_impact.constants import DEFAULT_LMAX
        
        config = PowerSpectrumConfig()
        assert config.lmax == DEFAULT_LMAX
        assert config.ell_min is None
        assert config.ell_max is None
        assert config.binning is False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from bar_impact.processing.power_spectrum import PowerSpectrumConfig
        
        config = PowerSpectrumConfig(
            lmax=512,
            ell_min=100,
            ell_max=400,
            binning=True,
            bin_width=20,
        )
        assert config.lmax == 512
        assert config.ell_min == 100
        assert config.ell_max == 400
        assert config.binning is True
        assert config.bin_width == 20


class TestL1NormConfig:
    """Tests for L1NormConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from bar_impact.processing.l1_norms import L1NormConfig
        
        config = L1NormConfig()
        assert config.nscales == 5
        assert config.nbins == 40
        assert config.noise_std is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from bar_impact.processing.l1_norms import L1NormConfig
        
        config = L1NormConfig(
            nscales=4,
            nbins=30,
            noise_std=0.01,
            min_snr=-3.0,
            max_snr=3.0,
        )
        assert config.nscales == 4
        assert config.nbins == 30
        assert config.noise_std == 0.01
        assert config.min_snr == -3.0
        assert config.max_snr == 3.0


class TestPeakCountConfig:
    """Tests for PeakCountConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from bar_impact.processing.peak_counts import PeakCountConfig
        
        config = PeakCountConfig()
        assert config.nscales == 5
        assert config.nbins == 40
        assert config.min_val == -4.0
        assert config.max_val == 4.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from bar_impact.processing.peak_counts import PeakCountConfig
        
        config = PeakCountConfig(
            nscales=3,
            nbins=20,
            min_val=-5.0,
            max_val=5.0,
        )
        assert config.nscales == 3
        assert config.nbins == 20
        assert config.min_val == -5.0
        assert config.max_val == 5.0


class TestProcessorIntegration:
    """Integration tests for processors."""
    
    @pytest.fixture
    def realistic_map(self):
        """Create a more realistic test map."""
        nside = 128
        lmax = 256
        
        # Create a map with realistic power spectrum
        ell = np.arange(lmax + 1)
        cl = np.zeros(lmax + 1)
        # Simple power law
        cl[2:] = 1e-8 * (ell[2:] / 100.0) ** (-2)
        
        map_data = hp.synfast(cl, nside, lmax=lmax)
        return map_data
    
    def test_power_spectrum_consistent(self, realistic_map):
        """Test that power spectrum is consistent with input."""
        from bar_impact.processing import PowerSpectrumProcessor, compute_power_spectrum
        
        processor = PowerSpectrumProcessor(lmax=256)
        
        # Both methods should give same result
        cls1 = processor.process_single(realistic_map)
        cls2 = compute_power_spectrum(realistic_map, lmax=256)
        
        np.testing.assert_array_almost_equal(cls1, cls2)
    
    def test_processor_output_suffix(self):
        """Test output suffix generation."""
        from bar_impact.processing import PowerSpectrumProcessor
        from bar_impact.processing.base import ProcessingConfig
        
        config = ProcessingConfig(add_noise=True, noise_level=0.26)
        processor = PowerSpectrumProcessor(config=config, lmax=1024)
        
        suffix = processor.get_output_suffix(bin_number=2)
        assert "_cls" in suffix
        assert "_bin2" in suffix
        assert "_noisy_s0.26" in suffix
        assert ".npy" in suffix


class TestModuleImports:
    """Test that all module imports work correctly."""
    
    def test_import_from_processing(self):
        """Test imports from processing module."""
        from bar_impact.processing import (
            BaseProcessor,
            ProcessingConfig,
            PowerSpectrumProcessor,
            PowerSpectrumConfig,
            L1NormProcessor,
            L1NormConfig,
            PeakCountProcessor,
            PeakCountConfig,
            compute_power_spectrum,
            compute_cross_power_spectrum,
            compute_l1_norms,
            compute_peak_counts,
            apply_bnt_transform,
        )
        
        # All should be importable
        assert BaseProcessor is not None
        assert PowerSpectrumProcessor is not None
        assert L1NormProcessor is not None
        assert PeakCountProcessor is not None
    
    def test_import_from_main_package(self):
        """Test imports from main bar_impact package."""
        from bar_impact import (
            PowerSpectrumProcessor,
            L1NormProcessor,
            PeakCountProcessor,
            compute_power_spectrum,
        )
        
        assert PowerSpectrumProcessor is not None
        assert L1NormProcessor is not None
        assert PeakCountProcessor is not None
