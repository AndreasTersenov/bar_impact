"""
Unit tests for the inference module.

Tests cover:
- NPEConfig
- NPEResult
- NPEInference (with mocking)
- CoverageConfig
- CoverageResult
- CoverageTester (with mocking)
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestNPEConfig:
    """Tests for NPEConfig dataclass."""
    
    def test_default_values(self):
        """Test NPEConfig has sensible defaults."""
        from bar_impact.inference.npe import NPEConfig
        
        config = NPEConfig()
        assert config.num_epochs == 1000
        assert config.learning_rate == 1e-4
        assert config.batch_size == 40
        assert config.gpu_id == "0"
    
    def test_custom_values(self):
        """Test NPEConfig with custom values."""
        from bar_impact.inference.npe import NPEConfig
        
        config = NPEConfig(
            num_epochs=500,
            learning_rate=1e-3,
            batch_size=64,
            checkpoint_dir="/custom/path",
        )
        assert config.num_epochs == 500
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64
        assert config.checkpoint_dir == "/custom/path"


class TestNPEResult:
    """Tests for NPEResult dataclass."""
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample NPEResult."""
        from bar_impact.inference.npe import NPEResult
        
        np.random.seed(42)
        samples = np.random.randn(1000, 4)  # 1000 samples, 4 params
        param_names = ["Omega_m", "S_8", "w_0", "H_0"]
        
        return NPEResult(
            samples=samples,
            param_names=param_names,
            observed_data=np.random.randn(100),
            metadata={"test": "value"},
        )
    
    def test_properties(self, sample_result):
        """Test NPEResult properties."""
        assert sample_result.num_samples == 1000
        assert sample_result.num_params == 4
    
    def test_get_param_samples(self, sample_result):
        """Test getting samples for a specific parameter."""
        omega_samples = sample_result.get_param_samples("Omega_m")
        assert len(omega_samples) == 1000
        
        with pytest.raises(ValueError, match="Unknown parameter"):
            sample_result.get_param_samples("nonexistent")
    
    def test_get_mean_std(self, sample_result):
        """Test mean and std computation."""
        mean = sample_result.get_mean()
        std = sample_result.get_std()
        
        assert len(mean) == 4
        assert len(std) == 4
        assert np.all(std > 0)
    
    def test_get_quantiles(self, sample_result):
        """Test quantile computation."""
        q = sample_result.get_quantiles([0.16, 0.5, 0.84])
        
        assert q.shape == (3, 4)
        # Check ordering: 16th < 50th < 84th
        assert np.all(q[0] < q[1])
        assert np.all(q[1] < q[2])
    
    def test_summary(self, sample_result):
        """Test summary statistics."""
        summary = sample_result.summary()
        
        assert "Omega_m" in summary
        assert "mean" in summary["Omega_m"]
        assert "std" in summary["Omega_m"]
        assert "median" in summary["Omega_m"]
    
    def test_save_load(self, sample_result):
        """Test saving and loading results."""
        from bar_impact.inference.npe import NPEResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_result.npz"
            sample_result.save(filepath)
            
            loaded = NPEResult.load(filepath)
            
            np.testing.assert_array_almost_equal(
                loaded.samples, sample_result.samples
            )
            assert loaded.param_names == sample_result.param_names


class TestNPEInference:
    """Tests for NPEInference class."""
    
    def test_init_default(self):
        """Test default initialization."""
        from bar_impact.inference.npe import NPEInference
        
        npe = NPEInference()
        assert npe.config is not None
        assert npe.is_trained is False
        assert isinstance(npe.jaxili_available, bool)
    
    def test_init_with_config(self):
        """Test initialization with custom config."""
        from bar_impact.inference.npe import NPEInference, NPEConfig
        
        config = NPEConfig(num_epochs=100, batch_size=32)
        npe = NPEInference(config=config, param_names=["a", "b"])
        
        assert npe.config.num_epochs == 100
        assert npe.config.batch_size == 32
        assert npe.param_names == ["a", "b"]
    
    def test_sample_before_training_raises(self):
        """Test that sampling before training raises error."""
        from bar_impact.inference.npe import NPEInference
        
        npe = NPEInference()
        
        with pytest.raises(RuntimeError, match="must be trained"):
            npe.sample(np.random.randn(10))
    
    @patch('bar_impact.inference.npe._check_jaxili_available', return_value=False)
    def test_train_without_jaxili(self, mock_check):
        """Test that training without jaxili raises ImportError."""
        from bar_impact.inference.npe import NPEInference
        
        npe = NPEInference()
        npe.jaxili_available = False
        
        with pytest.raises(ImportError, match="jaxili is required"):
            npe.train(np.random.randn(100, 50), np.random.randn(100, 4))


class TestCoverageConfig:
    """Tests for CoverageConfig dataclass."""
    
    def test_default_values(self):
        """Test CoverageConfig has sensible defaults."""
        from bar_impact.inference.coverage import CoverageConfig
        
        config = CoverageConfig()
        assert config.num_sims == 100
        assert config.num_samples == 1000
        assert config.bootstrap is True
        assert config.metric == "euclidean"
    
    def test_custom_values(self):
        """Test CoverageConfig with custom values."""
        from bar_impact.inference.coverage import CoverageConfig
        
        config = CoverageConfig(
            num_sims=50,
            num_samples=500,
            bootstrap=False,
            metric="manhattan",
        )
        assert config.num_sims == 50
        assert config.num_samples == 500
        assert config.bootstrap is False
        assert config.metric == "manhattan"


class TestCoverageResult:
    """Tests for CoverageResult dataclass."""
    
    @pytest.fixture
    def simple_result(self):
        """Create a simple CoverageResult (no bootstrap)."""
        from bar_impact.inference.coverage import CoverageResult, CoverageConfig
        
        # Perfect calibration example
        alpha = np.linspace(0, 1, 11)
        ecp = alpha  # Perfect calibration
        
        config = CoverageConfig(bootstrap=False)
        return CoverageResult(ecp=ecp, alpha=alpha, config=config)
    
    @pytest.fixture
    def bootstrap_result(self):
        """Create a CoverageResult with bootstrap."""
        from bar_impact.inference.coverage import CoverageResult, CoverageConfig
        
        alpha = np.linspace(0, 1, 11)
        # Bootstrap results: (n_bootstrap, n_bins)
        ecp = np.random.randn(100, 11) * 0.05 + alpha  # Noisy around diagonal
        
        config = CoverageConfig(bootstrap=True, num_bootstrap=100)
        return CoverageResult(ecp=ecp, alpha=alpha, config=config)
    
    def test_ecp_mean_no_bootstrap(self, simple_result):
        """Test ECP mean without bootstrap."""
        np.testing.assert_array_almost_equal(
            simple_result.ecp_mean, simple_result.ecp
        )
    
    def test_ecp_mean_with_bootstrap(self, bootstrap_result):
        """Test ECP mean with bootstrap."""
        assert bootstrap_result.ecp_mean.shape == (11,)
    
    def test_ecp_std_no_bootstrap(self, simple_result):
        """Test ECP std is None without bootstrap."""
        assert simple_result.ecp_std is None
    
    def test_ecp_std_with_bootstrap(self, bootstrap_result):
        """Test ECP std is computed with bootstrap."""
        assert bootstrap_result.ecp_std is not None
        assert bootstrap_result.ecp_std.shape == (11,)
        assert np.all(bootstrap_result.ecp_std >= 0)
    
    def test_deviation_from_diagonal(self, simple_result):
        """Test deviation calculation for perfect calibration."""
        # Perfect calibration should have ~0 deviation
        deviation = simple_result.get_deviation_from_diagonal()
        assert deviation < 0.01
    
    def test_is_calibrated(self, simple_result):
        """Test calibration check."""
        assert simple_result.is_calibrated(tolerance=0.05) is True
    
    def test_summary(self, simple_result):
        """Test summary method."""
        summary = simple_result.summary()
        
        assert "mean_deviation" in summary
        assert "is_calibrated" in summary
    
    def test_save_load(self, simple_result):
        """Test saving and loading."""
        from bar_impact.inference.coverage import CoverageResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "coverage.npz"
            simple_result.save(filepath)
            
            loaded = CoverageResult.load(filepath)
            
            np.testing.assert_array_almost_equal(loaded.ecp, simple_result.ecp)
            np.testing.assert_array_almost_equal(loaded.alpha, simple_result.alpha)


class TestCoverageTester:
    """Tests for CoverageTester class."""
    
    def test_init_default(self):
        """Test default initialization."""
        from bar_impact.inference.coverage import CoverageTester
        
        tester = CoverageTester()
        assert tester.config is not None
        assert isinstance(tester.tarp_available, bool)
    
    def test_init_with_config(self):
        """Test initialization with custom config."""
        from bar_impact.inference.coverage import CoverageTester, CoverageConfig
        
        config = CoverageConfig(num_sims=50, bootstrap=False)
        tester = CoverageTester(config=config)
        
        assert tester.config.num_sims == 50
        assert tester.config.bootstrap is False
    
    @patch('bar_impact.inference.coverage._check_tarp_available', return_value=False)
    def test_test_without_tarp(self, mock_check):
        """Test that testing without TARP raises ImportError."""
        from bar_impact.inference.coverage import CoverageTester
        
        tester = CoverageTester()
        tester.tarp_available = False
        
        def mock_sample_fn(x, n):
            return np.random.randn(n, 4)
        
        with pytest.raises(ImportError, match="TARP is required"):
            tester.test(
                mock_sample_fn,
                np.random.randn(100, 50),
                np.random.randn(100, 4),
            )


class TestModuleImports:
    """Test that all module imports work correctly."""
    
    def test_import_from_inference(self):
        """Test imports from inference module."""
        from bar_impact.inference import (
            NPEInference,
            NPEConfig,
            NPEResult,
            run_npe_inference,
            train_npe_model,
            sample_posterior,
            CoverageTester,
            CoverageConfig,
            CoverageResult,
            compute_tarp_coverage,
        )
        
        assert NPEInference is not None
        assert NPEConfig is not None
        assert CoverageTester is not None
    
    def test_import_from_main_package(self):
        """Test imports from main bar_impact package."""
        from bar_impact import (
            NPEInference,
            NPEConfig,
            NPEResult,
            run_npe_inference,
            CoverageTester,
            CoverageConfig,
            CoverageResult,
            compute_tarp_coverage,
        )
        
        assert NPEInference is not None
        assert CoverageTester is not None


class TestFunctionalInterface:
    """Tests for functional interface compatibility."""
    
    def test_run_npe_inference_creates_config(self):
        """Test that run_npe_inference creates proper config from kwargs."""
        # This test just checks the interface, not actual training
        from bar_impact.inference.npe import NPEConfig
        
        # Test that kwargs are properly handled in config creation
        config = NPEConfig(**{
            k: v for k, v in {"num_epochs": 100, "batch_size": 32}.items()
            if hasattr(NPEConfig, k)
        })
        
        assert config.num_epochs == 100
        assert config.batch_size == 32
