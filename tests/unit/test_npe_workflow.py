"""
Unit tests for utils.npe_workflow module.

Tests cover:
- NPE initialization
- Training/loading workflow
- Posterior sampling
- Triangle plot generation
- Standard cosmological parameter configuration

Note: These tests require JAX, jaxili, and getdist, and are skipped if any are not available.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import availability flags from conftest
from tests.conftest import HAS_GETDIST, HAS_JAX

# Check if all required dependencies are available for npe_workflow
HAS_NPE_WORKFLOW_DEPS = HAS_JAX and HAS_GETDIST

# Skip marker for tests requiring npe_workflow dependencies
requires_npe_workflow = pytest.mark.skipif(
    not HAS_NPE_WORKFLOW_DEPS, reason="JAX and getdist required for npe_workflow tests"
)

# Conditionally import jax.numpy
if HAS_JAX:
    import jax.numpy as jnp
else:
    jnp = None  # Will be skipped in tests


@requires_npe_workflow
class TestNPEInitialization:
    """Tests for NPE initialization."""

    def test_initialize_npe_basic(self):
        """Test basic NPE initialization."""
        from bar_impact.utils.npe_workflow import initialize_npe

        # Create mock data
        params = jnp.array(np.random.randn(100, 6).astype(np.float32))
        data = jnp.array(np.random.randn(100, 120).astype(np.float32))

        # Initialize NPE
        with patch("bar_impact.utils.npe_workflow.NPE") as mock_npe_class:
            mock_inference = MagicMock()
            mock_npe_class.return_value = mock_inference
            mock_inference.append_simulations.return_value = mock_inference

            result = initialize_npe(params, data)

            # Verify NPE was created and simulations appended
            mock_npe_class.assert_called_once()
            mock_inference.append_simulations.assert_called_once()
            assert result is mock_inference

    def test_initialize_npe_with_correct_shapes(self):
        """Test that NPE initialization passes correct shapes."""
        from bar_impact.utils.npe_workflow import initialize_npe

        n_sims, n_params, n_features = 50, 6, 80
        params = jnp.array(np.random.randn(n_sims, n_params).astype(np.float32))
        data = jnp.array(np.random.randn(n_sims, n_features).astype(np.float32))

        with patch("bar_impact.utils.npe_workflow.NPE") as mock_npe_class:
            mock_inference = MagicMock()
            mock_npe_class.return_value = mock_inference
            mock_inference.append_simulations.return_value = mock_inference

            initialize_npe(params, data)

            # Check that append_simulations was called with correct arguments
            call_args = mock_inference.append_simulations.call_args
            assert call_args[0][0].shape == (n_sims, n_params)
            assert call_args[0][1].shape == (n_sims, n_features)


@requires_npe_workflow
class TestTrainOrLoadNPE:
    """Tests for train_or_load_npe workflow function."""

    def test_train_new_model_basic(self):
        """Test training a new model."""
        from bar_impact.utils.npe_workflow import train_or_load_npe

        mock_inference = MagicMock()
        checkpoint_path = "/tmp/test_checkpoint"

        train_params = {
            "num_epochs": 10,
            "learning_rate": 1e-4,
            "batch_size": 32,
        }

        mock_metrics = MagicMock()
        mock_density = MagicMock()
        mock_inference.train.return_value = (mock_metrics, mock_density)

        result = train_or_load_npe(
            inference=mock_inference,
            checkpoint_path=checkpoint_path,
            should_train=True,
            train_params=train_params,
        )

        assert len(result) == 3
        assert result[0] is mock_inference
        assert result[1] is mock_metrics
        assert result[2] is mock_density
        mock_inference.train.assert_called_once()

    def test_train_with_nan_retry(self):
        """Test training with NaN retry mechanism."""
        from bar_impact.utils.npe_workflow import train_or_load_npe

        mock_inference = MagicMock()
        params = jnp.array(np.random.randn(50, 6).astype(np.float32))
        data = jnp.array(np.random.randn(50, 80).astype(np.float32))

        train_params = {
            "params": params,
            "data": data,
            "num_epochs": 10,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "max_retries": 5,
        }

        with patch(
            "bar_impact.utils.npe_workflow.train_npe_with_nan_retry"
        ) as mock_train:
            mock_train.return_value = (mock_inference, MagicMock(), MagicMock())

            train_or_load_npe(
                inference=mock_inference,
                checkpoint_path="/tmp/test",
                should_train=True,
                train_params=train_params,
            )

            # Verify NaN retry was used
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs["max_retries"] == 5

    def test_load_existing_model(self):
        """Test loading an existing model."""
        from bar_impact.utils.npe_workflow import train_or_load_npe

        mock_inference = MagicMock()
        mock_density = MagicMock()
        mock_inference.load.return_value = mock_density

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint")
            os.makedirs(checkpoint_path)

            result = train_or_load_npe(
                inference=mock_inference,
                checkpoint_path=checkpoint_path,
                should_train=False,
                train_params=None,
            )

            assert len(result) == 3
            assert result[0] is mock_inference
            assert result[1] is None  # No metrics when loading
            assert result[2] is mock_density
            mock_inference.load.assert_called_once_with(checkpoint_path)

    def test_load_missing_checkpoint_raises_error(self):
        """Test that loading from missing checkpoint raises error."""
        from bar_impact.utils.npe_workflow import train_or_load_npe

        mock_inference = MagicMock()
        nonexistent_path = "/nonexistent/path/checkpoint"

        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            train_or_load_npe(
                inference=mock_inference,
                checkpoint_path=nonexistent_path,
                should_train=False,
                train_params=None,
            )

    def test_train_without_params_raises_error(self):
        """Test that training without train_params raises error."""
        from bar_impact.utils.npe_workflow import train_or_load_npe

        mock_inference = MagicMock()

        with pytest.raises(ValueError, match="train_params required"):
            train_or_load_npe(
                inference=mock_inference,
                checkpoint_path="/tmp/test",
                should_train=True,
                train_params=None,
            )


@requires_npe_workflow
class TestTrianglePlot:
    """Tests for triangle plot generation."""

    def test_create_triangle_plot_basic(self, tmp_path):
        """Test basic triangle plot creation."""
        from bar_impact.utils.npe_workflow import create_triangle_plot

        # Mock samples
        n_samples, n_params = 1000, 6
        samples = np.random.randn(n_samples, n_params)

        output_path = tmp_path / "test_triangle.pdf"

        with patch("bar_impact.utils.npe_workflow.plots") as mock_plots, patch(
            "bar_impact.utils.npe_workflow.plt"
        ) as mock_plt:
            mock_plotter = MagicMock()
            mock_plots.get_subplot_plotter.return_value = mock_plotter

            result = create_triangle_plot(
                samples=samples,
                sample_label="Test Sample",
                output_path=str(output_path),
                color="blue",
            )

            assert result == str(output_path)
            mock_plots.get_subplot_plotter.assert_called_once()
            mock_plotter.triangle_plot.assert_called_once()
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()

    def test_create_triangle_plot_with_fiducial(self, tmp_path):
        """Test triangle plot with fiducial values."""
        from bar_impact.utils.npe_workflow import create_triangle_plot

        samples = np.random.randn(500, 6)

        param_config = {
            "labels": [
                r"$\Omega_m$",
                r"$S_8$",
                r"$w_0$",
                r"$H_0$",
                r"$n_s$",
                r"$\Omega_b$",
            ],
            "fiducial_values": np.array([[0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493]]),
        }

        output_path = tmp_path / "test_with_fiducial.pdf"

        with patch("bar_impact.utils.npe_workflow.plots") as mock_plots, patch(
            "bar_impact.utils.npe_workflow.plt"
        ):
            mock_plotter = MagicMock()
            mock_plots.get_subplot_plotter.return_value = mock_plotter

            create_triangle_plot(
                samples=samples,
                sample_label="Test",
                output_path=str(output_path),
                param_config=param_config,
            )

            # Verify markers were added to plot
            call_kwargs = mock_plotter.triangle_plot.call_args[1]
            assert "markers" in call_kwargs

    def test_create_triangle_plot_creates_directory(self, tmp_path):
        """Test that triangle plot creates output directory."""
        from bar_impact.utils.npe_workflow import create_triangle_plot

        samples = np.random.randn(500, 6)
        nested_dir = tmp_path / "nested" / "output" / "dir"
        output_path = nested_dir / "plot.pdf"

        with patch("bar_impact.utils.npe_workflow.plots"), patch(
            "bar_impact.utils.npe_workflow.plt"
        ):
            create_triangle_plot(
                samples=samples, sample_label="Test", output_path=str(output_path)
            )

            # Directory should be created
            assert nested_dir.exists()


@requires_npe_workflow
class TestSampleAndSavePosterior:
    """Tests for sample_and_save_posterior workflow function."""

    def test_sample_and_save_basic(self, tmp_path):
        """Test basic posterior sampling and saving."""
        from bar_impact.utils.npe_workflow import sample_and_save_posterior

        mock_posterior = MagicMock()
        mock_samples = np.random.randn(1000, 6)
        mock_posterior.sample.return_value = mock_samples

        observation = np.random.randn(120)

        output_config = {
            "samples_dir": str(tmp_path / "samples"),
            "output_dir": str(tmp_path / "plots"),
            "base_filename": "test_run",
            "num_samples": 1000,
            "random_seed": 42,
            "sample_label": "Test Sample",
            "color": "blue",
        }

        with patch(
            "bar_impact.utils.npe_workflow.create_triangle_plot"
        ) as mock_triangle:
            mock_triangle.return_value = str(tmp_path / "plots" / "test_run.pdf")

            samples_path, plot_path = sample_and_save_posterior(
                posterior=mock_posterior,
                observation=observation,
                output_config=output_config,
            )

            # Verify sampling was called
            mock_posterior.sample.assert_called_once()

            # Verify directories were created
            assert Path(samples_path).exists()
            assert samples_path.endswith(".npy")

            # Verify triangle plot was created
            mock_triangle.assert_called_once()

    def test_sample_and_save_with_custom_param_config(self, tmp_path):
        """Test sampling with custom parameter configuration."""
        from bar_impact.utils.npe_workflow import sample_and_save_posterior

        mock_posterior = MagicMock()
        mock_samples = np.random.randn(500, 3)
        mock_posterior.sample.return_value = mock_samples

        observation = np.random.randn(80)

        custom_params = {
            "names": ["param1", "param2", "param3"],
            "labels": [r"$p_1$", r"$p_2$", r"$p_3$"],
        }

        output_config = {
            "samples_dir": str(tmp_path / "samples"),
            "output_dir": str(tmp_path / "plots"),
            "base_filename": "custom_run",
            "num_samples": 500,
            "random_seed": 123,
            "sample_label": "Custom",
            "param_config": custom_params,
            "color": "red",
        }

        with patch(
            "bar_impact.utils.npe_workflow.create_triangle_plot"
        ) as mock_triangle:
            mock_triangle.return_value = str(tmp_path / "plots" / "custom_run.pdf")

            sample_and_save_posterior(
                posterior=mock_posterior,
                observation=observation,
                output_config=output_config,
            )

            # Verify custom config was passed to triangle plot
            call_kwargs = mock_triangle.call_args[1]
            assert call_kwargs["param_config"] == custom_params


@requires_npe_workflow
class TestStandardCosmologyConfig:
    """Tests for standard cosmological parameter configuration."""

    def test_standard_cosmo_params_exists(self):
        """Test that STANDARD_COSMO_PARAMS is defined."""
        from bar_impact.utils.npe_workflow import STANDARD_COSMO_PARAMS

        assert "names" in STANDARD_COSMO_PARAMS
        assert "labels" in STANDARD_COSMO_PARAMS
        assert "fiducial_values" in STANDARD_COSMO_PARAMS

    def test_standard_cosmo_params_structure(self):
        """Test structure of STANDARD_COSMO_PARAMS."""
        from bar_impact.utils.npe_workflow import STANDARD_COSMO_PARAMS

        # Check expected 6 cosmological parameters
        assert len(STANDARD_COSMO_PARAMS["names"]) == 6
        assert len(STANDARD_COSMO_PARAMS["labels"]) == 6

        # Check fiducial values shape
        fiducial = STANDARD_COSMO_PARAMS["fiducial_values"]
        assert fiducial.shape == (1, 6)

    def test_standard_cosmo_params_values(self):
        """Test that standard cosmology has reasonable values."""
        from bar_impact.utils.npe_workflow import STANDARD_COSMO_PARAMS

        expected_names = ["Omega_m", "S_8", "w_0", "H_0", "n_s", "Omega_b"]
        assert STANDARD_COSMO_PARAMS["names"] == expected_names

        # Check fiducial values are in reasonable ranges
        fiducial = np.array(STANDARD_COSMO_PARAMS["fiducial_values"][0])

        # Omega_m should be between 0 and 1
        assert 0.0 < fiducial[0] < 1.0

        # S_8 should be around 0.8
        assert 0.5 < fiducial[1] < 1.2

        # w_0 should be around -1
        assert -2.0 < fiducial[2] < 0.0

        # H_0 should be in reasonable range (km/s/Mpc)
        assert 50.0 < fiducial[3] < 100.0


@requires_npe_workflow
class TestPrintFunctions:
    """Tests for print helper functions."""

    def test_print_analysis_summary(self, capsys):
        """Test analysis summary printing."""
        from bar_impact.utils.npe_workflow import print_analysis_summary

        config = {
            "simulation_type": "baryonified",
            "fiducial_type": "nobaryons",
            "bin_desc": "bin2",
            "scale_desc": "scales123",
            "noisy": True,
            "noise_level": 0.26,
            "checkpoint_name": "test_checkpoint",
        }

        print_analysis_summary(config)

        captured = capsys.readouterr()
        assert "Configuration Summary" in captured.out
        assert "baryonified" in captured.out
        assert "nobaryons" in captured.out
        assert "bin2" in captured.out
        assert "scales123" in captured.out
        assert "0.26" in captured.out

    def test_print_completion_summary(self, capsys):
        """Test completion summary printing."""
        from bar_impact.utils.npe_workflow import print_completion_summary

        result_paths = {
            "checkpoint": "/path/to/checkpoint",
            "samples": "/path/to/samples.npy",
            "triangle_plot": "/path/to/plot.pdf",
        }

        print_completion_summary(result_paths, coverage_test=False)

        captured = capsys.readouterr()
        assert "Inference Complete" in captured.out
        assert "/path/to/checkpoint" in captured.out
        assert "/path/to/samples.npy" in captured.out
        assert "/path/to/plot.pdf" in captured.out

    def test_print_completion_with_coverage(self, capsys):
        """Test completion summary with coverage test."""
        from bar_impact.utils.npe_workflow import print_completion_summary

        result_paths = {
            "checkpoint": "/path/to/checkpoint",
            "samples": "/path/to/samples.npy",
            "triangle_plot": "/path/to/plot.pdf",
            "coverage_plot": "/path/to/coverage.pdf",
        }

        print_completion_summary(result_paths, coverage_test=True)

        captured = capsys.readouterr()
        assert "coverage.pdf" in captured.out


@requires_npe_workflow
class TestSetupJAXEnvironment:
    """Tests for JAX environment setup."""

    def test_setup_jax_environment_gpu(self):
        """Test JAX setup with GPU."""
        from bar_impact.utils.npe_workflow import setup_jax_environment

        with patch("bar_impact.utils.npe_workflow.os.environ", {}) as mock_env, patch(
            "bar_impact.utils.npe_workflow.jax.devices"
        ) as mock_devices:
            mock_devices.return_value = ["GPU:0", "GPU:1"]

            setup_jax_environment(gpu_id="0,1", force_cpu=False)

            assert "CUDA_VISIBLE_DEVICES" in mock_env
            assert mock_env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_setup_jax_environment_force_cpu(self):
        """Test JAX setup forcing CPU."""
        from bar_impact.utils.npe_workflow import setup_jax_environment

        with patch("bar_impact.utils.npe_workflow.os.environ", {}) as mock_env, patch(
            "bar_impact.utils.npe_workflow.jax.config"
        ) as mock_config:
            setup_jax_environment(gpu_id="0", force_cpu=True)

            # Should set JAX to use CPU
            assert mock_config.update.called or "JAX_PLATFORMS" in mock_env
