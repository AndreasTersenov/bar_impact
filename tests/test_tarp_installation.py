#!/usr/bin/env python3
"""
Tests to verify TARP package is accessible and working.
This module runs minimal TARP coverage tests with synthetic data.

Note: These tests are skipped if TARP is not installed.
"""

import os
import sys
import numpy as np
import pytest

# Check if TARP is available
try:
    # Add tarp package to path (for local installation)
    tarp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tarp', 'src')
    if tarp_path not in sys.path:
        sys.path.insert(0, tarp_path)
    from tarp import get_tarp_coverage
    HAS_TARP = True
except ImportError:
    HAS_TARP = False
    get_tarp_coverage = None

pytestmark = pytest.mark.skipif(not HAS_TARP, reason="TARP not installed")


@pytest.fixture
def synthetic_tarp_data():
    """Generate synthetic data for TARP testing."""
    np.random.seed(42)

    num_samples = 100
    num_sims = 50
    num_dims = 3

    # True parameter values
    theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))

    # Posterior samples (Gaussian around true values)
    log_sigma = np.random.uniform(low=-2, high=-0.5, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    samples = np.random.normal(
        loc=theta[np.newaxis, :, :],
        scale=sigma[np.newaxis, :, :],
        size=(num_samples, num_sims, num_dims)
    )

    return samples, theta


def test_tarp_import():
    """Test that TARP can be imported."""
    assert HAS_TARP, "TARP should be available for this test"
    assert get_tarp_coverage is not None


def test_tarp_coverage_basic(synthetic_tarp_data):
    """Test basic TARP coverage computation."""
    samples, theta = synthetic_tarp_data

    ecp, alpha = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=False
    )

    # Check output shapes
    assert ecp.shape == alpha.shape
    assert len(ecp) > 0

    # Check values are in valid range
    assert np.all(ecp >= 0) and np.all(ecp <= 1)
    assert np.all(alpha >= 0) and np.all(alpha <= 1)


def test_tarp_coverage_calibration(synthetic_tarp_data):
    """Test that coverage deviation is reasonable."""
    samples, theta = synthetic_tarp_data

    ecp, alpha = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=False
    )

    # Check if coverage is reasonable (well-calibrated posteriors)
    mean_deviation = np.mean(np.abs(ecp - alpha))

    # For well-specified posteriors, deviation should be small
    # Note: With small test data, some deviation is expected
    assert mean_deviation < 0.5, f"Coverage deviation too large: {mean_deviation:.3f}"


def test_tarp_bootstrap(synthetic_tarp_data):
    """Test TARP bootstrap functionality."""
    samples, theta = synthetic_tarp_data

    ecp_boot, alpha_boot = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=10
    )

    # Bootstrap should return 2D arrays
    assert ecp_boot.ndim == 2
    assert ecp_boot.shape[0] == 10  # num_bootstrap


def test_tarp_different_metrics(synthetic_tarp_data):
    """Test TARP with different distance metrics."""
    samples, theta = synthetic_tarp_data

    for metric in ["euclidean"]:  # Can add more metrics if supported
        ecp, alpha = get_tarp_coverage(
            samples=samples,
            theta=theta,
            references="random",
            metric=metric,
            norm=True,
            bootstrap=False
        )

        assert len(ecp) > 0, f"Failed for metric: {metric}"
