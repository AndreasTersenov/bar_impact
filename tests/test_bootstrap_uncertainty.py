#!/usr/bin/env python3
"""
Test to verify that TARP bootstrap uncertainties are reasonable.
This test compares bootstrap variance to ensure it's in expected range.

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
def bootstrap_test_data():
    """Generate test data for bootstrap uncertainty testing."""
    np.random.seed(42)

    num_samples = 200
    num_sims = 100
    num_dims = 5

    # True parameter values
    theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))

    # Posterior samples (Gaussian around true values with varying uncertainty)
    log_sigma = np.random.uniform(low=-2, high=-0.5, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    samples = np.random.normal(
        loc=theta[np.newaxis, :, :],
        scale=sigma[np.newaxis, :, :],
        size=(num_samples, num_sims, num_dims)
    )

    return samples, theta


def test_bootstrap_shape(bootstrap_test_data):
    """Test that bootstrap returns correct shape."""
    samples, theta = bootstrap_test_data

    ecp_boot, alpha = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=50,
        seed=42
    )

    # Bootstrap should return 2D array with num_bootstrap rows
    assert ecp_boot.ndim == 2
    assert ecp_boot.shape[0] == 50


def test_bootstrap_uncertainty_reasonable(bootstrap_test_data):
    """Test that bootstrap uncertainties are non-trivial."""
    samples, theta = bootstrap_test_data

    ecp_boot, alpha = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=50,
        seed=42
    )

    # Compute bootstrap standard deviation
    ecp_std = np.std(ecp_boot, axis=0)

    # Mean uncertainty should be reasonable (not zero, not huge)
    mean_std = np.mean(ecp_std)
    assert mean_std > 0.001, f"Bootstrap std too small: {mean_std:.6f}"
    assert mean_std < 0.5, f"Bootstrap std unexpectedly large: {mean_std:.4f}"


def test_bootstrap_variation(bootstrap_test_data):
    """Test that bootstrap samples show meaningful variation."""
    samples, theta = bootstrap_test_data

    ecp_boot, alpha = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=50,
        seed=42
    )

    # Check variation across bootstrap samples
    variation = np.max(ecp_boot, axis=0) - np.min(ecp_boot, axis=0)
    mean_variation = np.mean(variation)

    # Should have some meaningful variation
    assert mean_variation > 0.01, f"Bootstrap variation too small: {mean_variation:.4f}"


def test_bootstrap_reproducibility(bootstrap_test_data):
    """Test that bootstrap results are reproducible with same seed."""
    samples, theta = bootstrap_test_data

    ecp_boot1, alpha1 = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=20,
        seed=123
    )

    ecp_boot2, alpha2 = get_tarp_coverage(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=True,
        num_bootstrap=20,
        seed=123
    )

    # Same seed should give same results
    np.testing.assert_array_almost_equal(ecp_boot1, ecp_boot2)
    np.testing.assert_array_almost_equal(alpha1, alpha2)
