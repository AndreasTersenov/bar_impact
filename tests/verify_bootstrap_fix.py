#!/usr/bin/env python3
"""
Test to verify that bootstrap behavior with the seed fix works correctly.
This demonstrates the impact of the fix on bootstrap uncertainties.

Note: These tests are skipped if TARP is not installed.
"""

import os
import sys

import numpy as np
import pytest

# Check if TARP is available
try:
    # Add tarp package to path (for local installation)
    tarp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tarp", "src")
    if tarp_path not in sys.path:
        sys.path.insert(0, tarp_path)
    from tarp.drp import _get_tarp_coverage_bootstrap, _get_tarp_coverage_single

    HAS_TARP_INTERNAL = True
except ImportError:
    HAS_TARP_INTERNAL = False
    _get_tarp_coverage_bootstrap = None
    _get_tarp_coverage_single = None

pytestmark = pytest.mark.skipif(
    not HAS_TARP_INTERNAL, reason="TARP internal functions not available"
)


@pytest.fixture
def bootstrap_fix_test_data():
    """Generate test data for bootstrap fix verification."""
    np.random.seed(123)

    num_samples = 150
    num_sims = 80
    num_dims = 4

    # True parameter values
    theta = np.random.uniform(low=-5, high=5, size=(num_sims, num_dims))

    # Posterior samples
    log_sigma = np.random.uniform(low=-1.5, high=-0.5, size=(num_sims, num_dims))
    sigma = np.exp(log_sigma)
    samples = np.random.normal(
        loc=theta[np.newaxis, :, :],
        scale=sigma[np.newaxis, :, :],
        size=(num_samples, num_sims, num_dims),
    )

    return samples, theta


def test_bootstrap_with_fix_has_variation(bootstrap_fix_test_data):
    """Test that fixed bootstrap shows proper variation across iterations."""
    samples, theta = bootstrap_fix_test_data

    ecp_boot_fixed, alpha = _get_tarp_coverage_bootstrap(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        num_alpha_bins=None,
        num_bootstrap=30,
        norm=True,
        seed=42,
    )

    # Compute bootstrap standard deviation
    ecp_std_fixed = np.std(ecp_boot_fixed, axis=0)

    # With the fix, there should be meaningful variation
    mean_std = np.mean(ecp_std_fixed)
    assert mean_std > 0.001, f"Bootstrap std too small after fix: {mean_std:.6f}"

    # Also check variation across bootstrap samples at midpoint
    mid_idx = len(alpha) // 2
    variation_fixed = np.std(
        [ecp_boot_fixed[i, mid_idx] for i in range(len(ecp_boot_fixed))]
    )
    assert variation_fixed > 0.001, (
        f"No variation across bootstrap samples: {variation_fixed:.6f}"
    )


def test_bootstrap_different_samples_vary(bootstrap_fix_test_data):
    """Test that different bootstrap samples have different values."""
    samples, theta = bootstrap_fix_test_data

    ecp_boot, alpha = _get_tarp_coverage_bootstrap(
        samples=samples,
        theta=theta,
        references="random",
        metric="euclidean",
        num_alpha_bins=None,
        num_bootstrap=10,
        norm=True,
        seed=42,
    )

    # First 5 bootstrap samples at midpoint should not all be identical
    mid_idx = len(alpha) // 2
    midpoint_values = [ecp_boot[i, mid_idx] for i in range(min(5, len(ecp_boot)))]

    # Check that values are not all the same
    unique_values = len({round(v, 6) for v in midpoint_values})
    assert unique_values > 1, "All bootstrap samples at midpoint are identical"
