"""
Pytest configuration and fixtures for BAR_IMPACT tests.
"""

import pytest
import numpy as np
import healpy as hp
import tempfile
from pathlib import Path


@pytest.fixture
def small_nside():
    """Small nside for fast tests."""
    return 64


@pytest.fixture
def sample_map_array(small_nside):
    """Generate a sample convergence map array."""
    npix = hp.nside2npix(small_nside)
    return np.random.randn(npix) * 0.01


@pytest.fixture
def sample_multi_bin_array(small_nside):
    """Generate sample maps for 4 redshift bins."""
    npix = hp.nside2npix(small_nside)
    return np.random.randn(4, npix) * 0.01


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_cosmo_params():
    """Generate sample cosmological parameters."""
    return np.array([0.3, 0.8, -1.0, 70.0, 0.96, 0.05])


@pytest.fixture
def sample_param_names():
    """Standard cosmological parameter names."""
    return ["Omega_m", "S_8", "w_0", "H_0", "n_s", "Omega_b"]
