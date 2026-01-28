"""
Pytest configuration and fixtures for BAR_IMPACT tests.

This module provides:
- Common fixtures for test data
- Skip markers for optional dependencies (pycs, pymaster, jax, jaxili, tarp)
- Test collection configuration
"""

import pytest
import numpy as np
import healpy as hp
import tempfile
from pathlib import Path


# =============================================================================
# Optional Dependency Detection
# =============================================================================

def _check_pycs_available():
    """Check if pycs (CosmoStat) is available and working."""
    try:
        from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere, get_wtpeaks_sphere
        return True
    except (ImportError, NameError, AttributeError):
        # NameError and AttributeError can occur if pycs has internal issues
        return False
    except Exception:
        # Catch any other unexpected errors during import
        return False


def _check_pymaster_available():
    """Check if pymaster (NaMaster) is available."""
    try:
        import pymaster as nmt
        return True
    except ImportError:
        return False


def _check_jax_available():
    """Check if JAX is available."""
    try:
        import jax
        import jax.numpy as jnp
        return True
    except ImportError:
        return False


def _check_jaxili_available():
    """Check if jaxili is available."""
    try:
        from jaxili.inference import NPE
        return True
    except ImportError:
        return False


def _check_tarp_available():
    """Check if TARP is available."""
    try:
        from tarp import get_tarp_coverage
        return True
    except ImportError:
        return False


def _check_getdist_available():
    """Check if getdist is available."""
    try:
        from getdist import plots
        return True
    except ImportError:
        return False


# Availability flags
HAS_PYCS = _check_pycs_available()
HAS_NAMASTER = _check_pymaster_available()
HAS_JAX = _check_jax_available()
HAS_JAXILI = _check_jaxili_available()
HAS_TARP = _check_tarp_available()
HAS_GETDIST = _check_getdist_available()


# =============================================================================
# Skip Markers
# =============================================================================

requires_pycs = pytest.mark.skipif(
    not HAS_PYCS,
    reason="pycs (CosmoStat) not installed"
)

requires_namaster = pytest.mark.skipif(
    not HAS_NAMASTER,
    reason="pymaster (NaMaster) not installed"
)

requires_jax = pytest.mark.skipif(
    not HAS_JAX,
    reason="JAX not installed"
)

requires_jaxili = pytest.mark.skipif(
    not HAS_JAXILI,
    reason="jaxili not installed"
)

requires_tarp = pytest.mark.skipif(
    not HAS_TARP,
    reason="TARP not installed"
)

requires_getdist = pytest.mark.skipif(
    not HAS_GETDIST,
    reason="getdist not installed"
)


# =============================================================================
# Common Fixtures
# =============================================================================

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


# =============================================================================
# Fixtures for Optional Dependencies
# =============================================================================

@pytest.fixture
def jax_random_key():
    """Get a JAX random key (skipped if JAX not available)."""
    if not HAS_JAX:
        pytest.skip("JAX not available")
    import jax
    return jax.random.PRNGKey(42)


@pytest.fixture
def jnp_array():
    """Get jax.numpy for array operations (skipped if JAX not available)."""
    if not HAS_JAX:
        pytest.skip("JAX not available")
    import jax.numpy as jnp
    return jnp


# =============================================================================
# Test Collection Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_pycs: mark test as requiring pycs library"
    )
    config.addinivalue_line(
        "markers", "requires_namaster: mark test as requiring pymaster library"
    )
    config.addinivalue_line(
        "markers", "requires_jax: mark test as requiring JAX"
    )
    config.addinivalue_line(
        "markers", "requires_jaxili: mark test as requiring jaxili"
    )
    config.addinivalue_line(
        "markers", "requires_tarp: mark test as requiring TARP"
    )
    config.addinivalue_line(
        "markers", "requires_getdist: mark test as requiring getdist"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
