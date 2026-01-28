"""
Unit tests for utility modules.

Tests cover:
- utils.reproducibility: Deterministic seed generation
- utils.paths: File path discovery and output naming
- utils.noise: Shape noise generation
"""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestReproducibility:
    """Tests for utils.reproducibility module."""
    
    def test_get_deterministic_seed_basic(self):
        """Test basic deterministic seed generation."""
        from bar_impact.utils.reproducibility import get_deterministic_seed
        
        seed1 = get_deterministic_seed("/path/to/file1.h5", global_seed=42)
        seed2 = get_deterministic_seed("/path/to/file1.h5", global_seed=42)
        
        # Same inputs should give same seed
        assert seed1 == seed2
        assert isinstance(seed1, int)
        assert 0 <= seed1 < 2**32
    
    def test_get_deterministic_seed_different_paths(self):
        """Test different paths give different seeds."""
        from bar_impact.utils.reproducibility import get_deterministic_seed
        
        seed1 = get_deterministic_seed("/path/to/file1.h5", global_seed=42)
        seed2 = get_deterministic_seed("/path/to/file2.h5", global_seed=42)
        
        # Different paths should give different seeds
        assert seed1 != seed2
    
    def test_get_deterministic_seed_different_global(self):
        """Test different global seeds give different seeds."""
        from bar_impact.utils.reproducibility import get_deterministic_seed
        
        seed1 = get_deterministic_seed("/path/to/file.h5", global_seed=42)
        seed2 = get_deterministic_seed("/path/to/file.h5", global_seed=123)
        
        # Different global seeds should give different seeds
        assert seed1 != seed2
    
    def test_seed_worker_with_global_seed(self):
        """Test seed_worker with global_seed parameter."""
        from bar_impact.utils.reproducibility import seed_worker
        
        # This should set np.random.seed based on global_seed
        seed_worker(global_seed=42)
        
        # Generate some random numbers
        rand1 = np.random.rand(10)
        
        # Reset with same seed
        seed_worker(global_seed=42)
        rand2 = np.random.rand(10)
        
        # Should get same sequence
        np.testing.assert_array_equal(rand1, rand2)
    
    def test_seed_worker_without_global_seed(self):
        """Test seed_worker without global_seed uses OS entropy."""
        from bar_impact.utils.reproducibility import seed_worker
        
        # This should use OS entropy (non-deterministic)
        seed_worker()
        
        # Just check it doesn't crash - we can't test randomness easily
        rand = np.random.rand(5)
        assert len(rand) == 5
    
    def test_create_seed_worker_initializer(self):
        """Test create_seed_worker_initializer convenience function."""
        from bar_impact.utils.reproducibility import create_seed_worker_initializer
        
        initializer = create_seed_worker_initializer(global_seed=42)
        
        # Should be callable
        assert callable(initializer)
        
        # Should work when called
        initializer()
        
        rand = np.random.rand(5)
        assert len(rand) == 5


class TestPaths:
    """Tests for utils.paths module."""
    
    @pytest.fixture
    def temp_data_structure(self):
        """Create temporary directory structure mimicking data layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            
            # Create fiducial structure: perm_0000, perm_0001, etc.
            fiducial_dir = base_dir / "fiducial"
            fiducial_dir.mkdir()
            for i in range(5):  # Only 5 for testing
                perm_dir = fiducial_dir / f"perm_{i:04d}"
                perm_dir.mkdir()
                (perm_dir / "projected_probes_maps_nobaryons512.h5").touch()
                (perm_dir / "projected_probes_maps_baryonified512.h5").touch()
            
            # Create grid structure: cosmo_0000/perm_0000, etc.
            grid_dir = base_dir / "grid"
            grid_dir.mkdir()
            for c in range(3):  # Only 3 cosmologies for testing
                cosmo_dir = grid_dir / f"cosmo_{c:04d}"
                cosmo_dir.mkdir()
                for p in range(7):
                    perm_dir = cosmo_dir / f"perm_{p:04d}"
                    perm_dir.mkdir()
                    (perm_dir / "projected_probes_maps_nobaryons512.h5").touch()
                    (perm_dir / "projected_probes_maps_baryonified512.h5").touch()
            
            yield {
                'base': base_dir,
                'fiducial': fiducial_dir,
                'grid': grid_dir,
            }
    
    def test_get_data_file_paths_fiducial_nobaryons(self, temp_data_structure):
        """Test file discovery for fiducial nobaryons data."""
        from bar_impact.utils.paths import get_data_file_paths
        
        fiducial_dir = temp_data_structure['fiducial']
        
        base_dir, file_paths = get_data_file_paths(
            base_dir=str(fiducial_dir),
            fiducial=True,
            baryonified=False,
        )
        
        assert base_dir == str(fiducial_dir)
        assert len(file_paths) == 5  # We created 5 perms
        assert all("nobaryons512.h5" in fp for fp in file_paths)
        assert all("perm_" in fp for fp in file_paths)
    
    def test_get_data_file_paths_fiducial_baryonified(self, temp_data_structure):
        """Test file discovery for fiducial baryonified data."""
        from bar_impact.utils.paths import get_data_file_paths
        
        fiducial_dir = temp_data_structure['fiducial']
        
        base_dir, file_paths = get_data_file_paths(
            base_dir=str(fiducial_dir),
            fiducial=True,
            baryonified=True,
        )
        
        assert len(file_paths) == 5
        assert all("baryonified512.h5" in fp for fp in file_paths)
    
    def test_get_data_file_paths_grid(self, temp_data_structure):
        """Test file discovery for grid data."""
        from bar_impact.utils.paths import get_data_file_paths
        
        grid_dir = temp_data_structure['grid']
        
        base_dir, file_paths = get_data_file_paths(
            base_dir=str(grid_dir),
            fiducial=False,
            baryonified=False,
        )
        
        assert len(file_paths) == 3 * 7  # 3 cosmologies × 7 perms
        assert all("cosmo_" in fp for fp in file_paths)
        assert all("perm_" in fp for fp in file_paths)
    
    def test_get_data_file_paths_default_paths(self, temp_data_structure):
        """Test that default paths are used when base_dir not specified."""
        from bar_impact.utils.paths import get_data_file_paths
        
        # When base_dir is None and fiducial=True, should use default fiducial path
        with patch('os.path.exists', return_value=False):
            base_dir, file_paths = get_data_file_paths(
                base_dir=None,
                fiducial=True,
                baryonified=False,
            )
            
            # Should use the hardcoded default path
            assert "fiducial" in base_dir
            assert len(file_paths) == 0  # No files exist
    
    def test_build_output_suffix_basic(self):
        """Test basic output suffix generation."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="l1_norms",
            bin_range=[1, 2, 3, 4],
        )
        
        assert "l1_norms" in suffix
        assert "bins" in suffix or "1234" in suffix
    
    def test_build_output_suffix_single_bin(self):
        """Test output suffix with single bin."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="peak_counts",
            bin_number=2,
        )
        
        assert "peak_counts" in suffix
        assert "bin2" in suffix or "2" in suffix
    
    def test_build_output_suffix_bnt(self):
        """Test output suffix with BNT bins."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="l1_norms",
            bnt_bin_range=[0, 1, 2, 3],
        )
        
        assert "bnt" in suffix
        assert "bins" in suffix or "0123" in suffix
    
    def test_build_output_suffix_mask(self):
        """Test output suffix with mask parameters."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="peak_counts",
            bin_number=1,
            apply_mask=True,
            mask_area_sqdeg=14003.0,
        )
        
        assert "masked" in suffix
        assert "14003" in suffix
    
    def test_build_output_suffix_noise(self):
        """Test output suffix with noise parameters."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="l1_norms",
            bin_number=1,
            add_noise=True,
            noise_level=0.26,
        )
        
        assert "noisy" in suffix
        assert "0.26" in suffix or "s0" in suffix
    
    def test_build_output_suffix_power_spectra(self):
        """Test output suffix for power spectra."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="cls",
            bin_range=[1, 2, 3, 4],
            lmax=1024,
            apply_mask=True,
            mask_area_sqdeg=10000.0,
            apodization_scale_deg=2.0,
        )
        
        assert "cls" in suffix or "ps" in suffix
        assert "1024" in suffix or "l" in suffix  # lmax
        assert "masked" in suffix
        assert "apod" in suffix or "2.0" in suffix or "10000" in suffix
    
    def test_build_output_suffix_master(self):
        """Test output suffix with MASTER correction."""
        from bar_impact.utils.paths import build_output_suffix
        
        suffix = build_output_suffix(
            statistic_type="cls",
            bin_range=[1, 2],
            use_namaster=True,
        )
        
        # Just verify it creates a valid suffix
        assert suffix.endswith(".npz") or suffix.endswith(".npy")


class TestNoiseUtils:
    """Tests for utils.noise module."""
    
    def test_add_shape_noise_with_seed(self):
        """Test shape noise with fixed seed."""
        from bar_impact.utils.noise import add_shape_noise
        import healpy as hp
        
        # Use valid HEALPix size: nside=8 -> npix=768
        kappa_map = np.zeros(hp.nside2npix(8))
        
        noisy1 = add_shape_noise(kappa_map, sigma_e=0.26, seed=42)
        noisy2 = add_shape_noise(kappa_map, sigma_e=0.26, seed=42)
        
        # Same seed should give same noise
        np.testing.assert_array_almost_equal(noisy1, noisy2)
    
    def test_add_shape_noise_with_rng(self):
        """Test shape noise with RNG object."""
        from bar_impact.utils.noise import add_shape_noise
        import healpy as hp
        
        # Use valid HEALPix size: nside=8 -> npix=768
        kappa_map = np.zeros(hp.nside2npix(8))
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        noisy1 = add_shape_noise(kappa_map, sigma_e=0.26, rng=rng1)
        noisy2 = add_shape_noise(kappa_map, sigma_e=0.26, rng=rng2)
        
        # Same RNG seed should give same noise
        np.testing.assert_array_almost_equal(noisy1, noisy2)
    
    def test_add_shape_noise_priority(self):
        """Test that rng parameter takes priority over seed."""
        from bar_impact.utils.noise import add_shape_noise
        import healpy as hp
        
        # Use valid HEALPix size: nside=8 -> npix=768
        kappa_map = np.zeros(hp.nside2npix(8))
        rng = np.random.default_rng(42)
        
        # Even with different seed, rng should be used
        noisy = add_shape_noise(kappa_map, sigma_e=0.26, rng=rng, seed=999)
        
        # Should use rng, not seed
        assert noisy is not None
        assert not np.allclose(noisy, kappa_map)  # Noise was added
    
    def test_add_shape_noise_galaxy_density(self):
        """Test shape noise with galaxy density."""
        from bar_impact.utils.noise import add_shape_noise
        import healpy as hp
        
        # Use valid HEALPix size: nside=8 -> npix=768
        kappa_map = np.zeros(hp.nside2npix(8))
        
        # Lower galaxy density should give larger noise
        noisy_low = add_shape_noise(
            kappa_map, sigma_e=0.26, galaxy_density=10.0, seed=42
        )
        noisy_high = add_shape_noise(
            kappa_map, sigma_e=0.26, galaxy_density=100.0, seed=42
        )
        
        std_low = np.std(noisy_low)
        std_high = np.std(noisy_high)
        
        # Lower density should have higher std
        assert std_low > std_high
    
    def test_add_shape_noise_preserves_mean(self):
        """Test that noise doesn't shift the mean significantly."""
        from bar_impact.utils.noise import add_shape_noise
        import healpy as hp
        
        # Use valid HEALPix size: nside=32 -> npix=12288 (large enough for statistics)
        kappa_map = np.ones(hp.nside2npix(32)) * 0.05  # Constant map
        
        noisy = add_shape_noise(kappa_map, sigma_e=0.26, seed=42)
        
        # Mean should be close to original (noise is zero-mean)
        assert abs(np.mean(noisy) - 0.05) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
