"""
Unit tests for the core module.

Tests for ConvergenceMap, SurveyMask, and DataVector classes.
"""


import healpy as hp
import numpy as np
import pytest


class TestConvergenceMap:
    """Tests for the ConvergenceMap class."""

    @pytest.fixture
    def sample_map_data(self):
        """Create sample map data for testing."""
        nside = 64  # Small nside for fast tests
        npix = hp.nside2npix(nside)
        return np.random.randn(npix) * 0.01, nside

    def test_creation_basic(self, sample_map_data):
        """Test basic map creation."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data, nside=nside)

        assert kappa.nside == nside
        assert kappa.npix == hp.nside2npix(nside)
        assert not kappa.is_noisy
        assert not kappa.is_bnt_transformed

    def test_creation_infer_nside(self, sample_map_data):
        """Test that nside is inferred from data length."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data)  # No nside provided

        assert kappa.nside == nside

    def test_creation_wrong_size(self):
        """Test that wrong data size raises error."""
        from bar_impact.core import ConvergenceMap

        with pytest.raises(ValueError, match="does not match"):
            ConvergenceMap(data=np.zeros(100), nside=64)

    def test_add_shape_noise(self, sample_map_data):
        """Test shape noise addition."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data, nside=nside)

        # Add noise with fixed seed for reproducibility
        kappa_noisy = kappa.add_shape_noise(sigma_e=0.26, seed=42)

        assert kappa_noisy.is_noisy
        assert kappa_noisy.noise_level == 0.26
        assert not np.allclose(kappa.data, kappa_noisy.data)

        # Original should be unchanged
        assert not kappa.is_noisy

    def test_add_shape_noise_inplace(self, sample_map_data):
        """Test in-place noise addition."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data.copy(), nside=nside)
        original_data = kappa.data.copy()

        result = kappa.add_shape_noise(sigma_e=0.26, seed=42, inplace=True)

        assert result is kappa  # Same object
        assert kappa.is_noisy
        assert not np.allclose(original_data, kappa.data)

    def test_add_shape_noise_reproducible(self, sample_map_data):
        """Test that noise is reproducible with seed."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa1 = ConvergenceMap(data=data, nside=nside)
        kappa2 = ConvergenceMap(data=data, nside=nside)

        noisy1 = kappa1.add_shape_noise(sigma_e=0.26, seed=42)
        noisy2 = kappa2.add_shape_noise(sigma_e=0.26, seed=42)

        np.testing.assert_array_equal(noisy1.data, noisy2.data)

    def test_compute_power_spectrum(self, sample_map_data):
        """Test power spectrum computation."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data, nside=nside)

        lmax = 100
        cls = kappa.compute_power_spectrum(lmax=lmax)

        assert len(cls) == lmax + 1
        assert np.all(cls >= 0)  # Power spectrum should be non-negative

    def test_compute_power_spectrum_with_ell(self, sample_map_data):
        """Test power spectrum with ell values returned."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data, nside=nside)

        lmax = 100
        cls, ell = kappa.compute_power_spectrum(lmax=lmax, return_ell=True)

        assert len(cls) == len(ell)
        np.testing.assert_array_equal(ell, np.arange(lmax + 1))

    def test_copy(self, sample_map_data):
        """Test that copy creates independent object."""
        from bar_impact.core import ConvergenceMap

        data, nside = sample_map_data
        kappa = ConvergenceMap(data=data, nside=nside, bin_number=1)
        kappa_copy = kappa.copy()

        # Modify copy
        kappa_copy.data[0] = 999.0

        # Original should be unchanged
        assert kappa.data[0] != 999.0


class TestConvergenceMapCollection:
    """Tests for the ConvergenceMapCollection class."""

    @pytest.fixture
    def sample_collection(self):
        """Create sample map collection."""
        from bar_impact.core import ConvergenceMap, ConvergenceMapCollection

        nside = 64
        npix = hp.nside2npix(nside)
        maps = [
            ConvergenceMap(
                data=np.random.randn(npix) * 0.01, nside=nside, bin_number=i + 1
            )
            for i in range(4)
        ]
        return ConvergenceMapCollection(maps)

    def test_creation(self, sample_collection):
        """Test collection creation."""
        assert sample_collection.n_bins == 4
        assert sample_collection.nside == 64

    def test_to_array(self, sample_collection):
        """Test conversion to array."""
        arr = sample_collection.to_array()

        assert arr.shape == (4, hp.nside2npix(64))

    def test_apply_bnt_transform(self, sample_collection):
        """Test BNT transform application."""
        bnt_collection = sample_collection.apply_bnt_transform()

        assert bnt_collection.n_bins == 4
        assert all(m.is_bnt_transformed for m in bnt_collection.maps)

        # Data should be different after transform
        orig_arr = sample_collection.to_array()
        bnt_arr = bnt_collection.to_array()
        assert not np.allclose(orig_arr, bnt_arr)

    def test_apply_bnt_custom_matrix(self, sample_collection):
        """Test BNT with custom matrix."""
        # Identity matrix should leave data unchanged
        custom = np.eye(4)
        bnt_collection = sample_collection.apply_bnt_transform(bnt_matrix=custom)

        orig_arr = sample_collection.to_array()
        bnt_arr = bnt_collection.to_array()
        np.testing.assert_array_almost_equal(orig_arr, bnt_arr)


class TestSurveyMask:
    """Tests for the SurveyMask class."""

    def test_create_disk_mask(self):
        """Test disk mask creation."""
        from bar_impact.core import SurveyMask

        nside = 64
        mask = SurveyMask.create_disk_mask(nside=nside, target_area_sqdeg=14000.0)

        assert mask.nside == nside
        assert 0 < mask.f_sky < 1
        assert mask.is_binary

    def test_create_apodized_mask(self):
        """Test apodized mask creation."""
        from bar_impact.core import SurveyMask

        nside = 64
        mask = SurveyMask.create_apodized_disk_mask(
            nside=nside,
            target_area_sqdeg=14000.0,
            apodization_deg=5.0,  # Large apodization for test
        )

        assert mask.nside == nside
        assert mask.apodization_deg == 5.0
        # Apodized mask should not be strictly binary
        assert not mask.is_binary

    def test_full_sky_mask(self):
        """Test full sky mask."""
        from bar_impact.core import SurveyMask

        nside = 64
        mask = SurveyMask.full_sky(nside=nside)

        assert mask.f_sky == 1.0
        assert np.all(mask.data == 1.0)

    def test_mask_caching(self):
        """Test that masks are cached."""
        from bar_impact.core import SurveyMask, clear_mask_cache

        clear_mask_cache()

        mask1 = SurveyMask.create_disk_mask(nside=64, target_area_sqdeg=14000.0)
        mask2 = SurveyMask.create_disk_mask(nside=64, target_area_sqdeg=14000.0)

        # Should be the same object due to caching
        assert mask1 is mask2


class TestDataVector:
    """Tests for the DataVector class."""

    def test_creation(self):
        """Test data vector creation."""
        from bar_impact.core import DataVector

        data = np.array([1.0, 2.0, 3.0, 4.0])
        dv = DataVector(data=data, statistic_type="l1_norm")

        assert dv.n_features == 4
        assert dv.statistic_type == "l1_norm"

    def test_concatenate(self):
        """Test data vector concatenation."""
        from bar_impact.core import DataVector

        dv1 = DataVector(data=np.array([1, 2]), statistic_type="l1_norm")
        dv2 = DataVector(data=np.array([3, 4, 5]), statistic_type="power_spectrum")

        combined = dv1.concatenate(dv2)

        assert combined.n_features == 5
        np.testing.assert_array_equal(combined.data, [1, 2, 3, 4, 5])

    def test_save_load_npz(self, tmp_path):
        """Test saving and loading in npz format."""
        from bar_impact.core import DataVector

        data = np.array([1.0, 2.0, 3.0])
        params = np.array([0.3, 0.8])
        dv = DataVector(
            data=data,
            statistic_type="l1_norm",
            cosmology_params=params,
            metadata={"bin": 1},
        )

        filepath = tmp_path / "test.npz"
        dv.save(filepath)

        loaded = DataVector.load(filepath)

        np.testing.assert_array_equal(loaded.data, data)
        np.testing.assert_array_equal(loaded.cosmology_params, params)
        assert loaded.statistic_type == "l1_norm"


class TestDataVectorCollection:
    """Tests for the DataVectorCollection class."""

    def test_creation(self):
        """Test collection creation."""
        from bar_impact.core import DataVectorCollection

        data = np.random.randn(100, 20)
        params = np.random.randn(100, 6)

        collection = DataVectorCollection(
            data_vectors=data, parameters=params, statistic_type="l1_norm"
        )

        assert collection.n_simulations == 100
        assert collection.n_features == 20
        assert collection.n_params == 6

    def test_filter_zero_variance(self):
        """Test zero variance filtering."""
        from bar_impact.core import DataVectorCollection

        # Create data with one constant column
        data = np.random.randn(100, 20)
        data[:, 5] = 0.0  # Zero variance column
        params = np.random.randn(100, 6)

        collection = DataVectorCollection(
            data_vectors=data, parameters=params, statistic_type="l1_norm"
        )

        filtered, mask = collection.filter_zero_variance(verbose=False)

        assert filtered.n_features == 19
        assert not mask[5]

    def test_train_test_split(self):
        """Test train/test splitting."""
        from bar_impact.core import DataVectorCollection

        data = np.random.randn(100, 20)
        params = np.random.randn(100, 6)

        collection = DataVectorCollection(
            data_vectors=data, parameters=params, statistic_type="l1_norm"
        )

        train, test = collection.train_test_split(test_fraction=0.2, seed=42)

        assert train.n_simulations == 80
        assert test.n_simulations == 20


class TestConstants:
    """Tests for constants module."""

    def test_bnt_matrix_shape(self):
        """Test BNT matrix has correct shape."""
        from bar_impact.constants import BNT_MATRIX_DEFAULT

        assert BNT_MATRIX_DEFAULT.shape == (4, 4)

    def test_bnt_matrix_lower_triangular(self):
        """Test BNT matrix is lower triangular."""
        from bar_impact.constants import BNT_MATRIX_DEFAULT

        # Upper triangle (excluding diagonal) should be zero
        upper = np.triu(BNT_MATRIX_DEFAULT, k=1)
        assert np.allclose(upper, 0)

    def test_get_bnt_matrix_default(self):
        """Test get_bnt_matrix returns correct default."""
        from bar_impact.constants import BNT_MATRIX_DEFAULT, get_bnt_matrix

        matrix = get_bnt_matrix(n_bins=4)
        np.testing.assert_array_equal(matrix, BNT_MATRIX_DEFAULT)

    def test_get_bnt_matrix_custom(self):
        """Test get_bnt_matrix with custom matrix."""
        from bar_impact.constants import get_bnt_matrix

        custom = np.eye(4) * 2
        matrix = get_bnt_matrix(n_bins=4, custom_matrix=custom)
        np.testing.assert_array_equal(matrix, custom)

    def test_get_bnt_matrix_wrong_bins(self):
        """Test get_bnt_matrix raises for unsupported bins."""
        from bar_impact.constants import get_bnt_matrix

        with pytest.raises(ValueError, match="only available for n_bins=4"):
            get_bnt_matrix(n_bins=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
