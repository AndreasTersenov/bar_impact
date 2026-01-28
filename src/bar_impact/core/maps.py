"""
Convergence map data structure for BAR_IMPACT.

This module provides the ConvergenceMap class for representing and
manipulating HEALPix weak lensing convergence (kappa) maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import h5py
import healpy as hp
import numpy as np

from bar_impact.constants import (
    COSMOGRID_MAP_KEY_TEMPLATE,
    DEFAULT_GALAXY_DENSITY,
    DEFAULT_LMAX,
    DEFAULT_SIGMA_E,
    get_bnt_matrix,
)

if TYPE_CHECKING:
    from bar_impact.core.masks import SurveyMask


__all__ = ["ConvergenceMap", "ConvergenceMapCollection"]


@dataclass
class ConvergenceMap:
    """
    Representation of a HEALPix weak lensing convergence map.

    This class encapsulates a convergence (kappa) map along with its
    metadata, and provides methods for common operations like adding
    noise, applying masks, and computing summary statistics.

    Parameters
    ----------
    data : np.ndarray
        HEALPix map data array.
    nside : int, optional
        HEALPix resolution parameter. If not provided, will be inferred
        from the data array length.
    bin_number : int, optional
        Tomographic redshift bin number (1-indexed).
    simulation_type : str, optional
        Type of simulation: "baryonified" or "nobaryons".
    cosmology_id : str, optional
        Identifier for the cosmology (e.g., "cosmo_0001").
    permutation_id : str, optional
        Identifier for the realization (e.g., "perm_0001").
    is_noisy : bool
        Whether shape noise has been added.
    noise_level : float, optional
        Shape noise level (sigma_e) if noise was added.
    is_bnt_transformed : bool
        Whether BNT transform has been applied.

    Attributes
    ----------
    npix : int
        Number of pixels in the map.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(12 * 512**2) * 0.01
    >>> kappa = ConvergenceMap(data, nside=512, bin_number=1)
    >>> kappa.npix
    3145728
    >>> kappa.nside
    512

    >>> # Add shape noise
    >>> kappa_noisy = kappa.add_shape_noise(sigma_e=0.26)
    >>> kappa_noisy.is_noisy
    True
    """

    data: np.ndarray
    nside: int = field(default=None)
    bin_number: Optional[int] = None
    simulation_type: Optional[str] = None
    cosmology_id: Optional[str] = None
    permutation_id: Optional[str] = None
    is_noisy: bool = False
    noise_level: Optional[float] = None
    is_bnt_transformed: bool = False
    bnt_bin: Optional[int] = None

    def __post_init__(self):
        """Validate and set derived attributes."""
        self.data = np.asarray(self.data, dtype=np.float64)

        # Infer nside if not provided
        if self.nside is None:
            self.nside = hp.npix2nside(len(self.data))

        # Validate data length
        expected_npix = hp.nside2npix(self.nside)
        if len(self.data) != expected_npix:
            raise ValueError(
                f"Data length {len(self.data)} does not match nside={self.nside} "
                f"(expected {expected_npix} pixels)"
            )

    @property
    def npix(self) -> int:
        """Number of pixels in the map."""
        return len(self.data)

    @property
    def pixel_area_sr(self) -> float:
        """Pixel area in steradians."""
        return hp.nside2pixarea(self.nside)

    @property
    def pixel_area_arcmin2(self) -> float:
        """Pixel area in square arcminutes."""
        return hp.nside2pixarea(self.nside, degrees=True) * 3600

    @classmethod
    def from_h5(
        cls,
        filepath: Union[str, Path],
        bin_number: int,
        key_template: str = COSMOGRID_MAP_KEY_TEMPLATE,
        **kwargs,
    ) -> "ConvergenceMap":
        """
        Load a convergence map from a CosmoGRID HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.
        bin_number : int
            Tomographic bin number (1-indexed).
        key_template : str, optional
            HDF5 key template with {bin_number} placeholder.
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        ConvergenceMap
            Loaded convergence map.

        Examples
        --------
        >>> kappa = ConvergenceMap.from_h5(
        ...     "projected_probes_maps_baryonified512.h5",
        ...     bin_number=2
        ... )
        """
        filepath = Path(filepath)
        map_key = key_template.format(bin_number=bin_number)

        with h5py.File(filepath, "r") as f:
            if map_key not in f:
                available = list(f.keys())
                raise KeyError(
                    f"Key '{map_key}' not found in {filepath}. "
                    f"Available keys: {available}"
                )
            data = np.array(f[map_key])

        # Try to infer simulation type from filename
        simulation_type = kwargs.pop("simulation_type", None)
        if simulation_type is None:
            if "baryonified" in filepath.name:
                simulation_type = "baryonified"
            elif "nobaryons" in filepath.name:
                simulation_type = "nobaryons"

        return cls(
            data=data,
            bin_number=bin_number,
            simulation_type=simulation_type,
            **kwargs,
        )

    @classmethod
    def from_fits(
        cls,
        filepath: Union[str, Path],
        field: int = 0,
        **kwargs,
    ) -> "ConvergenceMap":
        """
        Load a convergence map from a FITS file.

        Parameters
        ----------
        filepath : str or Path
            Path to the FITS file.
        field : int, optional
            Field index to read (default: 0).
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        ConvergenceMap
            Loaded convergence map.
        """
        filepath = Path(filepath)
        data = hp.read_map(str(filepath), field=field, verbose=False)
        return cls(data=data, **kwargs)

    def copy(self) -> "ConvergenceMap":
        """Create a deep copy of this map."""
        return ConvergenceMap(
            data=self.data.copy(),
            nside=self.nside,
            bin_number=self.bin_number,
            simulation_type=self.simulation_type,
            cosmology_id=self.cosmology_id,
            permutation_id=self.permutation_id,
            is_noisy=self.is_noisy,
            noise_level=self.noise_level,
            is_bnt_transformed=self.is_bnt_transformed,
            bnt_bin=self.bnt_bin,
        )

    def add_shape_noise(
        self,
        sigma_e: float = DEFAULT_SIGMA_E,
        galaxy_density: float = DEFAULT_GALAXY_DENSITY,
        seed: Optional[int] = None,
        inplace: bool = False,
    ) -> "ConvergenceMap":
        """
        Add shape noise to the convergence map.

        Parameters
        ----------
        sigma_e : float, optional
            Intrinsic ellipticity dispersion (default: 0.26).
        galaxy_density : float, optional
            Galaxy number density in arcmin^-2 (default: 6.75).
        seed : int, optional
            Random seed for reproducibility.
        inplace : bool, optional
            If True, modify this map in place. Otherwise return a new map.

        Returns
        -------
        ConvergenceMap
            Map with added shape noise (self if inplace=True).

        Notes
        -----
        The shape noise variance per pixel is:

        .. math::
            \\sigma_{\\rm pix}^2 = \\frac{\\sigma_e^2}{n_{\\rm gal} \\cdot A_{\\rm pix}}

        where :math:`n_{\\rm gal}` is the galaxy density and
        :math:`A_{\\rm pix}` is the pixel area.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        # Calculate noise per pixel
        sigma_pix = sigma_e / np.sqrt(galaxy_density * self.pixel_area_arcmin2)
        noise = rng.normal(loc=0.0, scale=sigma_pix, size=self.npix)

        if inplace:
            self.data += noise
            self.is_noisy = True
            self.noise_level = sigma_e
            return self
        else:
            result = self.copy()
            result.data = self.data + noise
            result.is_noisy = True
            result.noise_level = sigma_e
            return result

    def apply_mask(
        self,
        mask: "SurveyMask",
        fill_value: float = 0.0,
        inplace: bool = False,
    ) -> "ConvergenceMap":
        """
        Apply a survey mask to the map.

        Parameters
        ----------
        mask : SurveyMask
            Survey mask to apply.
        fill_value : float, optional
            Value to use for masked pixels (default: 0.0).
        inplace : bool, optional
            If True, modify this map in place.

        Returns
        -------
        ConvergenceMap
            Masked map.
        """
        if mask.nside != self.nside:
            raise ValueError(
                f"Mask nside ({mask.nside}) does not match map nside ({self.nside})"
            )

        if inplace:
            self.data = np.where(mask.data > 0, self.data, fill_value)
            return self
        else:
            result = self.copy()
            result.data = np.where(mask.data > 0, self.data, fill_value)
            return result

    def compute_power_spectrum(
        self,
        lmax: int = DEFAULT_LMAX,
        return_ell: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the angular power spectrum of the map.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole (default: 1024).
        return_ell : bool, optional
            If True, also return the multipole values.

        Returns
        -------
        cls : np.ndarray
            Angular power spectrum C_ell.
        ell : np.ndarray, optional
            Multipole values (if return_ell=True).
        """
        alm = hp.map2alm(self.data, lmax=lmax)
        cls = hp.alm2cl(alm)

        if return_ell:
            ell = np.arange(len(cls))
            return cls, ell
        return cls

    def compute_cross_power_spectrum(
        self,
        other: "ConvergenceMap",
        lmax: int = DEFAULT_LMAX,
    ) -> np.ndarray:
        """
        Compute the cross power spectrum with another map.

        Parameters
        ----------
        other : ConvergenceMap
            Another convergence map.
        lmax : int, optional
            Maximum multipole (default: 1024).

        Returns
        -------
        np.ndarray
            Cross power spectrum C_ell^{ab}.
        """
        if self.nside != other.nside:
            raise ValueError(f"Map nsides do not match: {self.nside} vs {other.nside}")

        alm1 = hp.map2alm(self.data, lmax=lmax)
        alm2 = hp.map2alm(other.data, lmax=lmax)
        return hp.alm2cl(alm1, alm2)

    def to_fits(
        self,
        filepath: Union[str, Path],
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Save the map to a FITS file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        overwrite : bool, optional
            Whether to overwrite existing file.
        **kwargs
            Additional arguments passed to healpy.write_map.
        """
        hp.write_map(str(filepath), self.data, overwrite=overwrite, **kwargs)

    def __repr__(self) -> str:
        parts = [f"ConvergenceMap(nside={self.nside}"]
        if self.bin_number is not None:
            parts.append(f", bin={self.bin_number}")
        if self.is_noisy:
            parts.append(f", noisy(σ={self.noise_level})")
        if self.is_bnt_transformed:
            parts.append(f", bnt_bin={self.bnt_bin}")
        parts.append(")")
        return "".join(parts)


@dataclass
class ConvergenceMapCollection:
    """
    Collection of convergence maps across multiple redshift bins.

    This class provides methods for working with tomographic data,
    including BNT transforms and cross-correlations.

    Parameters
    ----------
    maps : List[ConvergenceMap]
        List of convergence maps, one per redshift bin.

    Attributes
    ----------
    n_bins : int
        Number of redshift bins.
    nside : int
        HEALPix resolution (must be same for all maps).

    Examples
    --------
    >>> maps = [ConvergenceMap.from_h5(file, bin_number=i) for i in range(1, 5)]
    >>> collection = ConvergenceMapCollection(maps)
    >>> bnt_collection = collection.apply_bnt_transform()
    """

    maps: List[ConvergenceMap]

    def __post_init__(self):
        """Validate the collection."""
        if len(self.maps) == 0:
            raise ValueError("Collection must contain at least one map")

        # Check all maps have same nside
        nsides = {m.nside for m in self.maps}
        if len(nsides) > 1:
            raise ValueError(f"All maps must have same nside, got {nsides}")

        # Sort by bin number if available
        if all(m.bin_number is not None for m in self.maps):
            self.maps = sorted(self.maps, key=lambda m: m.bin_number)

    @property
    def n_bins(self) -> int:
        """Number of maps in the collection."""
        return len(self.maps)

    @property
    def nside(self) -> int:
        """HEALPix resolution parameter."""
        return self.maps[0].nside

    @classmethod
    def from_h5(
        cls,
        filepath: Union[str, Path],
        bin_numbers: List[int] = None,
        **kwargs,
    ) -> "ConvergenceMapCollection":
        """
        Load multiple bins from a CosmoGRID HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.
        bin_numbers : List[int], optional
            List of bin numbers to load (default: [1, 2, 3, 4]).
        **kwargs
            Additional arguments passed to ConvergenceMap.from_h5.

        Returns
        -------
        ConvergenceMapCollection
            Collection of loaded maps.
        """
        if bin_numbers is None:
            bin_numbers = [1, 2, 3, 4]
        maps = [
            ConvergenceMap.from_h5(filepath, bin_number=b, **kwargs)
            for b in bin_numbers
        ]
        return cls(maps)

    def to_array(self) -> np.ndarray:
        """
        Convert to a 2D numpy array.

        Returns
        -------
        np.ndarray
            Array of shape (n_bins, npix).
        """
        return np.array([m.data for m in self.maps])

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        nside: Optional[int] = None,
        **kwargs,
    ) -> "ConvergenceMapCollection":
        """
        Create collection from a 2D array.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_bins, npix).
        nside : int, optional
            HEALPix resolution parameter.
        **kwargs
            Additional arguments passed to ConvergenceMap constructor.

        Returns
        -------
        ConvergenceMapCollection
            Collection of maps.
        """
        maps = [
            ConvergenceMap(data=d, nside=nside, bin_number=i + 1, **kwargs)
            for i, d in enumerate(data)
        ]
        return cls(maps)

    def add_shape_noise(
        self,
        sigma_e: float = DEFAULT_SIGMA_E,
        galaxy_density: float = DEFAULT_GALAXY_DENSITY,
        seed: Optional[int] = None,
    ) -> "ConvergenceMapCollection":
        """
        Add shape noise to all maps in the collection.

        Parameters
        ----------
        sigma_e : float, optional
            Intrinsic ellipticity dispersion.
        galaxy_density : float, optional
            Galaxy number density in arcmin^-2.
        seed : int, optional
            Base random seed. Each map uses seed+bin_number.

        Returns
        -------
        ConvergenceMapCollection
            Collection with noisy maps.
        """
        noisy_maps = []
        for i, m in enumerate(self.maps):
            map_seed = None if seed is None else seed + i
            noisy_maps.append(
                m.add_shape_noise(
                    sigma_e=sigma_e,
                    galaxy_density=galaxy_density,
                    seed=map_seed,
                )
            )
        return ConvergenceMapCollection(noisy_maps)

    def apply_bnt_transform(
        self,
        bnt_matrix: Optional[np.ndarray] = None,
    ) -> "ConvergenceMapCollection":
        """
        Apply BNT transform to the map collection.

        Parameters
        ----------
        bnt_matrix : np.ndarray, optional
            Custom BNT matrix. If None, uses the default 4-bin matrix.

        Returns
        -------
        ConvergenceMapCollection
            BNT-transformed maps.

        Raises
        ------
        ValueError
            If matrix dimensions don't match number of bins.
        """
        matrix = get_bnt_matrix(n_bins=self.n_bins, custom_matrix=bnt_matrix)

        # Stack maps and apply transform
        stacked = self.to_array()  # (n_bins, npix)
        transformed = matrix @ stacked  # (n_bins, npix)

        # Create new maps with BNT metadata
        bnt_maps = []
        for i, (orig_map, new_data) in enumerate(zip(self.maps, transformed)):
            new_map = ConvergenceMap(
                data=new_data,
                nside=self.nside,
                bin_number=orig_map.bin_number,
                simulation_type=orig_map.simulation_type,
                cosmology_id=orig_map.cosmology_id,
                permutation_id=orig_map.permutation_id,
                is_noisy=orig_map.is_noisy,
                noise_level=orig_map.noise_level,
                is_bnt_transformed=True,
                bnt_bin=i,  # 0-indexed BNT bin
            )
            bnt_maps.append(new_map)

        return ConvergenceMapCollection(bnt_maps)

    def compute_all_power_spectra(
        self,
        lmax: int = DEFAULT_LMAX,
        include_cross: bool = True,
    ) -> dict:
        """
        Compute all auto and cross power spectra.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole.
        include_cross : bool, optional
            Whether to include cross-spectra (default: True).

        Returns
        -------
        dict
            Dictionary with keys (i, j) for bin pairs and Cls as values.
            Auto-spectra have i==j, cross-spectra have i<j.
        """
        # First compute all alms
        alms = [hp.map2alm(m.data, lmax=lmax) for m in self.maps]

        cls_dict = {}

        # Auto-spectra
        for i in range(self.n_bins):
            cls_dict[(i, i)] = hp.alm2cl(alms[i])

        # Cross-spectra
        if include_cross:
            for i in range(self.n_bins):
                for j in range(i + 1, self.n_bins):
                    cls_dict[(i, j)] = hp.alm2cl(alms[i], alms[j])

        return cls_dict

    def __getitem__(self, idx: int) -> ConvergenceMap:
        return self.maps[idx]

    def __len__(self) -> int:
        return len(self.maps)

    def __iter__(self):
        return iter(self.maps)

    def __repr__(self) -> str:
        return f"ConvergenceMapCollection(n_bins={self.n_bins}, nside={self.nside})"
