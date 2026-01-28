"""
Survey mask data structures for BAR_IMPACT.

This module provides the SurveyMask class for representing and
creating survey footprint masks for partial-sky analysis.
"""

from __future__ import annotations

import numpy as np
import healpy as hp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, Dict

from bar_impact.constants import (
    DEFAULT_NSIDE,
    DEFAULT_MASK_AREA_SQDEG,
    DEFAULT_MASK_CENTER,
    FULL_SKY_AREA_SQDEG,
)


__all__ = ["SurveyMask"]


# Global cache for masks to avoid recomputation
_MASK_CACHE: Dict[tuple, "SurveyMask"] = {}


@dataclass
class SurveyMask:
    """
    Survey footprint mask for partial-sky analysis.
    
    This class represents a binary or weighted mask that defines
    the survey footprint. It supports various mask geometries and
    apodization schemes.
    
    Parameters
    ----------
    data : np.ndarray
        HEALPix mask array. Values should be in [0, 1] where
        0 = masked and 1 = unmasked.
    nside : int, optional
        HEALPix resolution. Inferred from data if not provided.
    area_sqdeg : float, optional
        Effective survey area in square degrees.
    f_sky : float, optional
        Fraction of sky covered. Computed from data if not provided.
    center_coords : Tuple[float, float], optional
        Center coordinates (lon, lat) in degrees.
    apodization_deg : float, optional
        Apodization width in degrees (0 if not apodized).
        
    Attributes
    ----------
    npix : int
        Number of pixels in the mask.
        
    Examples
    --------
    >>> mask = SurveyMask.create_disk_mask(
    ...     nside=512,
    ...     target_area_sqdeg=14000.0,
    ...     center_coords=(0.0, 90.0)
    ... )
    >>> mask.f_sky
    0.339...
    
    >>> # Apply to a map
    >>> masked_map = np.where(mask.data > 0, kappa_map, 0)
    """
    
    data: np.ndarray
    nside: int = field(default=None)
    area_sqdeg: Optional[float] = None
    f_sky: Optional[float] = None
    center_coords: Optional[Tuple[float, float]] = None
    apodization_deg: float = 0.0
    angular_radius_deg: Optional[float] = None
    
    def __post_init__(self):
        """Validate and compute derived attributes."""
        self.data = np.asarray(self.data, dtype=np.float32)
        
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
        
        # Compute f_sky if not provided
        if self.f_sky is None:
            self.f_sky = float(np.mean(self.data))
        
        # Compute area if not provided
        if self.area_sqdeg is None:
            self.area_sqdeg = self.f_sky * FULL_SKY_AREA_SQDEG
    
    @property
    def npix(self) -> int:
        """Number of pixels in the mask."""
        return len(self.data)
    
    @property
    def is_binary(self) -> bool:
        """Check if mask is strictly binary (0 or 1)."""
        unique_vals = np.unique(self.data)
        return len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))
    
    @property
    def n_unmasked_pixels(self) -> int:
        """Number of unmasked pixels (where mask > 0)."""
        return int(np.sum(self.data > 0))
    
    @classmethod
    def create_disk_mask(
        cls,
        nside: int = DEFAULT_NSIDE,
        target_area_sqdeg: float = DEFAULT_MASK_AREA_SQDEG,
        center_coords: Tuple[float, float] = DEFAULT_MASK_CENTER,
        use_cache: bool = True,
    ) -> "SurveyMask":
        """
        Create a circular (disk) mask on the sphere.
        
        This creates a spherical cap mask centered at the specified
        coordinates with an area matching the target.
        
        Parameters
        ----------
        nside : int, optional
            HEALPix resolution (default: 512).
        target_area_sqdeg : float, optional
            Target unmasked area in square degrees (default: 14000).
        center_coords : Tuple[float, float], optional
            Center (lon, lat) in degrees (default: (0, 90) = North Pole).
        use_cache : bool, optional
            Whether to use cached masks (default: True).
            
        Returns
        -------
        SurveyMask
            Circular survey mask.
            
        Notes
        -----
        For a spherical cap with solid angle Ω, the angular radius θ
        satisfies: Ω = 2π(1 - cos(θ))
        
        Examples
        --------
        >>> mask = SurveyMask.create_disk_mask(
        ...     nside=512,
        ...     target_area_sqdeg=14000.0
        ... )
        >>> print(f"f_sky = {mask.f_sky:.3f}")
        f_sky = 0.339
        """
        # Check cache
        cache_key = (
            int(nside),
            float(target_area_sqdeg),
            float(center_coords[0]),
            float(center_coords[1]),
            0.0,  # no apodization
        )
        if use_cache and cache_key in _MASK_CACHE:
            return _MASK_CACHE[cache_key]
        
        # Compute angular radius for target area
        # Ω = 2π(1 - cos(θ)) => cos(θ) = 1 - Ω/(2π)
        # In terms of area: Ω/4π = target_area / full_sky_area
        frac = target_area_sqdeg / FULL_SKY_AREA_SQDEG
        angular_radius_rad = np.arccos(1 - 2 * frac)
        angular_radius_deg = np.degrees(angular_radius_rad)
        
        # Convert center to HEALPix convention (theta, phi)
        lon, lat = center_coords
        theta_center = np.radians(90.0 - lat)  # co-latitude
        phi_center = np.radians(lon)
        center_vec = hp.ang2vec(theta_center, phi_center)
        
        # Query disk pixels
        disc_pixels = hp.query_disc(nside, center_vec, angular_radius_rad)
        
        # Create mask array
        npix = hp.nside2npix(nside)
        mask_data = np.zeros(npix, dtype=np.float32)
        mask_data[disc_pixels] = 1.0
        
        # Compute actual f_sky
        f_sky = float(np.mean(mask_data))
        
        mask = cls(
            data=mask_data,
            nside=nside,
            area_sqdeg=f_sky * FULL_SKY_AREA_SQDEG,
            f_sky=f_sky,
            center_coords=center_coords,
            apodization_deg=0.0,
            angular_radius_deg=angular_radius_deg,
        )
        
        # Cache if requested
        if use_cache:
            _MASK_CACHE[cache_key] = mask
        
        return mask
    
    @classmethod
    def create_apodized_disk_mask(
        cls,
        nside: int = DEFAULT_NSIDE,
        target_area_sqdeg: float = DEFAULT_MASK_AREA_SQDEG,
        center_coords: Tuple[float, float] = DEFAULT_MASK_CENTER,
        apodization_deg: float = 2.0,
        apodization_type: str = 'C1',
        use_cache: bool = True,
    ) -> "SurveyMask":
        """
        Create an apodized (smooth edge) circular mask.
        
        The mask smoothly transitions from 1 to 0 over the apodization
        region at the mask boundary.
        
        Parameters
        ----------
        nside : int, optional
            HEALPix resolution.
        target_area_sqdeg : float, optional
            Target unmasked area.
        center_coords : Tuple[float, float], optional
            Center (lon, lat) in degrees.
        apodization_deg : float, optional
            Width of the apodization region in degrees (default: 2.0).
        apodization_type : str, optional
            Type of apodization: 'C1' (cosine taper) or 'C2' (polynomial).
            C2 is recommended for power spectrum analysis (default: 'C1').
        use_cache : bool, optional
            Whether to use cached masks.
            
        Returns
        -------
        SurveyMask
            Apodized circular mask.
            
        Notes
        -----
        Apodization helps reduce edge effects in power spectrum
        estimation.
        
        **C1 (cosine taper)**: Smooth but not twice differentiable
        
        .. math::
            w(x) = \\frac{1}{2}\\left[1 + \\cos(\\pi x)\\right]
        
        **C2 (polynomial)**: Twice differentiable, better for power spectra
        
        .. math::
            w(x) = \\begin{cases}
                1 - 2x^2 & x < 0.5 \\\\
                2(1-x)^2 & x \\geq 0.5
            \\end{cases}
            
        where x is the normalized distance within the transition region.
        """
        # Check cache
        cache_key = (
            int(nside),
            float(target_area_sqdeg),
            float(center_coords[0]),
            float(center_coords[1]),
            float(apodization_deg),
            str(apodization_type),
        )
        if use_cache and cache_key in _MASK_CACHE:
            return _MASK_CACHE[cache_key]
        
        # Start with binary disk mask
        base_mask = cls.create_disk_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
            use_cache=False,  # Don't cache intermediate
        )
        
        if apodization_deg <= 0:
            return base_mask
        
        # Convert center to vector
        lon, lat = center_coords
        theta_center = np.radians(90.0 - lat)
        phi_center = np.radians(lon)
        center_vec = hp.ang2vec(theta_center, phi_center)
        
        # Get all pixel positions
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        pixel_vecs = hp.ang2vec(theta, phi)
        
        # Compute angular distance from center for each pixel
        # dot product gives cos(angular distance)
        cos_dist = np.sum(pixel_vecs * center_vec, axis=1)
        angular_dist_deg = np.degrees(np.arccos(np.clip(cos_dist, -1, 1)))
        
        # Define transition region
        # Note: apodization_deg specifies the half-width on each side, so total
        # transition width is 2 * apodization_deg (matching original scripts)
        inner_radius = base_mask.angular_radius_deg - apodization_deg
        outer_radius = base_mask.angular_radius_deg + apodization_deg
        transition_width = 2 * apodization_deg

        # Create apodized mask
        mask_data = np.zeros(npix, dtype=np.float32)

        # Inner region: fully unmasked
        inner_mask = angular_dist_deg < max(0, inner_radius)
        mask_data[inner_mask] = 1.0

        # Transition region: cosine taper or polynomial
        transition_mask = (angular_dist_deg >= max(0, inner_radius)) & (angular_dist_deg < outer_radius)
        if np.any(transition_mask):
            frac = (angular_dist_deg[transition_mask] - max(0, inner_radius)) / transition_width
            
            if apodization_type == 'C1':
                # C1 continuous (cosine taper): smooth but not twice-differentiable
                mask_data[transition_mask] = 0.5 * (1 + np.cos(np.pi * frac))
            elif apodization_type == 'C2':
                # C2 continuous (polynomial): twice differentiable, better for power spectra
                taper = np.where(frac < 0.5, 1.0 - 2 * frac**2, 2 * (1 - frac)**2)
                mask_data[transition_mask] = taper.astype(np.float32)
            else:
                raise ValueError(f"Unknown apodization_type: {apodization_type}. "
                               f"Must be 'C1' or 'C2'.")
        
        # Compute effective f_sky (weighted by mask values)
        f_sky = float(np.mean(mask_data))
        
        mask = cls(
            data=mask_data,
            nside=nside,
            area_sqdeg=f_sky * FULL_SKY_AREA_SQDEG,
            f_sky=f_sky,
            center_coords=center_coords,
            apodization_deg=apodization_deg,
            angular_radius_deg=base_mask.angular_radius_deg,
        )
        
        if use_cache:
            _MASK_CACHE[cache_key] = mask
        
        return mask
    
    @classmethod
    def from_fits(
        cls,
        filepath: Union[str, Path],
        field: int = 0,
        **kwargs,
    ) -> "SurveyMask":
        """
        Load a mask from a FITS file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the FITS file.
        field : int, optional
            Field index to read.
        **kwargs
            Additional arguments passed to constructor.
            
        Returns
        -------
        SurveyMask
            Loaded mask.
        """
        data = hp.read_map(str(filepath), field=field, verbose=False)
        return cls(data=data, **kwargs)
    
    @classmethod
    def full_sky(cls, nside: int = DEFAULT_NSIDE) -> "SurveyMask":
        """
        Create a full-sky (all ones) mask.
        
        Parameters
        ----------
        nside : int, optional
            HEALPix resolution.
            
        Returns
        -------
        SurveyMask
            Full-sky mask with all pixels = 1.
        """
        npix = hp.nside2npix(nside)
        return cls(
            data=np.ones(npix, dtype=np.float32),
            nside=nside,
            area_sqdeg=FULL_SKY_AREA_SQDEG,
            f_sky=1.0,
        )
    
    def copy(self) -> "SurveyMask":
        """Create a deep copy of this mask."""
        return SurveyMask(
            data=self.data.copy(),
            nside=self.nside,
            area_sqdeg=self.area_sqdeg,
            f_sky=self.f_sky,
            center_coords=self.center_coords,
            apodization_deg=self.apodization_deg,
            angular_radius_deg=self.angular_radius_deg,
        )
    
    def invert(self) -> "SurveyMask":
        """
        Return an inverted mask (1 - mask).
        
        Returns
        -------
        SurveyMask
            Inverted mask.
        """
        inverted = self.copy()
        inverted.data = 1.0 - self.data
        inverted.f_sky = 1.0 - self.f_sky
        inverted.area_sqdeg = FULL_SKY_AREA_SQDEG - self.area_sqdeg
        return inverted
    
    def to_fits(
        self,
        filepath: Union[str, Path],
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Save the mask to a FITS file.
        
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
        apod_str = f", apod={self.apodization_deg}°" if self.apodization_deg > 0 else ""
        return (
            f"SurveyMask(nside={self.nside}, f_sky={self.f_sky:.3f}, "
            f"area={self.area_sqdeg:.0f} sq.deg{apod_str})"
        )


def clear_mask_cache() -> None:
    """Clear the global mask cache."""
    _MASK_CACHE.clear()


def get_cached_mask(
    nside: int = DEFAULT_NSIDE,
    target_area_sqdeg: float = DEFAULT_MASK_AREA_SQDEG,
    center_coords: Tuple[float, float] = DEFAULT_MASK_CENTER,
    apodization_deg: float = 0.0,
) -> SurveyMask:
    """
    Get a cached mask, creating it if necessary.
    
    This is a convenience function that maintains backward compatibility
    with the original scripts.
    
    Parameters
    ----------
    nside : int
        HEALPix resolution.
    target_area_sqdeg : float
        Target area in square degrees.
    center_coords : Tuple[float, float]
        Center (lon, lat) in degrees.
    apodization_deg : float
        Apodization width in degrees.
        
    Returns
    -------
    SurveyMask
        Cached or newly created mask.
    """
    if apodization_deg > 0:
        return SurveyMask.create_apodized_disk_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
            apodization_deg=apodization_deg,
            use_cache=True,
        )
    else:
        return SurveyMask.create_disk_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
            use_cache=True,
        )
