"""
Noise generation utilities for simulating observational effects.

This module provides functions for adding shape noise and other
observational effects to simulated maps.
"""

import numpy as np
from typing import Optional


def add_shape_noise(
    map_data: np.ndarray,
    sigma_e: float = 0.26,
    nside: Optional[int] = None,
    area_deg2: Optional[float] = None,
    ngal_arcmin2: float = 30.0,
    seed: Optional[int] = None,
    inplace: bool = False
) -> np.ndarray:
    """
    Add shape noise to a convergence map.
    
    Shape noise is added as Gaussian random noise with variance
    determined by the galaxy number density and intrinsic ellipticity.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input convergence map (HEALPix format)
    sigma_e : float, optional
        Intrinsic ellipticity dispersion (default: 0.26)
    nside : int, optional
        HEALPix resolution parameter (auto-detected if not provided)
    area_deg2 : float, optional
        Survey area in square degrees (full sky if not provided)
    ngal_arcmin2 : float, optional
        Galaxy number density in arcmin^-2 (default: 30.0)
    seed : int, optional
        Random seed for reproducibility
    inplace : bool, optional
        If True, modify map_data in place (default: False)
        
    Returns
    -------
    np.ndarray
        Map with added shape noise
        
    Notes
    -----
    The shape noise variance per pixel is calculated as:
    
    .. math::
        \\sigma_{\\kappa}^2 = \\frac{\\sigma_e^2}{2 n_{\\rm gal} A_{\\rm pix}}
        
    where :math:`n_{\\rm gal}` is the galaxy density and 
    :math:`A_{\\rm pix}` is the pixel area.
    
    Examples
    --------
    >>> import healpy as hp
    >>> nside = 512
    >>> kappa = np.random.randn(hp.nside2npix(nside)) * 0.01
    >>> kappa_noisy = add_shape_noise(kappa, sigma_e=0.26, nside=nside)
    >>> np.std(kappa_noisy) > np.std(kappa)
    True
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    if not inplace:
        map_data = map_data.copy()
    
    # Get HEALPix parameters
    if nside is None:
        import healpy as hp
        nside = hp.npix2nside(len(map_data))
    
    # Calculate pixel area
    pixel_area_sr = 4 * np.pi / len(map_data)  # steradians
    pixel_area_deg2 = pixel_area_sr * (180 / np.pi)**2
    pixel_area_arcmin2 = pixel_area_deg2 * 3600  # arcmin^2
    
    # Calculate number of galaxies per pixel
    if area_deg2 is not None:
        # Partial sky survey
        npix_survey = area_deg2 / pixel_area_deg2
        fsky = npix_survey / len(map_data)
    else:
        # Full sky
        fsky = 1.0
    
    ngal_per_pixel = ngal_arcmin2 * pixel_area_arcmin2 * fsky
    
    # Calculate shape noise variance
    if ngal_per_pixel > 0:
        sigma_noise = sigma_e / np.sqrt(2 * ngal_per_pixel)
    else:
        sigma_noise = 0.0
    
    # Add Gaussian noise
    if sigma_noise > 0:
        noise = rng.normal(0, sigma_noise, size=len(map_data))
        map_data += noise
    
    return map_data


def generate_shape_noise_realization(
    npix: int,
    sigma_e: float = 0.26,
    ngal_arcmin2: float = 30.0,
    nside: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a shape noise realization.
    
    Parameters
    ----------
    npix : int
        Number of pixels (must be valid HEALPix npix)
    sigma_e : float
        Intrinsic ellipticity dispersion
    ngal_arcmin2 : float
        Galaxy number density
    nside : int, optional
        HEALPix nside (auto-detected if not provided)
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Shape noise map
    """
    noise_map = np.zeros(npix)
    return add_shape_noise(
        noise_map,
        sigma_e=sigma_e,
        nside=nside,
        ngal_arcmin2=ngal_arcmin2,
        seed=seed,
        inplace=True
    )


def estimate_noise_level(
    noisy_map: np.ndarray,
    clean_map: Optional[np.ndarray] = None
) -> float:
    """
    Estimate the noise level in a map.
    
    Parameters
    ----------
    noisy_map : np.ndarray
        Map with noise
    clean_map : np.ndarray, optional
        Clean map without noise (if available)
        
    Returns
    -------
    float
        Estimated noise standard deviation
    """
    if clean_map is not None:
        noise = noisy_map - clean_map
        return np.std(noise)
    else:
        # Estimate from high-ell modes (rough approximation)
        return np.std(noisy_map)
