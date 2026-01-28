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
    ngal_arcmin2: float = 6.75,  # Matches DEFAULT_GALAXY_DENSITY in constants.py
    galaxy_density: Optional[float] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
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
    galaxy_density : float, optional
        Alias for ngal_arcmin2 (for backward compatibility)
    seed : int, optional
        Random seed for reproducibility (legacy interface)
    rng : np.random.Generator, optional
        Modern random number generator for reproducibility.
        If provided, takes precedence over seed.
    inplace : bool, optional
        If True, modify map_data in place (default: False)
        
    Returns
    -------
    np.ndarray
        Map with added shape noise
        
    Notes
    -----
    The shape noise variance per pixel for convergence is:
    
    .. math::
        \\sigma_{\\kappa}^2 = \\frac{\\sigma_e^2}{n_{\\rm gal} A_{\\rm pix}}
        
    where :math:`n_{\\rm gal}` is the galaxy density and 
    :math:`A_{\\rm pix}` is the pixel area.
    
    Note: For shear components, there would be a factor of 2 in the denominator
    because shear has two independent components. For convergence (a scalar),
    we do NOT include this factor.
    
    Examples
    --------
    >>> import healpy as hp
    >>> nside = 512
    >>> kappa = np.random.randn(hp.nside2npix(nside)) * 0.01
    >>> kappa_noisy = add_shape_noise(kappa, sigma_e=0.26, nside=nside)
    >>> np.std(kappa_noisy) > np.std(kappa)
    True
    
    >>> # Using modern RNG for reproducibility
    >>> rng = np.random.default_rng(42)
    >>> kappa_noisy1 = add_shape_noise(kappa, rng=rng)
    >>> rng = np.random.default_rng(42)
    >>> kappa_noisy2 = add_shape_noise(kappa, rng=rng)
    >>> np.allclose(kappa_noisy1, kappa_noisy2)
    True
    """
    # Handle backward compatibility alias
    if galaxy_density is not None:
        ngal_arcmin2 = galaxy_density
    
    # Set up random number generator
    if rng is not None:
        # Modern Generator interface (preferred)
        pass
    elif seed is not None:
        # Legacy seed interface
        rng = np.random.RandomState(seed)
    else:
        # Default: use global random state
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
    
    # Calculate shape noise variance
    # For convergence: sigma_kappa = sigma_e / sqrt(n_gal * A_pix)
    # (No factor of 2 - that's only for shear components)
    if ngal_arcmin2 > 0:
        sigma_noise = sigma_e / np.sqrt(ngal_arcmin2 * pixel_area_arcmin2)
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
    ngal_arcmin2: float = 6.75,  # Matches DEFAULT_GALAXY_DENSITY in constants.py
    galaxy_density: Optional[float] = None,
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
    galaxy_density : float, optional
        Alias for ngal_arcmin2 (for backward compatibility)
    nside : int, optional
        HEALPix nside (auto-detected if not provided)
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Shape noise map
    """
    # Handle backward compatibility
    if galaxy_density is not None:
        ngal_arcmin2 = galaxy_density
    
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
