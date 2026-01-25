"""
L1 norm computation for convergence maps.

This module provides the L1NormProcessor class for computing L1 norms
of wavelet coefficients from HEALPix convergence maps.

The L1 norm is a summary statistic that captures non-Gaussian information
in the convergence field through wavelet decomposition.
"""

from __future__ import annotations

import numpy as np
import healpy as hp
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

from bar_impact.processing.base import BaseProcessor, ProcessingConfig
from bar_impact.constants import DEFAULT_NSIDE


__all__ = ["L1NormProcessor", "L1NormConfig", "compute_l1_norms"]


# Default parameters for L1 norm computation
DEFAULT_NSCALES = 5  # Number of wavelet scales
DEFAULT_NBINS = 40   # Number of histogram bins for L1 norm
DEFAULT_MIN_SNR = -4.0  # Minimum SNR for fine scales
DEFAULT_MAX_SNR = 4.0   # Maximum SNR for fine scales
DEFAULT_MIN_SNR_COARSE = -3.0  # Minimum SNR for coarse scale
DEFAULT_MAX_SNR_COARSE = 3.0   # Maximum SNR for coarse scale


def _check_pycs_available():
    """Check if pycs is available for wavelet computations."""
    try:
        from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere
        return True
    except ImportError:
        return False


def compute_l1_norms(
    map_data: np.ndarray,
    nscales: int = DEFAULT_NSCALES,
    nbins: int = DEFAULT_NBINS,
    mask: Optional[np.ndarray] = None,
    noise_std: Optional[float] = None,
    min_snr: float = DEFAULT_MIN_SNR,
    max_snr: float = DEFAULT_MAX_SNR,
    min_snr_coarse: Optional[float] = DEFAULT_MIN_SNR_COARSE,
    max_snr_coarse: Optional[float] = DEFAULT_MAX_SNR_COARSE,
) -> np.ndarray:
    """
    Compute L1 norms of wavelet coefficients.
    
    This function wraps the pycs get_wtl1_sphere function which computes
    L1 norms of starlet wavelet coefficients at multiple scales.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix convergence map.
    nscales : int, optional
        Number of wavelet scales (default: 5).
    nbins : int, optional
        Number of bins for the L1 norm histogram (default: 40).
    mask : np.ndarray, optional
        Binary mask (1=keep, 0=exclude). Same shape as map_data.
    noise_std : float, optional
        Noise standard deviation for SNR calculation.
    min_snr : float, optional
        Minimum SNR for fine scale bins.
    max_snr : float, optional
        Maximum SNR for fine scale bins.
    min_snr_coarse : float, optional
        Minimum SNR for coarse scale bins.
    max_snr_coarse : float, optional
        Maximum SNR for coarse scale bins.
        
    Returns
    -------
    np.ndarray
        L1 norms for each wavelet scale.
        
    Raises
    ------
    ImportError
        If pycs is not installed.
        
    Notes
    -----
    This function requires the pycs library (CosmoStat) to be installed.
    Install via: pip install pycs or conda install -c cosmostat pycs
    """
    try:
        from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere
    except ImportError:
        raise ImportError(
            "pycs library is required for L1 norm computation. "
            "Install via: pip install pycs"
        )
    
    _, l1norms = get_wtl1_sphere(
        map_data,
        nscales=nscales,
        nbins=nbins,
        Mask=mask,
        noise_std=noise_std,
        min_snr=min_snr,
        max_snr=max_snr,
        min_snr_coarse=min_snr_coarse,
        max_snr_coarse=max_snr_coarse,
    )
    
    return np.array(l1norms)


@dataclass
class L1NormConfig(ProcessingConfig):
    """
    Configuration for L1 norm processing.
    
    Parameters
    ----------
    nscales : int
        Number of wavelet scales for decomposition.
    nbins : int
        Number of bins for the L1 norm histogram.
    noise_std : float, optional
        Noise standard deviation for SNR computation.
    min_snr : float
        Minimum SNR for fine scale histogram bins.
    max_snr : float
        Maximum SNR for fine scale histogram bins.
    min_snr_coarse : float
        Minimum SNR for coarse scale histogram bins.
    max_snr_coarse : float
        Maximum SNR for coarse scale histogram bins.
    """
    
    nscales: int = DEFAULT_NSCALES
    nbins: int = DEFAULT_NBINS
    noise_std: Optional[float] = None
    min_snr: float = DEFAULT_MIN_SNR
    max_snr: float = DEFAULT_MAX_SNR
    min_snr_coarse: float = DEFAULT_MIN_SNR_COARSE
    max_snr_coarse: float = DEFAULT_MAX_SNR_COARSE


class L1NormProcessor(BaseProcessor):
    """
    Processor for computing L1 norms of wavelet coefficients.
    
    This processor computes L1 norms from convergence maps using
    spherical starlet wavelets. L1 norms capture non-Gaussian
    information that is lost in the power spectrum.
    
    Parameters
    ----------
    config : L1NormConfig, optional
        Configuration for processing. Uses defaults if not provided.
    nscales : int, optional
        Number of wavelet scales. Overrides config if provided.
    nbins : int, optional
        Number of histogram bins. Overrides config if provided.
        
    Attributes
    ----------
    nscales : int
        Number of wavelet scales.
    nbins : int
        Number of histogram bins.
    pycs_available : bool
        Whether pycs is available for computation.
        
    Examples
    --------
    >>> from bar_impact.processing import L1NormProcessor
    >>> processor = L1NormProcessor(nscales=5, nbins=40)
    >>> 
    >>> # Process a single map
    >>> l1 = processor.process_single(map_data)
    >>> l1.shape
    (200,)  # 5 scales * 40 bins
    >>> 
    >>> # Process with mask
    >>> l1 = processor.process_single(map_data, mask=survey_mask)
    
    Notes
    -----
    This processor requires the pycs library (CosmoStat) for wavelet
    computations. Install via: pip install pycs
    """
    
    statistic_type = "l1_norm"
    
    def __init__(
        self,
        config: Optional[L1NormConfig] = None,
        nscales: Optional[int] = None,
        nbins: Optional[int] = None,
        noise_std: Optional[float] = None,
        min_snr: Optional[float] = None,
        max_snr: Optional[float] = None,
        min_snr_coarse: Optional[float] = None,
        max_snr_coarse: Optional[float] = None,
    ):
        # Create config if not provided
        if config is None:
            config = L1NormConfig()
        
        super().__init__(config)
        
        # Override config values if explicitly provided
        self.nscales = nscales if nscales is not None else getattr(config, 'nscales', DEFAULT_NSCALES)
        self.nbins = nbins if nbins is not None else getattr(config, 'nbins', DEFAULT_NBINS)
        self.noise_std = noise_std if noise_std is not None else getattr(config, 'noise_std', None)
        self.min_snr = min_snr if min_snr is not None else getattr(config, 'min_snr', DEFAULT_MIN_SNR)
        self.max_snr = max_snr if max_snr is not None else getattr(config, 'max_snr', DEFAULT_MAX_SNR)
        self.min_snr_coarse = min_snr_coarse if min_snr_coarse is not None else getattr(config, 'min_snr_coarse', DEFAULT_MIN_SNR_COARSE)
        self.max_snr_coarse = max_snr_coarse if max_snr_coarse is not None else getattr(config, 'max_snr_coarse', DEFAULT_MAX_SNR_COARSE)
        
        # Check pycs availability
        self.pycs_available = _check_pycs_available()
    
    def process_single(
        self,
        map_data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        nscales: Optional[int] = None,
        nbins: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute L1 norms for a single map.
        
        Parameters
        ----------
        map_data : np.ndarray
            Input HEALPix convergence map.
        mask : np.ndarray, optional
            Binary mask for the computation.
        nscales : int, optional
            Number of scales. Uses processor default if not provided.
        nbins : int, optional
            Number of bins. Uses processor default if not provided.
        **kwargs
            Additional arguments (ignored).
            
        Returns
        -------
        np.ndarray
            L1 norms, shape (nscales * nbins,).
            
        Raises
        ------
        ImportError
            If pycs is not installed.
        """
        if not self.pycs_available:
            raise ImportError(
                "pycs library is required for L1 norm computation. "
                "Install via: pip install pycs"
            )
        
        _nscales = nscales if nscales is not None else self.nscales
        _nbins = nbins if nbins is not None else self.nbins
        
        return compute_l1_norms(
            map_data,
            nscales=_nscales,
            nbins=_nbins,
            mask=mask,
            noise_std=self.noise_std,
            min_snr=self.min_snr,
            max_snr=self.max_snr,
            min_snr_coarse=self.min_snr_coarse,
            max_snr_coarse=self.max_snr_coarse,
        )
    
    def get_output_shape(self) -> Tuple[int]:
        """
        Get the output shape for L1 norms.
        
        Returns
        -------
        tuple
            Shape of the output L1 norm array.
        """
        return (self.nscales * self.nbins,)
    
    def get_output_suffix(
        self,
        bin_number: Optional[int] = None,
        bnt_bin: Optional[int] = None,
    ) -> str:
        """Generate output filename suffix."""
        parts = ["_l1"]
        
        if bnt_bin is not None:
            parts.append(f"_bnt{bnt_bin+1}")
        elif bin_number is not None:
            parts.append(f"_bin{bin_number}")
        
        if self.config.apply_mask:
            area = int(round(self.config.mask_area_sqdeg))
            parts.append(f"_masked_{area}sqdeg")
        
        if self.config.add_noise:
            parts.append(f"_noisy_s{self.config.noise_level:.2f}")
        
        parts.append(f"_scales{self.nscales}_bins{self.nbins}")
        parts.append(".npy")
        return "".join(parts)


# Backwards compatibility - function-style interface
def process_l1_norms(
    map_data: np.ndarray,
    nside: int = DEFAULT_NSIDE,
    num_scales: int = DEFAULT_NSCALES,
    add_noise: bool = True,
    noise_level: float = 0.26,
    **kwargs,
) -> np.ndarray:
    """
    Process a map to compute L1 norms (functional interface).
    
    This function provides a simple interface for computing L1 norms
    without instantiating a processor object.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix convergence map.
    nside : int, optional
        HEALPix resolution (for noise calculation).
    num_scales : int, optional
        Number of wavelet scales.
    add_noise : bool, optional
        Whether to add shape noise.
    noise_level : float, optional
        Shape noise level.
    **kwargs
        Additional arguments passed to processor.
        
    Returns
    -------
    np.ndarray
        L1 norms for each wavelet scale.
    """
    config = ProcessingConfig(add_noise=add_noise, noise_level=noise_level)
    processor = L1NormProcessor(config=config, nscales=num_scales)
    dv = processor.process(map_data, apply_preprocessing=add_noise)
    return dv.data


def compute_wavelet_transform(
    map_data: np.ndarray,
    num_scales: int = DEFAULT_NSCALES,
) -> Tuple[np.ndarray, ...]:
    """
    Compute wavelet transform of a spherical map.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix map.
    num_scales : int, optional
        Number of scales for the wavelet decomposition.
        
    Returns
    -------
    tuple of np.ndarray
        Wavelet coefficients at each scale.
        
    Raises
    ------
    ImportError
        If pycs is not installed.
    """
    try:
        from pycs.sparsity.mrs.mrs_starlet import CMRStarlet
    except ImportError:
        raise ImportError(
            "pycs library is required for wavelet computation. "
            "Install via: pip install pycs"
        )
    
    nside = hp.get_nside(map_data)
    starlet = CMRStarlet(nside=nside, nscales=num_scales)
    starlet.decompose(map_data)
    
    return tuple(starlet.coef[i] for i in range(num_scales))
