"""
Peak count computation for convergence maps.

This module provides the PeakCountProcessor class for computing
peak count statistics from HEALPix convergence maps.

Peak counts capture non-Gaussian information in the convergence field
by measuring the distribution of local maxima at different SNR levels.
"""

from __future__ import annotations

import numpy as np
import healpy as hp
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

from bar_impact.processing.base import BaseProcessor, ProcessingConfig
from bar_impact.constants import DEFAULT_NSIDE


__all__ = ["PeakCountProcessor", "PeakCountConfig", "compute_peak_counts"]


# Default parameters for peak count computation
DEFAULT_NSCALES = 5  # Number of wavelet scales
DEFAULT_NBINS = 40   # Number of SNR bins
DEFAULT_MIN_VAL = -4.0  # Minimum SNR value
DEFAULT_MAX_VAL = 4.0   # Maximum SNR value


def _check_pycs_available():
    """Check if pycs is available for peak computations."""
    try:
        from pycs.astro.wl.hos_peaks_l1 import get_wtpeaks_sphere
        return True
    except ImportError:
        return False


def compute_peak_counts(
    map_data: np.ndarray,
    nscales: int = DEFAULT_NSCALES,
    nbins: int = DEFAULT_NBINS,
    noise_std: Optional[float] = None,
    min_val: float = DEFAULT_MIN_VAL,
    max_val: float = DEFAULT_MAX_VAL,
) -> np.ndarray:
    """
    Compute peak counts at multiple wavelet scales.
    
    This function wraps the pycs get_wtpeaks_sphere function which computes
    peak counts in starlet wavelet coefficient maps.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix convergence map.
    nscales : int, optional
        Number of wavelet scales (default: 5).
    nbins : int, optional
        Number of SNR bins for counting (default: 40).
    noise_std : float, optional
        Noise standard deviation for SNR calculation.
    min_val : float, optional
        Minimum SNR value for binning.
    max_val : float, optional
        Maximum SNR value for binning.
        
    Returns
    -------
    np.ndarray
        Peak counts at each scale and SNR bin.
        
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
        from pycs.astro.wl.hos_peaks_l1 import get_wtpeaks_sphere
    except ImportError:
        raise ImportError(
            "pycs library is required for peak count computation. "
            "Install via: pip install pycs"
        )
    
    peak_counts, _ = get_wtpeaks_sphere(
        map_data,
        nscales=nscales,
        noise_std=noise_std,
        nbins=nbins,
        Min=min_val,
        Max=max_val,
        verbose=False,
    )
    
    return np.array(peak_counts)


@dataclass
class PeakCountConfig(ProcessingConfig):
    """
    Configuration for peak count processing.
    
    Parameters
    ----------
    nscales : int
        Number of wavelet scales for decomposition.
    nbins : int
        Number of SNR bins for peak histogram.
    noise_std : float, optional
        Noise standard deviation for SNR computation.
    min_val : float
        Minimum SNR value for histogram.
    max_val : float
        Maximum SNR value for histogram.
    """
    
    nscales: int = DEFAULT_NSCALES
    nbins: int = DEFAULT_NBINS
    noise_std: Optional[float] = None
    min_val: float = DEFAULT_MIN_VAL
    max_val: float = DEFAULT_MAX_VAL


class PeakCountProcessor(BaseProcessor):
    """
    Processor for computing peak counts from convergence maps.
    
    This processor computes peak counts from convergence maps using
    spherical starlet wavelets. Peak counts capture non-Gaussian
    information that complements power spectra and L1 norms.
    
    Parameters
    ----------
    config : PeakCountConfig, optional
        Configuration for processing. Uses defaults if not provided.
    nscales : int, optional
        Number of wavelet scales. Overrides config if provided.
    nbins : int, optional
        Number of SNR bins. Overrides config if provided.
        
    Attributes
    ----------
    nscales : int
        Number of wavelet scales.
    nbins : int
        Number of SNR bins.
    pycs_available : bool
        Whether pycs is available for computation.
        
    Examples
    --------
    >>> from bar_impact.processing import PeakCountProcessor
    >>> processor = PeakCountProcessor(nscales=5, nbins=40)
    >>> 
    >>> # Process a single map
    >>> peaks = processor.process_single(map_data)
    >>> peaks.shape
    (200,)  # 5 scales * 40 bins
    
    Notes
    -----
    This processor requires the pycs library (CosmoStat) for wavelet
    computations. Install via: pip install pycs
    """
    
    statistic_type = "peak_counts"
    
    def __init__(
        self,
        config: Optional[PeakCountConfig] = None,
        nscales: Optional[int] = None,
        nbins: Optional[int] = None,
        noise_std: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        # Create config if not provided
        if config is None:
            config = PeakCountConfig()
        
        super().__init__(config)
        
        # Override config values if explicitly provided
        self.nscales = nscales if nscales is not None else getattr(config, 'nscales', DEFAULT_NSCALES)
        self.nbins = nbins if nbins is not None else getattr(config, 'nbins', DEFAULT_NBINS)
        self.noise_std = noise_std if noise_std is not None else getattr(config, 'noise_std', None)
        self.min_val = min_val if min_val is not None else getattr(config, 'min_val', DEFAULT_MIN_VAL)
        self.max_val = max_val if max_val is not None else getattr(config, 'max_val', DEFAULT_MAX_VAL)
        
        # Check pycs availability
        self.pycs_available = _check_pycs_available()
    
    def process_single(
        self,
        map_data: np.ndarray,
        nscales: Optional[int] = None,
        nbins: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute peak counts for a single map.
        
        Parameters
        ----------
        map_data : np.ndarray
            Input HEALPix convergence map.
        nscales : int, optional
            Number of scales. Uses processor default if not provided.
        nbins : int, optional
            Number of bins. Uses processor default if not provided.
        **kwargs
            Additional arguments (ignored).
            
        Returns
        -------
        np.ndarray
            Peak counts, shape (nscales * nbins,).
            
        Raises
        ------
        ImportError
            If pycs is not installed.
        """
        if not self.pycs_available:
            raise ImportError(
                "pycs library is required for peak count computation. "
                "Install via: pip install pycs"
            )
        
        _nscales = nscales if nscales is not None else self.nscales
        _nbins = nbins if nbins is not None else self.nbins
        
        return compute_peak_counts(
            map_data,
            nscales=_nscales,
            nbins=_nbins,
            noise_std=self.noise_std,
            min_val=self.min_val,
            max_val=self.max_val,
        )
    
    def get_output_shape(self) -> Tuple[int]:
        """
        Get the output shape for peak counts.
        
        Returns
        -------
        tuple
            Shape of the output peak count array.
        """
        return (self.nscales * self.nbins,)
    
    def get_output_suffix(
        self,
        bin_number: Optional[int] = None,
        bnt_bin: Optional[int] = None,
    ) -> str:
        """Generate output filename suffix."""
        parts = ["_peaks"]
        
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


def identify_peaks(
    map_data: np.ndarray,
    threshold: float = 0.0,
) -> List[Tuple[int, float]]:
    """
    Identify peak locations and values in a HEALPix map.
    
    A pixel is considered a peak if its value is higher than all
    its neighbors.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix map.
    threshold : float, optional
        Minimum peak height to include (default: 0.0).
        
    Returns
    -------
    list of tuples
        List of (pixel_index, peak_value) tuples.
        
    Notes
    -----
    This is a simple implementation for identifying peaks directly
    in HEALPix maps. For wavelet-based peak analysis, use the
    PeakCountProcessor instead.
    """
    nside = hp.get_nside(map_data)
    npix = len(map_data)
    peaks = []
    
    for ipix in range(npix):
        neighbors = hp.get_all_neighbours(nside, ipix)
        # Remove invalid neighbor indices (-1)
        neighbors = neighbors[neighbors >= 0]
        
        pixel_val = map_data[ipix]
        if pixel_val > threshold and np.all(pixel_val > map_data[neighbors]):
            peaks.append((ipix, pixel_val))
    
    return peaks


# Backwards compatibility - function-style interface
def process_peak_counts(
    map_data: np.ndarray,
    snr_bins: Optional[np.ndarray] = None,
    add_noise: bool = False,
    noise_level: float = 0.26,
    **kwargs,
) -> np.ndarray:
    """
    Process a map to compute peak counts (functional interface).
    
    This function provides a simple interface for computing peak counts
    without instantiating a processor object.
    
    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix convergence map.
    snr_bins : np.ndarray, optional
        SNR bin edges (not directly used - for interface compatibility).
    add_noise : bool, optional
        Whether to add shape noise.
    noise_level : float, optional
        Shape noise level.
    **kwargs
        Additional arguments passed to processor.
        
    Returns
    -------
    np.ndarray
        Peak counts.
    """
    config = ProcessingConfig(add_noise=add_noise, noise_level=noise_level)
    processor = PeakCountProcessor(config=config)
    dv = processor.process(map_data, apply_preprocessing=add_noise)
    return dv.data
