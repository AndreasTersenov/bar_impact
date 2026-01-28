"""
Power spectrum computation for convergence maps.

This module provides the PowerSpectrumProcessor class for computing
angular power spectra from HEALPix convergence maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import healpy as hp
import numpy as np

from bar_impact.constants import DEFAULT_LMAX
from bar_impact.processing.base import BaseProcessor, ProcessingConfig

__all__ = [
    "PowerSpectrumProcessor",
    "compute_power_spectrum",
    "compute_cross_power_spectrum",
]


def compute_power_spectrum(
    map_data: np.ndarray,
    lmax: int = DEFAULT_LMAX,
) -> np.ndarray:
    """
    Compute the angular power spectrum of a HEALPix map.

    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix convergence map.
    lmax : int, optional
        Maximum multipole to compute (default: 1024).

    Returns
    -------
    np.ndarray
        Angular power spectrum C_ell, with length lmax+1.

    Examples
    --------
    >>> import healpy as hp
    >>> nside = 512
    >>> map_data = np.random.randn(hp.nside2npix(nside)) * 0.01
    >>> cls = compute_power_spectrum(map_data, lmax=100)
    >>> len(cls)
    101
    """
    alm = hp.map2alm(map_data, lmax=lmax)
    return hp.alm2cl(alm)


def compute_cross_power_spectrum(
    map1: np.ndarray,
    map2: np.ndarray,
    lmax: int = DEFAULT_LMAX,
) -> np.ndarray:
    """
    Compute the cross power spectrum between two HEALPix maps.

    Parameters
    ----------
    map1 : np.ndarray
        First HEALPix map.
    map2 : np.ndarray
        Second HEALPix map.
    lmax : int, optional
        Maximum multipole to compute.

    Returns
    -------
    np.ndarray
        Cross power spectrum C_ell^{12}.
    """
    alm1 = hp.map2alm(map1, lmax=lmax)
    alm2 = hp.map2alm(map2, lmax=lmax)
    return hp.alm2cl(alm1, alm2)


@dataclass
class PowerSpectrumConfig(ProcessingConfig):
    """
    Configuration for power spectrum processing.

    Parameters
    ----------
    lmax : int
        Maximum multipole for power spectrum calculation.
    ell_min : int, optional
        Minimum multipole to include in output (for binning).
    ell_max : int, optional
        Maximum multipole to include in output.
    binning : bool
        Whether to apply multipole binning.
    bin_width : int
        Width of multipole bins (if binning).
    """

    lmax: int = DEFAULT_LMAX
    ell_min: Optional[int] = None
    ell_max: Optional[int] = None
    binning: bool = False
    bin_width: int = 10


class PowerSpectrumProcessor(BaseProcessor):
    """
    Processor for computing angular power spectra from convergence maps.

    This processor computes the angular power spectrum C_ell from
    HEALPix convergence maps, with optional ell-range selection and binning.

    Parameters
    ----------
    config : PowerSpectrumConfig, optional
        Configuration for processing. Uses defaults if not provided.
    lmax : int, optional
        Maximum multipole. Overrides config if provided.

    Attributes
    ----------
    lmax : int
        Maximum multipole for computation.
    ell_range : tuple or None
        (ell_min, ell_max) for output selection.

    Examples
    --------
    >>> from bar_impact.processing import PowerSpectrumProcessor
    >>> processor = PowerSpectrumProcessor(lmax=1024)
    >>>
    >>> # Process a single map
    >>> cls = processor.process_single(map_data)
    >>> cls.shape
    (1025,)
    >>>
    >>> # Process with ell selection
    >>> processor = PowerSpectrumProcessor(lmax=1024, ell_min=100, ell_max=500)
    >>> cls = processor.process_single(map_data)
    >>> cls.shape
    (401,)
    """

    statistic_type = "power_spectrum"

    def __init__(
        self,
        config: Optional[PowerSpectrumConfig] = None,
        lmax: Optional[int] = None,
        ell_min: Optional[int] = None,
        ell_max: Optional[int] = None,
    ):
        # Create config if not provided
        if config is None:
            config = PowerSpectrumConfig()

        super().__init__(config)

        # Override config values if explicitly provided
        self.lmax = lmax if lmax is not None else getattr(config, "lmax", DEFAULT_LMAX)
        self.ell_min = (
            ell_min if ell_min is not None else getattr(config, "ell_min", None)
        )
        self.ell_max = (
            ell_max if ell_max is not None else getattr(config, "ell_max", None)
        )
        self.binning = getattr(config, "binning", False)
        self.bin_width = getattr(config, "bin_width", 10)

    def process_single(
        self,
        map_data: np.ndarray,
        lmax: Optional[int] = None,
        return_ell: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the power spectrum of a single map.

        Parameters
        ----------
        map_data : np.ndarray
            Input HEALPix convergence map.
        lmax : int, optional
            Maximum multipole. Uses processor default if not provided.
        return_ell : bool, optional
            If True, also return the multipole values.
        **kwargs
            Ignored (for interface compatibility).

        Returns
        -------
        cls : np.ndarray
            Angular power spectrum.
        ell : np.ndarray, optional
            Multipole values (if return_ell=True).
        """
        _lmax = lmax if lmax is not None else self.lmax

        # Compute full power spectrum
        cls = compute_power_spectrum(map_data, lmax=_lmax)
        ell = np.arange(len(cls))

        # Apply ell range selection
        if self.ell_min is not None or self.ell_max is not None:
            ell_min = self.ell_min if self.ell_min is not None else 0
            ell_max = self.ell_max if self.ell_max is not None else len(cls) - 1
            mask = (ell >= ell_min) & (ell <= ell_max)
            cls = cls[mask]
            ell = ell[mask]

        # Apply binning if configured
        if self.binning:
            cls, ell = self._bin_spectrum(cls, ell)

        if return_ell:
            return cls, ell
        return cls

    def process_cross(
        self,
        map1: np.ndarray,
        map2: np.ndarray,
        lmax: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute cross power spectrum between two maps.

        Parameters
        ----------
        map1 : np.ndarray
            First HEALPix map.
        map2 : np.ndarray
            Second HEALPix map.
        lmax : int, optional
            Maximum multipole.

        Returns
        -------
        np.ndarray
            Cross power spectrum.
        """
        _lmax = lmax if lmax is not None else self.lmax

        cls = compute_cross_power_spectrum(map1, map2, lmax=_lmax)

        # Apply ell range selection
        if self.ell_min is not None or self.ell_max is not None:
            ell = np.arange(len(cls))
            ell_min = self.ell_min if self.ell_min is not None else 0
            ell_max = self.ell_max if self.ell_max is not None else len(cls) - 1
            mask = (ell >= ell_min) & (ell <= ell_max)
            cls = cls[mask]

        return cls

    def process_all_cross_spectra(
        self,
        maps: List[np.ndarray],
        include_auto: bool = True,
        lmax: Optional[int] = None,
    ) -> dict:
        """
        Compute all auto and cross power spectra for multiple maps.

        Parameters
        ----------
        maps : List[np.ndarray]
            List of HEALPix maps.
        include_auto : bool, optional
            Whether to include auto power spectra (default: True).
        lmax : int, optional
            Maximum multipole.

        Returns
        -------
        dict
            Dictionary with keys (i, j) and power spectra as values.
            Auto-spectra have i==j, cross-spectra have i<j.
        """
        _lmax = lmax if lmax is not None else self.lmax
        n_maps = len(maps)

        # Compute all alms first (more efficient)
        alms = [hp.map2alm(m, lmax=_lmax) for m in maps]

        cls_dict = {}

        for i in range(n_maps):
            # Auto spectrum
            if include_auto:
                cls = hp.alm2cl(alms[i])
                cls_dict[(i, i)] = self._apply_ell_selection(cls)

            # Cross spectra
            for j in range(i + 1, n_maps):
                cls = hp.alm2cl(alms[i], alms[j])
                cls_dict[(i, j)] = self._apply_ell_selection(cls)

        return cls_dict

    def _apply_ell_selection(self, cls: np.ndarray) -> np.ndarray:
        """Apply ell range selection to a power spectrum."""
        if self.ell_min is None and self.ell_max is None:
            return cls

        ell = np.arange(len(cls))
        ell_min = self.ell_min if self.ell_min is not None else 0
        ell_max = self.ell_max if self.ell_max is not None else len(cls) - 1
        mask = (ell >= ell_min) & (ell <= ell_max)
        return cls[mask]

    def _bin_spectrum(
        self,
        cls: np.ndarray,
        ell: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin the power spectrum.

        Parameters
        ----------
        cls : np.ndarray
            Power spectrum values.
        ell : np.ndarray
            Multipole values.

        Returns
        -------
        binned_cls : np.ndarray
            Binned power spectrum.
        binned_ell : np.ndarray
            Bin centers.
        """
        n_bins = len(ell) // self.bin_width
        if n_bins == 0:
            return cls, ell

        # Trim to exact multiple of bin_width
        trim_len = n_bins * self.bin_width
        cls_trim = cls[:trim_len]
        ell_trim = ell[:trim_len]

        # Reshape and average
        binned_cls = cls_trim.reshape(n_bins, self.bin_width).mean(axis=1)
        binned_ell = ell_trim.reshape(n_bins, self.bin_width).mean(axis=1)

        return binned_cls, binned_ell

    def get_output_suffix(
        self,
        bin_number: Optional[int] = None,
        bnt_bin: Optional[int] = None,
    ) -> str:
        """Generate output filename suffix."""
        parts = ["_cls"]

        if bnt_bin is not None:
            parts.append(f"_bnt{bnt_bin + 1}")
        elif bin_number is not None:
            parts.append(f"_bin{bin_number}")

        if self.config.apply_mask:
            area = int(round(self.config.mask_area_sqdeg))
            parts.append(f"_masked_{area}sqdeg")

        if self.config.add_noise:
            parts.append(f"_noisy_s{self.config.noise_level:.2f}")

        if self.lmax != DEFAULT_LMAX:
            parts.append(f"_lmax{self.lmax}")

        parts.append(".npy")
        return "".join(parts)


# Backwards compatibility - function-style interface
def process_power_spectrum(
    map_data: np.ndarray,
    lmax: int = DEFAULT_LMAX,
    add_noise: bool = False,
    noise_level: float = 0.26,
    **kwargs,
) -> np.ndarray:
    """
    Process a map to compute power spectrum (functional interface).

    This function provides a simple interface for computing power spectra
    without instantiating a processor object.

    Parameters
    ----------
    map_data : np.ndarray
        Input HEALPix convergence map.
    lmax : int, optional
        Maximum multipole.
    add_noise : bool, optional
        Whether to add shape noise.
    noise_level : float, optional
        Shape noise level (if adding noise).
    **kwargs
        Additional arguments passed to processor.

    Returns
    -------
    np.ndarray
        Angular power spectrum.
    """
    config = ProcessingConfig(add_noise=add_noise, noise_level=noise_level)
    processor = PowerSpectrumProcessor(config=config, lmax=lmax)
    dv = processor.process(map_data, apply_preprocessing=add_noise)
    return dv.data
