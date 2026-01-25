"""
Data aggregation utilities for cosmological analysis.

This module provides classes and functions for aggregating processed results
from multiple simulations, files, or realizations.
"""

from __future__ import annotations

import os
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any


__all__ = [
    "ResultsAggregator",
    "AggregationConfig",
    "aggregate_results",
    "aggregate_l1_norms",
    "aggregate_power_spectra",
    "load_datavectors",
]


@dataclass
class AggregationConfig:
    """
    Configuration for data aggregation.
    
    Parameters
    ----------
    filter_nans : bool
        Whether to filter out NaN values.
    filter_infs : bool
        Whether to filter out infinite values.
    filter_zeros : bool
        Whether to filter out all-zero entries.
    compute_statistics : bool
        Whether to compute mean/std statistics.
    verbose : bool
        Whether to print progress information.
    """
    
    filter_nans: bool = True
    filter_infs: bool = True
    filter_zeros: bool = False
    compute_statistics: bool = True
    verbose: bool = True


class ResultsAggregator:
    """
    Aggregator for processing results from multiple files.
    
    This class provides methods for loading and aggregating data vectors,
    power spectra, and other summary statistics from multiple files.
    
    Parameters
    ----------
    config : AggregationConfig, optional
        Configuration for aggregation.
        
    Examples
    --------
    >>> from bar_impact.analysis import ResultsAggregator
    >>> 
    >>> # Load L1 norms from multiple files
    >>> aggregator = ResultsAggregator()
    >>> data = aggregator.load_from_pattern("outputs/l1_norms_*.npy")
    >>> print(f"Loaded {data.shape[0]} samples")
    >>> 
    >>> # Load and combine multiple bins
    >>> data = aggregator.load_multi_bin(
    ...     ["bin1_l1.npy", "bin2_l1.npy"],
    ...     axis=1
    ... )
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config if config is not None else AggregationConfig()
        self._loaded_data = {}
    
    def load_from_pattern(
        self,
        pattern: str,
        sort: bool = True,
    ) -> np.ndarray:
        """
        Load and concatenate data from files matching a glob pattern.
        
        Parameters
        ----------
        pattern : str
            Glob pattern to match files (e.g., "data/*.npy").
        sort : bool
            Whether to sort files before loading.
            
        Returns
        -------
        np.ndarray
            Concatenated data from all matching files.
        """
        files = glob.glob(pattern)
        if sort:
            files = sorted(files)
        
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        if self.config.verbose:
            print(f"Found {len(files)} files matching pattern")
        
        return self.load_from_files(files)
    
    def load_from_files(
        self,
        file_paths: List[Union[str, Path]],
        axis: int = 0,
    ) -> np.ndarray:
        """
        Load and concatenate data from a list of files.
        
        Parameters
        ----------
        file_paths : list
            List of file paths to load.
        axis : int
            Axis along which to concatenate.
            
        Returns
        -------
        np.ndarray
            Concatenated data.
        """
        arrays = []
        
        for path in file_paths:
            data = np.load(path, allow_pickle=True)
            
            # Apply filters
            if self.config.filter_nans:
                valid_mask = ~np.any(np.isnan(data), axis=-1 if data.ndim > 1 else None)
                if data.ndim > 1:
                    data = data[valid_mask]
            
            if self.config.filter_infs:
                valid_mask = ~np.any(np.isinf(data), axis=-1 if data.ndim > 1 else None)
                if data.ndim > 1:
                    data = data[valid_mask]
            
            arrays.append(data)
            
            if self.config.verbose:
                print(f"Loaded {path}: shape {data.shape}")
        
        result = np.concatenate(arrays, axis=axis)
        
        if self.config.verbose:
            print(f"Combined shape: {result.shape}")
        
        return result
    
    def load_multi_bin(
        self,
        file_paths: List[Union[str, Path]],
        axis: int = 1,
    ) -> np.ndarray:
        """
        Load data from multiple redshift bins and concatenate.
        
        Parameters
        ----------
        file_paths : list
            List of file paths, one per bin.
        axis : int
            Axis along which to concatenate bins (default: 1 for features).
            
        Returns
        -------
        np.ndarray
            Combined data with all bins.
        """
        arrays = []
        
        for path in file_paths:
            data = np.load(path, allow_pickle=True)
            arrays.append(data)
            
            if self.config.verbose:
                print(f"Loaded bin data from {path}: shape {data.shape}")
        
        # Ensure all arrays have same shape on non-concat axis
        result = np.concatenate(arrays, axis=axis)
        
        if self.config.verbose:
            print(f"Combined multi-bin shape: {result.shape}")
        
        return result
    
    def load_with_parameters(
        self,
        data_path: Union[str, Path],
        params_path: Union[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data vectors and corresponding parameters.
        
        Parameters
        ----------
        data_path : str or Path
            Path to data vectors file.
        params_path : str or Path
            Path to parameters file.
            
        Returns
        -------
        data : np.ndarray
            Data vectors.
        params : np.ndarray
            Parameter values.
        """
        data = np.load(data_path, allow_pickle=True)
        params = np.load(params_path, allow_pickle=True)
        
        if len(data) != len(params):
            raise ValueError(
                f"Data and params have different lengths: {len(data)} vs {len(params)}"
            )
        
        if self.config.verbose:
            print(f"Loaded data: {data.shape}, params: {params.shape}")
        
        return data, params
    
    def filter_by_mask(
        self,
        data: np.ndarray,
        params: Optional[np.ndarray] = None,
        valid_indices: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Filter data and parameters by a mask or index array.
        
        Parameters
        ----------
        data : np.ndarray
            Data to filter.
        params : np.ndarray, optional
            Parameters to filter (if provided).
        valid_indices : np.ndarray, optional
            Indices to keep.
        mask : np.ndarray, optional
            Boolean mask (True = keep).
            
        Returns
        -------
        filtered_data : np.ndarray
            Filtered data.
        filtered_params : np.ndarray or None
            Filtered parameters (if provided).
        """
        if valid_indices is not None:
            data = data[valid_indices]
            if params is not None:
                params = params[valid_indices]
        elif mask is not None:
            data = data[mask]
            if params is not None:
                params = params[mask]
        
        return data, params
    
    def compute_statistics(
        self,
        data: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute summary statistics for data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data, shape (n_samples, n_features).
            
        Returns
        -------
        dict
            Dictionary with 'mean', 'std', 'median', 'min', 'max'.
        """
        return {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "median": np.median(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
        }
    
    def select_scales(
        self,
        data: np.ndarray,
        scale_indices: List[int],
        nbins_per_scale: int = 40,
    ) -> np.ndarray:
        """
        Select specific wavelet scales from L1 norm data.
        
        Parameters
        ----------
        data : np.ndarray
            L1 norm data, shape (n_samples, n_scales, nbins).
        scale_indices : list of int
            Indices of scales to select (0-indexed).
        nbins_per_scale : int
            Number of bins per scale.
            
        Returns
        -------
        np.ndarray
            Selected and flattened data.
        """
        if data.ndim == 2:
            # Data is already flattened, reshape first
            n_samples = data.shape[0]
            n_total = data.shape[1]
            n_scales = n_total // nbins_per_scale
            data = data.reshape(n_samples, n_scales, nbins_per_scale)
        
        selected = data[:, scale_indices, :]
        return selected.reshape(selected.shape[0], -1)


# Functional interface for backwards compatibility

def aggregate_results(
    file_pattern: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Aggregate processed results from multiple files.
    
    Parameters
    ----------
    file_pattern : str
        Glob pattern to match input files.
    output_path : str, optional
        Path to save aggregated results.
    **kwargs
        Additional options passed to AggregationConfig.
        
    Returns
    -------
    dict
        Dictionary containing aggregated data and statistics.
    """
    config = AggregationConfig(**{k: v for k, v in kwargs.items() if hasattr(AggregationConfig, k)})
    aggregator = ResultsAggregator(config=config)
    
    data = aggregator.load_from_pattern(file_pattern)
    
    result = {"data": data}
    if config.compute_statistics:
        result.update(aggregator.compute_statistics(data))
    
    if output_path:
        np.savez(output_path, **result)
    
    return result


def aggregate_l1_norms(
    file_paths: List[str],
    scale_indices: Optional[List[int]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Aggregate L1 norm results from multiple files.
    
    Parameters
    ----------
    file_paths : list of str
        Paths to L1 norm files.
    scale_indices : list of int, optional
        Scales to select. If None, uses all scales.
    **kwargs
        Additional options.
        
    Returns
    -------
    np.ndarray
        Aggregated L1 norms.
    """
    config = AggregationConfig(**{k: v for k, v in kwargs.items() if hasattr(AggregationConfig, k)})
    aggregator = ResultsAggregator(config=config)
    
    data = aggregator.load_from_files(file_paths, axis=0)
    
    if scale_indices is not None:
        data = aggregator.select_scales(data, scale_indices)
    
    return data


def aggregate_power_spectra(
    file_paths: List[str],
    ell_range: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Aggregate power spectrum results.
    
    Parameters
    ----------
    file_paths : list of str
        Paths to power spectrum files.
    ell_range : tuple of int, optional
        (ell_min, ell_max) to select.
    **kwargs
        Aggregation options.
        
    Returns
    -------
    dict
        Aggregated power spectra with 'cls' and optional 'ell' keys.
    """
    config = AggregationConfig(**{k: v for k, v in kwargs.items() if hasattr(AggregationConfig, k)})
    aggregator = ResultsAggregator(config=config)
    
    cls = aggregator.load_from_files(file_paths, axis=0)
    
    if ell_range is not None:
        ell_min, ell_max = ell_range
        cls = cls[..., ell_min:ell_max+1]
        ell = np.arange(ell_min, ell_max + 1)
    else:
        ell = np.arange(cls.shape[-1])
    
    result = {"cls": cls, "ell": ell}
    
    if config.compute_statistics:
        result["cls_mean"] = np.mean(cls, axis=0)
        result["cls_std"] = np.std(cls, axis=0)
    
    return result


def load_datavectors(
    data_path: Union[str, Path],
    params_path: Union[str, Path],
    filter_invalid: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data vectors and parameters for NPE training.
    
    Parameters
    ----------
    data_path : str or Path
        Path to data vectors.
    params_path : str or Path
        Path to parameters.
    filter_invalid : bool
        Whether to filter out invalid (NaN/Inf) entries.
        
    Returns
    -------
    data : np.ndarray
        Data vectors.
    params : np.ndarray
        Parameters.
    """
    config = AggregationConfig(filter_nans=filter_invalid, filter_infs=filter_invalid)
    aggregator = ResultsAggregator(config=config)
    return aggregator.load_with_parameters(data_path, params_path)
