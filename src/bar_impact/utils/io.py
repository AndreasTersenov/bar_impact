"""
Input/Output utilities for BAR_IMPACT.

This module provides functions for loading and saving data files,
including HEALPix maps, HDF5 files, and numpy arrays.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import healpy as hp
import numpy as np


def load_healpy_map(filepath: Union[str, Path], field: int = 0, **kwargs) -> np.ndarray:
    """
    Load a HEALPix map from file.

    Parameters
    ----------
    filepath : str or Path
        Path to the HEALPix map file
    field : int, optional
        Field index to load (default: 0)
    **kwargs
        Additional arguments passed to healpy.read_map

    Returns
    -------
    np.ndarray
        HEALPix map array

    Examples
    --------
    >>> kappa_map = load_healpy_map('convergence_map.fits')
    >>> kappa_map.shape
    (3145728,)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Map file not found: {filepath}")

    return hp.read_map(str(filepath), field=field, **kwargs)


def save_results(
    data: Union[np.ndarray, Dict[str, Any]],
    filepath: Union[str, Path],
    format: str = "npz",
    **kwargs,
) -> None:
    """
    Save analysis results to file.

    Parameters
    ----------
    data : np.ndarray or dict
        Data to save
    filepath : str or Path
        Output file path
    format : str, optional
        File format: 'npz', 'npy', 'hdf5', or 'fits'
        (default: 'npz')
    **kwargs
        Additional format-specific arguments

    Examples
    --------
    >>> results = {'l1_norms': np.array([1, 2, 3]), 'scales': [1, 2, 3, 4]}
    >>> save_results(results, 'output.npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            np.savez(filepath, data=data)
    elif format == "npy":
        if isinstance(data, dict):
            raise ValueError("Cannot save dict to .npy format, use 'npz' instead")
        np.save(filepath, data)
    elif format == "hdf5":
        _save_hdf5(filepath, data, **kwargs)
    elif format == "fits":
        if isinstance(data, np.ndarray):
            hp.write_map(str(filepath), data, **kwargs)
        else:
            raise ValueError("FITS format only supports numpy arrays")
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_hdf5(
    filepath: Path, data: Union[np.ndarray, Dict[str, Any]], **kwargs
) -> None:
    """Save data to HDF5 file."""
    with h5py.File(filepath, "w") as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.create_dataset(key, data=value)
        else:
            f.create_dataset("data", data=data)


def load_results(
    filepath: Union[str, Path], format: Optional[str] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load analysis results from file.

    Parameters
    ----------
    filepath : str or Path
        Path to results file
    format : str, optional
        File format (auto-detected from extension if not provided)

    Returns
    -------
    np.ndarray or dict
        Loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if format is None:
        format = filepath.suffix.lstrip(".")

    if format == "npz":
        loaded = np.load(filepath)
        return {key: loaded[key] for key in loaded.files}
    elif format == "npy":
        return np.load(filepath)
    elif format in ["hdf5", "h5"]:
        return _load_hdf5(filepath)
    elif format == "fits":
        return hp.read_map(str(filepath))
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_hdf5(filepath: Path) -> Dict[str, np.ndarray]:
    """Load data from HDF5 file."""
    data = {}
    with h5py.File(filepath, "r") as f:
        for key in f:
            data[key] = f[key][()]
    return data


def find_files(pattern: str, base_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """
    Find files matching a glob pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern (e.g., '*.fits', 'sim_*/kappa*.h5')
    base_dir : str or Path, optional
        Base directory to search in (default: current directory)

    Returns
    -------
    list of Path
        List of matching file paths

    Examples
    --------
    >>> files = find_files('*_nobaryons_*.fits', base_dir='/data/sims')
    >>> len(files)
    1000
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    return sorted(base_dir.glob(pattern))
