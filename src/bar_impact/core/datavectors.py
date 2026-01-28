"""
Data vector containers for summary statistics.

This module provides the DataVector class for storing and manipulating
summary statistics used in simulation-based inference.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import json


__all__ = ["DataVector", "DataVectorCollection"]


@dataclass
class DataVector:
    """
    Container for summary statistics used in inference.
    
    A DataVector holds the summary statistics computed from one or more
    maps, along with metadata about how it was computed. This provides
    a unified interface for different types of summary statistics
    (L1 norms, power spectra, peak counts, etc.).
    
    Parameters
    ----------
    data : np.ndarray
        The summary statistic values. Shape depends on the statistic type.
    statistic_type : str
        Type of summary statistic: "l1_norm", "power_spectrum", 
        "peak_counts", "cross_power_spectrum", etc.
    cosmology_params : np.ndarray, optional
        Associated cosmological parameters.
    param_names : List[str], optional
        Names of the cosmological parameters.
    metadata : dict, optional
        Additional metadata (bin numbers, scales, noise level, etc.).
        
    Attributes
    ----------
    n_features : int
        Number of features (length of data vector).
        
    Examples
    --------
    >>> l1_norms = np.array([0.1, 0.15, 0.2, 0.25])  # 4 scales
    >>> dv = DataVector(
    ...     data=l1_norms,
    ...     statistic_type="l1_norm",
    ...     metadata={"scales": [0, 1, 2, 3], "bin_number": 2}
    ... )
    >>> dv.n_features
    4
    """
    
    data: np.ndarray
    statistic_type: str
    cosmology_params: Optional[np.ndarray] = None
    param_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize inputs."""
        self.data = np.atleast_1d(np.asarray(self.data, dtype=np.float64))
        
        if self.cosmology_params is not None:
            self.cosmology_params = np.atleast_1d(
                np.asarray(self.cosmology_params, dtype=np.float64)
            )
    
    @property
    def n_features(self) -> int:
        """Number of features in the data vector."""
        return self.data.size
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data array."""
        return self.data.shape
    
    def flatten(self) -> np.ndarray:
        """Return flattened data vector."""
        return self.data.ravel()
    
    def normalize(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        return_stats: bool = False,
    ) -> Union["DataVector", Tuple["DataVector", np.ndarray, np.ndarray]]:
        """
        Normalize the data vector to zero mean and unit variance.
        
        Parameters
        ----------
        mean : np.ndarray, optional
            Precomputed mean for each feature. If None, computed from data.
        std : np.ndarray, optional
            Precomputed std for each feature. If None, computed from data.
        return_stats : bool, optional
            If True, also return the mean and std used.
            
        Returns
        -------
        DataVector
            Normalized data vector.
        mean, std : np.ndarray
            Only if return_stats=True.
        """
        if mean is None:
            mean = np.mean(self.data)
        if std is None:
            std = np.std(self.data)
            # Avoid division by zero
            std = np.where(std < 1e-10, 1.0, std)
        
        normalized = DataVector(
            data=(self.data - mean) / std,
            statistic_type=self.statistic_type,
            cosmology_params=self.cosmology_params,
            param_names=self.param_names,
            metadata={**self.metadata, "normalized": True, "norm_mean": mean, "norm_std": std},
        )
        
        if return_stats:
            return normalized, mean, std
        return normalized
    
    def concatenate(self, other: "DataVector") -> "DataVector":
        """
        Concatenate with another data vector.
        
        Parameters
        ----------
        other : DataVector
            Another data vector to concatenate.
            
        Returns
        -------
        DataVector
            Concatenated data vector.
        """
        new_data = np.concatenate([self.data.ravel(), other.data.ravel()])
        
        # Combine metadata
        combined_metadata = {
            "concatenated_from": [self.statistic_type, other.statistic_type],
            "original_shapes": [self.shape, other.shape],
            **self.metadata,
        }
        
        return DataVector(
            data=new_data,
            statistic_type="combined",
            cosmology_params=self.cosmology_params,
            param_names=self.param_names,
            metadata=combined_metadata,
        )
    
    def select_features(
        self,
        indices: Union[List[int], np.ndarray, slice],
    ) -> "DataVector":
        """
        Select a subset of features.
        
        Parameters
        ----------
        indices : array-like or slice
            Indices of features to select.
            
        Returns
        -------
        DataVector
            Data vector with selected features.
        """
        selected_data = self.data.ravel()[indices]
        
        return DataVector(
            data=selected_data,
            statistic_type=self.statistic_type,
            cosmology_params=self.cosmology_params,
            param_names=self.param_names,
            metadata={**self.metadata, "selected_indices": indices},
        )
    
    def filter_zero_variance(
        self,
        min_variance: float = 1e-10,
    ) -> Tuple["DataVector", np.ndarray]:
        """
        Remove features with zero or near-zero variance.
        
        This is important for NPE training where zero-variance features
        can cause numerical issues.
        
        Parameters
        ----------
        min_variance : float, optional
            Minimum variance threshold.
            
        Returns
        -------
        DataVector
            Filtered data vector.
        valid_mask : np.ndarray
            Boolean mask indicating which features were kept.
            
        Notes
        -----
        This method is primarily useful when applied to DataVectorCollection
        where variance is computed across multiple simulations.
        """
        # For a single data vector, we can only check for constant values
        valid_mask = np.ones(self.n_features, dtype=bool)
        
        filtered = DataVector(
            data=self.data[valid_mask],
            statistic_type=self.statistic_type,
            cosmology_params=self.cosmology_params,
            param_names=self.param_names,
            metadata={**self.metadata, "variance_filtered": True},
        )
        
        return filtered, valid_mask
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "data": self.data.tolist(),
            "statistic_type": self.statistic_type,
            "metadata": self.metadata,
        }
        if self.cosmology_params is not None:
            result["cosmology_params"] = self.cosmology_params.tolist()
        if self.param_names is not None:
            result["param_names"] = self.param_names
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataVector":
        """Create from dictionary."""
        return cls(
            data=np.array(d["data"]),
            statistic_type=d["statistic_type"],
            cosmology_params=np.array(d["cosmology_params"]) if "cosmology_params" in d else None,
            param_names=d.get("param_names"),
            metadata=d.get("metadata", {}),
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save data vector to file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path. Format determined by extension:
            - .npz: NumPy compressed archive
            - .json: JSON format
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == ".npz":
            save_dict = {"data": self.data, "statistic_type": self.statistic_type}
            if self.cosmology_params is not None:
                save_dict["cosmology_params"] = self.cosmology_params
            if self.param_names is not None:
                save_dict["param_names"] = np.array(self.param_names, dtype=object)
            save_dict["metadata"] = np.array([json.dumps(self.metadata)])
            np.savez(filepath, **save_dict)
        elif filepath.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            # Default to npz
            self.save(filepath.with_suffix(".npz"))
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DataVector":
        """
        Load data vector from file.
        
        Parameters
        ----------
        filepath : str or Path
            Input file path.
            
        Returns
        -------
        DataVector
            Loaded data vector.
        """
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            with open(filepath) as f:
                return cls.from_dict(json.load(f))
        else:
            # Assume npz
            with np.load(filepath, allow_pickle=True) as data:
                metadata = {}
                if "metadata" in data:
                    metadata = json.loads(str(data["metadata"][0]))
                
                return cls(
                    data=data["data"],
                    statistic_type=str(data["statistic_type"]),
                    cosmology_params=data.get("cosmology_params"),
                    param_names=list(data["param_names"]) if "param_names" in data else None,
                    metadata=metadata,
                )
    
    def __repr__(self) -> str:
        shape_str = f"shape={self.shape}"
        return f"DataVector(type='{self.statistic_type}', {shape_str})"
    
    def __len__(self) -> int:
        return self.n_features


@dataclass
class DataVectorCollection:
    """
    Collection of data vectors from multiple simulations.
    
    This class is designed for storing training data for NPE, where
    each simulation produces a data vector and associated parameters.
    
    Parameters
    ----------
    data_vectors : np.ndarray
        Array of shape (n_simulations, n_features) containing all data vectors.
    parameters : np.ndarray
        Array of shape (n_simulations, n_params) containing cosmological parameters.
    statistic_type : str
        Type of summary statistic.
    param_names : List[str], optional
        Names of the cosmological parameters.
    metadata : dict, optional
        Additional metadata.
        
    Attributes
    ----------
    n_simulations : int
        Number of simulations in the collection.
    n_features : int
        Number of features per data vector.
    n_params : int
        Number of cosmological parameters.
        
    Examples
    --------
    >>> # Create from simulation results
    >>> data = np.random.randn(1000, 20)  # 1000 sims, 20 features
    >>> params = np.random.randn(1000, 6)  # 6 cosmological parameters
    >>> collection = DataVectorCollection(
    ...     data_vectors=data,
    ...     parameters=params,
    ...     statistic_type="l1_norm"
    ... )
    >>> collection.n_simulations
    1000
    """
    
    data_vectors: np.ndarray
    parameters: np.ndarray
    statistic_type: str
    param_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate inputs."""
        self.data_vectors = np.atleast_2d(np.asarray(self.data_vectors, dtype=np.float64))
        self.parameters = np.atleast_2d(np.asarray(self.parameters, dtype=np.float64))
        
        if self.data_vectors.shape[0] != self.parameters.shape[0]:
            raise ValueError(
                f"Number of data vectors ({self.data_vectors.shape[0]}) "
                f"does not match number of parameter sets ({self.parameters.shape[0]})"
            )
    
    @property
    def n_simulations(self) -> int:
        """Number of simulations."""
        return self.data_vectors.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features per data vector."""
        return self.data_vectors.shape[1]
    
    @property
    def n_params(self) -> int:
        """Number of cosmological parameters."""
        return self.parameters.shape[1]
    
    def filter_zero_variance(
        self,
        min_variance: float = 1e-10,
        verbose: bool = True,
    ) -> Tuple["DataVectorCollection", np.ndarray]:
        """
        Remove features with zero or near-zero variance across simulations.
        
        Parameters
        ----------
        min_variance : float, optional
            Minimum variance threshold.
        verbose : bool, optional
            Print information about filtering.
            
        Returns
        -------
        DataVectorCollection
            Filtered collection.
        valid_mask : np.ndarray
            Boolean mask indicating which features were kept.
        """
        variances = np.var(self.data_vectors, axis=0)
        valid_mask = variances > min_variance
        
        n_removed = np.sum(~valid_mask)
        
        if verbose and n_removed > 0:
            print(f"Zero-variance filtering:")
            print(f"  Removed {n_removed} / {self.n_features} features")
            print(f"  Remaining features: {np.sum(valid_mask)}")
        
        filtered = DataVectorCollection(
            data_vectors=self.data_vectors[:, valid_mask],
            parameters=self.parameters,
            statistic_type=self.statistic_type,
            param_names=self.param_names,
            metadata={
                **self.metadata,
                "variance_filtered": True,
                "n_removed_features": int(n_removed),
                "original_n_features": self.n_features,
            },
        )
        
        return filtered, valid_mask
    
    def normalize(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> Tuple["DataVectorCollection", np.ndarray, np.ndarray]:
        """
        Normalize data vectors to zero mean and unit variance.
        
        Parameters
        ----------
        mean : np.ndarray, optional
            Precomputed mean for each feature.
        std : np.ndarray, optional
            Precomputed std for each feature.
            
        Returns
        -------
        DataVectorCollection
            Normalized collection.
        mean : np.ndarray
            Mean used for normalization.
        std : np.ndarray
            Std used for normalization.
        """
        if mean is None:
            mean = np.mean(self.data_vectors, axis=0)
        if std is None:
            std = np.std(self.data_vectors, axis=0)
            std = np.where(std < 1e-10, 1.0, std)
        
        normalized = DataVectorCollection(
            data_vectors=(self.data_vectors - mean) / std,
            parameters=self.parameters,
            statistic_type=self.statistic_type,
            param_names=self.param_names,
            metadata={**self.metadata, "normalized": True},
        )
        
        return normalized, mean, std
    
    def train_test_split(
        self,
        test_fraction: float = 0.2,
        seed: Optional[int] = None,
    ) -> Tuple["DataVectorCollection", "DataVectorCollection"]:
        """
        Split into training and test sets.
        
        Parameters
        ----------
        test_fraction : float, optional
            Fraction of data to use for testing.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        train : DataVectorCollection
            Training set.
        test : DataVectorCollection
            Test set.
        """
        rng = np.random.RandomState(seed)
        n_test = int(self.n_simulations * test_fraction)
        indices = rng.permutation(self.n_simulations)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        train = DataVectorCollection(
            data_vectors=self.data_vectors[train_idx],
            parameters=self.parameters[train_idx],
            statistic_type=self.statistic_type,
            param_names=self.param_names,
            metadata={**self.metadata, "split": "train"},
        )
        
        test = DataVectorCollection(
            data_vectors=self.data_vectors[test_idx],
            parameters=self.parameters[test_idx],
            statistic_type=self.statistic_type,
            param_names=self.param_names,
            metadata={**self.metadata, "split": "test"},
        )
        
        return train, test
    
    def get_single(self, idx: int) -> DataVector:
        """
        Get a single data vector from the collection.
        
        Parameters
        ----------
        idx : int
            Index of the data vector.
            
        Returns
        -------
        DataVector
            Single data vector.
        """
        return DataVector(
            data=self.data_vectors[idx],
            statistic_type=self.statistic_type,
            cosmology_params=self.parameters[idx],
            param_names=self.param_names,
            metadata=self.metadata,
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save collection to file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path (.npz format).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "data_vectors": self.data_vectors,
            "parameters": self.parameters,
            "statistic_type": np.array([self.statistic_type]),
        }
        if self.param_names is not None:
            save_dict["param_names"] = np.array(self.param_names, dtype=object)
        save_dict["metadata"] = np.array([json.dumps(self.metadata)])
        
        np.savez(filepath, **save_dict)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DataVectorCollection":
        """
        Load collection from file.
        
        Parameters
        ----------
        filepath : str or Path
            Input file path.
            
        Returns
        -------
        DataVectorCollection
            Loaded collection.
        """
        filepath = Path(filepath)
        
        with np.load(filepath, allow_pickle=True) as data:
            metadata = {}
            if "metadata" in data:
                metadata = json.loads(str(data["metadata"][0]))
            
            return cls(
                data_vectors=data["data_vectors"],
                parameters=data["parameters"],
                statistic_type=str(data["statistic_type"][0]),
                param_names=list(data["param_names"]) if "param_names" in data else None,
                metadata=metadata,
            )
    
    def __len__(self) -> int:
        return self.n_simulations
    
    def __getitem__(self, idx: int) -> DataVector:
        return self.get_single(idx)
    
    def __repr__(self) -> str:
        return (
            f"DataVectorCollection(type='{self.statistic_type}', "
            f"n_sims={self.n_simulations}, n_features={self.n_features})"
        )
