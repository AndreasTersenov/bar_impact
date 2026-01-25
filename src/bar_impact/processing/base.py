"""
Base processor class for summary statistics computation.

This module provides the abstract base class that all summary statistic
processors inherit from, ensuring a consistent interface.
"""

from __future__ import annotations

import os
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from bar_impact.core.maps import ConvergenceMap, ConvergenceMapCollection
from bar_impact.core.masks import SurveyMask
from bar_impact.core.datavectors import DataVector, DataVectorCollection


__all__ = ["BaseProcessor", "ProcessingConfig"]


@dataclass
class ProcessingConfig:
    """
    Configuration for batch processing operations.
    
    Parameters
    ----------
    add_noise : bool
        Whether to add shape noise to maps.
    noise_level : float
        Shape noise level (sigma_e) if adding noise.
    galaxy_density : float
        Galaxy number density in arcmin^-2.
    apply_mask : bool
        Whether to apply a survey mask.
    mask_area_sqdeg : float
        Mask area in square degrees (if applying mask).
    mask_center : tuple
        Mask center coordinates (lon, lat) in degrees.
    n_workers : int
        Number of parallel workers for batch processing.
    verbose : bool
        Whether to print progress information.
    force_overwrite : bool
        Whether to overwrite existing output files.
    output_dir : str or Path, optional
        Directory for output files.
        
    Examples
    --------
    >>> config = ProcessingConfig(
    ...     add_noise=True,
    ...     noise_level=0.26,
    ...     apply_mask=True,
    ...     mask_area_sqdeg=14000.0
    ... )
    """
    
    add_noise: bool = True
    noise_level: float = 0.26
    galaxy_density: float = 6.75
    apply_mask: bool = False
    mask_area_sqdeg: float = 14000.0
    mask_center: tuple = (0.0, 90.0)
    n_workers: int = 1
    verbose: bool = False
    force_overwrite: bool = False
    output_dir: Optional[Union[str, Path]] = None
    random_seed: Optional[int] = None


class BaseProcessor(ABC):
    """
    Abstract base class for summary statistic processors.
    
    This class defines the interface that all processors must implement,
    and provides common functionality for batch processing.
    
    Parameters
    ----------
    config : ProcessingConfig, optional
        Configuration for processing. Uses defaults if not provided.
        
    Attributes
    ----------
    config : ProcessingConfig
        The processing configuration.
    statistic_type : str
        Name of the summary statistic (set by subclasses).
        
    Examples
    --------
    Subclasses must implement the `process_single` method:
    
    >>> class MyProcessor(BaseProcessor):
    ...     statistic_type = "my_statistic"
    ...     
    ...     def process_single(self, map_data):
    ...         return np.sum(np.abs(map_data))
    """
    
    # Subclasses must set this
    statistic_type: str = "base"
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._mask_cache: Dict[tuple, SurveyMask] = {}
    
    @abstractmethod
    def process_single(
        self,
        map_data: Union[np.ndarray, ConvergenceMap],
        **kwargs,
    ) -> np.ndarray:
        """
        Process a single map to compute the summary statistic.
        
        This method must be implemented by all subclasses.
        
        Parameters
        ----------
        map_data : np.ndarray or ConvergenceMap
            Input convergence map.
        **kwargs
            Additional processor-specific parameters.
            
        Returns
        -------
        np.ndarray
            Computed summary statistic.
        """
        pass
    
    def process(
        self,
        map_data: Union[np.ndarray, ConvergenceMap],
        apply_preprocessing: bool = True,
        **kwargs,
    ) -> DataVector:
        """
        Process a map with optional preprocessing (noise, masking).
        
        Parameters
        ----------
        map_data : np.ndarray or ConvergenceMap
            Input convergence map.
        apply_preprocessing : bool, optional
            Whether to apply noise and masking based on config.
        **kwargs
            Additional processor-specific parameters.
            
        Returns
        -------
        DataVector
            Computed summary statistic wrapped in a DataVector.
        """
        # Convert to ConvergenceMap if necessary
        if isinstance(map_data, np.ndarray):
            map_obj = ConvergenceMap(data=map_data)
        else:
            map_obj = map_data
        
        # Apply preprocessing if requested
        if apply_preprocessing:
            map_obj = self._preprocess(map_obj)
        
        # Compute statistic
        result = self.process_single(map_obj.data, **kwargs)
        
        # Wrap in DataVector
        metadata = {
            "nside": map_obj.nside,
            "is_noisy": map_obj.is_noisy,
            "noise_level": map_obj.noise_level,
            "bin_number": map_obj.bin_number,
        }
        if self.config.apply_mask:
            metadata["mask_area_sqdeg"] = self.config.mask_area_sqdeg
        
        return DataVector(
            data=result,
            statistic_type=self.statistic_type,
            metadata=metadata,
        )
    
    def process_collection(
        self,
        collection: ConvergenceMapCollection,
        apply_bnt: bool = False,
        bnt_matrix: Optional[np.ndarray] = None,
        concatenate: bool = True,
        **kwargs,
    ) -> Union[DataVector, List[DataVector]]:
        """
        Process a collection of maps (multiple redshift bins).
        
        Parameters
        ----------
        collection : ConvergenceMapCollection
            Collection of convergence maps.
        apply_bnt : bool, optional
            Whether to apply BNT transform before processing.
        bnt_matrix : np.ndarray, optional
            Custom BNT matrix (uses default if None).
        concatenate : bool, optional
            Whether to concatenate results into single DataVector.
        **kwargs
            Additional processor-specific parameters.
            
        Returns
        -------
        DataVector or List[DataVector]
            Computed statistics, concatenated if requested.
        """
        # Apply BNT if requested
        if apply_bnt:
            collection = collection.apply_bnt_transform(bnt_matrix=bnt_matrix)
        
        # Process each map
        results = []
        for kappa_map in collection:
            dv = self.process(kappa_map, apply_preprocessing=True, **kwargs)
            results.append(dv)
        
        if concatenate and len(results) > 1:
            # Concatenate all data vectors
            combined_data = np.concatenate([dv.data.ravel() for dv in results])
            return DataVector(
                data=combined_data,
                statistic_type=self.statistic_type,
                metadata={
                    "n_bins": len(results),
                    "original_shapes": [dv.shape for dv in results],
                    "is_bnt": apply_bnt,
                },
            )
        elif len(results) == 1:
            return results[0]
        else:
            return results
    
    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        bin_numbers: Union[int, List[int]] = 1,
        cosmology_params: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
        progress: bool = True,
        **kwargs,
    ) -> DataVectorCollection:
        """
        Process multiple files in parallel.
        
        Parameters
        ----------
        file_paths : List[str or Path]
            Paths to input HDF5 files.
        bin_numbers : int or List[int], optional
            Redshift bin number(s) to process.
        cosmology_params : np.ndarray, optional
            Cosmological parameters for each file, shape (n_files, n_params).
        param_names : List[str], optional
            Names of cosmological parameters.
        progress : bool, optional
            Whether to show progress bar.
        **kwargs
            Additional processor-specific parameters.
            
        Returns
        -------
        DataVectorCollection
            Collection of computed data vectors.
        """
        # Normalize bin_numbers to list
        if isinstance(bin_numbers, int):
            bin_numbers = [bin_numbers]
        
        n_files = len(file_paths)
        all_data_vectors = []
        
        # Process files
        if self.config.n_workers > 1:
            # Parallel processing
            all_data_vectors = self._process_batch_parallel(
                file_paths, bin_numbers, progress, **kwargs
            )
        else:
            # Sequential processing
            iterator = tqdm(file_paths, desc=f"Processing {self.statistic_type}") if progress else file_paths
            for filepath in iterator:
                try:
                    dv = self._process_file(filepath, bin_numbers, **kwargs)
                    if dv is not None:
                        all_data_vectors.append(dv)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Error processing {filepath}: {e}")
        
        if not all_data_vectors:
            raise ValueError("No files were successfully processed")
        
        # Stack into collection
        data_array = np.array([dv.data.ravel() for dv in all_data_vectors])
        
        # Use provided params or create placeholder
        if cosmology_params is None:
            cosmology_params = np.zeros((len(all_data_vectors), 1))
        
        return DataVectorCollection(
            data_vectors=data_array,
            parameters=cosmology_params[:len(all_data_vectors)],
            statistic_type=self.statistic_type,
            param_names=param_names,
            metadata={
                "n_bins": len(bin_numbers),
                "bin_numbers": bin_numbers,
                "config": {
                    "add_noise": self.config.add_noise,
                    "noise_level": self.config.noise_level,
                    "apply_mask": self.config.apply_mask,
                    "mask_area_sqdeg": self.config.mask_area_sqdeg,
                },
            },
        )
    
    def _process_file(
        self,
        filepath: Union[str, Path],
        bin_numbers: List[int],
        **kwargs,
    ) -> Optional[DataVector]:
        """
        Process a single file (internal method).
        
        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file.
        bin_numbers : List[int]
            Bin numbers to process.
        **kwargs
            Additional parameters.
            
        Returns
        -------
        DataVector or None
            Processed data vector, or None if processing failed.
        """
        try:
            if len(bin_numbers) == 1:
                # Single bin
                kappa = ConvergenceMap.from_h5(filepath, bin_number=bin_numbers[0])
                return self.process(kappa, apply_preprocessing=True, **kwargs)
            else:
                # Multiple bins - use collection
                collection = ConvergenceMapCollection.from_h5(
                    filepath, bin_numbers=bin_numbers
                )
                return self.process_collection(
                    collection, 
                    concatenate=True,
                    **kwargs
                )
        except Exception as e:
            if self.config.verbose:
                print(f"Error processing {filepath}: {e}")
            return None
    
    def _process_batch_parallel(
        self,
        file_paths: List[Union[str, Path]],
        bin_numbers: List[int],
        progress: bool,
        **kwargs,
    ) -> List[DataVector]:
        """
        Process batch in parallel (internal method).
        
        Note: Due to pickling constraints, this uses a simpler approach
        that may not preserve all processor state.
        """
        # For now, fall back to sequential processing
        # Full parallel support would require more careful handling
        results = []
        iterator = tqdm(file_paths, desc=f"Processing {self.statistic_type}") if progress else file_paths
        for filepath in iterator:
            try:
                dv = self._process_file(filepath, bin_numbers, **kwargs)
                if dv is not None:
                    results.append(dv)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error processing {filepath}: {e}")
        return results
    
    def _preprocess(self, map_obj: ConvergenceMap) -> ConvergenceMap:
        """
        Apply preprocessing (noise, masking) to a map.
        
        Parameters
        ----------
        map_obj : ConvergenceMap
            Input map.
            
        Returns
        -------
        ConvergenceMap
            Preprocessed map.
        """
        result = map_obj.copy()
        
        # Add noise if configured
        if self.config.add_noise and not result.is_noisy:
            result = result.add_shape_noise(
                sigma_e=self.config.noise_level,
                galaxy_density=self.config.galaxy_density,
                seed=self.config.random_seed,
            )
        
        # Apply mask if configured
        if self.config.apply_mask:
            mask = self._get_cached_mask(result.nside)
            result = result.apply_mask(mask)
        
        return result
    
    def _get_cached_mask(self, nside: int) -> SurveyMask:
        """Get or create a cached mask."""
        cache_key = (
            nside,
            self.config.mask_area_sqdeg,
            self.config.mask_center[0],
            self.config.mask_center[1],
        )
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = SurveyMask.create_disk_mask(
                nside=nside,
                target_area_sqdeg=self.config.mask_area_sqdeg,
                center_coords=self.config.mask_center,
            )
        return self._mask_cache[cache_key]
    
    def get_output_suffix(
        self,
        bin_number: Optional[int] = None,
        bnt_bin: Optional[int] = None,
    ) -> str:
        """
        Generate output filename suffix based on configuration.
        
        Parameters
        ----------
        bin_number : int, optional
            Redshift bin number.
        bnt_bin : int, optional
            BNT bin number.
            
        Returns
        -------
        str
            Filename suffix.
        """
        parts = [f"_{self.statistic_type}"]
        
        if bnt_bin is not None:
            parts.append(f"_bnt{bnt_bin+1}")
        elif bin_number is not None:
            parts.append(f"_bin{bin_number}")
        
        if self.config.apply_mask:
            area = int(round(self.config.mask_area_sqdeg))
            parts.append(f"_masked_{area}sqdeg")
        
        if self.config.add_noise:
            parts.append(f"_noisy_s{self.config.noise_level:.2f}")
        
        parts.append(".npy")
        return "".join(parts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(statistic_type='{self.statistic_type}')"
