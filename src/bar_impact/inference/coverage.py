"""
Coverage testing for posterior validation using TARP.

This module provides classes and functions for assessing the quality of
posterior estimates using the TARP (Tests of Accuracy with Random Points)
method.

References
----------
Lemos, Coogan et al. 2023: https://arxiv.org/abs/2302.03026
"""

from __future__ import annotations

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
from pathlib import Path


__all__ = [
    "CoverageTester",
    "CoverageConfig",
    "CoverageResult",
    "compute_tarp_coverage",
]


def _check_tarp_available():
    """Check if TARP is available."""
    try:
        # Try local tarp first (in the repo)
        tarp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'tarp', 'src'
        )
        if tarp_path not in sys.path:
            sys.path.insert(0, tarp_path)
        from tarp import get_tarp_coverage
        return True
    except ImportError:
        try:
            # Try installed tarp
            from tarp import get_tarp_coverage
            return True
        except ImportError:
            return False


def _get_tarp_function():
    """Get the TARP coverage function."""
    try:
        tarp_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'tarp', 'src'
        )
        if tarp_path not in sys.path:
            sys.path.insert(0, tarp_path)
        from tarp import get_tarp_coverage
        return get_tarp_coverage
    except ImportError:
        from tarp import get_tarp_coverage
        return get_tarp_coverage


@dataclass
class CoverageConfig:
    """
    Configuration for coverage testing.
    
    Parameters
    ----------
    num_sims : int
        Number of simulations to use for coverage testing.
    num_samples : int
        Number of posterior samples per simulation.
    references : str
        Reference point strategy ("random" or custom array).
    metric : str
        Distance metric ("euclidean" or "manhattan").
    bootstrap : bool
        Whether to use bootstrap for uncertainty estimation.
    num_bootstrap : int
        Number of bootstrap iterations.
    normalize : bool
        Whether to normalize parameter space.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
    """
    
    num_sims: int = 100
    num_samples: int = 1000
    references: str = "random"
    metric: str = "euclidean"
    bootstrap: bool = True
    num_bootstrap: int = 100
    normalize: bool = True
    seed: int = 42
    verbose: bool = True


@dataclass
class CoverageResult:
    """
    Container for coverage test results.
    
    Parameters
    ----------
    ecp : np.ndarray
        Expected coverage probability. Shape (n_bins+1,) without bootstrap,
        or (n_bootstrap, n_bins+1) with bootstrap.
    alpha : np.ndarray
        Credibility levels (bin edges).
    config : CoverageConfig
        Configuration used for the test.
    metadata : dict
        Additional metadata.
        
    Attributes
    ----------
    is_calibrated : bool
        Whether the posterior appears well-calibrated (within tolerance).
    """
    
    ecp: np.ndarray
    alpha: np.ndarray
    config: CoverageConfig = field(default_factory=CoverageConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def ecp_mean(self) -> np.ndarray:
        """Mean ECP (handles bootstrap case)."""
        if self.config.bootstrap and self.ecp.ndim == 2:
            return np.mean(self.ecp, axis=0)
        return self.ecp
    
    @property
    def ecp_std(self) -> Optional[np.ndarray]:
        """Standard deviation of ECP (only with bootstrap)."""
        if self.config.bootstrap and self.ecp.ndim == 2:
            return np.std(self.ecp, axis=0)
        return None
    
    def get_deviation_from_diagonal(self) -> float:
        """
        Compute mean absolute deviation from perfect calibration.
        
        Returns
        -------
        float
            Mean absolute deviation from the diagonal.
        """
        ecp = self.ecp_mean
        return float(np.mean(np.abs(ecp - self.alpha)))
    
    def is_calibrated(self, tolerance: float = 0.05) -> bool:
        """
        Check if posterior is well-calibrated.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed mean deviation from diagonal.
            
        Returns
        -------
        bool
            Whether calibration is within tolerance.
        """
        return self.get_deviation_from_diagonal() < tolerance
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of coverage test.
        
        Returns
        -------
        dict
            Summary statistics including calibration metrics.
        """
        summary = {
            "mean_deviation": self.get_deviation_from_diagonal(),
            "is_calibrated": self.is_calibrated(),
            "num_sims_used": self.metadata.get("num_sims_used", self.config.num_sims),
            "num_samples": self.config.num_samples,
        }
        
        if self.ecp_std is not None:
            summary["max_uncertainty"] = float(np.max(self.ecp_std))
            summary["mean_uncertainty"] = float(np.mean(self.ecp_std))
        
        return summary
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save coverage results to file.
        
        Parameters
        ----------
        filepath : str or Path
            Output filepath.
        """
        np.savez(
            filepath,
            ecp=self.ecp,
            alpha=self.alpha,
            bootstrap=self.config.bootstrap,
            metadata=self.metadata,
        )
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CoverageResult":
        """
        Load coverage results from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved results.
            
        Returns
        -------
        CoverageResult
            Loaded results.
        """
        data = np.load(filepath, allow_pickle=True)
        config = CoverageConfig(bootstrap=bool(data.get("bootstrap", False)))
        return cls(
            ecp=data["ecp"],
            alpha=data["alpha"],
            config=config,
            metadata=data["metadata"].item() if "metadata" in data else {},
        )


class CoverageTester:
    """
    Coverage tester using TARP method.
    
    This class provides methods for assessing posterior calibration using
    the TARP (Tests of Accuracy with Random Points) method.
    
    Parameters
    ----------
    config : CoverageConfig, optional
        Configuration for coverage testing.
        
    Attributes
    ----------
    config : CoverageConfig
        Current configuration.
    tarp_available : bool
        Whether TARP is available.
        
    Examples
    --------
    >>> from bar_impact.inference import CoverageTester, CoverageConfig
    >>> 
    >>> # Create tester
    >>> config = CoverageConfig(num_sims=100, num_samples=1000)
    >>> tester = CoverageTester(config=config)
    >>> 
    >>> # Run coverage test
    >>> result = tester.test(npe_model, data_vectors, parameters)
    >>> 
    >>> # Check results
    >>> print(f"Calibration deviation: {result.get_deviation_from_diagonal():.3f}")
    >>> print(f"Is calibrated: {result.is_calibrated()}")
    
    Notes
    -----
    This class uses the TARP package for coverage computation.
    
    References
    ----------
    Lemos, Coogan et al. 2023: https://arxiv.org/abs/2302.03026
    """
    
    def __init__(self, config: Optional[CoverageConfig] = None):
        self.config = config if config is not None else CoverageConfig()
        self.tarp_available = _check_tarp_available()
    
    def _ensure_tarp(self):
        """Ensure TARP is available."""
        if not self.tarp_available:
            raise ImportError(
                "TARP is required for coverage testing. "
                "Install via: pip install tarp"
            )
    
    def test(
        self,
        sample_fn: Callable[[np.ndarray, int], np.ndarray],
        data_vectors: np.ndarray,
        true_parameters: np.ndarray,
        test_indices: Optional[np.ndarray] = None,
    ) -> CoverageResult:
        """
        Run coverage test using a sampling function.
        
        Parameters
        ----------
        sample_fn : callable
            Function that takes (observation, num_samples) and returns samples.
        data_vectors : np.ndarray
            Data vectors, shape (n_sims, n_features).
        true_parameters : np.ndarray
            True parameter values, shape (n_sims, n_params).
        test_indices : np.ndarray, optional
            Indices of simulations to use. Random if not provided.
            
        Returns
        -------
        CoverageResult
            Coverage test results.
        """
        self._ensure_tarp()
        
        n_total = data_vectors.shape[0]
        n_test = min(self.config.num_sims, n_total)
        
        # Select test simulations
        if test_indices is None:
            np.random.seed(self.config.seed)
            test_indices = np.random.choice(n_total, size=n_test, replace=False)
        else:
            test_indices = test_indices[:n_test]
        
        test_data = data_vectors[test_indices]
        test_params = true_parameters[test_indices]
        
        if self.config.verbose:
            print(f"Running TARP coverage test with {n_test} simulations")
            print(f"Generating {self.config.num_samples} samples per simulation")
        
        # Generate posterior samples for each test simulation
        all_samples = []
        for i, x_obs in enumerate(test_data):
            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_test} simulations")
            
            samples = sample_fn(x_obs, self.config.num_samples)
            all_samples.append(samples)
        
        # Stack: (num_samples, n_sims, n_params)
        all_samples = np.stack(all_samples, axis=1)
        
        if self.config.verbose:
            print(f"Posterior samples shape: {all_samples.shape}")
            print("Computing TARP coverage...")
        
        # Compute TARP coverage
        get_tarp_coverage = _get_tarp_function()
        ecp, alpha = get_tarp_coverage(
            samples=all_samples,
            theta=test_params,
            references=self.config.references,
            metric=self.config.metric,
            num_alpha_bins=None,
            norm=self.config.normalize,
            bootstrap=self.config.bootstrap,
            num_bootstrap=self.config.num_bootstrap if self.config.bootstrap else 100,
            seed=self.config.seed,
        )
        
        if self.config.verbose:
            print("Coverage test completed!")
        
        return CoverageResult(
            ecp=ecp,
            alpha=alpha,
            config=self.config,
            metadata={
                "num_sims_used": n_test,
                "test_indices": test_indices,
            },
        )
    
    def test_npe(
        self,
        npe_model,
        data_vectors: np.ndarray,
        true_parameters: np.ndarray,
        test_indices: Optional[np.ndarray] = None,
    ) -> CoverageResult:
        """
        Run coverage test on an NPE model.
        
        Parameters
        ----------
        npe_model : NPEInference
            Trained NPE model with sample() method.
        data_vectors : np.ndarray
            Data vectors.
        true_parameters : np.ndarray
            True parameters.
        test_indices : np.ndarray, optional
            Indices to test.
            
        Returns
        -------
        CoverageResult
            Coverage results.
        """
        def sample_fn(x_obs, num_samples):
            result = npe_model.sample(x_obs, num_samples=num_samples)
            return result.samples
        
        return self.test(sample_fn, data_vectors, true_parameters, test_indices)
    
    def test_from_samples(
        self,
        samples: np.ndarray,
        true_parameters: np.ndarray,
    ) -> CoverageResult:
        """
        Run coverage test from pre-computed samples.
        
        Parameters
        ----------
        samples : np.ndarray
            Pre-computed posterior samples, shape (num_samples, n_sims, n_params).
        true_parameters : np.ndarray
            True parameter values, shape (n_sims, n_params).
            
        Returns
        -------
        CoverageResult
            Coverage results.
        """
        self._ensure_tarp()
        
        if self.config.verbose:
            print(f"Running TARP coverage from {samples.shape[1]} pre-computed samples")
        
        get_tarp_coverage = _get_tarp_function()
        ecp, alpha = get_tarp_coverage(
            samples=samples,
            theta=true_parameters,
            references=self.config.references,
            metric=self.config.metric,
            num_alpha_bins=None,
            norm=self.config.normalize,
            bootstrap=self.config.bootstrap,
            num_bootstrap=self.config.num_bootstrap if self.config.bootstrap else 100,
            seed=self.config.seed,
        )
        
        return CoverageResult(
            ecp=ecp,
            alpha=alpha,
            config=self.config,
            metadata={"num_sims_used": samples.shape[1]},
        )


def compute_tarp_coverage(
    samples: np.ndarray,
    true_parameters: np.ndarray,
    bootstrap: bool = True,
    num_bootstrap: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TARP coverage from posterior samples.
    
    This is a simple functional interface for TARP coverage computation.
    
    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (num_samples, n_sims, n_params).
    true_parameters : np.ndarray
        True parameter values, shape (n_sims, n_params).
    bootstrap : bool
        Whether to use bootstrap.
    num_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed.
        
    Returns
    -------
    ecp : np.ndarray
        Expected coverage probability.
    alpha : np.ndarray
        Credibility levels.
    """
    if not _check_tarp_available():
        raise ImportError("TARP is required. Install via: pip install tarp")
    
    get_tarp_coverage = _get_tarp_function()
    return get_tarp_coverage(
        samples=samples,
        theta=true_parameters,
        references="random",
        metric="euclidean",
        num_alpha_bins=None,
        norm=True,
        bootstrap=bootstrap,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
