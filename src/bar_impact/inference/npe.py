"""
Neural Posterior Estimation for cosmological inference.

This module provides classes and functions for training and running NPE
using the jaxili library for simulation-based inference on cosmological data.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union, List
from pathlib import Path


__all__ = [
    "NPEInference",
    "NPEConfig", 
    "NPEResult",
    "run_npe_inference",
    "train_npe_model",
    "sample_posterior",
]


def _check_jaxili_available():
    """Check if jaxili is available."""
    try:
        from jaxili.inference import NPE
        return True
    except ImportError:
        return False


def _check_jax_available():
    """Check if JAX is available."""
    try:
        import jax
        import jax.numpy as jnp
        return True
    except ImportError:
        return False


@dataclass
class NPEConfig:
    """
    Configuration for Neural Posterior Estimation.
    
    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for optimization.
    batch_size : int
        Training batch size.
    checkpoint_dir : str or Path
        Directory for saving model checkpoints.
    checkpoint_name : str, optional
        Name for the checkpoint. Auto-generated if not provided.
    gpu_id : str
        GPU index to use (e.g., "0", "1").
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print training progress.
        
    Examples
    --------
    >>> config = NPEConfig(
    ...     num_epochs=1000,
    ...     learning_rate=1e-4,
    ...     batch_size=40,
    ...     checkpoint_dir="./checkpoints"
    ... )
    """
    
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 40
    checkpoint_dir: Union[str, Path] = "./checkpoints"
    checkpoint_name: Optional[str] = None
    gpu_id: str = "0"
    random_seed: int = 42
    verbose: bool = True


@dataclass
class NPEResult:
    """
    Container for NPE inference results.
    
    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (num_samples, num_params).
    param_names : list of str
        Names of the parameters.
    observed_data : np.ndarray
        The observed data used for inference.
    metadata : dict
        Additional metadata about the inference.
        
    Attributes
    ----------
    num_samples : int
        Number of posterior samples.
    num_params : int
        Number of parameters.
    """
    
    samples: np.ndarray
    param_names: List[str] = field(default_factory=list)
    observed_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_samples(self) -> int:
        """Number of posterior samples."""
        return self.samples.shape[0]
    
    @property
    def num_params(self) -> int:
        """Number of parameters."""
        return self.samples.shape[1]
    
    def get_param_samples(self, param_name: str) -> np.ndarray:
        """
        Get samples for a specific parameter.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter.
            
        Returns
        -------
        np.ndarray
            Samples for the specified parameter.
        """
        if param_name not in self.param_names:
            raise ValueError(f"Unknown parameter: {param_name}. Available: {self.param_names}")
        idx = self.param_names.index(param_name)
        return self.samples[:, idx]
    
    def get_mean(self) -> np.ndarray:
        """Get posterior mean for each parameter."""
        return np.mean(self.samples, axis=0)
    
    def get_std(self) -> np.ndarray:
        """Get posterior standard deviation for each parameter."""
        return np.std(self.samples, axis=0)
    
    def get_quantiles(self, q: Union[float, List[float]] = [0.16, 0.5, 0.84]) -> np.ndarray:
        """
        Get posterior quantiles.
        
        Parameters
        ----------
        q : float or list of float
            Quantile(s) to compute.
            
        Returns
        -------
        np.ndarray
            Quantiles, shape (len(q), num_params).
        """
        return np.quantile(self.samples, q, axis=0)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all parameters.
        
        Returns
        -------
        dict
            Dictionary with parameter names as keys and statistics as values.
        """
        quantiles = self.get_quantiles([0.16, 0.5, 0.84])
        summary = {}
        for i, name in enumerate(self.param_names):
            summary[name] = {
                "mean": float(np.mean(self.samples[:, i])),
                "std": float(np.std(self.samples[:, i])),
                "median": float(quantiles[1, i]),
                "lower_68": float(quantiles[0, i]),
                "upper_68": float(quantiles[2, i]),
            }
        return summary
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save results to a numpy file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the results.
        """
        np.savez(
            filepath,
            samples=self.samples,
            param_names=self.param_names,
            observed_data=self.observed_data,
            metadata=self.metadata,
        )
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NPEResult":
        """
        Load results from a numpy file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to load results from.
            
        Returns
        -------
        NPEResult
            Loaded results.
        """
        data = np.load(filepath, allow_pickle=True)
        return cls(
            samples=data["samples"],
            param_names=list(data["param_names"]),
            observed_data=data["observed_data"] if "observed_data" in data else None,
            metadata=data["metadata"].item() if "metadata" in data else {},
        )


class NPEInference:
    """
    Neural Posterior Estimation inference wrapper.
    
    This class wraps the jaxili NPE implementation for easy use in
    cosmological inference pipelines.
    
    Parameters
    ----------
    config : NPEConfig, optional
        Configuration for the inference. Uses defaults if not provided.
    param_names : list of str, optional
        Names of the cosmological parameters.
        
    Attributes
    ----------
    config : NPEConfig
        Configuration settings.
    is_trained : bool
        Whether the model has been trained.
    jaxili_available : bool
        Whether jaxili is installed.
        
    Examples
    --------
    >>> from bar_impact.inference import NPEInference, NPEConfig
    >>> from bar_impact.constants import COSMO_PARAM_NAMES
    >>> 
    >>> # Create inference object
    >>> config = NPEConfig(num_epochs=500, batch_size=64)
    >>> npe = NPEInference(config=config, param_names=COSMO_PARAM_NAMES)
    >>> 
    >>> # Train on simulation data
    >>> npe.train(data_vectors, parameters)
    >>> 
    >>> # Sample posterior for observed data
    >>> result = npe.sample(observed_data, num_samples=10000)
    >>> print(result.summary())
    
    Notes
    -----
    This class requires jaxili to be installed. Install via:
    pip install jaxili
    """
    
    def __init__(
        self,
        config: Optional[NPEConfig] = None,
        param_names: Optional[List[str]] = None,
    ):
        self.config = config if config is not None else NPEConfig()
        self.param_names = param_names or []
        
        self._inference = None
        self._posterior = None
        self._is_trained = False
        
        # Check dependencies
        self.jaxili_available = _check_jaxili_available()
        self.jax_available = _check_jax_available()
        
        # Set GPU
        if self.jax_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id
    
    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained
    
    def _ensure_jaxili(self):
        """Ensure jaxili is available."""
        if not self.jaxili_available:
            raise ImportError(
                "jaxili is required for NPE inference. "
                "Install via: pip install jaxili"
            )
    
    def _to_jax_array(self, arr: np.ndarray):
        """Convert numpy array to JAX array."""
        import jax.numpy as jnp
        return jnp.array(arr)
    
    def train(
        self,
        data_vectors: np.ndarray,
        parameters: np.ndarray,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the NPE model on simulation data.
        
        Parameters
        ----------
        data_vectors : np.ndarray
            Training data vectors, shape (n_sims, n_features).
        parameters : np.ndarray
            Training parameters, shape (n_sims, n_params).
        checkpoint_name : str, optional
            Name for checkpoint. Overrides config if provided.
            
        Returns
        -------
        dict
            Training metrics (loss history, etc.).
            
        Raises
        ------
        ImportError
            If jaxili is not installed.
        """
        self._ensure_jaxili()
        from jaxili.inference import NPE
        
        # Convert to JAX arrays
        params_jax = self._to_jax_array(parameters)
        data_jax = self._to_jax_array(data_vectors)
        
        # Set up checkpoint path
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        name = checkpoint_name or self.config.checkpoint_name
        if name is None:
            name = f"npe_checkpoint_{data_vectors.shape[1]}features"
        checkpoint_path = str(checkpoint_dir / name)
        
        if self.config.verbose:
            print(f"Training NPE with {data_vectors.shape[0]} simulations, "
                  f"{data_vectors.shape[1]} features, {parameters.shape[1]} parameters")
            print(f"Checkpoint path: {checkpoint_path}")
        
        # Initialize and train
        self._inference = NPE()
        self._inference = self._inference.append_simulations(params_jax, data_jax)
        
        if self.config.verbose:
            print(f"Starting training for {self.config.num_epochs} epochs...")
        
        metrics, _ = self._inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            training_batch_size=self.config.batch_size,
        )
        
        # Build posterior
        self._posterior = self._inference.build_posterior()
        self._is_trained = True
        
        if self.config.verbose:
            print("Training completed successfully")
        
        return {"metrics": metrics, "checkpoint_path": checkpoint_path}
    
    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a trained model from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint directory.
        """
        self._ensure_jaxili()
        from jaxili.inference import NPE
        
        self._inference = NPE()
        self._inference.load(str(checkpoint_path))
        self._posterior = self._inference.build_posterior()
        self._is_trained = True
        
        if self.config.verbose:
            print(f"Loaded model from {checkpoint_path}")
    
    def sample(
        self,
        observed_data: np.ndarray,
        num_samples: int = 10000,
        seed: Optional[int] = None,
    ) -> NPEResult:
        """
        Sample from the posterior given observed data.
        
        Parameters
        ----------
        observed_data : np.ndarray
            Observed data vector, shape (n_features,).
        num_samples : int, optional
            Number of posterior samples to draw.
        seed : int, optional
            Random seed for sampling.
            
        Returns
        -------
        NPEResult
            Container with posterior samples and metadata.
            
        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before sampling. Call train() or load() first.")
        
        import jax.random as random
        
        seed = seed if seed is not None else self.config.random_seed
        key = random.PRNGKey(seed)
        
        samples = self._posterior.sample(
            x=observed_data,
            num_samples=num_samples,
            key=key,
        )
        
        return NPEResult(
            samples=np.array(samples),
            param_names=self.param_names,
            observed_data=observed_data,
            metadata={
                "num_samples": num_samples,
                "seed": seed,
            },
        )
    
    def sample_batch(
        self,
        observed_data_batch: np.ndarray,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample from posterior for multiple observations.
        
        Parameters
        ----------
        observed_data_batch : np.ndarray
            Batch of observed data, shape (n_obs, n_features).
        num_samples : int
            Number of samples per observation.
        seed : int, optional
            Random seed.
            
        Returns
        -------
        np.ndarray
            Samples, shape (num_samples, n_obs, n_params).
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before sampling.")
        
        import jax
        import jax.random as random
        
        seed = seed if seed is not None else self.config.random_seed
        master_key = random.PRNGKey(seed)
        
        all_samples = []
        for i, obs in enumerate(observed_data_batch):
            sample_key, master_key = jax.random.split(master_key)
            samples = self._posterior.sample(
                x=obs,
                num_samples=num_samples,
                key=sample_key,
            )
            all_samples.append(np.array(samples))
        
        # Stack: (num_samples, n_obs, n_params)
        return np.stack(all_samples, axis=1)


# Functional interface for backwards compatibility

def run_npe_inference(
    data_vectors: np.ndarray,
    parameters: np.ndarray,
    observed_data: Optional[np.ndarray] = None,
    num_samples: int = 10000,
    param_names: Optional[List[str]] = None,
    config: Optional[NPEConfig] = None,
    **kwargs,
) -> Union[NPEResult, Dict[str, Any]]:
    """
    Run Neural Posterior Estimation on cosmological data.
    
    This is a convenience function that trains an NPE model and optionally
    samples from the posterior.
    
    Parameters
    ----------
    data_vectors : np.ndarray
        Training data vectors from simulations, shape (n_sims, n_features).
    parameters : np.ndarray
        Corresponding cosmological parameters, shape (n_sims, n_params).
    observed_data : np.ndarray, optional
        Observed data to infer parameters for. If None, only training is done.
    num_samples : int, optional
        Number of posterior samples to draw.
    param_names : list of str, optional
        Names of the parameters.
    config : NPEConfig, optional
        Configuration for the inference.
    **kwargs
        Additional keyword arguments passed to NPEConfig.
        
    Returns
    -------
    NPEResult or dict
        If observed_data is provided, returns NPEResult with posterior samples.
        Otherwise, returns dict with training info.
    """
    # Create config from kwargs if not provided
    if config is None:
        config = NPEConfig(**{k: v for k, v in kwargs.items() if hasattr(NPEConfig, k)})
    
    # Create and train
    npe = NPEInference(config=config, param_names=param_names)
    train_result = npe.train(data_vectors, parameters)
    
    # Sample if observed data provided
    if observed_data is not None:
        result = npe.sample(observed_data, num_samples=num_samples)
        result.metadata.update(train_result)
        return result
    
    return train_result


def train_npe_model(
    data_vectors: np.ndarray,
    parameters: np.ndarray,
    config: Optional[NPEConfig] = None,
    **kwargs,
) -> NPEInference:
    """
    Train a Neural Posterior Estimation model.
    
    Parameters
    ----------
    data_vectors : np.ndarray
        Training data vectors.
    parameters : np.ndarray
        Training parameters.
    config : NPEConfig, optional
        Training configuration.
    **kwargs
        Additional arguments for NPEConfig.
        
    Returns
    -------
    NPEInference
        Trained NPE inference object.
    """
    if config is None:
        config = NPEConfig(**{k: v for k, v in kwargs.items() if hasattr(NPEConfig, k)})
    
    npe = NPEInference(config=config)
    npe.train(data_vectors, parameters)
    return npe


def sample_posterior(
    model: NPEInference,
    observed_data: np.ndarray,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from the posterior distribution.
    
    Parameters
    ----------
    model : NPEInference
        Trained NPE model.
    observed_data : np.ndarray
        Observed data.
    num_samples : int
        Number of samples to draw.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    np.ndarray
        Posterior samples.
    """
    result = model.sample(observed_data, num_samples=num_samples, seed=seed)
    return result.samples
