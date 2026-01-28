"""
Inference utilities for Neural Posterior Estimation (NPE).

This module provides helper functions for NPE workflows, including
TARP coverage testing and training with NaN handling.
"""

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "run_tarp_coverage_test",
    "plot_tarp_coverage",
    "train_npe_with_nan_retry",
]


def run_tarp_coverage_test(
    posterior: Any,
    data: np.ndarray,
    params: np.ndarray,
    num_test_sims: int = 100,
    num_samples: int = 1000,
    seed: int = 42,
    bootstrap: bool = False,
    num_bootstrap: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TARP coverage test on posterior estimator.

    Uses TARP (Test of Accuracy with Random Points) to assess whether
    posterior coverage is well-calibrated by comparing expected coverage
    probability (ECP) against credibility levels.

    Parameters
    ----------
    posterior : object
        Trained posterior object from NPE.
    data : np.ndarray
        Full training data vector, shape (n_sims, n_features).
    params : np.ndarray
        True parameter values, shape (n_sims, n_params).
    num_test_sims : int
        Number of simulations to use for testing.
    num_samples : int
        Number of posterior samples per simulation.
    seed : int
        Random seed for reproducibility.
    bootstrap : bool
        Whether to use bootstrap for uncertainty estimation.
    num_bootstrap : int
        Number of bootstrap iterations.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    ecp : np.ndarray
        Expected coverage probability.
    alpha : np.ndarray
        Credibility levels.

    Notes
    -----
    Requires JAX and the TARP package.
    """
    try:
        import jax  # noqa: I001
        from jax import random
        from tarp import get_tarp_coverage
    except ImportError as e:
        raise ImportError(
            f"TARP coverage testing requires JAX and TARP package: {e}"
        ) from e

    if verbose:
        print("\n" + "=" * 60)
        print("Running TARP Coverage Test")
        print("=" * 60)

    # Select subset of simulations
    n_total = data.shape[0]
    n_test = min(num_test_sims, n_total)

    np.random.seed(seed)
    test_indices = np.random.choice(n_total, size=n_test, replace=False)

    if verbose:
        print(f"Using {n_test} simulations from training set")
        print(f"Generating {num_samples} posterior samples per simulation")

    test_data = data[test_indices]
    test_params = params[test_indices]

    # Generate posterior samples for each test simulation
    all_samples = []
    master_key = random.PRNGKey(seed)

    if verbose:
        print("Generating posterior samples...")

    for i, x_obs in enumerate(test_data):
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_test} simulations processed")

        sample_key, master_key = jax.random.split(master_key)
        samples = posterior.sample(x=x_obs, num_samples=num_samples, key=sample_key)
        all_samples.append(np.array(samples))

    # Stack samples: shape (n_samples, n_sims, n_params)
    all_samples = np.stack(all_samples, axis=1)

    if verbose:
        print(f"Posterior samples shape: {all_samples.shape}")
        print(f"True parameters shape: {test_params.shape}")
        print("\nComputing TARP coverage...")

    # Compute TARP coverage
    ecp, alpha = get_tarp_coverage(
        samples=all_samples,
        theta=test_params,
        references="random",
        metric="euclidean",
        num_alpha_bins=None,
        norm=True,
        bootstrap=bootstrap,
        num_bootstrap=num_bootstrap if bootstrap else 100,
        seed=seed,
    )

    if verbose:
        print("TARP coverage computation complete!")
        print("=" * 60 + "\n")

    return ecp, alpha


def plot_tarp_coverage(
    ecp: np.ndarray,
    alpha: np.ndarray,
    output_path: str,
    bootstrap: bool = False,
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 300,
) -> None:
    """
    Plot TARP coverage diagnostics.

    Parameters
    ----------
    ecp : np.ndarray
        Expected coverage probability from TARP.
        Can be 1D array or 2D with bootstrap samples.
    alpha : np.ndarray
        Credibility levels.
    output_path : str
        Path to save plot (without extension).
    bootstrap : bool
        Whether bootstrap uncertainties are included.
    figsize : tuple
        Figure size (width, height).
    dpi : int
        Resolution for saved figure.
    """
    plt.figure(figsize=figsize)

    if bootstrap and ecp.ndim == 2:
        # Plot mean with uncertainty band
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std = np.std(ecp, axis=0)
        plt.plot(alpha, ecp_mean, "b-", linewidth=2, label="TARP coverage")
        plt.fill_between(
            alpha,
            ecp_mean - ecp_std,
            ecp_mean + ecp_std,
            alpha=0.3,
            color="blue",
            label="Bootstrap uncertainty",
        )
    else:
        # Single run or mean already computed
        ecp_to_plot = np.mean(ecp, axis=0) if ecp.ndim == 2 else ecp
        plt.plot(alpha, ecp_to_plot, "b-", linewidth=2, label="TARP coverage")

    # Plot ideal calibration
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Ideal calibration")

    # Formatting
    plt.xlabel("Credibility Level", fontsize=12)
    plt.ylabel("Expected Coverage Probability", fontsize=12)
    plt.title("TARP Coverage Diagnostic", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    # Save plot
    plot_path = f"{output_path}_tarp_coverage.pdf"
    plt.savefig(plot_path, transparent=True, dpi=dpi)
    print(f"Saved TARP coverage plot to {plot_path}")
    plt.close()

    # Save coverage data
    data_path = f"{output_path}_tarp_coverage_data.npz"
    if bootstrap:
        np.savez(data_path, ecp=ecp, alpha=alpha, ecp_mean=np.mean(ecp, axis=0))
    else:
        np.savez(data_path, ecp=ecp, alpha=alpha)
    print(f"Saved TARP coverage data to {data_path}")


def train_npe_with_nan_retry(
    inference: Any,
    checkpoint_path: str,
    params: np.ndarray,
    data: np.ndarray,
    num_epochs: int = 1000,
    learning_rate: float = 1e-4,
    batch_size: int = 40,
    max_retries: int = 10,
    verbose: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Train NPE with automatic retry on NaN loss.

    Sometimes the loss initializes at NaN due to bad random initialization.
    This function will retry training with a fresh initialization if NaN is detected.

    Parameters
    ----------
    inference : object
        NPE inference object with simulations already appended.
    checkpoint_path : str
        Path to save model checkpoints.
    params : np.ndarray
        Parameter array (for reinitializing if needed).
    data : np.ndarray
        Data array (for reinitializing if needed).
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for optimizer.
    batch_size : int
        Training batch size.
    max_retries : int
        Maximum number of retry attempts.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    inference : object
        Trained inference object.
    metrics : object
        Training metrics.
    density_estimator : object
        Trained density estimator.

    Raises
    ------
    RuntimeError
        If all retry attempts fail due to NaN loss.

    Notes
    -----
    Only checks train_loss and val_loss for NaN. Test loss can sometimes
    be NaN due to evaluation issues even when training succeeded.
    """
    try:
        import jax.numpy as jnp
        from jaxili.inference import NPE
    except ImportError as err:
        raise ImportError("This function requires jaxili and JAX") from err

    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training attempt {attempt}/{max_retries}")
            print(f"{'=' * 60}")
            print(f"Training for {num_epochs} epochs...")

        metrics, density_estimator = inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            training_batch_size=batch_size,
        )

        # Check for NaN in training or validation loss
        has_nan = False
        nan_source = None

        if hasattr(metrics, "train_loss"):
            train_loss = np.array(metrics.train_loss)
            if np.any(np.isnan(train_loss)):
                has_nan = True
                nan_source = "training loss"
                if verbose:
                    first_nan_idx = np.where(np.isnan(train_loss))[0][0]
                    print(
                        f"  ❌ NaN detected in training loss at epoch {first_nan_idx}"
                    )

        if hasattr(metrics, "val_loss") and not has_nan:
            val_loss = np.array(metrics.val_loss)
            if np.any(np.isnan(val_loss)):
                has_nan = True
                nan_source = "validation loss"
                if verbose:
                    first_nan_idx = np.where(np.isnan(val_loss))[0][0]
                    print(
                        f"  ❌ NaN detected in validation loss at epoch {first_nan_idx}"
                    )

        if not has_nan:
            if verbose:
                print("  ✅ Training successful!")
                if hasattr(metrics, "train_loss"):
                    final_train_loss = metrics.train_loss[-1]
                    print(f"  Final training loss: {final_train_loss:.6f}")
                if hasattr(metrics, "val_loss"):
                    final_val_loss = metrics.val_loss[-1]
                    print(f"  Final validation loss: {final_val_loss:.6f}")
            return inference, metrics, density_estimator

        # NaN detected, retry if attempts remain
        if attempt < max_retries:
            if verbose:
                print("  Retrying with fresh initialization...")
            # Reinitialize inference object
            inference = NPE()
            inference = inference.append_simulations(jnp.array(params), jnp.array(data))
        else:
            if verbose:
                print(f"  All {max_retries} attempts failed")

    raise RuntimeError(
        f"Training failed after {max_retries} attempts due to NaN in {nan_source}"
    )
