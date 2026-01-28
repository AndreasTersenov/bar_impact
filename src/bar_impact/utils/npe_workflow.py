"""
NPE Workflow Utilities

Common workflow functions for NPE inference scripts, including:
- NPE initialization and training/loading
- Posterior sampling
- Triangle plot generation with standard cosmological parameters
- Output file management
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from jaxili.inference import NPE

# Standard cosmological parameter configuration
STANDARD_COSMO_PARAMS = {
    "names": ["Omega_m", "S_8", "w_0", "H_0", "n_s", "Omega_b"],
    "labels": [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"],
    "fiducial_values": jnp.array(
        [[2.600e-01, 8.400e-01, -1.000e00, 6.736e01, 9.649e-01, 4.930e-02]]
    ),
}


def initialize_npe(params, data):
    """
    Initialize NPE inference object with simulations.

    Args:
        params: Parameter array (n_sims, n_params)
        data: Data array (n_sims, n_features)

    Returns:
        NPE inference object with simulations appended
    """
    inference = NPE()
    inference = inference.append_simulations(params, data)
    return inference


def train_or_load_npe(inference, checkpoint_path, should_train, train_params=None):
    """
    Train a new NPE model or load existing one from checkpoint.

    Args:
        inference: NPE inference object with simulations
        checkpoint_path: Path to checkpoint directory
        should_train: Whether to train (True) or load (False)
        train_params: Dictionary with training parameters:
            - num_epochs: Number of training epochs
            - learning_rate: Learning rate
            - batch_size: Batch size
            - params: Parameter array (for NaN retry)
            - data: Data array (for NaN retry)
            - max_retries: Maximum NaN retry attempts (default: 10)

    Returns:
        tuple: (inference, metrics, density_estimator) if training
               (inference, None, density_estimator) if loading
    """
    if should_train:
        if train_params is None:
            raise ValueError("train_params required when should_train=True")

        print(f"\nTraining for {train_params['num_epochs']} epochs...")

        # Check if we should use NaN retry
        if "params" in train_params and "data" in train_params:
            # Use NaN-resilient training (imports locally to avoid circular dependency)
            from bar_impact.utils.inference import train_npe_with_nan_retry

            inference, metrics, density_estimator = train_npe_with_nan_retry(
                inference=inference,
                checkpoint_path=checkpoint_path,
                params=train_params["params"],
                data=train_params["data"],
                num_epochs=train_params["num_epochs"],
                learning_rate=train_params["learning_rate"],
                batch_size=train_params["batch_size"],
                max_retries=train_params.get("max_retries", 10),
            )
        else:
            # Standard training without NaN retry
            metrics, density_estimator = inference.train(
                checkpoint_path=checkpoint_path,
                num_epochs=train_params["num_epochs"],
                learning_rate=train_params["learning_rate"],
                training_batch_size=train_params["batch_size"],
            )

        print(f"Training completed. Model saved to {checkpoint_path}")
        return inference, metrics, density_estimator
    else:
        print(f"\nLoading model from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_path}\n"
                f"Use --train to train a new model first."
            )

        density_estimator = inference.load(checkpoint_path)
        print("Model loaded successfully")
        return inference, None, density_estimator


def create_triangle_plot(
    samples, sample_label, output_path, param_config=None, color="blue", dpi=300
):
    """
    Create and save a cosmological parameter triangle plot.

    Args:
        samples: Posterior samples array (n_samples, n_params)
        sample_label: Descriptive label for the sample
        output_path: Full path where to save the plot
        param_config: Optional dict with 'labels' and 'fiducial_values' (uses STANDARD_COSMO_PARAMS if None)
        color: Color for contours and lines (default: 'blue')
        dpi: Resolution for saved plot (default: 300)

    Returns:
        str: Path where plot was saved
    """
    if param_config is None:
        param_config = STANDARD_COSMO_PARAMS

    # Convert samples to numpy if needed
    samples_np = np.array(samples)

    # Create MCSamples object
    mc_samples = MCSamples(
        samples=samples_np,
        names=param_config["labels"],
        label=sample_label,
    )

    # Configure plot settings
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4

    # Create triangle plot with markers if fiducial values provided
    plot_kwargs = {
        "filled": True,
        "line_args": [{"color": color}],
        "contour_colors": [color],
    }

    if "fiducial_values" in param_config:
        true_params = param_config["fiducial_values"]
        plot_kwargs["markers"] = dict(zip(param_config["labels"], true_params[0]))

    g.triangle_plot([mc_samples], **plot_kwargs)

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, transparent=True, dpi=dpi)
    plt.close()

    print(f"Saved triangle plot to {output_path}")
    return output_path


def sample_and_save_posterior(posterior, observation, output_config):
    """
    Sample from posterior, save samples, and create triangle plot.

    Args:
        posterior: Trained posterior object
        observation: Observation vector to condition on (fiducial data)
        output_config: Dictionary with output configuration:
            - samples_dir: Directory to save samples
            - output_dir: Directory to save plots
            - base_filename: Base name for output files
            - num_samples: Number of samples to draw
            - random_seed: Seed for sampling
            - sample_label: Label for triangle plot
            - run: Optional run number to append to filenames
            - color: Optional color for triangle plot (default: 'blue')
            - param_config: Optional parameter configuration (uses STANDARD_COSMO_PARAMS if None)

    Returns:
        tuple: (samples_path, plot_path) - Paths to saved samples and triangle plot
    """
    from jax import random as jax_random

    # Sample from posterior
    print("\nSampling from posterior...")
    key = jax_random.PRNGKey(output_config["random_seed"])
    samples = posterior.sample(
        x=observation, num_samples=output_config["num_samples"], key=key
    )
    samples_np = np.array(samples)
    print(
        f"Generated {output_config['num_samples']} samples, shape: {samples_np.shape}"
    )

    # Construct output filenames
    base_name = output_config["base_filename"]
    if output_config.get("run") is not None:
        base_name += f"_run{output_config['run']}"

    # Save samples
    os.makedirs(output_config["samples_dir"], exist_ok=True)
    samples_filename = f"{base_name}_samples.npy"
    samples_path = os.path.join(output_config["samples_dir"], samples_filename)
    np.save(samples_path, samples_np)
    print(f"Saved samples to {samples_path}")

    # Create triangle plot
    print("\nCreating triangle plot...")
    plot_filename = f"{base_name}_triangle.pdf"
    plot_path = os.path.join(output_config["output_dir"], plot_filename)

    create_triangle_plot(
        samples=samples_np,
        sample_label=output_config["sample_label"],
        output_path=plot_path,
        param_config=output_config.get("param_config"),
        color=output_config.get("color", "blue"),
        dpi=300,
    )

    return samples_path, plot_path


def setup_jax_environment(gpu_id="0", force_cpu=False, enable_x64=True):
    """
    Configure JAX device and precision settings.

    Args:
        gpu_id: GPU device ID to use (default: '0')
        force_cpu: Force CPU usage instead of GPU (default: False)
        enable_x64: Enable 64-bit precision (default: True)

    Returns:
        list: JAX devices being used
    """
    import jax

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if force_cpu:
        jax.config.update("jax_platform_name", "cpu")
        print("Forcing CPU usage")

    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    devices = jax.devices()
    print(f"Device used by JAX: {devices}")
    return devices


def print_analysis_summary(config):
    """
    Print a formatted summary of analysis configuration.

    Args:
        config: Dictionary with configuration keys:
            - simulation_type: Type of simulation
            - fiducial_type: Type of fiducial
            - description: Description of analysis (e.g., bin/scale info)
            - noisy: Whether using noisy data
            - noise_level: Noise level if noisy
            - masked: Whether using masked data
            - mask_info: Mask description if masked
            - checkpoint_name: Checkpoint identifier
            - additional: Optional dict of additional info to display
    """
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Simulation type: {config['simulation_type']}")
    print(f"Fiducial type: {config['fiducial_type']}")

    # Support both 'description' and separate 'bin_desc'/'scale_desc'
    if "description" in config:
        print(f"Analysis: {config['description']}")
    elif "bin_desc" in config and "scale_desc" in config:
        print(f"Analysis: {config['bin_desc']}, {config['scale_desc']}")

    print(
        f"Noisy: {config.get('noisy', False)}"
        + (
            f" (noise level: {config.get('noise_level')})"
            if config.get("noisy")
            else ""
        )
    )

    if config.get("masked"):
        mask_info = config.get("mask_info")
        if mask_info:
            print(f"Masked: True ({mask_info})")
        elif config.get("mask_area_sqdeg"):
            print(f"Masked: True ({config.get('mask_area_sqdeg')} sq deg)")
        else:
            print("Masked: True")

    # Print new_normalization if present
    if "new_normalization" in config and config["new_normalization"]:
        print("New normalization: True")

    if "additional" in config:
        for key, value in config["additional"].items():
            print(f"{key}: {value}")

    print(f"Checkpoint: {config['checkpoint_name']}")
    print("=" * 60 + "\n")


def print_completion_summary(paths, run_coverage_test=False):
    """
    Print a formatted summary of completed analysis with output file paths.

    Args:
        paths: Dictionary with output paths:
            - checkpoint or checkpoint_path: Path to model checkpoint
            - samples: Path to saved samples
            - triangle_plot: Path to triangle plot
            - coverage_plot: Optional path to TARP coverage plot
        run_coverage_test: Whether coverage test was run
    """
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    # Support both 'checkpoint' and 'checkpoint_path'
    checkpoint = paths.get("checkpoint_path") or paths.get("checkpoint", "N/A")
    print(f"Checkpoint: {checkpoint}")
    print(f"Samples: {paths.get('samples', 'N/A')}")
    print(f"Triangle plot: {paths.get('triangle_plot', 'N/A')}")

    if run_coverage_test and "coverage_plot" in paths:
        print(f"Coverage plot: {paths['coverage_plot']}")

    print("=" * 60)
