#!/usr/bin/env python3
"""
Example: NPE Inference with Auto+Cross Power Spectra using the bar_impact package

This example shows how to run Neural Posterior Estimation (NPE) inference
using combined auto and cross power spectra.

Equivalent to running:
    python scripts/run_npe_inference_auto_cross_ps.py \
        --simulation-type nobaryons --fiducial-type baryonified \
        --bins 1,2,3,4 --lmax 2048 --lower-cut 100 --upper-cut 450 \
        --noisy --noise-level 0.26 --train --gpu 1 --rebin 10
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

# Set GPU before importing JAX
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp
import jax.random as random

# Import from the new package
from bar_impact.inference import NPEInference, NPEConfig
from bar_impact.analysis import (
    ResultsAggregator,
    aggregate_power_spectra,
    PosteriorPlotter,
    plot_triangle,
)


# ============================================================================
# Configuration (equivalent to argparse arguments)
# ============================================================================

# Data paths
DATA_DIR = "/home/tersenov/CosmoGridV1/stage3_forecast"
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"

# Analysis configuration
SIMULATION_TYPE = "nobaryons"
FIDUCIAL_TYPE = "baryonified"
BINS = [1, 2, 3, 4]
USE_BNT = True

# Power spectrum configuration
LMAX = 2048
LOWER_CUT = 100
UPPER_CUT = 450
REBIN_FACTOR = 10

# Noise configuration
NOISY = True
NOISE_LEVEL = 0.26
MASKED = True
MASK_AREA_SQDEG = 10000.0

# Training configuration
TRAIN = True
EPOCHS = 1000
BATCH_SIZE = 40
LEARNING_RATE = 1e-4

# Sampling configuration
NUM_SAMPLES = 3000
RANDOM_SEED = 1


# ============================================================================
# Data Loading and Processing
# ============================================================================

def rebin_spectrum(cls: np.ndarray, factor: int) -> np.ndarray:
    """
    Rebin power spectrum by averaging adjacent bins.
    
    Parameters
    ----------
    cls : np.ndarray
        Power spectrum, shape (n_samples, n_ell) or (n_ell,)
    factor : int
        Rebinning factor
        
    Returns
    -------
    np.ndarray
        Rebinned spectrum
    """
    if cls.ndim == 1:
        n_bins = len(cls) // factor
        return cls[:n_bins * factor].reshape(n_bins, factor).mean(axis=1)
    else:
        n_samples, n_ell = cls.shape
        n_bins = n_ell // factor
        return cls[:, :n_bins * factor].reshape(n_samples, n_bins, factor).mean(axis=2)


def load_power_spectrum_data(
    data_dir: str,
    sim_type: str,
    bins: list,
    use_bnt: bool,
    lower_cut: int,
    upper_cut: int,
    rebin_factor: int,
    noisy: bool,
    noise_level: float,
    masked: bool,
    mask_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and combine auto+cross power spectrum data.
    
    Returns combined data vector and cosmological parameters.
    """
    aggregator = ResultsAggregator()
    
    # Build filename components
    bnt_tag = "_bnt" if use_bnt else ""
    mask_tag = f"_masked_{int(mask_area)}sqdeg" if masked else ""
    noise_tag = f"_noisy_s{noise_level:.2f}" if noisy else ""
    
    data_vectors = []
    
    # Load auto power spectra
    print("Loading auto power spectra...")
    for bin_num in bins:
        filename = f"all_ps{bnt_tag}_auto_{bin_num}{mask_tag}{noise_tag}.npy"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            # Try alternative naming
            alt_filename = f"combined_bnt_ps_bin{bin_num}{mask_tag}{noise_tag}.npy"
            filepath = os.path.join(data_dir, alt_filename)
        
        if os.path.exists(filepath):
            cls = np.load(filepath)
            print(f"  Loaded auto_{bin_num}: {cls.shape}")
            
            # Apply ell cuts
            cls = cls[:, lower_cut:upper_cut+1]
            
            # Rebin if requested
            if rebin_factor > 1:
                cls = rebin_spectrum(cls, rebin_factor)
            
            print(f"    After cuts/rebin: {cls.shape}")
            data_vectors.append(cls)
        else:
            print(f"  Warning: {filename} not found, skipping")
    
    # Load cross power spectra
    print("Loading cross power spectra...")
    for bin_i, bin_j in combinations(bins, 2):
        filename = f"all_ps{bnt_tag}_cross_{bin_i}_{bin_j}{mask_tag}{noise_tag}.npy"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            cls = np.load(filepath)
            print(f"  Loaded cross_{bin_i}_{bin_j}: {cls.shape}")
            
            # Apply ell cuts
            cls = cls[:, lower_cut:upper_cut+1]
            
            # Rebin
            if rebin_factor > 1:
                cls = rebin_spectrum(cls, rebin_factor)
            
            print(f"    After cuts/rebin: {cls.shape}")
            data_vectors.append(cls)
        else:
            print(f"  Warning: {filename} not found, skipping")
    
    # Combine all spectra
    if not data_vectors:
        raise ValueError("No power spectrum data found!")
    
    combined_data = np.concatenate(data_vectors, axis=1)
    print(f"\nCombined data vector shape: {combined_data.shape}")
    
    # Load cosmological parameters
    params_file = os.path.join(data_dir, "cosmo_params_halofit.npy")
    params = np.load(params_file)
    print(f"Loaded parameters: {params.shape}")
    
    # Ensure matching lengths
    n_samples = min(len(combined_data), len(params))
    combined_data = combined_data[:n_samples]
    params = params[:n_samples]
    
    return combined_data, params


def normalize_data(data: np.ndarray, params: np.ndarray) -> tuple:
    """Normalize data and parameters for training."""
    # Log-transform power spectra (they are positive)
    # Handle any negative values from noise
    data_safe = np.clip(data, 1e-20, None)
    data_log = np.log10(data_safe)
    
    # Data normalization
    data_mean = np.mean(data_log, axis=0)
    data_std = np.std(data_log, axis=0)
    data_std[data_std < 1e-10] = 1.0
    data_normalized = (data_log - data_mean) / data_std
    
    # Parameter normalization
    params_mean = np.mean(params, axis=0)
    params_std = np.std(params, axis=0)
    params_normalized = (params - params_mean) / params_std
    
    normalization = {
        "data_mean": data_mean,
        "data_std": data_std,
        "params_mean": params_mean,
        "params_std": params_std,
        "use_log": True,
    }
    
    return data_normalized, params_normalized, normalization


# ============================================================================
# Inference Pipeline
# ============================================================================

def run_inference():
    """Main inference pipeline."""
    
    print("=" * 60)
    print("NPE Inference (Power Spectra) with bar_impact package")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load and prepare data
    # -------------------------------------------------------------------------
    print("\n--- Loading data ---")
    
    data, params = load_power_spectrum_data(
        DATA_DIR, SIMULATION_TYPE, BINS, USE_BNT,
        LOWER_CUT, UPPER_CUT, REBIN_FACTOR,
        NOISY, NOISE_LEVEL, MASKED, MASK_AREA_SQDEG
    )
    
    # Normalize
    data_norm, params_norm, normalization = normalize_data(data, params)
    print(f"Normalized data: mean={np.mean(data_norm):.4f}, std={np.std(data_norm):.4f}")
    
    # Filter zero-variance features
    variances = np.var(data_norm, axis=0)
    valid_mask = variances > 1e-10
    n_removed = np.sum(~valid_mask)
    if n_removed > 0:
        print(f"Removing {n_removed} zero-variance features")
        data_norm = data_norm[:, valid_mask]
        normalization["valid_mask"] = valid_mask
    
    print(f"Final data shape: {data_norm.shape}")
    
    # -------------------------------------------------------------------------
    # Configure and create NPE
    # -------------------------------------------------------------------------
    print("\n--- Configuring NPE ---")
    
    npe_config = NPEConfig(
        n_features=data_norm.shape[1],
        n_params=params_norm.shape[1],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )
    
    npe = NPEInference(config=npe_config)
    
    # Build checkpoint name
    bins_str = "".join(map(str, BINS))
    bnt_tag = "_bnt" if USE_BNT else ""
    checkpoint_name = f"npe_ps{bnt_tag}_{SIMULATION_TYPE}_bins{bins_str}_l{LOWER_CUT}-{UPPER_CUT}_r{REBIN_FACTOR}"
    if NOISY:
        checkpoint_name += f"_noisy_s{NOISE_LEVEL}"
    if MASKED:
        checkpoint_name += f"_masked_{int(MASK_AREA_SQDEG)}sqdeg"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    # -------------------------------------------------------------------------
    # Train or load model
    # -------------------------------------------------------------------------
    if TRAIN:
        print("\n--- Training NPE ---")
        npe.train(
            data_norm, params_norm,
            checkpoint_path=checkpoint_path,
        )
        print(f"Model saved to {checkpoint_path}")
    else:
        print("\n--- Loading trained model ---")
        npe.load(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    
    # -------------------------------------------------------------------------
    # Sample posterior for test case
    # -------------------------------------------------------------------------
    print("\n--- Sampling posterior for test simulation ---")
    
    # Use a random simulation as test
    test_idx = 0
    test_data = data_norm[test_idx:test_idx+1]
    true_params = params[test_idx]
    
    print(f"True parameters: {true_params}")
    
    # Sample posterior
    key = random.PRNGKey(RANDOM_SEED)
    samples_norm = npe.sample(test_data, NUM_SAMPLES, key)
    
    # Denormalize samples
    samples = samples_norm * normalization["params_std"] + normalization["params_mean"]
    
    print(f"Generated {samples.shape[0]} posterior samples")
    print(f"Posterior mean: {np.mean(samples, axis=0)}")
    print(f"Posterior std: {np.std(samples, axis=0)}")
    
    # Save samples
    samples_path = os.path.join(OUTPUT_DIR, f"posterior_samples_{checkpoint_name}.npy")
    np.save(samples_path, samples)
    print(f"Saved samples to {samples_path}")
    
    # -------------------------------------------------------------------------
    # Plot posterior
    # -------------------------------------------------------------------------
    print("\n--- Creating triangle plot ---")
    
    param_names = ["Om", "S8"]  # Adjust based on your parameters
    param_labels = [r"$\Omega_m$", r"$S_8$"]
    
    # Create truth dict
    truth_values = {
        param_names[i]: true_params[i] for i in range(len(param_names))
    }
    
    fig = plot_triangle(
        samples[:, :2],  # First two parameters
        param_names=param_names,
        param_labels=param_labels,
    )
    
    plot_path = os.path.join(OUTPUT_DIR, f"posterior_{checkpoint_name}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved triangle plot to {plot_path}")
    
    # -------------------------------------------------------------------------
    # Save normalization for later use
    # -------------------------------------------------------------------------
    norm_path = os.path.join(CHECKPOINT_DIR, f"normalization_{checkpoint_name}.npz")
    np.savez(norm_path, **{k: v for k, v in normalization.items() if isinstance(v, np.ndarray)})
    print(f"Saved normalization to {norm_path}")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
