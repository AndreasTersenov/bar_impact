#!/usr/bin/env python3
"""
Example: NPE Inference for L1 Norms using the bar_impact package

This example shows how to run Neural Posterior Estimation (NPE) inference
using L1 norm data vectors with the new package structure.

Equivalent to running:
    python scripts/run_npe_inference.py \
        --simulation-type baryonified --fiducial-type nobaryons \
        --bins 1,2,3,4 --scales 0,1,2,3,4 \
        --noisy --noise-level 0.26 \
        --train --run-coverage-test --gpu 1 --new-normalization
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set GPU before importing JAX
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import jax.random as random

# Use jaxili directly like the original script
from jaxili.inference import NPE

# Import getdist for plotting (like original)
from getdist import plots, MCSamples

# Import from the new package
from bar_impact.core import ConvergenceMap, SurveyMask
from bar_impact.processing import L1NormProcessor
from bar_impact.inference import train_with_nan_retry


# ============================================================================
# Configuration (equivalent to argparse arguments)
# ============================================================================

# Data paths - adjust these to your setup
# Option 1: Use data from example_l1_norm_processing.py output
L1_NORMS_DIR = "./outputs/l1_norms"
USE_LOCAL_OUTPUT = True  # Set to True to use local outputs, False to use CosmoGRID aggregated data

# Option 2: Use pre-aggregated CosmoGRID data (recommended for training)
# These files have shape (n_cosmologies * n_perms, nscales, nbins) and match the params file
COSMOGRID_DATA_DIR = "/home/tersenov/CosmoGridV1/stage3_forecast/grid"

# Cosmological parameters from CosmoGRID (auto-selected based on SIMULATION_TYPE):
COSMOGRID_PARAMS_DIR = "/home/tersenov/CosmoGridV1/stage3_forecast/grid"

CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs/npe_results"

# Analysis configuration
SIMULATION_TYPE = "nobaryons"  # "baryonified" or "nobaryons" - for training data
FIDUCIAL_TYPE = "nobaryons"      # For fiducial reference
BINS = [1, 2, 3, 4]
SCALES = [0, 1, 2, 3, 4]  # Wavelet scales to use (0-indexed) - MUST match what's in the data!
NOISE_LEVEL = 0.26
NOISY = True
MASKED = True  # Use masked data to test with newly processed mask_area=14002.0 files
MASK_AREA_SQDEG = 14002  # Match the processing script's mask area

# Training configuration
TRAIN = True
EPOCHS = 1000
BATCH_SIZE = 40
LEARNING_RATE = 1e-4

# Sampling configuration
NUM_SAMPLES = 3000
RANDOM_SEED = 1

# Coverage testing
RUN_COVERAGE_TEST = True
COVERAGE_NUM_SIMS = 100
COVERAGE_NUM_SAMPLES = 1000
COVERAGE_BOOTSTRAP = True
COVERAGE_NUM_BOOTSTRAP = 100


# ============================================================================
# Data Loading
# ============================================================================

def load_l1_norm_data(
    l1_dir: str,
    sim_type: str,  # "grid_baryonified", "grid_nobaryons", "fiducial_nobaryons"
    bins: list,
    scales: list,
    noisy: bool,
    noise_level: float,
    masked: bool = False,
    mask_area: int = None,
) -> np.ndarray:
    """
    Load L1 norm data vectors from combined files.
    
    The combined files have shape (n_samples, 40) where 40 = nbins.
    Since the original data was (5, 40) = (nscales, nbins), the combined
    files seem to have only one scale or are already flattened.
    
    Returns
    -------
    data : np.ndarray
        Data vectors of shape (n_simulations, n_features)
    """
    all_data = []
    
    for bin_num in bins:
        # Try multiple filename patterns (with and without mask)
        noise_suffix = f"_noisy_s{noise_level:.2f}" if noisy else ""
        
        possible_filenames = []
        if masked and mask_area:
            # Try masked version first
            possible_filenames.append(
                f"combined_l1_norms_{sim_type}_bin{bin_num}_masked_{mask_area}sqdeg{noise_suffix}.npy"
            )
        # Also try unmasked version
        possible_filenames.append(
            f"combined_l1_norms_{sim_type}_bin{bin_num}{noise_suffix}.npy"
        )
        # Try masked version even if masked=False (in case fiducial was processed with mask)
        if mask_area and not masked:
            possible_filenames.append(
                f"combined_l1_norms_{sim_type}_bin{bin_num}_masked_{mask_area}sqdeg{noise_suffix}.npy"
            )
        
        data = None
        for filename in possible_filenames:
            filepath = os.path.join(l1_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                print(f"Loaded {filename}: shape {data.shape}")
                break
        
        if data is None:
            raise FileNotFoundError(f"Data file not found. Tried: {possible_filenames}")
        
        # The data should have shape (n_samples, n_features)
        # If it has 3D shape (n_samples, nscales, nbins), select scales
        if len(data.shape) == 3:
            # Select specified scales and flatten
            selected = data[:, scales, :]  # (n_samples, len(scales), nbins)
            data = selected.reshape(data.shape[0], -1)
            print(f"  After scale selection: {data.shape}")
        
        all_data.append(data)
    
    # Combine all bins
    combined_data = np.concatenate(all_data, axis=1)
    print(f"Combined data shape: {combined_data.shape}")
    
    return combined_data


def load_cosmogrid_l1_data(
    data_dir: str,
    sim_type: str,  # "baryonified" or "nobaryons"
    bins: list,
    scales: list,
    noisy: bool,
    noise_level: float,
    masked: bool = False,
    mask_area: int = None,
) -> np.ndarray:
    """
    Load L1 norm data from pre-aggregated CosmoGRID files.
    
    These files have shape (n_samples, nscales, nbins) where:
    - n_samples = n_cosmologies * n_permutations (matches params file)
    - nscales = 5 (wavelet scales)
    - nbins = 40 (L1 norm bins)
    
    Returns
    -------
    data : np.ndarray
        Data vectors of shape (n_simulations, n_features)
    """
    all_data = []
    
    for bin_num in bins:
        # Construct filename matching CosmoGRID naming convention
        mask_suffix = f"_masked_{mask_area}sqdeg" if masked else ""
        noise_suffix = f"_noisy_s{noise_level:.2f}" if noisy else ""
        
        # Try different naming patterns used in CosmoGRID
        # IMPORTANT: Try files WITHOUT _new_normalization first (they have correct scale)
        possible_filenames = [
            # Standard patterns (correct normalization)
            f"all_l1_norms_grid_{sim_type}_bin{bin_num}{mask_suffix}{noise_suffix}.npy",
            f"all_bnt_l1_norms_grid_{sim_type}_bin{bin_num}{mask_suffix}{noise_suffix}.npy",
            f"all_l1_norms_{sim_type}_bin{bin_num}{mask_suffix}{noise_suffix}.npy",
            f"all_bnt_l1_norms_{sim_type}_bin{bin_num}{mask_suffix}{noise_suffix}.npy",
            # New normalization versions (values ~15000x larger - avoid unless specifically needed)
            f"all_l1_norms_grid_{sim_type}_bin{bin_num}{mask_suffix}{noise_suffix}_new_normalization.npy",
            f"all_bnt_l1_norms_grid_{sim_type}_bin{bin_num}{mask_suffix}{noise_suffix}_new_normalization.npy",
        ]
        
        data = None
        for filename in possible_filenames:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                print(f"Loaded {filename}: shape {data.shape}")
                break
        
        if data is None:
            print(f"Available files for bin {bin_num}:")
            import glob
            pattern = os.path.join(data_dir, f"*l1*{sim_type}*bin{bin_num}*.npy")
            for f in sorted(glob.glob(pattern)):
                print(f"  {os.path.basename(f)}")
            raise FileNotFoundError(f"No matching L1 norm file found in {data_dir} for bin {bin_num}")
        
        # Select scales: data shape is (n_samples, nscales, nbins)
        if len(data.shape) == 3:
            selected = data[:, scales, :]  # (n_samples, len(scales), nbins)
            data = selected.reshape(data.shape[0], -1)
            print(f"  After scale selection (scales {scales}): {data.shape}")
        elif len(data.shape) == 2:
            # Already flattened, assume all scales
            print(f"  Data already 2D, using as-is")
        
        all_data.append(data)
    
    # Combine all bins
    combined_data = np.concatenate(all_data, axis=1)
    print(f"Combined data shape: {combined_data.shape}")
    
    return combined_data


def load_params(params_file: str) -> np.ndarray:
    """
    Load cosmological parameters.
    
    CosmoGRID has separate params files for baryonified and nobaryons:
    - cosmo_params.npy for nobaryons (16965 rows)
    - cosmo_params_baryonified.npy for baryonified (16966 rows)
    
    Parameters
    ----------
    params_file : str
        Path to the cosmological parameters file
    
    Returns
    -------
    params : np.ndarray
        Parameters of shape (n_samples, n_params)
    """
    params = np.load(params_file)
    print(f"Loaded parameters: shape {params.shape}")
    
    return params


def normalize_data(data: np.ndarray, params: np.ndarray) -> tuple:
    """Normalize data and parameters for training."""
    # Data normalization (per-feature)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_std[data_std < 1e-10] = 1.0  # Avoid division by zero
    data_normalized = (data - data_mean) / data_std
    
    # Parameter normalization (per-parameter)
    params_mean = np.mean(params, axis=0)
    params_std = np.std(params, axis=0)
    params_normalized = (params - params_mean) / params_std
    
    normalization = {
        "data_mean": data_mean,
        "data_std": data_std,
        "params_mean": params_mean,
        "params_std": params_std,
    }
    
    return data_normalized, params_normalized, normalization


# ============================================================================
# Inference Pipeline
# ============================================================================

def filter_zero_variance_bins(data: np.ndarray, min_variance: float = 1e-10, verbose: bool = True):
    """
    Identify and filter out bins (features) with zero or near-zero variance.
    
    This is CRITICAL for NPE training - zero-variance features cause NaN loss.
    
    Args:
        data: np.ndarray of shape (n_samples, n_features)
        min_variance: Minimum variance threshold (default: 1e-10)
        verbose: Whether to print information about filtered bins
        
    Returns:
        valid_mask: Boolean array indicating which bins to keep
        n_removed: Number of bins removed
    """
    # Compute variance across samples for each feature
    variances = np.var(data, axis=0)
    
    # Create mask for valid (non-zero variance) bins
    valid_mask = variances > min_variance
    
    n_total = len(valid_mask)
    n_valid = np.sum(valid_mask)
    n_removed = n_total - n_valid
    
    if verbose:
        print(f"\nZero-variance bin filtering:")
        print(f"  Total bins: {n_total}")
        print(f"  Valid bins (variance > {min_variance}): {n_valid}")
        print(f"  Removed bins: {n_removed}")
        
        if n_removed > 0:
            zero_var_indices = np.where(~valid_mask)[0]
            if len(zero_var_indices) <= 20:
                print(f"  Removed bin indices: {zero_var_indices.tolist()}")
            else:
                print(f"  First 20 removed bin indices: {zero_var_indices[:20].tolist()}...")
    
    return valid_mask, n_removed


def run_inference():
    """Main inference pipeline."""
    
    print("=" * 60)
    print("NPE Inference with bar_impact package")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Load and prepare training data (grid cosmologies)
    # -------------------------------------------------------------------------
    print("\n--- Loading training data (grid cosmologies) ---")
    
    if USE_LOCAL_OUTPUT:
        # Load from local outputs (from example_l1_norm_processing.py)
        # Note: This data may not align with params - see warning below
        sim_type = f"grid_{SIMULATION_TYPE}"
        data = load_l1_norm_data(
            l1_dir=L1_NORMS_DIR,
            sim_type=sim_type,
            bins=BINS,
            scales=SCALES,
            noisy=NOISY,
            noise_level=NOISE_LEVEL,
            masked=MASKED,
            mask_area=MASK_AREA_SQDEG,
        )
    else:
        # Load from pre-aggregated CosmoGRID files (recommended)
        data = load_cosmogrid_l1_data(
            data_dir=COSMOGRID_DATA_DIR,
            sim_type=SIMULATION_TYPE,
            bins=BINS,
            scales=SCALES,
            noisy=NOISY,
            noise_level=NOISE_LEVEL,
            masked=MASKED,
            mask_area=MASK_AREA_SQDEG,
        )
    
    # Load cosmological parameters - select correct file based on simulation type
    # baryonified uses cosmo_params_baryonified.npy, nobaryons uses cosmo_params.npy
    params_suffix = "_baryonified" if SIMULATION_TYPE == "baryonified" else ""
    params_file = os.path.join(COSMOGRID_PARAMS_DIR, f"cosmo_params{params_suffix}.npy")
    params = load_params(params_file)
    print(f"Loaded params for {SIMULATION_TYPE}: {params_file}")
    
    # Verify data and params match
    if len(data) != len(params):
        raise ValueError(f"Data ({len(data)}) and params ({len(params)}) size mismatch! "
                        f"Make sure SIMULATION_TYPE='{SIMULATION_TYPE}' matches your data.")
    
    print(f"Using {len(data)} samples for training")
    
    # Debug: Show parameter ranges in training data
    print(f"Training parameters shape: {params.shape}")
    print(f"Training parameters - mean: {np.mean(params, axis=0)}")
    print(f"Training parameters - std: {np.std(params, axis=0)}")
    print(f"Training parameters - min: {np.min(params, axis=0)}")
    print(f"Training parameters - max: {np.max(params, axis=0)}")
    
    # CRITICAL: Filter out zero-variance bins before training
    # This prevents NaN loss during NPE training
    valid_bin_mask, n_removed = filter_zero_variance_bins(data, min_variance=1e-10, verbose=True)
    data_filtered = data[:, valid_bin_mask]
    print(f"Data shape after filtering: {data_filtered.shape}")
    
    # NOTE: We do NOT manually normalize! jaxili's NPE handles normalization internally.
    # Manually normalizing causes double-normalization and breaks training.
    # Just store raw statistics for fiducial data processing
    data_mean = np.mean(data_filtered, axis=0)
    data_std = np.std(data_filtered, axis=0)
    
    # Store info for applying to fiducial data later
    normalization = {
        "valid_bin_mask": valid_bin_mask,
        "data_mean": data_mean,
        "data_std": data_std,
    }
    
    print(f"Data statistics: mean={np.mean(data_filtered):.4f}, std={np.std(data_filtered):.4f}")
    
    # -------------------------------------------------------------------------
    # Create NPE and Train (following original script exactly)
    # -------------------------------------------------------------------------
    print("\n--- Setting up NPE ---")
    
    # Build checkpoint name
    bins_str = "".join(map(str, BINS))
    scales_str = "".join(map(str, SCALES))
    mask_tag = f"_masked_{MASK_AREA_SQDEG}sqdeg" if MASKED else ""
    checkpoint_name = f"npe_{SIMULATION_TYPE}_bins{bins_str}_scales{scales_str}{mask_tag}"
    if NOISY:
        checkpoint_name += f"_noisy_s{NOISE_LEVEL}"
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Convert to JAX arrays (like original)
    params_jax = jnp.array(params)
    data_jax = jnp.array(data_filtered)
    
    # Initialize NPE (like original)
    inference = NPE()
    inference = inference.append_simulations(params_jax, data_jax)
    print("Added simulations to NPE")
    
    # Train or load model
    if TRAIN:
        print(f"\n--- Training NPE for {EPOCHS} epochs (with NaN retry) ---")
        # Use train_with_nan_retry to handle numerical instability
        inference, metrics, density_estimator = train_with_nan_retry(
            inference=inference,
            params=params_jax,
            data=data_jax,
            checkpoint_path=checkpoint_path,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_retries=10,
            verbose=True,
        )
        print("Training completed successfully")
        print(f"Model saved to {checkpoint_path}")
    else:
        print("\n--- Loading trained model ---")
        inference.load(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    
    # Build posterior (like original)
    posterior = inference.build_posterior()
    print("Built posterior")
    
    # -------------------------------------------------------------------------
    # Load fiducial and sample posterior
    # -------------------------------------------------------------------------
    # Load fiducial and sample posterior
    # -------------------------------------------------------------------------
    print("\n--- Loading fiducial data and sampling posterior ---")
    
    # Load fiducial data - use local if available, otherwise CosmoGRID
    fid_means = []
    
    if USE_LOCAL_OUTPUT:
        # Load from locally processed files
        print("Loading fiducial from local outputs...")
        for bin_num in BINS:
            mask_suffix = f"_masked_{int(MASK_AREA_SQDEG)}sqdeg" if MASKED else ""
            fid_filename = f"combined_l1_norms_fiducial_{FIDUCIAL_TYPE}_bin{bin_num}{mask_suffix}_noisy_s{NOISE_LEVEL:.2f}.npy"
            fid_path = os.path.join(L1_NORMS_DIR, fid_filename)
            
            if not os.path.exists(fid_path):
                raise FileNotFoundError(f"Local fiducial file not found: {fid_path}")
            
            fid_full = np.load(fid_path)
            fid_mean = np.mean(fid_full, axis=0)  # Average over realizations
            fid_means.append(fid_mean)
            print(f"Loaded fiducial from {fid_filename}, shape: {fid_full.shape}")
    else:
        # Load from CosmoGRID pre-processed files
        print("Loading fiducial from CosmoGRID...")
        fiducial_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial"
        
        for bin_num in BINS:
            # Construct fiducial filename to match training data normalization
            # CRITICAL: Must use same normalization as training data!
            if MASKED:
                fid_filename = f"all_l1_norms_fiducial_{FIDUCIAL_TYPE}_bin{bin_num}_masked_{int(MASK_AREA_SQDEG)}sqdeg_noisy_s{NOISE_LEVEL:.2f}.npy"
            else:
                fid_filename = f"all_l1_norms_fiducial_{FIDUCIAL_TYPE}_bin{bin_num}_noisy_s{NOISE_LEVEL:.2f}.npy"
            
            fid_path = os.path.join(fiducial_dir, fid_filename)
            
            if not os.path.exists(fid_path):
                print(f"ERROR: Fiducial file not found: {fid_path}")
                # Try alternative patterns
                alt_patterns = [
                    f"all_bnt_l1_norms_fiducial_{FIDUCIAL_TYPE}_bin{bin_num}_noisy_s{NOISE_LEVEL:.2f}.npy",
                    f"all_l1_norms_fiducial_{FIDUCIAL_TYPE}_bin{bin_num}_noisy_s{NOISE_LEVEL:.2f}.npy",
                ]
                for alt in alt_patterns:
                    alt_path = os.path.join(fiducial_dir, alt)
                    if os.path.exists(alt_path):
                        fid_path = alt_path
                        fid_filename = alt
                        break
                else:
                    raise FileNotFoundError(f"Fiducial file not found: {fid_path}")
            
            fid_full = np.load(fid_path)
            fid_mean = np.mean(fid_full, axis=0)  # Average over realizations
            fid_means.append(fid_mean)
            print(f"Loaded fiducial from {fid_filename}, shape: {fid_full.shape}")
    
    # Process fiducial data according to scale selection (like original)
    fid_data_list = []
    for i, fid_mean in enumerate(fid_means):
        # fid_mean has shape (nscales, nbins) = (5, 40)
        bin_fid_scales = []
        for scale_idx in SCALES:
            scale_data = fid_mean[scale_idx]  # Shape: (nbins,) = (40,)
            bin_fid_scales.append(scale_data)
        
        # Concatenate scales for this bin
        bin_fid_data = np.concatenate(bin_fid_scales)
        fid_data_list.append(bin_fid_data)
    
    # Concatenate all bins' fiducial data
    fiducial_mean_filtered = np.concatenate(fid_data_list)
    print(f"Combined fiducial data shape (before filtering): {fiducial_mean_filtered.shape}")
    
    # Apply the same zero-variance bin mask used for training data
    fiducial_mean_filtered = fiducial_mean_filtered[normalization["valid_bin_mask"]]
    print(f"Combined fiducial data shape (after filtering): {fiducial_mean_filtered.shape}")
    
    try:
        master_key = random.PRNGKey(RANDOM_SEED)
        sample_key, master_key = random.split(master_key)
        samples = posterior.sample(
            x=fiducial_mean_filtered,
            num_samples=NUM_SAMPLES,
            key=sample_key,
        )
        samples = np.array(samples)  # Convert to numpy
        
        # Debug: check samples range
        print(f"Posterior samples shape: {samples.shape}")
        print(f"Posterior samples - mean: {np.mean(samples, axis=0)}")
        print(f"Posterior samples - std: {np.std(samples, axis=0)}")
        print(f"Posterior samples - min: {np.min(samples, axis=0)}")
        print(f"Posterior samples - max: {np.max(samples, axis=0)}")
        
        print(f"Generated {samples.shape[0]} posterior samples")
        
        # Save samples
        samples_path = os.path.join(OUTPUT_DIR, f"posterior_samples_{checkpoint_name}.npy")
        np.save(samples_path, samples)
        print(f"Saved samples to {samples_path}")
        
        # -------------------------------------------------------------------------
        # Plot posterior
        # -------------------------------------------------------------------------
        print("\n--- Creating triangle plot ---")
        
        # Check if samples have valid range
        samples_std = np.std(samples, axis=0)
        samples_min = np.min(samples, axis=0)
        samples_max = np.max(samples, axis=0)
        
        # Check for infinite or NaN values
        if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
            print("  WARNING: Samples contain NaN or Inf values - skipping plot")
            print(f"  NaN count: {np.sum(np.isnan(samples))}")
            print(f"  Inf count: {np.sum(np.isinf(samples))}")
        elif np.all(samples_max - samples_min < 1e-10):
            print("  WARNING: Samples have no dynamic range - skipping plot")
            print(f"  Sample range: min={samples_min}, max={samples_max}")
        else:
            # Parameter names depend on what's in your params file
            # CosmoGRID typically has: Om, Ob, h, ns, sigma8, w0
            # But original uses: Om, S8, w0, H0, ns, Ob
            labels = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"][:params.shape[1]]
            param_names = ["Om", "S8", "w0", "H0", "ns", "Ob"][:params.shape[1]]
            
            # Create triangle plot using getdist (like original script)
            try:
                from getdist import plots, MCSamples
                
                # Create descriptive sample label
                fiducial_desc = f"{FIDUCIAL_TYPE}"
                if NOISY:
                    fiducial_desc += f"_n{NOISE_LEVEL:.2f}"
                sample_label = f"{SIMULATION_TYPE} vs {fiducial_desc}, bins{''.join(map(str, BINS))}, scales{''.join(map(str, SCALES))}"
                
                samples_mc = MCSamples(
                    samples=np.array(samples),
                    names=labels,
                    label=sample_label,
                )
                
                g = plots.get_subplot_plotter()
                g.settings.figure_legend_frame = False
                g.settings.alpha_filled_add = 0.4
                
                g.triangle_plot([samples_mc], filled=True,
                               line_args=[{'color': 'blue'}],
                               contour_colors=['blue'])
                
                plot_path = os.path.join(OUTPUT_DIR, f"posterior_{checkpoint_name}.pdf")
                plt.savefig(plot_path, transparent=True, dpi=150)
                plt.close()
                print(f"Saved triangle plot to {plot_path}")
            except ImportError:
                print("  getdist package not installed, skipping triangle plot")
                print("  Install with: pip install getdist")
    
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print(f"  Make sure the CosmoGRID fiducial files exist")
    
    # -------------------------------------------------------------------------
    # Coverage testing (simplified - can be added later if needed)
    # -------------------------------------------------------------------------
    if RUN_COVERAGE_TEST:
        print("\n--- Coverage testing not implemented in this simplified example ---")
        print("For coverage testing, refer to scripts/run_npe_inference.py")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
