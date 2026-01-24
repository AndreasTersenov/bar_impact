#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/run_npe_peak_counts_inference.py

import os
import sys
import argparse
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
from jaxili.inference import NPE
from getdist import plots, MCSamples

# Add tarp package to path if needed
tarp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tarp', 'src')
if tarp_path not in sys.path:
    sys.path.insert(0, tarp_path)
from tarp import get_tarp_coverage

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NPE inference on CosmoGRID peak counts")
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, 
                        default='/home/tersenov/CosmoGridV1/stage3_forecast',
                        help="Base directory for data")
    
    parser.add_argument("--simulation-type", type=str, choices=["baryonified", "nobaryons"],
                        default="baryonified", 
                        help="Type of simulation to use for training (baryonified or nobaryons)")
    
    # Analysis configuration
    bin_group = parser.add_mutually_exclusive_group(required=False)
    bin_group.add_argument("--bin", type=int, default=2, 
                        help="Which redshift bin to analyze")
    bin_group.add_argument("--bins", type=str, 
                        help="Comma-separated list of redshift bins to analyze for tomographic inference")
    
    # BNT configuration
    parser.add_argument("--bnt", action="store_true", 
                        help="Use BNT-transformed data")
    
    bnt_bin_group = parser.add_mutually_exclusive_group(required=False)
    bnt_bin_group.add_argument("--bnt-bin", type=int, default=3,
                        help="Which BNT bin to analyze (0-3, default=3 corresponds to bin4)")
    bnt_bin_group.add_argument("--bnt-bins", type=str,
                        help="Comma-separated list of BNT bins to analyze for tomographic inference")
    
    # Create a mutually exclusive group for scale selection
    scale_group = parser.add_mutually_exclusive_group(required=False)
    scale_group.add_argument("--scale", type=int, default=0, 
                        help="Which scale index to analyze (0-indexed). Use for single scale analysis.")
    scale_group.add_argument("--scales", type=str, 
                        help="Comma-separated list of scale indices to analyze (0-indexed). Use for multi-scale analysis.")
    scale_group.add_argument("--scales-per-bin", type=str,
                        help="Semicolon-separated scale specifications per bin (0-indexed). "
                             "E.g., '1,2,3;0,1,2,3;0,1,2,3;0,1,2,3' applies scales [1,2,3] to bin 1, "
                             "[0,1,2,3] to bins 2,3,4. Use for BNT analysis with different cuts per bin.")
    
    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    
    # Fiducial configuration  
    parser.add_argument("--fiducial-type", type=str, choices=["baryonified", "nobaryons"],
                        default=None,  # Will default to match simulation-type if not specified
                        help="Type of fiducial (baryonified or nobaryons). If not specified, matches --simulation-type")
    
    # Training parameters
    parser.add_argument("--train", action="store_true", 
                        help="Train model (if not specified, will try to load existing model)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--epochs", type=int, default=1000, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=40, 
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                        help="Learning rate")
    
    # Sampling parameters
    parser.add_argument("--num-samples", type=int, default=3000, 
                        help="Number of posterior samples to generate")
    parser.add_argument("--random-seed", type=int, default=1, 
                        help="Random seed for sampling")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number to append to output filenames (for multiple runs)")
    
    # Coverage testing parameters
    parser.add_argument("--run-coverage-test", action="store_true",
                        help="Run TARP coverage test to assess posterior quality")
    parser.add_argument("--coverage-num-sims", type=int, default=100,
                        help="Number of simulations to use for coverage testing (default: 100)")
    parser.add_argument("--coverage-num-samples", type=int, default=1000,
                        help="Number of posterior samples per simulation for coverage testing (default: 1000)")
    parser.add_argument("--coverage-bootstrap", action="store_true",
                        help="Use bootstrap to estimate coverage uncertainties")
    parser.add_argument("--coverage-num-bootstrap", type=int, default=100,
                        help="Number of bootstrap iterations for coverage uncertainties (default: 100)")
    parser.add_argument("--coverage-seed", type=int, default=42,
                        help="Random seed for coverage testing")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="/home/tersenov/software/bar_impact/outputs/plots",
                        help="Directory to save output plots")
    parser.add_argument("--samples-dir", type=str, default="/home/tersenov/software/bar_impact/outputs/samples",
                        help="Directory to save posterior samples")
    
    # GPU configuration
    parser.add_argument("--gpu", type=str, default="0", 
                        help="GPU index to use")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage instead of GPU")
    
    parser.add_argument("--new-normalization", action="store_true",
                        help="Use data with new normalization.")
    
    parser.add_argument("--masked", action="store_true",
                        help="Use masked datavectors (Euclid-like sky mask)")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the sky mask in square degrees (default: 14000).")
    
    args = parser.parse_args()
    
    # Set fiducial type to match simulation type if not specified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type
    
    # Set mask suffixes
    if args.masked:
        area_tag = int(round(args.mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
        mask_label = f"masked_{area_tag}sqdeg"
    else:
        area_tag = None
        mask_suffix = ""
        mask_label = ""
    
    args.mask_area_tag = area_tag
    args.mask_suffix = mask_suffix
    args.mask_label = mask_label
    
    return args

def train_with_nan_retry(inference, checkpoint_path, args, params, data, max_retries=10):
    """
    Train NPE with automatic retry on NaN loss.
    
    Sometimes the loss initializes at NaN due to bad random initialization.
    This function will retry training with a fresh initialization if NaN is detected.
    
    Args:
        inference: NPE inference object (will be recreated on retry)
        checkpoint_path: Path to save checkpoints
        args: Command-line arguments
        params: Parameter array
        data: Data array
        max_retries: Maximum number of retry attempts
    
    Returns:
        tuple: (inference, metrics, density_estimator) on success
    
    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"Training attempt {attempt}/{max_retries}")
        print(f"{'='*60}")
        
        print(f"Training for {args.epochs} epochs...")
        metrics, density_estimator = inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            training_batch_size=args.batch_size
        )
        
        # Check for NaN in training or validation loss only
        # Note: We only check train_loss and val_loss, NOT test_loss
        # Test loss can sometimes be NaN due to evaluation issues even when training succeeded
        has_nan = False
        nan_source = None
        
        # Check training loss (indicates bad initialization)
        if hasattr(metrics, 'train_loss'):
            train_loss = metrics.train_loss
            if isinstance(train_loss, (list, np.ndarray)):
                if np.any(np.isnan(train_loss)):
                    has_nan = True
                    nan_source = 'training loss'
            elif np.isnan(train_loss):
                has_nan = True
                nan_source = 'training loss'
        
        # Check validation loss (indicates training instability)
        if not has_nan and hasattr(metrics, 'val_loss'):
            val_loss = metrics.val_loss
            if isinstance(val_loss, (list, np.ndarray)):
                if np.any(np.isnan(val_loss)):
                    has_nan = True
                    nan_source = 'validation loss'
            elif np.isnan(val_loss):
                has_nan = True
                nan_source = 'validation loss'
        
        # Warn about test loss NaN but don't trigger retry
        if hasattr(metrics, 'test_loss') and np.isnan(metrics.test_loss):
            print(f"⚠ Note: Test loss is NaN (evaluation issue, not affecting trained model)")
            
        if not has_nan:
            print(f"✓ Training completed successfully on attempt {attempt}")
            return inference, metrics, density_estimator
        
        print(f"⚠ NaN detected in {nan_source} during attempt {attempt}. Reinitializing...")
        
        if attempt < max_retries:
            # Reinitialize NPE for next attempt
            inference = NPE()
            inference = inference.append_simulations(params, data)
    
    raise RuntimeError(f"Training failed after {max_retries} attempts due to NaN loss")

def run_tarp_coverage_test(posterior, combined_data_vector, params, args):
    """
    Run TARP coverage test on the posterior estimator.
    
    This function samples from the posterior for multiple simulations from the
    training set, then uses TARP to assess whether the posterior coverage is well-calibrated.
    
    Args:
        posterior: Trained posterior object from NPE
        combined_data_vector: Full training data vector (n_sims, n_features)
        params: True parameter values for all simulations (n_sims, n_params)
        args: Command-line arguments
        
    Returns:
        ecp: Expected coverage probability
        alpha: Credibility levels
    """
    print("\n" + "="*60)
    print("Running TARP Coverage Test")
    print("="*60)
    
    # Select subset of simulations for coverage testing
    n_total_sims = combined_data_vector.shape[0]
    n_test_sims = min(args.coverage_num_sims, n_total_sims)
    
    # Randomly select test simulations
    np.random.seed(args.coverage_seed)
    test_indices = np.random.choice(n_total_sims, size=n_test_sims, replace=False)
    
    print(f"Using {n_test_sims} simulations from training set for coverage testing")
    print(f"Generating {args.coverage_num_samples} posterior samples per simulation")
    
    # Extract test data and parameters
    test_data = combined_data_vector[test_indices]
    test_params = params[test_indices]
    
    # Convert to numpy for TARP
    test_data_np = np.array(test_data)
    test_params_np = np.array(test_params)
    
    # Generate posterior samples for each test simulation
    all_samples = []
    master_key = random.PRNGKey(args.coverage_seed)
    
    print("Generating posterior samples for each test simulation...")
    for i, x_obs in enumerate(test_data_np):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_test_sims} simulations")
        
        sample_key, master_key = jax.random.split(master_key)
        samples = posterior.sample(
            x=x_obs, num_samples=args.coverage_num_samples, key=sample_key
        )
        all_samples.append(np.array(samples))
    
    # Stack samples into shape (n_samples, n_sims, n_dims)
    all_samples = np.stack(all_samples, axis=1)
    
    print(f"Posterior samples shape: {all_samples.shape}")
    print(f"True parameters shape: {test_params_np.shape}")
    
    # Compute TARP coverage
    print("\nComputing TARP coverage...")
    ecp, alpha = get_tarp_coverage(
        samples=all_samples,
        theta=test_params_np,
        references="random",
        metric="euclidean",
        num_alpha_bins=None,
        norm=True,
        bootstrap=args.coverage_bootstrap,
        num_bootstrap=args.coverage_num_bootstrap if args.coverage_bootstrap else 100,
        seed=args.coverage_seed
    )
    
    print("TARP coverage computation complete!")
    print("="*60 + "\n")
    
    return ecp, alpha

def plot_tarp_coverage(ecp, alpha, args, output_dir, filename_base):
    """
    Plot TARP coverage diagnostics.
    
    Args:
        ecp: Expected coverage probability from TARP
        alpha: Credibility levels from TARP
        args: Command-line arguments
        output_dir: Directory to save plots
        filename_base: Base filename for saved plot
    """
    plt.figure(figsize=(6, 6))
    
    if args.coverage_bootstrap:
        # ecp has shape (n_bootstrap, n_bins+1)
        # Compute mean and std across bootstrap samples
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std = np.std(ecp, axis=0)
        
        # Plot mean coverage with error band
        plt.plot(alpha, ecp_mean, 'b-', linewidth=2, label='TARP Coverage')
        plt.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std, 
                        alpha=0.3, color='blue', label='Bootstrap uncertainty')
    else:
        # ecp is 1D array
        plt.plot(alpha, ecp, 'b-', linewidth=2, label='TARP Coverage')
    
    # Plot ideal calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Ideal calibration')
    
    # Formatting
    plt.xlabel('Credibility Level', fontsize=12)
    plt.ylabel('Expected Coverage Probability', fontsize=12)
    plt.title('TARP Coverage Diagnostic', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save plot
    coverage_plot_path = os.path.join(output_dir, f"{filename_base}_tarp_coverage.pdf")
    plt.savefig(coverage_plot_path, transparent=True, dpi=300)
    print(f"Saved TARP coverage plot to {coverage_plot_path}")
    
    plt.close()
    
    # Save coverage data
    coverage_data_path = os.path.join(output_dir, f"{filename_base}_tarp_coverage_data.npz")
    if args.coverage_bootstrap:
        np.savez(coverage_data_path, ecp=ecp, alpha=alpha, 
                ecp_mean=ecp_mean, ecp_std=ecp_std,
                bootstrap=True)
    else:
        np.savez(coverage_data_path, ecp=ecp, alpha=alpha, bootstrap=False)
    print(f"Saved TARP coverage data to {coverage_data_path}")

def construct_paths(args):
    """Construct file paths based on provided arguments."""
    # Params file path - this doesn't change with bins
    params_filename = f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    params_path = os.path.join(args.data_dir, "grid", params_filename)
    
    # Parse bin options
    if args.bins:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
        is_multi_bin = True
    else:
        bin_indices = [args.bin]
        bin_desc = f"bin{args.bin}"
        is_multi_bin = False
    
    # Parse BNT bin options
    if args.bnt and args.bnt_bins:
        bnt_bin_indices = [int(b.strip()) for b in args.bnt_bins.split(',')]
        bnt_bin_desc = f"bntbins{''.join([str(b+1) for b in bnt_bin_indices])}"
        is_multi_bnt_bin = True
    elif args.bnt:
        bnt_bin_indices = [args.bnt_bin]
        bnt_bin_desc = f"bnt{args.bnt_bin+1}"
        is_multi_bnt_bin = False
    
    # Peak counts datavector paths for each bin
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    normalization_suffix = "_new_normalization" if args.new_normalization else ""
    peak_counts_paths = []
    fiducial_paths = []
    
    if args.bnt:
        peak_counts_prefix = "all_bnt_peak_counts"
        fiducial_prefix = "all_bnt_peak_counts"
        
        # For BNT mode, use the bnt bins
        for bnt_bin_idx in bnt_bin_indices:
            bin_spec = f"bin{bnt_bin_idx+1}"
            
            # Grid path
            peak_counts_filename = f"{peak_counts_prefix}_grid_{args.simulation_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            peak_counts_path = os.path.join(args.data_dir, "grid", peak_counts_filename)
            peak_counts_paths.append(peak_counts_path)
            
            # Fiducial path
            fiducial_filename = f"{fiducial_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
            fiducial_paths.append(fiducial_path)
        
        # Set bin description for paths and file names
        if is_multi_bnt_bin:
            bin_spec_for_output = bnt_bin_desc
        else:
            bin_spec_for_output = f"bnt{args.bnt_bin+1}"
    else:
        peak_counts_prefix = "all_peak_counts"
        fiducial_prefix = "all_peak_counts"
        
        # For non-BNT mode, use the regular bins
        for bin_idx in bin_indices:
            bin_spec = f"bin{bin_idx}"
            
            # Grid path
            peak_counts_filename = f"{peak_counts_prefix}_grid_{args.simulation_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            peak_counts_path = os.path.join(args.data_dir, "grid", peak_counts_filename)
            peak_counts_paths.append(peak_counts_path)
            
            # Fiducial path
            fiducial_filename = f"{fiducial_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
            fiducial_paths.append(fiducial_path)
        
        # Set bin description for paths and file names
        if is_multi_bin:
            bin_spec_for_output = bin_desc
        else:
            bin_spec_for_output = f"bin{args.bin}"
    
    return params_path, peak_counts_paths, fiducial_paths, bin_spec_for_output

def main():
    args = parse_arguments()
    
    # Construct file paths
    params_path, peak_counts_paths, fiducial_paths, bin_spec = construct_paths(args)
    print(f"Using parameters file: {params_path}")
    print(f"Using peak counts files: {peak_counts_paths}")
    print(f"Using fiducial files: {fiducial_paths}")
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Configure JAX device usage
    if args.force_cpu:
        jax.config.update('jax_platform_name', 'cpu')
        print("Forcing CPU usage")
    
    # Additional JAX configurations for stability
    jax.config.update("jax_enable_x64", True)  # Use 64-bit precision
    
    print("Device used by jax:", jax.devices())

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)

    # Load and process peak counts data from each bin
    peak_counts_full_bins = []
    for peak_counts_path in peak_counts_paths:
        peak_counts_full = np.load(peak_counts_path, allow_pickle=True)
        peak_counts_full_bins.append(peak_counts_full)
        print(f"Loaded peak counts data from {peak_counts_path}, shape: {peak_counts_full.shape}")

    # Extract scale data - either single scale, multiple scales, or per-bin scales
    if args.scales_per_bin:
        # Parse semicolon-separated per-bin scale specifications
        scales_per_bin = [[int(s.strip()) for s in bin_scales.split(',')] 
                          for bin_scales in args.scales_per_bin.split(';')]
        scale_desc = f"scales{''.join([str(s+1) for s in scales_per_bin[0]])}_perbin"
        print(f"Using per-bin scales configuration: {scales_per_bin}")
        
        # Verify we have the right number of scale specs for the bins
        if len(scales_per_bin) != len(peak_counts_full_bins):
            raise ValueError(f"Number of per-bin scale specs ({len(scales_per_bin)}) must match number of bins ({len(peak_counts_full_bins)})")
        
        # Process each bin's data with its specific scale selection
        bin_data_list = []
        for bin_idx, (peak_counts_full, scale_indices) in enumerate(zip(peak_counts_full_bins, scales_per_bin)):
            # Extract and concatenate scales for this bin
            peak_counts_scales = []
            for scale_idx in scale_indices:
                peak_counts_scales.append(peak_counts_full[:, scale_idx])
            
            # Concatenate along feature dimension (axis=1)
            bin_data = np.concatenate([scale_data.reshape(scale_data.shape[0], -1) 
                                      for scale_data in peak_counts_scales], axis=1)
            bin_data_list.append(bin_data)
            print(f"Bin {bin_idx}: using scales {[s+1 for s in scale_indices]}, shape: {bin_data.shape}")
        
        # Now concatenate all bins together
        peak_counts_scale = np.concatenate(bin_data_list, axis=1)
        
    elif args.scales:
        # Parse comma-separated scales
        scale_indices = [int(s.strip()) for s in args.scales.split(',')]
        scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
        print(f"Using multiple scales: {[s+1 for s in scale_indices]}")
        
        # Process each bin's data with selected scales
        bin_data_list = []
        for peak_counts_full in peak_counts_full_bins:
            # Extract and concatenate scales for this bin
            peak_counts_scales = []
            for scale_idx in scale_indices:
                peak_counts_scales.append(peak_counts_full[:, scale_idx])
            
            # Concatenate along feature dimension (axis=1)
            bin_data = np.concatenate([scale_data.reshape(scale_data.shape[0], -1) 
                                      for scale_data in peak_counts_scales], axis=1)
            bin_data_list.append(bin_data)
        
        # Now concatenate all bins together
        peak_counts_scale = np.concatenate(bin_data_list, axis=1)
        
    else:
        # Single scale case
        scale_desc = f"scale{args.scale+1}"
        print(f"Using single {scale_desc}")
        
        # Process each bin's data with the selected scale
        bin_data_list = []
        for peak_counts_full in peak_counts_full_bins:
            bin_data_list.append(peak_counts_full[:, args.scale])
        
        # Now concatenate all bins together
        peak_counts_scale = np.concatenate(bin_data_list, axis=1)
    
    print(f"Combined peak counts datavector shape: {peak_counts_scale.shape}")

    # Convert to JAX arrays with correct dtypes
    params = jnp.array(params, dtype=jnp.float32)
    peak_counts_scale = jnp.array(peak_counts_scale, dtype=jnp.float32)

    # Create checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration
    if args.bnt:
        datavector_desc = f"bnt_pc_weights_{args.simulation_type}_{bin_spec}_{scale_desc}"
    else:
        datavector_desc = f"pc_weights_{args.simulation_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        datavector_desc += f"_{args.mask_label}"
    if args.new_normalization:
        datavector_desc += "_new_normalization"
    
    checkpoint_name = f"cosmoGRID_{datavector_desc}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Checkpoint path: {checkpoint_path}")

    # Initialize NPE
    inference = NPE()
    inference = inference.append_simulations(params, peak_counts_scale)
    print("Added simulations to NPE")

    # Train or load the model
    if args.train:
        print(f"Starting NPE training for {args.epochs} epochs (with NaN retry)...")
        print(f"Training data shapes - params: {params.shape}, data: {peak_counts_scale.shape}")
        print(f"Data ranges - params: [{params.min():.3f}, {params.max():.3f}], data: [{peak_counts_scale.min():.3f}, {peak_counts_scale.max():.3f}]")
        
        try:
            inference, metrics, density_estimator = train_with_nan_retry(
                inference=inference,
                checkpoint_path=checkpoint_path,
                args=args,
                params=params,
                data=peak_counts_scale,
                max_retries=10
            )
            print("Training completed")
        except Exception as e:
            print(f"Training failed with error: {e}")
            print("This might be a GPU memory or numerical stability issue.")
            print("Try running with --force-cpu flag or check your CUDA installation.")
            raise
    else:
        print("Attempting to load existing model...")
        try:
            inference.load(checkpoint_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Please use --train to train a new model")
            return

    # Build posterior
    posterior = inference.build_posterior()
    print("Built posterior")

    # Run TARP coverage test if requested
    if args.run_coverage_test:
        # Construct filename base for coverage test outputs
        if args.bnt:
            simulation_type = "bnt"
        else:
            simulation_type = "nobaryons"
        
        # Parse bin description
        if args.bins:
            bin_spec = f"bins{''.join(map(str, args.bins))}"
        else:
            bin_spec = f"bin{args.bin}"
        
        # Parse scale description
        if args.scales:
            scale_desc = f"scales{''.join(map(str, args.scales))}"
        else:
            scale_desc = f"scale{args.scale}"
        
        # Noise and mask tags
        noise_tag = f"noisy_s{args.noise:.2f}"
        
        # Normalization tag
        norm_tag = "new_normalization"
        
        # Construct coverage filename base
        coverage_filename_base = f"cosmoGRID_peak_counts_{simulation_type}_{bin_spec}_{scale_desc}_{noise_tag}{args.mask_suffix}_{norm_tag}"
        if args.bin_ranges:
            bin_ranges_str = "_".join([f"{b[0]}-{b[1]}" for b in args.bin_ranges])
            coverage_filename_base += f"_cross_{bin_ranges_str}"
        
        # Run coverage test (use numpy versions before jnp conversion)
        ecp, alpha = run_tarp_coverage_test(posterior, np.array(peak_counts_scale), np.array(params), args)
        
        # Plot coverage
        plot_tarp_coverage(ecp, alpha, args, args.output_dir, coverage_filename_base)

    # Load fiducial peak counts data for each bin
    fid_means = []
    for fiducial_path in fiducial_paths:
        fid_full = np.load(fiducial_path, allow_pickle=True)
        fid_mean = np.mean(fid_full, axis=0)
        fid_means.append(fid_mean)
        print(f"Loaded fiducial peak counts data from {fiducial_path}, shape: {fid_full.shape}")
    
    # Process fiducial data according to scale selection
    fid_data_list = []
    if args.scales_per_bin:
        # Extract and concatenate scales for each bin's fiducial with per-bin configuration
        scales_per_bin = [[int(s.strip()) for s in bin_scales.split(',')] 
                          for bin_scales in args.scales_per_bin.split(';')]
        for fid_mean, scale_indices in zip(fid_means, scales_per_bin):
            bin_fid_scales = []
            for scale_idx in scale_indices:
                bin_fid_scales.append(fid_mean[scale_idx])
            
            # Concatenate scales for this bin's fiducial
            bin_fid_data = np.concatenate([scale_data.reshape(-1) 
                                         for scale_data in bin_fid_scales])
            fid_data_list.append(bin_fid_data)
    elif args.scales:
        # Extract and concatenate scales for each bin's fiducial
        for fid_mean in fid_means:
            bin_fid_scales = []
            for scale_idx in scale_indices:
                bin_fid_scales.append(fid_mean[scale_idx])
            
            # Concatenate scales for this bin's fiducial
            bin_fid_data = np.concatenate([scale_data.reshape(-1) 
                                         for scale_data in bin_fid_scales])
            fid_data_list.append(bin_fid_data)
    else:
        # Single scale case for each bin
        for fid_mean in fid_means:
            fid_data_list.append(fid_mean[args.scale])
    
    # Concatenate all bins' fiducial data
    fid_mean_scale = np.concatenate(fid_data_list)
    print(f"Combined fiducial peak counts data shape: {fid_mean_scale.shape}")

    # Sample from the posterior
    print("Sampling from posterior...")
    num_samples = args.num_samples
    master_key = random.PRNGKey(args.random_seed)
    sample_key, master_key = jax.random.split(master_key)
    samples = posterior.sample(
        x=fid_mean_scale, num_samples=num_samples, key=sample_key
    )
    print(f"Generated {num_samples} samples")

    # True parameters for plotting
    true_params = jnp.array([[2.600e-01, 8.400e-01, -1.000e+00, 6.736e+01, 9.649e-01, 4.930e-02]])

    # Create visualization
    labels = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
    
    # Create descriptive sample label
    fiducial_desc = f"{args.fiducial_type}"
    if args.noisy:
        fiducial_desc += f"_n{args.noise_level:.2f}"
    
    if args.bnt:
        sample_label = f"{args.simulation_type} BNT PC vs {fiducial_desc} fid, {bin_spec}, {scale_desc}"
    else:
        sample_label = f"{args.simulation_type} PC vs {fiducial_desc} fid, {bin_spec}, {scale_desc}"
    
    samples_bin_scale = MCSamples(
        samples=samples,
        names=labels,
        label=sample_label,
    )

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4

    g.triangle_plot([samples_bin_scale], filled=True,
                   line_args=[{'color': 'red'}],
                   contour_colors=['red'],
                   markers={
                       label: val for label, val in zip(labels, true_params[0])
                   })

    # Save plot with descriptive filename
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.bnt:
        plot_filename = f"posterior_bnt_pc_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    else:
        plot_filename = f"posterior_pc_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        plot_filename += f"_{args.mask_label}"
    if args.new_normalization:
        plot_filename += "_new_normalization"
    if args.run is not None:
        plot_filename += f"_run{args.run}"
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    if args.bnt:
        samples_filename = f"posterior_samples_bnt_pc_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    else:
        samples_filename = f"posterior_samples_pc_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        samples_filename += f"_{args.mask_label}"
    if args.new_normalization:
        samples_filename += "_new_normalization"
    if args.run is not None:
        samples_filename += f"_run{args.run}"
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin_scale.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")

if __name__ == "__main__":
    main()
