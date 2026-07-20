#!/usr/bin/env python3
"""
NPE Inference using locally processed L1 norm files.
This is a copy of run_npe_inference.py modified to read from
the output of example_l1_norm_processing.py
"""

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

def filter_zero_variance_bins(data, min_variance=1e-10, verbose=True):
    """
    Identify and filter out bins (features) with zero or near-zero variance.
    
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NPE inference using locally processed L1 norms")
    
    # Data configuration
    parser.add_argument("--l1-dir", type=str, 
                        default='./outputs/l1_norms',
                        help="Directory containing L1 norm outputs from example_l1_norm_processing.py")
    
    parser.add_argument("--params-dir", type=str,
                        default='/home/tersenov/CosmoGridV1/stage3_forecast/grid',
                        help="Directory containing cosmological parameters")
    
    parser.add_argument("--simulation-type", type=str, choices=["baryonified", "nobaryons"],
                        default="baryonified", 
                        help="Type of simulation to use for training (baryonified or nobaryons)")
    
    # Analysis configuration
    bin_group = parser.add_mutually_exclusive_group(required=False)
    bin_group.add_argument("--bin", type=int, default=2, 
                        help="Which redshift bin to analyze")
    bin_group.add_argument("--bins", type=str, 
                        help="Comma-separated list of redshift bins to analyze for tomographic inference")
    
    # Scale selection
    scale_group = parser.add_mutually_exclusive_group(required=False)
    scale_group.add_argument("--scale", type=int, default=0, 
                        help="Which scale index to analyze (0-indexed). Use for single scale analysis.")
    scale_group.add_argument("--scales", type=str, 
                        help="Comma-separated list of scale indices to analyze (0-indexed). Use for multi-scale analysis.")
    
    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    parser.add_argument("--masked", action="store_true",
                        help="Use masked datavectors")
    parser.add_argument("--mask-area-sqdeg", type=float, default=35002.0,
                        help="Area of the sky mask in square degrees")
    
    # Fiducial configuration  
    parser.add_argument("--fiducial-type", type=str, choices=["baryonified", "nobaryons"],
                        default=None,
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
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./outputs/plots",
                        help="Directory to save output plots")
    parser.add_argument("--samples-dir", type=str, default="./outputs/samples",
                        help="Directory to save posterior samples")
    
    # GPU configuration
    parser.add_argument("--gpu", type=str, default="0", 
                        help="GPU index to use")
    
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

def construct_paths(args):
    """Construct file paths based on provided arguments."""
    # Params file path
    params_filename = f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    params_path = os.path.join(args.params_dir, params_filename)
    
    # Parse bin options
    if args.bins:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
    else:
        bin_indices = [args.bin]
        bin_desc = f"bin{args.bin}"
    
    # Datavector paths for each bin
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    l1_paths = []
    fiducial_paths = []
    
    # Use the "combined_l1_norms_" prefix from example_l1_norm_processing.py
    for bin_idx in bin_indices:
        # Grid path - matches example_l1_norm_processing.py output
        l1_filename = f"combined_l1_norms_grid_{args.simulation_type}_bin{bin_idx}{args.mask_suffix}{noise_suffix}.npy"
        l1_path = os.path.join(args.l1_dir, l1_filename)
        l1_paths.append(l1_path)
        
        # Fiducial path - matches example_l1_norm_processing.py output
        fiducial_filename = f"combined_l1_norms_fiducial_{args.fiducial_type}_bin{bin_idx}{args.mask_suffix}{noise_suffix}.npy"
        fiducial_path = os.path.join(args.l1_dir, fiducial_filename)
        fiducial_paths.append(fiducial_path)
    
    return params_path, l1_paths, fiducial_paths, bin_desc

def main():
    args = parse_arguments()
    
    # Construct file paths
    params_path, l1_paths, fiducial_paths, bin_spec = construct_paths(args)
    print(f"Using parameters file: {params_path}")
    print(f"Using datavector files: {l1_paths}")
    print(f"Using fiducial files: {fiducial_paths}")
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Device used by jax:", jax.devices())

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)
    print(f"Loaded parameters: shape {params.shape}")

    # Load and process data from each bin
    l1_full_bins = []
    for l1_path in l1_paths:
        l1_full = np.load(l1_path, allow_pickle=True)
        l1_full_bins.append(l1_full)
        print(f"Loaded data from {l1_path}, shape: {l1_full.shape}")

    # Extract scale data - either single scale or multiple scales
    if args.scales:
        # Parse comma-separated scales
        scale_indices = [int(s.strip()) for s in args.scales.split(',')]
        scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
        print(f"Using multiple scales: {[s+1 for s in scale_indices]} (1-indexed)")
        
        # Process each bin's data with selected scales
        bin_data_list = []
        for i, l1_full in enumerate(l1_full_bins):
            # Extract and concatenate scales for this bin
            l1_scales = []
            for scale_idx in scale_indices:
                scale_data = l1_full[:, scale_idx]  # Shape: (n_samples, nbins)
                l1_scales.append(scale_data)
            
            # Concatenate along feature dimension (axis=1)
            bin_data = np.concatenate(l1_scales, axis=1)
            bin_data_list.append(bin_data)
            print(f"  Bin {i+1}: selected scales {[s+1 for s in scale_indices]}, shape {bin_data.shape}")
        
        # Concatenate all bins together
        l1_scale = np.concatenate(bin_data_list, axis=1)
        
    else:
        # Single scale case
        scale_desc = f"scale{args.scale+1}"
        print(f"Using single {scale_desc} (1-indexed)")
        
        # Process each bin's data with the selected scale
        bin_data_list = []
        for i, l1_full in enumerate(l1_full_bins):
            scale_data = l1_full[:, args.scale]  # Shape: (n_samples, nbins)
            bin_data_list.append(scale_data)
            print(f"  Bin {i+1}: scale {args.scale+1}, shape {scale_data.shape}")
        
        # Concatenate all bins together
        l1_scale = np.concatenate(bin_data_list, axis=1)
    
    print(f"Combined datavector shape (before filtering): {l1_scale.shape}")
    
    # Filter out zero-variance bins to prevent NaN loss during training
    valid_bin_mask, n_removed = filter_zero_variance_bins(l1_scale, min_variance=1e-10, verbose=True)
    l1_scale_filtered = l1_scale[:, valid_bin_mask]
    print(f"Combined datavector shape (after filtering): {l1_scale_filtered.shape}")
    
    # Use the filtered data for training
    l1_scale = l1_scale_filtered

    # Convert to JAX arrays
    params_jax = jnp.array(params)
    l1_scale_jax = jnp.array(l1_scale)

    # Create checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration
    datavector_desc = f"{args.simulation_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        datavector_desc += f"_{args.mask_label}"
    
    checkpoint_name = f"npe_local_{datavector_desc}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Checkpoint path: {checkpoint_path}")

    # Initialize NPE
    inference = NPE()
    inference = inference.append_simulations(params_jax, l1_scale_jax)
    print("Added simulations to NPE")

    # Train or load the model
    if args.train:
        print(f"Starting NPE training for {args.epochs} epochs...")
        metrics, density_estimator = inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            training_batch_size=args.batch_size
        )
        print("Training completed")
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

    # Load fiducial data for each bin
    print("\nLoading fiducial data...")
    fid_means = []
    for fiducial_path in fiducial_paths:
        if not os.path.exists(fiducial_path):
            print(f"ERROR: Fiducial file not found: {fiducial_path}")
            print("Please run example_l1_norm_processing.py with PROCESS_FIDUCIAL=True")
            return
        
        fid_full = np.load(fiducial_path, allow_pickle=True)
        fid_mean = np.mean(fid_full, axis=0)
        fid_means.append(fid_mean)
        print(f"Loaded fiducial data from {fiducial_path}, shape: {fid_full.shape}")
    
    # Process fiducial data according to scale selection
    fid_data_list = []
    if args.scales:
        # Extract and concatenate scales for each bin's fiducial
        for i, fid_mean in enumerate(fid_means):
            bin_fid_scales = []
            for scale_idx in scale_indices:
                scale_data = fid_mean[scale_idx]  # Shape: (nbins,)
                bin_fid_scales.append(scale_data)
            
            # Concatenate scales for this bin's fiducial
            bin_fid_data = np.concatenate(bin_fid_scales)
            fid_data_list.append(bin_fid_data)
    else:
        # Single scale case for each bin
        for i, fid_mean in enumerate(fid_means):
            scale_data = fid_mean[args.scale]  # Shape: (nbins,)
            fid_data_list.append(scale_data)
    
    # Concatenate all bins' fiducial data
    fid_mean_scale = np.concatenate(fid_data_list)
    print(f"Combined fiducial data shape (before filtering): {fid_mean_scale.shape}")
    
    # Apply the same zero-variance bin mask used for training data
    fid_mean_scale = fid_mean_scale[valid_bin_mask]
    print(f"Combined fiducial data shape (after filtering): {fid_mean_scale.shape}")

    # Sample from the posterior
    print("\nSampling from posterior...")
    num_samples = args.num_samples
    master_key = random.PRNGKey(args.random_seed)
    sample_key, master_key = jax.random.split(master_key)
    samples = posterior.sample(
        x=fid_mean_scale, num_samples=num_samples, key=sample_key
    )
    samples_np = np.array(samples)
    print(f"Generated {num_samples} samples")
    
    # Print sample statistics
    print(f"\nPosterior sample statistics:")
    print(f"  Mean: {np.mean(samples_np, axis=0)}")
    print(f"  Std:  {np.std(samples_np, axis=0)}")
    print(f"  Min:  {np.min(samples_np, axis=0)}")
    print(f"  Max:  {np.max(samples_np, axis=0)}")

    # True parameters for plotting (fiducial cosmology)
    true_params = jnp.array([[2.600e-01, 8.400e-01, -1.000e+00, 6.736e+01, 9.649e-01, 4.930e-02]])

    # Create visualization
    labels = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
    
    # Create descriptive sample label
    fiducial_desc = f"{args.fiducial_type}"
    if args.noisy:
        fiducial_desc += f"_n{args.noise_level:.2f}"
    
    sample_label = f"{args.simulation_type} DV vs {fiducial_desc} fid, {bin_spec}, {scale_desc}"
    
    samples_mc = MCSamples(
        samples=samples_np,
        names=labels,
        label=sample_label,
    )

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4

    g.triangle_plot([samples_mc], filled=True,
                   line_args=[{'color': 'blue'}],
                   contour_colors=['blue'],
                   markers={
                       label: val for label, val in zip(labels, true_params[0])
                   })

    # Save plot with descriptive filename
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_filename = f"posterior_local_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        plot_filename += f"_{args.mask_label}"
    plot_filename += ".pdf"
    
    plot_path = os.path.join(args.output_dir, plot_filename)
    plt.savefig(plot_path, transparent=True)
    print(f"\nSaved plot to {plot_path}")

    # Save posterior samples
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = f"posterior_samples_local_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        samples_filename += f"_{args.mask_label}"
    samples_filename += ".npy"
    
    samples_path = os.path.join(args.samples_dir, samples_filename)
    np.save(samples_path, samples_np)
    print(f"Saved posterior samples to {samples_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
