#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/run_npe_peak_counts_inference.py

import os
import argparse
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
from jaxili.inference import NPE
from getdist import plots, MCSamples

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
    
    # Create a mutually exclusive group for scale selection
    scale_group = parser.add_mutually_exclusive_group(required=False)
    scale_group.add_argument("--scale", type=int, default=0, 
                        help="Which scale index to analyze (0-indexed). Use for single scale analysis.")
    scale_group.add_argument("--scales", type=str, 
                        help="Comma-separated list of scale indices to analyze (0-indexed). Use for multi-scale analysis.")
    
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
    
    args = parser.parse_args()
    
    # Set fiducial type to match simulation type if not specified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type
    
    return args

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
    
    # Peak counts datavector paths for each bin
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    normalization_suffix = "_new_normalization" if args.new_normalization else ""
    peak_counts_paths = []
    fiducial_paths = []
    
    peak_counts_prefix = "all_peak_counts"
    fiducial_prefix = "all_peak_counts"
    
    # For peak counts mode, use the regular bins
    for bin_idx in bin_indices:
        bin_spec = f"bin{bin_idx}"
        
        # Grid path
        peak_counts_filename = f"{peak_counts_prefix}_grid_{args.simulation_type}_{bin_spec}{noise_suffix}{normalization_suffix}.npy"
        peak_counts_path = os.path.join(args.data_dir, "grid", peak_counts_filename)
        peak_counts_paths.append(peak_counts_path)
        
        # Fiducial path
        fiducial_filename = f"{fiducial_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{noise_suffix}{normalization_suffix}.npy"
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

    # Extract scale data - either single scale or multiple scales
    if args.scales:
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
    datavector_desc = f"ps_weights_{args.simulation_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
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
        print(f"Starting NPE training for {args.epochs} epochs...")
        print(f"Training data shapes - params: {params.shape}, data: {peak_counts_scale.shape}")
        print(f"Data ranges - params: [{params.min():.3f}, {params.max():.3f}], data: [{peak_counts_scale.min():.3f}, {peak_counts_scale.max():.3f}]")
        
        try:
            metrics, density_estimator = inference.train(
                checkpoint_path=checkpoint_path,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                training_batch_size=args.batch_size
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

    # Load fiducial peak counts data for each bin
    fid_means = []
    for fiducial_path in fiducial_paths:
        fid_full = np.load(fiducial_path, allow_pickle=True)
        fid_mean = np.mean(fid_full, axis=0)
        fid_means.append(fid_mean)
        print(f"Loaded fiducial peak counts data from {fiducial_path}, shape: {fid_full.shape}")
    
    # Process fiducial data according to scale selection
    fid_data_list = []
    if args.scales:
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
    
    plot_filename = f"posterior_pc_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.new_normalization:
        plot_filename += "_new_normalization"
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = f"posterior_samples_pc_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.new_normalization:
        samples_filename += "_new_normalization"
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin_scale.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")

if __name__ == "__main__":
    main()
