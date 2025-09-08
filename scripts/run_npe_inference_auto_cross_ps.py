#!/usr/bin/env python3
"""
NPE Inference with Auto + Cross Power Spectra - Run NPE inference using combined auto and cross power spectra.
"""

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
    parser = argparse.ArgumentParser(description="Run NPE inference on CosmoGRID auto + cross power spectra")
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, 
                        default='/home/tersenov/CosmoGridV1/stage3_forecast',
                        help="Base directory for data")
    
    parser.add_argument("--simulation-type", type=str, choices=["baryonified", "nobaryons"],
                        default="baryonified", 
                        help="Type of simulation to use for training (baryonified or nobaryons)")
    
    # Analysis configuration - for auto+cross we need multiple bins
    parser.add_argument("--bins", type=str, default="1,2,3,4",
                        help="Comma-separated list of redshift bins to analyze (default: 1,2,3,4)")
    
    # BNT configuration
    parser.add_argument("--bnt", action="store_true", 
                        help="Use BNT-transformed power spectra")
    parser.add_argument("--bnt-bins", type=str, default="0,1,2,3",
                        help="Comma-separated list of BNT bins to analyze (default: 0,1,2,3)")

    # Power Spectrum processing options
    parser.add_argument("--lower-cut", type=int, default=30,
                        help="Lower multipole cut for the power spectrum (l_min).")
    parser.add_argument("--upper-cut", type=int, default=1024,
                        help="Upper multipole cut for the power spectrum (l_max).")
    parser.add_argument("--rebin", type=int, default=1,
                        help="Rebinning factor for the power spectrum. Default is 1 (no rebinning).")

    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    
    # Cross power spectra configuration
    parser.add_argument("--cross-data-dir", type=str,
                        help="Directory containing aggregated cross power spectra files. If not specified, uses data-dir.")
    parser.add_argument("--auto-only", action="store_true",
                        help="Use only auto power spectra (for comparison)")
    parser.add_argument("--cross-only", action="store_true",
                        help="Use only cross power spectra (for comparison)")
    
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
    
    # Debugging options
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information about data loading and processing")
    
    args = parser.parse_args()
    
    # Set fiducial type to match simulation type if not specified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type
    
    # Set cross data directory if not specified
    if args.cross_data_dir is None:
        args.cross_data_dir = args.data_dir
    
    # Validate conflicting options
    if args.auto_only and args.cross_only:
        parser.error("Cannot specify both --auto-only and --cross-only")
    
    return args

def construct_auto_paths(args):
    """Construct file paths for auto power spectra based on provided arguments."""
    # Params file path - this doesn't change with bins
    params_filename = f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    params_path = os.path.join(args.data_dir, "grid", params_filename)
    
    # Parse bin options
    if args.bnt:
        bin_indices = [int(b.strip()) for b in args.bnt_bins.split(',')]
        bin_desc = f"bins{''.join([str(b+1) for b in bin_indices])}"
        data_prefix = "all_bnt_cls"
        bin_prefix = "bin"
        bin_suffix_list = [f"{b+1}" for b in bin_indices]
    else:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
        data_prefix = "all_cls"
        bin_prefix = "bin"
        bin_suffix_list = bin_indices

    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    
    auto_data_paths = []
    auto_fiducial_paths = []
    
    for i, bin_idx in enumerate(bin_indices):
        bin_spec = f"{bin_prefix}{bin_suffix_list[i]}"
        # Auto data path (grid)
        data_filename = f"{data_prefix}_grid_{args.simulation_type}_{bin_spec}{noise_suffix}.npy"
        data_path = os.path.join(args.data_dir, "new_grid", data_filename)
        if not os.path.exists(data_path):
             data_path = os.path.join(args.data_dir, "grid", data_filename)
        auto_data_paths.append(data_path)
        
        # Auto fiducial path
        fiducial_filename = f"{data_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{noise_suffix}.npy"
        fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
        auto_fiducial_paths.append(fiducial_path)
        
    return params_path, auto_data_paths, auto_fiducial_paths, bin_desc

def construct_cross_paths(args, bin_desc):
    """Construct file paths for aggregated cross power spectra."""
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    
    if args.bnt:
        # For BNT, use the correct BNT cross power spectrum naming
        data_filename = f"all_bnt_cross_cls_grid_{args.simulation_type}_{bin_desc}{noise_suffix}.npy"
        fiducial_filename = f"all_bnt_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{noise_suffix}.npy"
    else:
        # For regular bins
        data_filename = f"all_cross_cls_grid_{args.simulation_type}_{bin_desc}{noise_suffix}.npy"
        fiducial_filename = f"all_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{noise_suffix}.npy"
    
    # Look for cross power spectra files in the cross data directory
    cross_data_path = os.path.join(args.cross_data_dir, "new_grid", data_filename)
    if not os.path.exists(cross_data_path):
        cross_data_path = os.path.join(args.cross_data_dir, "grid", data_filename)
    
    cross_fiducial_path = os.path.join(args.cross_data_dir, "fiducial", fiducial_filename)
    
    return cross_data_path, cross_fiducial_path

def rebin_cls(cls, factor=2):
    """Rebins a 1D power spectrum by a given factor."""
    if factor <= 1:
        return cls
    n = len(cls) // factor
    cls_rebinned = np.zeros(n)
    for i in range(n):
        cls_rebinned[i] = np.mean(cls[i*factor:(i+1)*factor])
    return cls_rebinned

def load_and_process_auto_spectra(auto_data_paths, args):
    """Load and process auto power spectra."""
    auto_data_list = []
    for data_path in auto_data_paths:
        cls_full = np.load(data_path, allow_pickle=True)
        auto_data_list.append(cls_full)
        if args.verbose:
            print(f"Loaded auto data from {os.path.basename(data_path)}, shape: {cls_full.shape}")
    
    # Process auto power spectra: apply cuts and rebinning
    processed_auto_list = []
    for cls_full in auto_data_list:
        # Apply cuts
        cls_cut = cls_full[:, args.lower_cut:args.upper_cut]
        
        # Apply rebinning if specified
        if args.rebin > 1:
            cls_rebinned_list = [rebin_cls(cl, args.rebin) for cl in cls_cut]
            cls_processed = np.array(cls_rebinned_list)
        else:
            cls_processed = cls_cut
        
        processed_auto_list.append(cls_processed)
    
    # Concatenate all auto bins together
    auto_data_vector = np.concatenate(processed_auto_list, axis=1)
    
    return auto_data_vector

def load_and_process_cross_spectra(cross_data_path, args):
    """Load and process aggregated cross power spectra."""
    cross_cls_full = np.load(cross_data_path, allow_pickle=True)
    if args.verbose:
        print(f"Loaded cross data from {os.path.basename(cross_data_path)}, shape: {cross_cls_full.shape}")
    
    # Apply cuts and rebinning to cross power spectra
    # Apply cuts
    cross_cls_cut = cross_cls_full[:, args.lower_cut:args.upper_cut]
    
    # Apply rebinning if specified
    if args.rebin > 1:
        cross_cls_rebinned_list = [rebin_cls(cl, args.rebin) for cl in cross_cls_cut]
        cross_data_vector = np.array(cross_cls_rebinned_list)
    else:
        cross_data_vector = cross_cls_cut
    
    return cross_data_vector

def load_and_process_auto_fiducial(auto_fiducial_paths, args):
    """Load and process auto fiducial data."""
    auto_fid_means = []
    for fiducial_path in auto_fiducial_paths:
        fid_full = np.load(fiducial_path, allow_pickle=True)
        fid_mean = np.mean(fid_full, axis=0)
        auto_fid_means.append(fid_mean)
        if args.verbose:
            print(f"Loaded auto fiducial data from {os.path.basename(fiducial_path)}, shape: {fid_full.shape}")
    
    # Process auto fiducial data according to cuts and rebinning
    auto_fid_data_list = []
    for fid_mean in auto_fid_means:
        # Apply cuts
        fid_cut = fid_mean[args.lower_cut:args.upper_cut]
        
        # Apply rebinning
        if args.rebin > 1:
            fid_processed = rebin_cls(fid_cut, args.rebin)
        else:
            fid_processed = fid_cut
        
        auto_fid_data_list.append(fid_processed)
    
    # Concatenate all auto bins' fiducial data
    auto_fid_mean_processed = np.concatenate(auto_fid_data_list)
    
    return auto_fid_mean_processed

def load_and_process_cross_fiducial(cross_fiducial_path, args):
    """Load and process cross fiducial data."""
    cross_fid_full = np.load(cross_fiducial_path, allow_pickle=True)
    cross_fid_mean = np.mean(cross_fid_full, axis=0)
    if args.verbose:
        print(f"Loaded cross fiducial data from {os.path.basename(cross_fiducial_path)}, shape: {cross_fid_full.shape}")
    
    # Process cross fiducial data according to cuts and rebinning
    # Apply cuts
    cross_fid_cut = cross_fid_mean[args.lower_cut:args.upper_cut]
    
    # Apply rebinning
    if args.rebin > 1:
        cross_fid_processed = rebin_cls(cross_fid_cut, args.rebin)
    else:
        cross_fid_processed = cross_fid_cut
    
    return cross_fid_processed

def main():
    args = parse_arguments()
    
    # Construct file paths for auto power spectra
    params_path, auto_data_paths, auto_fiducial_paths, bin_desc = construct_auto_paths(args)
    
    # Construct file paths for cross power spectra
    cross_data_path, cross_fiducial_path = construct_cross_paths(args, bin_desc)
    
    print(f"Using parameters file: {params_path}")
    print(f"Using auto datavector files: {auto_data_paths}")
    print(f"Using cross datavector file: {cross_data_path}")
    print(f"Using auto fiducial files: {auto_fiducial_paths}")
    print(f"Using cross fiducial file: {cross_fiducial_path}")
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Device used by jax:", jax.devices())

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)

    # Load and process data
    data_vector_parts = []
    fid_vector_parts = []
    
    if not args.cross_only:
        # Load auto power spectra
        auto_data_vector = load_and_process_auto_spectra(auto_data_paths, args)
        auto_fid_vector = load_and_process_auto_fiducial(auto_fiducial_paths, args)
        
        data_vector_parts.append(auto_data_vector)
        fid_vector_parts.append(auto_fid_vector)
        
        print(f"Auto datavector shape: {auto_data_vector.shape}")
        print(f"Auto fiducial shape: {auto_fid_vector.shape}")
    
    if not args.auto_only:
        # Load cross power spectra
        cross_data_vector = load_and_process_cross_spectra(cross_data_path, args)
        cross_fid_vector = load_and_process_cross_fiducial(cross_fiducial_path, args)
        
        data_vector_parts.append(cross_data_vector)
        fid_vector_parts.append(cross_fid_vector)
        
        print(f"Cross datavector shape: {cross_data_vector.shape}")
        print(f"Cross fiducial shape: {cross_fid_vector.shape}")
    
    # Combine auto and cross power spectra
    combined_data_vector = np.concatenate(data_vector_parts, axis=1)
    combined_fid_vector = np.concatenate(fid_vector_parts)
    
    print(f"Combined datavector shape: {combined_data_vector.shape}")
    print(f"Combined fiducial shape: {combined_fid_vector.shape}")

    # Process power spectra description
    ps_desc = f"l{args.lower_cut}-{args.upper_cut}"
    if args.rebin > 1:
        ps_desc += f"_r{args.rebin}"
    print(f"Processing power spectra with: {ps_desc}")

    # Convert to JAX arrays
    params = jnp.array(params)
    combined_data_vector = jnp.array(combined_data_vector)

    # Create checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration
    if args.auto_only:
        spectra_type = "auto"
    elif args.cross_only:
        spectra_type = "cross"
    else:
        spectra_type = "auto_cross"
    
    # Add BNT prefix to checkpoint name if using BNT data
    if args.bnt:
        datavector_desc = f"{args.simulation_type}_bnt_{bin_desc}_{ps_desc}_{spectra_type}"
        checkpoint_name = f"cosmoGRID_bnt_ps_weights_{datavector_desc}"
    else:
        datavector_desc = f"{args.simulation_type}_{bin_desc}_{ps_desc}_{spectra_type}"
        checkpoint_name = f"cosmoGRID_ps_weights_{datavector_desc}"
    
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Checkpoint path: {checkpoint_path}")

    # Initialize NPE
    inference = NPE()
    inference = inference.append_simulations(params, combined_data_vector)
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

    # Sample from the posterior
    print("Sampling from posterior...")
    num_samples = args.num_samples
    master_key = random.PRNGKey(args.random_seed)
    sample_key, master_key = jax.random.split(master_key)
    samples = posterior.sample(
        x=combined_fid_vector, num_samples=num_samples, key=sample_key
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
    
    if args.auto_only:
        analysis_type = "BNT Auto Cls" if args.bnt else "Auto Cls"
    elif args.cross_only:
        analysis_type = "BNT Cross Cls" if args.bnt else "Cross Cls"
    else:
        analysis_type = "BNT Auto+Cross Cls" if args.bnt else "Auto+Cross Cls"
    
    sample_label = f"{args.simulation_type} {analysis_type} vs {fiducial_desc} fid, {bin_desc}, {ps_desc}"
    
    samples_bin_scale = MCSamples(
        samples=samples,
        names=labels,
        label=sample_label,
    )

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4

    g.triangle_plot([samples_bin_scale], filled=True,
                   line_args=[{'color': 'blue'}],
                   contour_colors=['blue'],
                   markers={
                       label: val for label, val in zip(labels, true_params[0])
                   })

    # Save plot with descriptive filename
    os.makedirs(args.output_dir, exist_ok=True)
    
    bnt_prefix = "bnt_" if args.bnt else ""
    plot_filename = f"posterior_{bnt_prefix}ps_{spectra_type}_{args.simulation_type}_vs_{args.fiducial_type}_{bin_desc}_{ps_desc}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = f"posterior_samples_{bnt_prefix}ps_{spectra_type}_{args.simulation_type}_vs_{args.fiducial_type}_{bin_desc}_{ps_desc}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin_scale.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")

if __name__ == "__main__":
    main()
