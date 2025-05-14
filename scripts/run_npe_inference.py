#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/run_npe_inference.py

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
    parser = argparse.ArgumentParser(description="Run NPE inference on CosmoGRID simulations")
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, 
                        default='/home/tersenov/CosmoGridV1/stage3_forecast',
                        help="Base directory for data")
    
    parser.add_argument("--simulation-type", type=str, choices=["baryonified", "nobaryons"],
                        default="baryonified", 
                        help="Type of simulation to use for training (baryonified or nobaryons)")
    
    # Analysis configuration
    parser.add_argument("--bin", type=int, default=2, 
                        help="Which redshift bin to analyze")
    
    # BNT configuration
    parser.add_argument("--bnt", action="store_true", 
                        help="Use BNT-transformed data")
    parser.add_argument("--bnt-bin", type=int, default=3,
                        help="Which BNT bin to analyze (0-3, default=3 corresponds to bin4)")
    
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
    
    args = parser.parse_args()
    
    # Set fiducial type to match simulation type if not specified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type
    
    return args

def construct_paths(args):
    """Construct file paths based on provided arguments."""
    # Params file path
    params_filename = f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    params_path = os.path.join(args.data_dir, "grid", params_filename)
    
    # Datavector path
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    
    # Handle BNT vs regular l1-norms
    if args.bnt:
        # For BNT data, we use the bnt-bin+1 as the bin number in the file name
        bin_spec = f"bin{args.bnt_bin+1}"
        l1_prefix = "all_bnt_l1_norms"
        fiducial_prefix = "all_bnt_l1_norms"
    else:
        # For regular data, we use the standard bin
        bin_spec = f"bin{args.bin}"
        l1_prefix = "all_l1_norms"
        fiducial_prefix = "all_l1_norms"
    
    l1_filename = f"{l1_prefix}_grid_{args.simulation_type}_{bin_spec}{noise_suffix}.npy"
    l1_path = os.path.join(args.data_dir, "grid", l1_filename)
    
    # Fiducial path - use same noise settings as datavector
    fiducial_filename = f"{fiducial_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{noise_suffix}.npy"
    fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
    
    return params_path, l1_path, fiducial_path

def main():
    args = parse_arguments()
    
    # Construct file paths
    params_path, l1_path, fiducial_path = construct_paths(args)
    print(f"Using parameters file: {params_path}")
    print(f"Using datavector file: {l1_path}")
    print(f"Using fiducial file: {fiducial_path}")
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Device used by jax:", jax.devices())

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)

    # Load bin data
    l1_full = np.load(l1_path, allow_pickle=True)
    print(f"Loaded data shapes: params {params.shape}, l1_full {l1_full.shape}")

    # Extract scale data - either single scale or multiple scales
    if args.scales:
        # Parse comma-separated scales
        scale_indices = [int(s.strip()) for s in args.scales.split(',')]
        scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
        print(f"Using multiple scales: {[s+1 for s in scale_indices]}")
        
        # Extract and concatenate scales
        l1_scales = []
        for scale_idx in scale_indices:
            l1_scales.append(l1_full[:, scale_idx])
        
        # Concatenate along feature dimension (axis=1)
        l1_scale = np.concatenate([scale_data.reshape(scale_data.shape[0], -1) 
                                  for scale_data in l1_scales], axis=1)
    else:
        # Single scale case (existing behavior)
        l1_scale = l1_full[:, args.scale]
        scale_desc = f"scale{args.scale+1}"
        print(f"Using single {scale_desc}")
    
    print(f"L1 {scale_desc} shape: {l1_scale.shape}")

    # Convert to JAX arrays
    params = jnp.array(params)
    l1_scale = jnp.array(l1_scale)

    # Create checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration
    # Include BNT information in the checkpoint name if using BNT
    if args.bnt:
        bin_spec = f"bnt{args.bnt_bin+1}"
    else:
        bin_spec = f"bin{args.bin}"
        
    datavector_desc = f"{args.simulation_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
    
    checkpoint_name = f"cosmoGRID_weights_{datavector_desc}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Checkpoint path: {checkpoint_path}")

    # Initialize NPE
    inference = NPE()
    inference = inference.append_simulations(params, l1_scale)
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
            # Note: Replace this with actual model loading code if jaxili.NPE requires different loading procedure
            inference.load(checkpoint_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Please use --train to train a new model")
            return

    # Build posterior
    posterior = inference.build_posterior()
    print("Built posterior")

    # Load fiducial data
    fid_full = np.load(fiducial_path, allow_pickle=True)
    print(f"Fiducial data shape: {np.array(fid_full).shape}")

    # Compute mean of fiducial data
    fid_mean = np.mean(fid_full, axis=0)
    
    # Process fiducial data according to scale selection
    if args.scales:
        # Extract and concatenate scales for fiducial
        fid_scales = []
        for scale_idx in scale_indices:
            fid_scales.append(fid_mean[scale_idx])
        
        # Concatenate scales for fiducial
        fid_mean_scale = np.concatenate([scale_data.reshape(-1) 
                                       for scale_data in fid_scales])
    else:
        # Single scale case
        fid_mean_scale = fid_mean[args.scale]
    
    print(f"Fiducial mean {scale_desc} shape: {fid_mean_scale.shape}")

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
    
    # Include BNT information in the label if using BNT
    if args.bnt:
        bin_desc = f"bnt{args.bnt_bin+1}"
    else:
        bin_desc = f"bin{args.bin}"
        
    sample_label = f"{args.simulation_type} DV vs {fiducial_desc} fid, {bin_desc}, {scale_desc}"
    
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
    
    # Include BNT information in the filename if using BNT
    if args.bnt:
        bin_spec = f"bnt{args.bnt_bin+1}"
    else:
        bin_spec = f"bin{args.bin}"
        
    plot_filename = f"posterior_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = f"posterior_samples_{args.simulation_type}_vs_{args.fiducial_type}_{bin_spec}_{scale_desc}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin_scale.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")

if __name__ == "__main__":
    main()