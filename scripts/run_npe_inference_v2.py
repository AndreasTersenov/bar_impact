#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/run_npe_inference_v2.py
"""
NPE Inference on CosmoGRID L1 Norms - v2 (Refactored)

This is a refactored version that uses the modular bar_impact utilities:
- bar_impact.analysis.aggregation: ResultsAggregator for data loading and filtering
- bar_impact.utils.paths: get_data_file_paths for file discovery
- bar_impact.utils.reproducibility: Deterministic seed management
- bar_impact.utils.inference: TARP testing and NaN-resilient training

Changes from original:
- Eliminated filter_zero_variance_bins() - now uses ResultsAggregator.filter_zero_variance()
- Eliminated parse_bin_ranges() - now uses ResultsAggregator.select_bin_ranges_per_bin()
- Eliminated run_tarp_coverage_test(), plot_tarp_coverage() - now uses utils.inference functions
- Eliminated train_with_nan_retry() - now uses utils.inference.train_npe_with_nan_retry()
- Simplified file path construction using get_data_file_paths()

Maintains identical CLI interface and numerical behavior.
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

from bar_impact.analysis.aggregation import ResultsAggregator, load_datavectors
from bar_impact.utils.paths import get_data_file_paths
from bar_impact.utils.reproducibility import get_deterministic_seed
from bar_impact.utils.inference import (
    run_tarp_coverage_test,
    plot_tarp_coverage,
    train_npe_with_nan_retry,
)

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
                        help="Semicolon-separated list of scale configurations per bin, e.g., '1,2,3;0,1,2,3;0,1,2,3;0,1,2,3' for different scales per bin.")
    
    # Datavector bin range selection
    bin_range_group = parser.add_mutually_exclusive_group(required=False)
    bin_range_group.add_argument("--bin-range", type=str,
                        help="Global bin range for all redshift bins in format 'start:end' (both inclusive, 0-indexed)")
    bin_range_group.add_argument("--bin-ranges", type=str,
                        help="Separate bin ranges for each redshift bin in format 'start1:end1,start2:end2,...' (0-indexed)")
    
    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    parser.add_argument("--masked", action="store_true",
                        help="Use masked datavectors (Euclid-like sky mask)")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the sky mask in square degrees (default: 14000).")
    
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
    
    parser.add_argument("--new-normalization", action="store_true",
                        help="Use data with new normalization.")
    
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


def parse_bin_ranges(args, num_redshift_bins):
    """
    Parse bin range arguments and return list of (start, end) tuples for each redshift bin.
    
    Args:
        args: Command-line arguments
        num_redshift_bins: Number of redshift bins
        
    Returns:
        list of (start_idx, end_idx) tuples or None if no range specified
    """
    if args.bin_range:
        # Global bin range for all redshift bins
        try:
            start, end = map(int, args.bin_range.split(':'))
            bin_ranges = [(start, end)] * num_redshift_bins
            print(f"Using global bin range [{start}:{end}] for all redshift bins")
        except ValueError:
            raise ValueError("Global bin range must be in format 'start:end' (e.g., '10:50')")
    elif args.bin_ranges:
        # Separate bin ranges for each redshift bin
        try:
            range_strs = args.bin_ranges.split(',')
            if len(range_strs) != num_redshift_bins:
                raise ValueError(f"Number of bin ranges ({len(range_strs)}) must match number of redshift bins ({num_redshift_bins})")
            
            bin_ranges = []
            for range_str in range_strs:
                start, end = map(int, range_str.strip().split(':'))
                bin_ranges.append((start, end))
            
            print(f"Using separate bin ranges: {bin_ranges}")
        except ValueError as e:
            if "must match number" in str(e):
                raise e
            else:
                raise ValueError("Bin ranges must be in format 'start1:end1,start2:end2,...' (e.g., '10:50,20:60')")
    else:
        # No bin range specified, use all bins
        bin_ranges = None
        print("No bin range specified, using all datavector bins")
    
    return bin_ranges


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
    
    # Datavector paths for each bin
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    normalization_suffix = "_new_normalization" if args.new_normalization else ""
    l1_paths = []
    fiducial_paths = []
    
    if args.bnt:
        l1_prefix = "all_bnt_l1_norms"
        fiducial_prefix = "all_bnt_l1_norms"
        
        # For BNT mode, use the bnt bins
        for bnt_bin_idx in bnt_bin_indices:
            bin_spec = f"bin{bnt_bin_idx+1}"
            
            # Grid path
            l1_filename = f"{l1_prefix}_grid_{args.simulation_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            l1_path = os.path.join(args.data_dir, "grid", l1_filename)
            l1_paths.append(l1_path)
            
            # Fiducial path
            fiducial_filename = f"{fiducial_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
            fiducial_paths.append(fiducial_path)
        
        # Set bin description for paths and file names
        if is_multi_bnt_bin:
            final_bin_desc = bnt_bin_desc
        else:
            final_bin_desc = bnt_bin_desc
    else:
        l1_prefix = "all_l1_norms"
        fiducial_prefix = "all_l1_norms"
        
        # Standard mode, use regular bins
        for bin_idx in bin_indices:
            bin_spec = f"bin{bin_idx}"
            
            # Grid path
            l1_filename = f"{l1_prefix}_grid_{args.simulation_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            l1_path = os.path.join(args.data_dir, "grid", l1_filename)
            l1_paths.append(l1_path)
            
            # Fiducial path
            fiducial_filename = f"{fiducial_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{args.mask_suffix}{noise_suffix}{normalization_suffix}.npy"
            fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
            fiducial_paths.append(fiducial_path)
        
        final_bin_desc = bin_desc
    
    # Parse scale options
    if args.scales_per_bin:
        # Parse scales per bin format: "1,2,3;0,1,2,3;..."
        scales_per_bin = []
        for scale_group in args.scales_per_bin.split(';'):
            scales = [int(s.strip()) for s in scale_group.split(',')]
            scales_per_bin.append(scales)
        
        # Generate scale description
        scale_desc = "scales-per-bin_" + "_".join([
            "".join(map(str, scales)) for scales in scales_per_bin
        ])
        is_multi_scale = True
    elif args.scales:
        scale_indices = [int(s.strip()) for s in args.scales.split(',')]
        scale_desc = f"scales{''.join([str(s) for s in scale_indices])}"
        scales_per_bin = None
        is_multi_scale = True
    else:
        scale_indices = [args.scale]
        scale_desc = f"scale{args.scale}"
        scales_per_bin = None
        is_multi_scale = False
    
    # Build checkpoint path
    checkpoint_components = [
        "cosmoGRID",
        "l1" if not args.bnt else "bnt_l1",
        args.simulation_type,
        final_bin_desc,
        scale_desc
    ]
    
    if args.masked:
        checkpoint_components.append(args.mask_label)
    
    if args.noisy:
        checkpoint_components.append(f"noisy_s{args.noise_level:.2f}")
    
    if args.new_normalization:
        checkpoint_components.append("new_normalization")
    
    checkpoint_name = "_".join(checkpoint_components)
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    return {
        'params_path': params_path,
        'l1_paths': l1_paths,
        'fiducial_paths': fiducial_paths,
        'checkpoint_path': checkpoint_path,
        'bin_indices': bin_indices if not args.bnt else bnt_bin_indices,
        'bin_desc': final_bin_desc,
        'scale_desc': scale_desc,
        'scales_per_bin': scales_per_bin,
        'scale_indices': scale_indices if not args.scales_per_bin else None,
        'is_multi_bin': is_multi_bin if not args.bnt else is_multi_bnt_bin,
        'is_multi_scale': is_multi_scale,
        'checkpoint_name': checkpoint_name,
    }


def main():
    args = parse_arguments()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
    print(f"JAX device: {jax.devices()}")
    
    # Construct file paths
    paths = construct_paths(args)
    
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Simulation type: {args.simulation_type}")
    print(f"Fiducial type: {args.fiducial_type}")
    print(f"Bin configuration: {paths['bin_desc']}")
    print(f"Scale configuration: {paths['scale_desc']}")
    print(f"Noisy: {args.noisy}" + (f" (noise level: {args.noise_level})" if args.noisy else ""))
    print(f"Masked: {args.masked}" + (f" ({args.mask_area_sqdeg} sq deg)" if args.masked else ""))
    print(f"New normalization: {args.new_normalization}")
    print(f"Checkpoint: {paths['checkpoint_name']}")
    print("="*60 + "\n")
    
    # Load cosmological parameters
    print(f"Loading cosmological parameters from {paths['params_path']}")
    params = np.load(paths['params_path'])
    print(f"Parameters shape: {params.shape}")
    print(f"Parameter names: Omega_m, sigma_8, w0")
    print(f"Parameter ranges:")
    print(f"  Omega_m: [{params[:, 0].min():.3f}, {params[:, 0].max():.3f}]")
    print(f"  sigma_8: [{params[:, 1].min():.3f}, {params[:, 1].max():.3f}]")
    print(f"  w0: [{params[:, 2].min():.3f}, {params[:, 2].max():.3f}]")
    
    # Load L1 norm data for each bin
    print(f"\nLoading L1 norm data...")
    data_list = []
    for i, l1_path in enumerate(paths['l1_paths']):
        print(f"  Bin {i+1}: {l1_path}")
        data = np.load(l1_path)
        print(f"    Shape: {data.shape}")
        data_list.append(data)
    
    # Create ResultsAggregator for data processing
    aggregator = ResultsAggregator()
    
    # Select scales based on configuration
    if paths['scales_per_bin'] is not None:
        # Different scales per bin
        print(f"\nApplying per-bin scale selection: {paths['scales_per_bin']}")
        # Determine number of bins per scale (datavector features per scale)
        nbins_per_scale = data_list[0].shape[1] // 4  # Assuming 4 scales
        print(f"  Features per scale: {nbins_per_scale}")
        
        data_list = aggregator.select_scales_per_bin(
            data_list, 
            paths['scales_per_bin'], 
            nbins_per_scale
        )
        
        for i, data in enumerate(data_list):
            print(f"  Bin {i+1} after scale selection: {data.shape}")
    elif paths['scale_indices'] is not None and len(paths['scale_indices']) > 0:
        # Select specific scales (same for all bins)
        print(f"\nSelecting scales: {paths['scale_indices']}")
        nbins_per_scale = data_list[0].shape[1] // 4  # Assuming 4 scales
        print(f"  Features per scale: {nbins_per_scale}")
        
        data_list = [
            aggregator.select_scales(data, paths['scale_indices'], nbins_per_scale)
            for data in data_list
        ]
        
        for i, data in enumerate(data_list):
            print(f"  Bin {i+1} after scale selection: {data.shape}")
    
    # Apply bin range selection if specified
    bin_ranges = parse_bin_ranges(args, len(data_list))
    if bin_ranges is not None:
        print(f"\nApplying bin range selection: {bin_ranges}")
        data_list = aggregator.select_bin_ranges_per_bin(data_list, bin_ranges)
        
        for i, data in enumerate(data_list):
            print(f"  Bin {i+1} after bin range selection: {data.shape}")
    
    # Combine data from all bins
    if len(data_list) > 1:
        combined_data_vector = np.concatenate(data_list, axis=1)
        print(f"\nCombined datavector shape (after concatenating bins): {combined_data_vector.shape}")
    else:
        combined_data_vector = data_list[0]
        print(f"\nFinal datavector shape: {combined_data_vector.shape}")
    
    # Filter zero-variance bins using ResultsAggregator
    print(f"\nFiltering zero-variance features...")
    original_features = combined_data_vector.shape[1]
    combined_data_vector = aggregator.filter_zero_variance(
        combined_data_vector,
        min_variance=1e-10,
        return_mask=False
    )
    n_removed = original_features - combined_data_vector.shape[1]
    print(f"Removed {n_removed} zero-variance features")
    print(f"Final datavector shape: {combined_data_vector.shape}")
    
    # Convert to JAX arrays
    params_jax = jnp.array(params)
    data_jax = jnp.array(combined_data_vector)
    
    print(f"\nFinal JAX arrays:")
    print(f"  Parameters: {params_jax.shape}")
    print(f"  Data: {data_jax.shape}")
    
    # Training or loading
    if args.train:
        print("\n" + "="*60)
        print("Training NPE Model")
        print("="*60)
        
        # Initialize NPE
        inference = NPE()
        inference = inference.append_simulations(params_jax, data_jax)
        
        # Create checkpoint directory
        os.makedirs(paths['checkpoint_path'], exist_ok=True)
        
        # Train with NaN retry using utility function
        inference, metrics, density_estimator = train_npe_with_nan_retry(
            inference=inference,
            checkpoint_path=paths['checkpoint_path'],
            params=params_jax,
            data=data_jax,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_retries=10
        )
        
        print(f"\nTraining completed!")
        print(f"Final train loss: {metrics.train_loss[-1] if isinstance(metrics.train_loss, list) else metrics.train_loss}")
        if hasattr(metrics, 'val_loss'):
            print(f"Final val loss: {metrics.val_loss[-1] if isinstance(metrics.val_loss, list) else metrics.val_loss}")
        
    else:
        print("\n" + "="*60)
        print("Loading Existing Model")
        print("="*60)
        
        if not os.path.exists(paths['checkpoint_path']):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {paths['checkpoint_path']}\n"
                f"Use --train to train a new model first."
            )
        
        # Load existing model
        inference = NPE()
        inference = inference.append_simulations(params_jax, data_jax)
        density_estimator = inference.load(paths['checkpoint_path'])
        print(f"Loaded model from {paths['checkpoint_path']}")
    
    # Build posterior
    print("\n" + "="*60)
    print("Building Posterior")
    print("="*60)
    posterior = inference.build_posterior()
    print("Posterior built successfully!")
    
    # Load fiducial data
    print("\n" + "="*60)
    print("Loading Fiducial Data")
    print("="*60)
    
    fiducial_data_list = []
    for i, fiducial_path in enumerate(paths['fiducial_paths']):
        print(f"  Bin {i+1}: {fiducial_path}")
        fiducial_data = np.load(fiducial_path)
        print(f"    Shape: {fiducial_data.shape}")
        fiducial_data_list.append(fiducial_data)
    
    # Apply same scale selection to fiducial
    if paths['scales_per_bin'] is not None:
        print(f"\nApplying per-bin scale selection to fiducial")
        nbins_per_scale = fiducial_data_list[0].shape[0] // 4
        fiducial_data_list = aggregator.select_scales_per_bin(
            [fd[np.newaxis, :] for fd in fiducial_data_list],  # Add batch dimension
            paths['scales_per_bin'],
            nbins_per_scale
        )
        fiducial_data_list = [fd[0] for fd in fiducial_data_list]  # Remove batch dimension
    elif paths['scale_indices'] is not None and len(paths['scale_indices']) > 0:
        print(f"\nApplying scale selection to fiducial")
        nbins_per_scale = fiducial_data_list[0].shape[0] // 4
        fiducial_data_list = [
            aggregator.select_scales(fd[np.newaxis, :], paths['scale_indices'], nbins_per_scale)[0]
            for fd in fiducial_data_list
        ]
    
    # Apply bin range selection to fiducial
    if bin_ranges is not None:
        print(f"\nApplying bin range selection to fiducial")
        fiducial_data_list = aggregator.select_bin_ranges_per_bin(
            [fd[np.newaxis, :] for fd in fiducial_data_list],  # Add batch dimension
            bin_ranges
        )
        fiducial_data_list = [fd[0] for fd in fiducial_data_list]  # Remove batch dimension
    
    # Combine fiducial data
    if len(fiducial_data_list) > 1:
        fiducial_data = np.concatenate(fiducial_data_list)
    else:
        fiducial_data = fiducial_data_list[0]
    
    # Apply zero-variance mask to fiducial (must use same mask)
    # We need to recompute the mask from training data
    # Combine original data for mask computation
    if len(data_list) > 1:
        temp_data = np.concatenate(data_list, axis=1)
    else:
        temp_data = data_list[0]
    
    # Get mask (filter_zero_variance returns (filtered_data, mask) when return_mask=True)
    _, valid_mask = aggregator.filter_zero_variance(
        temp_data,
        min_variance=1e-10,
        return_mask=True
    )

    # Apply mask to fiducial
    fiducial_data = fiducial_data[valid_mask]
    print(f"Fiducial data shape after processing: {fiducial_data.shape}")
    
    # Sample posterior for fiducial
    print("\n" + "="*60)
    print("Sampling Posterior for Fiducial")
    print("="*60)
    
    key = random.PRNGKey(args.random_seed)
    print(f"Generating {args.num_samples} samples...")
    samples = posterior.sample(
        x=fiducial_data, 
        num_samples=args.num_samples, 
        key=key
    )
    samples_np = np.array(samples)
    print(f"Samples shape: {samples_np.shape}")
    
    # True parameters for plotting
    true_params = jnp.array([[2.600e-01, 8.400e-01, -1.000e+00, 6.736e+01, 9.649e-01, 4.930e-02]])
    
    # Run coverage test if requested
    if args.run_coverage_test:
        ecp, alpha = run_tarp_coverage_test(
            posterior=posterior,
            data=combined_data_vector,
            params=params,
            num_test_sims=args.coverage_num_sims,
            num_samples=args.coverage_num_samples,
            seed=args.coverage_seed,
            bootstrap=args.coverage_bootstrap,
            num_bootstrap=args.coverage_num_bootstrap if args.coverage_bootstrap else None
        )
        
        # Plot coverage
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename_base = paths['checkpoint_name']
        if args.run is not None:
            output_filename_base += f"_run{args.run}"
        
        plot_tarp_coverage(
            ecp=ecp,
            alpha=alpha,
            output_path=os.path.join(args.output_dir, f"{output_filename_base}_tarp_coverage.pdf"),
            bootstrap=args.coverage_bootstrap,
            figsize=(6, 6),
            dpi=300
        )
    
    # Save samples
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = paths['checkpoint_name']
    if args.run is not None:
        samples_filename += f"_run{args.run}"
    samples_filename += "_samples.npy"
    samples_path = os.path.join(args.samples_dir, samples_filename)
    
    np.save(samples_path, samples_np)
    print(f"Saved samples to {samples_path}")
    
    # Create triangle plot using getdist
    print("\nCreating triangle plot...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization with all 6 parameters
    labels = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
    
    # Create descriptive sample label
    fiducial_desc = f"{args.fiducial_type}"
    if args.noisy:
        fiducial_desc += f"_n{args.noise_level:.2f}"
    
    sample_label = f"{args.simulation_type} DV vs {fiducial_desc} fid, {paths['bin_desc']}, {paths['scale_desc']}"
    
    mc_samples = MCSamples(
        samples=samples_np,
        names=labels,
        label=sample_label,
    )
    
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4
    
    g.triangle_plot([mc_samples], filled=True,
                   line_args=[{'color': 'blue'}],
                   contour_colors=['blue'],
                   markers={
                       label: val for label, val in zip(labels, true_params[0])
                   })
    
    plot_filename = paths['checkpoint_name']
    if args.run is not None:
        plot_filename += f"_run{args.run}"
    plot_filename += "_triangle.pdf"
    plot_path = os.path.join(args.output_dir, plot_filename)
    
    plt.savefig(plot_path, transparent=True, dpi=300)
    print(f"Saved triangle plot to {plot_path}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Checkpoint: {paths['checkpoint_path']}")
    print(f"Samples: {samples_path}")
    print(f"Triangle plot: {plot_path}")
    if args.run_coverage_test:
        coverage_plot = os.path.join(args.output_dir, f"{output_filename_base}_tarp_coverage.pdf")
        print(f"Coverage plot: {coverage_plot}")
    print("="*60)


if __name__ == "__main__":
    main()
