#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/run_npe_peak_counts_inference_v3.py
"""
NPE Inference on CosmoGRID Peak Counts - v3 (Workflow Utilities)

This version builds on v2 by utilizing the bar_impact.utils.npe_workflow module
to eliminate ~130 lines of duplicate workflow code:

New utilities used:
- setup_jax_environment(): JAX configuration
- initialize_npe(): NPE initialization
- train_or_load_npe(): Unified train/load with NaN retry
- sample_and_save_posterior(): Complete sampling, saving, and plotting workflow
- print_analysis_summary(): Formatted configuration output
- print_completion_summary(): Formatted results output

Maintains identical CLI interface and numerical behavior to v2.
"""

import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp

from bar_impact.analysis.aggregation import ResultsAggregator
from bar_impact.utils.paths import get_data_file_paths
from bar_impact.utils.reproducibility import get_deterministic_seed
from bar_impact.utils.inference import run_tarp_coverage_test, plot_tarp_coverage
from bar_impact.utils.npe_workflow import (
    STANDARD_COSMO_PARAMS,
    initialize_npe,
    train_or_load_npe,
    sample_and_save_posterior,
    setup_jax_environment,
    print_analysis_summary,
    print_completion_summary,
)


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


def construct_paths(args):
    """Construct file paths based on provided arguments."""
    # Params file path
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
    
    # Parse scale options
    if args.scales_per_bin:
        # Parse semicolon-separated per-bin scale specifications
        scales_per_bin = [[int(s.strip()) for s in bin_scales.split(',')] 
                          for bin_scales in args.scales_per_bin.split(';')]
        scale_desc = f"scales{''.join([str(s+1) for s in scales_per_bin[0]])}_perbin"
    elif args.scales:
        scale_indices = [int(s.strip()) for s in args.scales.split(',')]
        scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
        scales_per_bin = None
    else:
        scale_indices = [args.scale]
        scale_desc = f"scale{args.scale+1}"
        scales_per_bin = None
    
    # Build checkpoint path
    checkpoint_components = [
        "cosmoGRID",
        "pc" if not args.bnt else "bnt_pc",
        args.simulation_type,
        bin_spec_for_output,
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
        'peak_counts_paths': peak_counts_paths,
        'fiducial_paths': fiducial_paths,
        'checkpoint_path': checkpoint_path,
        'bin_spec': bin_spec_for_output,
        'scale_desc': scale_desc,
        'scales_per_bin': scales_per_bin if args.scales_per_bin else None,
        'scale_indices': scale_indices if not args.scales_per_bin else None,
        'checkpoint_name': checkpoint_name,
    }


def main():
    args = parse_arguments()
    
    # Construct file paths
    paths = construct_paths(args)
    
    print(f"Using parameters file: {paths['params_path']}")
    print(f"Using peak counts files: {paths['peak_counts_paths']}")
    print(f"Using fiducial files: {paths['fiducial_paths']}")
    
    # Setup JAX environment using workflow utility
    setup_jax_environment(gpu_id=args.gpu, force_cpu=args.force_cpu, enable_x64=True)
    
    # Print analysis summary using workflow utility
    print_analysis_summary({
        'simulation_type': args.simulation_type,
        'fiducial_type': args.fiducial_type,
        'bin_desc': paths['bin_spec'],
        'scale_desc': paths['scale_desc'],
        'noisy': args.noisy,
        'noise_level': args.noise_level if args.noisy else None,
        'masked': args.masked,
        'mask_area_sqdeg': args.mask_area_sqdeg if args.masked else None,
        'new_normalization': args.new_normalization,
        'checkpoint_name': paths['checkpoint_name'],
    })
    
    # Load cosmological parameters
    print(f"Loading cosmological parameters from {paths['params_path']}")
    params = np.load(paths['params_path'], allow_pickle=True)
    print(f"Parameters shape: {params.shape}")
    
    # Load and process peak counts data from each bin
    print(f"\nLoading peak counts data...")
    peak_counts_full_bins = []
    for peak_counts_path in paths['peak_counts_paths']:
        peak_counts_full = np.load(peak_counts_path, allow_pickle=True)
        peak_counts_full_bins.append(peak_counts_full)
        print(f"  Loaded from {peak_counts_path}, shape: {peak_counts_full.shape}")
    
    # Create ResultsAggregator for data processing
    aggregator = ResultsAggregator()
    
    # Extract scale data - either single scale, multiple scales, or per-bin scales
    if paths['scales_per_bin'] is not None:
        # Per-bin scale selection
        scales_per_bin = paths['scales_per_bin']
        print(f"\nUsing per-bin scales configuration: {scales_per_bin}")
        
        # Verify we have the right number of scale specs for the bins
        if len(scales_per_bin) != len(peak_counts_full_bins):
            raise ValueError(f"Number of per-bin scale specs ({len(scales_per_bin)}) must match number of bins ({len(peak_counts_full_bins)})")
        
        # Process each bin's data with its specific scale selection
        # peak_counts_full has shape (n_sims, n_scales, n_features_per_scale)
        # We need to select specific scales and concatenate
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
            print(f"  Bin {bin_idx+1}: using scales {[s+1 for s in scale_indices]}, shape: {bin_data.shape}")
            
    elif paths['scale_indices'] is not None:
        # Same scales for all bins
        scale_indices = paths['scale_indices']
        print(f"\nSelecting scales: {[s+1 for s in scale_indices]}")
        
        bin_data_list = []
        for bin_idx, peak_counts_full in enumerate(peak_counts_full_bins):
            peak_counts_scales = []
            for scale_idx in scale_indices:
                peak_counts_scales.append(peak_counts_full[:, scale_idx])
            
            bin_data = np.concatenate([scale_data.reshape(scale_data.shape[0], -1) 
                                      for scale_data in peak_counts_scales], axis=1)
            bin_data_list.append(bin_data)
            print(f"  Bin {bin_idx+1}: shape: {bin_data.shape}")
    
    # Combine data from all bins
    if len(bin_data_list) > 1:
        combined_data_vector = np.concatenate(bin_data_list, axis=1)
        print(f"\nCombined datavector shape (after concatenating bins): {combined_data_vector.shape}")
    else:
        combined_data_vector = bin_data_list[0]
        print(f"\nFinal datavector shape: {combined_data_vector.shape}")
    
    # Convert to JAX arrays
    params_jax = jnp.array(params)
    data_jax = jnp.array(combined_data_vector)
    
    print(f"\nFinal JAX arrays:")
    print(f"  Parameters: {params_jax.shape}")
    print(f"  Data: {data_jax.shape}")
    
    # Initialize NPE using workflow utility
    inference = initialize_npe(params_jax, data_jax)
    
    # Train or load using workflow utility
    inference, metrics, density_estimator = train_or_load_npe(
        inference=inference,
        checkpoint_path=paths['checkpoint_path'],
        should_train=args.train,
        train_params={
            'params': params_jax,
            'data': data_jax,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'max_retries': 10,
        }
    )
    
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
    
    fiducial_full_bins = []
    for fiducial_path in paths['fiducial_paths']:
        print(f"  Loading: {fiducial_path}")
        fiducial_full = np.load(fiducial_path, allow_pickle=True)
        fiducial_full_bins.append(fiducial_full)
        print(f"    Shape: {fiducial_full.shape}")
    
    # Apply same scale selection to fiducial
    fiducial_data_list = []
    if paths['scales_per_bin'] is not None:
        print(f"\nApplying per-bin scale selection to fiducial")
        for bin_idx, (fiducial_full, scale_indices) in enumerate(zip(fiducial_full_bins, paths['scales_per_bin'])):
            fiducial_scales = []
            for scale_idx in scale_indices:
                fiducial_scales.append(fiducial_full[scale_idx])
            
            fiducial_bin = np.concatenate([scale_data.flatten() for scale_data in fiducial_scales])
            fiducial_data_list.append(fiducial_bin)
    elif paths['scale_indices'] is not None:
        print(f"\nApplying scale selection to fiducial")
        for fiducial_full in fiducial_full_bins:
            fiducial_scales = []
            for scale_idx in paths['scale_indices']:
                fiducial_scales.append(fiducial_full[scale_idx])
            
            fiducial_bin = np.concatenate([scale_data.flatten() for scale_data in fiducial_scales])
            fiducial_data_list.append(fiducial_bin)
    
    # Combine fiducial data
    if len(fiducial_data_list) > 1:
        fiducial_data = np.concatenate(fiducial_data_list)
    else:
        fiducial_data = fiducial_data_list[0]
    
    print(f"Fiducial data shape after processing: {fiducial_data.shape}")
    
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
    
    # Sample and save posterior using workflow utility
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    samples_filename = paths['checkpoint_name']
    if args.run is not None:
        samples_filename += f"_run{args.run}"
    
    # Create descriptive sample label
    fiducial_desc = f"{args.fiducial_type}"
    if args.noisy:
        fiducial_desc += f"_n{args.noise_level:.2f}"
    
    sample_label = f"{args.simulation_type} PC vs {fiducial_desc} fid, {paths['bin_spec']}, {paths['scale_desc']}"
    
    samples_path, plot_path = sample_and_save_posterior(
        posterior=posterior,
        observation=fiducial_data,
        output_config={
            'samples_dir': args.samples_dir,
            'output_dir': args.output_dir,
            'base_filename': samples_filename,
            'num_samples': args.num_samples,
            'random_seed': args.random_seed,
            'sample_label': sample_label,
            'param_config': STANDARD_COSMO_PARAMS,
            'color': 'red',
        }
    )
    
    # Print completion summary using workflow utility
    result_paths = {
        'checkpoint': paths['checkpoint_path'],
        'samples': samples_path,
        'triangle_plot': plot_path,
    }
    
    if args.run_coverage_test:
        result_paths['coverage_plot'] = os.path.join(
            args.output_dir, 
            f"{output_filename_base}_tarp_coverage.pdf"
        )
    
    print_completion_summary(result_paths, args.run_coverage_test)


if __name__ == "__main__":
    main()
