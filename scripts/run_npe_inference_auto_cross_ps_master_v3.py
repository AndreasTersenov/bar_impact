#!/usr/bin/env python3
"""
NPE Inference with Auto + Cross Power Spectra (MASTER-corrected) - v3 (Workflow Utilities)

This version builds on v2 by utilizing the bar_impact.utils.npe_workflow module
to eliminate ~130 lines of duplicate workflow code:

New utilities used:
- setup_jax_environment(): JAX configuration
- initialize_npe(): NPE initialization
- train_or_load_npe(): Unified train/load with NaN retry
- sample_and_save_posterior(): Complete sampling, saving, and plotting workflow
- print_analysis_summary(): Formatted configuration output
- print_completion_summary(): Formatted results output

Maintains all specialized power spectra processing logic (multipole cutting, rebinning, 
cross-pair selection) as these are domain-specific.

Maintains identical CLI interface and numerical behavior to v2.
"""
import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp

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
    parser = argparse.ArgumentParser(description="Run NPE inference on MASTER-corrected CosmoGRID auto + cross power spectra")
    
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
    parser.add_argument("--bnt-cross-abs", action="store_true",
                        help="Take absolute values of BNT cross spectra before rebinning (handles negative values)")

    # Power Spectrum processing options
    parser.add_argument("--lmax", type=int, default=1024,
                        help="Maximum multipole (lmax) used when computing power spectra. Must match the processing script's --lmax. Default is 1024.")
    parser.add_argument("--lower-cut", type=int, default=30,
                        help="Lower multipole cut for the power spectrum (l_min).")
    
    upper_cut_group = parser.add_mutually_exclusive_group(required=False)
    upper_cut_group.add_argument("--upper-cut", type=int, default=1024,
                        help="Upper multipole cut for the power spectrum (l_max).")
    upper_cut_group.add_argument("--upper-cuts", type=str,
                        help="Comma-separated list of upper multipole cuts for each bin (l_max).")
    
    parser.add_argument("--rebin", type=int, default=1,
                        help="Rebinning factor for the power spectrum. Default is 1 (no rebinning).")

    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    
    # MASTER-specific options
    parser.add_argument("--masked", action="store_true",
                        help="Use MASTER-corrected masked power spectra (required for this script)")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the sky mask in square degrees (default: 14000).")
    parser.add_argument("--apodization-scale-deg", type=float, default=2.0,
                        help="Apodization scale in degrees (default: 2.0). Must match processing.")
    
    # Cross power spectra configuration
    parser.add_argument("--cross-data-dir", type=str,
                        help="Directory containing aggregated cross power spectra files. If not specified, uses data-dir.")
    parser.add_argument("--cross-pairs", type=str, default=None,
                        help="Comma-separated list of cross power spectrum pairs to include, e.g., '1,3;1,4;2,4' for (1,3), (1,4), and (2,4). If not specified, all cross pairs are used.")
    parser.add_argument("--auto-only", action="store_true",
                        help="Use only auto power spectra (for comparison)")
    parser.add_argument("--cross-only", action="store_true",
                        help="Use only cross power spectra (for comparison)")
    
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
                        help="Force CPU execution (ignore GPU)")
    
    # Debugging options
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information about data loading and processing")
    
    # Run identifier for multiple runs
    parser.add_argument("--run", type=int, default=None,
                        help="Run number identifier. If specified, appends '_runN' to output filenames for multiple runs.")
    
    args = parser.parse_args()
    
    # Set fiducial type to match simulation type if not specified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type
    
    # Set cross data directory if not specified
    if args.cross_data_dir is None:
        args.cross_data_dir = args.data_dir
    
    # Validate conflicting options
    if args.auto_only and args.cross_only:
        raise ValueError("Cannot specify both --auto-only and --cross-only")
    
    # Validate upper_cut doesn't exceed lmax
    if args.upper_cut > args.lmax:
        raise ValueError(f"--upper-cut ({args.upper_cut}) cannot exceed --lmax ({args.lmax})")

    # Construct mask suffix for MASTER-corrected files
    if args.masked:
        area_tag = int(round(args.mask_area_sqdeg))
        apod_tag = f"apod{args.apodization_scale_deg:.1f}"
        mask_suffix = f"_masked_{area_tag}sqdeg_{apod_tag}_master"
        mask_label = f"Masked ({area_tag} sq deg, {apod_tag}, MASTER)"
    else:
        mask_suffix = "_master"
        mask_label = "Full-sky_master"

    args.mask_area_tag = area_tag if args.masked else None
    args.mask_suffix = mask_suffix
    args.mask_label = mask_label
    
    return args

def parse_cross_pairs(cross_pairs_str):
    """Parse cross pairs string into list of tuples."""
    if cross_pairs_str is None:
        return None
    
    pairs = []
    for pair_str in cross_pairs_str.split(';'):
        i, j = map(int, pair_str.split(','))
        pairs.append((i, j))
    return pairs

def parse_upper_cuts(args):
    """Parse upper cuts and validate against number of bins."""
    # Determine number of bins
    if args.bnt:
        num_bins = len(args.bnt_bins.split(','))
    else:
        num_bins = len(args.bins.split(','))
    
    # Parse upper cuts
    if args.upper_cuts:
        upper_cuts = [int(x) for x in args.upper_cuts.split(',')]
        if len(upper_cuts) != num_bins:
            raise ValueError(f"Number of upper cuts ({len(upper_cuts)}) must match number of bins ({num_bins})")
    else:
        # Use single upper_cut for all bins
        upper_cuts = [args.upper_cut] * num_bins
    
    return upper_cuts

def get_cross_indices_for_pairs(bin_indices, cross_pairs):
    """
    Calculate which indices in the cross power spectra array correspond to the requested pairs.
    
    For bins [1,2,3,4], the cross power spectra are ordered as:
    (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    
    Args:
        bin_indices: List of bin indices used in the analysis
        cross_pairs: List of tuples specifying which cross pairs to include
        
    Returns:
        List of indices into the cross power spectra array
    """
    if cross_pairs is None:
        return None
    
    # Generate all possible cross pairs for the given bins
    all_cross_pairs = []
    for i in range(len(bin_indices)):
        for j in range(i+1, len(bin_indices)):
            all_cross_pairs.append((bin_indices[i], bin_indices[j]))
    
    # Find indices of requested pairs
    selected_indices = []
    for pair in cross_pairs:
        if pair in all_cross_pairs:
            selected_indices.append(all_cross_pairs.index(pair))
        else:
            raise ValueError(f"Cross pair {pair} not found in available pairs {all_cross_pairs}")
    
    return selected_indices

def construct_auto_paths(args):
    """Construct file paths for auto power spectra based on provided arguments."""
    # Params file path - this doesn't change with bins
    params_filename = f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    params_path = os.path.join(args.data_dir, "grid", params_filename)
    
    # Parse bin options
    if args.bnt:
        # BNT mode: bnt_bins are 0-indexed, but files use 1-indexed bin numbers
        bnt_bin_indices = [int(b) for b in args.bnt_bins.split(',')]
        # For file paths, convert to 1-indexed: bnt_bin 0 -> bin 1, etc.
        bin_indices = [b + 1 for b in bnt_bin_indices]
        # bin_desc for cross spectra files uses regular "bins" format with 1-indexed numbers
        bin_desc = f"bins{''.join(map(str, bin_indices))}"
        map_type = args.simulation_type
    else:
        bin_indices = [int(b) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join(map(str, bin_indices))}"
        map_type = args.simulation_type

    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    mask_suffix = getattr(args, "mask_suffix", "")
    lmax_suffix = f"_lmax{args.lmax}" if args.lmax != 1024 else ""
    
    auto_data_paths = []
    auto_fiducial_paths = []
    
    for i, bin_idx in enumerate(bin_indices):
        if args.bnt:
            # BNT mode: use "all_bnt_cls_" prefix with 1-indexed bin number
            data_filename = f"all_bnt_cls_grid_{map_type}_bin{bin_idx}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
            fiducial_filename = f"all_bnt_cls_fiducial_{args.fiducial_type}_bin{bin_idx}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        else:
            data_filename = f"all_cls_grid_{map_type}_bin{bin_idx}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
            fiducial_filename = f"all_cls_fiducial_{args.fiducial_type}_bin{bin_idx}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        
        auto_data_paths.append(os.path.join(args.data_dir, "new_grid", data_filename))
        auto_fiducial_paths.append(os.path.join(args.data_dir, "fiducial", fiducial_filename))
        
    return params_path, auto_data_paths, auto_fiducial_paths, bin_desc

def construct_cross_paths(args, bin_desc):
    """Construct file paths for aggregated cross power spectra."""
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    mask_suffix = getattr(args, "mask_suffix", "")
    lmax_suffix = f"_lmax{args.lmax}" if args.lmax != 1024 else ""
    
    if args.bnt:
        # BNT mode: use "all_bnt_cross_cls_" prefix with regular bin_desc (bins1234)
        data_filename = f"all_bnt_cross_cls_grid_{args.simulation_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        fiducial_filename = f"all_bnt_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
    else:
        data_filename = f"all_cross_cls_grid_{args.simulation_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        fiducial_filename = f"all_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
    
    # Look for cross power spectra files in the cross data directory
    cross_data_path = os.path.join(args.cross_data_dir, "new_grid", data_filename)
    if not os.path.exists(cross_data_path):
        raise FileNotFoundError(f"Cross power spectra data file not found: {cross_data_path}")
    
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

def load_and_process_auto_spectra(auto_data_paths, args, upper_cuts=None):
    """Load and process auto power spectra.
    
    Args:
        auto_data_paths: List of paths to auto power spectra files
        args: Argument namespace
        upper_cuts: List of upper multipole cuts for each bin (optional)
    
    NOTE: For MASTER-corrected data, NO f_sky correction is applied here
          because the power spectra were already properly normalized during processing.
    """
    auto_data_list = []
    
    # Use single upper_cut if upper_cuts not provided
    if upper_cuts is None:
        upper_cuts = [args.upper_cut] * len(auto_data_paths)
    
    for data_path in auto_data_paths:
        cls_full = np.load(data_path, allow_pickle=True)
        auto_data_list.append(cls_full)
        if args.verbose:
            print(f"Loaded auto data from {os.path.basename(data_path)}, shape: {cls_full.shape}")
    
    # Process auto power spectra: apply cuts and rebinning
    # NOTE: For MASTER-corrected masked data, we do NOT apply f_sky correction
    # because the spectra were already properly corrected during processing
    processed_auto_list = []
    for i, cls_full in enumerate(auto_data_list):
        upper_cut = upper_cuts[i]
        
        # Determine ell offset and binning based on data source:
        # - Full-sky (args.masked=False): Uses healpy, data starts at ell=0
        # - Masked (args.masked=True): Uses NaMaster, data starts at ell=2
        # Also account for binning: nlb=1 for lmax<=1024, nlb=2 for lmax<=1500, nlb=4 for lmax>1500
        
        if args.masked:
            # NaMaster data: ells start at 2 (monopole/dipole excluded)
            ell_offset = 2
        else:
            # Full-sky healpy data: ells start at 0
            ell_offset = 0
        
        if args.lmax > 1500:
            # Data is binned with nlb=4
            ell_per_bin = 4.0
            # For binned NaMaster data, first bin center is at ~(2+5)/2=3.5 for nlb=4
            # Approximate: bin_index ≈ (ell - ell_offset) / nlb
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        elif args.lmax > 1024:
            # Data is binned with nlb=2
            ell_per_bin = 2.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        else:
            # Unbinned data (nlb=1)
            # Index = ell - ell_offset
            lower_idx = args.lower_cut - ell_offset
            cut_idx = upper_cut - ell_offset
        
        # Safety check for valid indices
        lower_idx = max(0, lower_idx)
        cut_idx = min(cls_full.shape[1], cut_idx)
        
        # Apply lower and upper cuts
        cls_cut = cls_full[:, lower_idx:cut_idx]
        
        if args.verbose:
            actual_ell_start = lower_idx + ell_offset if args.lmax <= 1024 else "binned"
            actual_ell_end = cut_idx + ell_offset - 1 if args.lmax <= 1024 else "binned"
            print(f"Bin {i}: ell_offset={ell_offset}, indices [{lower_idx}:{cut_idx}] -> ells [{actual_ell_start}, {actual_ell_end}], shape: {cls_cut.shape}")
        
        # Apply rebinning if requested
        if args.rebin > 1:
            cls_rebinned = np.array([rebin_cls(cl, args.rebin) for cl in cls_cut])
            processed_auto_list.append(cls_rebinned)
            if args.verbose:
                print(f"Bin {i}: Rebinned by factor {args.rebin}, final shape: {cls_rebinned.shape}")
        else:
            processed_auto_list.append(cls_cut)
    
    # Concatenate all auto bins together
    auto_data_vector = np.concatenate(processed_auto_list, axis=1)
    
    if args.verbose:
        print(f"Final auto data vector shape: {auto_data_vector.shape}")
    
    return auto_data_vector

def load_and_process_cross_spectra(cross_data_path, args, cross_indices=None, n_bins=None, upper_cuts=None):
    """Load and process aggregated cross power spectra.
    
    Args:
        cross_data_path: Path to the cross power spectra file
        args: Argument namespace
        cross_indices: Optional list of cross-pair indices to select
        n_bins: Number of bins (needed to infer multipole range per cross-pair)
        upper_cuts: List of upper multipole cuts for each bin (optional)
    
    NOTE: For MASTER-corrected data, NO f_sky correction is applied here.
    """
    cross_cls_full = np.load(cross_data_path, allow_pickle=True)
    if args.verbose:
        print(f"Loaded cross data from {os.path.basename(cross_data_path)}, shape: {cross_cls_full.shape}")
    
    # Use single upper_cut if upper_cuts not provided
    if upper_cuts is None:
        upper_cuts = [args.upper_cut] * (n_bins if n_bins else 1)
    
    # Calculate expected number of cross pairs and infer original multipole range
    if n_bins is not None:
        expected_cross_pairs = n_bins * (n_bins - 1) // 2
        # Each cross-pair should have the same ell range as auto spectra
        # Total features = expected_cross_pairs * n_ells_per_pair
        n_ells_per_pair = cross_cls_full.shape[1] // expected_cross_pairs
        if args.verbose:
            print(f"Expected {expected_cross_pairs} cross pairs, {n_ells_per_pair} ells per pair")
    else:
        # Assume single pair
        expected_cross_pairs = 1
        n_ells_per_pair = cross_cls_full.shape[1]
    
    # Create mapping of cross-pair index to bin pair
    # For bins [1,2,3,4], pairs are ordered as: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    cross_pair_to_bins = []
    for i in range(n_bins):
        for j in range(i+1, n_bins):
            cross_pair_to_bins.append((i, j))
    
    # Determine ell offset based on data source (same logic as auto spectra)
    if args.masked:
        # NaMaster data: ells start at 2 (monopole/dipole excluded)
        ell_offset = 2
    else:
        # Full-sky healpy data: ells start at 0
        ell_offset = 0
    
    # Apply cuts to each cross-pair individually
    cross_cls_cut_list = []
    
    for pair_idx in range(expected_cross_pairs):
        # Extract this cross-pair's data
        start_idx = pair_idx * n_ells_per_pair
        end_idx = (pair_idx + 1) * n_ells_per_pair
        cross_cls_this_pair = cross_cls_full[:, start_idx:end_idx]
        
        # Determine cuts for this pair (use max of the two bins' cuts)
        if pair_idx < len(cross_pair_to_bins):
            bin_i, bin_j = cross_pair_to_bins[pair_idx]
            upper_cut = max(upper_cuts[bin_i], upper_cuts[bin_j])
        else:
            upper_cut = args.upper_cut
        
        # Apply cuts (accounting for ell offset and binning like in auto spectra)
        if args.lmax > 1500:
            ell_per_bin = 4.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        elif args.lmax > 1024:
            ell_per_bin = 2.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        else:
            # Unbinned data (nlb=1)
            lower_idx = args.lower_cut - ell_offset
            cut_idx = upper_cut - ell_offset
        
        # Safety check for valid indices
        lower_idx = max(0, lower_idx)
        cut_idx = min(cross_cls_this_pair.shape[1], cut_idx)
        
        cross_cls_cut = cross_cls_this_pair[:, lower_idx:cut_idx]
        cross_cls_cut_list.append(cross_cls_cut)
        
        if args.verbose:
            print(f"Cross pair {pair_idx}: Applied cut at {upper_cut}, shape: {cross_cls_cut.shape}")
    
    if args.verbose:
        print(f"Total cross pairs processed: {len(cross_cls_cut_list)}")
    
    # Apply rebinning to each cross-pair separately (before selection/concatenation)
    if args.rebin > 1:
        cross_cls_cut_list = [
            np.array([rebin_cls(cl, args.rebin) for cl in cross_cls_cut])
            for cross_cls_cut in cross_cls_cut_list
        ]
        if args.verbose:
            print(f"Applied rebinning factor {args.rebin} to all cross pairs")
    
    # If specific cross indices are requested, select only those
    if cross_indices is not None:
        cross_cls_cut_list = [cross_cls_cut_list[i] for i in cross_indices]
        if args.verbose:
            print(f"Selected {len(cross_indices)} cross pairs: indices {cross_indices}")
    
    # Concatenate all (selected) cross pairs
    cross_data_vector = np.concatenate(cross_cls_cut_list, axis=1)
    
    if args.verbose:
        print(f"Final cross data vector shape: {cross_data_vector.shape}")
    
    return cross_data_vector

def load_and_process_auto_fiducial(auto_fiducial_paths, args, upper_cuts=None):
    """Load and process auto fiducial data.
    
    Args:
        auto_fiducial_paths: List of paths to auto fiducial files
        args: Argument namespace
        upper_cuts: List of upper multipole cuts for each bin (optional)
    
    NOTE: For MASTER-corrected data, NO f_sky correction is applied.
    """
    auto_fid_means = []
    
    # Use single upper_cut if upper_cuts not provided
    if upper_cuts is None:
        upper_cuts = [args.upper_cut] * len(auto_fiducial_paths)
    
    for fiducial_path in auto_fiducial_paths:
        fid_full = np.load(fiducial_path, allow_pickle=True)
        fid_mean = np.mean(fid_full, axis=0)
        auto_fid_means.append(fid_mean)
        if args.verbose:
            print(f"Loaded fiducial from {os.path.basename(fiducial_path)}, shape: {fid_mean.shape}")
    
    # Determine ell offset based on data source (same logic as training data)
    if args.masked:
        ell_offset = 2  # NaMaster data starts at ell=2
    else:
        ell_offset = 0  # Full-sky healpy data starts at ell=0
    
    # Process auto fiducial data according to cuts and rebinning
    # NO f_sky correction for MASTER data
    auto_fid_data_list = []
    for i, fid_mean in enumerate(auto_fid_means):
        upper_cut = upper_cuts[i]
        
        # Apply cuts (accounting for ell offset and binning)
        if args.lmax > 1500:
            ell_per_bin = 4.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        elif args.lmax > 1024:
            ell_per_bin = 2.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        else:
            # Unbinned data (nlb=1)
            lower_idx = args.lower_cut - ell_offset
            cut_idx = upper_cut - ell_offset
        
        # Safety check for valid indices
        lower_idx = max(0, lower_idx)
        cut_idx = min(len(fid_mean), cut_idx)
        
        fid_cut = fid_mean[lower_idx:cut_idx]
        
        # Apply rebinning
        if args.rebin > 1:
            fid_rebinned = rebin_cls(fid_cut, args.rebin)
            auto_fid_data_list.append(fid_rebinned)
        else:
            auto_fid_data_list.append(fid_cut)
    
    # Concatenate all auto bins' fiducial data
    auto_fid_mean_processed = np.concatenate(auto_fid_data_list)
    
    if args.verbose:
        print(f"Final auto fiducial vector shape: {auto_fid_mean_processed.shape}")
    
    return auto_fid_mean_processed

def load_and_process_cross_fiducial(cross_fiducial_path, args, cross_indices=None, n_bins=None, upper_cuts=None):
    """Load and process cross fiducial data.
    
    Args:
        cross_fiducial_path: Path to the cross fiducial file
        args: Argument namespace
        cross_indices: Optional list of cross-pair indices to select
        n_bins: Number of bins (needed to infer multipole range per cross-pair)
        upper_cuts: List of upper multipole cuts for each bin (optional)
    
    NOTE: For MASTER-corrected data, NO f_sky correction is applied.
    """
    cross_fid_full = np.load(cross_fiducial_path, allow_pickle=True)
    cross_fid_mean = np.mean(cross_fid_full, axis=0)
    if args.verbose:
        print(f"Loaded cross fiducial from {os.path.basename(cross_fiducial_path)}, shape: {cross_fid_mean.shape}")
    
    # Use single upper_cut if upper_cuts not provided
    if upper_cuts is None:
        upper_cuts = [args.upper_cut] * (n_bins if n_bins else 1)
    
    # Calculate expected number of cross pairs
    if n_bins is not None:
        expected_cross_pairs = n_bins * (n_bins - 1) // 2
        n_ells_per_pair = len(cross_fid_mean) // expected_cross_pairs
    else:
        expected_cross_pairs = 1
        n_ells_per_pair = len(cross_fid_mean)
    
    # Create mapping of cross-pair index to bin pair
    cross_pair_to_bins = []
    for i in range(n_bins):
        for j in range(i+1, n_bins):
            cross_pair_to_bins.append((i, j))
    
    # Determine ell offset based on data source (same logic as training data)
    if args.masked:
        ell_offset = 2  # NaMaster data starts at ell=2
    else:
        ell_offset = 0  # Full-sky healpy data starts at ell=0
    
    # Apply cuts to each cross-pair individually
    cross_fid_cut_list = []
    
    for pair_idx in range(expected_cross_pairs):
        # Extract this cross-pair's data
        start_idx = pair_idx * n_ells_per_pair
        end_idx = (pair_idx + 1) * n_ells_per_pair
        cross_fid_this_pair = cross_fid_mean[start_idx:end_idx]
        
        # Determine cuts for this pair
        if pair_idx < len(cross_pair_to_bins):
            bin_i, bin_j = cross_pair_to_bins[pair_idx]
            upper_cut = max(upper_cuts[bin_i], upper_cuts[bin_j])
        else:
            upper_cut = args.upper_cut
        
        # Apply cuts (accounting for ell offset and binning)
        if args.lmax > 1500:
            ell_per_bin = 4.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        elif args.lmax > 1024:
            ell_per_bin = 2.0
            lower_idx = int((args.lower_cut - ell_offset) / ell_per_bin)
            cut_idx = int((upper_cut - ell_offset) / ell_per_bin)
        else:
            # Unbinned data (nlb=1)
            lower_idx = args.lower_cut - ell_offset
            cut_idx = upper_cut - ell_offset
        
        # Safety check for valid indices
        lower_idx = max(0, lower_idx)
        cut_idx = min(len(cross_fid_this_pair), cut_idx)
        
        cross_fid_cut = cross_fid_this_pair[lower_idx:cut_idx]
        cross_fid_cut_list.append(cross_fid_cut)
    
    if args.verbose:
        print(f"Processed {len(cross_fid_cut_list)} cross pairs for fiducial")
    
    # Apply rebinning to each cross-pair separately (before selection/concatenation)
    if args.rebin > 1:
        cross_fid_cut_list = [
            rebin_cls(cross_fid_cut, args.rebin)
            for cross_fid_cut in cross_fid_cut_list
        ]
        if args.verbose:
            print(f"Applied rebinning factor {args.rebin} to fiducial cross pairs")
    
    # If specific cross indices are requested, select only those
    if cross_indices is not None:
        cross_fid_cut_list = [cross_fid_cut_list[i] for i in cross_indices]
        if args.verbose:
            print(f"Selected {len(cross_indices)} fiducial cross pairs")
    
    # Concatenate all (selected) cross pairs
    cross_fid_processed = np.concatenate(cross_fid_cut_list)
    
    if args.verbose:
        print(f"Final cross fiducial vector shape: {cross_fid_processed.shape}")
    
    return cross_fid_processed


def main():
    args = parse_arguments()

    if not args.masked:
        print("\n" + "="*60)
        print("WARNING: This script is designed for MASTER-corrected data")
        print("Please use --masked flag for MASTER-corrected files")
        print("For full-sky data, use run_npe_inference_auto_cross_ps.py")
        print("="*60 + "\n")
    
    # Parse upper cuts
    upper_cuts = parse_upper_cuts(args)
    print(f"Using upper cuts: {upper_cuts}")
    
    # Parse cross pairs if specified
    cross_pairs = parse_cross_pairs(args.cross_pairs)
    if cross_pairs and args.verbose:
        print(f"Selected cross pairs: {cross_pairs}")
    
    # Construct file paths for auto power spectra
    params_path, auto_data_paths, auto_fiducial_paths, bin_desc = construct_auto_paths(args)
    
    # Determine which bin indices are being used for cross pair calculation
    if args.bnt:
        bin_indices = [int(b) for b in args.bnt_bins.split(',')]
    else:
        bin_indices = [int(b) for b in args.bins.split(',')]
    
    # Number of bins (needed to correctly parse aggregated cross spectra files)
    n_bins = len(bin_indices)
    
    # Calculate cross indices if specific pairs are requested
    cross_indices = get_cross_indices_for_pairs(bin_indices, cross_pairs)
    if cross_indices and args.verbose:
        print(f"Cross pair indices: {cross_indices}")
    
    # Construct file paths for cross power spectra
    cross_data_path, cross_fiducial_path = construct_cross_paths(args, bin_desc)
    
    print(f"\nUsing parameters file: {params_path}")
    print(f"Using {n_bins} bins: {bin_indices}")
    print(f"Using lmax: {args.lmax}")
    print(f"MASTER correction: {args.masked}")
    if args.masked:
        print(f"Mask configuration: {args.mask_area_sqdeg} sq deg, apod={args.apodization_scale_deg}°")
    print(f"\nUsing auto datavector files: {auto_data_paths}")
    print(f"Using cross datavector file: {cross_data_path}")
    print(f"Using auto fiducial files: {auto_fiducial_paths}")
    print(f"Using cross fiducial file: {cross_fiducial_path}")
    if cross_pairs:
        print(f"Using cross pairs: {cross_pairs}")
    
    # Validate that required files exist
    missing_files = []
    if not os.path.exists(params_path):
        missing_files.append(params_path)
    
    if not args.cross_only:
        for path in auto_data_paths + auto_fiducial_paths:
            if not os.path.exists(path):
                missing_files.append(path)
    
    if not args.auto_only:
        if not os.path.exists(cross_data_path):
            missing_files.append(cross_data_path)
        if not os.path.exists(cross_fiducial_path):
            missing_files.append(cross_fiducial_path)
    
    if missing_files:
        print("\n" + "="*60)
        print("ERROR: Missing required files:")
        print("="*60)
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run cross_power_spectrum_processing_master.py first with:")
        print("  --apply-mask")
        print(f"  --mask-area-sqdeg {args.mask_area_sqdeg}")
        print(f"  --apodization-scale-deg {args.apodization_scale_deg}")
        print(f"  --lmax {args.lmax}")
        if args.noisy:
            print(f"  --noise-level {args.noise_level}")
        print("  --aggregate-for-inference")
        print("="*60 + "\n")
        return

    # Setup JAX environment using workflow utility
    setup_jax_environment(gpu_id=args.gpu, force_cpu=args.force_cpu)

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)

    # Load and process data
    data_vector_parts = []
    fid_vector_parts = []
    
    if not args.cross_only:
        print("\nLoading auto power spectra...")
        auto_data_vector = load_and_process_auto_spectra(auto_data_paths, args, upper_cuts)
        data_vector_parts.append(auto_data_vector)
        
        print("Loading auto fiducial...")
        auto_fid_vector = load_and_process_auto_fiducial(auto_fiducial_paths, args, upper_cuts)
        fid_vector_parts.append(auto_fid_vector)
    
    if not args.auto_only:
        print("\nLoading cross power spectra...")
        cross_data_vector = load_and_process_cross_spectra(cross_data_path, args, cross_indices, n_bins, upper_cuts)
        data_vector_parts.append(cross_data_vector)
        
        print("Loading cross fiducial...")
        cross_fid_vector = load_and_process_cross_fiducial(cross_fiducial_path, args, cross_indices, n_bins, upper_cuts)
        fid_vector_parts.append(cross_fid_vector)
    
    # Combine auto and cross power spectra
    combined_data_vector = np.concatenate(data_vector_parts, axis=1)
    combined_fid_vector = np.concatenate(fid_vector_parts)
    
    print(f"\nCombined datavector shape: {combined_data_vector.shape}")
    print(f"Combined fiducial shape: {combined_fid_vector.shape}")

    # Process power spectra description
    if len(set(upper_cuts)) == 1:
        ps_desc = f"l{args.lower_cut}-{args.upper_cut}"
        # For checkpoint path (filesystem-safe): same format
        ps_desc_checkpoint = ps_desc
        if args.rebin > 1:
            ps_desc += f"_r{args.rebin}"
            ps_desc_checkpoint += f"_r{args.rebin}"
    else:
        # For display/filename: use list format [340, 1024, 1024, 1024]
        ps_desc = f"l{args.lower_cut}-{upper_cuts}"
        # For checkpoint path: use underscore-separated format to avoid brackets
        cuts_str = "_".join(map(str, upper_cuts))
        ps_desc_checkpoint = f"l{args.lower_cut}-{cuts_str}"
        if args.rebin > 1:
            ps_desc += f"_r{args.rebin}"
            ps_desc_checkpoint += f"_r{args.rebin}"
    print(f"Processing power spectra with: {ps_desc}")
    
    # Save the full datavector set for verification
    os.makedirs(args.samples_dir, exist_ok=True)
    datavector_filename = f"datavectors_npe_input_{args.simulation_type}_{bin_desc}_{ps_desc}"
    if args.masked:
        datavector_filename += args.mask_suffix
    if args.noisy:
        datavector_filename += f"_noisy_s{args.noise_level:.2f}"
    datavector_filename += ".npy"
    datavector_path = os.path.join(args.samples_dir, datavector_filename)
    np.save(datavector_path, combined_data_vector)
    print(f"Saved full NPE input datavectors to: {datavector_path}")
    print(f"  Shape: {combined_data_vector.shape} (n_simulations × n_features)")
    
    # Create descriptive filename
    if args.auto_only:
        spectra_desc = "auto"
    elif args.cross_only:
        spectra_desc = "cross"
    else:
        spectra_desc = "auto_cross"
    
    if cross_pairs and not args.auto_only:
        pair_str = "_".join([f"{i}-{j}" for i, j in cross_pairs])
        spectra_type = f"{spectra_desc}_{pair_str}"
    else:
        spectra_type = spectra_desc
    
    bnt_prefix = "bnt_" if args.bnt else ""
    bnt_abs_suffix = "_abs" if (args.bnt and hasattr(args, 'bnt_cross_abs') and args.bnt_cross_abs) else ""
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    mask_suffix = args.mask_suffix if args.masked else ""
    run_suffix = f"_run{args.run}" if args.run is not None else ""
    
    # Save first example from training data
    example_train_filename = f"example_train_datavector_{bnt_prefix}{spectra_desc}_{args.simulation_type}_{bin_desc}_{ps_desc}{bnt_abs_suffix}{mask_suffix}{noise_suffix}.npy"
    np.save(os.path.join(args.samples_dir, example_train_filename), combined_data_vector[0])
    print(f"Saved example training datavector to {example_train_filename}")
    print(f"  Shape: {combined_data_vector[0].shape}, first 10 values: {combined_data_vector[0][:10]}")
    
    # Save fiducial observation
    example_fid_filename = f"example_fiducial_datavector_{bnt_prefix}{spectra_desc}_{args.fiducial_type}_{bin_desc}_{ps_desc}{bnt_abs_suffix}{mask_suffix}{noise_suffix}.npy"
    np.save(os.path.join(args.samples_dir, example_fid_filename), combined_fid_vector)
    print(f"Saved fiducial datavector to {example_fid_filename}")
    print(f"  Shape: {combined_fid_vector.shape}, first 10 values: {combined_fid_vector[:10]}")
    
    # Convert to JAX arrays
    params = jnp.array(params)
    combined_data_vector = jnp.array(combined_data_vector)

    # Create checkpoint path (use realpath to resolve symlinks for jaxili compatibility)
    checkpoint_dir = os.path.realpath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration
    if args.auto_only:
        spectra_type_base = "auto"
    elif args.cross_only:
        spectra_type_base = "cross"
    else:
        spectra_type_base = "auto_cross"
    
    # Add cross pairs information to spectra type if specific pairs are selected
    if cross_pairs and not args.auto_only:
        pair_str = "_".join([f"{i}-{j}" for i, j in cross_pairs])
        spectra_type_full = f"{spectra_type_base}_{pair_str}"
    else:
        spectra_type_full = spectra_type_base
    
    # Add BNT prefix to checkpoint name if using BNT data
    if args.bnt:
        checkpoint_name = f"cosmoGRID_bnt_ps_weights_{args.simulation_type}_{bin_desc}_{ps_desc_checkpoint}"
    else:
        checkpoint_name = f"cosmoGRID_ps_weights_{args.simulation_type}_{bin_desc}_{ps_desc_checkpoint}"

    if args.masked:
        checkpoint_name += args.mask_suffix

    if args.noisy:
        checkpoint_name += f"_noisy_s{args.noise_level:.2f}"

    if args.bnt:
        checkpoint_name += f"_{spectra_type_full}"
    else:
        checkpoint_name += f"_{spectra_type_full}"

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Print analysis summary using workflow utility
    print_analysis_summary({
        'simulation_type': args.simulation_type,
        'fiducial_type': args.fiducial_type,
        'bin_desc': bin_desc,
        'scale_desc': ps_desc,
        'noisy': args.noisy,
        'noise_level': args.noise_level if args.noisy else None,
        'masked': args.masked,
        'mask_area_sqdeg': args.mask_area_sqdeg if args.masked else None,
        'new_normalization': False,
        'checkpoint_name': checkpoint_name,
    })

    # Initialize NPE using workflow utility
    inference = initialize_npe(params, combined_data_vector)
    
    # Train or load using workflow utility
    inference, metrics, density_estimator = train_or_load_npe(
        inference=inference,
        checkpoint_path=checkpoint_path,
        should_train=args.train,
        train_params={
            'params': params,
            'data': combined_data_vector,
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

    # Coverage testing if requested
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
        
        # Create base filename for coverage plots
        coverage_base = f"posterior_{bnt_prefix}ps_{spectra_type_full}_{args.simulation_type}_vs_{args.fiducial_type}_{bin_desc}_{ps_desc}{bnt_abs_suffix}"
        if args.masked:
            coverage_base += args.mask_suffix
        if args.noisy:
            coverage_base += f"_noisy_s{args.noise_level:.2f}"
        
        plot_tarp_coverage(
            ecp=ecp,
            alpha=alpha,
            output_path=os.path.join(args.output_dir, f"{coverage_base}_tarp_coverage.pdf"),
            bootstrap=args.coverage_bootstrap,
            figsize=(6, 6),
            dpi=300
        )

    # Sample and save posterior using workflow utility
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create descriptive sample label
    fiducial_desc = f"{args.fiducial_type}"
    if args.masked:
        fiducial_desc += f" ({args.mask_label})"
    if args.noisy:
        fiducial_desc += f" noisy s={args.noise_level}"
    
    if args.auto_only:
        analysis_type = f"auto {bin_desc}"
    elif args.cross_only:
        if cross_pairs:
            pair_str = "+".join([f"({i},{j})" for i, j in cross_pairs])
            analysis_type = f"cross {pair_str}"
        else:
            analysis_type = f"cross {bin_desc}"
    else:
        if cross_pairs:
            pair_str = "+".join([f"({i},{j})" for i, j in cross_pairs])
            analysis_type = f"auto+cross {bin_desc} (cross: {pair_str})"
        else:
            analysis_type = f"auto+cross {bin_desc}"
    
    sample_label = f"{args.simulation_type} {analysis_type} vs {fiducial_desc} fid, {ps_desc}"
    if args.masked:
        sample_label += " MASTER"
    
    # Construct base filename for samples and plots
    samples_base = f"posterior_samples_{bnt_prefix}ps_{spectra_type}_{args.simulation_type}_vs_{args.fiducial_type}_{bin_desc}_{ps_desc_checkpoint}{bnt_abs_suffix}"
    if args.masked:
        samples_base += args.mask_suffix
    if args.noisy:
        samples_base += f"_noisy_s{args.noise_level:.2f}"
    if args.run is not None:
        samples_base += f"_run{args.run}"
    
    samples_path, plot_path = sample_and_save_posterior(
        posterior=posterior,
        observation=combined_fid_vector,
        output_config={
            'samples_dir': args.samples_dir,
            'output_dir': args.output_dir,
            'base_filename': samples_base,
            'num_samples': args.num_samples,
            'random_seed': args.random_seed,
            'sample_label': sample_label,
            'param_config': STANDARD_COSMO_PARAMS,
            'color': 'blue',
        }
    )
    
    # Print completion summary using workflow utility
    result_paths = {
        'checkpoint': checkpoint_path,
        'samples': samples_path,
        'triangle_plot': plot_path,
        'datavector': datavector_path,
    }
    
    if args.run_coverage_test:
        result_paths['coverage_plot'] = os.path.join(
            args.output_dir, 
            f"{coverage_base}_tarp_coverage.pdf"
        )
    
    print_completion_summary(result_paths, args.run_coverage_test)


if __name__ == "__main__":
    main()
