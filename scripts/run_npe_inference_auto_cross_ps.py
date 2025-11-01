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
import sys
# Add tarp package to path if needed
tarp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tarp', 'src')
if tarp_path not in sys.path:
    sys.path.insert(0, tarp_path)
from tarp import get_tarp_coverage

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
    parser.add_argument("--lmax", type=int, default=1024,
                        help="Maximum multipole (lmax) used when computing power spectra. Must match the processing script's --lmax. Default is 1024.")
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
    parser.add_argument("--masked", action="store_true",
                        help="Use masked power spectra (Euclid-like sky mask)")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the sky mask in square degrees (default: 14000).")
    
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
    
    # Validate upper_cut doesn't exceed lmax
    if args.upper_cut > args.lmax:
        parser.error(f"--upper-cut ({args.upper_cut}) cannot exceed --lmax ({args.lmax})")

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

def parse_cross_pairs(cross_pairs_str):
    """Parse cross pairs string into list of tuples."""
    if cross_pairs_str is None:
        return None
    
    pairs = []
    for pair_str in cross_pairs_str.split(';'):
        i, j = pair_str.split(',')
        pairs.append((int(i.strip()), int(j.strip())))
    return pairs

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
        for j in range(i + 1, len(bin_indices)):
            all_cross_pairs.append((bin_indices[i], bin_indices[j]))
    
    # Find indices of requested pairs
    selected_indices = []
    for pair in cross_pairs:
        try:
            idx = all_cross_pairs.index(pair)
            selected_indices.append(idx)
        except ValueError:
            print(f"Warning: Cross pair {pair} not found in available pairs {all_cross_pairs}")
    
    return selected_indices

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
    mask_suffix = getattr(args, "mask_suffix", "")
    lmax_suffix = f"_lmax{args.lmax}" if args.lmax != 1024 else ""
    
    auto_data_paths = []
    auto_fiducial_paths = []
    
    for i, bin_idx in enumerate(bin_indices):
        bin_spec = f"{bin_prefix}{bin_suffix_list[i]}"
        # Auto data path (grid)
        data_filename = f"{data_prefix}_grid_{args.simulation_type}_{bin_spec}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        data_path = os.path.join(args.data_dir, "new_grid", data_filename)
        if not os.path.exists(data_path):
             data_path = os.path.join(args.data_dir, "grid", data_filename)
        auto_data_paths.append(data_path)
        
        # Auto fiducial path
        fiducial_filename = f"{data_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        fiducial_path = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial", fiducial_filename)
        auto_fiducial_paths.append(fiducial_path)
        
    return params_path, auto_data_paths, auto_fiducial_paths, bin_desc

def construct_cross_paths(args, bin_desc):
    """Construct file paths for aggregated cross power spectra."""
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    mask_suffix = getattr(args, "mask_suffix", "")
    lmax_suffix = f"_lmax{args.lmax}" if args.lmax != 1024 else ""
    
    if args.bnt:
        # For BNT, use the correct BNT cross power spectrum naming
        data_filename = f"all_bnt_cross_cls_grid_{args.simulation_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        fiducial_filename = f"all_bnt_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
    else:
        # For regular bins
        data_filename = f"all_cross_cls_grid_{args.simulation_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
        fiducial_filename = f"all_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{mask_suffix}{noise_suffix}{lmax_suffix}.npy"
    
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

def get_fsky_from_npz(file_path, verbose=False):
    """
    Extract f_sky from a processed .npz file (if masked).
    Returns None if no mask metadata is present.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        if 'mask_f_sky' in data.files:
            f_sky = float(data['mask_f_sky'])
            if verbose:
                print(f"  Found f_sky = {f_sky:.4f} in {os.path.basename(file_path)}")
            return f_sky
        else:
            if verbose:
                print(f"  No mask metadata in {os.path.basename(file_path)}")
            return None
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not read mask metadata from {os.path.basename(file_path)}: {e}")
        return None

def load_and_process_auto_spectra(auto_data_paths, args):
    """Load and process auto power spectra."""
    auto_data_list = []
    f_sky = None
    
    for data_path in auto_data_paths:
        cls_full = np.load(data_path, allow_pickle=True)
        auto_data_list.append(cls_full)
        if args.verbose:
            print(f"Loaded auto data from {os.path.basename(data_path)}, shape: {cls_full.shape}")
        
        # Extract f_sky from first file if masked
        if args.masked and f_sky is None:
            # Try to get f_sky from a corresponding .npz file
            # The .npy aggregated files don't have metadata, so we need to look elsewhere
            # For now, compute f_sky from mask parameters
            pass
    
    # If masked, compute f_sky from mask parameters
    if args.masked and f_sky is None:
        # Import locally to avoid circular dependency
        import healpy as hp
        # Compute f_sky based on mask area
        total_area_sqdeg = 41252.96125  # 4*pi*(180/pi)^2
        f_sky = args.mask_area_sqdeg / total_area_sqdeg
        if args.verbose:
            print(f"Computed f_sky = {f_sky:.4f} from mask area {args.mask_area_sqdeg:.0f} sq deg")
    
    # Process auto power spectra: apply cuts, f_sky correction, and rebinning
    processed_auto_list = []
    for cls_full in auto_data_list:
        # Apply cuts
        cls_cut = cls_full[:, args.lower_cut:args.upper_cut]
        
        # Apply f_sky correction if masked
        if args.masked and f_sky is not None:
            cls_cut = cls_cut / f_sky
            if args.verbose:
                print(f"Applied f_sky correction: divided by {f_sky:.4f}")
        
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

def load_and_process_cross_spectra(cross_data_path, args, cross_indices=None, n_bins=None):
    """Load and process aggregated cross power spectra.
    
    Args:
        cross_data_path: Path to the cross power spectra file
        args: Argument namespace
        cross_indices: Optional list of cross-pair indices to select
        n_bins: Number of bins (needed to infer multipole range per cross-pair)
    """
    cross_cls_full = np.load(cross_data_path, allow_pickle=True)
    if args.verbose:
        print(f"Loaded cross data from {os.path.basename(cross_data_path)}, shape: {cross_cls_full.shape}")
    
    # Compute f_sky if masked
    f_sky = None
    if args.masked:
        total_area_sqdeg = 41252.96125  # 4*pi*(180/pi)^2
        f_sky = args.mask_area_sqdeg / total_area_sqdeg
        if args.verbose:
            print(f"Computed f_sky = {f_sky:.4f} from mask area {args.mask_area_sqdeg:.0f} sq deg")
    
    # Calculate expected number of cross pairs and infer original multipole range
    if n_bins is not None:
        expected_cross_pairs = n_bins * (n_bins - 1) // 2
        n_ell_original = cross_cls_full.shape[1] // expected_cross_pairs
        
        if args.verbose:
            print(f"Expected {expected_cross_pairs} cross pairs with {n_ell_original} multipoles each")
            print(f"Total columns: {cross_cls_full.shape[1]} = {expected_cross_pairs} × {n_ell_original}")
    else:
        # Fallback: assume standard multipole range if n_bins not provided
        # This is a guess and may not be correct!
        print("Warning: n_bins not provided, assuming standard multipole range")
        n_ell_original = 1024  # Common default
        expected_cross_pairs = cross_cls_full.shape[1] // n_ell_original
    
    # Apply cuts to each cross-pair individually
    n_multipoles_cut = args.upper_cut - args.lower_cut
    cross_cls_cut_list = []
    
    for i in range(expected_cross_pairs):
        # Extract this cross-pair's full multipole range
        start_col = i * n_ell_original
        end_col = (i + 1) * n_ell_original
        cross_pair_full = cross_cls_full[:, start_col:end_col]
        
        # Apply multipole cuts to this cross-pair
        cross_pair_cut = cross_pair_full[:, args.lower_cut:args.upper_cut]
        
        # Apply f_sky correction if masked
        if args.masked and f_sky is not None:
            cross_pair_cut = cross_pair_cut / f_sky
        
        cross_cls_cut_list.append(cross_pair_cut)
    
    if args.verbose:
        print(f"Applied cuts to {len(cross_cls_cut_list)} cross pairs")
        print(f"Each cross pair now has {n_multipoles_cut} multipoles (l={args.lower_cut} to l={args.upper_cut})")
        if args.masked and f_sky is not None:
            print(f"Applied f_sky correction: divided by {f_sky:.4f}")
    
    # Apply rebinning to each cross-pair separately (before selection/concatenation)
    if args.rebin > 1:
        cross_cls_rebinned_list = []
        for cross_pair_cut in cross_cls_cut_list:
            # Rebin each simulation's cross-pair spectrum
            rebinned_sims = [rebin_cls(cl, args.rebin) for cl in cross_pair_cut]
            cross_cls_rebinned_list.append(np.array(rebinned_sims))
        cross_cls_cut_list = cross_cls_rebinned_list
        
        if args.verbose:
            print(f"Applied rebinning with factor {args.rebin} to each cross pair")
            print(f"Each cross pair now has {cross_cls_cut_list[0].shape[1]} multipoles after rebinning")
    
    # If specific cross indices are requested, select only those
    if cross_indices is not None:
        if args.verbose:
            print(f"Selecting cross indices: {cross_indices} out of {expected_cross_pairs} total cross pairs")
        
        # Select only the requested cross pairs
        selected_cross_cls = [cross_cls_cut_list[idx] for idx in cross_indices]
        cross_cls_cut_list = selected_cross_cls
    
    # Concatenate all (selected) cross pairs
    cross_data_vector = np.concatenate(cross_cls_cut_list, axis=1)
    
    if args.verbose:
        print(f"Final cross data shape: {cross_data_vector.shape}")
    
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
    
    # Compute f_sky if masked
    f_sky = None
    if args.masked:
        total_area_sqdeg = 41252.96125  # 4*pi*(180/pi)^2
        f_sky = args.mask_area_sqdeg / total_area_sqdeg
        if args.verbose:
            print(f"Computed f_sky = {f_sky:.4f} from mask area {args.mask_area_sqdeg:.0f} sq deg")
    
    # Process auto fiducial data according to cuts, f_sky correction, and rebinning
    auto_fid_data_list = []
    for fid_mean in auto_fid_means:
        # Apply cuts
        fid_cut = fid_mean[args.lower_cut:args.upper_cut]
        
        # Apply f_sky correction if masked
        if args.masked and f_sky is not None:
            fid_cut = fid_cut / f_sky
        
        # Apply rebinning
        if args.rebin > 1:
            fid_processed = rebin_cls(fid_cut, args.rebin)
        else:
            fid_processed = fid_cut
        
        auto_fid_data_list.append(fid_processed)
    
    if args.verbose and args.masked and f_sky is not None:
        print(f"Applied f_sky correction to auto fiducial: divided by {f_sky:.4f}")
    
    # Concatenate all auto bins' fiducial data
    auto_fid_mean_processed = np.concatenate(auto_fid_data_list)
    
    return auto_fid_mean_processed

def load_and_process_cross_fiducial(cross_fiducial_path, args, cross_indices=None, n_bins=None):
    """Load and process cross fiducial data.
    
    Args:
        cross_fiducial_path: Path to the cross fiducial file
        args: Argument namespace
        cross_indices: Optional list of cross-pair indices to select
        n_bins: Number of bins (needed to infer multipole range per cross-pair)
    """
    cross_fid_full = np.load(cross_fiducial_path, allow_pickle=True)
    cross_fid_mean = np.mean(cross_fid_full, axis=0)
    if args.verbose:
        print(f"Loaded cross fiducial data from {os.path.basename(cross_fiducial_path)}, shape: {cross_fid_full.shape}")
    
    # Compute f_sky if masked
    f_sky = None
    if args.masked:
        total_area_sqdeg = 41252.96125  # 4*pi*(180/pi)^2
        f_sky = args.mask_area_sqdeg / total_area_sqdeg
        if args.verbose:
            print(f"Computed f_sky = {f_sky:.4f} from mask area {args.mask_area_sqdeg:.0f} sq deg")
    
    # Calculate expected number of cross pairs and infer original multipole range
    if n_bins is not None:
        expected_cross_pairs = n_bins * (n_bins - 1) // 2
        n_ell_original = len(cross_fid_mean) // expected_cross_pairs
        
        if args.verbose:
            print(f"Expected {expected_cross_pairs} cross pairs with {n_ell_original} multipoles each")
    else:
        # Fallback: assume standard multipole range
        print("Warning: n_bins not provided for fiducial, assuming standard multipole range")
        n_ell_original = 1024
        expected_cross_pairs = len(cross_fid_mean) // n_ell_original
    
    # Apply cuts to each cross-pair individually
    n_multipoles_cut = args.upper_cut - args.lower_cut
    cross_fid_cut_list = []
    
    for i in range(expected_cross_pairs):
        # Extract this cross-pair's full multipole range
        start_idx = i * n_ell_original
        end_idx = (i + 1) * n_ell_original
        cross_pair_full = cross_fid_mean[start_idx:end_idx]
        
        # Apply multipole cuts to this cross-pair
        cross_pair_cut = cross_pair_full[args.lower_cut:args.upper_cut]
        
        # Apply f_sky correction if masked
        if args.masked and f_sky is not None:
            cross_pair_cut = cross_pair_cut / f_sky
        
        cross_fid_cut_list.append(cross_pair_cut)
    
    if args.verbose:
        print(f"Applied cuts to {len(cross_fid_cut_list)} cross pairs in fiducial")
        if args.masked and f_sky is not None:
            print(f"Applied f_sky correction to cross fiducial: divided by {f_sky:.4f}")
    
    # Apply rebinning to each cross-pair separately (before selection/concatenation)
    if args.rebin > 1:
        cross_fid_rebinned_list = []
        for cross_pair_cut in cross_fid_cut_list:
            # Rebin this cross-pair's fiducial spectrum
            rebinned = rebin_cls(cross_pair_cut, args.rebin)
            cross_fid_rebinned_list.append(rebinned)
        cross_fid_cut_list = cross_fid_rebinned_list
        
        if args.verbose:
            print(f"Applied rebinning with factor {args.rebin} to each cross pair in fiducial")
            print(f"Each cross pair now has {len(cross_fid_cut_list[0])} multipoles after rebinning")
    
    # If specific cross indices are requested, select only those
    if cross_indices is not None:
        if args.verbose:
            print(f"Selecting fiducial cross indices: {cross_indices} out of {expected_cross_pairs} total cross pairs")
        
        # Select only the requested cross pairs
        selected_cross_fid = [cross_fid_cut_list[idx] for idx in cross_indices]
        cross_fid_cut_list = selected_cross_fid
    
    # Concatenate all (selected) cross pairs
    cross_fid_processed = np.concatenate(cross_fid_cut_list)
    
    if args.verbose:
        print(f"Final cross fiducial shape: {cross_fid_processed.shape}")
    
    return cross_fid_processed

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
            print(f"  Processing simulation {i+1}/{n_test_sims}")
        
        # Generate samples from posterior
        sample_key, master_key = jax.random.split(master_key)
        samples_i = posterior.sample(
            x=jnp.array(x_obs), 
            num_samples=args.coverage_num_samples, 
            key=sample_key
        )
        all_samples.append(np.array(samples_i))
    
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

def main():
    args = parse_arguments()

    if args.masked:
        print(f"Using masked spectra with area ≈ {args.mask_area_sqdeg:.0f} sq deg (suffix: {args.mask_suffix})")
    
    # Parse cross pairs if specified
    cross_pairs = parse_cross_pairs(args.cross_pairs)
    if cross_pairs and args.verbose:
        print(f"Selected cross pairs: {cross_pairs}")
    
    # Construct file paths for auto power spectra
    params_path, auto_data_paths, auto_fiducial_paths, bin_desc = construct_auto_paths(args)
    
    # Determine which bin indices are being used for cross pair calculation
    if args.bnt:
        bnt_bin_indices = [int(b.strip()) for b in args.bnt_bins.split(',')]
        # For cross pair naming, BNT bins are labeled as 1,2,3,4 (corresponding to BNT bins 0,1,2,3)
        bin_indices = [b+1 for b in bnt_bin_indices]
    else:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
    
    # Number of bins (needed to correctly parse aggregated cross spectra files)
    n_bins = len(bin_indices)
    
    # Calculate cross indices if specific pairs are requested
    cross_indices = get_cross_indices_for_pairs(bin_indices, cross_pairs)
    if cross_indices and args.verbose:
        print(f"Cross indices to select: {cross_indices}")
    
    # Construct file paths for cross power spectra
    cross_data_path, cross_fiducial_path = construct_cross_paths(args, bin_desc)
    
    print(f"Using parameters file: {params_path}")
    print(f"Using {n_bins} bins: {bin_indices}")
    print(f"Using lmax: {args.lmax}")
    print(f"Using auto datavector files: {auto_data_paths}")
    print(f"Using cross datavector file: {cross_data_path}")
    print(f"Using auto fiducial files: {auto_fiducial_paths}")
    print(f"Using cross fiducial file: {cross_fiducial_path}")
    if cross_pairs:
        print(f"Using only cross pairs: {cross_pairs}")
    
    # Validate that required files exist
    missing_files = []
    if not os.path.exists(params_path):
        missing_files.append(params_path)
    
    if not args.cross_only:
        for path in auto_data_paths:
            if not os.path.exists(path):
                missing_files.append(path)
        for path in auto_fiducial_paths:
            if not os.path.exists(path):
                missing_files.append(path)
    
    if not args.auto_only:
        if not os.path.exists(cross_data_path):
            missing_files.append(cross_data_path)
        if not os.path.exists(cross_fiducial_path):
            missing_files.append(cross_fiducial_path)
    
    if missing_files:
        print("\n" + "="*60)
        print("ERROR: Required files not found!")
        print("="*60)
        for f in missing_files:
            print(f"  ✗ {f}")
        print("\nPossible causes:")
        print(f"  1. The data files were processed with a different --lmax (current: {args.lmax})")
        print(f"  2. The data files were processed with different noise settings (current: noisy={args.noisy}, level={args.noise_level})")
        if args.masked:
            print(f"  3. The data files were processed without masking or with a different mask area (current suffix: {args.mask_suffix or 'none'})")
            print(f"  4. The data files don't exist yet - run cross_power_spectrum_processing.py with --apply-mask first")
        else:
            print(f"  3. The data files don't exist yet - run cross_power_spectrum_processing.py first")
        print("\nTip: Check the actual filenames in the data directories to match --lmax, --noisy, and --noise-level")
        print("="*60 + "\n")
        return
    
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
        # Load cross power spectra with optional selection
        cross_data_vector = load_and_process_cross_spectra(cross_data_path, args, cross_indices, n_bins)
        cross_fid_vector = load_and_process_cross_fiducial(cross_fiducial_path, args, cross_indices, n_bins)
        
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
        cross_pairs_str = "_".join([f"{i}-{j}" for i, j in cross_pairs])
        if args.cross_only:
            spectra_desc = f"cross_{cross_pairs_str}"
        else:
            spectra_desc = f"auto_cross_{cross_pairs_str}"
    
    bnt_prefix = "bnt_" if args.bnt else ""
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    mask_suffix = args.mask_suffix if args.masked else ""
    
    # Save first example from training data
    example_train_filename = f"example_train_datavector_{bnt_prefix}{spectra_desc}_{args.simulation_type}_{bin_desc}_{ps_desc}{mask_suffix}{noise_suffix}.npy"
    np.save(os.path.join(args.samples_dir, example_train_filename), combined_data_vector[0])
    print(f"Saved example training datavector to {example_train_filename}")
    print(f"  Shape: {combined_data_vector[0].shape}, first 10 values: {combined_data_vector[0][:10]}")
    
    # Save fiducial observation
    example_fid_filename = f"example_fiducial_datavector_{bnt_prefix}{spectra_desc}_{args.fiducial_type}_{bin_desc}_{ps_desc}{mask_suffix}{noise_suffix}.npy"
    np.save(os.path.join(args.samples_dir, example_fid_filename), combined_fid_vector)
    print(f"Saved fiducial datavector to {example_fid_filename}")
    print(f"  Shape: {combined_fid_vector.shape}, first 10 values: {combined_fid_vector[:10]}")
    
    
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
    
    # Add cross pairs information to spectra type if specific pairs are selected
    if cross_pairs and not args.auto_only:
        cross_pairs_str = "_".join([f"{i}-{j}" for i, j in cross_pairs])
        if args.cross_only:
            spectra_type = f"cross_{cross_pairs_str}"
        else:
            spectra_type = f"auto_cross_{cross_pairs_str}"
    
    # Add BNT prefix to checkpoint name if using BNT data
    if args.bnt:
        datavector_desc = f"{args.simulation_type}_bnt_{bin_desc}_{ps_desc}_{spectra_type}"
    else:
        datavector_desc = f"{args.simulation_type}_{bin_desc}_{ps_desc}_{spectra_type}"

    if args.masked:
        datavector_desc += f"_{args.mask_label}"

    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"

    if args.bnt:
        checkpoint_name = f"cosmoGRID_bnt_ps_weights_{datavector_desc}"
    else:
        checkpoint_name = f"cosmoGRID_ps_weights_{datavector_desc}"

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

    # Run TARP coverage test if requested
    if args.run_coverage_test:
        ecp, alpha = run_tarp_coverage_test(posterior, combined_data_vector, params, args)
        
        # Create filename base for coverage plots
        bnt_prefix = "bnt_" if args.bnt else ""
        if args.auto_only:
            spectra_type = "auto"
        elif args.cross_only:
            spectra_type = "cross"
        else:
            spectra_type = "auto_cross"
        
        if cross_pairs and not args.auto_only:
            cross_pairs_str = "_".join([f"{i}-{j}" for i, j in cross_pairs])
            if args.cross_only:
                spectra_type = f"cross_{cross_pairs_str}"
            else:
                spectra_type = f"auto_cross_{cross_pairs_str}"
        
        coverage_filename_base = f"posterior_{bnt_prefix}ps_{spectra_type}_{args.simulation_type}_vs_{args.fiducial_type}_{bin_desc}_{ps_desc}"
        if args.masked:
            coverage_filename_base += f"_{args.mask_label}"
        if args.noisy:
            coverage_filename_base += f"_noisy_s{args.noise_level:.2f}"
        
        # Plot and save coverage diagnostics
        plot_tarp_coverage(ecp, alpha, args, args.output_dir, coverage_filename_base)

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
    if args.masked:
        fiducial_desc += f"_masked_{args.mask_area_tag}sqdeg"
    if args.noisy:
        fiducial_desc += f"_n{args.noise_level:.2f}"
    
    if args.auto_only:
        analysis_type = "BNT Auto Cls" if args.bnt else "Auto Cls"
    elif args.cross_only:
        base_type = "BNT Cross Cls" if args.bnt else "Cross Cls"
        if cross_pairs:
            cross_pairs_str = ",".join([f"({i},{j})" for i, j in cross_pairs])
            analysis_type = f"{base_type} {cross_pairs_str}"
        else:
            analysis_type = base_type
    else:
        base_type = "BNT Auto+Cross Cls" if args.bnt else "Auto+Cross Cls"
        if cross_pairs:
            cross_pairs_str = ",".join([f"({i},{j})" for i, j in cross_pairs])
            analysis_type = f"{base_type} {cross_pairs_str}"
        else:
            analysis_type = base_type
    
    sample_label = f"{args.simulation_type} {analysis_type} vs {fiducial_desc} fid, {bin_desc}, {ps_desc}"
    if args.masked:
        sample_label += f", masked {args.mask_area_tag} sq deg"
    
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
    if args.masked:
        plot_filename += f"_{args.mask_label}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = f"posterior_samples_{bnt_prefix}ps_{spectra_type}_{args.simulation_type}_vs_{args.fiducial_type}_{bin_desc}_{ps_desc}"
    if args.masked:
        samples_filename += f"_{args.mask_label}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin_scale.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")

if __name__ == "__main__":
    main()
