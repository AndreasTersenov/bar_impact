#!/usr/bin/env python3
"""
Compute tension statistics between nobaryons and baryonified fiducial posteriors for L1-norm.

This script calculates Q_DM tension metrics for all combinations of:
- mask-area-sqdeg: 2001.0, 5001.0, 10001.0, 14001.0, 28001.0, 35001.0
- scales: "0,1,2,3" and "1,2,3"

Outputs two tables:
1. Full 6-parameter tension
2. Subset 3-parameter tension (Omega_m, S_8, w_0)

Usage:
  python scripts/compute_tension_statistics_l1.py [--run N] [--bnt]
  
  --run N: Optional run number to load samples with '_runN' suffix
  --bnt: Use BNT-transformed data samples
"""

import argparse
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path
from getdist import MCSamples
import tensiometer.utilities as utilities
from tensiometer import gaussian_tension

# Parameter labels
LABELS = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
LABELS_SUBSET = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$"]
SUBSET_INDICES = [0, 1, 2]  # First 3 parameters

# Sample directory
SAMPLES_DIR = Path("/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples")

# Parameter ranges (matching submit_npe_inference_l1_parameter_sweep_parallel.py)
MASK_AREAS = [2001.0, 2002.0, 5001.0, 10001.0, 14001.0, 28001.0, 35001.0]
SCALE_CONFIGS = ["0,1,2,3", "1,2,3"]  # Wavelet configurations

# Run number (set via command line)
RUN_NUMBER = None

# BNT flag (set via command line)
USE_BNT = False

def get_sample_filename(fiducial_type, mask_area, scales):
    """Construct filename for posterior samples."""
    # Construct scale description (scales are 1-indexed in filename)
    scale_indices = [int(s.strip()) for s in scales.split(',')]
    scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
    
    # For BNT mode with "1,2,3" scales, we use per-bin scales so add "_perbin" suffix
    if USE_BNT and scales == "1,2,3":
        scale_desc += "_perbin"
    
    # Construct bin description
    # For BNT mode: use "bntbins1234", for standard mode: use "bins1234"
    bin_spec = "bntbins1234" if USE_BNT else "bins1234"
    
    run_suffix = f"_run{RUN_NUMBER}" if RUN_NUMBER is not None else ""
    
    # Format: posterior_samples_{simulation_type}_vs_{fiducial_type}_{bin_spec}_{scale_desc}_noisy_s{noise}_masked_{mask}sqdeg_new_normalization_npe.npy
    # Note: BNT mode does NOT add "bnt_" prefix - the bin_spec contains "bntbins" instead
    filename = (
        f"posterior_samples_nobaryons_vs_{fiducial_type}_{bin_spec}_{scale_desc}_"
        f"noisy_s0.26_masked_{int(mask_area)}sqdeg_new_normalization{run_suffix}_npe.npy"
    )
    return SAMPLES_DIR / filename

def load_mcsamples(fiducial_type, mask_area, scales, labels):
    """Load posterior samples and create MCSamples object."""
    filepath = get_sample_filename(fiducial_type, mask_area, scales)
    
    if not filepath.exists():
        return None
    
    samples = np.load(filepath)
    scale_tag = scales.replace(",", "")
    label = f"{fiducial_type}, scales={scale_tag}, mask={int(mask_area)}"
    
    return MCSamples(samples=samples, names=labels, label=label)

def subset_mcsamples(mcsamples, indices, subset_labels):
    """Extract subset of parameters from MCSamples."""
    subset_samples = mcsamples.samples[:, indices]
    return MCSamples(samples=subset_samples, names=subset_labels, label=mcsamples.label)

def compute_tension(mcsamples1, mcsamples2):
    """Compute Q_DM tension statistics between two MCSamples."""
    try:
        Q_DM, Q_DM_dofs = gaussian_tension.Q_DM(mcsamples1, mcsamples2)
        Q_DM_P = scipy.stats.chi2.cdf(Q_DM, Q_DM_dofs)
        Q_DM_nsigma = utilities.from_confidence_to_sigma(Q_DM_P)
        return Q_DM, Q_DM_dofs, Q_DM_P, Q_DM_nsigma
    except Exception as e:
        print(f"Error computing tension: {e}")
        return None, None, None, None

def main():
    global RUN_NUMBER, USE_BNT
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute L1-norm tension statistics between nobaryons and baryonified posteriors")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number to load samples with '_runN' suffix")
    parser.add_argument("--bnt", action="store_true",
                        help="Use BNT-transformed data samples")
    args = parser.parse_args()
    
    RUN_NUMBER = args.run
    USE_BNT = args.bnt
    run_suffix = f"_run{RUN_NUMBER}" if RUN_NUMBER is not None else ""
    bnt_suffix = "_bnt" if USE_BNT else ""
    
    print("="*80)
    print("Computing L1-Norm Tension Statistics: nobaryons vs baryonified")
    if USE_BNT:
        print("MODE: BNT-transformed data")
    if RUN_NUMBER is not None:
        print(f"RUN NUMBER: {RUN_NUMBER}")
    print("="*80)
    print()
    
    # Storage for results
    results_full = []
    results_subset = []
    
    # Iterate over all combinations
    total = len(MASK_AREAS) * len(SCALE_CONFIGS)
    count = 0
    
    for mask_area in MASK_AREAS:
        for scales in SCALE_CONFIGS:
            count += 1
            scale_tag = scales.replace(",", "")
            print(f"Processing [{count}/{total}]: mask_area={int(mask_area)}, scales={scale_tag}")
            
            # Load samples for both fiducial types
            mcsamples_nobar = load_mcsamples("nobaryons", mask_area, scales, LABELS)
            mcsamples_bar = load_mcsamples("baryonified", mask_area, scales, LABELS)
            
            if mcsamples_nobar is None:
                print(f"  ⚠ Missing: nobaryons sample")
                continue
            if mcsamples_bar is None:
                print(f"  ⚠ Missing: baryonified sample")
                continue
            
            # Compute full 6-parameter tension
            Q_DM, dofs, P, nsigma = compute_tension(mcsamples_nobar, mcsamples_bar)
            if Q_DM is not None:
                results_full.append({
                    'mask_area': int(mask_area),
                    'scales': scale_tag,
                    'Q_DM': Q_DM,
                    'dofs': dofs,
                    'P': P,
                    'nsigma': nsigma
                })
                print(f"  Full (6 params): Q_DM={Q_DM:.2f}, dofs={dofs}, P={P:.5f}, nsigma={nsigma:.3f}")
            
            # Compute subset 3-parameter tension
            mcsamples_nobar_sub = subset_mcsamples(mcsamples_nobar, SUBSET_INDICES, LABELS_SUBSET)
            mcsamples_bar_sub = subset_mcsamples(mcsamples_bar, SUBSET_INDICES, LABELS_SUBSET)
            
            Q_DM_sub, dofs_sub, P_sub, nsigma_sub = compute_tension(mcsamples_nobar_sub, mcsamples_bar_sub)
            if Q_DM_sub is not None:
                results_subset.append({
                    'mask_area': int(mask_area),
                    'scales': scale_tag,
                    'Q_DM': Q_DM_sub,
                    'dofs': dofs_sub,
                    'P': P_sub,
                    'nsigma': nsigma_sub
                })
                print(f"  Subset (3 params): Q_DM={Q_DM_sub:.2f}, dofs={dofs_sub}, P={P_sub:.5f}, nsigma={nsigma_sub:.3f}")
            
            print()
    
    # Create DataFrames
    df_full = pd.DataFrame(results_full)
    df_subset = pd.DataFrame(results_subset)
    
    # Save to CSV
    output_dir = Path("outputs/tension_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_full = output_dir / f"tension_l1{bnt_suffix}_nobaryons_vs_baryonified_full_6params{run_suffix}.csv"
    csv_subset = output_dir / f"tension_l1{bnt_suffix}_nobaryons_vs_baryonified_subset_3params{run_suffix}.csv"
    
    df_full.to_csv(csv_full, index=False, float_format='%.5f')
    df_subset.to_csv(csv_subset, index=False, float_format='%.5f')
    
    print("="*80)
    print("Results saved!")
    print("="*80)
    print(f"Full 6-parameter table: {csv_full}")
    print(f"Subset 3-parameter table: {csv_subset}")
    print()
    
    # Print summary tables
    print("="*80)
    print("FULL 6-PARAMETER TENSION TABLE")
    print("="*80)
    print(df_full.to_string(index=False))
    print()
    
    print("="*80)
    print("SUBSET 3-PARAMETER TENSION TABLE (Omega_m, S_8, w_0)")
    print("="*80)
    print(df_subset.to_string(index=False))
    print()
    
    # Create pivot tables for easier viewing
    if len(df_full) > 0:
        print("="*80)
        print("FULL 6-PARAMETER TENSION - nsigma (rows=mask_area, cols=scales)")
        print("="*80)
        pivot_full = df_full.pivot(index='mask_area', columns='scales', values='nsigma')
        print(pivot_full.to_string(float_format=lambda x: f'{x:.3f}'))
        print()
        
        # Save pivot table
        pivot_csv_full = output_dir / f"tension_l1{bnt_suffix}_nobaryons_vs_baryonified_full_6params_pivot{run_suffix}.csv"
        pivot_full.to_csv(pivot_csv_full, float_format='%.3f')
        print(f"Saved pivot table to: {pivot_csv_full}")
        print()
    
    if len(df_subset) > 0:
        print("="*80)
        print("SUBSET 3-PARAMETER TENSION - nsigma (rows=mask_area, cols=scales)")
        print("="*80)
        pivot_subset = df_subset.pivot(index='mask_area', columns='scales', values='nsigma')
        print(pivot_subset.to_string(float_format=lambda x: f'{x:.3f}'))
        print()
        
        # Save pivot table
        pivot_csv_subset = output_dir / f"tension_l1{bnt_suffix}_nobaryons_vs_baryonified_subset_3params_pivot{run_suffix}.csv"
        pivot_subset.to_csv(pivot_csv_subset, float_format='%.3f')
        print(f"Saved pivot table to: {pivot_csv_subset}")
        print()

if __name__ == "__main__":
    main()
