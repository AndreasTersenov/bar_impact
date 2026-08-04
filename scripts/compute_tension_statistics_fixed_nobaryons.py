#!/usr/bin/env python3
"""
Compute tension statistics between nobaryons and baryonified fiducial posteriors.

This is a modified version where for mask_area=5000 sqdeg, the nobaryons samples
are ALWAYS loaded from run 3, regardless of the --run argument.

This script calculates Q_DM tension metrics for all combinations of:
- upper-cut: 520 to 1020 (step 20)
- mask-area-sqdeg: 2000.0, 5000.0, 10000.0, 14000.0, 28000.0

Outputs two tables:
1. Full 6-parameter tension
2. Subset 3-parameter tension (Omega_m, sigma_8, w_0)

Usage:
  python scripts/compute_tension_statistics_fixed_nobaryons.py [--run N]
  
  --run N: Optional run number to load samples with '_runN' suffix
           Note: For mask_area=5000, nobaryons always uses run 3
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
LABELS = [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
LABELS_SUBSET = [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$"]
SUBSET_INDICES = [0, 1, 2]  # First 3 parameters

# Sample directory
SAMPLES_DIR = Path("/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples")

# Parameter ranges
UPPER_CUTS = list(range(340, 1021, 20))  # 520, 540, ..., 1000, 1020
MASK_AREAS = [2000.0, 5000.0, 10000.0, 14000.0, 28000.0, 35000.0]

# Run number (set via command line)
RUN_NUMBER = None

# Fixed run number for nobaryons at mask_area=5000
FIXED_NOBARYONS_RUN_FOR_5000 = 3

def get_sample_filename(fiducial_type, upper_cut, mask_area, run_number=None):
    """Construct filename for posterior samples."""
    # Determine lmax based on mask_area
    lmax = 1530 if mask_area == 14000.0 else 1535
    
    run_suffix = f"_run{run_number}" if run_number is not None else ""
    
    filename = (
        f"posterior_samples_ps_auto_cross_nobaryons_vs_{fiducial_type}_"
        f"bins1234_l100-{upper_cut}_r10_masked_{int(mask_area)}sqdeg_apod2.0_master_noisy_s0.26{run_suffix}.npy"
    )
    return SAMPLES_DIR / filename

def load_mcsamples(fiducial_type, upper_cut, mask_area, labels, run_number=None):
    """Load posterior samples and create MCSamples object."""
    filepath = get_sample_filename(fiducial_type, upper_cut, mask_area, run_number)
    
    if not filepath.exists():
        return None
    
    samples = np.load(filepath)
    run_str = f"run{run_number}" if run_number is not None else "default"
    label = f"{fiducial_type}, ucut={upper_cut}, mask={int(mask_area)}, {run_str}"
    
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
    global RUN_NUMBER
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute tension statistics between nobaryons and baryonified posteriors")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number to load samples with '_runN' suffix")
    args = parser.parse_args()
    
    RUN_NUMBER = args.run
    run_suffix = f"_run{RUN_NUMBER}" if RUN_NUMBER is not None else ""
    
    print("="*80)
    print("Computing Tension Statistics: nobaryons vs baryonified")
    print("*** FIXED NOBARYONS VERSION: mask_area=5000 always uses run 3 for nobaryons ***")
    if RUN_NUMBER is not None:
        print(f"RUN NUMBER: {RUN_NUMBER}")
    print("="*80)
    print()
    
    # Storage for results
    results_full = []
    results_subset = []
    
    # Iterate over all combinations
    total = len(UPPER_CUTS) * len(MASK_AREAS)
    count = 0
    
    for upper_cut in UPPER_CUTS:
        for mask_area in MASK_AREAS:
            count += 1
            print(f"Processing [{count}/{total}]: upper_cut={upper_cut}, mask_area={int(mask_area)}")
            
            # Determine which run to use for nobaryons
            # For mask_area=5000, always use run 3 for nobaryons
            if mask_area == 5000.0:
                nobaryons_run = FIXED_NOBARYONS_RUN_FOR_5000
                print(f"  → Using fixed run {nobaryons_run} for nobaryons (mask_area=5000)")
            else:
                nobaryons_run = RUN_NUMBER
            
            # Load samples for both fiducial types
            mcsamples_nobar = load_mcsamples("nobaryons", upper_cut, mask_area, LABELS, run_number=nobaryons_run)
            mcsamples_bar = load_mcsamples("baryonified", upper_cut, mask_area, LABELS, run_number=RUN_NUMBER)
            
            if mcsamples_nobar is None:
                print(f"  ⚠ Missing: nobaryons sample (run={nobaryons_run})")
                continue
            if mcsamples_bar is None:
                print(f"  ⚠ Missing: baryonified sample (run={RUN_NUMBER})")
                continue
            
            # Compute full 6-parameter tension
            Q_DM, dofs, P, nsigma = compute_tension(mcsamples_nobar, mcsamples_bar)
            if Q_DM is not None:
                results_full.append({
                    'upper_cut': upper_cut,
                    'mask_area': int(mask_area),
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
                    'upper_cut': upper_cut,
                    'mask_area': int(mask_area),
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
    
    # Save to CSV (with _fixed_nobaryons suffix to distinguish from regular version)
    output_dir = Path("outputs/tension_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_full = output_dir / f"tension_nobaryons_vs_baryonified_full_6params_fixed_nobaryons{run_suffix}.csv"
    csv_subset = output_dir / f"tension_nobaryons_vs_baryonified_subset_3params_fixed_nobaryons{run_suffix}.csv"
    
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
    print("SUBSET 3-PARAMETER TENSION TABLE (Omega_m, sigma_8, w_0)")
    print("="*80)
    print(df_subset.to_string(index=False))
    print()
    
    # Create pivot tables for easier viewing
    if len(df_full) > 0:
        print("="*80)
        print("FULL 6-PARAMETER TENSION - nsigma (rows=upper_cut, cols=mask_area)")
        print("="*80)
        pivot_full = df_full.pivot(index='upper_cut', columns='mask_area', values='nsigma')
        print(pivot_full.to_string(float_format=lambda x: f'{x:.3f}'))
        print()
        
        # Save pivot table
        pivot_csv_full = output_dir / f"tension_nobaryons_vs_baryonified_full_6params_pivot_fixed_nobaryons{run_suffix}.csv"
        pivot_full.to_csv(pivot_csv_full, float_format='%.3f')
        print(f"Saved pivot table to: {pivot_csv_full}")
        print()
    
    if len(df_subset) > 0:
        print("="*80)
        print("SUBSET 3-PARAMETER TENSION - nsigma (rows=upper_cut, cols=mask_area)")
        print("="*80)
        pivot_subset = df_subset.pivot(index='upper_cut', columns='mask_area', values='nsigma')
        print(pivot_subset.to_string(float_format=lambda x: f'{x:.3f}'))
        print()
        
        # Save pivot table
        pivot_csv_subset = output_dir / f"tension_nobaryons_vs_baryonified_subset_3params_pivot_fixed_nobaryons{run_suffix}.csv"
        pivot_subset.to_csv(pivot_csv_subset, float_format='%.3f')
        print(f"Saved pivot table to: {pivot_csv_subset}")
        print()

if __name__ == "__main__":
    main()
