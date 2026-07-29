#!/usr/bin/env python3
"""
Compute tension statistics between nobaryons and baryonified fiducial posteriors using FULL SKY power spectra.

This script calculates Q_DM tension metrics for both normal and BNT-transformed data:
- ell-max: l30-1024 (common for both normal and BNT)
- rebinning factor: r10, r20 (common for both normal and BNT)

Outputs two tables per mode:
1. Full 6-parameter tension
2. Subset 3-parameter tension (Omega_m, S_8, w_0)

Usage:
  python scripts/compute_tension_statistics_fullsky.py
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

# Parameter ranges for full sky - using values common to both normal and BNT
ELL_MAX_VALUES = [1024]  # Common ell_max for both normal and BNT
REBINNING_FACTORS = [10, 20]  # Common rebinning factors for both normal and BNT

def get_sample_filename(fiducial_type, ell_max, rebin_factor, use_bnt=False):
    """Construct filename for posterior samples (full sky, no masking)."""
    if use_bnt:
        # BNT samples use 'bntbins1234' instead of 'bins1234'
        filename = (
            f"posterior_samples_ps_nobaryons_vs_{fiducial_type}_"
            f"bntbins1234_l30-{ell_max}_r{rebin_factor}_noisy_s0.26_npe.npy"
        )
    else:
        # Normal samples
        filename = (
            f"posterior_samples_ps_auto_cross_nobaryons_vs_{fiducial_type}_"
            f"bins1234_l30-{ell_max}_r{rebin_factor}_noisy_s0.26_npe.npy"
        )
    return SAMPLES_DIR / filename

def load_mcsamples(fiducial_type, ell_max, rebin_factor, labels, use_bnt=False):
    """Load posterior samples and create MCSamples object."""
    filepath = get_sample_filename(fiducial_type, ell_max, rebin_factor, use_bnt)
    
    if not filepath.exists():
        return None
    
    samples = np.load(filepath)
    bnt_label = " (BNT)" if use_bnt else ""
    label = f"{fiducial_type}{bnt_label}, lmax={ell_max}, r={rebin_factor}"
    
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
    print("="*80)
    print("Computing Tension Statistics: nobaryons vs baryonified (FULL SKY)")
    print("="*80)
    print()
    
    # Process both normal and BNT modes
    for use_bnt in [False, True]:
        mode_name = "BNT" if use_bnt else "Normal"
        mode_suffix = "_bnt" if use_bnt else ""
        
        print("="*80)
        print(f"MODE: {mode_name}")
        print("="*80)
        print()
        
        # Storage for results
        results_full = []
        results_subset = []
        
        # Iterate over all combinations
        total = len(ELL_MAX_VALUES) * len(REBINNING_FACTORS)
        count = 0
        
        for ell_max in ELL_MAX_VALUES:
            for rebin_factor in REBINNING_FACTORS:
                count += 1
                print(f"Processing [{count}/{total}]: ell_max={ell_max}, rebin_factor={rebin_factor}")
                
                # Load samples for both fiducial types
                mcsamples_nobar = load_mcsamples("nobaryons", ell_max, rebin_factor, LABELS, use_bnt)
                mcsamples_bar = load_mcsamples("baryonified", ell_max, rebin_factor, LABELS, use_bnt)
                
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
                        'ell_max': ell_max,
                        'rebin_factor': rebin_factor,
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
                        'ell_max': ell_max,
                        'rebin_factor': rebin_factor,
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
        
        csv_full = output_dir / f"tension_fullsky{mode_suffix}_nobaryons_vs_baryonified_full_6params.csv"
        csv_subset = output_dir / f"tension_fullsky{mode_suffix}_nobaryons_vs_baryonified_subset_3params.csv"
        
        df_full.to_csv(csv_full, index=False, float_format='%.5f')
        df_subset.to_csv(csv_subset, index=False, float_format='%.5f')
        
        print("="*80)
        print(f"Results saved for {mode_name} mode!")
        print("="*80)
        print(f"Full 6-parameter table: {csv_full}")
        print(f"Subset 3-parameter table: {csv_subset}")
        print()
        
        # Print summary tables
        print("="*80)
        print(f"FULL 6-PARAMETER TENSION TABLE ({mode_name})")
        print("="*80)
        print(df_full.to_string(index=False))
        print()
        
        print("="*80)
        print(f"SUBSET 3-PARAMETER TENSION TABLE (Omega_m, S_8, w_0) ({mode_name})")
        print("="*80)
        print(df_subset.to_string(index=False))
        print()
        
        # Create pivot tables for easier viewing
        if len(df_full) > 0:
            print("="*80)
            print(f"FULL 6-PARAMETER TENSION - nsigma (rows=ell_max, cols=rebin_factor) ({mode_name})")
            print("="*80)
            pivot_full = df_full.pivot(index='ell_max', columns='rebin_factor', values='nsigma')
            print(pivot_full.to_string(float_format=lambda x: f'{x:.3f}'))
            print()
            
            # Save pivot table
            pivot_csv_full = output_dir / f"tension_fullsky{mode_suffix}_nobaryons_vs_baryonified_full_6params_pivot.csv"
            pivot_full.to_csv(pivot_csv_full, float_format='%.3f')
            print(f"Saved pivot table to: {pivot_csv_full}")
            print()
        
        if len(df_subset) > 0:
            print("="*80)
            print(f"SUBSET 3-PARAMETER TENSION - nsigma (rows=ell_max, cols=rebin_factor) ({mode_name})")
            print("="*80)
            pivot_subset = df_subset.pivot(index='ell_max', columns='rebin_factor', values='nsigma')
            print(pivot_subset.to_string(float_format=lambda x: f'{x:.3f}'))
            print()
            
            # Save pivot table
            pivot_csv_subset = output_dir / f"tension_fullsky{mode_suffix}_nobaryons_vs_baryonified_subset_3params_pivot.csv"
            pivot_subset.to_csv(pivot_csv_subset, float_format='%.3f')
            print(f"Saved pivot table to: {pivot_csv_subset}")
            print()
        
        print()  # Extra space between modes

if __name__ == "__main__":
    main()
