#!/usr/bin/env python3
"""
Inspect saved datavector examples to verify correct structure.
This helps debug and verify that cross power spectra are being handled correctly.
"""

import numpy as np
import sys
import os
import glob

def inspect_datavector(filepath, n_bins=4, lower_cut=30, upper_cut=1024, rebin=1):
    """
    Inspect a saved datavector and break down its structure.
    
    Args:
        filepath: Path to the .npy file
        n_bins: Number of redshift bins
        lower_cut: Lower multipole cut
        upper_cut: Upper multipole cut
        rebin: Rebinning factor
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    data = np.load(filepath)
    print(f"Total shape: {data.shape}")
    print(f"Total length: {len(data)}")
    
    # Calculate expected sizes
    n_multipoles_cut = upper_cut - lower_cut
    n_multipoles_rebinned = n_multipoles_cut // rebin
    n_auto = n_bins * n_multipoles_rebinned
    n_cross_pairs = n_bins * (n_bins - 1) // 2
    n_cross = n_cross_pairs * n_multipoles_rebinned
    
    print(f"\nExpected structure:")
    print(f"  Number of bins: {n_bins}")
    print(f"  Multipole range after cuts: l={lower_cut} to l={upper_cut} ({n_multipoles_cut} values)")
    print(f"  After rebinning by {rebin}: {n_multipoles_rebinned} values per spectrum")
    print(f"  Auto spectra: {n_bins} bins × {n_multipoles_rebinned} = {n_auto}")
    print(f"  Cross spectra: {n_cross_pairs} pairs × {n_multipoles_rebinned} = {n_cross}")
    
    # Determine what type of datavector this is
    filename = os.path.basename(filepath)
    
    if "AUTO_ONLY" in filename:
        print(f"\nThis is AUTO ONLY data")
        print(f"  Expected length: {n_auto}")
        if len(data) == n_auto:
            print(f"  ✓ Length matches!")
        else:
            print(f"  ✗ Length mismatch! Got {len(data)}, expected {n_auto}")
        
        # Show breakdown by bin
        print(f"\nBreakdown by bin:")
        for i in range(n_bins):
            start = i * n_multipoles_rebinned
            end = (i + 1) * n_multipoles_rebinned
            if end <= len(data):
                print(f"  Bin {i+1}: indices [{start}:{end}], mean={np.mean(data[start:end]):.6e}, std={np.std(data[start:end]):.6e}")
    
    elif "CROSS_ONLY" in filename:
        print(f"\nThis is CROSS ONLY data")
        
        # Try to figure out how many cross pairs from the filename
        if "cross_1-2_1-3_1-4_2-3_2-4" in filename:
            # Specific pairs selected
            selected_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4)]
            n_selected = len(selected_pairs)
            expected_length = n_selected * n_multipoles_rebinned
            print(f"  Selected cross pairs: {selected_pairs}")
            print(f"  Expected length: {n_selected} pairs × {n_multipoles_rebinned} = {expected_length}")
        else:
            # All cross pairs
            expected_length = n_cross
            print(f"  All cross pairs (total: {n_cross_pairs})")
            print(f"  Expected length: {expected_length}")
        
        if len(data) == expected_length:
            print(f"  ✓ Length matches!")
        else:
            print(f"  ✗ Length mismatch! Got {len(data)}, expected {expected_length}")
        
        # Show breakdown by cross pair
        print(f"\nBreakdown by cross pair:")
        actual_pairs = len(data) // n_multipoles_rebinned
        for i in range(actual_pairs):
            start = i * n_multipoles_rebinned
            end = (i + 1) * n_multipoles_rebinned
            if end <= len(data):
                print(f"  Cross pair {i+1}: indices [{start}:{end}], mean={np.mean(data[start:end]):.6e}, std={np.std(data[start:end]):.6e}")
    
    else:
        # Combined auto + cross
        print(f"\nThis is AUTO + CROSS combined data")
        
        # Try to figure out cross pair count from filename
        if "cross_1-2_1-3_1-4_2-3_2-4" in filename:
            selected_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4)]
            n_selected_cross = len(selected_pairs)
            n_cross_component = n_selected_cross * n_multipoles_rebinned
            print(f"  Selected cross pairs: {selected_pairs}")
        else:
            n_cross_component = n_cross
            print(f"  All cross pairs")
        
        expected_total = n_auto + n_cross_component
        print(f"  Expected: {n_auto} (auto) + {n_cross_component} (cross) = {expected_total}")
        
        if len(data) == expected_total:
            print(f"  ✓ Length matches!")
        else:
            print(f"  ✗ Length mismatch! Got {len(data)}, expected {expected_total}")
        
        # Show auto component
        print(f"\nAuto component (first {n_auto} values):")
        for i in range(n_bins):
            start = i * n_multipoles_rebinned
            end = (i + 1) * n_multipoles_rebinned
            if end <= len(data):
                print(f"  Bin {i+1}: indices [{start}:{end}], mean={np.mean(data[start:end]):.6e}, std={np.std(data[start:end]):.6e}")
        
        # Show cross component
        print(f"\nCross component (next {n_cross_component} values):")
        actual_cross_pairs = n_cross_component // n_multipoles_rebinned
        for i in range(actual_cross_pairs):
            start = n_auto + i * n_multipoles_rebinned
            end = n_auto + (i + 1) * n_multipoles_rebinned
            if end <= len(data):
                print(f"  Cross pair {i+1}: indices [{start}:{end}], mean={np.mean(data[start:end]):.6e}, std={np.std(data[start:end]):.6e}")
    
    # Show some sample values
    print(f"\nFirst 20 values:")
    print(data[:20])
    
    print(f"\nLast 20 values:")
    print(data[-20:])
    
    # Statistics
    print(f"\nOverall statistics:")
    print(f"  Min: {np.min(data):.6e}")
    print(f"  Max: {np.max(data):.6e}")
    print(f"  Mean: {np.mean(data):.6e}")
    print(f"  Std: {np.std(data):.6e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect saved datavector examples")
    parser.add_argument("files", nargs="*", help="Specific files to inspect. If not provided, searches in outputs/samples/")
    parser.add_argument("--samples-dir", type=str, 
                       default="/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples",
                       help="Directory containing saved datavector examples")
    parser.add_argument("--n-bins", type=int, default=4,
                       help="Number of bins (default: 4)")
    parser.add_argument("--lower-cut", type=int, default=30,
                       help="Lower multipole cut (default: 30)")
    parser.add_argument("--upper-cut", type=int, default=1024,
                       help="Upper multipole cut (default: 1024)")
    parser.add_argument("--rebin", type=int, default=1,
                       help="Rebinning factor (default: 1)")
    
    args = parser.parse_args()
    
    if args.files:
        files = args.files
    else:
        # Find all example datavector files
        pattern = os.path.join(args.samples_dir, "example_*.npy")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"No example datavector files found in {args.samples_dir}")
            print(f"Run the inference script first to generate example files.")
            return
    
    print(f"Found {len(files)} file(s) to inspect")
    
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        inspect_datavector(filepath, 
                          n_bins=args.n_bins,
                          lower_cut=args.lower_cut,
                          upper_cut=args.upper_cut,
                          rebin=args.rebin)


if __name__ == "__main__":
    main()
