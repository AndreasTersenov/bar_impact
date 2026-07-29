#!/usr/bin/env python3
"""
Example: Power Spectrum Processing using the bar_impact package

This example shows how to process convergence maps to compute cross power spectra
with BNT transformation and MASTER mode-coupling correction.

Equivalent to running:
    python scripts/bnt_cross_power_spectrum_processing_master.py \
        --lmax 1535 --num-workers 20 \
        --apply-mask --mask-area-sqdeg 10000 --apodization-scale-deg 2.0 \
        --save-combined --aggregate-for-inference --fiducial --baryonified
"""

import os
import numpy as np
import h5py
import glob
import healpy as hp
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from itertools import combinations
from tqdm import tqdm

# Import from the new package
from bar_impact.core import ConvergenceMap, SurveyMask
from bar_impact.processing import PowerSpectrumProcessor, PowerSpectrumConfig
from bar_impact.analysis import ResultsAggregator, aggregate_power_spectra
from bar_impact.constants import BNT_MATRIX


# ============================================================================
# Configuration (equivalent to argparse arguments)
# ============================================================================

DATA_DIR = "/home/tersenov/CosmoGridV1/stage3_forecast"
LMAX = 1535
NUM_WORKERS = 20
APPLY_MASK = True
MASK_AREA_SQDEG = 10000.0
APODIZATION_SCALE_DEG = 2.0
SAVE_COMBINED = True
AGGREGATE_FOR_INFERENCE = True
PROCESS_FIDUCIAL = True
PROCESS_BARYONIFIED = True
OUTPUT_DIR = "./outputs/power_spectra"
NSIDE = 512
NOISE_LEVEL = 0.26
BINS = [1, 2, 3, 4]


# ============================================================================
# Processing Functions  
# ============================================================================

def get_file_list(data_dir: str, sim_type: str = "baryonified") -> list:
    """Get list of simulation files to process."""
    pattern = os.path.join(data_dir, sim_type, "*.h5")
    return sorted(glob.glob(pattern))


def load_all_bins(file_path: str, bins: list, nside: int = 512) -> dict:
    """Load convergence maps for all bins from an HDF5 file."""
    maps = {}
    with h5py.File(file_path, "r") as f:
        for bin_num in bins:
            key = f"kg/stage3_lensing{bin_num}"
            data = np.array(f[key])
            maps[bin_num] = ConvergenceMap(data=data, nside=nside, bin_number=bin_num)
    return maps


def add_noise_to_maps(maps: dict, noise_level: float) -> dict:
    """Add shape noise to all maps."""
    noisy_maps = {}
    for bin_num, kmap in maps.items():
        noisy_maps[bin_num] = kmap.add_shape_noise(sigma_e=noise_level)
    return noisy_maps


def apply_bnt_transform(maps: dict) -> dict:
    """Apply BNT transformation to maps."""
    # Stack maps in bin order
    bin_nums = sorted(maps.keys())
    stacked = np.array([maps[b].data for b in bin_nums])
    
    # Apply BNT transformation
    transformed = BNT_MATRIX @ stacked
    
    # Create new maps
    bnt_maps = {}
    for i, bin_num in enumerate(bin_nums):
        bnt_maps[bin_num] = ConvergenceMap(
            data=transformed[i],
            nside=maps[bin_num].nside,
            bin_number=bin_num,
        )
    return bnt_maps


def process_single_file(
    file_path: str,
    processor: PowerSpectrumProcessor,
    mask: SurveyMask | None,
    noise_level: float,
    output_dir: str,
    apply_bnt: bool = True,
) -> dict | None:
    """
    Process a single file to compute auto and cross power spectra.
    
    Returns a dict with keys like 'auto_1', 'auto_2', 'cross_1_2', etc.
    """
    try:
        # Load all bins
        maps = load_all_bins(file_path, BINS, NSIDE)
        
        # Add shape noise BEFORE BNT (for physical consistency)
        if noise_level > 0:
            maps = add_noise_to_maps(maps, noise_level)
        
        # Apply BNT transform
        if apply_bnt:
            maps = apply_bnt_transform(maps)
        
        # Apply mask if requested
        if mask:
            for bin_num in maps:
                masked_data = maps[bin_num].data * mask.data
                maps[bin_num] = ConvergenceMap(
                    data=masked_data,
                    nside=NSIDE,
                    bin_number=bin_num,
                )
        
        results = {}
        bin_nums = sorted(maps.keys())
        
        # Compute auto power spectra
        for bin_num in bin_nums:
            cls = processor.process_single(maps[bin_num].data)
            results[f"auto_{bin_num}"] = cls
        
        # Compute cross power spectra
        for bin_i, bin_j in combinations(bin_nums, 2):
            cls = processor.compute_cross_spectrum(
                maps[bin_i].data, 
                maps[bin_j].data
            )
            results[f"cross_{bin_i}_{bin_j}"] = cls
        
        return results
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def combine_and_save_results(
    all_results: list,
    output_dir: str,
    sim_type: str,
    mask_area: float | None,
    noise_level: float,
) -> dict:
    """Combine results from all files and save."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all keys from first result
    keys = list(all_results[0].keys())
    
    combined = {}
    mask_tag = f"_masked_{int(mask_area)}sqdeg" if mask_area else ""
    noise_tag = f"_noisy_s{noise_level:.2f}" if noise_level > 0 else ""
    
    for key in keys:
        # Stack results for this key
        stacked = np.array([r[key] for r in all_results if r is not None])
        combined[key] = stacked
        
        # Save individual file
        output_path = os.path.join(
            output_dir, 
            f"{sim_type}_bnt_{key}{mask_tag}{noise_tag}.npy"
        )
        np.save(output_path, stacked)
        print(f"Saved {key}: shape {stacked.shape}")
    
    return combined


def aggregate_for_inference(
    combined_results: dict,
    output_dir: str,
    sim_type: str,
    ell_range: tuple = (100, 1024),
) -> np.ndarray:
    """
    Aggregate power spectra into a single data vector for NPE inference.
    
    Combines auto and cross spectra, applies ell cuts.
    """
    # Order: auto_1, auto_2, auto_3, auto_4, cross_1_2, cross_1_3, etc.
    ordered_keys = []
    
    # Auto spectra
    for i in BINS:
        ordered_keys.append(f"auto_{i}")
    
    # Cross spectra
    for i, j in combinations(BINS, 2):
        ordered_keys.append(f"cross_{i}_{j}")
    
    # Extract and concatenate
    ell_min, ell_max = ell_range
    vectors = []
    
    for key in ordered_keys:
        if key in combined_results:
            cls = combined_results[key]
            # Apply ell cut
            cls_cut = cls[:, ell_min:ell_max+1]
            vectors.append(cls_cut)
    
    # Concatenate along feature axis
    datavector = np.concatenate(vectors, axis=1)
    
    # Save
    output_path = os.path.join(
        output_dir,
        f"{sim_type}_bnt_datavector_l{ell_min}-{ell_max}.npy"
    )
    np.save(output_path, datavector)
    print(f"Saved data vector: {datavector.shape} to {output_path}")
    
    return datavector


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def main():
    """Main processing pipeline."""
    
    print("=" * 60)
    print("Power Spectrum Processing with bar_impact package")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create apodized mask if requested
    mask = None
    if APPLY_MASK:
        mask = SurveyMask.create_apodized_mask(
            nside=NSIDE,
            target_area_sqdeg=MASK_AREA_SQDEG,
            center_coords=(0.0, 90.0),
            apodization_scale_deg=APODIZATION_SCALE_DEG,
        )
        print(f"Created apodized mask: area={mask.area_sqdeg:.1f} sq.deg, f_sky={mask.f_sky:.4f}")
    
    # Configure processor
    config = PowerSpectrumConfig(
        nside=NSIDE,
        lmax=LMAX,
        use_master=True,  # Use NaMaster for mode-coupling correction
    )
    processor = PowerSpectrumProcessor(config=config)
    
    # Set mask for processor (needed for MASTER)
    if mask:
        processor.set_mask(mask.data)
    
    # Process baryonified simulations
    if PROCESS_BARYONIFIED:
        print("\n--- Processing baryonified simulations ---")
        files = get_file_list(DATA_DIR, "baryonified")
        print(f"Found {len(files)} files")
        
        all_results = []
        for f in tqdm(files, desc="Processing"):
            result = process_single_file(
                f, processor, mask, NOISE_LEVEL, OUTPUT_DIR
            )
            if result:
                all_results.append(result)
        
        if SAVE_COMBINED and all_results:
            combined = combine_and_save_results(
                all_results, OUTPUT_DIR, "baryonified",
                MASK_AREA_SQDEG if mask else None, NOISE_LEVEL
            )
            
            if AGGREGATE_FOR_INFERENCE:
                aggregate_for_inference(combined, OUTPUT_DIR, "baryonified")
    
    # Process nobaryons (fiducial)
    if PROCESS_FIDUCIAL:
        print("\n--- Processing nobaryons (fiducial) simulations ---")
        files = get_file_list(DATA_DIR, "nobaryons")
        print(f"Found {len(files)} files")
        
        all_results = []
        for f in tqdm(files, desc="Processing"):
            result = process_single_file(
                f, processor, mask, NOISE_LEVEL, OUTPUT_DIR
            )
            if result:
                all_results.append(result)
        
        if SAVE_COMBINED and all_results:
            combined = combine_and_save_results(
                all_results, OUTPUT_DIR, "nobaryons",
                MASK_AREA_SQDEG if mask else None, NOISE_LEVEL
            )
            
            if AGGREGATE_FOR_INFERENCE:
                aggregate_for_inference(combined, OUTPUT_DIR, "nobaryons")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
