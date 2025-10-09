#!/usr/bin/env python3
"""
BNT Cross Power Spectrum Processing Script - Processes cosmological data files to compute cross power spectra between different BNT-transformed redshift bins.
"""

import os
import h5py
import healpy as hp
import numpy as np
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import combinations


# BNT transformation matrix
BNT_MATRIX = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [-1.        ,  1.        ,  0.        ,  0.        ],
                       [ 0.4521097 , -1.4521097 ,  1.        ,  0.        ],
                       [ 0.        ,  0.25127807, -1.251278  ,  1.        ]])


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """
    Adds shape noise to a full-sky Healpix convergence (kappa) map.
    """
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise


def get_cross_power_spectra(maps_dict, lmax=1024):
    """
    Computes auto and cross power spectra for multiple redshift bins.
    
    Parameters:
    - maps_dict: dict, dictionary with bin numbers as keys and HEALPix maps as values
    - lmax: int, the maximum multipole to compute
    
    Returns:
    - cls_dict: dict, dictionary with (bin_i, bin_j) tuples as keys and Cls as values
    """
    # Convert all maps to alms first
    alms_dict = {}
    for bin_num, map_data in maps_dict.items():
        alms_dict[bin_num] = hp.map2alm(map_data, lmax=lmax)
    
    # Compute all auto and cross power spectra
    cls_dict = {}
    bin_numbers = sorted(maps_dict.keys())
    
    # Auto power spectra (diagonal elements)
    for bin_num in bin_numbers:
        cls_dict[(bin_num, bin_num)] = hp.alm2cl(alms_dict[bin_num])
    
    # Cross power spectra (off-diagonal elements)
    for bin_i, bin_j in combinations(bin_numbers, 2):
        cls_dict[(bin_i, bin_j)] = hp.alm2cl(alms_dict[bin_i], alms_dict[bin_j])
    
    return cls_dict


def process_file(file_path, bnt_bin_range=[0, 1, 2, 3], noise_level=0.26, add_noise=True, 
                 lmax=1024, cross_only=False, verbose=False):
    """
    Process a single file: extract kappa maps for all redshift bins, apply BNT transform, 
    compute cross power spectra for the specified BNT bins, and save results.
    """
    
    # Define output filename
    bnt_bin_str = "".join([str(b+1) for b in bnt_bin_range])  # Convert 0-based to 1-based for filename
    # include lmax in per-file suffix when different from default
    lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""
    if add_noise:
        if cross_only:
            suffix = f"_bnt_cross_cls_bins{bnt_bin_str}_noisy_s{noise_level:.2f}{lmax_suffix}.npz"
        else:
            suffix = f"_bnt_all_cls_bins{bnt_bin_str}_noisy_s{noise_level:.2f}{lmax_suffix}.npz"
    else:
        if cross_only:
            suffix = f"_bnt_cross_cls_bins{bnt_bin_str}{lmax_suffix}.npz"
        else:
            suffix = f"_bnt_all_cls_bins{bnt_bin_str}{lmax_suffix}.npz"
    
    save_path = file_path.replace(".h5", suffix)
    
    # Skip if file already exists
    if os.path.exists(save_path):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, BNT cross-Cls file already exists.")
        return save_path
    
    try:
        # Load all 4 kappa maps
        kgs = []
        with h5py.File(file_path, "r") as f:
            for i in range(4):
                map_key = f"kg/stage3_lensing{i+1}"
                kgs.append(np.array(f[map_key]))
        
        # Add shape noise if requested (before BNT transform)
        if add_noise:
            kgs = [add_shape_noise(kg, sigma_e=noise_level) for kg in kgs]
        
        # Apply BNT transform
        kgs = np.array(kgs)  # Convert to numpy array for matrix multiplication
        kgs_bnt = BNT_MATRIX @ kgs
        
        # Create maps dictionary for the specified BNT bins
        maps_dict = {}
        for bnt_bin in bnt_bin_range:
            maps_dict[bnt_bin+1] = kgs_bnt[bnt_bin]  # Use 1-based indexing for consistency
        
        # Compute cross power spectra
        cls_dict = get_cross_power_spectra(maps_dict, lmax=lmax)
        
        # Filter to cross-correlations only if requested
        if cross_only:
            cls_dict = {(i, j): cls for (i, j), cls in cls_dict.items() if i != j}
        
        # Prepare data for saving
        save_dict = {}
        for (i, j), cls in cls_dict.items():
            save_dict[f"cls_{i}_{j}"] = cls
        
        # Add metadata
        save_dict['bnt_bin_range'] = np.array(bnt_bin_range)
        save_dict['bin_range'] = np.array(list(maps_dict.keys()))  # 1-based BNT bin numbers
        save_dict['lmax'] = lmax
        
        # Save results as compressed npz file
        np.savez_compressed(save_path, **save_dict)
        
        if verbose:
            spectra_type = "cross" if cross_only else "auto+cross"
            print(f"Processed: {os.path.basename(file_path)} -> {os.path.basename(save_path)} ({len(cls_dict)} BNT {spectra_type} spectra)")
        
        return save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def aggregate_for_inference(processed_files, output_dir, bnt_bin_range=[0, 1, 2, 3], 
                            dataset_type="grid", map_type="nobaryons", 
                            noise_level=0.26, add_noise=True, lmax=1024, verbose=False):
    """
    Aggregate processed .npz files into the format expected by run_npe_inference_auto_cross_ps.py.
    
    Creates:
    - Separate .npy files for each BNT bin's auto power spectrum
    - One combined .npy file for all cross power spectra
    
    Parameters:
    - processed_files: list of paths to processed .npz files
    - output_dir: directory to save aggregated files
    - bnt_bin_range: list of BNT bin numbers (0-indexed)
    - dataset_type: "grid" or "fiducial"
    - map_type: "nobaryons" or "baryonified"
    - noise_level: noise level for filename
    - add_noise: whether noise was added
    - verbose: print detailed information
    """
    if not processed_files:
        print("No files to aggregate!")
        return
    
    # Filter out None values and check file existence
    valid_files = [f for f in processed_files if f is not None and os.path.exists(f)]
    
    if not valid_files:
        print("No valid processed files found for aggregation!")
        return
    
    print(f"\nAggregating {len(valid_files)} BNT files for inference...")
    
    # Load all files and organize by spectrum type
    auto_spectra = {(b+1): [] for b in bnt_bin_range}  # Use 1-based indexing
    cross_spectra_parts = []
    
    # Determine the order of cross pairs for consistent concatenation
    bnt_bins_1based = [b+1 for b in bnt_bin_range]
    cross_pairs_ordered = list(combinations(sorted(bnt_bins_1based), 2))
    cross_spectra = {pair: [] for pair in cross_pairs_ordered}
    
    for file_path in tqdm(valid_files, desc="Loading files"):
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Extract auto spectra
            for bnt_bin in bnt_bin_range:
                bin_1based = bnt_bin + 1
                key = f"cls_{bin_1based}_{bin_1based}"
                if key in data.files:
                    auto_spectra[bin_1based].append(data[key])
            
            # Extract cross spectra in order
            for pair in cross_pairs_ordered:
                i, j = pair
                key = f"cls_{i}_{j}"
                if key in data.files:
                    cross_spectra[pair].append(data[key])
                    
        except Exception as e:
            if verbose:
                print(f"Error loading {os.path.basename(file_path)}: {e}")
            continue
    
    # Convert lists to arrays
    for bin_1based in auto_spectra:
        if auto_spectra[bin_1based]:
            auto_spectra[bin_1based] = np.array(auto_spectra[bin_1based])
    
    for pair in cross_spectra:
        if cross_spectra[pair]:
            cross_spectra[pair] = np.array(cross_spectra[pair])
    
    # Prepare filename suffixes
    noise_suffix = f"_noisy_s{noise_level:.2f}" if add_noise else ""
    lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""
    
    # Save individual auto power spectra
    print("\nSaving auto power spectra...")
    for bnt_bin in bnt_bin_range:
        bin_1based = bnt_bin + 1
        if auto_spectra[bin_1based].size > 0:
            # Filename: all_bnt_cls_<dataset>_<maptype>_bin<N>.npy
            auto_filename = f"all_bnt_cls_{dataset_type}_{map_type}_bin{bin_1based}{noise_suffix}{lmax_suffix}.npy"
            auto_path = os.path.join(output_dir, auto_filename)
            np.save(auto_path, auto_spectra[bin_1based])
            
            if verbose:
                print(f"  Saved {auto_filename}, shape: {auto_spectra[bin_1based].shape}")
    
    # Save combined cross power spectra
    print("\nSaving combined cross power spectra...")
    cross_data_parts = []
    for pair in cross_pairs_ordered:
        if cross_spectra[pair].size > 0:
            cross_data_parts.append(cross_spectra[pair])
    
    if cross_data_parts:
        # Concatenate all cross spectra along the multipole dimension (axis=1)
        # Result shape: (n_files, total_cross_spectrum_length)
        cross_data_combined = np.concatenate(cross_data_parts, axis=1)
        # Filename: all_bnt_cross_cls_<dataset>_<maptype>_bins<1234>.npy
        bnt_bin_str = "".join([str(b+1) for b in bnt_bin_range])
        cross_filename = f"all_bnt_cross_cls_{dataset_type}_{map_type}_bins{bnt_bin_str}{noise_suffix}{lmax_suffix}.npy"
        cross_path = os.path.join(output_dir, cross_filename)
        np.save(cross_path, cross_data_combined)

        if verbose:
            print(f"  Saved {cross_filename}, shape: {cross_data_combined.shape}")
    
    print(f"\n✓ Aggregation complete! Files saved to: {output_dir}")
    print(f"  Auto spectra: {len([b for b in bnt_bin_range if auto_spectra[b+1].size > 0])} bins")
    print(f"  Cross spectra: {len(cross_data_parts)} pairs combined")
    
    # Print expected filenames for inference
    print(f"\nFiles ready for run_npe_inference_auto_cross_ps.py:")
    for bnt_bin in bnt_bin_range:
        bin_1based = bnt_bin + 1
        if auto_spectra[bin_1based].size > 0:
            print(f"  - all_bnt_cls_{dataset_type}_{map_type}_bin{bin_1based}{noise_suffix}{lmax_suffix}.npy")
    if cross_data_parts:
        print(f"  - all_bnt_cross_cls_{dataset_type}_{map_type}_bins{bnt_bin_str}{noise_suffix}{lmax_suffix}.npy")


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(description="Process HEALPix maps with BNT transform to compute cross power spectra between BNT bins.")
    
    # Main processing options
    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", 
                        help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")
    parser.add_argument("--bnt-bin-range", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="List of BNT bins to include in cross-correlation analysis (0-indexed, default: 0 1 2 3).")
    parser.add_argument("--cross-only", action="store_true",
                        help="Only compute cross power spectra, exclude auto power spectra.")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Algorithm parameters
    parser.add_argument("--lmax", type=int, default=1024, 
                        help="Maximum multipole (l) for power spectrum calculation.")
    
    # Execution options
    parser.add_argument("--num-workers", type=int, default=70,
                        help="Number of worker processes.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    # Output options
    parser.add_argument("--save-combined", action="store_true",
                        help="Create a summary file listing all processed files.")
    parser.add_argument("--combined-output", 
                        help="Path for combined summary file.")
    
    # Aggregation for inference
    parser.add_argument("--aggregate-for-inference", action="store_true",
                        help="Aggregate processed files into format for run_npe_inference_auto_cross_ps.py")
    parser.add_argument("--inference-output-dir", type=str,
                        help="Output directory for aggregated inference files. If not specified, uses base_dir/new_grid or base_dir.")
    
    args = parser.parse_args()
    
    # Validate BNT bin range
    if len(args.bnt_bin_range) < 2 and args.cross_only:
        print("Error: Need at least 2 BNT bins for cross-correlation analysis.")
        return
    
    # Check that BNT bins are in valid range [0, 3]
    for bnt_bin in args.bnt_bin_range:
        if bnt_bin < 0 or bnt_bin > 3:
            print(f"Error: BNT bin {bnt_bin} is out of range [0, 3].")
            return
    
    # Set the base directory based on fiducial flag or override
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    
    # Set the filename based on the baryonified flag
    if args.baryonified:
        filename = "projected_probes_maps_baryonified512.h5"
    else:
        filename = "projected_probes_maps_nobaryons512.h5"
    
    # Set permutation directories based on fiducial flag
    if args.fiducial:
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
    else:
        cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
        perm_dirs = [f"perm_{i:04d}" for i in range(7)]
        file_paths = [
            os.path.join(base_dir, cosmo, perm, filename)
            for cosmo in cosmo_dirs
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
        ]
    
    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Computing BNT cross power spectra for BNT bins: {args.bnt_bin_range} (0-indexed)")
    
    if args.cross_only:
        cross_pairs = list(combinations(args.bnt_bin_range, 2))
        print(f"BNT cross-correlation pairs (0-indexed): {cross_pairs}")
    else:
        auto_pairs = [(b, b) for b in args.bnt_bin_range]
        cross_pairs = list(combinations(args.bnt_bin_range, 2))
        print(f"BNT auto power spectra (0-indexed): {auto_pairs}")
        print(f"BNT cross power spectra (0-indexed): {cross_pairs}")
    
    # Determine suffix for output files
    bnt_bin_str = "".join([str(b+1) for b in args.bnt_bin_range])  # Convert to 1-based for filename
    if args.no_noise:
        if args.cross_only:
            suffix = f"_bnt_cross_cls_bins{bnt_bin_str}.npz"
        else:
            suffix = f"_bnt_all_cls_bins{bnt_bin_str}.npz"
    else:
        if args.cross_only:
            suffix = f"_bnt_cross_cls_bins{bnt_bin_str}_noisy_s{args.noise_level:.2f}.npz"
        else:
            suffix = f"_bnt_all_cls_bins{bnt_bin_str}_noisy_s{args.noise_level:.2f}.npz"
    
    print(f"Output suffix: {suffix}")
    
    # Process files in parallel with progress bar
    with mp.Pool(processes=args.num_workers) as pool:
        process_func = partial(
            process_file,
            bnt_bin_range=args.bnt_bin_range,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            cross_only=args.cross_only,
            verbose=args.verbose
        )
        results = list(tqdm(
            pool.imap(process_func, file_paths),
            total=len(file_paths),
            desc="Processing files"
        ))
    
    # Count successful files
    successful = [r for r in results if r is not None]
    processed = len([r for r in successful if os.path.exists(r)])
    print(f"Processing complete: {processed}/{len(file_paths)} files processed")
    
    # Optionally save combined summary
    if args.save_combined and successful:
        combined_output = args.combined_output
        if not combined_output:
            dataset_name = "fiducial" if args.fiducial else "grid"
            map_suffix = "baryonified" if args.baryonified else "nobaryons"
            
            bnt_bin_str = "".join([str(b+1) for b in args.bnt_bin_range])
            spectra_type = "cross" if args.cross_only else "all"
            
            if args.no_noise:
                combined_output = os.path.join(base_dir, f"bnt_{spectra_type}_cls_summary_{dataset_name}_{map_suffix}_bins{bnt_bin_str}.txt")
            else:
                combined_output = os.path.join(base_dir, f"bnt_{spectra_type}_cls_summary_{dataset_name}_{map_suffix}_bins{bnt_bin_str}_noisy_s{args.noise_level:.2f}.txt")
        
        print(f"Creating summary file: {os.path.basename(combined_output)}")
        
        with open(combined_output, 'w') as f:
            f.write("# BNT Cross Power Spectrum Processing Summary\n")
            f.write(f"# Dataset: {dataset_type} {map_type}\n")
            f.write(f"# BNT Bins (0-indexed): {args.bnt_bin_range}\n")
            f.write(f"# Cross-only: {args.cross_only}\n")
            f.write(f"# Noise level: {args.noise_level if not args.no_noise else 'None'}\n")
            f.write(f"# Lmax: {args.lmax}\n")
            f.write(f"# Total files processed: {processed}/{len(file_paths)}\n")
            f.write("# Processed files:\n")
            
            for file_path in successful:
                if os.path.exists(file_path):
                    f.write(f"{file_path}\n")
        
        print(f"Summary saved to: {os.path.basename(combined_output)}")
        
        # Also save an example of how to load the data
        example_file = combined_output.replace('.txt', '_example_load.py')
        with open(example_file, 'w') as f:
            f.write('''# Example of how to load and use the BNT cross power spectra files
import numpy as np
import glob

# Load a single file
filename = "your_file_bnt_cross_cls_bins1234_noisy_s0.26.npz"
data = np.load(filename)

# Print available power spectra
print("Available BNT power spectra:")
for key in data.files:
    if key.startswith('cls_'):
        print(f"  {key}: {data[key].shape}")

# Access specific BNT cross power spectra
''')
            if args.cross_only:
                cross_pairs = list(combinations([b+1 for b in args.bnt_bin_range], 2))  # Convert to 1-based
                for i, j in cross_pairs:
                    f.write(f"cross_{i}_{j} = data['cls_{i}_{j}']  # Cross power spectrum between BNT bins {i} and {j}\n")
            else:
                f.write("# BNT auto power spectra:\n")
                for b in args.bnt_bin_range:
                    f.write(f"auto_{b+1}_{b+1} = data['cls_{b+1}_{b+1}']  # Auto power spectrum of BNT bin {b+1}\n")
                f.write("\n# BNT cross power spectra:\n")
                cross_pairs = list(combinations([b+1 for b in args.bnt_bin_range], 2))  # Convert to 1-based
                for i, j in cross_pairs:
                    f.write(f"cross_{i}_{j} = data['cls_{i}_{j}']  # Cross power spectrum between BNT bins {i} and {j}\n")
            
            f.write('''
# Load metadata
bnt_bin_range = data['bnt_bin_range']  # 0-indexed BNT bin range
bin_range = data['bin_range']  # 1-indexed bin range used in file
lmax = data['lmax']

# Load multiple files
pattern = "path/to/files/*bnt_cross_cls_bins1234*.npz"
files = glob.glob(pattern)
all_data = []
for file in files:
    data = np.load(file)
    all_data.append(data)
''')
        
        print(f"Example loading script saved to: {os.path.basename(example_file)}")
    
    # Aggregate files for inference if requested
    if args.aggregate_for_inference:
        # Determine output directory
        if args.inference_output_dir:
            inference_output_dir = args.inference_output_dir
        else:
            if args.fiducial:
                inference_output_dir = os.path.join(base_dir, "cosmo_fiducial")
            else:
                inference_output_dir = os.path.join(os.path.dirname(base_dir.rstrip('/')), "new_grid")
        
        # Create output directory if it doesn't exist
        os.makedirs(inference_output_dir, exist_ok=True)
        
        # Determine dataset type
        dataset_type = "fiducial" if args.fiducial else "grid"
        map_type = "baryonified" if args.baryonified else "nobaryons"
        
        # Call aggregation function
        aggregate_for_inference(
            processed_files=successful,
            output_dir=inference_output_dir,
            bnt_bin_range=args.bnt_bin_range,
            dataset_type=dataset_type,
            map_type=map_type,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            lmax=args.lmax,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
