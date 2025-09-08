#!/usr/bin/env python3
"""
BNT Cross Power Spectrum Aggregation Script - Aggregates BNT cross power spectra .npz files into combined arrays.
"""

import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from itertools import combinations


def find_bnt_cross_power_files(base_dir, pattern="*bnt_cross_cls_bins*.npz"):
    """
    Find all BNT cross power spectrum files in a directory tree.
    
    Parameters:
    - base_dir: str, base directory to search
    - pattern: str, glob pattern to match files
    
    Returns:
    - list of file paths
    """
    search_pattern = os.path.join(base_dir, "**", pattern)
    files = glob.glob(search_pattern, recursive=True)
    return sorted(files)


def load_and_validate_bnt_file(file_path, expected_keys=None, verbose=False):
    """
    Load and validate a BNT cross power spectrum file.
    
    Parameters:
    - file_path: str, path to the .npz file
    - expected_keys: list, expected BNT cross power spectrum keys
    - verbose: bool, print detailed information
    
    Returns:
    - dict with cross power spectra or None if invalid
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Extract cross power spectra
        cls_data = {}
        metadata = {}
        
        for key in data.files:
            if key.startswith('cls_'):
                cls_data[key] = data[key]
            else:
                metadata[key] = data[key]
        
        if not cls_data:
            if verbose:
                print(f"No BNT cross power spectra found in {os.path.basename(file_path)}")
            return None
        
        # Validate expected keys if provided
        if expected_keys:
            missing_keys = set(expected_keys) - set(cls_data.keys())
            if missing_keys:
                if verbose:
                    print(f"Missing keys {missing_keys} in {os.path.basename(file_path)}")
                return None
        
        # Validate that all spectra have the same length
        lengths = [len(cls) for cls in cls_data.values()]
        if len(set(lengths)) > 1:
            if verbose:
                print(f"Inconsistent spectrum lengths in {os.path.basename(file_path)}: {lengths}")
            return None
        
        return {'cls_data': cls_data, 'metadata': metadata, 'file_path': file_path}
        
    except Exception as e:
        if verbose:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
        return None


def aggregate_bnt_cross_power_spectra(file_paths, bnt_bin_range=[1, 2, 3, 4], cross_only=True, verbose=False):
    """
    Aggregate BNT cross power spectra from multiple files.
    
    Parameters:
    - file_paths: list, paths to .npz files
    - bnt_bin_range: list, expected BNT bin numbers (1-based for file compatibility)
    - cross_only: bool, whether files contain only cross spectra
    - verbose: bool, print detailed information
    
    Returns:
    - dict with aggregated spectra and metadata
    """
    # Determine expected keys (BNT bins use 1-based indexing in files)
    if cross_only:
        expected_keys = [f"cls_{i}_{j}" for i, j in combinations(bnt_bin_range, 2)]
    else:
        # Include auto spectra
        auto_keys = [f"cls_{i}_{i}" for i in bnt_bin_range]
        cross_keys = [f"cls_{i}_{j}" for i, j in combinations(bnt_bin_range, 2)]
        expected_keys = auto_keys + cross_keys
    
    if verbose:
        print(f"Expected BNT cross power spectrum keys: {expected_keys}")
    
    # Load and validate all files
    valid_data = []
    skipped_files = 0
    
    for file_path in tqdm(file_paths, desc="Loading BNT files"):
        data = load_and_validate_bnt_file(file_path, expected_keys, verbose)
        if data is not None:
            valid_data.append(data)
        else:
            skipped_files += 1
    
    if not valid_data:
        print("No valid BNT files found!")
        return None
    
    print(f"Loaded {len(valid_data)} valid BNT files, skipped {skipped_files} files")
    
    # Aggregate the data
    aggregated = {}
    n_files = len(valid_data)
    
    # Get the spectrum length from the first valid file
    first_cls = list(valid_data[0]['cls_data'].values())[0]
    spectrum_length = len(first_cls)
    
    # Initialize arrays for each BNT cross power spectrum
    for key in expected_keys:
        aggregated[key] = np.zeros((n_files, spectrum_length))
    
    # Fill the arrays
    for i, data in enumerate(valid_data):
        for key in expected_keys:
            if key in data['cls_data']:
                aggregated[key][i] = data['cls_data'][key]
            else:
                # This should not happen if validation worked correctly
                print(f"Warning: Missing key {key} in file {i}")
                aggregated[key][i] = np.nan
    
    # Create concatenated data vector for compatibility with inference pipeline
    # Order: all auto spectra first, then cross spectra in lexicographic order
    data_vector_parts = []
    ordered_keys = []
    
    if not cross_only:
        # Add auto spectra first
        auto_keys = [f"cls_{i}_{i}" for i in sorted(bnt_bin_range)]
        for key in auto_keys:
            if key in aggregated:
                data_vector_parts.append(aggregated[key])
                ordered_keys.append(key)
    
    # Add cross spectra
    cross_keys = [f"cls_{i}_{j}" for i, j in combinations(sorted(bnt_bin_range), 2)]
    for key in cross_keys:
        if key in aggregated:
            data_vector_parts.append(aggregated[key])
            ordered_keys.append(key)
    
    # Concatenate along the spectrum dimension to create data vector
    # Shape: (n_files, total_spectrum_length)
    concatenated_data = np.concatenate(data_vector_parts, axis=1)
    
    # Add metadata
    metadata = {
        'n_files': n_files,
        'spectrum_length': spectrum_length,
        'total_spectrum_length': concatenated_data.shape[1],
        'bnt_bin_range': np.array(bnt_bin_range),
        'cross_only': cross_only,
        'expected_keys': expected_keys,
        'ordered_keys': ordered_keys,
        'file_paths': [data['file_path'] for data in valid_data]
    }
    
    # Extract common metadata from first file
    if valid_data[0]['metadata']:
        first_meta = valid_data[0]['metadata']
        for key in ['lmax', 'bnt_bin_range']:
            if key in first_meta:
                metadata[key] = first_meta[key]
    
    return {
        'aggregated': aggregated, 
        'concatenated_data': concatenated_data,
        'metadata': metadata
    }


def main():
    """Main function to handle command-line arguments and run BNT aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate BNT cross power spectra .npz files into combined arrays.")
    
    # Input options
    parser.add_argument("--base-dir", required=True,
                        help="Base directory to search for BNT cross power spectrum files.")
    parser.add_argument("--pattern", default="*bnt_cross_cls_bins*.npz",
                        help="Glob pattern to match BNT files (default: *bnt_cross_cls_bins*.npz).")
    parser.add_argument("--bnt-bin-range", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Expected BNT bin range (1-based, default: 1 2 3 4).")
    parser.add_argument("--include-auto", action="store_true",
                        help="Include auto power spectra (for files with both auto and cross).")
    
    # Output options
    parser.add_argument("--output", 
                        help="Output file path. If not specified, will be auto-generated.")
    parser.add_argument("--output-format", choices=["npy", "npz", "hdf5"], default="npy",
                        help="Output format (default: npy for inference compatibility).")
    parser.add_argument("--inference-compatible", action="store_true", default=True,
                        help="Generate files compatible with run_npe_inference_ps.py (default: True).")
    
    # Processing options
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    
    # Add argument parsing for dataset inference
    parser.add_argument("--fiducial", action="store_true",
                        help="Input files are from fiducial dataset (helps with filename generation).")
    parser.add_argument("--baryonified", action="store_true", 
                        help="Input files are from baryonified simulations (helps with filename generation).")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist.")
        return
    
    # Find files
    print(f"Searching for BNT files in {args.base_dir} with pattern {args.pattern}")
    file_paths = find_bnt_cross_power_files(args.base_dir, args.pattern)
    
    if not file_paths:
        print("No matching BNT files found!")
        return
    
    print(f"Found {len(file_paths)} matching BNT files")
    
    # Aggregate data
    cross_only = not args.include_auto
    result = aggregate_bnt_cross_power_spectra(
        file_paths, 
        bnt_bin_range=args.bnt_bin_range, 
        cross_only=cross_only, 
        verbose=args.verbose
    )
    
    if result is None:
        print("BNT aggregation failed!")
        return
    
    aggregated = result['aggregated']
    concatenated_data = result['concatenated_data']
    metadata = result['metadata']
    
    # Print summary
    print(f"\nBNT Aggregation Summary:")
    print(f"  Files processed: {metadata['n_files']}")
    print(f"  Individual spectrum length: {metadata['spectrum_length']}")
    print(f"  Concatenated spectrum length: {metadata['total_spectrum_length']}")
    print(f"  BNT bin range: {metadata['bnt_bin_range']}")
    print(f"  Cross spectra only: {metadata['cross_only']}")
    print(f"  Available BNT spectra:")
    for key in metadata['expected_keys']:
        shape = aggregated[key].shape
        print(f"    {key}: {shape}")
    print(f"  Concatenated data shape: {concatenated_data.shape}")
    
    # Generate output filename if not provided
    if not args.output:
        bnt_bin_str = "".join(map(str, args.bnt_bin_range))
        
        # Try to infer dataset and map info from pattern or first file path
        sample_path = file_paths[0] if file_paths else ""
        
        if args.fiducial or "fiducial" in sample_path:
            dataset_info = "fiducial"
        elif "grid" in sample_path or "new_grid" in sample_path:
            dataset_info = "grid"
        else:
            dataset_info = "unknown"
        
        if args.baryonified or "baryonified" in sample_path:
            map_info = "baryonified"
        elif "nobaryons" in sample_path:
            map_info = "nobaryons"
        else:
            map_info = "unknown"
        
        # Infer noise info
        if "noisy" in sample_path:
            import re
            noise_match = re.search(r'noisy_s([\d.]+)', sample_path)
            if noise_match:
                noise_suffix = f"_noisy_s{noise_match.group(1)}"
            else:
                noise_suffix = "_noisy"
        else:
            noise_suffix = ""
        
        if args.inference_compatible and args.output_format == "npy":
            # Generate filename compatible with run_npe_inference_ps.py with BNT prefix
            if metadata['cross_only']:
                # For cross-only, use a special identifier
                filename = f"all_bnt_cross_cls_{dataset_info}_{map_info}_bins{bnt_bin_str}{noise_suffix}"
            else:
                # For auto+cross, use standard naming with BNT prefix
                filename = f"all_bnt_cls_{dataset_info}_{map_info}_bins{bnt_bin_str}{noise_suffix}"
        else:
            # Use descriptive naming for other formats
            spectra_type = "cross" if metadata['cross_only'] else "all"
            filename = f"aggregated_bnt_{spectra_type}_cls_{dataset_info}_{map_info}_bins{bnt_bin_str}{noise_suffix}"
        
        args.output = os.path.join(args.base_dir, f"{filename}{args.output_format}")
    
    # Save results
    print(f"\nSaving aggregated BNT data to: {args.output}")
    
    if args.output_format == "npy":
        # Save concatenated data as .npy (compatible with inference pipeline)
        np.save(args.output, concatenated_data)
        
        # Also save metadata and individual spectra as separate files if requested
        if not args.inference_compatible:
            metadata_file = args.output.replace('.npy', '_metadata.npz')
            save_dict = {}
            for key, data in aggregated.items():
                save_dict[key] = data
            for key, value in metadata.items():
                if key != 'file_paths':
                    save_dict[f"meta_{key}"] = value
            np.savez_compressed(metadata_file, **save_dict)
            print(f"Saved detailed BNT data to: {metadata_file}")
        
    elif args.output_format == "npz":
        # Save as npz
        save_dict = {}
        
        # Add aggregated spectra
        for key, data in aggregated.items():
            save_dict[key] = data
        
        # Add metadata
        for key, value in metadata.items():
            if key != 'file_paths':  # Skip file paths to avoid issues
                save_dict[f"meta_{key}"] = value
        
        np.savez_compressed(args.output, **save_dict)
        
    else:
        # Save as HDF5
        import h5py
        
        with h5py.File(args.output, 'w') as f:
            # Create groups
            spectra_group = f.create_group('bnt_cross_power_spectra')
            meta_group = f.create_group('metadata')
            
            # Save spectra
            for key, data in aggregated.items():
                spectra_group.create_dataset(key, data=data, compression='gzip')
            
            # Save metadata
            for key, value in metadata.items():
                if key == 'file_paths':
                    # Convert to byte strings for HDF5
                    str_dtype = h5py.special_dtype(vlen=str)
                    meta_group.create_dataset(key, data=value, dtype=str_dtype)
                elif isinstance(value, (list, np.ndarray)):
                    meta_group.create_dataset(key, data=value)
                else:
                    meta_group.attrs[key] = value
    
    print("BNT aggregation complete!")


if __name__ == "__main__":
    main()
