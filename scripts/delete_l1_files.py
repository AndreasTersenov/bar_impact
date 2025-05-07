#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/delete_l1_files.py

"""
Script to delete L1 norm files for a specific configuration to allow reprocessing.
"""

import os
import argparse
from tqdm import tqdm


def main():
    """Main function to handle command-line arguments and delete files."""
    parser = argparse.ArgumentParser(description="Delete L1 norm files for a specific configuration.")
    
    # Configuration options (matching the processing script)
    parser.add_argument("--fiducial", action="store_true",
                        help="Delete files for fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", 
                        help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Delete files for baryonified maps instead of nobaryons maps.")
    parser.add_argument("--bin-number", type=int, default=1, 
                        help="Bin number to delete")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Delete files without noise suffix.")
    
    # Combined file options
    parser.add_argument("--combined-only", action="store_true",
                        help="Delete only the combined file, not individual files.")
    
    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                        help="Show files that would be deleted without deleting them.")
    
    args = parser.parse_args()
    
    # Set the base directory based on fiducial flag or override
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    
    # Set the filename based on the baryonified flag
    if args.baryonified:
        filename_pattern = "projected_probes_maps_baryonified512"
    else:
        filename_pattern = "projected_probes_maps_nobaryons512"
    
    # Create the suffix pattern to match
    if args.no_noise:
        suffix = f"_l1_norms_bin{args.bin_number}.npy"
    else:
        suffix = f"_l1_norms_bin{args.bin_number}_noisy_s{args.noise_level:.2f}.npy"
    
    # Find files to delete
    files_to_delete = []
    
    # Handle combined file first
    dataset_name = "fiducial" if args.fiducial else "grid"
    map_suffix = "baryonified" if args.baryonified else "nobaryons"
    combined_filename = f"all_l1_norms_{dataset_name}_{map_suffix}_bin{args.bin_number}"
    if args.no_noise:
        combined_filename += ".npy"
    else:
        combined_filename += f"_noisy_s{args.noise_level:.2f}.npy"
    
    combined_path = os.path.join(base_dir, combined_filename)
    if os.path.exists(combined_path):
        files_to_delete.append(combined_path)
    
    # If we're only deleting the combined file, skip individual file search
    if not args.combined_only:
        # Set permutation directories based on fiducial flag
        if args.fiducial:
            perm_dirs = [f"perm_{i:04d}" for i in range(200)]  # "perm_0000" to "perm_0199"
            
            # Find all matching files for fiducial cosmology
            for perm in perm_dirs:
                perm_path = os.path.join(base_dir, perm)
                if not os.path.exists(perm_path):
                    continue
                    
                # Check for L1 norm file
                l1_file = os.path.join(perm_path, f"{filename_pattern}{suffix}")
                if os.path.exists(l1_file):
                    files_to_delete.append(l1_file)
        else:
            # Find all cosmology directories
            cosmo_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("cosmo_")])
            perm_dirs = [f"perm_{i:04d}" for i in range(7)]  # "perm_0000" to "perm_0006"
            
            # Find all matching files for grid cosmologies
            for cosmo in cosmo_dirs:
                cosmo_path = os.path.join(base_dir, cosmo)
                
                for perm in perm_dirs:
                    perm_path = os.path.join(cosmo_path, perm)
                    if not os.path.exists(perm_path):
                        continue
                        
                    # Check for L1 norm file
                    l1_file = os.path.join(perm_path, f"{filename_pattern}{suffix}")
                    if os.path.exists(l1_file):
                        files_to_delete.append(l1_file)
    
    # Report and confirm deletion
    if not files_to_delete:
        print(f"No files found matching the specified configuration.")
        return
    
    print(f"Found {len(files_to_delete)} files to delete:")
    
    # Show sample of files (first 5 and last 5 if there are many)
    if len(files_to_delete) <= 10:
        for file_path in files_to_delete:
            print(f"  {file_path}")
    else:
        for file_path in files_to_delete[:5]:
            print(f"  {file_path}")
        print(f"  ... and {len(files_to_delete) - 10} more files ...")
        for file_path in files_to_delete[-5:]:
            print(f"  {file_path}")
    
    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in files_to_delete) / (1024 * 1024)  # Convert to MB
    print(f"Total size: {total_size:.2f} MB")
    
    # Dry run mode - stop here
    if args.dry_run:
        print("DRY RUN: No files were deleted.")
        return
    
    # Confirm deletion
    confirm = input(f"Delete these {len(files_to_delete)} files? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Deletion cancelled.")
        return
    
    # Delete files with progress bar
    deleted_count = 0
    for file_path in tqdm(files_to_delete, desc="Deleting files"):
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Deletion complete: {deleted_count}/{len(files_to_delete)} files deleted.")


if __name__ == "__main__":
    main()