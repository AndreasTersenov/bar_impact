#!/usr/bin/env python3
"""
Quick test to verify the cross_power_spectrum_processing.py modifications work correctly.
"""

import os
import numpy as np
import tempfile
import shutil

def create_test_npz_files(test_dir, n_files=5, bin_range=[1, 2, 3, 4], lmax=1024):
    """Create mock .npz files for testing."""
    print(f"Creating {n_files} test .npz files in {test_dir}...")
    
    file_paths = []
    for i in range(n_files):
        # Create mock data
        data_dict = {}
        
        # Auto spectra
        for bin_num in bin_range:
            data_dict[f'cls_{bin_num}_{bin_num}'] = np.random.randn(lmax + 1)
        
        # Cross spectra
        from itertools import combinations
        for b1, b2 in combinations(bin_range, 2):
            data_dict[f'cls_{b1}_{b2}'] = np.random.randn(lmax + 1)
        
        # Metadata
        data_dict['bin_range'] = np.array(bin_range)
        data_dict['lmax'] = lmax
        
        # Save file
        file_path = os.path.join(test_dir, f'test_file_{i:03d}_all_cls_bins1234.npz')
        np.savez_compressed(file_path, **data_dict)
        file_paths.append(file_path)
    
    print(f"  ✓ Created {len(file_paths)} test files")
    return file_paths


def test_aggregation():
    """Test the aggregation function."""
    print("\n" + "="*60)
    print("Testing aggregation for inference")
    print("="*60)
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="test_cross_ps_")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create test files
        bin_range = [1, 2, 3, 4]
        n_files = 5
        lmax = 1024
        
        test_files = create_test_npz_files(test_dir, n_files=n_files, bin_range=bin_range, lmax=lmax)
        
        # Import the aggregation function
        import sys
        sys.path.insert(0, '/home/tersenov/software/bar_impact/scripts')
        from cross_power_spectrum_processing import aggregate_for_inference
        
        # Run aggregation
        print("\nRunning aggregation...")
        created_files = aggregate_for_inference(
            processed_files=test_files,
            output_dir=output_dir,
            bin_range=bin_range,
            dataset_type="grid",
            map_type="nobaryons",
            noise_level=0.26,
            add_noise=False,
            verbose=True
        )
        
        # Verify output files
        print("\n" + "="*60)
        print("Verification")
        print("="*60)
        
        success = True
        
        # Check auto files
        for bin_num in bin_range:
            expected_file = os.path.join(output_dir, f"all_cls_grid_nobaryons_bin{bin_num}.npy")
            if os.path.exists(expected_file):
                data = np.load(expected_file)
                expected_shape = (n_files, lmax + 1)
                if data.shape == expected_shape:
                    print(f"  ✓ Bin {bin_num}: {os.path.basename(expected_file)} - shape {data.shape} ✓")
                else:
                    print(f"  ✗ Bin {bin_num}: Wrong shape {data.shape}, expected {expected_shape}")
                    success = False
            else:
                print(f"  ✗ Bin {bin_num}: File not found")
                success = False
        
        # Check cross file
        bin_str = "".join(map(str, bin_range))
        expected_cross_file = os.path.join(output_dir, f"all_cross_cls_grid_nobaryons_bins{bin_str}.npy")
        if os.path.exists(expected_cross_file):
            data = np.load(expected_cross_file)
            # 6 cross pairs for 4 bins: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
            n_cross_pairs = 6
            expected_shape = (n_files, n_cross_pairs * (lmax + 1))
            if data.shape == expected_shape:
                print(f"  ✓ Cross: {os.path.basename(expected_cross_file)} - shape {data.shape} ✓")
            else:
                print(f"  ✗ Cross: Wrong shape {data.shape}, expected {expected_shape}")
                success = False
        else:
            print(f"  ✗ Cross: File not found")
            success = False
        
        # Final result
        print("\n" + "="*60)
        if success:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed")
        print("="*60)
        
        return success
        
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = test_aggregation()
    exit(0 if success else 1)
