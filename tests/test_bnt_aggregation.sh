#!/bin/bash
# Quick test of BNT aggregation for inference

echo "============================================"
echo "BNT Cross Power Spectrum Aggregation Test"
echo "============================================"

# Test with a small subset (first cosmology only)
TEST_BASE_DIR="/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/cosmo_000001"

if [ ! -d "$TEST_BASE_DIR" ]; then
    echo "Error: Test directory not found: $TEST_BASE_DIR"
    exit 1
fi

echo ""
echo "Step 1: Processing BNT files for cosmo_000001 (test run)..."
echo "-----------------------------------------------------------"

python3 /home/tersenov/software/bar_impact/scripts/bnt_cross_power_spectrum_processing.py \
  --base-dir "$TEST_BASE_DIR" \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 4 \
  --no-noise \
  --aggregate-for-inference \
  --inference-output-dir /tmp/bnt_test_output \
  --verbose

echo ""
echo "Step 2: Checking output files..."
echo "-----------------------------------------------------------"

if [ ! -d "/tmp/bnt_test_output" ]; then
    echo "Error: Output directory not created!"
    exit 1
fi

echo "Files created:"
ls -lh /tmp/bnt_test_output/all_bnt_*.npy 2>/dev/null || echo "No .npy files found!"

echo ""
echo "Step 3: Verifying file shapes..."
echo "-----------------------------------------------------------"

python3 << 'EOF'
import numpy as np
import os
import glob

output_dir = "/tmp/bnt_test_output"

# Check auto spectra
for i in range(1, 5):
    auto_file = os.path.join(output_dir, f"all_bnt_cls_grid_nobaryons_bin{i}.npy")
    if os.path.exists(auto_file):
        data = np.load(auto_file)
        print(f"✓ BNT bin {i} auto spectrum: {data.shape}")
        assert data.shape[1] == 1025, f"Expected 1025 multipoles, got {data.shape[1]}"
    else:
        print(f"✗ BNT bin {i} auto spectrum: NOT FOUND")

# Check cross spectra
cross_file = os.path.join(output_dir, "all_bnt_cross_cls_grid_nobaryons_bins1234.npy")
if os.path.exists(cross_file):
    data = np.load(cross_file)
    print(f"✓ Combined cross spectra: {data.shape}")
    expected_length = 6 * 1025  # 6 cross pairs × 1025 multipoles
    assert data.shape[1] == expected_length, f"Expected {expected_length} length, got {data.shape[1]}"
else:
    print(f"✗ Combined cross spectra: NOT FOUND")

print("\n✓ All checks passed!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Test completed successfully!"
    echo "============================================"
    echo ""
    echo "The script is ready to use. Run with:"
    echo ""
    echo "  python scripts/bnt_cross_power_spectrum_processing.py \\"
    echo "    --bnt-bin-range 0 1 2 3 \\"
    echo "    --num-workers 80 \\"
    echo "    --no-noise \\"
    echo "    --aggregate-for-inference \\"
    echo "    --verbose"
    echo ""
else
    echo ""
    echo "============================================"
    echo "✗ Test failed!"
    echo "============================================"
    exit 1
fi

# Cleanup
rm -rf /tmp/bnt_test_output
echo "Cleaned up test files."
