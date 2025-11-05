# BNT Cross Power Spectrum Inference Workflow

This guide explains how to use the modified `bnt_cross_power_spectrum_processing.py` script to process BNT-transformed power spectra and prepare them for inference with `run_npe_inference_auto_cross_ps.py`.

## New Feature: Direct Aggregation for Inference

The script now includes an `--aggregate-for-inference` flag that automatically:
1. Processes all HEALPix map files to compute BNT cross power spectra
2. Aggregates the results into the exact format expected by the inference script
3. Creates separate `.npy` files for each bin's auto spectrum and one combined file for all cross spectra

## Quick Start

### For Grid Data (No Noise)

```bash
python scripts/bnt_cross_power_spectrum_processing.py \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

**Output files** (saved to `/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/`):
- `all_bnt_cls_grid_nobaryons_bin1.npy` - Auto spectrum for BNT bin 1
- `all_bnt_cls_grid_nobaryons_bin2.npy` - Auto spectrum for BNT bin 2
- `all_bnt_cls_grid_nobaryons_bin3.npy` - Auto spectrum for BNT bin 3
- `all_bnt_cls_grid_nobaryons_bin4.npy` - Auto spectrum for BNT bin 4
- `all_bnt_cross_cls_grid_nobaryons_bins1234.npy` - Combined cross spectra

### For Grid Data (With Noise)

```bash
python scripts/bnt_cross_power_spectrum_processing.py \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --noise-level 0.26 \
  --aggregate-for-inference \
  --verbose
```

**Output files** (saved to `/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/`):
- `all_bnt_cls_grid_nobaryons_bin1_noisy_s0.26.npy`
- `all_bnt_cls_grid_nobaryons_bin2_noisy_s0.26.npy`
- `all_bnt_cls_grid_nobaryons_bin3_noisy_s0.26.npy`
- `all_bnt_cls_grid_nobaryons_bin4_noisy_s0.26.npy`
- `all_bnt_cross_cls_grid_nobaryons_bins1234_noisy_s0.26.npy`

### For Fiducial Data (No Noise)

```bash
python scripts/bnt_cross_power_spectrum_processing.py \
  --fiducial \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

**Output files** (saved to `/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/`):
- `all_bnt_cls_fiducial_nobaryons_bin1.npy`
- `all_bnt_cls_fiducial_nobaryons_bin2.npy`
- `all_bnt_cls_fiducial_nobaryons_bin3.npy`
- `all_bnt_cls_fiducial_nobaryons_bin4.npy`
- `all_bnt_cross_cls_fiducial_nobaryons_bins1234.npy`

### For Fiducial Data (With Noise)

```bash
python scripts/bnt_cross_power_spectrum_processing.py \
  --fiducial \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --noise-level 0.26 \
  --aggregate-for-inference \
  --verbose
```

## Custom Output Directory

You can specify a custom output directory for the aggregated files:

```bash
python scripts/bnt_cross_power_spectrum_processing.py \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --inference-output-dir /path/to/custom/output \
  --verbose
```

## Complete Workflow: Process Both Grid and Fiducial

```bash
#!/bin/bash

# 1. Process and aggregate grid data (no noise)
echo "Processing grid data..."
python scripts/bnt_cross_power_spectrum_processing.py \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose

# 2. Process and aggregate fiducial data (no noise)
echo "Processing fiducial data..."
python scripts/bnt_cross_power_spectrum_processing.py \
  --fiducial \
  --bnt-bin-range 0 1 2 3 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose

echo "All done! Files are ready for inference."
```

## Running Inference

After processing, you can run the inference with:

```bash
python scripts/run_npe_inference_auto_cross_ps.py \
  --simulation-type nobaryons \
  --bnt \
  --bnt-bins 0,1,2,3 \
  --train \
  --epochs 1000 \
  --verbose
```

## File Format

The aggregated files have the following structure:

### Auto Power Spectra Files
- **Filename**: `all_bnt_cls_{dataset}_{maptype}_bin{N}[_noisy_s{level}].npy`
- **Shape**: `(n_files, n_multipoles)`
- **n_multipoles**: 1025 (from ℓ=0 to ℓ=1024)

### Cross Power Spectra File
- **Filename**: `all_bnt_cross_cls_{dataset}_{maptype}_bins{1234}[_noisy_s{level}].npy`
- **Shape**: `(n_files, n_cross_pairs * n_multipoles)`
- **For 4 bins**: 6 cross pairs × 1025 multipoles = 6150 total length
- **Cross pairs order**: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)

## Command-Line Arguments

### Processing Options
- `--fiducial`: Process fiducial cosmology instead of grid
- `--base-dir`: Override default base directory
- `--baryonified`: Use baryonified maps instead of nobaryons
- `--bnt-bin-range`: BNT bins to include (0-indexed, default: 0 1 2 3)
- `--cross-only`: Only compute cross spectra (excludes auto spectra)

### Noise Options
- `--noise-level`: Shape noise level σₑ (default: 0.26)
- `--no-noise`: Don't add shape noise

### Algorithm Parameters
- `--lmax`: Maximum multipole (default: 1024)

### Execution Options
- `--num-workers`: Number of parallel workers (default: 70)
- `--verbose`: Print detailed progress

### Output Options
- `--save-combined`: Create summary file
- `--aggregate-for-inference`: **NEW** - Aggregate for inference
- `--inference-output-dir`: Custom output directory for aggregated files

## Verifying Output Files

Check that files were created correctly:

```bash
# List output files
ls -lh /home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_bnt_*.npy
ls -lh /home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/all_bnt_*.npy

# Check shape with Python
python3 << EOF
import numpy as np
# Check auto spectrum
auto = np.load('/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_bnt_cls_grid_nobaryons_bin1.npy')
print(f"Auto spectrum shape: {auto.shape}")  # Expected: (n_files, 1025)

# Check cross spectrum
cross = np.load('/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_bnt_cross_cls_grid_nobaryons_bins1234.npy')
print(f"Cross spectrum shape: {cross.shape}")  # Expected: (n_files, 6150)
EOF
```

## Troubleshooting

### No files to aggregate
- Make sure the processing step completed successfully
- Check that `.npz` files exist in the expected directories
- Verify file paths with `--verbose`

### Wrong output directory
- Use `--inference-output-dir` to specify custom location
- Check default paths: `new_grid/` for grid, `fiducial/cosmo_fiducial/` for fiducial

### Mismatched file counts
- Ensure all processed files have the expected BNT bins
- Use `--verbose` to see which files are being loaded
- Check for errors during the processing step

## Advantages of This Workflow

1. **One-step processing**: No need for separate aggregation scripts
2. **Correct format**: Automatically creates files in the exact format expected by inference
3. **Efficient**: Processes and aggregates in one run
4. **Flexible**: Can process with/without noise, grid/fiducial, custom output directories
5. **Safe**: Original `.npz` files are preserved; aggregated files are separate

## Old vs New Workflow

### Old Workflow (3 steps)
```bash
# 1. Process files (creates individual .npz files)
python scripts/bnt_cross_power_spectrum_processing.py --bnt-bin-range 0 1 2 3 --num-workers 80 --no-noise

# 2. Aggregate into one file
python scripts/aggregate_cross_power_spectra.py --base-dir ... --output aggregated.npz

# 3. Split into per-bin files
python scripts/split_auto_cross_for_inference.py --input-file aggregated.npz ...
```

### New Workflow (1 step)
```bash
# Process and aggregate in one step
python scripts/bnt_cross_power_spectrum_processing.py --bnt-bin-range 0 1 2 3 --num-workers 80 --no-noise --aggregate-for-inference
```

## BNT Bin Indexing

**Important**: BNT bins use 0-based indexing in the script arguments but 1-based indexing in file names and keys:

- Script argument: `--bnt-bin-range 0 1 2 3` (0-indexed)
- File names: `bin1`, `bin2`, `bin3`, `bin4` (1-based)
- Inference script: `--bnt-bins 0,1,2,3` (0-indexed)

The script handles this conversion automatically.
