# Cross Power Spectrum Processing Workflow

This guide explains how to use `cross_power_spectrum_processing.py` to process HEALPix maps and create inference-ready power spectra files.

## Overview

The script now has two modes:
1. **Standard mode**: Processes maps and saves individual `.npz` files per cosmology/permutation
2. **Inference mode** (with `--aggregate-for-inference`): Additionally aggregates all files into the format expected by `run_npe_inference_auto_cross_ps.py`

## Quick Start

### For Grid Data (nobaryons, no noise)

```bash
python scripts/cross_power_spectrum_processing.py \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

This will:
1. Process all grid cosmologies (7 perms × ~1000+ cosmologies)
2. Compute auto + cross power spectra for bins 1,2,3,4
3. Save individual `.npz` files in each cosmology/perm directory
4. **Automatically aggregate** into inference-ready files:
   - `all_cls_grid_nobaryons_bin1.npy`
   - `all_cls_grid_nobaryons_bin2.npy`
   - `all_cls_grid_nobaryons_bin3.npy`
   - `all_cls_grid_nobaryons_bin4.npy`
   - `all_cross_cls_grid_nobaryons_bins1234.npy`

### For Fiducial Data

```bash
python scripts/cross_power_spectrum_processing.py \
  --fiducial \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

This creates:
- `all_cls_fiducial_nobaryons_bin1.npy`
- `all_cls_fiducial_nobaryons_bin2.npy`
- `all_cls_fiducial_nobaryons_bin3.npy`
- `all_cls_fiducial_nobaryons_bin4.npy`
- `all_cross_cls_fiducial_nobaryons_bins1234.npy`

### With Noise

```bash
python scripts/cross_power_spectrum_processing.py \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --noise-level 0.26 \
  --aggregate-for-inference \
  --verbose
```

This creates files with `_noisy_s0.26` suffix.

### For Baryonified Maps

```bash
python scripts/cross_power_spectrum_processing.py \
  --baryonified \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

## Command-Line Arguments

### Processing Options
- `--fiducial`: Process fiducial cosmology (200 perms) instead of grid
- `--base-dir PATH`: Override default base directory
- `--baryonified`: Use baryonified maps instead of nobaryons
- `--bin-range 1 2 3 4`: Redshift bins to process (default: 1 2 3 4)
- `--cross-only`: Only compute cross spectra, exclude autos

### Noise Options
- `--no-noise`: Don't add shape noise (default adds noise)
- `--noise-level 0.26`: Shape noise level σ_e (default: 0.26)

### Algorithm Parameters
- `--lmax 1024`: Maximum multipole (default: 1024)

### Execution Options
- `--num-workers 80`: Number of parallel processes (default: 70)
- `--verbose`: Print detailed progress

### Output Options
- `--save-combined`: Create summary text file listing all processed files
- `--aggregate-for-inference`: **Aggregate into inference-ready format**
- `--inference-output-dir PATH`: Where to save inference files (default: base_dir)

## Output File Structure

### Without `--aggregate-for-inference`
Individual `.npz` files are created in each directory:
```
/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/
├── cosmo_000001/
│   ├── perm_0000/
│   │   ├── projected_probes_maps_nobaryons512.h5
│   │   └── projected_probes_maps_nobaryons512_all_cls_bins1234.npz  ← Created
│   ├── perm_0001/
│   │   └── projected_probes_maps_nobaryons512_all_cls_bins1234.npz  ← Created
...
```

Each `.npz` contains keys: `cls_1_1`, `cls_2_2`, `cls_3_3`, `cls_4_4` (autos) and `cls_1_2`, `cls_1_3`, `cls_1_4`, `cls_2_3`, `cls_2_4`, `cls_3_4` (crosses).

### With `--aggregate-for-inference`
Additional aggregated `.npy` files are created in the base directory:
```
/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/
├── all_cls_grid_nobaryons_bin1.npy              ← (n_files, 1025)
├── all_cls_grid_nobaryons_bin2.npy              ← (n_files, 1025)
├── all_cls_grid_nobaryons_bin3.npy              ← (n_files, 1025)
├── all_cls_grid_nobaryons_bin4.npy              ← (n_files, 1025)
├── all_cross_cls_grid_nobaryons_bins1234.npy    ← (n_files, 6×1025)
└── cosmo_000001/...
```

These `.npy` files are **exactly** what `run_npe_inference_auto_cross_ps.py` expects!

## Complete Workflow Example

### Process Grid and Fiducial Together

```bash
#!/bin/bash

# Process grid data
echo "Processing grid data..."
python scripts/cross_power_spectrum_processing.py \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose

# Process fiducial data
echo "Processing fiducial data..."
python scripts/cross_power_spectrum_processing.py \
  --fiducial \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose

echo "All done! Ready for inference."
```

### Run Inference

After processing, you can immediately run:

```bash
python scripts/run_npe_inference_auto_cross_ps.py \
  --simulation-type nobaryons \
  --bins 1,2,3,4 \
  --train \
  --epochs 1000 \
  --verbose
```

The inference script will automatically find:
- Auto spectra: `/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_cls_grid_nobaryons_bin*.npy`
- Cross spectra: `/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_cross_cls_grid_nobaryons_bins1234.npy`
- Fiducial auto: `/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/all_cls_fiducial_nobaryons_bin*.npy`
- Fiducial cross: `/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/all_cross_cls_fiducial_nobaryons_bins1234.npy`

## Verification

Check the created files:

```bash
# Check grid files
ls -lh /home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_*.npy

# Check fiducial files
ls -lh /home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/all_*.npy

# Inspect a file
python -c "
import numpy as np
data = np.load('/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/all_cls_grid_nobaryons_bin1.npy')
print(f'Shape: {data.shape}')
print(f'n_files: {data.shape[0]}, n_multipoles: {data.shape[1]}')
"
```

## Troubleshooting

### Files already exist
If you re-run the script, it will skip already-processed individual `.npz` files but will regenerate the aggregated `.npy` files if `--aggregate-for-inference` is used.

### Missing bins in some files
The script continues processing even if some bins are missing in individual files. Check the verbose output for warnings.

### Memory issues
If you run out of memory during aggregation (with very large datasets), reduce `--num-workers` or process in smaller batches.

### Wrong file names
The inference script expects specific naming conventions:
- Grid: `all_cls_grid_<maptype>_bin<N>.npy`
- Fiducial: `all_cls_fiducial_<maptype>_bin<N>.npy`
- Cross: `all_cross_cls_<dataset>_<maptype>_bins<1234>.npy`

Make sure you use the correct flags (`--fiducial`, `--baryonified`, `--no-noise`) consistently.

## Advanced Usage

### Custom output directory

```bash
python scripts/cross_power_spectrum_processing.py \
  --bin-range 1 2 3 4 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --inference-output-dir /path/to/custom/output \
  --verbose
```

### Specific bins only

```bash
python scripts/cross_power_spectrum_processing.py \
  --bin-range 1 2 3 \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

This creates files for bins 1,2,3 and cross pairs (1,2), (1,3), (2,3).

### Cross-correlations only (no autos)

```bash
python scripts/cross_power_spectrum_processing.py \
  --bin-range 1 2 3 4 \
  --cross-only \
  --num-workers 80 \
  --no-noise \
  --aggregate-for-inference \
  --verbose
```

Note: When using `--cross-only`, you won't get the individual auto files, only the combined cross file.

## Summary

**Before**: Multi-step process (process → aggregate → split)
**Now**: Single command with `--aggregate-for-inference` flag!

```bash
# Old workflow (3 steps):
python scripts/cross_power_spectrum_processing.py --bin-range 1 2 3 4 --num-workers 80 --no-noise
python scripts/aggregate_cross_power_spectra.py --base-dir ... --pattern ... --include-auto ...
python scripts/split_auto_cross_for_inference.py --input-file ... --output-dir ... --dataset-type ...

# New workflow (1 step):
python scripts/cross_power_spectrum_processing.py --bin-range 1 2 3 4 --num-workers 80 --no-noise --aggregate-for-inference
```

Done! 🎉
