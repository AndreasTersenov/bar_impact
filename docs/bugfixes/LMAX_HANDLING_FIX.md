# Fix: lmax Handling in File Paths

## Summary
Added support for non-default `lmax` values in filename construction to match the processing script's behavior.

## The Problem

The inference script was constructing filenames without considering the `lmax` parameter:
```python
# Old (WRONG):
data_filename = f"all_cls_grid_{args.simulation_type}_bin{bin}{noise_suffix}.npy"
cross_filename = f"all_cross_cls_grid_{args.simulation_type}_{bin_desc}{noise_suffix}.npy"
```

However, the processing script (`cross_power_spectrum_processing.py`) appends `_lmax{lmax}` to filenames when `lmax != 1024`:
```python
# From processing script:
lmax_suffix = f"_lmax{lmax}" if lmax != 1024 else ""
filename = f"all_cls_grid_nobaryons_bin1{noise_suffix}{lmax_suffix}.npy"
```

### Example Issue
If data was processed with `--lmax 2048`:
- Actual filename: `all_cls_grid_nobaryons_bin1_noisy_s0.26_lmax2048.npy`
- Inference script looked for: `all_cls_grid_nobaryons_bin1_noisy_s0.26.npy`
- Result: **FileNotFoundError**

## The Fix

### 1. Added `--lmax` Argument
```python
parser.add_argument("--lmax", type=int, default=1024,
                    help="Maximum multipole (lmax) used when computing power spectra. Must match the processing script's --lmax.")
```

### 2. Added Validation
```python
# Validate upper_cut doesn't exceed lmax
if args.upper_cut > args.lmax:
    parser.error(f"--upper-cut ({args.upper_cut}) cannot exceed --lmax ({args.lmax})")
```

### 3. Updated File Path Construction
Both `construct_auto_paths()` and `construct_cross_paths()` now include `lmax_suffix`:

```python
noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
lmax_suffix = f"_lmax{args.lmax}" if args.lmax != 1024 else ""

# Auto spectra:
data_filename = f"{data_prefix}_grid_{args.simulation_type}_{bin_spec}{noise_suffix}{lmax_suffix}.npy"

# Cross spectra:
data_filename = f"all_cross_cls_grid_{args.simulation_type}_{bin_desc}{noise_suffix}{lmax_suffix}.npy"
```

### 4. Added File Existence Validation
The script now checks if all required files exist before proceeding and provides helpful error messages:

```python
if missing_files:
    print("ERROR: Required files not found!")
    for f in missing_files:
        print(f"  ✗ {f}")
    print("\nPossible causes:")
    print(f"  1. The data files were processed with a different --lmax (current: {args.lmax})")
    print(f"  2. The data files were processed with different noise settings")
    print(f"  3. The data files don't exist yet - run cross_power_spectrum_processing.py first")
```

## Usage

### Default lmax (1024)
No changes needed - works as before:
```bash
python scripts/run_npe_inference_auto_cross_ps.py \
    --bins "1,2,3,4" \
    --simulation-type nobaryons \
    --train
```

### Non-default lmax
Specify the same lmax used during processing:
```bash
# If data was processed with --lmax 2048:
python scripts/run_npe_inference_auto_cross_ps.py \
    --bins "1,2,3,4" \
    --lmax 2048 \
    --upper-cut 2048 \
    --simulation-type nobaryons \
    --train
```

## Consistency Check

The `--lmax` parameter must match between:
1. **Processing script** (`cross_power_spectrum_processing.py --lmax N`)
2. **Inference script** (`run_npe_inference_auto_cross_ps.py --lmax N`)

The script will fail early with a clear error message if files with the correct lmax are not found.

## Related Files
- `run_npe_inference_auto_cross_ps.py` - Fixed file
- `cross_power_spectrum_processing.py` - Processing script that generates the files
