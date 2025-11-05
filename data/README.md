# Data Directory

This directory contains input data files for the BAR_IMPACT analyses.

## Contents

- `*.npy` - NumPy array files containing pre-computed data
- `*.fits` - FITS files with cosmological maps
- Other simulation inputs

## Data Files (as of repository state)

The following data files are included:
- `Gaia_EDR3_flux.npy` - Gaia early data release flux measurements
- `cov_scaled_cls_fid_kg*.npy` - Covariance matrices for power spectra
- `l1_list_kg_array*.npy` - L1 norm arrays
- `l1norm_list_*_4bins_200perms.npy` - L1 norms with permutations
- `scaled_fid_cls_arrays_kg*.npy` - Scaled fiducial power spectra
- `target_map_lss.npy` - Target large-scale structure map

## Storage Note

⚠️ **Large data files are gitignored by default**

Due to file size considerations, most `.npy` and `.fits` files are excluded from git tracking (see `.gitignore`). If you need to share or backup data:

1. Use external storage (e.g., cloud storage, shared drives)
2. Document data locations in this README
3. Provide download scripts if data is publicly available

## Adding New Data

When adding new data files:
1. Place them in this directory
2. Update this README with file descriptions
3. Ensure large files are properly gitignored
4. Consider using `data/raw/` and `data/processed/` subdirectories for organization
