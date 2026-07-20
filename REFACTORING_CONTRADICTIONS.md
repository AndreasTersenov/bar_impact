# Refactoring Contradictions Report

This document lists the contradictions found between the original scripts and the new modular organization, along with resolutions.

## 1. Shape Noise Function - Parameter Naming & Formula

### Contradiction
- **Scripts** (`l1_norm_processing_new_mask.py`, etc.): 
  - Use parameter name `galaxy_density`
  - Formula: `sigma_pix = sigma_e / sqrt(galaxy_density * pixel_area_arcmin2)`
  
- **Module** (`src/bar_impact/utils/noise.py`):
  - Uses parameter name `ngal_arcmin2`
  - Formula: `sigma_e / sqrt(2 * ngal_per_pixel)` (includes factor of 2)

### Physics Background
The factor of 2 in the module formula is **incorrect** for convergence (kappa). The factor of 2 applies to shear components (gamma) because:
- Shear has two independent components (gamma1, gamma2)
- Convergence is a single scalar quantity derived from both shear components
- For convergence: `sigma_kappa = sigma_e / sqrt(n_gal * A_pixel)`
- For shear: `sigma_gamma = sigma_e / sqrt(2 * n_gal * A_pixel)`

### Resolution
- **FIXED**: Update `noise.py` module to remove the factor of 2 for convergence
- **FIXED**: Add parameter alias for backward compatibility (`galaxy_density` → `ngal_arcmin2`)
- Added clarification in docstring about convergence vs shear noise

## 2. Mask Creation - Binary vs Apodized

### Contradiction
- **Scripts**: Use `create_euclid_mask()` which returns binary masks (values 0 or 1)
- **Module** (`src/bar_impact/core/masks.py`): Provides both `create_disk_mask()` (binary) and `create_apodized_disk_mask()` (smooth)

### Context
- Basic scripts (L1 norms, peak counts) work fine with binary masks
- Cross power spectrum MASTER script requires apodized masks for proper mode-coupling correction
- The module correctly implements both types

### Resolution
- **NO FIX NEEDED**: Module design is correct
- Refactored scripts will use appropriate mask type:
  - Binary masks for L1 norms and peak counts
  - Apodized masks for power spectrum MASTER correction

## 3. Default Constants

### Contradiction
- **Scripts**: Hardcode values like `center_coords=(0.0, 90.0)`, `nside=512`, etc.
- **Module**: Defines constants in `constants.py`:
  - `DEFAULT_MASK_CENTER = (0.0, 90.0)`
  - `DEFAULT_NSIDE = 512`
  - `DEFAULT_GALAXY_DENSITY = 6.75`
  - etc.

### Resolution
- **NO FIX NEEDED**: Module design is correct
- Refactored scripts will import and use constants from `bar_impact.constants`

## 4. BNT Matrix

### Contradiction
- **Scripts**: Define BNT matrix locally in each script
- **Module**: Centralizes in `constants.py` as `BNT_MATRIX_DEFAULT`

### Resolution  
- **NO FIX NEEDED**: Module design is correct
- Refactored scripts will import from `bar_impact.constants`

## 5. Processing Configuration

### Observation
- **Scripts**: Use argparse with many individual command-line arguments
- **Module**: Provides `ProcessingConfig` dataclass and `BaseProcessor` class

### Resolution
- **NO FIX NEEDED**: Module design is improvement
- Refactored scripts will:
  - Keep argparse interface for backward compatibility
  - Map arguments to `ProcessingConfig` internally
  - Use processor classes for cleaner code

## Summary of Module Fixes

### Files Modified:
1. `src/bar_impact/utils/noise.py` - Fixed convergence noise formula (removed factor of 2)

### Files That Were Already Correct:
- `src/bar_impact/core/masks.py` - Mask implementation is correct
- `src/bar_impact/constants.py` - Constants are properly defined
- `src/bar_impact/processing/*.py` - Processor classes are well-designed

## Refactored Scripts to Create:
1. `scripts/l1_norm_processing_v2.py` - Uses modular organization
2. `scripts/peak_counts_processing_v2.py` - Uses modular organization  
3. `scripts/cross_power_spectrum_processing_master_v2.py` - Uses modular organization
4. `scripts/bnt_l1_norm_processing_v2.py` - Uses modular organization with BNT
5. `scripts/bnt_peak_counts_processing_v2.py` - Uses modular organization with BNT
6. `scripts/bnt_cross_power_spectrum_processing_master_v2.py` - Uses modular organization with BNT

Each refactored script will:
- Import from `bar_impact.*` modules instead of duplicating code
- Use processor classes (`L1NormProcessor`, `PeakCountProcessor`, etc.)
- Maintain the same command-line interface for backward compatibility
- Be more maintainable and follow DRY principle
