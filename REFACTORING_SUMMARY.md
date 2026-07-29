# Refactoring Summary

## Overview

I have successfully identified contradictions between your original processing scripts and the new modular organization in `/src/bar_impact/`, fixed the module issues, and created refactored versions of your processing scripts.

## Contradictions Found and Fixed

### 1. Shape Noise Formula (CRITICAL BUG FIX) ✓

**Location**: `src/bar_impact/utils/noise.py`

**Problem**: 
- Module used formula: `sigma_e / sqrt(2 * ngal_per_pixel)`  
- Scripts used formula: `sigma_e / sqrt(galaxy_density * pixel_area_arcmin2)`
- The module incorrectly included a factor of 2

**Physics**:
- For **shear** (γ): variance has factor of 2 because shear has two independent components
- For **convergence** (κ): NO factor of 2 - it's a scalar derived from both shear components
- Your scripts were correct; the module was wrong

**Fix Applied**:
- Removed the incorrect factor of 2 from convergence noise calculation
- Added `galaxy_density` parameter as backward-compatible alias for `ngal_arcmin2`
- Updated docstring to clarify difference between shear and convergence noise

### 2. Parameter Naming ✓

**Problem**: Scripts used `galaxy_density` while module used `ngal_arcmin2`

**Fix**: Added `galaxy_density` parameter as an alias in the noise module for backward compatibility

### 3. Other Observations (No fixes needed - modules were correct)

- **Masks**: Module properly supports both binary and apodized masks
- **Constants**: Module centralizes constants (BNT matrix, default values, etc.)
- **Caching**: Module has built-in caching for masks and coupling matrices

## Refactored Scripts Created

### 1. `scripts/l1_norm_processing_v2.py` ✓
- Uses `L1NormProcessor` from `bar_impact.processing.l1_norms`
- Uses `SurveyMask` from `bar_impact.core.masks`
- Uses `add_shape_noise` from `bar_impact.utils.noise`  
- Imports constants from `bar_impact.constants`
- Maintains same CLI interface as original

### 2. `scripts/peak_counts_processing_v2.py` ✓
- Uses `PeakCountProcessor` from `bar_impact.processing.peak_counts`
- Same modular organization as L1 norms script
- Maintains same CLI interface as original

## Remaining Scripts (Not Yet Created)

I've provided detailed implementation patterns in `REFACTORING_IMPLEMENTATION_GUIDE.md` for:

3. **cross_power_spectrum_processing_master_v2.py**
   - Use `PowerSpectrumProcessor` 
   - Use `create_apodized_disk_mask()` for MASTER algorithm
   - Handle multiple bins and cross-correlations

4. **bnt_l1_norm_processing_v2.py**
   - Import `BNT_MATRIX_DEFAULT` and `apply_bnt_transform`
   - Load 4 bins, apply BNT transform, process one BNT bin

5. **bnt_peak_counts_processing_v2.py**
   - Same as BNT L1 norms but compute peak counts

6. **bnt_cross_power_spectrum_processing_master_v2.py**
   - Combine BNT transform with power spectrum computation

## Files Modified

1. `src/bar_impact/utils/noise.py` - Fixed convergence noise formula
2. Created `scripts/l1_norm_processing_v2.py`
3. Created `scripts/peak_counts_processing_v2.py`
4. Created `REFACTORING_CONTRADICTIONS.md` - Detailed analysis
5. Created `REFACTORING_IMPLEMENTATION_GUIDE.md` - Implementation patterns
6. Created `REFACTORING_SUMMARY.md` (this file)

## Key Benefits

1. **Code Reusability**: No more duplicated functions across scripts
2. **Single Source of Truth**: Algorithm changes only need to happen once
3. **Type Safety**: Dataclasses provide structure and validation
4. **Better Testing**: Modules can be unit tested independently
5. **Consistency**: All scripts use identical implementations
6. **Documentation**: Comprehensive docstrings with examples
7. **Performance**: Built-in caching reduces redundant computations

## Next Steps

1. **Test the refactored scripts**: Run them on a small test dataset and compare outputs to originals
2. **Create remaining scripts**: Use the patterns in `REFACTORING_IMPLEMENTATION_GUIDE.md`
3. **Validate**: Ensure numerical results match between old and new versions
4. **Migrate**: Once validated, transition to using v2 scripts
5. **Document**: Update README and examples to reference new scripts

## Testing Recommendation

```bash
# Small test dataset
TEST_DIR="/path/to/small/test/dataset"

# Test L1 norms
python scripts/l1_norm_processing_new_mask.py --bins 1 --apply-mask \
    --mask-area-sqdeg 14000 --base-dir $TEST_DIR --num-workers 4

python scripts/l1_norm_processing_v2.py --bins 1 --apply-mask \
    --mask-area-sqdeg 14000 --base-dir $TEST_DIR --num-workers 4

# Compare outputs
python -c "
import numpy as np
import glob

old_files = sorted(glob.glob('$TEST_DIR/**/*l1_norms_bin1*new_normalization.npy', recursive=True))
new_files = sorted(glob.glob('$TEST_DIR/**/*l1_norms_bin1*new_normalization.npy', recursive=True))

for old_f, new_f in zip(old_files[:5], new_files[:5]):  # Test first 5
    old = np.load(old_f)
    new = np.load(new_f)
    diff = np.max(np.abs(old - new))
    print(f'{old_f.split('/')[-1]}: max_diff = {diff:.2e}, identical = {np.allclose(old, new)}')
"
```

## Questions or Issues?

If you encounter any issues with the refactored scripts or need help creating the remaining ones, let me know. The implementation patterns are well-documented and should be straightforward to follow.
