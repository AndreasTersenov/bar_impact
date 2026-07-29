# Refactoring Implementation Guide

This document describes the completed refactoring work and provides guidance for creating the remaining scripts.

## Completed Work

### 1. Module Fixes ✓

**File**: `src/bar_impact/utils/noise.py`

**Issue**: The module had an incorrect factor of 2 in the convergence noise formula.

**Fix Applied**:
- Removed factor of 2 from convergence noise calculation  
- Changed from: `sigma_e / sqrt(2 * ngal_per_pixel)`
- To: `sigma_e / sqrt(ngal_per_pixel * pixel_area_arcmin2)`
- Added `galaxy_density` parameter as alias for `ngal_arcmin2` for backward compatibility
- Updated docstring to clarify convergence vs shear noise formulas

### 2. Refactored Scripts Created ✓

**Created Files**:
1. `scripts/l1_norm_processing_v2.py` - L1 norm processing using modular organization
2. `scripts/peak_counts_processing_v2.py` - Peak counts processing using modular organization

**Key Improvements**:
- Import from `bar_impact.*` modules instead of duplicating code
- Use processor classes (`L1NormProcessor`, `PeakCountProcessor`)
- Import constants from `bar_impact.constants`
- Use `SurveyMask.create_disk_mask()` from `bar_impact.core.masks`
- Use `add_shape_noise()` from `bar_impact.utils.noise`
- Maintain same command-line interface for backward compatibility
- Cleaner, more maintainable code following DRY principle

## Remaining Scripts to Create

### 3. Cross Power Spectrum Processing (MASTER)

**Template**: Adapt from `scripts/cross_power_spectrum_processing_master.py`

**Key Modules to Use**:
- `bar_impact.processing.power_spectrum.PowerSpectrumProcessor`
- `bar_impact.core.masks.SurveyMask.create_apodized_disk_mask()` (for MASTER)
- NaMaster library for mode-coupling correction

**Key Differences from L1/Peak Counts**:
- Process multiple bins together (tomographic cross-correlations)
- Use apodized masks instead of binary masks
- Compute coupling matrix using NaMaster
- Return dictionary of power spectra for all auto/cross combinations
- Support bin_edges for bandpower binning

**Implementation Pattern**:
```python
from bar_impact.processing.power_spectrum import (
    PowerSpectrumProcessor, 
    compute_cross_power_spectrum
)

# For each cosmology:
#   - Load all bins into maps_dict
#   - Create apodized mask if needed
#   - For each bin pair (i,j):
#       - Compute pseudo-Cl
#       - Deconvolve with coupling matrix
#       - Save results
```

### 4-6. BNT Transform Scripts

The BNT scripts follow the same pattern as their non-BNT counterparts, with these additions:

**Additional Imports**:
```python
from bar_impact.constants import BNT_MATRIX_DEFAULT, get_bnt_matrix
from bar_impact.processing.bnt_transforms import apply_bnt_transform
```

**Processing Flow**:
```python
# 1. Load all 4 bins into array (4, npix)
maps = np.array([kg_bin1, kg_bin2, kg_bin3, kg_bin4])

# 2. Apply BNT transform
bnt_maps = apply_bnt_transform(maps, bnt_matrix=BNT_MATRIX_DEFAULT)

# 3. Process the specified BNT bin (0-3)
bnt_map = bnt_maps[bnt_bin]

# 4. Compute statistic on bnt_map (L1 norms, peak counts, or power spectrum)
```

**Files to Create**:
- `scripts/bnt_l1_norm_processing_v2.py`
- `scripts/bnt_peak_counts_processing_v2.py`  
- `scripts/bnt_cross_power_spectrum_processing_master_v2.py`

**Key Differences from Regular Scripts**:
- Load 4 bins, not 1
- Apply BNT transform before computing statistic
- Argument is `--bnt-bin` (0-3) instead of `--bin-number` (1-4)
- Output filenames include `_bnt{bin}` instead of `_bin{number}`
- Per-BNT-bin coarse scale SNR ranges

## Refactoring Patterns

### Pattern 1: Imports
```python
# OLD (scripts)
def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    # ... implementation ...

def create_euclid_mask(nside=512, target_area_sqdeg=14000.0, ...):
    # ... implementation ...

# NEW (refactored)
from bar_impact.utils.noise import add_shape_noise
from bar_impact.core.masks import SurveyMask
from bar_impact.constants import DEFAULT_NSIDE, DEFAULT_SIGMA_E, ...
```

### Pattern 2: Mask Creation
```python
# OLD
MASK_CACHE = {}
def get_cached_mask(...):
    # ... manual caching ...
    
# NEW
mask = SurveyMask.create_disk_mask(
    nside=DEFAULT_NSIDE,
    target_area_sqdeg=mask_area_sqdeg,
    center_coords=mask_center,
    use_cache=True  # Built-in caching
)
mask_array = mask.data
```

### Pattern 3: Processor Usage
```python
# OLD
from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere
_, l1_norms = get_wtl1_sphere(kg, nscales=5, nbins=40, ...)

# NEW
from bar_impact.processing.l1_norms import L1NormProcessor, L1NormConfig

config = L1NormConfig(nscales=5, nbins=40, ...)
processor = L1NormProcessor(config=config)
l1_norms = processor.process_single(kg)
```

### Pattern 4: Constants
```python
# OLD
BNT_MATRIX = np.array([[1., 0., ...], ...])
DEFAULT_NOISE_LEVEL = 0.26
nside = 512

# NEW
from bar_impact.constants import (
    BNT_MATRIX_DEFAULT,
    DEFAULT_SIGMA_E,
    DEFAULT_NSIDE
)
```

## Testing Refactored Scripts

To verify the refactored scripts produce identical results:

```bash
# Test L1 norms
python scripts/l1_norm_processing_new_mask.py --bins 1 --apply-mask --mask-area-sqdeg 14000 --base-dir /path/to/test --num-workers 1
python scripts/l1_norm_processing_v2.py --bins 1 --apply-mask --mask-area-sqdeg 14000 --base-dir /path/to/test --num-workers 1

# Compare outputs
python -c "
import numpy as np
old = np.load('old_output.npy')
new = np.load('new_output.npy')
print('Max difference:', np.max(np.abs(old - new)))
print('Identical:', np.allclose(old, new))
"
```

## Benefits of Refactored Code

1. **Maintainability**: Changes to core algorithms only need to be made once in modules
2. **Testability**: Modules can be unit tested independently  
3. **Reusability**: Functions and classes can be imported by other scripts/notebooks
4. **Consistency**: All scripts use same implementations, eliminating drift
5. **Documentation**: Modules have comprehensive docstrings and type hints
6. **Type Safety**: Dataclasses provide structure and validation
7. **Performance**: Built-in caching in modules (e.g., masks, coupling matrices)

## Migration Strategy

1. ✓ Fix module contradictions (noise formula)
2. ✓ Create v2 scripts for L1 norms and peak counts  
3. Test v2 scripts against originals
4. Create remaining v2 scripts (cross power spectrum, BNT variants)
5. Once validated, consider deprecating original scripts
6. Update documentation and examples to use v2 scripts
