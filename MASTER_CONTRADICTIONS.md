# Contradictions Between Original MASTER Scripts and bar_impact Modules

## Overview
The original `cross_power_spectrum_processing_master.py` and `bnt_cross_power_spectrum_processing_master.py` scripts include advanced functionality not present in the bar_impact modules:

## Key Missing Features in Modules

### 1. **Apodization Type Parameter** (CRITICAL)
**Original Scripts:**
- Support two apodization types: `'C1'` (cosine taper) and `'C2'` (polynomial, twice differentiable)
- C2 is recommended for weak lensing power spectra
- Implementation:
  ```python
  if apodization_type == 'C1':
      taper = 0.5 * (1.0 + np.cos(np.pi * x))
  elif apodization_type == 'C2':
      taper = np.where(x < 0.5, 1.0 - 2 * x**2, 2 * (1 - x)**2)
  ```

**bar_impact.core.masks.SurveyMask.create_apodized_disk_mask():**
- Only implements C1 (cosine taper): `0.5 * (1 + np.cos(np.pi * frac))`
- No `apodization_type` parameter
- Less smooth for power spectrum analysis

**Impact:** Suboptimal apodization for power spectrum estimation (C2 is preferred for lensing)

---

### 2. **NaMaster Integration** (CRITICAL)
**Original Scripts:**
- Full NaMaster support for MASTER algorithm mode-coupling correction
- Functions: `get_coupling_matrix()`, `compute_power_spectra_master()`
- Caching of mode-coupling matrices (MCM_CACHE)
- Handles Nyquist limit (lmax ≤ 3*nside - 1)
- Adaptive binning strategy for high lmax

**bar_impact.processing.power_spectrum:**
- No NaMaster support
- Only naive pseudo-Cl computation via `hp.map2alm()` and `hp.alm2cl()`
- No mode-coupling correction → biased power spectra for masked data

**Impact:** Cannot produce unbiased power spectra from masked data

---

### 3. **Deterministic Random Seeds** (IMPORTANT)
**Original Scripts:**
- `get_deterministic_seed(file_path, global_seed)` - generates reproducible seeds from file paths
- Each file gets unique but deterministic seed
- Enables exact reproducibility

**bar_impact.utils.noise:**
- Uses system entropy: `np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))`
- Non-reproducible across runs

**Impact:** Cannot reproduce exact results

---

### 4. **Per-Bin Galaxy Densities** (MODERATE)
**Original Scripts:**
- Support different galaxy densities per redshift bin: `galaxy_densities=[n1, n2, n3, n4]`
- More realistic for tomographic surveys

**bar_impact Constants:**
- Single `DEFAULT_GALAXY_DENSITY = 6.75`
- `add_shape_noise()` takes one `galaxy_density` parameter

**Impact:** Less accurate noise modeling for multi-bin analysis

---

### 5. **Advanced Mask Validation**
**Original Scripts:**
- Validates mask/map consistency (nside matching)
- Checks effective f_sky after apodization
- Warns about Nyquist limit violations

**bar_impact Modules:**
- Basic validation only

---

### 6. **Aggregation for Inference**
**Original Scripts:**
- `aggregate_for_inference()` function
- Validates consistent metadata across files
- Creates inference-ready .npy files
- Comprehensive error reporting

**bar_impact Modules:**
- No aggregation utilities

---

## Contradictions Summary

| Feature | Original Scripts | bar_impact Modules | Severity |
|---------|------------------|-------------------|----------|
| Apodization Type (C1/C2) | ✓ Both supported | ✗ C1 only | CRITICAL |
| NaMaster/MASTER | ✓ Full support | ✗ None | CRITICAL |
| Deterministic Seeds | ✓ File-based | ✗ Random | IMPORTANT |
| Per-Bin Densities | ✓ Supported | ~ Single default | MODERATE |
| MCM Caching | ✓ Yes | ✗ None | IMPORTANT |
| Nyquist Validation | ✓ Yes | ✗ None | MODERATE |
| Aggregation Tools | ✓ Yes | ✗ None | LOW |

---

## Recommended Fixes

### Fix 1: Create `master_correction.py` Module
Create `src/bar_impact/processing/master_correction.py` with:
- NaMaster integration
- MCM computation and caching
- MASTER-corrected power spectrum computation
- Proper handling of spin-0 convergence fields

### Fix 2: Update `masks.py`
Add `apodization_type` parameter to `create_apodized_disk_mask()`:
- Support both 'C1' and 'C2' apodization
- Default to 'C2' for power spectrum applications

### Fix 3: Update `noise.py`
Add `rng` parameter to `add_shape_noise()`:
- Accept `np.random.Generator` for reproducibility
- Keep backwards compatibility with default random behavior

### Fix 4: Constants Update
Add `DEFAULT_GALAXY_DENSITIES = [6.75, 6.75, 6.75, 6.75]` for per-bin defaults

---

## Migration Strategy

1. **Immediate:** Create `master_correction.py` module (new file, no breaking changes)
2. **Immediate:** Update `masks.py` apodization (backwards compatible: default to 'C1')
3. **Optional:** Update `noise.py` for reproducibility (backwards compatible)
4. **Documentation:** Note that MASTER-corrected scripts require NaMaster: `pip install pymaster`
