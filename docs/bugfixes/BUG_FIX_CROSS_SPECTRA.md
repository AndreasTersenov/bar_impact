# Bug Fix: Cross Power Spectra Handling

## Summary
Fixed a critical bug in `run_npe_inference_auto_cross_ps.py` where cross power spectra were being incorrectly sliced, resulting in only the first cross-pair being used and all others being discarded.

## The Bug

### Location
- `load_and_process_cross_spectra()` (line ~267)
- `load_and_process_cross_fiducial()` (line ~335)

### Problem
The code was applying multipole cuts directly to the aggregated cross spectra array:
```python
cross_cls_cut = cross_cls_full[:, args.lower_cut:args.upper_cut]
```

This is fundamentally wrong because:
1. **Cross power spectra are concatenated along the column dimension**
   - For N cross-pairs with M multipoles each: shape = `(n_sims, N*M)`
   - Example: 6 pairs × 1024 multipoles = `(n_sims, 6144)` columns

2. **The slicing `[:, 30:1024]` takes only columns 30-1023**
   - This corresponds to multipoles l=30 to l=1023 of **only the first cross-pair**
   - All other 5 cross-pairs are immediately discarded!

3. **Downstream calculations were also wrong**
   - After the incorrect cut, `total_cross_pairs = cross_cls_cut.shape[1] // n_multipoles_cut`
   - Would calculate ~1 pair instead of the actual 6 pairs
   - Cross-pair selection logic would fail silently

### Impact
- **All analyses using cross power spectra were using only 1/N of the data**
- **Selected cross-pairs (via `--cross-pairs`) were not actually being selected**
- **Inference results were based on incomplete data**

## The Fix

### Correct Approach
1. **Calculate the number of expected cross-pairs** from the number of bins:
   ```python
   expected_cross_pairs = n_bins * (n_bins - 1) // 2
   ```

2. **Infer the original multipole range** per cross-pair:
   ```python
   n_ell_original = cross_cls_full.shape[1] // expected_cross_pairs
   ```

3. **Apply multipole cuts to each cross-pair individually**:
   ```python
   for i in range(expected_cross_pairs):
       start_col = i * n_ell_original
       end_col = (i + 1) * n_ell_original
       cross_pair_full = cross_cls_full[:, start_col:end_col]
       cross_pair_cut = cross_pair_full[:, args.lower_cut:args.upper_cut]
   ```

4. **Then select specific pairs if requested**:
   ```python
   if cross_indices is not None:
       selected_cross_cls = [cross_cls_cut_list[idx] for idx in cross_indices]
   ```

### Changes Made

#### Function Signatures
- Added `n_bins` parameter to both functions:
  - `load_and_process_cross_spectra(..., n_bins=None)`
  - `load_and_process_cross_fiducial(..., n_bins=None)`

#### Implementation
- Calculate expected number of cross-pairs from `n_bins`
- Infer original multipole range from data shape
- Apply cuts per cross-pair, not globally
- Added verbose logging for debugging
- Added fallback for when `n_bins` is not provided (with warning)

#### Call Sites
- Updated calls in `main()` to pass `n_bins=len(bin_indices)`
- Added explicit calculation and printing of `n_bins`

### Additional Fix
Clarified BNT bin indexing:
- BNT bins are 0-indexed internally (0,1,2,3)
- But labeled as 1,2,3,4 for cross-pair naming
- Separated `bnt_bin_indices` from `bin_indices` for clarity

## Verification

### What to Check
1. **Data shapes should now be correct**:
   - For 4 bins: 6 cross-pairs should be loaded (not just 1)
   - Cross data shape should be `(n_sims, 6 * n_multipoles_cut)`
   
2. **Cross-pair selection should work**:
   - `--cross-pairs "1,3;1,4"` should load exactly 2 cross-pairs
   - Not truncate to only the first pair's data

3. **Multipole cuts should apply to all pairs**:
   - Each of the 6 cross-pairs should have multipoles [30:1024]
   - Not just the first one

### Testing
Run with `--verbose` flag to see detailed information:
```bash
python scripts/run_npe_inference_auto_cross_ps.py \
    --bins 1,2,3,4 \
    --verbose \
    --lower-cut 30 \
    --upper-cut 1024
```

Check the output for:
- "Expected N cross pairs with M multipoles each"
- "Applied cuts to N cross pairs"
- Correct data shapes

## Impact Assessment

### Past Results
**All previous inference results using this script may be incorrect** because:
- Only 1/N of the cross power spectrum data was used
- The wrong cross-pair was being analyzed (always the first one)
- Cross-pair selection was not working

### Action Required
- **Rerun all analyses** that used cross power spectra
- **Compare old vs new results** to assess the impact
- **Update any publications or reports** that used the incorrect results

## Related Files
- `run_npe_inference_auto_cross_ps.py` - Fixed file
- Any analysis scripts that aggregate cross power spectra should be checked for similar bugs
