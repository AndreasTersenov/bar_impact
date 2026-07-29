# NPE Inference Scripts Refactoring Analysis

## Scripts to Refactor
1. `run_npe_inference.py` - L1 norms inference
2. `run_npe_peak_counts_inference.py` - Peak counts inference  
3. `run_npe_inference_auto_cross_ps_master.py` - Auto + Cross power spectra inference

## Existing Module Functionality

### Available in modules:
- ✅ `bar_impact.analysis.aggregation.load_datavectors()` - Load data + params
- ✅ `bar_impact.analysis.aggregation.ResultsAggregator` - Load/filter/process data
- ✅ `bar_impact.utils.paths.get_data_file_paths()` - File discovery
- ✅ `bar_impact.utils.reproducibility.get_deterministic_seed()` - Seed management
- ✅ `bar_impact.core.ConvergenceMap` - Map handling
- ✅ `bar_impact.processing.bnt.apply_bnt_transform()` - BNT transforms

### Key Patterns in Original Scripts:

1. **File Path Construction**: All three scripts duplicate path construction logic
   - Can use `get_data_file_paths()` from utils.paths
   - Scripts use pattern: `f"_l1_norms_bin{bin}_noisy_s{noise:.2f}_new_normalization.npy"`

2. **Zero-Variance Filtering**: Only `run_npe_inference.py` has this
   - Function: `filter_zero_variance_bins()`
   - Should be added to `ResultsAggregator` or new utility

3. **Bin Range Selection**: `run_npe_inference.py` has `parse_bin_ranges()` function
   - Selects specific ranges from datavectors
   - Should be added to `ResultsAggregator.select_bins()` or similar

4. **Scale Selection**: All scripts select wavelet scales
   - Module has: `ResultsAggregator.select_scales()`
   - But scripts use more complex per-bin scale selection
   - Need to verify module implementation matches

5. **TARP Coverage Testing**: Duplicated across all scripts
   - Functions: `run_tarp_coverage_test()`, `plot_tarp_coverage()`
   - Should be moved to `bar_impact.inference` or `bar_impact.analysis`

6. **NaN Retry Training**: Duplicated in L1 and peak counts scripts
   - Function: `train_with_nan_retry()`
   - Should be moved to `bar_impact.inference` module

## Identified Contradictions

### 1. File Naming Patterns
**Scripts use**: 
- L1 norms: `all_l1_norms_{simulation_type}_bin{bin}{mask_suffix}_noisy_s{noise}.npy`
- Peak counts: `all_peak_counts_{simulation_type}_bin{bin}{mask_suffix}_noisy_s{noise}.npy`
- Power spectra: Complex naming with lmax, apodization, etc.

**Module has**: `build_output_suffix()` in utils/paths.py but it's not widely used yet

**Resolution**: Scripts should use `build_output_suffix()` for consistency, OR we accept different naming conventions for processed vs aggregated files.

### 2. Scale Selection Logic
**Scripts have**: Per-bin scale selection with format like "1,2,3;0,1,2,3;0,1,2,3;0,1,2,3"

**Module has**: Simple scale selection for all bins together

**Resolution**: Need to enhance `ResultsAggregator.select_scales()` to support per-bin scales

### 3. Bin Range Selection
**Scripts have**: Parse bin ranges like "start:end" or "start1:end1,start2:end2,..."

**Module has**: No equivalent functionality

**Resolution**: Add `select_bin_range()` method to `ResultsAggregator`

### 4. Zero-Variance Filtering
**Scripts have**: `filter_zero_variance_bins()` in run_npe_inference.py only

**Module has**: Basic NaN/Inf filtering in `ResultsAggregator`

**Resolution**: Add `filter_zero_variance()` method to `ResultsAggregator`

## Refactoring Strategy

1. **Create new utility module**: `bar_impact/utils/inference.py`
   - Move TARP testing functions
   - Move NaN retry training function
   - Add helper functions for NPE workflows

2. **Enhance ResultsAggregator**:
   - Add `select_bin_range()` method
   - Add `filter_zero_variance()` method  
   - Enhance `select_scales()` to support per-bin selection

3. **Create v2 versions of scripts**:
   - Use `get_data_file_paths()` for path construction
   - Use enhanced `ResultsAggregator` for data loading
   - Use new `bar_impact.utils.inference` module for TARP and training
   - Keep minimal script-specific logic (argument parsing, plotting)

4. **Don't break existing behavior**:
   - Keep file naming conventions as-is in scripts
   - Modules provide utilities, scripts decide naming
   - No changes to numerical results

## No Major Contradictions Found

After analysis, there are **no fundamental algorithmic contradictions** between the scripts and modules. The scripts simply have more specialized functionality that doesn't exist in modules yet. The refactoring will:
- Move duplicated code to modules
- Add missing utility functions
- Maintain identical numerical behavior
- Improve code reusability
