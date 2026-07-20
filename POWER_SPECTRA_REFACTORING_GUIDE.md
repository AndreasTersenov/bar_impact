# Refactoring Guide for run_npe_inference_auto_cross_ps_master.py

## Overview

The `run_npe_inference_auto_cross_ps_master.py` script is highly specialized for power spectra processing with MASTER corrections. Due to its domain-specific complexity (1189 lines), a full rewrite would be error-prone and provide minimal benefit.

## Recommended Approach: Minimal Refactoring

Instead of a full v2 rewrite, we recommend a **targeted refactoring** that:
1. Eliminates only the duplicated TARP testing code (~200 lines)
2. Preserves all power spectra-specific processing logic
3. Maintains identical numerical behavior

## Code Duplication Analysis

### Functions to Refactor (can use bar_impact.utils.inference):

1. **run_tarp_coverage_test()** (lines 697-776, ~80 lines)
   - Replace with: `from bar_impact.utils.inference import run_tarp_coverage_test`
   - This function is identical to other inference scripts
   - Eliminates 80 lines of duplicated code

2. **plot_tarp_coverage()** (lines 777-830, ~53 lines)
   - Replace with: `from bar_impact.utils.inference import plot_tarp_coverage`
   - This function is identical to other inference scripts
   - Eliminates 53 lines of duplicated code

**Total Code Reduction: ~133 lines**

### Functions to Keep (domain-specific, not suitable for modules):

1. **parse_upper_cuts()** - Handles per-bin multipole cuts
2. **get_cross_indices_for_pairs()** - Cross-pair selection logic
3. **construct_auto_paths()** - Complex file path construction for auto/cross spectra
4. **construct_cross_paths()** - Cross power spectra file paths
5. **rebin_cls()** - Power spectrum rebinning
6. **load_and_process_auto_spectra()** - Multipole cutting, ell offset handling, binning
7. **load_and_process_cross_spectra()** - Cross spectra processing with pair selection
8. **load_and_process_auto_fiducial()** - Fiducial auto spectra processing
9. **load_and_process_cross_fiducial()** - Fiducial cross spectra processing

These functions contain power spectra-specific logic that:
- Is unique to this analysis type
- Requires domain expertise to modify
- Would be rarely reused by other scripts
- Would complicate modules with highly specialized code

## Step-by-Step Refactoring Instructions

### Step 1: Add imports
At the top of the script, add:
```python
from bar_impact.utils.inference import (
    run_tarp_coverage_test,
    plot_tarp_coverage,
)
```

### Step 2: Remove duplicated functions
Delete these function definitions from the script:
- `run_tarp_coverage_test()` (lines 697-776)
- `plot_tarp_coverage()` (lines 777-830)

### Step 3: Update main() function
In the main() function, find the coverage testing section and update it:

**Original:**
```python
if args.run_coverage_test:
    ecp, alpha = run_tarp_coverage_test(
        posterior, combined_data_vector, params, args
    )
    
    # Plot coverage
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename_base = checkpoint_name
    if args.run is not None:
        output_filename_base += f"_run{args.run}"
    
    plot_tarp_coverage(
        ecp, alpha, args, args.output_dir, output_filename_base
    )
```

**Refactored:**
```python
if args.run_coverage_test:
    ecp, alpha = run_tarp_coverage_test(
        posterior=posterior,
        data=combined_data_vector,
        params=params,
        num_test_sims=args.coverage_num_sims,
        num_samples=args.coverage_num_samples,
        seed=args.coverage_seed,
        bootstrap=args.coverage_bootstrap,
        num_bootstrap=args.coverage_num_bootstrap if args.coverage_bootstrap else None
    )
    
    # Plot coverage
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename_base = checkpoint_name
    if args.run is not None:
        output_filename_base += f"_run{args.run}"
    
    plot_tarp_coverage(
        ecp=ecp,
        alpha=alpha,
        output_path=os.path.join(args.output_dir, f"{output_filename_base}_tarp_coverage.pdf"),
        bootstrap=args.coverage_bootstrap,
        figsize=(6, 6),
        dpi=300
    )
```

### Step 4: Test
Run the script with coverage testing enabled to verify:
```bash
python run_npe_inference_auto_cross_ps_master.py \\
    --train \\
    --run-coverage-test \\
    --coverage-num-sims 10 \\
    --coverage-num-samples 100 \\
    --epochs 10  # Short run for testing
```

## Benefits of This Approach

1. **Minimal Risk**: Only touches TARP functions that are identical to other scripts
2. **Clear Benefit**: Eliminates 133 lines of duplicated code
3. **Maintainable**: Future TARP updates only need to happen in one place
4. **Preserves Domain Logic**: All power spectra processing remains intact
5. **Easy to Verify**: Can diff the outputs before/after refactoring

## Why Not Full Refactoring?

The remaining 1000+ lines contain highly specialized power spectra processing:
- **Complex multipole cutting logic** with ell offset handling for different data sources
- **Binning considerations** (nlb=1,2,4) based on lmax
- **Cross-pair selection** and ordering
- **Auto/cross concatenation** with flexible selection
- **BNT cross spectra absolute value handling**

Moving this logic to modules would:
- Create highly specialized modules used by only one script
- Require extensive testing of edge cases
- Risk introducing bugs in scientific analysis
- Provide minimal reusability benefit

## Alternative: Documentation

If the power spectra processing logic needs to be reused:
1. Keep it in this script
2. Add comprehensive docstrings
3. Create a notebook demonstrating the workflow
4. Reference this script as the canonical implementation

## Conclusion

**Recommended Action**: Minimal refactoring to eliminate only the TARP duplication (~133 lines).

**Not Recommended**: Full rewrite attempting to modularize all power spectra processing.

The script is already well-organized with clear function boundaries. The TARP refactoring provides the main benefit (reducing duplication) with minimal risk.
