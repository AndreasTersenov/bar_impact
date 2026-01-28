# Tests Directory

This directory contains test scripts for validating the BAR_IMPACT package functionality.

## Test Files

### Unit Tests (`unit/`)
- `test_core.py` - Tests for core map processing functionality
- `test_utils.py` - Tests for utility functions (reproducibility, paths, noise)
- `test_processing.py` - Tests for data processing pipelines
- `test_inference.py` - Tests for NPE inference utilities
- `test_analysis.py` - Tests for aggregation and visualization
- **`test_npe_workflow.py`** - NEW: Tests for NPE workflow utilities (initialization, training, sampling)
- **`test_aggregation.py`** - NEW: Tests for ResultsAggregator enhancements (scale selection, bin ranges, filtering)

### Integration Tests
- `test_bnt_aggregation.sh` - Test BNT aggregation workflow
- `test_bootstrap_uncertainty.py` - Test bootstrap uncertainty calculations
- `test_cross_ps_aggregation.py` - Test cross power spectrum aggregation
- `test_tarp_installation.py` - Test TARP installation and imports

### Verification Scripts
- `verify_bootstrap_fix.py` - Verify bootstrap bug fixes

## Running Tests

### Shell Scripts
```bash
cd tests
bash test_bnt_aggregation.sh
```

### Python Tests
```bash
cd tests
python test_bootstrap_uncertainty.py
python test_cross_ps_aggregation.py
python test_tarp_installation.py
```

### Unit Tests with pytest
Run all unit tests:
```bash
cd /home/tersenov/software/bar_impact
pytest tests/unit/ -v
```

Run specific test modules:
```bash
pytest tests/unit/test_npe_workflow.py -v
pytest tests/unit/test_aggregation.py -v
```

Run with coverage:
```bash
pytest tests/unit/ --cov=bar_impact --cov-report=html
```

## Adding New Tests

When adding functionality to `src/bar_impact/`, add corresponding tests here:
1. Create `test_<module_name>.py` following pytest conventions
2. Test both expected behavior and edge cases
3. Use fixtures for common test data
4. Aim for good code coverage

## Test Data

If tests require data files:
- Use small, synthetic test datasets
- Place test data in `tests/data/` 
- Document data requirements in test docstrings

## Recent Updates (January 2026)

### New Test Modules

#### `test_npe_workflow.py`
Comprehensive tests for `bar_impact.utils.npe_workflow` module:
- **NPE Initialization**: Tests for `initialize_npe()` function
- **Training/Loading**: Tests for `train_or_load_npe()` with NaN-retry support
- **Triangle Plots**: Tests for `create_triangle_plot()` with custom configs
- **Posterior Sampling**: Tests for `sample_and_save_posterior()` workflow
- **Config Validation**: Tests for `STANDARD_COSMO_PARAMS`
- **Helper Functions**: Print functions and JAX environment setup

Coverage: Basic functionality, custom configurations, error handling, file I/O, mocking

#### `test_aggregation.py`
Extensive tests for enhanced `ResultsAggregator` class:
- **Scale Selection**: `select_scales()`, `select_scales_per_bin()` methods
  - Single/multiple scales, flattened/3D inputs, per-bin configs
- **Bin Range Selection**: `select_bin_range()`, `select_bin_ranges_per_bin()`
  - Different ranges per bin, boundary conditions
- **Zero-Variance Filtering**: `filter_zero_variance()` with thresholds
  - Mask return, edge cases (all/none filtered)
- **Integration Tests**: Combined pipeline operations
- **Verbosity Control**: Config-based output testing

These tests validate the v3 NPE inference scripts and new workflow utilities.

## Test Coverage

- ✅ Core data processing
- ✅ Utility functions  
- ✅ NPE workflow utilities (NEW)
- ✅ Data aggregation enhancements (NEW)
- 🔄 Visualization (partial)
- 🔄 End-to-end pipelines (in progress)

