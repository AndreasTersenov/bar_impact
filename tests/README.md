# Tests Directory

This directory contains test scripts for validating the BAR_IMPACT package functionality.

## Test Files

### Unit Tests
- (To be added: pytest test files for package modules)

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

### Future: pytest Integration
Once the package is refactored with proper modules, run tests with:
```bash
pytest tests/
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
