# BAR_IMPACT Documentation

Documentation for the BAR_IMPACT package for analyzing baryon impact on cosmological weak lensing maps.

## Documentation Structure

### User Guides

- **[Installation](installation.rst)** - How to install the package
- **[Quick Start](quickstart.rst)** - Getting started with basic usage
- **[API Reference](api/)** - API documentation

### Workflows

Step-by-step guides for common analysis workflows:

- [BNT Inference Workflow](workflows/BNT_INFERENCE_WORKFLOW.md) - Running BNT (nulling) inference
- [Cross Power Spectrum Workflow](workflows/CROSS_POWER_SPECTRUM_WORKFLOW.md) - Processing cross power spectra
- [Cross Spectra Aggregation](workflows/WORKFLOW_CROSS_SPECTRA_AGGREGATION.md) - Aggregating results for inference

### Coverage Testing

- [TARP Coverage Testing](tarp/TARP_COVERAGE_TESTING.md) - Validating posterior quality with TARP
- [TARP Quick Reference](tarp/TARP_QUICK_REFERENCE.md) - Quick reference for TARP commands

### Advanced Topics

- [NPE Inference with Halofit](NPE_INFERENCE_HALOFIT_GUIDE.md) - Using Halofit predictions for inference

## Building Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.
