# BAR_IMPACT Documentation

Welcome to the BAR_IMPACT documentation! This directory contains comprehensive guides, workflows, and technical documentation for the package.

## 📚 Documentation Structure

### [Workflows](workflows/)
Step-by-step guides for common analysis workflows:
- [BNT Inference Workflow](workflows/BNT_INFERENCE_WORKFLOW.md) - Complete guide for running BNT inference
- [Cross Power Spectrum Workflow](workflows/CROSS_POWER_SPECTRUM_WORKFLOW.md) - Processing cross power spectra
- [Cross Spectra Aggregation Workflow](workflows/WORKFLOW_CROSS_SPECTRA_AGGREGATION.md) - Aggregating cross spectra results
- [Splitting Guide](workflows/SPLITTING_GUIDE.md) - Guide for data splitting strategies

### [TARP Coverage Testing](tarp/)
Documentation for Test of Accuracy with Random Points (TARP) coverage testing:
- [TARP Coverage Testing Guide](tarp/TARP_COVERAGE_TESTING.md) - Introduction and usage
- [TARP Quick Reference](tarp/TARP_QUICK_REFERENCE.md) - Quick reference guide
- [TARP Visual Guide](tarp/TARP_VISUAL_GUIDE.md) - Visual explanations
- [TARP Changes](tarp/TARP_CHANGES.md) - Changelog for TARP integration
- [TARP Scripts Summary](tarp/TARP_ALL_SCRIPTS_SUMMARY.md) - Overview of TARP scripts

### [Bug Fixes & Improvements](bugfixes/)
Historical documentation of bug fixes and improvements:
- [Bootstrap Fix](bugfixes/BOOTSTRAP_FIX.md) - Bootstrap uncertainty calculation fix
- [BNT RNG Fix Summary](bugfixes/BNT_RNG_FIX_SUMMARY.md) - Random number generation fix for BNT
- [RNG Seeding Fix](bugfixes/RNG_SEEDING_FIX.md) - General RNG seeding improvements
- [Cross Spectra Bug Fix](bugfixes/BUG_FIX_CROSS_SPECTRA.md) - Cross power spectrum bug fixes
- [Lmax Handling Fix](bugfixes/LMAX_HANDLING_FIX.md) - Maximum multipole handling improvements

### [Implementation Notes](implementation/)
Technical implementation details:
- [Implementation Summary](implementation/IMPLEMENTATION_SUMMARY.md) - Overall implementation notes

## 🚀 Quick Start

For new users, we recommend:
1. Read the main [README](../README.md) first
2. Follow a relevant workflow guide from [workflows/](workflows/)
3. Check [TARP documentation](tarp/) if you need coverage testing

## 📖 API Documentation

API documentation is available in the source code docstrings. To build HTML documentation:

```bash
# Coming soon - Sphinx documentation setup
```

## 🤝 Contributing

When adding new documentation:
- Place workflow guides in `workflows/`
- Place bug fix documentation in `bugfixes/`
- Place TARP-related docs in `tarp/`
- Place implementation notes in `implementation/`
- Update this index file

## 📝 Documentation Standards

- Use clear, descriptive titles
- Include code examples where appropriate
- Add links to related documentation
- Keep content up-to-date with code changes
