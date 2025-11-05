# Scripts Directory

This directory contains command-line scripts for processing cosmological data and running analyses.

## 📋 Available Scripts

### Data Processing

#### L1 Norm Processing
- **`l1_norm_processing.py`** - Process convergence maps to compute L1 norms of wavelet coefficients
- **`bnt_l1_norm_processing.py`** - Apply BNT transform and compute L1 norms

#### Power Spectrum Processing
- **`power_spectrum_processing.py`** - Compute angular power spectra
- **`bnt_power_spectrum_processing.py`** - BNT power spectrum processing
- **`cross_power_spectrum_processing.py`** - Cross power spectra between bins
- **`bnt_cross_power_spectrum_processing.py`** - BNT cross power spectra
- **`masked_power_spectrum_processing.py`** - Power spectra with masks

#### Peak Counting
- **`peak_counts_processing.py`** - Count peaks in convergence maps
- **`bnt_peak_counts_processing.py`** - Peak counting with BNT transform

### Data Aggregation
- **`aggregate_cross_power_spectra.py`** - Aggregate cross power spectrum results
- **`aggregate_bnt_cross_power_spectra.py`** - Aggregate BNT cross spectra

### Inference

#### Neural Posterior Estimation (NPE)
- **`run_npe_inference.py`** - Run NPE on L1 norms
- **`run_npe_inference_ps.py`** - Run NPE on power spectra
- **`run_npe_inference_auto_cross_ps.py`** - NPE on auto+cross power spectra
- **`run_npe_peak_counts_inference.py`** - NPE on peak counts

#### Fisher Forecasts
- **`run_fisher_forecast_ps.py`** - Fisher forecast for power spectra
- **`run_fisher_forecast_auto_cross_ps.py`** - Fisher forecast for auto+cross spectra

### Analysis & Visualization
- **`visualize_coverage_results.py`** - Visualize TARP coverage test results
- **`inspect_saved_datavectors.py`** - Inspect saved data vectors
- **`split_auto_cross_for_inference.py`** - Split auto/cross spectra for analysis

## 🚀 Quick Start

### Example: Process L1 Norms

```bash
# Process fiducial cosmology with default settings
python scripts/l1_norm_processing.py --fiducial

# Process with BNT transform
python scripts/bnt_l1_norm_processing.py --fiducial
```

### Example: Run NPE Inference

```bash
# Run inference on L1 norms
python scripts/run_npe_inference.py \\
    --data-file outputs/l1_norms_combined.npz \\
    --output-dir outputs/inference/

# With TARP coverage testing
python scripts/run_npe_inference.py \\
    --data-file outputs/l1_norms_combined.npz \\
    --output-dir outputs/inference/ \\
    --run-coverage
```

## 📚 Documentation

For detailed workflows, see:
- [docs/workflows/](../docs/workflows/) - Step-by-step workflow guides
- [docs/](../docs/) - Full documentation index

## 🗂️ Archive

Old script versions are kept in `archive/` for reference:
- `archive/bnt_l1_norm_processing.py` - Old version
- `archive/l1_norm_processing.py` - Old version
- `archive/run_npe_inference_ps copy.py` - Duplicate file
- `archive/QUICK_COMMANDS.sh` - Old utility script

## ⚙️ Script Organization

Scripts are organized by function:
- **Processing scripts** - Convert raw data to data vectors
- **Aggregation scripts** - Combine results from multiple runs
- **Inference scripts** - Perform statistical inference (NPE, Fisher)
- **Visualization scripts** - Create plots and diagnostics

## 🔧 Future Refactoring

As the package matures, these scripts will be refactored to use the `bar_impact` library:

```python
# Current approach
# Lots of functions defined in script

# Future approach
from bar_impact.processing import process_l1_norms
from bar_impact.inference import run_npe_inference

# Clean, reusable code
```

This refactoring is tracked in TODO Step 3.
