# BAR_IMPACT

[![CI](https://github.com/AndreasTersenov/bar_impact/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreasTersenov/bar_impact/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python package for analyzing the impact of baryonic physics on cosmological weak lensing maps using simulation-based inference.

## Overview

BAR_IMPACT provides tools for:

- **Summary Statistics**: L1 norms (wavelet coefficients), angular power spectra, peak counts
- **Tomographic Analysis**: BNT (Bernardeau-Nishimichi-Taruya) transform for nulling redshift correlations
- **Simulation-Based Inference**: Neural Posterior Estimation (NPE) with JAX
- **Posterior Validation**: TARP coverage testing

## Installation

```bash
# Basic installation
pip install -e .

# With inference dependencies (JAX, jaxili)
pip install -e ".[inference]"

# With development tools
pip install -e ".[dev]"

# All dependencies
pip install -e ".[all]"
```

### External Dependencies

- **[pycs](https://github.com/CosmoStat/cosmostat)** - Required for wavelet L1 norms and peak counts
- **[jaxili](https://github.com/jaxili)** - Required for NPE inference

## Quick Start

```python
from bar_impact.core import ConvergenceMap, SurveyMask
from bar_impact.processing import PowerSpectrumProcessor

# Load convergence map
kappa = ConvergenceMap.from_h5("simulation.h5", bin_number=1)

# Add shape noise and apply survey mask
kappa = kappa.add_shape_noise(sigma_e=0.26)
mask = SurveyMask.create_disk_mask(nside=512, target_area_sqdeg=14000)
kappa = kappa.apply_mask(mask)

# Compute power spectrum
processor = PowerSpectrumProcessor(lmax=1024)
cls = processor.process_single(kappa.data)
```

## Package Structure

```
src/bar_impact/
├── core/           # ConvergenceMap, SurveyMask, DataVector
├── processing/     # L1NormProcessor, PowerSpectrumProcessor, PeakCountProcessor
├── inference/      # NPEInference, CoverageTester
├── analysis/       # Aggregation and visualization
├── utils/          # I/O, noise generation, reproducibility
└── constants.py    # BNT matrices, default parameters
```

## Documentation

- [Installation Guide](docs/installation.rst)
- [Quick Start Tutorial](docs/quickstart.rst)
- [Workflow Guides](docs/workflows/)
- [TARP Coverage Testing](docs/tarp/)

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this package, please cite:

```bibtex
@software{bar_impact,
  author = {Tersenov, Andreas},
  title = {BAR_IMPACT: Baryon Impact Analysis for Weak Lensing},
  url = {https://github.com/AndreasTersenov/bar_impact}
}
```
