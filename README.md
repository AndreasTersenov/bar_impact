# BAR_IMPACT

**B**aryon **A**nalysis for Cosmological **R**esearch using **I**nference on **M**ap **P**rocessing with **A**nalysis **C**apabilities and **T**ools

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python package for analyzing the impact of baryonic physics on cosmological weak lensing maps through advanced statistical methods including wavelet analysis, power spectra, peak counting, and simulation-based inference.

## ✨ Features

- **📊 Multiple Analysis Methods**
  - Wavelet-based L1 norm calculations
  - Angular power spectrum analysis  
  - Peak counting statistics
  - Band-limited Nulling Transform (BNT)

- **🔬 Statistical Inference**
  - Neural Posterior Estimation (NPE) using JAX
  - Fisher information forecasts
  - TARP coverage testing for posterior validation

- **🗺️ Map Processing**
  - HEALPix convergence map processing
  - Shape noise simulation
  - Multi-scale wavelet decomposition

- **🧮 High Performance**
  - Multiprocessing support
  - Optimized for large simulation datasets
  - Batch processing capabilities

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AndreasTersenov/bar_impact.git
cd bar_impact

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

**External Dependencies:**
- [pycs (CosmoStat)](https://github.com/CosmoStat/cosmostat) - Required for wavelet analysis
- [jaxili](https://github.com/users/jaxili) - Optional, for NPE inference

See [docs/TARP_DEPENDENCY.md](docs/TARP_DEPENDENCY.md) for TARP installation.

### Basic Usage

```python
import numpy as np
from bar_impact.processing import apply_bnt_transform
from bar_impact.utils import load_healpy_map, add_shape_noise

# Load a convergence map
kappa_map = load_healpy_map('path/to/map.fits')

# Add observational noise
noisy_map = add_shape_noise(kappa_map, sigma_e=0.26, nside=512)

# Apply BNT transform to multiple redshift bins
maps = np.array([map_bin1, map_bin2, map_bin3, map_bin4])
bnt_maps = apply_bnt_transform(maps)
```

### Command-Line Scripts

Process L1 norms from convergence maps:
```bash
python scripts/l1_norm_processing.py --fiducial --save-combined
```

Run Neural Posterior Estimation:
```bash
python scripts/run_npe_inference.py \\
    --data-file outputs/l1_norms_combined.npz \\
    --output-dir outputs/inference/ \\
    --run-coverage
```

See [scripts/README.md](scripts/README.md) for all available scripts.

## 📁 Repository Structure

```
bar_impact/
├── src/bar_impact/          # Main package
│   ├── processing/          # Data processing modules
│   ├── inference/           # NPE and Fisher analysis
│   ├── analysis/            # Aggregation and visualization
│   └── utils/               # I/O and utilities
├── scripts/                 # Command-line scripts
├── notebooks/               # Jupyter notebooks
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── workflows/           # Step-by-step guides
│   ├── tarp/                # TARP coverage testing
│   └── bugfixes/            # Historical fixes
├── data/                    # Input data (gitignored)
├── outputs/                 # Results (gitignored)
└── examples/                # Usage examples
```

## 📚 Documentation

- **[Full Documentation](docs/README.md)** - Complete documentation index
- **[Workflows](docs/workflows/)** - Step-by-step analysis guides
- **[TARP Guide](docs/tarp/)** - Coverage testing documentation
- **[Scripts Reference](scripts/README.md)** - All available scripts

### Key Workflows

- [BNT Inference Workflow](docs/workflows/BNT_INFERENCE_WORKFLOW.md)
- [Cross Power Spectrum Analysis](docs/workflows/CROSS_POWER_SPECTRUM_WORKFLOW.md)
- [Data Aggregation](docs/workflows/WORKFLOW_CROSS_SPECTRA_AGGREGATION.md)

## 🔬 Science Background

This package analyzes the impact of baryonic physics on weak gravitational lensing observables. Baryonic processes (gas cooling, star formation, AGN feedback) affect the matter distribution and therefore the lensing signal.

**Key Methods:**
- **Wavelet L1 Norms**: Quantify non-Gaussian features induced by baryons
- **Band-limited Nulling (BNT)**: Decorrelate signals across redshift bins
- **Neural Posterior Estimation**: Infer cosmological parameters accounting for baryonic uncertainties

## 🛠️ Development Status

**Current Version**: 0.1.0 (Alpha)

The package is under active development. The core functionality is implemented in scripts, with ongoing refactoring to create a more modular library structure.

### Roadmap

- [x] Core processing scripts
- [x] Package structure and metadata
- [x] Documentation organization
- [ ] Extract core functionality to library modules
- [ ] Comprehensive test suite
- [ ] API documentation
- [ ] Example notebooks
- [ ] PyPI release

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CosmoStat](https://github.com/CosmoStat) for the pycs wavelet library
- [TARP](https://github.com/Ciela-Institute/tarp) for coverage testing tools
- [JAX](https://github.com/google/jax) ecosystem for numerical computing

## 📧 Contact

**Andreas Tersenov**
- GitHub: [@AndreasTersenov](https://github.com/AndreasTersenov)

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{bar_impact2025,
  author = {Tersenov, Andreas},
  title = {BAR\_IMPACT: Baryon Impact Analysis for Cosmological Maps},
  year = {2025},
  url = {https://github.com/AndreasTersenov/bar_impact}
}
```

---

**Status**: 🚧 Under Active Development | **Last Updated**: November 2025
