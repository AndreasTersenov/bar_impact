# TARP Dependency Information

## What is TARP?

TARP (Test of Accuracy with Random Points) is a Python package for performing statistical coverage tests to assess the quality of posterior estimators in simulation-based inference.

**Repository**: https://github.com/Ciela-Institute/tarp  
**Documentation**: https://tarp.readthedocs.io/  
**PyPI**: https://pypi.org/project/tarp/

## Installation

TARP can be installed via pip:

```bash
pip install tarp>=0.1.3
```

Or install with coverage testing support:

```bash
pip install "bar_impact[coverage]"
```

## Usage in BAR_IMPACT

TARP is used in this project for validating Neural Posterior Estimation (NPE) results. It helps ensure that posterior samples correctly cover the true parameter values.

### Scripts Using TARP

The following scripts in this repository use TARP:

- `scripts/run_npe_inference.py` - Main NPE inference with optional coverage testing
- `scripts/run_npe_inference_ps.py` - Power spectrum NPE with coverage
- `scripts/run_npe_peak_counts_inference.py` - Peak counts NPE with coverage
- `scripts/visualize_coverage_results.py` - Visualize TARP coverage results

### Example Usage

```python
from tarp import get_tarp_coverage
import numpy as np

# After running NPE and obtaining posterior samples
posterior_samples = npe.sample(observed_data, num_samples=10000)
true_parameters = np.array([0.3, 0.8, 0.7])  # True cosmology

# Compute TARP coverage
coverage = get_tarp_coverage(
    posterior_samples,
    true_parameters,
    references='random'  # or 'random_points'
)
```

## TARP in This Repository

### Subdirectory Structure

The `tarp/` subdirectory in this repository contains a **local copy** of the TARP package. This is **not recommended** for production use and exists for development purposes only.

**Recommendation**: Remove the `tarp/` subdirectory and install TARP via pip instead:

```bash
# Remove local copy
rm -rf tarp/

# Install from PyPI
pip install tarp>=0.1.3
```

### Why Not Include TARP Source?

1. **Maintenance**: TARP is actively developed and should be installed as a dependency
2. **Version Control**: Using pip ensures you get updates and bug fixes
3. **Clean Separation**: TARP is an independent package, not part of BAR_IMPACT
4. **Repository Size**: Reduces repository bloat

## Documentation

Comprehensive TARP documentation for this project is available in:

- `docs/tarp/TARP_COVERAGE_TESTING.md` - Usage guide
- `docs/tarp/TARP_QUICK_REFERENCE.md` - Quick reference
- `docs/tarp/TARP_VISUAL_GUIDE.md` - Visual explanations

## References

If you use TARP in your research, please cite:

```bibtex
@article{lemos2023tarp,
  title={Sampling-Based Accuracy Testing of Posterior Estimators for General Inference},
  author={Lemos, Pablo and Coogan, Adam and others},
  journal={arXiv preprint arXiv:2302.03026},
  year={2023}
}
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'tarp'`:

```bash
pip install tarp
```

### Version Issues

Check your TARP version:

```python
import tarp
print(tarp.__version__)
```

Ensure you have at least version 0.1.3:

```bash
pip install --upgrade tarp
```

### Local vs Installed

If you have both a local `tarp/` directory and an installed version, Python may import the wrong one. Remove the local copy:

```bash
rm -rf tarp/
```

## Integration with BAR_IMPACT

TARP is an **optional dependency**. BAR_IMPACT will work without it, but coverage testing features will not be available.

To enable full functionality:

```bash
# Install with all optional dependencies
pip install "bar_impact[all]"

# Or just coverage testing
pip install "bar_impact[coverage]"
```
