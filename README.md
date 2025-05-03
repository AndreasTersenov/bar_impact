# BAR_IMPACT: Baryon Impact Analysis for Cosmological Maps

This repository contains tools to analyze the impact of baryons on cosmological weak lensing maps through wavelet-based L1 norm calculations. The primary focus is on processing HEALPix maps from cosmological simulations to quantify the differences between baryonified and dark matter only (DMO) simulations.

## Overview

Baryonic processes can significantly impact cosmological observables, particularly weak lensing convergence (κ) maps. This repository provides scripts to:

1. Process large sets of cosmological simulation outputs
2. Apply shape noise to simulate observational conditions
3. Compute wavelet-based L1 norms that help quantify baryon-induced features
4. Analyze the statistical properties of these features across different scales
5. Perform Neural Posterior Estimation (NPE) to infer cosmological parameters

## Requirements

- Python 3.7+
- numpy
- healpy
- h5py
- tqdm
- pycs (Cosmostat library containing MRS wavelet tools)
- jaxili (JAX Implicit Likelihood Inference library)
- jax
- getdist (for visualization)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AndreasTersenov/bar_impact.git
cd bar_impact
```

2. Ensure you have the required dependencies:
```bash
pip install numpy healpy h5py tqdm jax jaxlib getdist
```

3. Install pycs following the instructions at the [CosmoStat repository](https://github.com/CosmoStat/cosmostat).

## Primary Script: L1 Norm Processing

The main script `scripts/l1_norm_processing.py` processes HEALPix convergence maps to extract L1 norms of wavelet coefficients.

### Usage

```bash
python scripts/l1_norm_processing.py [options]
```

### Key Options

```
--fiducial                Process fiducial cosmology instead of grid cosmologies
--base-dir DIR            Override default base directory for data
--baryonified             Use baryonified maps instead of nobaryons maps
--bin-number BIN          Bin number for redshift bin (default: 1)
--noise-level LEVEL       Shape noise level (sigma_e, default: 0.26)
--no-noise                Don't add shape noise to maps
--min-snr MIN             Minimum SNR value (default: -0.003)
--max-snr MAX             Maximum SNR value (default: 0.003)
--num-workers N           Number of worker processes (default: 70)
--shared-temp             Use a shared temporary directory
--verbose                 Print detailed progress information
--save-combined           Save combined L1 norms to a single file
--combined-output FILE    Path for combined output file
```

### Examples

Process fiducial cosmology maps with default settings:
```bash
python scripts/l1_norm_processing.py --fiducial
```

Process grid cosmology maps with baryons, no noise:
```bash
python scripts/l1_norm_processing.py --baryonified --no-noise
```

Process bin 2 maps with custom noise level and save combined results:
```bash
python scripts/l1_norm_processing.py --bin-number 2 --noise-level 0.3 --save-combined
```

## NPE Inference Script

After generating L1 norms, use the `run_npe_inference.py` script to perform Neural Posterior Estimation on the CosmoGRID simulations:

### Usage

```bash
python scripts/run_npe_inference.py [options]
```

### Key Options

```
# Data configuration
--data-dir DIR             Base directory for data (default: /home/tersenov/CosmoGridV1/stage3_forecast)
--simulation-type TYPE     Type of simulation to use for training: baryonified or nobaryons (default: baryonified)

# Analysis configuration
--bin BIN                  Which redshift bin to analyze (default: 2)

# Scale selection (mutually exclusive options)
--scale SCALE              Which scale index to analyze, 0-indexed (default: 0). Use for single scale analysis.
--scales SCALES            Comma-separated list of scale indices to analyze (0-indexed). Use for multi-scale analysis.

--noisy                    Use noisy datavectors
--noise-level LEVEL        Noise level when using noisy data (default: 0.26)

# Fiducial configuration
--fiducial-type TYPE       Type of fiducial data: baryonified or nobaryons (default: matches --simulation-type)

# Training parameters
--train                    Train model (if not specified, will try to load existing model)
--checkpoint-dir DIR       Directory to save/load model checkpoints (default: ./checkpoints)
--epochs N                 Number of training epochs (default: 1000)
--batch-size N             Training batch size (default: 40)
--learning-rate LR         Learning rate (default: 1e-4)

# Sampling parameters
--num-samples N            Number of posterior samples to generate (default: 3000)
--random-seed SEED         Random seed for sampling (default: 1)

# Output parameters
--output-dir DIR           Directory to save output plots (default: /home/tersenov/software/bar_impact/outputs/plots)
--samples-dir DIR          Directory to save posterior samples (default: /home/tersenov/software/bar_impact/outputs/samples)

# GPU configuration
--gpu INDEX                GPU index to use (default: 0)
```

### Examples

Basic training with default settings (baryonified simulations, bin 2, scale 0):
```bash
python scripts/run_npe_inference.py --train
```

Train model with nobaryons simulations and baryonified fiducial to test bias:
```bash
python scripts/run_npe_inference.py --simulation-type nobaryons --fiducial-type baryonified --train
```

Run inference on a different bin and scale with noisy data:
```bash
python scripts/run_npe_inference.py --bin 3 --scale 2 --noisy --train
```

Use a pre-trained model to generate more posterior samples:
```bash
python scripts/run_npe_inference.py --bin 2 --scale 0 --num-samples 10000
```

## Technical Details

### Shape Noise Addition

The code simulates realistic observational conditions by adding shape noise based on:
- Galaxy number density (default: 6.75 galaxies/arcmin²)
- Intrinsic ellipticity dispersion (σₑ, default: 0.26)
- HEALPix resolution (nside=512)

```python
def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise
```

### Wavelet L1 Norm Calculation

The L1 norm calculation follows these steps:
1. Apply undecimated wavelet transform to the spherical map
2. For each wavelet scale:
   - Normalize coefficients to the same energy level
   - Bin the coefficients by SNR values
   - Calculate L1 norm for each bin
3. Return the L1 norms for all scales and bins

### Neural Posterior Estimation

The NPE process follows these steps:
1. Load L1 norm data and cosmological parameters
2. Train a neural density estimator using JAX and the jaxili library
3. Build a posterior distribution from the trained model
4. Sample from the posterior given fiducial data
5. Generate visualizations using getdist

## Analysis Notebooks

The repository also includes Jupyter notebooks for analysis:

- `BNT_systematics.ipynb`: Analysis of Blind Nulling Transform (BNT) systematics
- `systematics.ipynb`: General systematics analysis
- `NPE_clean.ipynb`: Original notebook for Neural Posterior Estimation

## Data Structure

The scripts expect data in a specific format:

```
/path/to/CosmoGridV1/
├── stage3_forecast/
│   ├── fiducial/
│   │   └── cosmo_fiducial/
│   │       ├── perm_0000/
│   │       │   └── projected_probes_maps_*.h5
│   │       ├── perm_0001/
│   │       └── ...
│   └── grid/
│       ├── cosmo_000001/
│       │   ├── perm_0000/
│       │   │   └── projected_probes_maps_*.h5
│       │   └── ...
│       └── ...
```

Each HDF5 file contains weak lensing convergence maps at key path `kg/stage3_lensing{bin_number}`.

## Output

The L1 norm processing script produces `.npy` files containing L1 norm values for each processed map. When using `--save-combined`, it also creates a combined file with all processed L1 norms, with shape `(n_maps, n_scales, n_bins)`.

The NPE inference script produces:
1. Model checkpoints saved in the specified checkpoint directory
2. Triangle plots of posterior distributions saved as PDF files
3. Posterior samples saved as NumPy arrays

## Parallel Processing

The L1 norm processing script utilizes Python's multiprocessing capabilities to efficiently process large sets of maps in parallel, with options to:
- Control the number of worker processes
- Use a shared temporary directory to minimize disk operations
- Monitor progress with a detailed progress bar

## Contact

Andreas Tersenov - atersenov@physics.uoc.gr