# BAR_IMPACT: Baryon Impact Analysis for Cosmological Maps

This repository contains tools to analyze the impact of baryons on cosmological weak lensing maps through wavelet-based L1 norm calculations. The primary focus is on processing HEALPix maps from cosmological simulations to quantify the differences between baryonified and dark matter only (DMO) simulations.

## Overview

Baryonic processes can significantly impact cosmological observables, particularly weak lensing convergence (κ) maps. This repository provides scripts to:

1. Process large sets of cosmological simulation outputs
2. Apply shape noise to simulate observational conditions
3. Compute wavelet-based L1 norms that help quantify baryon-induced features
4. Analyze the statistical properties of these features across different scales

## Requirements

- Python 3.7+
- numpy
- healpy
- h5py
- tqdm
- pycs (Cosmostat library containing MRS wavelet tools)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bar_impact.git
cd bar_impact
```

2. Ensure you have the required dependencies:
```bash
pip install numpy healpy h5py tqdm
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

## Analysis Notebooks

The repository also includes Jupyter notebooks for analysis:

- `BNT_systematics.ipynb`: Analysis of Blind Nulling Transform (BNT) systematics
- `systematics.ipynb`: General systematics analysis

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

The script produces `.npy` files containing L1 norm values for each processed map. When using `--save-combined`, it also creates a combined file with all processed L1 norms, with shape `(n_maps, n_scales, n_bins)`.

## Parallel Processing

The script utilizes Python's multiprocessing capabilities to efficiently process large sets of maps in parallel, with options to:
- Control the number of worker processes
- Use a shared temporary directory to minimize disk operations
- Monitor progress with a detailed progress bar


## Contact

Andreas Tersenov - atersenov@physics.uoc.gr