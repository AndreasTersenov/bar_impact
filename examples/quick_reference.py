#!/usr/bin/env python3
"""
Quick Reference: Old Scripts vs New Package

This file shows the mapping between your old script commands
and how to achieve the same with the new bar_impact package.

================================================================================
PROCESSING WORKFLOWS
================================================================================

OLD: L1 Norm Processing
-----------------------
    python scripts/l1_norm_processing_new_mask.py \
        --bins 1,2,3,4 --noise-level 0.26 --apply-mask \
        --mask-area-sqdeg 35001.0 --num-workers 40 \
        --save-combined --fiducial --baryonified

NEW: Using bar_impact classes
-----------------------------
"""

# === L1 NORM PROCESSING ===

from bar_impact.core import ConvergenceMap, SurveyMask
from bar_impact.processing import L1NormProcessor, L1NormConfig

# Create mask
mask = SurveyMask.create_disk_mask(
    nside=512,
    target_area_sqdeg=35001.0,
    center_coords=(0.0, 90.0),
)

# Configure processor
config = L1NormConfig(nside=512, n_scales=5, nbins=40)
processor = L1NormProcessor(config=config)

# Process a single map
# (Previously done inside process_file() in the script)
kappa_map = ConvergenceMap(data=kappa_data, nside=512, bin_number=1)
kappa_noisy = kappa_map.add_shape_noise(sigma_e=0.26)
masked_data = kappa_noisy.data * mask.data
l1_norms = processor.process_single(masked_data)


"""
OLD: Power Spectrum Processing (with MASTER)
--------------------------------------------
    python scripts/bnt_cross_power_spectrum_processing_master.py \
        --lmax 1535 --num-workers 20 \
        --apply-mask --mask-area-sqdeg 10000 --apodization-scale-deg 2.0 \
        --save-combined --aggregate-for-inference --fiducial --baryonified

NEW: Using bar_impact classes
-----------------------------
"""

from bar_impact.processing import PowerSpectrumProcessor, PowerSpectrumConfig
from bar_impact.constants import BNT_MATRIX

# Create apodized mask
mask = SurveyMask.create_apodized_mask(
    nside=512,
    target_area_sqdeg=10000.0,
    center_coords=(0.0, 90.0),
    apodization_scale_deg=2.0,
)

# Configure processor
config = PowerSpectrumConfig(nside=512, lmax=1535, use_master=True)
processor = PowerSpectrumProcessor(config=config)
processor.set_mask(mask.data)

# Apply BNT transform
import numpy as np
maps_stacked = np.array([map1.data, map2.data, map3.data, map4.data])
bnt_maps = BNT_MATRIX @ maps_stacked

# Compute power spectrum
cls = processor.process_single(bnt_maps[0])


"""
================================================================================
INFERENCE WORKFLOWS
================================================================================

OLD: NPE Inference (L1 Norms)
-----------------------------
    python scripts/run_npe_inference.py \
        --simulation-type nobaryons --fiducial-type nobaryons \
        --bins 1,2,3,4 --scales 0,1,2,3,4 \
        --noisy --noise-level 0.26 \
        --train --run-coverage-test --gpu 1 --new-normalization

NEW: Using bar_impact classes
-----------------------------
"""

from bar_impact.inference import NPEInference, NPEConfig, CoverageTester, CoverageConfig
from bar_impact.analysis import ResultsAggregator, CoveragePlotter

# Load and prepare data
aggregator = ResultsAggregator()
data = aggregator.load_from_pattern("data/l1_norms_*.npy")
data = aggregator.select_scales(data, scale_indices=[0, 1, 2, 3, 4])

# Configure and train NPE
npe_config = NPEConfig(
    n_features=data.shape[1],
    n_params=2,  # Om, S8
    epochs=1000,
    batch_size=40,
)
npe = NPEInference(config=npe_config)
npe.train(data_normalized, params_normalized, checkpoint_path="./checkpoints/model")

# Sample posterior
import jax.random as random
samples = npe.sample(observation, n_samples=3000, key=random.PRNGKey(42))

# Run coverage test
coverage_config = CoverageConfig(n_sims=100, n_posterior_samples=1000)
tester = CoverageTester(config=coverage_config)
result = tester.run(npe, test_data, true_params)

# Plot coverage
plotter = CoveragePlotter()
fig = plotter.plot_coverage(result.ecp, result.alpha, result.ecp_std)


"""
OLD: NPE Inference (Power Spectra)
----------------------------------
    python scripts/run_npe_inference_auto_cross_ps.py \
        --simulation-type nobaryons --fiducial-type baryonified \
        --bins 1,2,3,4 --lmax 2048 --lower-cut 100 --upper-cut 450 \
        --noisy --noise-level 0.26 --train --gpu 1 --rebin 10

NEW: Using bar_impact classes
-----------------------------
"""

from bar_impact.analysis import aggregate_power_spectra, PosteriorPlotter

# Load power spectra with ell cuts
result = aggregate_power_spectra(
    file_paths=["auto_1.npy", "auto_2.npy", "cross_1_2.npy"],
    ell_range=(100, 450),
)
cls = result["cls"]

# Rebin
def rebin(cls, factor):
    n_bins = cls.shape[1] // factor
    return cls[:, :n_bins * factor].reshape(cls.shape[0], n_bins, factor).mean(axis=2)

cls_rebinned = rebin(cls, factor=10)

# Same NPE workflow as above...
npe = NPEInference(config=npe_config)
npe.train(data_normalized, params_normalized)

# Plot posterior with getdist
plotter = PosteriorPlotter()
fig = plotter.triangle_plot(
    samples,
    param_names=["Om", "S8"],
    param_labels=[r"$\Omega_m$", r"$S_8$"],
)


"""
================================================================================
KEY MAPPINGS
================================================================================

MASKS
-----
Old script functions:
    create_euclid_mask() -> SurveyMask.create_disk_mask()
    create_apodized_mask() -> SurveyMask.create_apodized_mask()

NOISE
-----
Old script functions:
    add_shape_noise() -> ConvergenceMap.add_shape_noise()

BNT TRANSFORM
-------------
Old script constant:
    BNT_MATRIX -> from bar_impact.constants import BNT_MATRIX

POWER SPECTRA
-------------
Old script functions:
    compute_auto_spectrum() -> PowerSpectrumProcessor.process_single()
    compute_cross_spectrum() -> PowerSpectrumProcessor.compute_cross_spectrum()

L1 NORMS
--------
Old script functions:
    get_wtl1_sphere() -> L1NormProcessor.process_single() (wraps pycs)

NPE INFERENCE
-------------
Old script usage:
    from jaxili.inference import NPE -> from bar_impact.inference import NPEInference
    
    npe = NPE(...)
    npe.train(...)
    ->
    npe = NPEInference(config=NPEConfig(...))
    npe.train(...)

COVERAGE TESTING
----------------
Old script usage:
    from tarp import get_tarp_coverage
    ->
    from bar_impact.inference import CoverageTester
    tester = CoverageTester(config=CoverageConfig(...))
    result = tester.run(npe, test_data, true_params)

PLOTTING
--------
Old script usage:
    from getdist import plots, MCSamples
    samples = MCSamples(...)
    g = plots.get_subplot_plotter()
    g.triangle_plot(...)
    ->
    from bar_impact.analysis import PosteriorPlotter
    plotter = PosteriorPlotter()
    fig = plotter.triangle_plot(samples, ...)

================================================================================
RUNNING THE EXAMPLES
================================================================================

To run the example scripts:

    # L1 norm processing
    python examples/example_l1_norm_processing.py
    
    # Power spectrum processing  
    python examples/example_power_spectrum_processing.py
    
    # NPE inference with L1 norms
    python examples/example_npe_inference_l1.py
    
    # NPE inference with power spectra
    python examples/example_npe_inference_power_spectrum.py

Or import and use as a library:

    from bar_impact.core import ConvergenceMap, SurveyMask
    from bar_impact.processing import L1NormProcessor, PowerSpectrumProcessor
    from bar_impact.inference import NPEInference, CoverageTester
    from bar_impact.analysis import ResultsAggregator, PosteriorPlotter

"""

if __name__ == "__main__":
    print(__doc__)
