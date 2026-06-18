# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BAR_IMPACT** analyzes the impact of baryonic physics on cosmological weak-lensing
convergence maps. It computes summary statistics (wavelet L1 norms, peak counts,
angular power spectra), optionally applies a BNT transform to null tomographic
cross-correlations, and runs Neural Posterior Estimation (NPE) for simulation-based
inference plus downstream tension statistics.

## Where the real work lives: `scripts/`, not `src/`

Read this before trusting the import paths in the rest of the repo.

The project is in a **half-finished refactor**. There are two layers, and they
disagree about how things work:

1. **`scripts/` — the source of truth.** The analyses that actually run are
   *standalone* scripts. They import third-party libraries directly (`healpy`,
   `h5py`, `pycs`, `jaxili`, `pymaster`, `tensiometer`, `getdist`, `tarp`) and do
   **not** use the `bar_impact` package. Constants like the BNT matrix are
   hard-coded inside each script (e.g. `BNT_MATRIX = np.array([...])` in
   `bnt_l1_norm_processing.py`). Only 2 of ~49 scripts import the library.

2. **`src/bar_impact/` — an aspirational library** extracted from the scripts but
   never completed. Treat it as scaffolding, not canonical. Specifically:
   - `core/` is **empty** (no modules).
   - `constants.py` **has no source file** — only a stale `__pycache__/*.pyc` and
     an `egg-info/SOURCES.txt` entry remain. So `from bar_impact.constants import …`
     raises `ImportError`. The `*_v2`/`*_template` scripts that depend on it
     (e.g. `bnt_l1_norm_processing_v2.py`) **do not currently import**.

When asked to change analysis behavior, edit the relevant standalone script.
Only touch `src/bar_impact/` if the task is explicitly about the library refactor.

The many root-level `*.md` files (`REFACTORING_*.md`, `MASTER_CONTRADICTIONS.md`,
`NPE_REFACTORING_SUMMARY.md`, `INFERENCE_SCRIPT_ANALYSIS.md`, `QUICK_REFERENCE.py`,
etc.) are artifacts of that incomplete refactor. They are **not authoritative** —
verify against the scripts before believing them.

## The library API (`src/bar_impact/`), as it actually exists

Purely functional — there are **no classes** (`ConvergenceMap`, `SurveyMask`,
`DataVector`, `*Processor`, `*Config` do not exist). Top-level re-exports:

```python
from bar_impact.processing import (
    process_l1_norms, process_power_spectrum, process_peak_counts,
    apply_bnt_transform,        # bnt_transforms.py also: get_bnt_matrix(n_bins=4)
)
from bar_impact.inference import run_npe_inference   # fisher.py: run_fisher_forecast
from bar_impact.analysis import aggregate_results     # aggregation.py, visualization.py
from bar_impact.utils import (
    load_healpy_map, add_shape_noise, find_files, load_results, save_results,
)
```

Module map: `processing/` (l1_norms, power_spectrum, peak_counts, bnt_transforms),
`inference/` (npe, fisher), `analysis/` (aggregation, visualization),
`utils/` (io, noise), plus `cli.py` (entry point `bar-impact`).

## Development Commands

### Installation
```bash
pip install -e ".[all]"             # everything (inference, coverage, dev)
pip install -e ".[inference]"       # JAX + jaxili + getdist for NPE
pip install -e ".[dev]"             # pytest/black/isort/flake8/mypy
```

### Runtime dependencies the scripts need but pip won't install
Several hard imports are **not** in `pyproject.toml` / `requirements.txt`:
- **pycs** (CosmoStat) — wavelet starlet transform + L1/peak statistics
  (`pycs.sparsity.mrs.mrs_starlet.CMRStarlet`,
  `pycs.astro.wl.hos_peaks_l1.get_wtl1_sphere` / `get_wtpeaks_sphere`).
- **jaxili** — NPE (`jaxili.inference.NPE`).
- **pymaster** (NaMaster) — mode-coupling-corrected power spectra in `*_master*` scripts.
- **tensiometer** — tension statistics in `compute_tension_statistics*.py`.
- **tarp** — coverage testing (`[coverage]` extra; see `docs/tarp/`).
- **getdist** — posterior corner plots.

### Testing
Tests are effectively flat under `tests/` — a `tests/unit/` directory exists but
is **empty** (only `__pycache__`), so point `pytest` at `tests/` itself. Coverage
flags are baked into `pyproject.toml` `addopts`, so plain `pytest` already reports
coverage.
```bash
pytest tests/ -v
pytest tests/test_cross_ps_aggregation.py -v   # a real test file
pytest tests/ -k bootstrap
```
Existing tests: `test_bootstrap_uncertainty.py`, `test_combined_file_loading.py`,
`test_cross_ps_aggregation.py`, `test_tarp_installation.py`, `verify_bootstrap_fix.py`.

### Linting / formatting (line-length 88, isort black profile)
```bash
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/
mypy src/bar_impact/
```

## Scripts: conventions and real entry points

Naming conventions in `scripts/`:
- `bnt_*` — applies a BNT transform (matrix hard-coded in the script).
- `*_master*` — NaMaster mode-coupling correction for power spectra.
- `*_halofit*` — uses Halofit theory predictions.
- `*_v2` / `*_template` — the library-based refactor (depend on the missing
  `bar_impact.constants`; **currently broken** — see above).
- `archive/` — deprecated.
- `diagnostics/` — ad-hoc investigation scripts, **not** part of the production
  pipeline. This is where the active research questions live: Fisher constraining
  power (`fisher_constraining_power.py`, `verify_fisher_constraining_power.py`),
  PS-vs-HOS `w0` degeneracy (`fisher_ps_vs_hos_degeneracy.py`,
  `analyze_hos_w0_origin.py`), low-ell mean leakage (`ps_lowell_mean_leakage_diagnostic.py`),
  mean-subtraction gate (`fisher_gate_mean_subtraction.py`, `regression_submean_off.py`),
  full-sky healpy control, and the Fisher/getdist plotting scripts. Read these to
  understand *why* a convention exists before changing it; don't treat them as APIs.

Actual primary workflows (these files exist and run):
- **L1 norms**: `l1_norm_processing.py` (BNT: `bnt_l1_norm_processing.py`).
- **Peak counts**: `peak_counts_processing.py` (BNT: `bnt_peak_counts_processing.py`).
- **L1 + peaks together**: `l1_peaks_processing.py` (BNT: `bnt_l1_peaks_processing.py`)
  — computes both from a single starlet transform per map (~1.6× faster than the
  two scripts above run separately). Outputs are byte-identical to the standalone
  scripts and use the same filenames, so aggregation/NPE are unaffected. Prefer
  these when you need both statistics on the same maps. Caveat: L1 and peaks then
  share one shape-noise realization per map (the standalone scripts drew
  independent noise) — fine if analyzed separately, but it changes the L1↔peaks
  cross-covariance for a *joint* L1+peaks data vector. The BNT variant reuses the
  non-BNT script's stats core (`compute_l1_and_peaks`), so it depends on
  `l1_peaks_processing.py` being importable from `scripts/`.
- **Power spectra**: `cross_power_spectrum_processing_master.py` (NaMaster);
  aggregate with `aggregate_cross_power_spectra.py` /
  `aggregate_bnt_cross_power_spectra.py`.
- **NPE inference**: `run_npe_inference.py`, `run_npe_peak_counts_inference.py`
  (Fisher: `run_fisher_forecast_*.py`).
- **Tension statistics**: `compute_tension_statistics*.py` (uses `tensiometer`).

Parameter sweeps: `submit_*_parameter_sweep_parallel.py` fan out jobs with
`subprocess.Popen` (local concurrent processes), **not** a scheduler — which suits
a no-Slurm host. Set the concurrency cap inside the script.

Wavelet-HOS performance (`scripts/pycs_speedups.py`): the spherical starlet
(L1 norms + peak counts) is SHT-bound. The L1/peak scripts call
`pycs_speedups.enable(starlet_iter=1)` right after the pycs imports — this
monkeypatches `pycs...mrs_starlet.map2alm` to use `iter=1` instead of the
`iter=3` default (which runs ~7 SHTs/forward call, the original bottleneck) and
caches `hp.get_all_neighbours` for peaks. ~1.5× (L1) / ~1.7× (peaks) with
negligible accuracy change; patches are inherited by the `mp.Pool` fork workers
because `enable()` runs before the Pool is created. **Keep `OMP_NUM_THREADS=1`
for grid runs** (70 workers × 128 cores is the right throughput config; raising
OMP oversubscribes); raise threads only for small/interactive batches. `pymrs`/
pysap C++ bindings are *not* installed and are not the bottleneck (the
"slow python code" warning concerns the unused 2D starlet path). To speed up any
other script using `get_wtl1_sphere`/`get_wtpeaks_sphere`, add the same
`enable()` call after its pycs imports.

Masked HOS treatment (`--mask-correction` in the combined scripts): the standalone
L1/peak scripts (BNT and non-BNT) bin **every pixel on the sphere** (they never
pass `Mask=` to the HOS function), and the BNT scripts add noise *after* masking
(`(κ·mask)+n`), leaving a full noise field outside the footprint. For masked runs
this floods the statistics with noise from the unobserved sky (~3× the peaks at
f_sky≈0.34) — and the BNT case is not reproducible on real data, so it's invalid
for SBI, not just noisy. Full-sky runs are unaffected. `l1_peaks_processing.py`
and `bnt_l1_peaks_processing.py` take `--mask-correction` (default OFF → faithful
reproduction): it (1) measures the statistic only inside the footprint (passes
`Mask=` to `compute_l1_and_peaks`) and (2) for BNT uses order noise→mask→BNT
(outside = 0, reproducible on data). Corrected outputs carry a `_maskcorr` tag so
they never collide with the faithful ones. Verified to recover ≈ f_sky × full-sky
peak counts and to match pycs's own `Mask=` path bit-for-bit. Open refinements
(not yet implemented): per-scale boundary erosion (matters mainly for the coarse
scale) and per-BNT-bin `noise_std` (the BNT matrix amplifies noise ~1.6–1.8× in
bins 3–4, so a single fixed `noise_std` misplaces the SNR bins).

## Important domain notes

### Mask handling — per-statistic, and currently inconsistent
Verify against the specific script you are editing; the conventions differ and
have drifted:
- **L1 norms (`l1_norm_processing.py`)**: pre-multiplies the map by the mask
  (`kg = kg * mask`) and calls `get_wtl1_sphere(kg, …)` with **no** `Mask=` arg.
  ⚠️ A prior version of this file claimed the opposite (pass `Mask=` to pycs, do
  not pre-multiply, to avoid wavelet edge artifacts). The live script pre-multiplies.
  If correctness of masked L1 norms is in question, this is the first thing to check.
- **Peak counts**: multiply the map by the mask **before** `get_wtpeaks_sphere`.
- **BNT (L1 and peaks)**: multiply by the mask, then apply the BNT matrix, then
  process the BNT map without further masking.
- **Power spectra with NaMaster**: do **not** pre-multiply. The map is passed raw
  to `nmt.NmtField(mask, [map_data], …)`; NaMaster applies the mask internally.
  Pre-multiplying double-masks (mask² bias).

### Shape noise
`add_shape_noise()` (and the script copies) uses the convergence formula with **no
factor of 2** — convergence is a scalar field, unlike the two-component shear.

### Apodization
Apodized disk masks use the apodization width as the half-width on **each** side of
the boundary, so the total transition width is `2 × apodization_deg`.

### BNT transform
Lower-triangular matrix nulling cross-correlations between tomographic redshift
bins. Only the 4-bin matrix is used; other configurations need a custom matrix.
In the standalone scripts it is hard-coded, not imported.

### Power-spectrum file keys
Output files use `cls_{i}_{j}` keys (e.g. `cls_1_2` = cross-spectrum of bins 1, 2).

### Mean subtraction (`--subtract-mean` / `_submean`)
`cross_power_spectrum_processing_master.py` takes a `--subtract-mean` flag that
removes the per-patch mean before computing the spectrum and tags outputs with a
`_submean` suffix (it also records `mean_subtracted` in the saved dict). This
addresses low-ell mean ("mass-sheet") leakage and is the subject of active
investigation — the `ps-lowell-mean-leakage` branch and the gate scripts in
`scripts/diagnostics/`. When comparing power-spectrum runs, the `_submean` vs
no-suffix distinction matters; don't mix them.

### Simulation types
Maps come in two variants: `baryonified` (with baryonic feedback) and `nobaryons`
(dark-matter-only). Most scripts process both for comparison; output filenames and
posterior-sample files encode the comparison (e.g. `nobaryons_vs_baryonified`).

## Documentation
- `docs/workflows/` — step-by-step analysis guides (BNT inference, cross power
  spectrum, aggregation).
- `docs/tarp/` — TARP coverage testing.
- `docs/bugfixes/` — historical fixes (shape noise, RNG seeding, bootstrap).
- `docs/implementation/` — implementation summaries for the library refactor.
- `docs/_build/`, `docs/api/` — generated Sphinx output; do not hand-edit.
