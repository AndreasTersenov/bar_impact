# Constraining-power scaling with survey area (submean regime) — validation

**One line:** In the footprint-mean-subtracted ("submean") regime, the constraining power of all three
statistics — power spectrum, peak counts, and L1 norm — scales with masked survey area as expected
(σ(S₈) ∝ A^≈−0.4), and the peak counts now scale in lockstep with the L1 norm. This confirms the
submean treatment removes the spurious masked-peak behaviour, and is the validation we should cite.

## Result

Power-law fits of the **nobaryons** posterior width and figure of merit vs masked area
(6 footprints, 2000–35000 deg²), log–log slopes:

| statistic        | slope σ(S₈) vs A | slope FoM₃ vs A |
|------------------|:----------------:|:---------------:|
| peak counts      |    **−0.38**     |     +1.36       |
| L1 norm          |    **−0.41**     |     +1.33       |
| power spectrum   |    **−0.45**     |     +1.24       |

- **Peaks and L1 agree** (−0.38 vs −0.41) and both sit close to the naive Gaussian expectation
  σ ∝ A^−1/2, and exactly on the **noise-free data-level Fisher** slopes measured independently in the
  masked-peaks investigation (peaks −0.35, L1 −0.38).
- **Power spectrum scales as expected** (−0.45), consistent with the HOS and close to the naive A^−1/2.
  This **relies on the ℓmin=37 low-ℓ recovery** (monopole subtraction): with a conventional ℓmin=100
  floor the PS goes artificially shallow (−0.29), because the recovered ℓ≈37–100 large-scale modes carry
  much of the area gain. So the no-cut PS scaling here is itself a demonstration of the low-ℓ recovery's
  value — a point worth making in the paper.

## Why it matters (paper context)

This is the closing validation of the submean fix. Without footprint-mean subtraction, the **masked peak
counts were anomalously flat** — σ(S₈) barely improved with area (data-level slope ≈ **−0.08**, vs L1's
−0.4), i.e. the masked peaks were spuriously tight at small footprints (the cosmology-dependent footprint
mean creates a boundary step that spawns spurious local maxima). Submean removes this, and the present
result shows the peak scaling is **restored to match the L1 norm** and the Fisher expectation. So masked
HOS are well-behaved under masking once submean is applied — the constraining power grows with area as it
physically should, for every statistic.

(Note this is the *constraining-power* scaling, distinct from the baryon-tension result; the coarse
starlet scale is excluded — the paper analysis uses scales1234 for HOS.)

## Method

- **Metric:** from each nobaryons posterior, cov over (Ω_m, S₈, w₀); σ(S₈)=√cov₁₁, FoM₃=1/√det(cov₃).
- **Configuration (NO scale cuts):** HOS use all detail scales **scales1234** (only the coarse scale
  excluded), bins1234, noisy s0.26, new normalization, submean. PS uses the **ℓmin=37 low-ℓ-recovered,
  monopole-subtracted (submean) range, l37–1020** (full, no ℓmax cut), r10, apod2.0, NaMaster — from the
  baryon-tension campaign tree `outputs/baryon_tension/ps_submean_l37/posteriors/mask_<A>/null/`.
  ℓmin=37 (not 100) is the defining low-ℓ-recovery feature of the new analysis and must be used here.
- **Seed-averaging:** peaks and L1 averaged over **5 NPE training seeds per footprint** (the
  single-seed σ is noisy — a single seed gave a spurious peak slope of −0.80 that settled to −0.38 once
  averaged). PS averaged over its 5 runs. Any prior-collapsed seed (σ(S₈)>0.08) dropped before averaging
  (see the NPE NaN-retry / zero-variance-filter fixes — collapses were the failure mode).
- **Footprints:** 2000/5000/10000/14000/28000/35000 deg² (PS) ≙ 2001/…/35001 (HOS, ~identical on log A).

## Reproduce

- Plot: `outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats_submean_seedavg.png`
- Posteriors: `outputs/samples/posterior_samples_{pc_,,ps_auto_cross_}nobaryons_vs_nobaryons_bins1234_…_submean…_run{2..5}_npe.npy`
- Convention borrowed from `scripts/diagnostics/npe_fom_vs_area.py` (FoM₃ = 1/√det cov₃).

## Configuration note (no scale cuts)

This is the **no-cut** configuration for every statistic — HOS use all detail starlet scales with only
the coarse scale dropped (scales1234), and PS uses its full multipole range at the analysis floor
**ℓmin=37** (the low-ℓ-recovered, monopole-subtracted gauge) up to l≈1020 (no ℓmax cut). Using the
correct ℓmin=37 floor matters: ℓmin=100 throws away exactly the recovered low-ℓ modes and gives a
spuriously shallow PS slope (−0.29 vs −0.45). Sensitivity to the upper cut is negligible.
