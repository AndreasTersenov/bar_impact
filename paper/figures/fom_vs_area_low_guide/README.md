# FoM3 vs survey area, guide dropped clear of the data

- **source**: `outputs/plots/submean_masked_peaks/fom_vs_area_all_stats_lowanchor`
- **generator commit**: `9c33642`
- **generated**: 2026-07-30T08:38:13+00:00
- **published**: 2026-08-03T08:56:39+00:00 at repo `c409dc2`
- **rows in values.csv**: 18

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10
- **peaks_l1**: wavelet scales1234 (four detail scales; coarse/mass-sheet excluded), submean, new_normalization, noisy sigma_e=0.26

## Caveats (from provenance)

- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, the data points are unaffected.
- Damaged posteriors are skipped (see [skip] lines in the run log), so n_seeds differs from the original campaign and each point averages a different subset.
- KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks and L1 are capped at 5 seeds even where 10 posteriors exist. Widening it would change the plotted values; n_seeds records what was actually used.
- The A^+3/2 guide is anchored on the MEASURED PS value at 14000 deg^2, read from the series rather than hardcoded, so it stays correct if the inputs change.
- FULL RESOLUTION — the regime where all three statistics are baryon-BIASED. This is constraining power available in principle, not at a baryon-safe cut.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
