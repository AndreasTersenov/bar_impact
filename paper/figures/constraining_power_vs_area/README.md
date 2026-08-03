# Constraining power vs survey area: sigma(S8) and FoM3

- **source**: `outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats`
- **generator commit**: `9c33642`
- **generated**: 2026-07-30T08:38:01+00:00
- **published**: 2026-08-03T08:56:39+00:00 at repo `c409dc2`
- **rows in values.csv**: 18

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10
- **peaks_l1**: wavelet scales1234 (four detail scales; coarse/mass-sheet excluded), submean, new_normalization, noisy sigma_e=0.26

## Presentation TODO before use

If this figure is used anywhere (paper or referee response) the x-axis tick labels need decluttering first — the log locator currently overlaps them. plot_fom_vs_area.py already solves this with a FixedLocator at 2k/5k/10k/20k/40k plus NullFormatter on the minors; port that here.

## Caveats (from provenance)

- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, the data points are unaffected.
- Damaged posteriors are skipped, so n_seeds differs from the original campaign and each point averages a different subset.
- KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks and L1 are capped at 5 seeds even where 10 posteriors exist. n_seeds records what was used.
- The two reference guides are HARDCODED anchors (0.0135 and 1.05e5 at 14000 deg^2) chosen pre-crash; unlike plot_fom_vs_area.py they are NOT read from the series, so they may no longer sit on the data. They are guides to the SLOPE, not fits.
- FULL RESOLUTION — the regime where all three statistics are baryon-BIASED.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
