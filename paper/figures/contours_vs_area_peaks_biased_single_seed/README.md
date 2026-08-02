# PAPER: Peak-count biased posteriors, all survey areas overlaid (single seed)

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_peaks_biased_l37-1020_single_seed`
- **generator commit**: `d8384e9`
- **generated**: 2026-07-30T15:47:08+00:00
- **published**: 2026-08-02T12:03:22+00:00 at repo `489dd05`
- **rows in values.csv**: 18

## Scales included

- **peaks_l1**: wavelet scales1234 — all four detail scales, coarse/mass-sheet excluded — submean (footprint-mean-subtracted) maps, new_normalization, noisy sigma_e=0.26

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| 2000 | 7 | 3.091e+04 | 3.356e+04 ± 3.61e+03 |
| 5000 | 7 | 1.206e+05 | 1.432e+05 ± 1.35e+04 |
| 10000 | 9 | 2.506e+05 | 3.175e+05 ± 6.07e+04 |
| 14000 | 7 | 4.028e+05 | 4.708e+05 ± 7.61e+04 |
| 28000 | 9 | 9.252e+05 | 1.165e+06 ± 2.41e+05 |
| 35000 | 9 | 1.114e+06 | 1.415e+06 ± 2.44e+05 |

fom3_pooled is computed from the pooled samples, i.e. from the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and therefore LOWERS the FoM. fom3_per_seed_mean is the mean of the per-seed FoM and is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot, so it is the value to use when comparing against those figures. Do not compare a pooled value against a per-seed one.

## Caveats (from provenance)

- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data unaffected.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
