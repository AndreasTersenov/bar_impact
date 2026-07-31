# PAPER: PS biased posteriors, all survey areas overlaid (single seed)

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_ps_biased_l37-1020_with_fullsky_single_seed`
- **generator commit**: `d8384e9`
- **generated**: 2026-07-30T15:29:27+00:00
- **published**: 2026-07-31T06:28:29+00:00 at repo `29261c9`
- **rows in values.csv**: 21

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| 2000 | 5 | 2.673e+04 | 2.722e+04 ± 1.42e+03 |
| 5000 | 5 | 7.972e+04 | 8.62e+04 ± 7.06e+03 |
| 10000 | 3 | 1.997e+05 | 2.097e+05 ± 5.36e+03 |
| 14000 | 5 | 2.449e+05 | 2.592e+05 ± 3.07e+04 |
| 28000 | 9 | 3.443e+05 | 4.367e+05 ± 5.23e+04 |
| 35000 | 3 | 5.067e+05 | 6.141e+05 ± 5.64e+04 |
| fullsky | 5 | 4.647e+05 | 6.489e+05 ± 1.35e+05 |

fom3_pooled is computed from the pooled samples, i.e. from the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and therefore LOWERS the FoM. fom3_per_seed_mean is the mean of the per-seed FoM and is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot, so it is the value to use when comparing against those figures. Do not compare a pooled value against a per-seed one.

## Caveats (from provenance)

- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data unaffected.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
