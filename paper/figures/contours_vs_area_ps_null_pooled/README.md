# PS null posteriors, all survey areas overlaid, full resolution

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_ps_null_l37-1020_with_fullsky`
- **generator commit**: `7cfd75b`
- **generated**: 2026-07-30T10:55:03+00:00
- **published**: 2026-08-03T09:36:14+00:00 at repo `17afa33`
- **rows in values.csv**: 21

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| 2000 | 5 | 2.632e+04 | 2.725e+04 ± 2.22e+03 |
| 5000 | 5 | 8.76e+04 | 9.052e+04 ± 7.77e+03 |
| 10000 | 5 | 2.179e+05 | 2.292e+05 ± 2.26e+04 |
| 14000 | 4 | 3.09e+05 | 3.438e+05 ± 2.99e+04 |
| 28000 | 10 | 6.495e+05 | 7.166e+05 ± 7.76e+04 |
| 35000 | 4 | 8.741e+05 | 9.658e+05 ± 1.55e+05 |
| fullsky | 5 | 1.104e+06 | 1.391e+06 ± 3.79e+05 |

fom3_pooled is computed from the pooled samples, i.e. from the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and therefore LOWERS the FoM. fom3_per_seed_mean is the mean of the per-seed FoM and is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot, so it is the value to use when comparing against those figures. Do not compare a pooled value against a per-seed one.

## Caveats (from provenance)

- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data unaffected.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
