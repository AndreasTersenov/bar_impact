# contours vs area peaks null pooled

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_peaks_null_l37-1020_pooled`
- **generator commit**: `unknown`
- **generated**: 2026-08-03T14:15:54+00:00
- **published**: 2026-08-03T21:10:33+00:00 at repo `9ef3572`
- **rows in values.csv**: 18

## Scales included

- **peaks_l1**: wavelet scales1234 — all four detail scales, coarse/mass-sheet excluded — submean (footprint-mean-subtracted) maps, new_normalization, noisy sigma_e=0.26

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| 2000 | 6 | 3.178e+04 | 3.327e+04 ± 3.56e+03 |
| 5000 | 10 | 9.329e+04 | 1.01e+05 ± 2.26e+04 |
| 10000 | 10 | 2.567e+05 | 2.842e+05 ± 4.89e+04 |
| 14000 | 9 | 3.801e+05 | 4.361e+05 ± 8.35e+04 |
| 28000 | 8 | 1.012e+06 | 1.201e+06 ± 3.76e+05 |
| 35000 | 10 | 1.494e+06 | 1.748e+06 ± 5.12e+05 |

fom3_pooled comes from the pooled samples, i.e. the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and so LOWERS the FoM. fom3_per_seed_mean is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot. Do not compare a pooled value against a per-seed one across figures.

## Known gaps

- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
