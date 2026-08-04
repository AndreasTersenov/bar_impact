# contours vs area l1 biased single seed with full sky

Same as contours_vs_area_l1_biased_single_seed but with a seventh, full-sky contour (black). Peaks has no counterpart: no full-sky peak-count posterior exists.

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_l1_biased_l37-1020_with_fullsky_single_seed`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T15:52:47+00:00
- **published**: 2026-08-04T17:09:04+00:00 at repo `a57658d9`
- **rows in values.csv**: 21

## Scales included

- **peaks_l1**: wavelet scales1234 — all four detail scales, coarse/mass-sheet excluded — submean (footprint-mean-subtracted) maps, new_normalization, noisy sigma_e=0.26

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| 2000 | 6 | 3.996e+04 | 5.803e+04 ± 2.38e+04 |
| 5000 | 9 | 2.548e+05 | 2.767e+05 ± 2.85e+04 |
| 10000 | 8 | 4.891e+05 | 5.737e+05 ± 7.97e+04 |
| 14000 | 5 | 7.338e+05 | 7.961e+05 ± 7.1e+04 |
| 28000 | 7 | 1.897e+06 | 2.174e+06 ± 4.01e+05 |
| 35000 | 9 | 1.916e+06 | 2.668e+06 ± 3.15e+05 |
| fullsky | 1 | 4.312e+06 | 4.312e+06 ± 0 |

fom3_pooled comes from the pooled samples, i.e. the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and so LOWERS the FoM. fom3_per_seed_mean is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot. Do not compare a pooled value against a per-seed one across figures.

## Known gaps

- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- FULL-SKY CONTOUR IS UNDER-SEEDED: 1 seed(s) against a median of 8 for the masked areas. Pooling adds training scatter, so this contour is tighter than a like-for-like full-sky measurement would be, for a reason unrelated to survey area. Do not read its size as constraining power. FoM3 scales as area^1.5, which is the check to apply.
- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
