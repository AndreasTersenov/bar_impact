# contours vs area l1 null pooled with full sky

Same as contours_vs_area_l1_null_pooled but with a seventh, full-sky contour (black). Peaks has no counterpart: no full-sky peak-count posterior exists.

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_l1_null_l37-1020_with_fullsky_pooled`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T15:52:47+00:00
- **published**: 2026-08-04T16:27:26+00:00 at repo `d9ca381`
- **rows in values.csv**: 21

## Scales included

- **peaks_l1**: wavelet scales1234 — all four detail scales, coarse/mass-sheet excluded — submean (footprint-mean-subtracted) maps, new_normalization, noisy sigma_e=0.26

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| 2000 | 6 | 6.263e+04 | 6.735e+04 ± 9.44e+03 |
| 5000 | 9 | 2.791e+05 | 2.982e+05 ± 4.21e+04 |
| 10000 | 9 | 4.861e+05 | 6.071e+05 ± 1.45e+05 |
| 14000 | 7 | 8.529e+05 | 9.065e+05 ± 7.52e+04 |
| 28000 | 8 | 2.237e+06 | 2.6e+06 ± 3.08e+05 |
| 35000 | 9 | 2.519e+06 | 3.14e+06 ± 4.34e+05 |
| fullsky | 2 | 3.122e+06 | 3.143e+06 ± 1.99e+05 |

fom3_pooled comes from the pooled samples, i.e. the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and so LOWERS the FoM. fom3_per_seed_mean is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot. Do not compare a pooled value against a per-seed one across figures.

## Known gaps

- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- FULL-SKY CONTOUR IS UNDER-SEEDED: 2 seed(s) against a median of 9 for the masked areas. Pooling adds training scatter, so this contour is tighter than a like-for-like full-sky measurement would be, for a reason unrelated to survey area. Do not read its size as constraining power. FoM3 scales as area^1.5, which is the check to apply.
- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
