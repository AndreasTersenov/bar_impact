# L1-norm null posteriors, all survey areas overlaid, full resolution

L1-norm null (nobaryons) posteriors for the six masked footprints, overlaid, using ALL FOUR detail wavelet scales (scales1234, coarse/mass-sheet excluded) on submean maps with the corrected mask treatment -- not the superseded pre-submean products. Sequential light-to-dark ramp encodes survey area.

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_l1_null_l37-1020`
- **generator commit**: `3d0116e`
- **generated**: 2026-07-30T12:41:18+00:00
- **published**: 2026-07-30T12:42:46+00:00 at repo `3d0116e`
- **rows in values.csv**: 18

## Scales included

- **peaks_l1**: wavelet scales1234 — all four detail scales, coarse/mass-sheet excluded — submean (footprint-mean-subtracted) maps, new_normalization, noisy sigma_e=0.26

## Caveats (from provenance)

- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data unaffected.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
