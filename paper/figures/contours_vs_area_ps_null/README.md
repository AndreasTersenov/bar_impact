# PS null posteriors, all survey areas overlaid, full resolution

Power-spectrum null (nobaryons) posteriors for all six masked footprints plus full sky, overlaid, at full map resolution ell=37-1020. Sequential light-to-dark ramp encodes survey area, which is an ordered quantity. CAVEAT: the full-sky contour comes from the healpy per-ell pipeline with no submean and no MASTER decoupling -- submean subtracts a FOOTPRINT mean, which exists only under a mask -- so it is not magnitude-comparable to the masked contours.

- **source**: `outputs/plots/contours_vs_area/contours_vs_area_ps_null_l37-1020_with_fullsky`
- **generator commit**: `7cfd75b`
- **generated**: 2026-07-30T10:55:03+00:00
- **published**: 2026-07-30T11:14:26+00:00 at repo `7cfd75b`
- **rows in values.csv**: 21

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10

## Caveats (from provenance)

- FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the higher-order statistics, this is the regime where the baryon bias is significant; the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.
- Contours pool all readable, non-collapsed seeds for the given role, so their width includes NPE seed-to-seed training scatter, not the posterior width of one seed. n_seeds per area is in the values CSV and varies with disk damage.
- Seeds are NOT matched between roles here (each role pools its own readable set), because these are single-role figures. Use plot_contours_three_stats.py, which pairs runs, for any null-to-biased comparison.
- Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the corrected mask treatment. The pre-submean products spuriously tighten the masked posteriors; the glob requires '_submean_' so a missing file surfaces as missing rather than being silently substituted.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data unaffected.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
