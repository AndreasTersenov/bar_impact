# Null posteriors, three statistics, 14000 deg2

Nobaryons-vs-nobaryons (unbiased) posteriors for the three statistics at the reference footprint, full resolution. Contours pool all surviving NPE training seeds, so their width includes seed-to-seed training scatter.

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_null_14000`
- **generator commit**: `62ebb46`
- **generated**: 2026-07-29T17:52:16+00:00
- **published**: 2026-07-30T07:31:51+00:00 at repo `b900627`
- **rows in values.csv**: 9

## Caveats (from provenance)

- Runs are used only as null/biased PAIRS; a run unreadable on either side is dropped from both, so the null-to-biased offset is like-for-like. See runs_dropped for what the disk failure removed.
- Contours pool all surviving NPE training seeds, so their width includes the seed-to-seed training scatter, not just the posterior width of one seed. --single-run gives the single-seed version.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) predates the lmin 100->37 recovery and the submean correction, so it is NOT numerically comparable to this figure.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
