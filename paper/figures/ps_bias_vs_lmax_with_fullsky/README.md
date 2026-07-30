# PS baryon bias vs cut, 6 footprints plus full sky

As the masked figure with a seventh full-sky panel. NOTE the full-sky panel uses the healpy 10-ell-bin pipeline while the masked panels use NaMaster nlb=4, so the scale-cut TREND is comparable across panels but the magnitudes are not.

- **source**: `outputs/plots/ps_submean_l37/nsigma_vs_lmax_with_fullsky`
- **generator commit**: `b91af33`
- **generated**: 2026-07-29T22:10:38+00:00
- **published**: 2026-07-30T07:31:34+00:00 at repo `b900627`
- **rows in values.csv**: 132

## Caveats (from provenance)

- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- Read from the campaign's aggregated table, which survived intact (n=5/5 on every row, 0 exclusions) — so unlike nsigma_vs_area these points are NOT re-averaged over a damaged subset and should reproduce the pre-crash figure exactly.
- The 6-param tables of this campaign are zero-length disk-failure casualties; only the 3-param subset is available.
- Threshold crossings are first-upcrossing linear interpolations; these curves are non-monotonic in places (2000 deg2 dips at lmax 940), so a global or last crossing would give a larger, less conservative lmax.
- Full sky uses the healpy 10-ell-bin pipeline vs the masked NaMaster nlb=4 (40-ell); the trend is comparable, the magnitude is not.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
