# PS baryon bias vs cut, 6 footprints plus full sky

- **source**: `outputs/plots/ps_submean_l37/nsigma_vs_lmax_with_fullsky`
- **generator commit**: `06e07006`
- **generated**: 2026-08-04T17:00:29+00:00
- **published**: 2026-08-04T17:00:48+00:00 at repo `06e07006`
- **rows in values.csv**: 132

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, rebin=10; upper cut varies along the x-axis over the step-40 grid 340..1020

## Caveats (from provenance)

- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.
- Read from the campaign's aggregated table, which survived intact (n=5/5 on every row, 0 exclusions) — so unlike nsigma_vs_area these points are NOT re-averaged over a damaged subset and should reproduce the pre-crash figure exactly.
- The 6-param tables of this campaign are zero-length disk-failure casualties; only the 3-param subset is available.
- Threshold crossings are first-upcrossing linear interpolations; these curves are non-monotonic in places (2000 deg2 dips at lmax 940), so a global or last crossing would give a larger, less conservative lmax.
- Full sky uses the healpy 10-ell-bin pipeline vs the masked NaMaster nlb=4 (40-ell); the trend is comparable, the magnitude is not.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
