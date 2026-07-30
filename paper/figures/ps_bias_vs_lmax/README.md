# PS baryon bias vs upper scale cut, 6 footprints

PAPER FIGURE. Gaussian Q_DM tension (Om, S8, w0) between the nobaryons null and baryonified biased posteriors, versus the power-spectrum upper multipole cut, one panel per masked footprint. The 0.3-sigma line is the baryon-safety tolerance; the dotted vertical marks the crossing. crossings.csv gives BOTH the adoptable cut (last cut still below tolerance) and the crossing (first cut that fails) -- they differ by one grid step and only the former is the cut to use. The diagnostic grey heading was removed for the paper.

- **source**: `outputs/plots/ps_submean_l37/nsigma_vs_lmax`
- **generator commit**: `9c33642`
- **generated**: 2026-07-30T08:37:33+00:00
- **published**: 2026-07-30T08:40:11+00:00 at repo `9c33642`
- **rows in values.csv**: 108

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, rebin=10; upper cut varies along the x-axis over the step-40 grid 340..1020

## Caveats (from provenance)

- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- Read from the campaign's aggregated table, which survived intact (n=5/5 on every row, 0 exclusions) — so unlike nsigma_vs_area these points are NOT re-averaged over a damaged subset and should reproduce the pre-crash figure exactly.
- The 6-param tables of this campaign are zero-length disk-failure casualties; only the 3-param subset is available.
- Threshold crossings are first-upcrossing linear interpolations; these curves are non-monotonic in places (2000 deg2 dips at lmax 940), so a global or last crossing would give a larger, less conservative lmax.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
