# Peak counts: BNT vs scale cut vs all scales (baryonified)

- **source**: `outputs/plots/hos_bnt_triangle/hos_bnt_peaks_baryonified`
- **generator commit**: `a3f63d23f6c425f005185ba5e2ba2fb219773eeb`
- **generated**: 2026-08-01T20:54:47Z
- **published**: 2026-08-02T08:10:17+00:00 at repo `cd90af1`
- **rows in values.csv**: 3

## Scales included

- **all**: scales1234 = internal 0,1,2,3 = all four wavelet scales
- **cut**: scales234 = internal 1,2,3 = three scales, finest dropped; legend [20',40',80']
- **bnt**: bntbins1234 scales1234

## Known gaps

- _values.csv has no seed-count column; n_seeds is the column that reveals a point averaging a different subset than it used to
- provenance is missing 'mplstyle'

## Caveats (from provenance)

- OLD CONVENTION: these NPE runs predate the lmin=37 / monopole-subtraction / MASTER recovery. Do not overlay on current-convention figures without rerunning.
- Single NPE run per arm (not seed-pooled), so the contour width carries the seed-to-seed training scatter of one seed only.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
