# FoM3 vs survey area, single panel

The single-panel figure-of-merit version, made for the referee. Unlike the two-panel figure, the A^+3/2 guide here is anchored on the MEASURED PS value at 14000 deg2 rather than a hardcoded constant.

- **source**: `outputs/plots/submean_masked_peaks/fom_vs_area_all_stats`
- **generator commit**: `b900627`
- **generated**: 2026-07-30T07:35:40+00:00
- **published**: 2026-07-30T07:36:02+00:00 at repo `b900627`
- **rows in values.csv**: 18

## Caveats (from provenance)

- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, the data points are unaffected.
- Damaged posteriors are skipped (see [skip] lines in the run log), so n_seeds differs from the original campaign and each point averages a different subset.
- KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks and L1 are capped at 5 seeds even where 10 posteriors exist. Widening it would change the plotted values; n_seeds records what was actually used.
- The A^+3/2 guide is anchored on the MEASURED PS value at 14000 deg^2, read from the series rather than hardcoded, so it stays correct if the inputs change.
- FULL RESOLUTION — the regime where all three statistics are baryon-BIASED. This is constraining power available in principle, not at a baryon-safe cut.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
