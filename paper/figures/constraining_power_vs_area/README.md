# Constraining power vs survey area: sigma(S8) and FoM3

Null-arm constraining power versus survey area for the three statistics, with fitted log-log slopes. Panel (a) sigma(S8), panel (b) FoM3. The two grey guides are HARDCODED pre-crash anchors, not fits -- read them as slope references only. Full resolution, i.e. the regime where all three statistics are baryon-biased.

- **source**: `outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats`
- **generator commit**: `b900627`
- **generated**: 2026-07-30T07:35:37+00:00
- **published**: 2026-07-30T07:36:02+00:00 at repo `b900627`
- **rows in values.csv**: 18

## Caveats (from provenance)

- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, the data points are unaffected.
- Damaged posteriors are skipped, so n_seeds differs from the original campaign and each point averages a different subset.
- KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks and L1 are capped at 5 seeds even where 10 posteriors exist. n_seeds records what was used.
- The two reference guides are HARDCODED anchors (0.0135 and 1.05e5 at 14000 deg^2) chosen pre-crash; unlike plot_fom_vs_area.py they are NOT read from the series, so they may no longer sit on the data. They are guides to the SLOPE, not fits.
- FULL RESOLUTION — the regime where all three statistics are baryon-BIASED.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
