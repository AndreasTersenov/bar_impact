# constraining power vs area

- **source**: `outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T16:21:58+00:00
- **published**: 2026-08-04T17:09:03+00:00 at repo `a57658d9`
- **rows in values.csv**: 18

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10
- **peaks_l1**: wavelet scales1234 (four detail scales; coarse/mass-sheet excluded), submean, new_normalization, noisy sigma_e=0.26

## Presentation TODO before use

If this figure is used anywhere (paper or referee response) the x-axis tick labels need decluttering first — the log locator currently overlaps them. plot_fom_vs_area.py already solves this with a FixedLocator at 2k/5k/10k/20k/40k plus NullFormatter on the minors; port that here.

## Known gaps

- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.
- Damaged posteriors are skipped, so n_seeds differs from the original campaign and each point averages a different subset.
- KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks and L1 are capped at 5 seeds even where 10 posteriors exist. n_seeds records what was used.
- The two reference guides are HARDCODED anchors (0.0135 and 1.05e5 at 14000 deg^2) chosen pre-crash; unlike plot_fom_vs_area.py they are NOT read from the series, so they may no longer sit on the data. They are guides to the SLOPE, not fits.
- FULL RESOLUTION — the regime where all three statistics are baryon-BIASED.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
