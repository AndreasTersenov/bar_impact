# contours three stats biased pooled

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_biased_14000`
- **generator commit**: `unknown`
- **generated**: 2026-08-03T14:15:48+00:00
- **published**: 2026-08-04T15:53:47+00:00 at repo `cade6ad`
- **rows in values.csv**: 9

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10
- **peaks_l1**: wavelet scales1234, submean, new_normalization, noisy s=0.26
- **power_spectrum_lmax**: 1020
- **power_spectrum_lmax_chosen_by**: no upper cut (full resolution)
- **hos_scale_tag**: scales1234
- **threshold_sigma**: 0.3

## Known gaps

- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- Runs are used only as null/biased PAIRS; a run unreadable on either side is dropped from both, so the null-to-biased offset is like-for-like. See runs_dropped for what the disk failure removed.
- Contours pool all surviving NPE training seeds, so their width includes the seed-to-seed training scatter, not just the posterior width of one seed. --single-run gives the single-seed version.
- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.
- The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) predates the lmin 100->37 recovery and the submean correction, so it is NOT numerically comparable to this figure.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
