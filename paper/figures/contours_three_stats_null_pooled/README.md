# Null posteriors, three statistics, 14000 deg2

Nobaryons-vs-nobaryons (unbiased) posteriors for the three statistics at the reference footprint. FULL RESOLUTION -- no scale cut applied; see the Scales included section for the exact multipoles and wavelet scales. Contours pool all surviving NPE training seeds, so their width includes seed-to-seed training scatter.

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_null_14000`
- **generator commit**: `7620563`
- **generated**: 2026-07-30T09:08:02+00:00
- **published**: 2026-07-30T09:18:17+00:00 at repo `7620563`
- **rows in values.csv**: 9

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10
- **peaks_l1**: wavelet scales1234, submean, new_normalization, noisy s=0.26
- **power_spectrum_lmax**: 1020
- **power_spectrum_lmax_chosen_by**: no upper cut (full resolution)
- **hos_scale_tag**: scales1234
- **threshold_sigma**: 0.3

## Caveats (from provenance)

- Runs are used only as null/biased PAIRS; a run unreadable on either side is dropped from both, so the null-to-biased offset is like-for-like. See runs_dropped for what the disk failure removed.
- Contours pool all surviving NPE training seeds, so their width includes the seed-to-seed training scatter, not just the posterior width of one seed. --single-run gives the single-seed version.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) predates the lmin 100->37 recovery and the submean correction, so it is NOT numerically comparable to this figure.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
