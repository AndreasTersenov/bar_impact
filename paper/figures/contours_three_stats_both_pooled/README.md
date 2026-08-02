# Null and biased overlaid, three statistics, 14000 deg2

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_both_14000`
- **generator commit**: `9c33642`
- **generated**: 2026-07-30T08:58:37+00:00
- **published**: 2026-08-02T12:03:22+00:00 at repo `489dd05`
- **rows in values.csv**: 18

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10
- **peaks_l1**: wavelet scales1234, submean, new_normalization, noisy s=0.26
- **power_spectrum_lmax**: 1020
- **power_spectrum_lmax_chosen_by**: no upper cut (full resolution)
- **hos_scale_tag**: scales1234
- **threshold_sigma**: 0.3

## Figure of merit

FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)

| contour | n seeds | FoM₃ pooled | FoM₃ per-seed mean ± std |
|---|---|---|---|
| Power spectrum / null | 4 | 3.09e+05 | 3.438e+05 ± 2.99e+04 |
| Power spectrum / biased | 4 | 2.456e+05 | 2.593e+05 ± 3.43e+04 |
| Peak counts / null | 7 | 3.802e+05 | 4.439e+05 ± 9.1e+04 |
| Peak counts / biased | 7 | 4.028e+05 | 4.708e+05 ± 7.61e+04 |
| L1 norm / null | 4 | 8.309e+05 | 8.919e+05 ± 9.16e+04 |
| L1 norm / biased | 4 | 7.484e+05 | 7.896e+05 ± 7.8e+04 |

fom3_pooled is computed from the pooled samples, i.e. from the covariance the DRAWN contour represents; pooling across NPE training seeds folds training scatter into the covariance and therefore LOWERS the FoM. fom3_per_seed_mean is the mean of the per-seed FoM and is what plot_fom_vs_area.py and plot_scaling_vs_area.py plot, so it is the value to use when comparing against those figures. Do not compare a pooled value against a per-seed one.

## Caveats (from provenance)

- Runs are used only as null/biased PAIRS; a run unreadable on either side is dropped from both, so the null-to-biased offset is like-for-like. See runs_dropped for what the disk failure removed.
- Contours pool all surviving NPE training seeds, so their width includes the seed-to-seed training scatter, not just the posterior width of one seed. --single-run gives the single-seed version.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) predates the lmin 100->37 recovery and the submean correction, so it is NOT numerically comparable to this figure.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
