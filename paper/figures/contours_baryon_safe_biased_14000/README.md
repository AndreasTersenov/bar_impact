# PAPER: baryon-safe contours, biased observation, three statistics, 14000 deg2

FLAGSHIP PAPER FIGURE. The baryonified observation analysed with a nobaryons-trained model, AFTER each statistic's baryon-safe scale cut: power spectrum restricted to lmax=460 (the largest step-40 cut whose 3-param Q_DM bias stays under 0.3 sigma -- 0.288 sigma; the next cut, 500, gives 0.413), and peaks and L1 with the finest wavelet scale dropped (scales234, i.e. detail scales 2-4 on submean maps). Compare contours_three_stats_biased, the same observation at full resolution, where all three are visibly pulled off truth. Omega_m and S8 now sit on the truth cross; a residual w0 offset remains, which is worth reading off values.csv rather than off the figure.

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales234`
- **generator commit**: `7cfd75b`
- **generated**: 2026-07-30T10:55:03+00:00
- **published**: 2026-07-30T11:13:26+00:00 at repo `7cfd75b`
- **rows in values.csv**: 9

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=460, rebin=10
- **peaks_l1**: wavelet scales234, submean, new_normalization, noisy s=0.26
- **power_spectrum_lmax**: 460
- **power_spectrum_lmax_chosen_by**: largest step-40 cut with mean tension < 0.3 sigma
- **hos_scale_tag**: scales234
- **threshold_sigma**: 0.3

## Caveats (from provenance)

- Runs are used only as null/biased PAIRS; a run unreadable on either side is dropped from both, so the null-to-biased offset is like-for-like. See runs_dropped for what the disk failure removed.
- Contours pool all surviving NPE training seeds, so their width includes the seed-to-seed training scatter, not just the posterior width of one seed. --single-run gives the single-seed version.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) predates the lmin 100->37 recovery and the submean correction, so it is NOT numerically comparable to this figure.
- The PS cut and the HOS cut are different KINDS of cut — multipoles vs a wavelet scale — so they are not ell-matched to each other. Each is the cut that keeps its own statistic under 0.3 sigma. Pass --ps-lmax 400 to instead reproduce the ell-matched pairing used by the Fisher baryon-safe figure.
- The PS lmax is the largest step-40 cut still BELOW 0.3 sigma, not the crossing. The crossing is the first cut that fails (500 at 14000 deg^2); adopting it would put a 0.41-sigma bias in a 'baryon-safe' figure.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
