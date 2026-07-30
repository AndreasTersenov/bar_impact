# APPENDIX: full-sky baryon-safe contours, three statistics, PS lmax=340

FULL SKY, all three statistics at MATCHED marginally-tolerated bias -- the symmetric comparison. PS lmax=340 (0.358+/-0.099), peaks scales234 (0.344+/-0.183), L1 scales234 (0.379+/-0.151); the three agree to within 0.04 sigma. NOTE all three sit ABOVE the nominal 0.3 tolerance and are justified by their error bars (mean-std < 0.3), so this is 'matched, marginally tolerated bias', NOT 'strictly baryon-safe' in the sense used at 14000 deg2. RESULT: L1 gains 2.33x in FoM3 over the PS (1.06e6 vs 4.53e5) while PEAK COUNTS GAIN ESSENTIALLY NOTHING (4.65e5, 1.03x) -- peaks beat the PS on Om and S8 but lose on w0, which cancels in the determinant. So the claim is L1-specific, not a generic higher-order-statistics claim. CAVEATS: the peaks contour uses the NON-submean full-sky product; that is exact for detail-only scale sets because a starlet detail coefficient is a difference of smoothed maps and a constant cancels identically, but its shape-noise realisation differs (the processing seeds from os.urandom). Peaks also have the widest seed scatter of the three.

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_biased_fullsky_bsafe_l340_scales234_pooled`
- **generator commit**: `bc8ebd2`
- **generated**: 2026-07-30T21:55:17+00:00
- **published**: 2026-07-30T22:03:48+00:00 at repo `bc8ebd2`
- **rows in values.csv**: 9

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=340, rebin=10
- **peaks_l1**: wavelet scales234, submean, new_normalization, noisy s=0.26
- **power_spectrum_lmax**: 340
- **power_spectrum_lmax_chosen_by**: explicit --ps-lmax
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
