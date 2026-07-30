# ROBUSTNESS: baryon-safe contours with the COARSE scale included, 14000 deg2 (single_seed)

ROBUSTNESS CHECK, not a headline figure. PS at lmax=460; peaks and L1 at scales2345, i.e. the baryon-safe detail scales PLUS the coarse/mass-sheet scale normally excluded. Adding the coarse scale does NOT reintroduce bias -- it reduces it (peaks 0.079+/-0.047 -> 0.034+/-0.004; L1 0.091+/-0.073 -> 0.082+/-0.027) and collapses the seed scatter by ~10x for peaks and ~3x for L1, making these the most stable bias measurements in the set. But it COSTS constraining power: FoM3 falls 0.67x for peaks (1.52e5 -> 1.02e5) and rises only 1.08x for L1 (2.91e5 -> 3.15e5). The coarse scale carries little cosmological information, so for peaks it dilutes the data vector. There is therefore no case for scales2345 on constraining-power grounds; its value is showing that the baryon-safety conclusion does not depend on excluding the mass-sheet scale.

- **source**: `outputs/plots/contours_three_stats/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales2345_single_seed`
- **generator commit**: `2cb7a65`
- **generated**: 2026-07-30T22:50:50+00:00
- **published**: 2026-07-30T22:51:29+00:00 at repo `2cb7a65`
- **rows in values.csv**: 9

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax=460, rebin=10
- **peaks_l1**: wavelet scales2345, submean, new_normalization, noisy s=0.26
- **power_spectrum_lmax**: 460
- **power_spectrum_lmax_chosen_by**: largest step-40 cut with mean tension < 0.3 sigma
- **hos_scale_tag**: scales2345
- **threshold_sigma**: 0.3

## Caveats (from provenance)

- Runs are used only as null/biased PAIRS; a run unreadable on either side is dropped from both, so the null-to-biased offset is like-for-like. See runs_dropped for what the disk failure removed.
- Contours pool all surviving NPE training seeds, so their width includes the seed-to-seed training scatter, not just the posterior width of one seed. --single-run gives the single-seed version.
- aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from pre-crash figures are expected, data points are unaffected.
- The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) predates the lmin 100->37 recovery and the submean correction, so it is NOT numerically comparable to this figure.
- The PS cut and the HOS cut are different KINDS of cut — multipoles vs a wavelet scale — so they are not ell-matched to each other. Each is the cut that keeps its own statistic under 0.3 sigma. Pass --ps-lmax 400 to instead reproduce the ell-matched pairing used by the Fisher baryon-safe figure.
- The PS lmax is the largest step-40 cut still BELOW 0.3 sigma, not the crossing. The crossing is the first cut that fails (500 at 14000 deg^2); adopting it would put a 0.41-sigma bias in a 'baryon-safe' figure.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
