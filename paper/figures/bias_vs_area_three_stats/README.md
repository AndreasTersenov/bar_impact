# bias vs area three stats

- **source**: `outputs/plots/submean_masked_peaks/nsigma_vs_area_fullres_noref`
- **generator commit**: `805e4cc`
- **generated**: 2026-08-03T12:32:13+00:00
- **published**: 2026-08-03T12:32:14+00:00 at repo `805e4cc`
- **rows in values.csv**: 18

## Scales included

- **power_spectrum**: monopole-subtracted MASTER, lmin=37, lmax~1020, rebin=10; no upper scale cut
- **peaks_l1**: wavelet detail scales 0,1,2,3 (coarse dropped), submean, new_normalization, noisy s=0.26, bins1234
- **note**: full resolution for every statistic — this figure measures how the baryon bias grows with area BEFORE any scale cut is applied.

## Caveats (from provenance)

- styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this figure sits beside the figures kept verbatim from it.
- Damaged run-pairs are skipped (see [skip] lines in the run log), so n_seeds differs from the original campaign and each mean is over a different subset.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
