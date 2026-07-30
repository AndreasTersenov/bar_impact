# BNT bin-1 vs non-BNT, optimal binning (rebin 40)

BNT bin-1-only cut against the standard non-BNT cut-all, with BNT at its optimal rebinning. THIS is the honest BNT figure: PLAN_bnt_optimal_binning.md records that the earlier rebin=10 version substantially OVERSTATED BNT's baryon mitigation, because raw NPE under-extracted and so inflated the BNT contours, hiding the bias. The per-panel percent-extracted annotation is a quoted historical result and is NOT reproducible from this repo -- see provenance.json.

- **source**: `outputs/plots/bnt_ps_bin1_submean_l37/nsigma_vs_lmax_bnt_bin1_allareas_optimal`
- **generator commit**: `701fba8`
- **generated**: 2026-07-29T16:58:19+00:00
- **published**: 2026-07-30T07:31:51+00:00 at repo `b900627`
- **rows in values.csv**: 216

## Known gaps

- provenance is missing 'mplstyle'

## Caveats (from provenance)

- Crossings are the lowest grid cut with mean >= threshold (no interpolation), matching how this campaign has always reported them.
- At-cut comparison is best-vs-best: BNT at rebin=40 vs non-BNT at rebin=10 (different binning) — see docs/PLAN_bnt_optimal_binning.md Caveats.
- rebin=40 means 3-run means (overnight budget), noisier than the 5-run default-binning variant, and coarser cut resolution (visible staircase).

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
