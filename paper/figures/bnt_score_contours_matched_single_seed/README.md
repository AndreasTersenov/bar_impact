# BNT vs non-BNT at a MATCHED scale cut (MOPED, single seed)

Single-seed companion to the matched-cut figure; each arm shows its most representative NDE seed. Ratio 1.314x. Note the 460-to-580 reference reads 1.085x here against 1.022x pooled: that inflation is compounded seed selection (the representative at 460 sits low in its ensemble, the one at 580 high), not physics. Quote the pooled value for that comparison.

- **source**: `plots/score_contours_matched_580_14000_single_seed`
- **generator commit**: `69586a219370d9a4bc00ed639396c383590f7375`
- **generated**: 2026-07-31T12:54:32Z
- **published**: 2026-08-01T12:21:27+00:00 at repo `5a91756`
- **rows in values.csv**: 3

## Scales included

- **scales_included**: BNT: bin-1 to ell<=580, bins 2-4 to ell<=1024; non-BNT: all bins to ell<=580
- **BNT_bin1**: [580, 1024, 1024, 1024]
- **nonBNT_cutall**: [580, 580, 580, 580]
- **BNT_bin1_ref**: [460, 1024, 1024, 1024]

## Caveats (from provenance)

- The analytic covariance behind the MOPED weights comes from the INTACT rebinned cache cov_rebinned_full_14000.npz, not from gaussian_cov_native_14000.npy: every native covariance and NaMaster workspace in cache_gaussian_cov/ was destroyed by the RAID0 failure (3 MiB stripe signature, ~20% zeroed). The substitution is exact at rebin=20 because a cut keeps whole leading bands and BNT commutes with the ell-rebin.
- Scale cuts are quantised to 80 in ell by the rebin=20 floor division, so ell_max=580 selects the same columns as its degenerate partner (540 or 620); the effective ell_max of the retained vector is lower than the label suggests. Quote the lower member of a pair.
- This is a CONSTRAINING-POWER comparison, not baryon mitigation. BNT bin-1 crosses 0.3 sigma at a LOWER ell_max (460) than non-BNT cut-all (620), so cutting only bin 1 does not control baryons better; its advantage is retaining more of the vector at equal unbiasedness.
- fom3_drawn describes exactly the samples plotted; fom3_seed_avg_cov removes between-seed mean scatter. They differ by 0.7-1.8% here.
- BNT@580 runs on 4 of 5 seeds; the rest were destroyed by the disk failure and are listed in seeds_skipped_disk_damage.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
