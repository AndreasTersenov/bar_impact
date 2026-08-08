# Appendix D - tomographic convergence maps, BNT basis, with shape noise

- **source**: `outputs/plots/tomographic_bnt_maps/noisy_bnt_transformed_maps_flatsky`
- **generator commit**: `41fbac1e5d4b8cc727d2a65c1a8c689d4535e717`
- **generated**: 2026-08-08T17:32:31Z
- **published**: 2026-08-08T17:32:53+00:00 at repo `41fbac1e`
- **rows in values.csv**: 4

## Scales included

- **scales_included**: NOT APPLICABLE -- this is a map-level figure. No wavelet decomposition, no multipole cut and no scale selection is applied; every panel is the full projected convergence field at the native NSIDE=512 resolution.
- **quantity_plotted**: kappa - <kappa> per panel (the per-panel mean is removed; see colour_scale.why_mean_subtracted)
- **tomographic_bins**: stage3_lensing1..4, increasing in source redshift

## Caveats (from provenance)

- BNT row 0 is the identity on bin 1, so 'BNT bin 1' and standard 'Bin 1' are the SAME field. Not four independent transforms.
- NO display smoothing. Per-pixel SNR is below 1 in every bin of the noisy maps (standard 0.19/0.31/0.50/0.61, BNT 0.19/0.11/0.10/0.07), so surviving structure in the high-z standard bins reads as faint large-scale mottling rather than crisp structure. The standard-vs-BNT SNR ratio at bin 4 is still ~9x.
- The colour range differs between the noiseless and the noisy figure (each row sets its own), so the two are NOT directly comparable in amplitude. They are comparable in texture and in within-row contrast.
- One realisation, one patch. Illustrative, not a statistical statement.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
