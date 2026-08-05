# Starlet scale to multipole response

Squared harmonic response W_j(l)^2 of each starlet scale, peak-normalised. Measured deterministically from single-pixel (white) inputs, so it carries no Monte Carlo error; cross-validated against 40 white-noise realisations (1.8% median agreement) and isotropic to 0.03%. Appendix material: it fixes the multipole band of each wavelet scale, which licenses the baryon-safe cut and the power-spectrum floor l >= 37. Method and interpretation: docs/starlet_scale_ell.md.

- **source**: `outputs/diagnostics/starlet_scale_ell/starlet_scale_ell`
- **generator commit**: `cd2a48f3a585e985dd26573fa65668f219ba4b9e`
- **generated**: 2026-08-05T08:48:35Z
- **published**: 2026-08-05T08:49:21+00:00 at repo `cd2a48f3`
- **rows in values.csv**: 5

## Scales included

- **note**: This figure MEASURES the scale-to-multipole mapping; it is not computed on a scale-cut data vector.
- **starlet**: nscale=5 -> wavelets 0-3 plus coarse
- **nside**: 512
- **ell_range_measured**: [0, 1535]
- **analysis_cut_shown**: shading marks the band removed by scales234 (wavelet 0), i.e. ell >= its half-power edge

## Measured multipole coverage per starlet scale

nside 512, nscale=5. Half-power = the multipole range over which the response is at least 50% of its peak. Bands OVERLAP; these are where each dominates, not sharp windows.

| scale | type | ell at peak | ell half-power range | role in the analysis |
|---|---|---|---|---|
| 0 | wav0 | 767 | 364 - 1535 | dropped by the baryon-safe cut (scales234) |
| 1 | wav1 | 228 | 142 - 336 | kept |
| 2 | wav2 | 114 | 71 - 168 | kept |
| 3 | wav3 | 57 | 36 - 84 | kept; its half-power edge sets the PS floor l>=37 |
| 4 | coarse | 0 | 0 - 24 | excluded throughout; the only band carrying the monopole |

Wavelet 0 is RESOLUTION-limited: it peaks near the Nyquist multipole 3*nside-1 = 1535, so its band moves with map resolution while the others do not. Angular-scale labels (~10 arcmin for wavelet 0, doubling thereafter) are dyadic smoothing-scale names and are NOT 10800/ell_peak -- a starlet band peaks at roughly half the multipole its label suggests.

## Caveats (from provenance)

- Measured at nside 512 with nscale=5. Wavelet 0 is RESOLUTION-limited, peaking near the Nyquist multipole 3*nside-1=1535, so its band moves with map resolution; the other bands do not.
- Half-power ranges summarise bands that OVERLAP substantially. 'scales234 covers ell 36-336' is where those bands dominate, not a sharp window.
- Measured on the FULL SPHERE. A mask couples multipoles, so the effective coverage of a masked map is broader than the table implies.
- theta ~ 10800/ell is a convention and is NOT how the scale labels were assigned: a starlet band peaks at roughly half the multipole its nominal scale size suggests. The measured ell ranges are authoritative.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
