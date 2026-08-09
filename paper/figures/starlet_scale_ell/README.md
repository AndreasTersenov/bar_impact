# Starlet scale to multipole response

- **source**: `outputs/diagnostics/starlet_scale_ell/starlet_scale_ell`
- **generator commit**: `33088a94f6f1df83725dea98a38283d67b023af1`
- **generated**: 2026-08-09T14:28:06Z
- **published**: 2026-08-09T14:28:35+00:00 at repo `33088a94`
- **rows in values.csv**: 5

## Scales included

- **note**: This figure MEASURES the scale-to-multipole mapping; it is not computed on a scale-cut data vector.
- **starlet**: nscale=5 -> wavelets j=1-4 (array bands wav0-wav3) plus coarse
- **nside**: 512
- **ell_range_measured**: [0, 1535]
- **analysis_cut_shown**: none. The shaded band that marked the scales234 cut was removed: this figure reports the measured mapping, and which scales a cut discards is stated in the text.

## Measured multipole coverage per starlet scale

nside 512, nscale=5. Half-power = the multipole range over which the response is at least 50% of its peak. j is the paper's 1-based scale index (j=1 is the finest wavelet); 'array band' is the transform's own 0-based name. Bands OVERLAP; these are where each dominates, not sharp windows.

| j | array band | ell at peak | ell half-power range | role in the analysis |
|---|---|---|---|---|
| 1 | wav0 | 767 | 364 - 1535 | dropped by the baryon-safe cut (j >= 2) |
| 2 | wav1 | 228 | 142 - 336 | kept |
| 3 | wav2 | 114 | 71 - 168 | kept |
| 4 | wav3 | 57 | 36 - 84 | kept; its half-power edge sets the PS floor l>=37 |
| coarse | coarse | 0 | 0 - 24 | excluded throughout; the only band carrying the monopole |

j=1 (array band wav0) is RESOLUTION-limited: it peaks near the Nyquist multipole 3*nside-1 = 1535, so its band moves with map resolution while the others do not. Angular-scale labels (~10 arcmin for j=1, doubling thereafter) are dyadic smoothing-scale names and are NOT 10800/ell_peak -- a starlet band peaks at roughly half the multipole its label suggests.

## Caveats (from provenance)

- Measured at nside 512 with nscale=5. Wavelet 0 is RESOLUTION-limited, peaking near the Nyquist multipole 3*nside-1=1535, so its band moves with map resolution; the other bands do not.
- Half-power ranges summarise bands that OVERLAP substantially. 'j>=2 covers ell 36-336' is where those bands dominate, not a sharp window.
- Measured on the FULL SPHERE. A mask couples multipoles, so the effective coverage of a masked map is broader than the table implies.
- theta ~ 10800/ell is a convention and is NOT how the scale labels were assigned: a starlet band peaks at roughly half the multipole its nominal scale size suggests. The measured ell ranges are authoritative.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
