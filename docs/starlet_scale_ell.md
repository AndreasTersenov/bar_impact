# Wavelet scale ↔ multipole correspondence (starlet, nside 512)

This note records the exact multipole (`ℓ`) ranges probed by each starlet wavelet scale used
for the L1-norm and peak-count statistics, and how they line up with the angular power-spectrum
multipole range. It is intended both as an internal reference and as candidate methods material
for the paper.

Figure: `outputs/diagnostics/starlet_scale_ell/starlet_scale_ell_clean.{png,pdf}`
Data: `outputs/diagnostics/starlet_scale_ell/starlet_transfer_data.npz`
Reproduce: `python scripts/diagnostics/starlet_scale_ell.py` (cosmostat_new venv; needs pycs).

## Method

The L1-norm and peak counts are computed on the undecimated isotropic **starlet** transform
(`CMRStarlet`, `nscale=5` → 4 wavelet/detail bands + 1 coarse/smooth scale) of the HEALPix
convergence maps. Each scale `j` acts as a fixed, isotropic linear filter `W_j(ℓ)` in harmonic
space, so the relevant quantity is the squared transfer function `W_j(ℓ)²`.

We measure it directly: push `N=40` white-noise HEALPix maps (nside 512, flat input `C_ℓ`)
through the starlet, and take the angular power spectrum of each scale's coefficient map.
Because the input `C_ℓ` is flat, the per-scale output `C_ℓ` **is** `W_j(ℓ)²` (up to a common
constant). Averaging over realizations and binning in log-`ℓ` gives smooth transfer functions.

## Result — ℓ coverage of each scale

Peak and half-power (≥ 50 % of peak) multipole ranges (nside 512, nscale 5):

| scale | type | ℓ at peak | ℓ half-power range | rough angular scale |
|------:|:-----|----------:|:-------------------|:--------------------|
| 0 | wavelet | ~840 (broad, res-limited) | ~370–1535 | finest (~10 arcmin) |
| 1 | wavelet | ~229  | ~143–335  | ~20 arcmin |
| 2 | wavelet | ~114  | ~72–167   | ~40 arcmin |
| 3 | wavelet | ~56   | ~37–83    | ~80 arcmin (largest wavelet) |
| 4 | coarse  | ~1–2  | ~0–24     | mean / largest scales |

(Values from `N=40` white-noise realizations, log-`ℓ` binned; reproducible via the script.)

(The "rough angular scale" column is the dyadic smoothing-scale label; note it is **not**
`10800/ℓ_peak` — a starlet band peaks at a lower `ℓ` than the nominal scale size suggests,
by a factor ~2. The measured `ℓ` ranges above are the authoritative quantity.)

## Interpretation

- Each **wavelet** scale is a band-pass filter probing a localized, dyadically-spaced range of
  multipoles. Together, the four wavelet scales used in the analysis (`scales1234`, i.e. wav0–3)
  span **ℓ ≈ 30–1535**, with the largest wavelet scale (wav3) reaching down to ℓ ≈ 30.
- The **coarse** scale is a low-pass filter capturing **ℓ ≲ 16–30** — the largest scales and the
  map mean. It is the only starlet scale sensitive to the (mass-sheet-degenerate) monopole; the
  wavelet bands are band-pass and largely insensitive to a constant offset. See
  `memory: l1-peak-coarse-not-mean-centered` — the coarse scale is **not** currently
  mean-centered, so it must be mean-subtracted (subtract the mask-weighted map mean before the
  transform) before it can be used.
- The finest scale (wav0) is **resolution-limited**: it peaks near the map's Nyquist multipole
  (3·nside−1 = 1535) rather than at a fixed physical scale.

## Implication for power-spectrum scale matching

Because the wavelet HOS (wav0–3) covers ℓ ≈ 30–1535, an angular power spectrum measured over
**ℓ ∈ [30, 1024]** is scale-consistent with the higher-order statistics as published (wavelet
scales only) — the largest wavelet scale already reaches ℓ ≈ 30, so no coarse scale is needed
to justify a low-ℓ floor of 30. The coarse scale would only extend the HOS to ℓ < 30 (below the
adopted floor) and is the leakage-prone scale, so it is deferred.

There is a soft edge: between the coarse (half-power ≤ 24) and wav3 (half-power ≥ 37) the HOS
response dips to ~25–30 % around ℓ ≈ 24–37, so ℓ = 30 is sampled by the HOS but not at full
weight. A strictly conservative match would place the PS floor at wav3's half-power edge
(ℓ ≈ 37) rather than 30.
