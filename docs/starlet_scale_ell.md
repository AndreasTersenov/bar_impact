# Wavelet scale ↔ multipole correspondence (starlet, nside 512)

Candidate appendix material. This note records the exact multipole (`ℓ`) band probed by each
starlet wavelet scale, how it is measured, and what it licenses in the analysis: the
higher-order scale cut, the power-spectrum multipole range, and the treatment of the coarse
scale.

**Figure** `outputs/diagnostics/starlet_scale_ell/starlet_scale_ell.{png,pdf}`
**Data** `starlet_scale_ell_data.npz` · **provenance** `_provenance.json` · **table** `_values.csv`
**Reproduce**

```
PYTHONNOUSERSITE=1 PYTHONPATH=/lustre/fswork/projects/rech/nzu/ulx34io/cosmostat_src \
  /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python \
  scripts/diagnostics/starlet_scale_ell.py --method both
```
or `sbatch scripts/jz/starlet_scale_ell.slurm`. `pycs` is a source checkout and needs
`PYTHONPATH`; `healpy`/`numpy` come from the `jaxili` env. No single environment has both.

## 1. What is being measured, and why it is well defined

The ℓ₁-norm and peak counts are computed on the undecimated isotropic **starlet** transform
(`CMRStarlet`, `nscale=5` → four wavelet/detail bands `w₀…w₃` plus one coarse band `c₄`) of the
HEALPix convergence maps.

The transform is **linear and isotropic**, so each scale acts as a fixed filter in harmonic
space: a scale's coefficient map has

```
a_lm^(j) = W_j(ℓ) · a_lm ,
```

and the quantity that matters for scale bookkeeping is the squared transfer function
`W_j(ℓ)²`. Because `W_j` depends on ℓ alone and not on m, "wavelet scale 2" *is* a definite
band of multipoles — the statement the appendix needs.

Concretely the starlet is built from successive smoothings, `c_j = S_j(map)` with
`w_j = c_{j-1} − c_j`, so the detail bands are differences of low-pass filters and are
therefore band-pass, while the final coarse band is the remaining low-pass.

## 2. Method — a deterministic measurement

**A single-pixel map is exactly white.** For a delta at direction `n̂`,
`a_lm = Ω_pix Y*_lm(n̂)`, hence

```
Σ_m |a_lm|² = Ω_pix² (2ℓ+1)/(4π)   ⟹   C_ℓ = Ω_pix²/(4π) ,
```

independent of ℓ. Pushing that map through the transform and taking the ratio of output to
input angular power spectra per multipole gives the response directly:

```
W_j(ℓ)² = C_ℓ[ coef_j ] / C_ℓ[ input ] .
```

Both sides are deterministic, so **the result carries no Monte Carlo error at any ℓ**, and no
binning is required. Dividing by the measured input `C_ℓ`, rather than assuming it is exactly
flat, also removes the pixel window, which departs from flat near the Nyquist multipole.

Eight delta positions spread over the sphere are averaged. This is *not* noise averaging —
there is no noise — but a check on pixelisation anisotropy, which a single position would probe
in only one direction. The measured spread across positions is **0.03 % median, 0.24 % max**
(where the response exceeds 5 % of peak), confirming the filter is isotropic to well below any
level that matters here.

**Cross-validation.** The earlier method pushed `N = 40` white-noise maps through the transform
and averaged. That is correct in expectation but noisy where there are few modes: the fractional
error on `C_ℓ` is `√(2 / ((2ℓ+1) N))`, ≈ 10 % at ℓ = 2. Running both (`--method both`) gives a
median agreement of **1.8 %** and a maximum of **13.5 %**, the disagreement concentrated at low ℓ
and consistent with the Monte Carlo error alone. The low-ℓ wobble on the coarse band in the
original figure was an artefact of the estimator, not structure in the filter; the earlier note
hid it by binning in log-ℓ, which also blurs the band edges this figure exists to show.

## 3. Result — ℓ coverage of each scale

Peak and half-power (≥ 50 % of peak) multipole ranges, nside 512, `nscale=5`:

| scale | type | ℓ at peak | ℓ half-power range | rough angular scale |
|------:|:-----|----------:|:-------------------|:--------------------|
| 0 | wavelet | 767 (broad, resolution-limited) | 364 – 1535 | finest, ~10′ |
| 1 | wavelet | 228 | 142 – 336 | ~20′ |
| 2 | wavelet | 114 | 71 – 168 | ~40′ |
| 3 | wavelet | 57 | 36 – 84 | ~80′ (largest wavelet) |
| 4 | coarse | 0 | 0 – 24 | mean / largest scales |

Two properties worth stating explicitly:

- The bands are **dyadic**: each wavelet peaks at roughly half the multipole of the previous one
  (767 → 228 → 114 → 57).
- Wavelet 0 is **resolution-limited**, not scale-limited: it peaks near the map's Nyquist
  multipole `3·nside − 1 = 1535` rather than at a fixed physical scale, so its band would move
  with map resolution while the others would not.

The "rough angular scale" column is the dyadic smoothing-scale label and is deliberately *not*
`10800/ℓ_peak`: a starlet band peaks at a lower ℓ than its nominal scale size suggests, by a
factor of about two. **The measured ℓ ranges are the authoritative quantity.**

### A note on ℓ → angular scale

A multipole ℓ is conventionally quoted as an angular scale `θ ≈ 180°/ℓ`, and since
`180° = 10800′`, as `θ[arcmin] ≈ 10800/ℓ`. It is a rule of thumb, not an identity — `360°/ℓ`
(full wavelength) and `2π/ℓ` are also in use — and for this transform it is actively
misleading, because a starlet band peaks at roughly half the multipole its nominal scale label
implies. Earlier versions of the figure carried a second axis showing `10800/ℓ` across the top;
it has been removed for exactly that reason, and because the paper figures carry no axis
titles beyond the two axes themselves. `--top-axis` restores it for internal use. If the
conversion is wanted by the reader, put it in the caption where it can be qualified.

## 4. What this licenses in the analysis

### 4.1 The baryon-safe cut is a multipole cut

The higher-order analysis keeps wavelets 1–3 (`scales234`) and drops wavelet 0. By the table,
that removes **ℓ ≳ 364** and retains **ℓ ≈ 36 – 336**. So "drop the finest wavelet scale" is a
statement about multipoles, and one that matches the measured bias behaviour: dropping wavelet 0
reduces the baryonic bias by a factor 40–190 depending on footprint, at a cost of 60–70 % of the
FoM (`outputs/diagnostics/hos_cut_safety_*.json`,
`outputs/diagnostics/baryon_safe_fom_table_errorbar.csv`).

That the baryon sensitivity is concentrated almost entirely in one band is expected — baryonic
feedback modifies the matter power spectrum mainly at `k ≳ 1 h/Mpc`, which at these source
redshifts projects to the ℓ ≳ few hundred that wavelet 0 covers.

### 4.2 The power-spectrum floor ℓ ≥ 37 is set by this figure

The largest wavelet, `w₃`, has its half-power edge at **ℓ = 36**. The power spectrum is analysed
over `ℓ ∈ [37, ℓ_max]`. That floor is therefore not a convention: it is essentially the lowest
multipole at which the higher-order statistics still retain at least half their peak response,
so PS and HOS are scale-matched at the low end. Going lower would give the PS access to
multipoles the wavelet bands barely measure, making the comparison unfair to the HOS.

There is a soft edge worth stating: between the coarse band (half-power ≤ 24) and `w₃`
(half-power ≥ 36) the total wavelet response dips to ~25–30 % around ℓ ≈ 24–36. Multipoles just
below 37 are sampled by the HOS, but not at full weight.

### 4.3 The coarse scale is the only mean-sensitive band

The coarse band is a low-pass filter reaching `ℓ = 0`, so it alone carries the map monopole —
the mass-sheet-degenerate mode. The wavelet bands are band-pass differences of low-pass filters,
so a constant offset cancels identically in them.

Two consequences, both acted on in the analysis:

- On the **masked** footprints the coarse scale must be used only on footprint-mean-subtracted
  ("submean") maps. See `memory: l1-peak-coarse-not-mean-centered`.
- On the **full sky**, "submean" is a monopole subtraction, so a detail-only scale set such as
  `scales234` is monopole-invariant *by construction* and the non-submean product may be used
  for it — but never for a set containing the coarse scale. `plot_contours_vs_area.py` and
  `plot_contours_three_stats.py` both refuse that substitution rather than silently applying it.

## 5. Caveats

- Measured at **nside 512** with `nscale=5`, matching the analysis maps. Wavelet 0's band is
  resolution-limited and would move with nside; the other bands would not.
- Half-power ranges summarise bands that overlap substantially. The bands are not disjoint, so
  "scales234 covers ℓ 36–336" describes where those bands dominate, not a sharp window.
- The response is measured on the **full sphere**. A mask couples multipoles; the wavelet bands
  themselves are unchanged, but the effective coverage on a masked map is broader than the table
  suggests.
