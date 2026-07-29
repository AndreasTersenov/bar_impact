# Why peaks/L1 have an opposite-sign w0 degeneracy to the power spectrum

**Status:** investigation complete (June 2026). This note collects everything we learned, with
the numbers, the interpretation, and paths to every figure/script, so it can be compiled into a
paper section. All work is on branch `bnt_inference`; analysis lives in `scripts/diagnostics/`.

> **FOR THE PAPER (decision):** present the **noisy** regime only — it is the actual analysis setup.
> The clean story is the noisy moment progression (`moments_w0/fig1_moment_flip.png`): restrict the
> L1 to its variance (the 2-point information) → degeneracy matches the PS; add the higher moments
> (non-Gaussian information) → the w₀ degeneracy rotates to the opposite sign. Do **not** present the
> noiseless results in the paper (they would require a noiseless 2-point baseline too, and the noisy
> result is already clear). The **noiseless analysis (§3–5 boxes) is a robustness check only** — it
> confirms the flip is not a noise artifact (survives, comparable size, with signal-matched binning),
> i.e. the answer to "isn't this just noise?". Keep it in your back pocket, not in the figures.

---

## 1. The observation

In the standard (non-BNT, full-sky) `nobaryons_vs_baryonified` analysis, the higher-order
statistics (HOS: wavelet **peak counts** and wavelet **L1-norm**, config `scales234`) show
parameter degeneracies that **agree with the power spectrum (PS) in the Ω_m–S₈ plane but have the
opposite sign in w₀**. From the NPE posteriors (`outputs/samples/posterior_samples_*`):

| pair | PS l100-400 | peaks sc234 | l1 sc234 |
|---|---|---|---|
| Ω_m–S₈ | −0.97 | −0.83 | −0.83 |
| **Ω_m–w₀** | **−0.38** | **+0.72** | **+0.82** |
| **S₈–w₀** | **+0.57** | **−0.24** | **−0.40** |

Overlay: `outputs/diagnostics/ps_vs_hos_degeneracies.png`.

The concern was whether this is a pipeline bug. A relevant clue: the HOS degeneracy orientation
**resembles the PS computed from the *wrong*, raw (non-mean-subtracted), low-ℓ (ℓ<100) masked data**
— which we independently know is corrupted by mass-sheet/monopole leakage. So the question split
into (a) is it a bug, and (b) if not, what is the physics.

Parameter order in all posteriors: `[Ω_m, S₈, w₀, H₀, n_s, Ω_b]`; column 1 is genuinely **S₈**
(fiducial 0.84), not σ₈ — verified across all three NPE scripts (no σ₈→S₈ transform; identical
`cosmo_params.npy`). So the comparison is apples-to-apples.

---

## 2. It is not a bug — five independent checks

We reproduce the degeneracy with a **Fisher forecast built directly from the grid data vectors**
(Jacobian by lstsq of the data vector vs `cosmo_params`, Hartlap-corrected sample covariance from
the fiducial perms). This bypasses the NPE entirely, and reproduces the NPE signs:
PS Ω_m–w₀ = −0.80, l1 = +0.60, peaks = +0.27. So the flip is **in the data vectors**, and the
checks below test whether the data property is physical or an artifact.

| # | hypothesis | test | result | verdict |
|---|---|---|---|---|
| 1 | NPE/training bug | Fisher from raw grid data | reproduces the flip (l1 +0.60 vs PS −0.80) | in the data, not NPE |
| 2 | mass-sheet / monopole (like the PS leakage) | detail starlet scales are mean-invariant? | adding/removing the map mean changes detail-scale L1 by ~1e-7..1e-5 rel; only the **coarse** scale moves | not the monopole (coarse excluded from sc234) |
| 3 | low-ℓ contamination | does it survive at the *smallest* scales? | wav0+wav1 (ℓ≈840+229) alone give l1 Ω_m–w₀ = +0.52 | not a low-ℓ effect |
| 4 | a pathological corner (one tomo bin / SNR tail) | per-tomographic-bin Fisher | every bin independently +0.81..+0.92 | broad, not a corner |
| 5 | cosmology-dependent **auto binning** of the SNR histogram | reprocess with **fixed** −13/13 bins | Ω_m–w₀ = +0.74 (vs auto +0.60) | not the binning |

Also ruled out by a pipeline audit: the `_new_normalization` tag is just an SNR convention
(divide by per-scale starlet norm and a *fixed* `noise_std=0.0146`), cosmology-independent;
grid↔`cosmo_params` alignment is the identical `sorted(cosmo)×perm_0000..6` walk in all three
pipelines (all grids 16965 rows). Scripts: `fisher_ps_vs_hos_degeneracy.py`,
`fixedbin_l1_w0_test.py`.

**Important pipeline facts uncovered along the way** (relevant to the methods section):
- The on-disk HOS data was produced with **auto per-scale SNR binning** (`get_wtl1_sphere(min_snr=None,
  max_snr=None)`), not the `−13/13` argparse default — confirmed by reproducing row 0 exactly
  (coarse total |L1| 8.21e8 with auto vs 0 with −13/13). Side effect: SNR bin edges differ per map.
- The **coarse scale carries the mean** and is ~80–100× the detail scales; it is *excluded* from
  `scales234`/`scales1234`, so the cited results are clean of it. (See
  `memory/l1-peak-coarse-not-mean-centered.md`.)
- Use the **cosmostat_new venv** (`cosmostat_chi` pycs) for any pycs work — it is the one that made
  the on-disk vectors.

---

## 3. Where it comes from: the variance is PS-like; the PDF shape is flipped

Decomposing the L1 data vector (detail scales, fixed bins, 300-grid/190-fid subset):

| representation | Ω_m–w₀ |
|---|---|
| PS (2-point) | −0.80 |
| L1 **broadband amplitude** (total + 2nd moment ≈ the Gaussian/variance content) | **−0.35** |
| L1 **full histogram** (keeps the non-Gaussian shape) | **+0.69** |

So **in the noisy measurement regime**, when the L1-norm is reduced to the information the power
spectrum already has (its variance), the w₀ degeneracy is PS-like; the flip appears when the full
non-Gaussian shape is kept. Figure: `outputs/diagnostics/fixedbin_l1/hos_w0_mechanism.png` (right).

> **⚠ The decomposition (not the flip) is regime-dependent — SETTLED.** Repeating noiseless, the
> binning-independent **moments** (the robust measure) give: variance **+0.41** (already flipped, NOT
> PS-like), full var+skew+kurt **+0.34** (flipped). So **the flip is real in the pure signal.** An
> earlier histogram number (full +0.06) was a *wrong-binning artifact*: the noisy-tuned −13/13 SNR bins
> spread 40 bins over ±13 while the noiseless signal lives in ±1–2 (measured σ = 0.44–3.47 per
> tomo+scale) → badly under-resolved. **Concrete confirmation:** redoing noiseless l1 with SNR bins
> matched to the signal (±4σ per tomo+scale, `noiseless_matchedbin_l1.py`) gives Ω_m–w₀ = **+0.66**
> (rebin3; +0.55/+0.23 coarser — all clearly flipped), i.e. **comparable to noisy (+0.68), NOT weaker
> once the binning is right.** See `moments_w0/fig_noiseless_settled.png`. Conclusion: the **overall
> HOS w₀ flip is a robust signal property (~+0.5–0.7 in both regimes)**; what is noisy-regime-specific
> is only the *contrast*
> "variance PS-like vs shape flipped" — noiselessly the variance itself is flipped, so do NOT present
> "the variance is the 2-pt-equivalent and is PS-like" as a clean signal-level mechanism.

---

## 4. Moment decomposition — which moment carries the flip

Computing the **statistical moments of the starlet SNR field per detail scale**
(variance/skewness/kurtosis; 350-grid/195-fid subset; `moments_w0_reprocess.py`):

| configuration | Ω_m–w₀ |
|---|---|
| PS l100-400 (reference) | −0.80 |
| **variance only** (Gaussian) | **−0.59** |
| variance + skewness | −0.18 |
| variance + skewness + kurtosis | **+0.68** |
| skewness alone | **+0.61** |
| kurtosis alone | +0.15 |

In this **noisy (measurement) regime** the variance looks PS-like (−0.59) and the non-Gaussian
moments look flipped (skewness alone +0.61). Figures: `moments_w0/fig1_moment_flip.png`,
`moments_w0/fig2_moment_responses.png`.

> **⚠ REGIME-DEPENDENCE — the moment-by-moment attribution is NOT robust.** Repeating the moment
> decomposition on **noiseless** maps (`moments_noiseless.npz`) flips the individual numbers:
> variance −0.59→**+0.41**, skewness alone +0.61→**−0.46**, kurtosis +0.15→−0.42. The *full*
> var+skew+kurt stays flipped in both regimes (+0.68 noisy, +0.34 noiseless), but **the attribution
> to a specific moment is an artifact of the noisy regime** and must NOT be claimed.
> Figure: `moments_w0/fig_noisy_vs_noiseless_moments.png`. So the safe statement is "the full
> non-Gaussian content flips," not "the skewness carries it."

---

## 5. Even/odd refinement — the variance is the *only* PS-like piece

It is tempting to map the split to symmetry about SNR=0 (variance = even, skewness = odd). The data
says it is **not** that clean. Splitting the histogram into its even (symmetric) and odd
(antisymmetric) parts about SNR=0:

| piece | Ω_m–w₀ |
|---|---|
| variance *moment* (for contrast) | −0.59 |
| EVEN part of the histogram | **+0.31** |
| ODD part of the histogram | **+0.54** |
| full L1 | +0.69 |

**Both** even and odd parts are flipped, because the even part of the *histogram* contains all the
higher even moments (kurtosis and up), which are non-Gaussian. The clean statement is therefore:
**the only PS-like piece of the L1-norm is its variance (a single number per scale — the exact
2-point information); any PDF shape beyond that, even or odd, is flipped.** Figure:
`moments_w0/fig3_even_odd_variance.png`.

This also answers the "does the degeneracy switch at SNR=0?" intuition: **no.** The sign change of
dL1/dΩ_m at SNR≈±1.5 is the **redistribution pivot** (raising Ω_m empties the bulk and fills both
tails — one coherent see-saw), not a degeneracy switch. Neither left-vs-right (+0.66/+0.64) nor
even-vs-odd (+0.31/+0.54) flips the sign.

---

## 6. Field-level mechanism (response profiles)

From the whitened responses dL1/dθ vs SNR (`hos_w0_mechanism.png` left, `fig3` left):

- **dL1/dΩ_m and dL1/dS₈ have the *same* shape**: strongly negative at the field mean (SNR≈0),
  positive in *both* wings (peaks and voids). Physically, more clustering moves probability from the
  bulk into the tails. They are identical → that is *why* Ω_m and S₈ are ~0.9 degenerate.
- **dL1/dw₀ is the opposite (and weaker) shape** — small positive bump in the bulk, negative in the
  near-wings. Physically, w₀→0 suppresses growth, the *reverse* redistribution (tails→bulk).

The power spectrum sees only the resulting *variance*; the peaks/L1 see the *shape* of the
redistribution, which gives w₀ a distinct projection relative to the Ω_m–S₈ plane → opposite-sign
degeneracy. (Caveat for the paper: the final sign comes from the full Ω_m–S₈ marginalization, not a
naive two-parameter response-sign reading.)

---

## 7. Physical interpretation (the "why") — what we can and cannot claim

Settled view after the noiseless tests (§3–4 boxes):

1. **The HOS w₀ flip is a genuine, robust property of the signal's non-Gaussianity** — opposite to
   the PS in *both* regimes (noisy full +0.68, noiseless full +0.34 from the binning-independent
   moments). Noise strengthens it but does not create it. Verified five ways (§2), matches the NPE.
   This is the complementary information and is what goes in the paper.
2. **What IS noisy-regime-specific is the *decomposition*, not the flip.** The tidy "variance =
   2-point-equivalent = PS-like, the non-Gaussian shape carries the flip" only holds *with noise*:
   noiseless, the variance itself is flipped (+0.41), so there is no PS-like piece. Therefore do NOT
   present a clean "variance PS-like / skewness carries it / because nonlinear collapse encodes the
   growth history" mechanism — it is not supported at the signal level.

The intuitive growth-history picture (variance ∝ linear amplitude; skewness ∝ nonlinear collapse,
responding to w₀ differently) remains a *plausible motivation* for why HOS see w₀ differently, but a
*quantitative* signal-level attribution would need a perturbation-theory / S₃ growth-history
prediction — not claimed here.

**Paper one-liner (settled, defensible):** *The wavelet HOS constrain w₀ along the opposite direction
to the power spectrum — a real, robust property of the field's non-Gaussianity (opposite-sign and
comparable in both the noisy analysis, +0.68, and the pure signal, ~+0.5–0.7 with signal-matched
binning; verified five ways; matches the NPE), and the
complementary information it provides. We do not attribute it to a specific moment, and the
"variance-is-PS-like / shape-carries-the-flip" split holds only in the noisy regime.*

---

## 8. Caveats

- Per-SNR-**region** localization (voids vs bulk vs peaks) was **noise-limited** at the subset size
  (n_fid ≈ 200 for ~60 features) and is *not* claimed. The robust statements are amplitude-vs-shape
  and moment-by-moment (few, stable features).
- The Fisher decompositions use a **subset** (300–350 grid cosmologies, 190–195 fiducial perms),
  fixed −13/13 bins, with shape noise added per the pipeline. The full NPE on the full 16965 grid
  gives the same signs.
- **Noiseless tests (done) did NOT sharpen — they revealed regime-dependence** (§3–4 boxes): the
  moment attribution and the amplitude-vs-shape contrast both change/weaken without noise (full l1
  +0.69→+0.06). The robust result is the *direction* of the flip in the noisy (measured) regime, not
  a clean signal-level mechanism. A fully fair noiseless comparison needs SNR binning matched to the
  signal range (the −13/13 bins are tuned to the noisy range).
- **Peaks vs l1:** peaks confirm the amplitude-vs-shape *direction* (amplitude −0.37, PS-like) but the
  peaks flip is much weaker in the linear Fisher (+0.09) than the NPE (+0.72) — peaks are a more
  non-linear statistic, so l1 is the cleaner Fisher vehicle. On-disk peaks are 40-bin (not the 31
  default), fixed range.

---

## 9. Figure inventory

| figure | shows |
|---|---|
| `outputs/diagnostics/ps_vs_hos_degeneracies.png` | NPE triangle: PS vs peaks vs l1 (the observation) |
| `outputs/diagnostics/fixedbin_l1/hos_w0_mechanism.png` | response profiles + amplitude(PS-like)-vs-shape(flip) bars |
| `outputs/diagnostics/fixedbin_l1/hos_w0_origin.png` | per-SNR-region Fisher (noise-limited) + per-SNR diagonal contribution |
| `outputs/diagnostics/moments_w0/fig1_moment_flip.png` | Ω_m–w₀ moment progression + (Ω_m,w₀) ellipse rotation |
| `outputs/diagnostics/moments_w0/fig2_moment_responses.png` | how variance/skewness/kurtosis respond to Ω_m/S₈/w₀ |
| `outputs/diagnostics/moments_w0/fig3_even_odd_variance.png` | even/odd response decomposition + variance-is-the-only-PS-like-piece (noisy regime) |
| `outputs/diagnostics/moments_w0/fig_noisy_vs_noiseless_moments.png` | **moment Ω_m–w₀ is noise-regime-dependent** (noisy vs noiseless; signs flip) |
| `outputs/diagnostics/moments_w0/fig1_moment_flip_noiseless.png` | noiseless analog of fig1 — **non-monotonic** (variance already +0.41); the clean noisy rotation does not hold |
| `outputs/diagnostics/moments_w0/fig_noiseless_settled.png` | **settles it:** histogram Fisher vs feature count — noisy converges to its moment (+0.68), noiseless robust value ~+0.34; the +0.06 was under-conditioned |
| `outputs/diagnostics/moments_w0/fig_fisher_contours_noisy_noiseless.png` | **(Ω_m,S₈,w₀) Fisher contours**: PS (anti-diag) vs noisy-l1 (flipped) vs noiseless-l1 (near-round) |

## 9b. PUBLICATION FIGURE — and how to remake / restyle it

**Figure:** `outputs/diagnostics/paper/fig_hos_w0_mechanism.{pdf,png}` (+ caption stub
`..._caption.txt`). Two panels, **noisy regime** (the paper's actual setup):
- (a) Ω_m–w₀ correlation as moments are added: PS −0.80, l1 variance −0.59 (≈ PS → the 2-point
  information in the l1 is PS-like), l1 +skew −0.18, l1 +skew+kurt **+0.68** (flipped).
- (b) the Ω_m–w₀ degeneracy *direction* (correlation ellipses) rotating from PS-like (anti-diagonal,
  PS & variance) to flipped (diagonal, full).

**Remake:** `python scripts/diagnostics/make_paper_figure_hos_w0.py` (jaxili or any numpy+matplotlib
python). **To restyle for the paper format**, edit ONLY the `STYLE` block at the top of that script
(figsize, fonts, colors, dpi; `ONE_COLUMN=True` for single-column; `ellipse_mode="cov"` for
actual-size ellipses) and the panel titles — the Fisher/physics code never needs to change. The
script has a full "HOW TO REMAKE / RESTYLE" header. **Input data** it reads (regenerate only if lost):
`outputs/diagnostics/moments_w0/moments.npz` (NOISY moments; remake with `moments_w0_reprocess.py` in
the cosmostat_new venv) and the PS l100-400 Cls in `CosmoGridV1/stage3_forecast`. The numbers it
prints are the values to quote in the text.

## 10. Script + data inventory

Scripts (`scripts/diagnostics/`): `fisher_ps_vs_hos_degeneracy.py` (Fisher PS vs HOS + scale scan),
`fixedbin_l1_w0_test.py` (fixed-bin verification), `fixedbin_l1_full.py` (full-res fixed-bin
reprocess), `analyze_hos_w0_origin.py` (SNR-region + response profiles), `moments_w0_reprocess.py`
(moment reprocess), `analyze_moments_w0.py` (moment Fisher + fig1/fig2), `fig_even_odd_variance.py`
(fig3).

Data: `fixedbin_l1/fixedbin_l1_full{,_noiseless}.npz` (l1 histogram, noisy + noiseless),
`moments_w0/moments{,_noiseless}.npz` (field var/skew/kurt, noisy + noiseless). Reprocesses take a
`NOISELESS=1` env toggle. Peaks read directly from on-disk `grid/all_peak_counts_grid_*` (fixed-bin,
full 16965 grid). Use the **cosmostat_new venv** for pycs reprocesses.

Config: nside=512, nscales=5, detail scales [1,2,3] = `scales234` (wav1/2/3 ≈ ℓ 229/114/56),
nbins=40, fixed min/max_snr=−13/13, noise_std=0.0146, σ_e=0.26, n_gal=6.75, fiducial
(Ω_m,S₈,w₀)=(0.26, 0.84, −1.0).

## 11. Open / next steps

1. **Noiseless moment run** — sharpen §4–5 (variance alone on the PS-like side, every higher moment
   on the flipped side).
2. **Confirm on peak counts** — repeat the moment decomposition so the paper states it for both HOS.
3. **Quantitative growth-history / perturbation-theory connection** for the skewness w₀-dependence.
4. **Package** the best panels (response profiles + moment flip + ellipse rotation) into one
   publication figure.
