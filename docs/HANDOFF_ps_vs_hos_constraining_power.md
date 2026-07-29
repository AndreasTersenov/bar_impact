# HANDOFF — PS vs HOS constraining-power comparison (start with Fisher)

**Goal:** quantify and compare the cosmological constraining power of the power spectrum (PS) vs the
higher-order statistics (HOS: wavelet l1-norm, peak counts), and PS+HOS combined. Key science
question: now that the masked PS can use ℓ≥37 (the recovered low-ℓ band, see
[[nlb4-submean-gate-passed]]), how much does that **shrink the PS-vs-HOS gap** vs the paper's ℓ≥100
PS? **Start with Fisher** (Gaussian, fast, no NPE) to get the picture simply; NPE later.

## Start simple: FULL-SKY, Fisher, no reprocess needed
All the data for a full-sky Fisher already exists. Do this first.

**Fisher recipe** (already coded — reuse `scripts/diagnostics/fisher_ps_vs_hos_degeneracy.py` and the
`_fisher_cov` / `ps_cov` / `moment_cov` helpers in `scripts/diagnostics/make_paper_figure_hos_w0.py`):
- Jacobian `J`: lstsq of the grid data vector on `[1, params-mean]`, take `coef[1:].T`.
- Covariance: `C = cov(fid_perms)`, Hartlap `(n_fid-n_data-2)/(n_fid-1) * inv(C)`.
- `F = J.T @ Cinv @ J`; param cov `= inv(F)`. Report **FoM**: σ(S8), σ(w0), the (Ω_m,S8) and
  (Ω_m,w0) 2D areas (`π·sqrt(det 2x2)`), and a 6-param FoM (`1/sqrt(det(param_cov))`).

**Params:** `CosmoGridV1/stage3_forecast/grid/cosmo_params.npy` → (16965,6) = [Ω_m, S8, w0, H0, ns, Ω_b].

**Data (full-sky, noisy s0.26):**
- PS auto: `new_grid/all_cls_grid_nobaryons_bin{1..4}_noisy_s0.26.npy` (16965,1025);
  cross `new_grid/all_cross_cls_grid_nobaryons_bins1234_noisy_s0.26.npy` (16965,6150 = 6×1025).
  Fid in `fiducial/cosmo_fiducial/all_cls_fiducial_...` / `all_cross_cls_fiducial_...` (200 perms).
- l1: `grid/all_l1_norms_grid_nobaryons_bin{1..4}_noisy_s0.26_new_normalization.npy` (16965,5,40);
  fid `fiducial/cosmo_fiducial/all_l1_norms_fiducial_...` (200,5,40). Use **scales234 = indices [1,2,3]**.
- peaks: `grid/all_peak_counts_grid_nobaryons_bin{1..4}_..._new_normalization.npy` (16965,5,40); fid same.

**Comparisons to make:**
1. PS at cuts **l100-1024 (paper) vs l37-1024 (recovered)** — how much does low-ℓ tighten PS?
2. PS vs l1(sc234) vs peaks(sc234) — the gap, on the matched footprint.
3. **PS+l1**, **PS+peaks** (concatenate data vectors) — does HOS add FoM over PS, and by how much,
   for the l100 vs l37 PS?
For a roughly scale-matched comparison, l1/peaks sc234 ≈ ℓ 56–229; compare against PS over a similar
ℓ window (e.g. l37-400) as well as the full l37-1024.

**Covariance invertibility:** n_fid≈200, so keep n_features < ~190 (rebin PS bandpowers / HOS SNR bins).

## Caveats / gotchas
- **Fisher UNDERSTATES HOS** (Gaussian; misses the non-linear gain — e.g. peaks Fisher degeneracy
  +0.09 vs NPE +0.72). So treat Fisher FoM as a *lower bound* on HOS; the NPE will favor HOS more.
- **Masked vs full-sky:** start full-sky (data ready). The paper analysis is *masked*; the masked PS
  at l37 needs the **6-mask nlb=4+submean production** (gated, **NOT yet run** — see
  [[nlb4-submean-gate-passed]]; the 5000-mask gate passed). Do masked after the full-sky Fisher.
- Full-sky PS needs **no** mean subtraction (monopole at ℓ=0). The HOS coarse scale carries the mean
  but **scales234 excludes it** ([[l1-peak-coarse-not-mean-centered]]).
- Use the **cosmostat_new venv** only if reprocessing maps (pycs/NaMaster); the Fisher itself runs in
  the jaxili env from the on-disk aggregates.
- HOS w0-degeneracy investigation is DONE and separate — see `docs/hos_w0_degeneracy_investigation.md`.

## Suggested first step
Write `scripts/diagnostics/fisher_constraining_power.py`: load PS(l100/l37) + l1 + peaks (full-sky),
build Fisher for each + combined, print a FoM table (σ(S8), σ(w0), 2D areas, 6-param FoM), and a bar
figure of FoM by probe/cut. Then iterate.

---

## RESULTS — full-sky Fisher (DONE + POLISHED, 2026-06-17)
Script: `scripts/diagnostics/fisher_constraining_power.py` (any numpy env; ~15 s). Outputs in
`outputs/diagnostics/constraining_power/`: `fisher_fom_bars.{png,pdf}` (standalone),
`fisher_combined_bars.{png,pdf}` (combined), `fisher_contours.{png,pdf}`, `fisher_fom_table.json`,
`fisher_covs.npz`. Method = whitened features (per-feature /std before the dead-column floor — fixes a
bug where a relative variance floor wiped the tiny-amplitude PS out of combined vectors) +
Hartlap-corrected empirical fid-perm covariance. **Headline Jacobian = LOCAL** (Gaussian kernel,
bw=1.0 in whitened param-std units → N_eff≈6519/16965 grid cosmologies feed the derivative at the
fiducial); `JAC_MODE` also supports "global"/"anchored" for the robustness table. Binning: standalone
uses rich PS edges `[37,68,100,140,200,280,400,560,760,1024]` + HOS scales234, SNR→5 bins; **combined
uses a coarser matched binning** (PS edges `[37,100,200,400,760,1024]`, SNR→4 bins) so the joint
covariance keeps Hartlap≈0.5 and the "gain from adding HOS" is at matched binning, not a feature-count
artifact.

> **UPDATE (fair scale pairing):** the full-ell regime now pairs the l1024 PS with HOS **scales1234**
> (idx 0,1,2,3 — keeps the smallest wavelet scale), and the baryon-safe regime pairs PS lmax400 with
> HOS **scales234** (drops it). Adding the finest scale boosts l1 ~×9 and peaks ~×8 in FoM6, and flips
> peaks from below-PS (sc234, ×0.6) to above-PS (sc1234, ×5.2 over PS l100-1024); l1 sc1234 is ×160
> over PS l100-1024. The table below shows scales234 (now the baryon-safe HOS); full-ell HOS numbers
> are l1 sc1234 FoM6 3.78e11 (σ(S8) 0.0089, σ(w0) 0.033) and peaks sc1234 1.21e10 (σ(w0) 0.051). See
> `fisher_fom_table.json` for the authoritative current values; memory `fisher-ps-vs-hos-constraining-power`.

(A) STANDALONE FoM (LOCAL Jacobian, rich binning, full-sky noisy s0.26):

| probe | nfeat | hart | σ(Ωm) | σ(S8) | σ(w0) | A(Ωm,w0) | r(Ωm,w0) | FoM6 |
|---|---|---|---|---|---|---|---|---|
| PS l100-1024 (paper) | 70 | 0.64 | 0.0089 | 0.0213 | 0.092 | 2.4e-3 | −0.33 | 2.35e9 |
| PS l37-1024 (recovered) | 90 | 0.54 | 0.0083 | 0.0166 | 0.071 | 1.8e-3 | −0.09 | 4.54e9 |
| PS l37-280 (HOS-ℓ-tight) | 50 | 0.74 | 0.0156 | 0.0321 | 0.087 | 4.2e-3 | −0.04 | 7.2e7 |
| l1 scales234 | 57 | 0.71 | 0.0106 | 0.0137 | 0.060 | 1.2e-3 | +0.81 | 4.10e10 |
| peaks scales234 | 60 | 0.69 | 0.0150 | 0.0220 | 0.082 | 3.4e-3 | +0.48 | 1.44e9 |

(B) COMBINED FoM (coarse matched binning, Hartlap≈0.5–0.79):

| probe | nfeat | hart | σ(S8) | σ(w0) | A(Ωm,w0) | FoM6 |
|---|---|---|---|---|---|---|
| PS l37-1024 [coarse] | 50 | 0.74 | 0.0188 | 0.070 | 2.1e-3 | 2.09e9 |
| l1 [coarse] | 48 | 0.75 | 0.0149 | 0.050 | 1.2e-3 | 2.71e10 |
| peaks [coarse] | 48 | 0.75 | 0.0230 | 0.093 | 3.7e-3 | 7.2e8 |
| PS l37 + l1 | 98 | 0.50 | 0.0089 | 0.036 | 6.2e-4 | 1.89e11 |
| PS l37 + peaks | 98 | 0.50 | 0.0103 | 0.048 | 9.4e-4 | 2.13e10 |
| PS l100 + peaks | 88 | 0.55 | 0.0110 | 0.047 | 1.0e-3 | 1.99e10 |

Four findings:
1. **Low-ℓ recovery (l37 vs l100) is the robust number:** σ(S8) ×0.78, σ(w0) ×0.77, FoM6 ×1.9.
   Survives all Jacobian modes. Recovering ℓ37–100 tightens the PS ~20–25%/param.
2. **l1 ≫ PS, peaks ≈/below PS — Fisher over-credits l1, under-credits peaks.** l1 FoM6 ×17 over PS
   l100 under the LOCAL derivative (×70 with global/anchored — the global linearization inflates a
   steeply-nonlinear statistic). peaks sit *below* PS l100 in this Gaussian-likelihood Fisher — that
   is precisely where the non-Gaussian gain only the NPE captures lives. **Absolute PS↔HOS gap needs
   the NPE.**
3. **PS+HOS is super-additive on w0 — the degeneracy flip pays off.** Because l1/peaks have the
   OPPOSITE (Ωm,w0) degeneracy to the PS, combining breaks it: PS l37 + l1 → σ(w0)=0.036 (vs 0.070
   PS, 0.050 l1) and (Ωm,w0) area 6.2e-4 (vs 2.1e-3, 1.2e-3); FoM6 ×90 over PS l37, ×7 over l1 alone.
   Even weak peaks help: PS+peaks σ(w0) 0.047–0.048 vs 0.070. This ties `hos_w0_degeneracy_
   investigation.md` directly to constraining power — a clean paper story.
4. **Contours** (`fisher_contours.png`) show the (Ωm,w0) flip (l1/peaks tilt opposite PS), l1 tightest.

Caveats (in the script docstring): empirical cov already has non-Gaussian VARIANCE (not a Gaussian-cov
approx); the linearized Jacobian is the real approximation and is NOT a clean bound (it inflated l1,
didn't rescue peaks); 200 perms may under-estimate HOS non-Gaussian covariance tails (optimistic HOS);
full-sky, not the masked paper footprint.

### Verification (DONE, `scripts/diagnostics/verify_fisher_constraining_power.py`) — 3/3 hard checks PASS
1. Whitening invariance: cov(raw) vs cov(whitened) differ by 2e-12 (provably a no-op).
2. Independent recompute (separate data load + bandpower code + einsum JᵀC⁻¹J) matches the main
   script to ~1e-12 on σ(S8), σ(w0), FoM6, r(Ωm,w0).
3. Reproduces the accepted `fisher_ps_vs_hos_degeneracy.py`: l1 r(Ωm,w0)=+0.60 and peaks=+0.27 match
   exactly; PS negative.
Honest method caveat (CHECK 4, linear-fit R²): PS R²=0.90(global)/0.95(local) → PS Fisher solid; l1
R²=0.66/0.81, peaks 0.77/0.81 → the HOS response is nonlinear so the linear Jacobian under-describes
it → the HOS *absolute* FoM is linearization-limited (this is the 70→17 swing). The local Jacobian
lifts l1 R² 0.66→0.81, justifying it as the headline. The low-ℓ PS recovery ratio rides on the R²≈0.95
PS fit and is unaffected. **Conclusion: code is correct; low-ℓ result trustworthy; HOS magnitude has a
quantified (R²) caveat, direction robust.**

### BARYON-SAFE regime (DONE) — `fisher_contours_baryon_safe.{png,pdf}`
PS lmax=400 (drops the high-ℓ baryon-contaminated band); HOS = scales234 (already excludes the finest
scale index 0). FoM (local Jacobian, full-sky noisy):

| probe | σ(S8) | σ(w0) | A(Ωm,w0) | r(Ωm,w0) | FoM6 |
|---|---|---|---|---|---|
| PS l100-400 (paper baryon-safe) | 0.037 | 0.115 | 5.1e-3 | −0.53 | 5.6e7 |
| PS l37-400 (recovered) | 0.025 | 0.075 | 2.9e-3 | −0.10 | 2.1e8 |
| l1 scales234 | 0.014 | 0.060 | 1.2e-3 | +0.81 | 4.1e10 |
| peaks scales234 | 0.022 | 0.082 | 3.4e-3 | +0.48 | 1.4e9 |

Take-aways: (a) the w0 flip PERSISTS in the baryon-safe regime (PS −0.5/−0.1 vs l1 +0.81, peaks +0.48).
(b) Cutting the PS at lmax=400 guts it (FoM6 2.35e9→5.6e7, ~×40 vs l100-1024), so the HOS lead is even
larger here than full-ℓ (l1 ×730, peaks ×26 over PS l100-400 — magnitudes linearization-limited, but
the direction confirms the paper's motivation for HOS at baryon-safe scales). (c) Low-ℓ recovery
matters MORE in the baryon-safe regime: PS l37-400 vs l100-400 = FoM6 ×3.8 (σ ×0.65–0.68), vs ×1.9
full-ℓ — recovering ℓ37–100 is most valuable exactly when high-ℓ is cut for baryon safety.

## Next (open)
- **Decisive step = NPE** to pin the PS↔HOS gap (Fisher can't: it over-credits l1, under-credits
  peaks). Full-sky `nobaryons_vs_nobaryons`, matched cuts (PS l100 vs l37 vs l1 vs peaks). No clean
  full-sky non-BNT HOS NPE samples exist on disk yet — would need fresh runs.
- **Masked production** (the gated 6-mask nlb=4+submean run) if the paper-footprint masked PS-vs-HOS
  is wanted instead of/after full-sky.
- Optional Fisher polish: make the local Jacobian the headline; trim combined features for Hartlap.
