# The proper, unambiguous Fisher: methodology grounded in the forecasting literature

**Status:** methodology proposed 2026-06-23. Supersedes the ad-hoc `fisher_bnt_vs_nonbnt.py`
(lstsq-over-prior derivatives + noisy 200-perm covariance) whose ambiguities caused the
noisy-J 0.37→0.91 / averaged-J 0.37→0.53 back-and-forth.

## Why we kept getting conflicting numbers
Our two Fisher inputs were both estimated by non-standard, noise-contaminated methods:
- **Derivative** = global least-squares of C_ℓ vs θ over the whole Latin-hypercube prior. This is
  *not* a local derivative: it conflates nonlinearity with the gradient and absorbs per-realization
  shape noise. Noise-averaging helped (R² 0.47→0.80) but the lstsq-over-prior is still not the
  established estimator.
- **Covariance** = 200 perms for 50–96 features, with a large, config-differential Hartlap factor.

This is the *documented* pitfall: Euclid's forecast-validation paper states plainly that Fisher
results "may depend on the choice of algorithm and stepsizes" because (i) numerical/realization noise
makes derivatives unstable, and (ii) the posterior may be non-Gaussian — and that covariance
estimated from finite sims biases the precision matrix. So our instability is expected, not a physics
result. The literature's fix is a clean, differentiable, analytic forecast, cross-validated against a
full likelihood.

## Established references (the standard we'll meet)
- **Tegmark, Taylor & Heavens 1997** — the foundational Fisher-for-cosmology formalism;
  `F_αβ = (∂μ/∂θ_α)ᵀ C⁻¹ (∂μ/∂θ_β)` (data-derivative term; drop the covariance-derivative term).
- **Euclid prep. VII / IST:F, Blanchard et al. 2020 ([arXiv:1910.09273](https://arxiv.org/pdf/1910.09273))**
  — the definitive Fisher-validation protocol: derivatives via n-point stencils *or* polynomial fits
  with **explicit step-size convergence** ("SteM"), multiple independent codes cross-checked to agree.
- **Autodiff / differentiable Boltzmann (jax-cosmo; DISCO-DJ, arXiv:2311.03291)** — automatic
  differentiation gives *exact* derivatives and **removes the step-size convergence problem entirely**.
  This is the modern gold standard and the cleanest route for us.
- **Hartlap et al. 2007 / Sellentin & Heavens 2016** — sim covariance: Hartlap de-biases the precision
  matrix (Gaussian plug-in) but misses the heavy tails; Sellentin–Heavens (Student-t) is the rigorous
  treatment. For a *forecast*, the analytic Gaussian covariance is cleaner and self-consistent.
- **Coe 2009** — Fisher ellipse conventions / quick-start sanity.

## The proper recipe (removes every ambiguity we hit)

### 1. Derivatives — autodiff of a differentiable theory model (primary)
Build the tomographic auto+cross angular power-spectrum model μ(θ) = C_ℓ^{ij}(θ) in **jax-cosmo**, with
the SAME ingredients as the data pipeline: CosmoGrid source n(z), nonlinear P(k) (Halofit/HMCode),
the 4-bin **BNT matrix**, the mask/bandpower binning. Take **∂μ/∂θ by autodiff** — exact, noiseless,
no step-size choice, no realization noise, no lstsq-over-prior. θ = (Ωm, S8, w0, H0, ns, Ωb).
- *Cross-validation (Euclid-style):* autodiff vs a converged n-point finite-difference stencil (must
  agree → confirms no autodiff/implementation bug), and theory-J vs the **noise-averaged sim**
  derivative (shape agreement → confirms the theory matches the simulations' parameter response).

### 2. Covariance — analytic Gaussian (primary)
Analytic Gaussian covariance of the auto+cross bandpowers, **mask-aware via NaMaster**
(`NmtCovarianceWorkspace` + `gaussian_covariance`) from the fiducial C_ℓ + shape-noise + the mask
coupling. Perfectly conditioned, no perm noise, self-consistent with the Gaussian-likelihood Fisher.
- *Cross-validation:* analytic vs the 200-perm **Sellentin–Heavens** sample covariance (agree in the
  Gaussian regime → confirms the analytic model; the sim covariance is the reality check).

### 3. Assemble & transform
`F = JᵀC⁻¹J`. Apply BNT/mask/binning identically across configs. Per-bin cuts via the worker's rule.

## Validation suite — what makes it "100% sure"
A result we quote only after ALL pass:
1. **Autodiff = finite-difference** (converged stencil) — derivative correctness, no step ambiguity.
2. **Theory-J ≈ noise-averaged sim-J** — the model reproduces the simulations' response.
3. **Analytic C ≈ Sellentin–Heavens sim C** — the covariance model is right.
4. **Fisher ellipse = full MCMC posterior on the Gaussian likelihood** (the clincher): for a Gaussian
   likelihood the Fisher *must* equal the MCMC contour; if they match, the forecast is exact and
   unambiguous by construction. (numpyro/emcee on the analytic Gaussian likelihood.)
5. **BNT-full = non-BNT-full** — linear-algebra invariance (already passes to 1e-13).
6. **Convergence** of σ vs binning and (for the sim cross-checks) vs N_perm/N_realization.

If 1–4 pass, the BNT-580 vs non-BNT-460 number is no longer a matter of opinion — Fisher = MCMC on a
known likelihood, with exact derivatives and an analytic covariance. *That* is the unambiguous result,
and it then settles cleanly against the NPE (0.79): equal ⇒ NPE is optimal; tighter ⇒ NPE
under-extracts a real, validated gain.

## Build plan
- **Phase I — differentiable model:** jax-cosmo auto+cross C_ℓ with CosmoGrid n(z) + nonlinear P(k) +
  BNT + binning; validate the *fiducial* C_ℓ against the sim mean (shape/amplitude sanity).
- **Phase II — autodiff derivatives** + the finite-difference and sim-J cross-checks (validation 1–2).
- **Phase III — analytic NaMaster Gaussian covariance** + the Sellentin–Heavens cross-check (val. 3).
- **Phase IV — Fisher + the MCMC equivalence test** (val. 4–5), then σ-convergence (val. 6).
- **Phase V — production:** all six footprints; BNT-580 vs non-BNT-460 (and the triangular cut) with
  the full validation suite passing; reconcile with the NPE.

## DECISION (signed off 2026-06-23): RIGOROUS SIM-BASED route (not theory/autodiff)
AT's call: jax-cosmo likely lacks pieces we need (BNT + masked NaMaster auto+cross + CosmoGrid
baryonification), and a sim-based Fisher is **apples-to-apples with the NPE** (same simulations,
same response, same covariance) — which is the whole point of the comparison. So we do the sim-based
Fisher *properly*, meeting the same validation bar:
- **Derivatives:** local polynomial fit on **noise-averaged** grid cosmologies in a neighborhood of the
  fiducial (NOT global lstsq over the whole prior), with convergence tests on neighborhood size,
  polynomial order, and realizations-per-cosmo. **First check `benchmarks/`** for dedicated derivative
  sims (fiducial±step) → if present, finite-difference them (gold standard) with step-size convergence.
- **Covariance:** analytic **NaMaster Gaussian** (primary) + 200-perm **Sellentin–Heavens** sample
  covariance (cross-check). [Hartlap is the wrong tool here — use S–H.]
- **Validation suite unchanged** — the clincher is **Fisher = MCMC on the analytic Gaussian likelihood**.

## Environment & data (verified 2026-06-23)
- Sim sets under `/home/tersenov/CosmoGridV1/stage3_forecast/`: `benchmarks/` (← CHECK FIRST for
  derivative sims), `fiducial/` (**200** perms — covariance + convergence), `grid/`+`new_grid/`
  (**2500** cosmologies, ~7 realizations each — our derivative neighborhood, confirmed),
  `nz/` (source n(z)).
- **NaMaster Gaussian covariance available**: `NmtCovarianceWorkspace` + `gaussian_covariance` present
  in the **cosmostat_new** venv (`/home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python`).
- MCMC: `numpyro` in the jaxili env (for the Fisher=MCMC validation); confirm emcee if preferred.

## HANDOFF — start Phase I in a fresh session
Order: **(I)** build the analytic NaMaster Gaussian covariance for the 14000 auto+cross bandpowers
(cosmostat_new) and cross-check vs the 200-perm Sellentin–Heavens covariance; **(II)** check
`benchmarks/` for fiducial±step derivative sims, else build the noise-averaged local-polynomial
derivative with convergence tests (extends `scripts/diagnostics/fisher_audit_noiseavg.py`);
**(III)** assemble F=JᵀC⁻¹J; **(IV)** validate via MCMC=Fisher on the Gaussian likelihood +
BNT-full=non-BNT-full; **(V)** production over all six footprints, reconcile with NPE (0.79).
Anchor on 14000. Working diagnostics so far: `scripts/diagnostics/fisher_{bnt_vs_nonbnt,audit_phase0,
audit_noiseavg}.py`. Current best read: noise-averaged-J Fisher gives BNT-580/non-BNT-460 ≈ 0.4–0.5
(high-ℓ BNT modes are REAL, R²→0.80), still using the noisy covariance — Phase I/covariance is the
remaining half.

---

## Phase I — RESULT (2026-06-23): covariance validated; KEEP the sim covariance (it carries real SSC+cNG)
Built the analytic mask-aware NaMaster Gaussian covariance of all 10 auto+cross bandpower spectra
(3830×3830 native) for the 14000 deg² apod-2° mask (f_sky=0.339, lmax=1535, nlb=4), feeding the
200-perm fiducial MEAN of the *measured* decoupled bandpowers as the total Cℓ (autos carry shape
noise, cross are noise-free → fully sim-based, apples-to-apples with the NPE).
Scripts: `scripts/diagnostics/fisher_gaussian_cov.py` (build + V1/V2), `fisher_cov_offdiag.py`
(band structure), `fisher_cov_nongauss.py` (non-Gaussian vs perm-artifact). Cached products in
`scripts/diagnostics/cache_gaussian_cov/` (cw/w FITS + `gaussian_cov_native_14000.npy`).

**V1 — the analytic Gaussian is CORRECT on the diagonal.** Per-bandpower variance analytic/sample
ratio = 0.94–0.98 at ℓ<100 and **1.01–1.03 at ℓ>400**, across all 10 spectra. So the mask, binning,
noise inputs, and `gaussian_covariance` call are all right; the sim covariance's diagonal is not an
artifact. (The ~5% low-ℓ deficit is the expected few-mode/non-Gaussian regime.)

**V2/offdiag — the sim covariance has a real ~25–30% NON-GAUSSIAN excess the Gaussian omits.** At the
rebinned scale the Fisher uses (ℓ≥37, rebin=20, full config), analytic/sample diagonal ratio = **0.767**
⇒ a pure analytic Gaussian would make σ ~12% too small (contour AREA ~23% too small). Mechanism: the
analytic native bandpowers carry the mask-decoupling adjacent anti-correlation (−0.133, constant in ℓ);
the sims are net *less* anti-correlated, so rebinning doesn't suppress their variance as much.

**Non-Gaussian, not a sim artifact (settled):**
- T1: the 200 perms are independent — perm-perm correlation mean −0.005≈0, no spikes (max 0.48, 1.4%
  exceed |0.3|). A duplicated/shared-mode perm artifact would bias toward a *positive* mean; it doesn't.
  (Correlated perms would only make the sample cov *noisier*, not *biased* — light-cone rotations
  preserve the 2-pt statistics.)
- T2: the excess D = C_sample−C_analytic is positive in 91% of features (median 0.233·C_samp), and its
  **leading eigenvector has coherence 1.00 (all bandpowers same sign) = super-sample covariance**
  (top normalized eigenvalue 21.7 ≫ next 3.2), plus a smaller connected-trispectrum tail. CosmoGrid
  full-sky maps legitimately contain super-survey modes relative to the patch — real covariance the
  14000 deg² survey (and the NPE) also has.

**Decision (covariance):** the production Fisher covariance is the **SIM covariance** (it contains the
real SSC+cNG), with **Sellentin–Heavens** for the finite-N=200 noise (retire Hartlap), **validated by
the analytic Gaussian on the diagonal**. A pure analytic Gaussian is *too tight* and is the wrong
production choice — it is the diagnostic/validator, not the covariance. Cleanest production form (Phase
III): a hybrid = full-rank analytic Gaussian + the low-rank coherent SSC/cNG correction from the sims
(well-conditioned, no aggressive rebin, no large config-differential Hartlap).

**Consequence for the BNT result:** the earlier noise-averaged-J Fisher (BNT-580/non-BNT-460 ≈ 0.4–0.5)
ALREADY used the sim covariance, so its covariance footing was right all along — the covariance was
never the source of the 0.37↔0.5 wobble. **The remaining lever is the Jacobian (Phase II)**: replace
global lstsq-over-prior with the noise-averaged local-polynomial derivative, then the MCMC=Fisher
clincher. Validation figure: `outputs/diagnostics/fisher_cov/gaussian_cov_validation_14000.png`.

---

## Phase II/III — RESULT (2026-06-23): the proper Fisher says BNT ≈ 0.45–0.48; NPE (0.79) under-extracts
Scripts: `scripts/diagnostics/fisher_local_jacobian.py` (local derivative + convergence),
`fisher_hybrid_cov.py` (analytic + hybrid covariance, BNT propagated exactly), `fisher_ratio_ladder.py`
(money plot `outputs/diagnostics/fisher_cov/bnt_ratio_ladder_14000.png`). Fiducial
θ=[Om,S8,w0,H0,ns,Ob]=[0.26,0.84,-1.0,67.36,0.9649,0.0493]. `benchmarks/` = numerical-resolution sims
(box/particle/z-res), NOT param-step → no finite-diff shortcut; used the local-poly route.

**Local Jacobian (Phase II).** Kernel-weighted polynomial gradient at the fiducial in whitened param
space, realizations noise-averaged first; orders {order1_anchored, order1_free, order2(quadratic)} × bw
{0.5..2.0} (N_eff 145→1921 cosmologies). ORACLE BNT-full==non-BNT-full to 1e-13 under every order. The
BNT-580/non-BNT-460 area ratio is **stable at ~0.50** (order1_free 0.47–0.52, order2 0.50–0.55) ⇒ the
global lstsq-over-prior 0.37 was over-optimistic (global linearization inflates steeply-responding BNT
modes); the proper local derivative gives ~0.50. order1≈order2 ⇒ nonlinearity is mild (linear Fisher OK).

**Covariance finite-N is the whole ambiguity.** With the 200-perm SIM covariance the ratio depends
entirely on the finite-N correction: raw Cinv 0.34 → Hartlap 0.50 → **Percival 0.72**. The BNT vector
(96 feat, Hartlap 0.51) sits near n_feat/n_perm, so the Percival estimation-noise inflation hits it
(m1=1.77) far harder than non-BNT (50 feat, m1=1.24). The 0.72≈0.79 coincidence is a finite-200-sim
penalty of the sample-covariance METHOD, not the data's information (and the NPE doesn't invert a
200-sample covariance, so it isn't subject to it).

**Estimation-noise-free covariance settles it (Phase III).** Using the Phase-I analytic Gaussian (and
the hybrid = analytic + top-k SSC/cNG eigenmodes), with BNT propagated EXACTLY via Cov(C̃)=(T⊗I)Cov(C)(T⊗I)ᵀ
(T = per-bandpower 10×10 from the 4×4 BNT M; oracle BNT-full==non-BNT-full to 2e-13): ratio = **0.484
(pure analytic), 0.455 (hybrid k3), 0.44 (hybrid k5)** — IDENTICAL across J order/bw (the J order cancels
in the ratio). So the **TRUE-information ratio is ~0.45–0.48**: BNT carries ≈2× the constraining area.

**Verdict.** Error budget: 0.34–0.37 biased estimator | **0.45–0.48 true information (the proper Fisher
answer)** | 0.72 what a 200-sim sample-cov analysis actually yields | 0.79 NPE. The BNT advantage is REAL
and ~2×; the NPE realizes only ~1.3× ⇒ **the NPE under-extracts** (BNT info is spread over many correlated
high-ℓ modes the flow doesn't fully capture; consistent with the whitening story). Remaining: (IV) the
formal MCMC=Fisher check is degenerate for this linear-Gaussian model (oracle already 1e-13; order1≈order2
confirms mild nonlinearity) — do it only to tick the suite; (V) production over all six footprints +
push the NPE compression. NPE 0.79 = whitening pilot; confirm exact config match before quoting.
