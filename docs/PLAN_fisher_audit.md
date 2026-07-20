# Plan: audit the Fisher pipeline and produce trustworthy Fisher forecasts

**Status:** proposed, awaiting sign-off. Created 2026-06-23.

## Why now
The Fisher forecast (BNT bin1-580 vs non-BNT-460: area 0.37×, σ(S8) 0.63×, σ(Ωm) 0.58×) disagrees
sharply with **two independent SBI methods** — whitening-NPE and (over-confidence-corrected) VMIM-NPE
both give ~0.79. The disagreement is concentrated in the **bins 2–4, ℓ≈460–1024** modes that BNT-580
keeps and non-BNT-460 cuts — and those modes are **noise-dominated** (high-ℓ autos = shape noise;
cross variance ∝ N_i·N_j at high ℓ). That is exactly the regime where a Fisher built from a noisy
least-squares Jacobian + a 200-perm sample covariance + a large, config-differential Hartlap factor is
least reliable. So either the forecast over-promised (most likely, given two simulators agree against
it) or the NPE is leaving real information on the table. We will not trust the Fisher — for the BNT
question or the paper — until every input survives an audit. This plan audits each, rebuilds the
Fisher on rigorous footing (chiefly an **analytic mask-aware Gaussian covariance**), and cross-checks
it against the NPE.

## What we have (and its weak points)
- `scripts/diagnostics/fisher_bnt_vs_nonbnt.py` — F = JᵀC⁻¹J, masked submean, 14000 deg².
  - **J** = *global* least-squares of C_ℓ vs params over the 16965-cosmo grid — an average response over
    the whole prior, **not the local derivative** at the fiducial; noisy for low-SNR modes; no R² gate.
  - **C** = 200 fiducial perms, Hartlap-corrected, **rebin=20** (chosen only so n_feat < n_perm).
  - Oracle BNT-full = non-BNT-full to 1e-13 ⇒ the **linear algebra is correct**; the suspects are the
    **inputs J and C**.
- `scripts/run_fisher_forecast_auto_cross_ps.py` — the older Fisher entry point (has `--bnt/--upper-cut/
  --rebin` but not `--masked/--subtract-mean/--lmax/--upper-cuts`). Reconcile/retire as part of Phase 0.

## Identified failure modes (the audit targets)
1. **Jacobian** — global linear fit ≠ local derivative; conflates nonlinearity (esp. w0) with the
   gradient; noisy & over-weighted for low-SNR high-ℓ modes; no R²/linearity diagnostics.
2. **Covariance** — 200 perms for ~50–240 features ⇒ poorly conditioned; Hartlap large *and differs
   between configs* (0.51 BNT-580 vs 0.74 non-BNT-460) so it distorts the *relative* result;
   Sellentin–Heavens not used; no analytic cross-check; estimation noise inflates Fisher info.
3. **Binning** — rebin=20 (Fisher) ≠ rebin=10 (NPE); convergence never established.
4. **Likelihood** — Gaussianity assumed, unchecked.
5. **No external validation** — never cross-checked against the NPE (where both should agree) or an
   analytic Gaussian covariance.

## Phases (each with a quantitative oracle)

### Phase 0 — Reproduce & instrument
Reproduce the 0.37 baseline; log per config: n_feat, cond(C), Hartlap factor and its differential,
per-bin SNR (signal/√Cov_diag) and per-bin Jacobian R². Re-confirm BNT-full = non-BNT-full (linear-
algebra correctness). Decide the fate of the two Fisher scripts (one audited engine going forward).
*Oracle:* baseline reproduces; BNT-full = non-BNT-full to ~1e-12.

### Phase A — Jacobian audit & fix
- Per-bin **R²** of the linear fit → map which modes (especially bins 2–4, ℓ>460) are well vs poorly
  determined.
- Compare derivative estimators: (i) global lstsq (current); (ii) **local** estimate (kNN/locally-
  weighted polynomial; gradient at the fiducial); (iii) **finite differences** if CosmoGrid has
  dedicated fiducial±step (derivative) sims — check.
- Sensitivity: does the BNT gain change with the estimator? Does it survive **R²-gating** (drop bins
  whose derivative is undetermined)?
*Oracle:* the BNT gain is stable across derivative methods and robust to R²-gating — or we've found it
isn't (and learned which modes are spurious).

### Phase B — Covariance audit + the analytic Gaussian covariance ← the main lever
- **Convergence:** σ(θ) vs N_perm (subsample 50/100/150/200; pull more fiducial perms if they exist).
  Does Fisher stabilize, or is it still moving at 200?
- **Conditioning & estimator:** cond(C); Hartlap vs **Sellentin–Heavens** (marginalizes covariance
  uncertainty — the right correction for limited perms); **Ledoit–Wolf shrinkage**. Quantify each on σ.
- **Analytic mask-aware Gaussian covariance via NaMaster** (`nmt.NmtCovarianceWorkspace` +
  `gaussian_covariance`): build the covariance of the auto+cross bandpowers from the fiducial C_ℓ +
  shape-noise + the mask's mode-coupling. Perfectly conditioned, no perm noise, the *right* covariance
  for masked PS in the Gaussian limit. **Adopt as the production covariance**; keep the 200-perm sim
  covariance as a cross-check.
*Oracle (recipe §5):* analytic vs sim covariance agree where the Gaussian limit holds; σ converged vs
N_perm; Sellentin–Heavens ≈ analytic.

### Phase C — Binning convergence & matched comparison
- σ(θ) vs rebin (10/20/40) using the **analytic** covariance (which stays conditioned at fine binning).
- Produce the Fisher at **rebin=10**, matched to the NPE — now feasible (analytic C, no n_feat<n_perm
  limit).
*Oracle:* σ converges with rebin; rebin-10 and rebin-20 agree within the convergence tolerance.

### Phase D — Decompose the BNT gain & cross-validate against the NPE
- **ℓ-decomposition:** cumulative BNT-580 vs non-BNT-460 gain as bins-2–4 modes ℓ=460→1024 are added in
  chunks; correlate each chunk's contribution with its R² and SNR. Localizes real-signal vs
  over-counted-noise.
- **Direct NPE measurement (model-free):** run a BNT config with bins 2–4 *also* cut at 460 — it differs
  from BNT-580 only by the high-ℓ bins-2–4 modes — so BNT-580 minus this = the NPE-extractable info in
  exactly those modes. Compare to the Fisher's prediction for the same increment. (This is the one
  short NPE run flagged earlier.)
- **Anchor:** confirm trusted-Fisher ≈ NPE in a high-SNR regime (low-ℓ-only, or the full vector); if they
  agree there and diverge only at high-ℓ, the issue is pinned to the high-ℓ modes.
*Oracle:* the trusted Fisher's BNT gain matches the model-free NPE increment within errors.

### Phase E — Deliverable: the trustworthy Fisher
Production Fisher = analytic NaMaster Gaussian covariance + Sellentin–Heavens sim cross-check, local/
R²-gated Jacobian, converged & matched binning, with a written error budget. Re-state BNT-580 vs
non-BNT-460 and **reconcile with the NPE (0.79)**: either the trusted Fisher now agrees (forecast was
optimistic ⇒ whitening near-optimal, the likely outcome) or a gain survives every audit (real,
recoverable ⇒ push the NPE). Extend the audited engine to the other footprints if wanted.

## Oracles / back-pressure (summary)
- BNT-full = non-BNT-full (linear algebra) — already passes.
- σ converges vs N_perm and vs rebin.
- Analytic vs sim covariance agree in the Gaussian regime.
- Trusted Fisher ≈ NPE in high-SNR regimes; and matches the model-free high-ℓ NPE increment.
- R²/SNR gating: modes the Fisher can't determine must not drive the result.
- Sellentin–Heavens vs Hartlap vs shrinkage consistent.

## Environment
- Analytic NaMaster Gaussian covariance → **cosmostat_new venv** (pymaster).
- Fisher numerics → any numpy.
- NPE cross-checks → jaxili. (Tension/getdist → aname.)

## Decisions (signed off 2026-06-23)
1. **Depth:** FULL audit, Phase 0 → E in order (methodical; every input audited before any number is
   quoted).
2. **Scope:** ALL SIX footprints (2000/5000/10000/14000/28000/35000) from the start — paper-ready set.
3. **Production covariance:** analytic NaMaster Gaussian as primary, 200-perm sim (Sellentin–Heavens) as
   cross-check.
4. **Jacobian:** adopt the local / R²-gated derivative if it differs materially from global lstsq.

Anchor each phase's *development* on 14000 (fastest feedback), then run the validated step across all six.

## Progress log
- 2026-06-23: plan signed off. Starting Phase 0 (reproduce & instrument) on the 14000 anchor.
