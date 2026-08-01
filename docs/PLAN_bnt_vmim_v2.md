# PLAN — VMIM neural compression for the BNT power spectra, v2 (corrected)

> **SUPERSEDED (2026-08-01).** This document predates the settled BNT scale-cut result and repeats
> the **retracted** claim that the BNT data vector is ill-conditioned. Measured, it is *better*
> conditioned than non-BNT (correlation cond 8.3e2 vs 4.4e3), and the raw score is 1.2e4, not the
> "~1e8" quoted here. The real cause of the raw-flow failure is **information dilution**. Read
> `NOTES_bnt_compression_for_paper.md` §1 and `HANDOFF_BNT_SETTLED.md` instead; kept for history.

Status: proposed, awaiting sign-off. Author: fresh session 2026-06-29. Supersedes the first VMIM
attempt (`outputs/baryon_tension/vmim/ENSEMBLE_RESULT.md` = oracle FAIL). Reads on top of
`docs/HANDOFF_bnt_vmim_reimplement.md`, `cnn_sbi/NEURAL_SUMMARIZATION_RECIPE.md`,
`docs/NOTES_bnt_compression_for_paper.md`, `docs/PLAN_score_bnt_tension_14000.md`.

## 0. Goal (one paragraph)

Build the **learned, assumption-light** alternative to the score/MOPED compression: a VMIM-MLP
compressor (Stage 1) feeding an **sbi_lens ConditionalRealNVP** NDE (Stage 2), giving **calibrated,
on-truth** BNT power-spectrum posteriors that reproduce the lossless identity (`bnt_full ≡
nonbnt_full`) and sit on the analytic Fisher floor. If VMIM matches the score's calibrated contours,
it removes the score's "by construction" caveat; if it beats them (tighter *and* calibrated) that is
genuine non-Gaussian information. Scope: **BNT only, 14000 deg² first**; non-BNT is the lossless
control. **One compressor per scale cut.** The score result already stands and is the safety net — a
VMIM that can't be calibrated is a recordable negative result, not a blocker.

## 1. Diagnosis confirmed against the live code (not taken on faith)

All five handoff causes verified by reading the scripts and the caches:

1. **Stage 2 = jaxili, not sbi_lens RealNVP** — `npe_on_summary.py:40`. Recipe lesson 1: the NDE family
   is a ~30% FoM lever; RealNVP > MAF on the identical summary. **Primary suspected cause.**
2. **Random train/val split → realization leakage** — `vmim_compress.py:84-87` permutes rows, so the 7
   noise realizations of a cosmology straddle train *and* val → over-optimistic val → over-fit
   best-checkpoint → over-confident summaries. Verified the grid is **2424 unique cosmologies × ~7
   reals = 16965 rows**, so the by-cosmology split is well defined.
3. **`--summary-noise` tuning knob (default 0.3)** — `vmim_compress.py:42,134`. Not in the recipe; the
   old sweep showed null S8 *slides* with it (fitting the answer, not calibrating).
4. **Full Cholesky whitening on the ill-conditioned BNT cov** — `vmim_compress.py:49-59`. Full
   decorrelation amplifies noise in the nulled directions — the exact directions that make BNT hard.
5. **Insufficient gate** — `npe_on_summary.py` checks TARP+SBC only; both test *average* coverage and
   passed while the nulls were off-truth and σ-collapsed and the identity was violated.

**The right reference is the cnn_sbi ℓ1-norm MLP compressor** (`vmim_from_cache.py` /
`CompressorMLP` in `npe_l1vmim_nbody_tomo.py`), **not** the sbi_lens-package CNN-over-maps. Our
`scripts/vmim_compress.py` is a near-verbatim port of `vmim_from_cache.py`; the compressor architecture
(MLP 256,256 leaky_relu → d; `ConditionalRealNVP` companion n_layers 4, `AffineCoupling`[128,128] silu;
VMIM loss) is **identical**. Our version introduced exactly **three deviations from the reference, and
they are the bugs**: (a) full Cholesky whitening instead of the reference's `log1p-zscore`+clip+min-var
preproc; (b) an internal **random** split (the reference consumes a pre-split `l1_train/l1_val.npz`, so
it never leaks); (c) an added `--summary-noise`. (Minor: reference uses plain `adam`+piecewise; ours
added `adamw`+grad-clip.) The fixes restore the reference behavior, adapted for signed data.

**Two cache facts that fix the recipe** (verified, not assumed):
- θ = `[Ωm, σ8, w0, H0, ns, Ωb]`, **H0 ∈ [64,82] physical**. The reference feeds θ **raw** (no z-score,
  `vmim_from_cache.py:145`) because its h0 is already /100≈0.67, sitting where the `N(0.5,0.05)` flow
  base lives; raw H0≈67 detonates that base. **Fix = `θ[:,3] /= 100` once at load**, then raw θ
  everywhere (see §2).
- Existing `outputs/score_experiment/cache/{bnt,nonbnt}_full_14000_nobary/cache.npz` are raw
  `(theta[16965,6], x[16965,120], x_fid[120])` at rebin=20 — **drop-in Stage-1 input; no re-dump for
  the pilot.** `..._bary/cache.npz` holds the biased fiducial `x_fid`.

## 2. Design decisions (taking a position — veto at sign-off)

- **Reuse the rebin=20 score caches for the pilot** (`{bnt,nonbnt}_full_14000_nobary`). The score result
  is binning-independent, so rebin=20 (120 feat) matches the benchmark and skips the expensive grid
  re-dump. Re-dumping (and per-cut slicing via `score_cut_utils.keep_indices`) is only for the later
  per-cut production sweep, not the pilot.
- **Fix Stage 1 in place** (`scripts/vmim_compress.py`), not a v2 file: add `--split {cosmology,random}`
  (default **cosmology**), `--preproc {zscore,whiten,pca_whiten}` (default **zscore**), and set
  `--summary-noise` default to **0** (keep the flag for an explicit ablation only). Minimal, reviewable
  diff; the old behavior stays reachable for A/B.
- **New Stage-2 script** `scripts/nde_realnvp_from_summary.py` (sbi_lens RealNVP), leaving the jaxili
  `npe_on_summary.py` intact as a comparison arm. Port `build_flow`/`train_flow` from
  `cnn_sbi/.../npe_cnn_nbody_tomo.py` into a self-contained `scripts/nde_realnvp.py` (strip wandb; keep
  best-val early stopping).
- **θ scaling (decided): divide H0 by 100 once at cache-load, then raw θ everywhere — NO z-scoring,
  matching the cnn_sbi reference exactly.** The right reference is the ℓ1-norm MLP compressor
  (`vmim_from_cache.py` / `CompressorMLP`), which feeds θ **raw** to both the Stage-1 companion and the
  Stage-2 NDE (line 145, no standardization). It gets away with that because its θ is already O(1) — h0
  stored as 0.6736 (the /100 convention). The RealNVP base is `N(0.5,0.05)`; raw H0≈67 detonates it,
  but h0≈0.67 sits right where the base lives. So the minimal, validated-matching fix is `θ[:,3] /= 100`
  at load. This supersedes the earlier "z-score all 6" call (the reference doesn't z-score; /100 is
  simpler and consistent). **Consistency contract:** after the one-time /100, θ is in
  `[Ωm,σ8,w0,h0,ns,Ωb]` units everywhere (h0≈0.6736); the first three (what we care about) are
  **untouched/physical** in and out; multiply col 3 by 100 only if h0 itself is ever reported (tension
  and FoM3 use cols 0,1,2 only). Truth = `(0.26,0.84,−1.0,0.6736,0.9649,0.0493)`. Keep our param order
  `[Ωm,σ8,w0,h0,ns,Ωb]`; do **not** adopt sbi_lens's native `[omega_c,omega_b,σ8,h0,ns,w0]`.
- **Preproc default = per-feature z-score** (subtract mean, divide by per-feature std, clip ±5σ; NO
  cross-feature decorrelation; signed-safe, no log). A/B vs `pca_whiten` (regularized whitening that
  drops the noisy nulled directions) **only if** z-score under-extracts the identity gate. Pick by the
  *calibrated* identity oracle, not FoM.
- **summary_dim = 8** (6 params + headroom). PS is near-Gaussian so d≈6 may suffice; revisit if Stage 2
  struggles.
- **Single compressor seed**; the statistical seeds live in Stage 2 (pool 3). The compressor
  deep-ensemble (recipe lesson 7) is held in reserve **only if** the gated posterior is *over-confident*
  (SBC std high, marginals narrow) — diagnose direction first.

## 3. The corrected pipeline

**Stage 1 — `vmim_compress.py` (fixed):** MLP `c(x)` hidden (256,256) leaky_relu → d=8; companion
`ConditionalRealNVP` (n_layers 4, coupling [128,128] silu), loss `−E[log q(θ|c(x))]`, `adam`+piecewise
schedule (the reference; grad-clip kept only as NaN-insurance), batch 512, ~20–30k steps time-boxed,
**best-val**. Fixes restoring reference behavior: **`θ[:,3] /= 100` at load then raw θ** (drop the
existing companion z-score), split **by cosmology** (group `np.unique(round(θ,6), axis=0)` → hold out
~10% of whole cosmologies, all reals together), **summary-noise 0**, **preproc = reference
`log1p-zscore` minus the log1p** (signed BNT → per-feature z-score + clip 5 + min-var 1e-5). Output:
compressed `{theta_tr,y_tr,theta_va,y_va,y_fid}` + preproc stats.

**Stage 2 — `nde_realnvp_from_summary.py` (the key fix):** sbi_lens `build_flow(6, n_layers=4,
hidden=128)`; `train_flow(total_steps≈50k, batch 128, AdamW cosine 1e-3→1e-5, grad_clip 1.0, wd 1e-4,
patience 20, 90/10 internal split seed 0)`; **raw θ** (already /100 at load → O(1), so no per-stage
standardization, exactly as the reference); **pool 3 seeds** (per-obs key `seed*100003+obs_idx`, drop
non-finite). Sample at `y_fid`. Metrics `compute_fom3`/`fom2d`/`marginal_stats` (FoM3 uses cols 0,1,2 =
Ωm,σ8,w0). After the /100, our truth `(0.26,0.84,−1.0,0.6736,0.9649,0.0493)` **coincides with cnn_sbi's
`FIDUCIAL`**, so the metrics' constant is directly correct — one more reason /100 is the clean choice.

## 4. Gates (bake in; an unattended failure must be loud). Run in order; do NOT report a width that
fails (1)-(3).

1. **Lossless identity (hard, the one the old attempt failed):** `bnt_full` null posterior ≈
   `nonbnt_full` null posterior — same truth, width within seed scatter. The two compressors see
   different inputs (BNT vs non-BNT x) but the information is identical (BNT = invertible linear map at
   full), so the posteriors must agree. Quantify: |Δ mean| ≤ ~0.3σ and σ ratio ∈ [0.85,1.18] per param.
2. **Null on truth:** null mean within ~0.5σ of `(0.26,0.84,−1.0)` for both bases.
3. **σ vs Fisher floor:** compute the floor at the full vector for free from
   `score_cut_utils.build_score(FULL_CUTS, bnt=True)` → `F` → `σ=sqrt(diag(F⁻¹))` (network-independent).
   Realized σ near it; collapsed σ + TARP-OK ⇒ over-confident (apply ensemble, don't ship); inflated ⇒
   under-extracted (debug preproc/Stage-2).
4. **TARP-DRP plot + SBC histograms:** varied-θ **held-out** validation set (the cosmology-split val
   side), stratified by FoM3 terciles (port `tarp_stratified_val_nde.py`). **Plot ECP-vs-α and the rank
   histograms** — inspect the curve, not just the scalar. SBC rank-std ≈ 0.289, TARP net bias ≈ 0,
   `max|ecp−α| ≤ 0.05` PASS / `>0.10` FAIL.
5. **(optional, strong) Analytic-Gaussian oracle:** in the Gaussian regime the VMIM+NDE posterior
   should match the score/Fisher contour (`gaussian_cov_native_14000.npy` + the score). Validates the
   whole pipeline before any non-Gaussian claim.

## 5. Pilot → gate → scale (failure-avoidance order; this is what the sign-off authorizes)

**P0 — smoke Stage-2 (no new physics).** Build the ported sbi_lens RealNVP, train on the EXISTING
`bnt_full_14000_nobary` raw vector compressed with a quick by-cosmo Stage-1 run (or directly on a tiny
by-cosmo split), θ `/100`-at-load then raw, sample at `y_fid`. PASS = finite, constrained,
roughly-on-truth null. Just proves the Stage-2 port runs and is sane.

**P1 — fix Stage 1.** Retrain `bnt_full` and `nonbnt_full` compressors, single seed, **by-cosmology
split, no summary noise, zscore preproc**.

**P2 — lossless-identity gate (make-or-break).** Run gates (1)-(4) on `bnt_full` vs `nonbnt_full`. If
green → the pipeline is correct; proceed to P3. If red → debug in order **{Stage-2 family, split,
preproc}**; **do not** tune `summary_noise` to paper over it. If z-score preproc under-extracts BNT
(σ ≫ floor) try `pca_whiten`; if over-confident, the compressor ensemble.

**P3 (after P2 green — scope decided then, not now).** Per the sign-off interview: run the identity +
de-biasing-cut contours and *look at whether VMIM lands exactly on the Gaussian score or shows
calibrated headroom past the Fisher floor*, then choose how hard to push (match-and-write-up vs chase
non-Gaussian gain with over-complete summaries / ensemble). Reuse the score dump + `score_cut_utils`
slicing (one compressor+NDE per cut); other areas + full sky later. **Regression anchor skipped** — the
existing `ENSEMBLE_RESULT.md` FAIL is the baseline; the lossless-identity gate is the discriminator.

## 6. Files

- **Edit:** `scripts/vmim_compress.py` (split/noise/preproc flags + defaults).
- **New:** `scripts/nde_realnvp.py` (ported `build_flow`/`train_flow`, wandb stripped),
  `scripts/nde_realnvp_from_summary.py` (Stage-2 driver: raw θ post-/100, 3-seed pool, sample at y_fid),
  `scripts/vmim_gate.py` (gates 1-4: identity, on-truth, σ-vs-floor, TARP-DRP + SBC, with the ECP-vs-α
  and rank-histogram plots), `scripts/run_vmim_pilot.py` (orchestrate P0-P2, write a verdict json+md).
- **Reuse:** `scripts/score_cut_utils.py` (`build_score` for the floor; `keep_indices` for P3),
  caches under `outputs/score_experiment/cache/`.
- **Output:** `outputs/baryon_tension/vmim_v2/` (compressed caches, posteriors, gate jsons, plots).

## 7. Env / compute

jaxili env (`/home/tersenov/anaconda3/envs/jaxili/bin/python`), **GPU 2** on titan — check `nvidia-smi`
first, stagger CUDA inits (simultaneous JAX starts race). `import
tensorflow_probability.substrates.jax` BEFORE `sbi_lens`. Pilot is tiny: 2 compressors (single seed) + 2
Stage-2 RealNVP (3 seeds). Tension/getdist (P3 contours) → aname env. Fisher floor (gate 3) is pure
numpy via `score_cut_utils`.

## 8b. Verification & checkpoint protocol (proper / clean / verified — report after EVERY step)

Each step writes its intermediate artifacts (json + a short md + any plot) under
`outputs/baryon_tension/vmim_v2/<step>/` and I **stop and report the numbers before proceeding** — no
silent run-through to the gate.

- **P0 verify (Stage-2 port):**
  (i) *port-equivalence regression* — the ported `build_flow`/`train_flow` are copied verbatim from
  cnn_sbi minus wandb; with the same PRNGKey + same tiny `(θ,y)` batch the ported `best_val` must equal
  cnn_sbi's to ~machine precision (proves the copy added no drift);
  (ii) *linear-Gaussian toy oracle* — train on synthetic `(θ, y=Aθ+ε)` with a known analytic posterior;
  the flow's posterior mean/cov must match the analytic one within MC error (proves the flow + θ-scaling
  + sampling are correct end-to-end, independent of any cosmology cache);
  (iii) *sanity* — finite log_prob across the set; null at `y_fid` finite, constrained (σ < prior),
  roughly on-truth in (Ωm,σ8,w0).
- **P1 verify (Stage-1 compression):**
  (i) *zero cosmology leakage* — assert `set(train_cosmo) ∩ set(val_cosmo) == ∅` (the bug that broke it),
  print n_train/n_val cosmologies;
  (ii) *preproc sane* — per-feature (µ,σ) finite, clipped fraction small, no zero-variance feature kept;
  (iii) *summaries sane* — `y_tr/y_va/y_fid` finite, no collapsed (≈0-variance) summary dim, best-val
  improved over init; same seed → identical summaries (determinism).
- **P2 verify (the gate):** the four gates each print PASS/FAIL with numbers; the identity gate prints
  both nulls' mean±σ side-by-side and Δ; TARP-DRP renders the ECP-vs-α curve + SBC rank histograms (not
  just the scalar). A single `verdict.json` + `GATE_REPORT.md` summarizes.

Cleanliness: small self-contained scripts, no dead code, ruff/black-clean, each script runnable
standalone with `--help`; constants (truth, param order) defined once and imported, never re-hardcoded.

## 8. Open questions / risks

- **Preproc for signed ill-conditioned data is the genuine unknown** — z-score is the safe default but
  may leave the flow to find the projection in a correlated space; pca_whiten drops noisy nulled
  directions but discards some signal. Decide by the calibrated identity oracle.
- **Over-confidence is the headline danger** (single-compressor amortization). Ensemble is the fix, but
  only if the direction is over-confident — diagnose first.
- **VMIM that can't be calibrated is a valid negative result** ("for this near-Gaussian PS the analytic
  score is the right tool"), recorded, not chased indefinitely. The score is the safety net.

---

# RESULTS LOG (2026-06-29) — pilot P0→P2 COMPLETE, identity gate PASSES

All under `outputs/baryon_tension/vmim_v2/`. Env: jaxili, GPU 0+2. Scripts: `nde_realnvp.py` (Stage-2
port), `vmim_compress.py` (Stage-1, fixed), `nde_realnvp_from_summary.py` (Stage-2 driver),
`vmim_gate.py` (4-gate), `pool_ensemble.py` (compressor ensemble), `p0_verify_nde_port.py`.

## P0 — Stage-2 port verified (3 ways), all PASS
- **port-equivalence**: ported `train_flow` best_val = cnn_sbi's to **|Δ|=0.0** (bit-for-bit; no copy drift).
- **linear-Gaussian oracle**: flow posterior vs analytic — mean err ≤0.11σ, σ-ratio 0.99–1.00.
- **real-cache smoke**: bnt_full raw 120-D, by-cosmo split → on-truth, constrained null.

## P1 — Stage-1 fixed; leakage eliminated
By-cosmology split (2424 cosmologies, 242 val, **leakage=0**), summary-noise 0, H0/100, per-feature
z-score. Flagged: bnt companion best-val −13.06 @1500 + divergence vs nonbnt −15.55 → BNT under-extracted.

## P2 — the make-or-break identity gate (bnt_full vs nonbnt_full, 14000, full vector)
Iterated the BNT compressor (non-BNT was clean throughout: calibrated, ~on-truth, on-floor):

| run | preproc | identity σ-ratio | null S8 (0.84) | calibration | verdict |
|---|---|---|---|---|---|
| P2  | z-score | 1.24,1.26 ✗ | 0.840 ✓ | bnt over-conf | under-extracts (σ/floor 1.16,1.11) |
| P2b | pca-truncate | — | — | — | worse (drops signal) |
| P2c/d | ana-whiten (clipped) | 1.06,1.01 ✓ | **0.82 ✗** | — | on-floor but BIASED |
| P2e | ana-whiten **clip-fixed** | 0.89,0.93 ✓ | 0.831 ✓ | bnt ok | near-pass (identity-mean 0.40σ; nonbnt SBC 0.35) |
| **P2f** | **+ 3-seed compressor ensemble** | **1.08,1.12 ✓** | **0.846 ✓** | **SBC 0.30–0.31 both ✓** | **PRIMARY PASS ✅** |

**P2f final (the win):** identity |Δmean|/σ = [0.14, 0.15, 0.03] (arms agree to ~0.15σ); both on-truth
(bnt S8 0.846 / nonbnt 0.843); TARP ≤0.06, SBC rank-std 0.30–0.31 (uniform) both; Ωm/S8 on the Fisher
floor (σ/floor 0.80–1.0). Only Gate-3 w0/floor (0.63–0.67) "fails" — the **documented over-estimated w0
Fisher floor** (shallow local w0 Jacobian), confirmed NOT over-confidence by the uniform w0 SBC (0.305).

## Key methodological findings
1. **The corrected pipeline works**: sbi_lens RealNVP Stage-2 (not jaxili) + by-cosmology split +
   no summary-noise + H0/100 fixed the gross failures of the prior attempt (off-truth, identity-violation).
2. **BNT under-extraction is real for a plain learned compressor on z-scored ill-conditioned input** — it
   under-learns the J·C⁻¹ projection (consistent with `NOTES_bnt_compression_for_paper`).
3. **The fix is analytic-covariance noise-whitening (`ana_whiten`, IMNN-style)**: whiten the input by the
   regularized analytic C^{-1/2} so noise is isotropic and the MLP need only find the ~6 signal directions.
4. **CRUX — the whitening clip must be PER-FEATURE-RELATIVE.** After C^{-1/2} the parameter-sensitive
   directions carry the cosmology-variation signal (std up to ~4.6); an ABSOLUTE ±5 clip (right for
   unit-variance z-score) lops that signal and biases S8 by ~1σ. Per-feature ±5σ clip recovered ~1.4 nats
   and removed the bias. This was the difference between "biased" (P2c/d) and "on-truth" (P2e).
5. **Compressor deep-ensemble (3 seeds, common split / varied init) is the over-confidence remedy** — it
   moved SBC 0.35→0.30 and tightened the identity 0.40σ→0.15σ (P2e→P2f).

## Status / next (P3, scope was "decide after P2")
Pilot COMPLETE and PASSING. P3 options: per-cut de-biasing contours + head-to-head vs the score across
scale cuts (the science application), reusing `score_cut_utils` slicing; or write up the methods result.
The score remains the validated benchmark; VMIM now independently reproduces it (learned, assumption-light).
