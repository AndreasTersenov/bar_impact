# HANDOFF — re-implement VMIM neural compression for the BNT power spectra (correctly)

**Read this first, then `/home/tersenov/software/cnn_sbi/NEURAL_SUMMARIZATION_RECIPE.md` (the validated
recipe from AT's other project), then `docs/NOTES_bnt_compression_for_paper.md` (why the BNT vector is
hard).** Fresh-session handoff, 2026-06-26.

## 1. The one-paragraph goal

The BNT auto+cross power-spectrum data vector is ill-conditioned (the BNT nulling makes it
anti-correlated, signed, with information in many low-S/N high-ℓ modes — see `NOTES §1`). A raw
normalizing-flow NPE under-extracts it. We have ALREADY solved this with **score/MOPED compression**
(analytic `J`, `C` → 6 sufficient statistics), which gives calibrated, on-truth, Fisher-floor BNT
contours (`docs/PLAN_score_bnt_tension_14000.md`). The score works but is **"by construction"** — it
uses the analytic Fisher `J` and `C`, so "score reaches Fisher" is partly definitional, and it is only
Gaussian-optimal. **This task builds the learned, assumption-light alternative: a VMIM-MLP neural
compressor + an expressive NDE**, exactly as in AT's `cnn_sbi` recipe (which worked very well on
wavelet ℓ1 statistics). If a properly-done VMIM reproduces the score's calibrated contours, it removes
the "by construction" caveat and strengthens the paper; if it *beats* the score (tighter AND
calibrated), that is genuine non-Gaussian information the Gaussian score cannot reach.

**Scope:** BNT power spectra ONLY. The non-BNT (cut-all) results are well-understood and as-expected
(score and Fisher agree); **leave non-BNT alone** except as the lossless-identity control (§5). Work at
**14000 deg² first** (we have all the caches/benchmarks there); generalize to other areas later.

## 2. The benchmark to match (the score/MOPED result at 14000)

The VMIM posteriors must be compared against these — they are calibrated (TARP/SBC), on-truth, and sit
on the Fisher floor. Source: `docs/PLAN_score_bnt_tension_14000.md`.

- **Lossless identity (the hard oracle):** at the FULL vector (ℓmax=1024 on every bin), BNT ≡ non-BNT.
  The score reproduces this to machine precision at the summary level. A correct VMIM must give
  `bnt_full` null ≈ `nonbnt_full` null (same truth, same width). **The previous attempt FAILED exactly
  here** (§3).
- **Fisher floor σ (the σ anchor)** for BNT bin-1 @ ℓmax460 (bins 2-4 full): σ(Ωm,S8,w0) =
  [0.0154, 0.0282, 0.1058], FoM3 = 2.04e5. The score realized σ = [0.0161, 0.0273, 0.0842] — *on* the
  floor. A calibrated VMIM should land near this; a VMIM that is much tighter is over-confident (gate
  it), much wider is under-extracted.
- **Null on truth:** (Ωm, S8, w0) = (0.26, 0.84, −1.0). The score nulls sit on truth; the raw NPE
  nulls sit off (S8≈0.86-0.88) — being on-truth is a primary reliability signal.

The score is the **safety net**: it already gives a defensible BNT result. VMIM is an enhancement /
independent cross-check, NOT a dependency. Don't let a hard VMIM chase block the paper.

## 3. What the previous VMIM attempt did, and exactly why it failed

Code: `scripts/vmim_compress.py` (Stage 1), `scripts/npe_on_summary.py` (Stage 2),
`scripts/run_vmim_overnight.py` + `scripts/run_vmim_ensemble.py` (orchestrators). Results:
`outputs/baryon_tension/vmim/MORNING_SUMMARY.md` (PARTIAL), `.../ENSEMBLE_RESULT.md` (**oracle FAIL**).
The plan that built it: `docs/PLAN_bnt_neural_compression.md` — note its "Adaptations" section lists,
in its own words, several of the very deviations that broke it.

The symptoms (from the result docs): TARP passed but the nulls were **off-truth and σ-collapsed**, the
result moved with a tuning knob, and the **`bnt_full` null disagreed with `nonbnt_full`** (0.8385 vs
0.8152 — violates the lossless identity → the ENSEMBLE_RESULT oracle verdict is literally `FAIL`). The
BNT/non-BNT payoff ratio swung 0.76 → 1.46 across runs — not reproducible. Five concrete causes, each
a deviation from the recipe:

1. **Stage 2 used `jaxili` NPE, not the validated `sbi_lens` ConditionalRealNVP** (`npe_on_summary.py:40`
   `from jaxili.inference import NPE`). The recipe's Stage 2 is the sbi_lens RealNVP (`build_flow` /
   `train_flow`), and **recipe lesson 1 is "the NDE choice is a large lever"** (RealNVP gave ~30% higher
   FoM than a MAF on the *identical* summary). `PLAN_bnt_neural_compression.md` explicitly deferred this
   ("Stage 2 = jaxili … sweep later") — that shortcut is the user's caveat #2 and the most likely
   primary cause. **FIX: port the sbi_lens RealNVP for Stage 2.**

2. **Random train/val split → realization leakage.** `vmim_compress.py:84-87` splits by a random
   `val_frac` permutation. The grid is **2424 unique cosmologies × 7 noise realizations = 16965 rows**
   (verified). A random split scatters the 7 realizations of each cosmology across train AND val, so the
   val loss measures "predict θ from a fresh-noise realization of a *training* cosmology" — over-
   optimistic → the best-val checkpoint is over-fit → over-confident summaries. **FIX: split by UNIQUE
   COSMOLOGY** (all 7 realizations of a cosmology go to the same side; use `np.unique(round(theta))` to
   group, hold out whole cosmologies). This is the user's caveat #1.

3. **An ad-hoc `--summary-noise` info-bottleneck used as a tuning knob.** `vmim_compress.py:42,134`
   adds Gaussian noise to `c(x)` during training. It is NOT in the recipe, and the sweep showed the null
   S8 *slides* with it (0.76 → 0.80 → 0.855 as noise 0.3 → 0.6 → 1.0) — i.e. it is a free parameter being
   tuned to hit truth, which is fitting the answer, not calibrating. **FIX: remove it.** The recipe's
   principled over-confidence remedy is the **compressor deep-ensemble** (recipe lesson 7): train K=2-3
   compressors with different seeds and pool their posteriors per observation — apply ONLY if the gated
   single-compressor posterior is mildly over-confident, and diagnose the direction first.

4. **Cholesky whitening preprocessing on the ill-conditioned BNT covariance.** `vmim_compress.py:49-59`
   replaces the recipe's `log1p-zscore` with full-rank `L⁻¹(x−µ)` (because the BNT cross-spectra are
   *signed*, so log1p is invalid — a real constraint). But full decorrelation of an ill-conditioned
   covariance **amplifies noise in the near-degenerate (nulled) directions** — exactly the directions
   that make BNT hard — so it can feed the MLP a noise-dominated input. **FIX: A/B test the
   preprocessing.** A signed-safe, robust default is **per-feature standardization** (subtract mean,
   divide by per-feature std — NO cross-feature decorrelation), then clip at ±a few σ. Optionally
   compare: (a) per-feature z-score, (b) a *regularized*/PCA-truncated whitening that drops the noisy
   nulled directions, (c) the current full Cholesky whitening. Pick by *calibrated* performance on the
   lossless-identity oracle, not by FoM.

5. **The gate was insufficient.** It checked TARP (and SBC), but TARP passed while the result was wrong.
   TARP/SBC test *average* coverage over the prior — they can pass while the posterior is locally biased
   at the fiducial. **FIX: gate on ALL of** (a) the **lossless identity** `bnt_full` null ≈ `nonbnt_full`
   null (hard — this is what failed), (b) **null on truth** (within ~0.5σ), (c) **σ near the Fisher
   floor** (not collapsed, not inflated), (d) **TARP** — and actually *look at the TARP-DRP plot*
   (ECP vs α curve), not just the max-deviation scalar (user caveat #3), and (e) **SBC rank histograms**
   ≈ uniform (rank-std ≈ 0.289). A summary that fails (a) or (b) is rejected regardless of TARP.

## 4. The correct re-implementation (the two-stage recipe, fixed)

Follow `NEURAL_SUMMARIZATION_RECIPE.md` faithfully; the only project-specific change is the input cache
and the signed-safe preprocessing. Keep Stage 1 and Stage 2 **separate** (train compressor → freeze →
train NDE on frozen summaries), so the NDE family is a swappable lever (recipe §0, lesson 1).

**Stage 1 — VMIM compressor** (mostly reuse `scripts/vmim_compress.py`, with the fixes):
- MLP `c(x)`: hidden (256,256), leaky_relu, linear → `summary_dim` (try d=8, i.e. 6 params + headroom;
  recipe lesson 4). Companion `q(θ|y)` = sbi_lens ConditionalRealNVP (n_layers 4, coupling [128,128]
  silu), discarded after Stage 1. Loss `L = −E[log q(θ|c(x))]`. Adam(W), piecewise-const LR ×0.7,
  batch 512, ~30k steps (time-boxed), keep **best-validation**.
- **Split by cosmology** (fix #2). **No summary noise** (fix #3). **Preproc = signed-safe per-feature
  standardization by default**, A/B vs regularized whitening (fix #4). Keep the θ-z-scoring for the
  companion (raw cosmo params span H0~67 vs Ωm~0.26 → RealNVP log_prob diverges; the existing code
  already does this at `vmim_compress.py:96-101` — keep it).
- Output: compressed `{theta, y}` train/val + compressed fiducial `y_fid`, raw θ kept.

**Stage 2 — NDE on frozen summaries (THE key fix):** port the **sbi_lens RealNVP** `build_flow` /
`train_flow` from the cnn_sbi reference (`scripts/sbi/train_nde_from_compressed.py` +
`scripts/sbi/npe_cnn_nbody_tomo.py`). Capacity (n_layers 4, hidden 128), ~50k steps, Adam cosine LR
1e-3→1e-5, grad clip 1.0, weight decay 1e-4, early stopping; **pool 3 estimator seeds** (per-obs sample
pooling). Sample the posterior at `y_fid`. (Keep the jaxili path available as a comparison arm, but the
*production* Stage-2 is sbi_lens RealNVP — recipe §2, lesson 1.)

**Exact Stage-2 port (verified from the cnn_sbi source):**
- **Flow:** `build_flow(n_cosmo_params, n_layers, hidden)` →
  `cnn_sbi/scripts/sbi/npe_cnn_nbody_tomo.py:3647-3677`. It is `ConditionalRealNVP(n_layers=4,
  bijector_fn=partial(AffineCoupling, layers=[hidden,hidden], activation=jax.nn.silu))` with
  `hidden=128`; returns `(nf_logp, nf_sample)` as `hk.without_apply_rng(hk.transform(...))`.
  `nf_logp(theta, y) = NF(n_cosmo)(y).log_prob(theta).squeeze()`; `nf_sample(y, n) = NF()(y).sample(n)`.
- **Train:** `train_flow(rng, nf_logp, dataset_train, dataset_val, n_cosmo, summary_dim, total_steps=50000,
  batch_size=128, save_every=2000, save_dir, lr_init=1e-3, end_lr=1e-5, grad_clip=1.0, weight_decay=1e-4,
  patience=20)` → `npe_cnn_nbody_tomo.py:3746-3903`. Optimizer = `optax.chain(clip_by_global_norm(1.0),
  adamw(cosine_decay_schedule(1e-3, total_steps, alpha=end_lr/lr_init), weight_decay=1e-4))`; keeps the
  best-val params with early stopping (`patience=20`); 90/10 internal split (deterministic `seed 0`).
- **Wrapper to copy:** `train_nde_from_compressed.py:158-193` (`train_sbilens_realnvp`) shows the whole
  build→train→sample call. **Sampling:** broadcast `y_fid` to `(M, summary_dim)`, then
  `nf_sample.apply(best_params, key, y_cond, M)`; filter non-finite rows.
- **Seed pooling:** per-obs `key = PRNGKey(seed*100003 + obs_idx)`, concatenate the 3 seeds' samples,
  drop non-finite (`train_nde_from_compressed.py:291-310`).
- **Metrics:** reuse `compute_fom3` (= `exp(-0.5·logdet(cov3))` on Ωm,S8,w0), `fom2d`, `marginal_stats`
  from `cnn_sbi/scripts/sbi/train_jaxili_from_compressed.py:56-92`.
- **Imports:** `from npe_cnn_nbody_tomo import build_flow, train_flow`; you'll either copy those two
  functions into this repo (they're ~250 lines, self-contained on `sbi_lens` + `optax` + `haiku`) or add
  `cnn_sbi/scripts/sbi` to `sys.path`. Materialize `tensorflow_probability.substrates.jax` BEFORE
  `sbi_lens` (lazy-loader bug).

**CRITICAL ADAPTATION — z-score θ for the flow.** The cnn_sbi reference uses `h_0 ≈ 0.6736` (units of
H0/100), so its 6-param θ is all O(1) and the RealNVP `log_prob` is stable. **Our θ has `H0 = 67.36`,
`ns = 0.96`, `Ob = 0.049`** — wildly different scales → the RealNVP `log_prob` diverges (the existing
Stage-1 code hits exactly this and z-scores θ for the companion at `vmim_compress.py:96-101`). So the
**Stage-2 RealNVP must train on z-scored θ** (fit (µ,σ) on train θ, transform, and un-transform the
posterior samples back to physical units before computing tension/FoM). Do not skip this — it is a
likely silent-divergence trap.

**Gate thresholds (from `gate_verdict.py`):** TARP max-deviation `≤0.05` = PASS, `>0.10` = FAIL;
SBC rank-std target band `[0.275, 0.305]`; TARP net bias ≈ 0. Use `tarp.get_tarp_coverage(samples_tarp,
theta, references="random", num_bootstrap=200, norm=True)` on a **varied-θ validation set** (not just
the fiducial), stratified by FoM3 terciles (`tarp_stratified_val_nde.py`). Inspect the ECP-vs-α plot.

**Per scale cut → per compressor.** The data vector changes with the cut (and the BNT x-cut = min
rule), so **each scale cut needs its own compressor + NDE** (the user flagged this). Reuse the
cut-aware slicing already built for the score pipeline so you don't re-dump per cut:
`scripts/score_cut_utils.py` (`keep_indices(cuts)` slices the full dump to any cut;
`build_score(...)` shows the J/C reference). Dump the FULL vector once
(`run_npe_inference_auto_cross_ps_master.py --dump-cache`), slice to each cut, compress, infer.

## 5. Validation oracles & gates (bake in — an unattended failure must be loud)

Run these in order; do NOT report a width that fails any of (1)-(3):
1. **Lossless identity (hard):** `bnt_full` null posterior ≈ `nonbnt_full` null posterior (same truth,
   width within scatter). This is THE check the old attempt failed. If VMIM can't reproduce it, the
   compression is wrong — stop and debug (likely the split, the preproc, or the Stage-2 family).
2. **Null on truth:** null mean within ~0.5σ of (0.26, 0.84, −1.0), for both BNT and non-BNT.
3. **σ vs Fisher floor:** realized σ near the floor (§2). Collapsed σ + TARP-OK = over-confident →
   apply the compressor ensemble (fix #3), don't ship.
4. **TARP-DRP plot + SBC histograms:** inspect the ECP-vs-α curve and the rank histograms, not just the
   scalar; SBC rank-std ≈ 0.289, TARP net bias ≈ 0.
5. **Analytic-Gaussian oracle (recipe §5, optional but strong):** we HAVE an analytic Gaussian
   covariance (`scripts/diagnostics/cache_gaussian_cov/gaussian_cov_native_14000.npy`) and the score's
   Fisher contours. In the Gaussian regime the VMIM+NDE posterior should match the score/Fisher — use it
   to validate the whole pipeline before claiming any non-Gaussian gain.

## 6. File map / environment

- **Recipe (read):** `/home/tersenov/software/cnn_sbi/NEURAL_SUMMARIZATION_RECIPE.md`. Reference code:
  `cnn_sbi/scripts/sbi/{vmim_from_cache.py, train_nde_from_compressed.py, npe_cnn_nbody_tomo.py
  (build_flow/train_flow), tarp_stratified_val_nde.py}`.
- **This repo — to fix/replace:** `scripts/vmim_compress.py` (Stage 1 — fix split/noise/preproc),
  `scripts/npe_on_summary.py` (Stage 2 — REPLACE jaxili with sbi_lens RealNVP, or add a new
  `scripts/nde_realnvp_from_summary.py`), orchestrators `scripts/run_vmim_{overnight,ensemble}.py`
  (rebuild the gate per §5).
- **Cut-aware dump/slice (reuse):** `scripts/score_cut_utils.py`, worker
  `scripts/run_npe_inference_auto_cross_ps_master.py --dump-cache`. Existing fixed-cut caches:
  `outputs/score_experiment/cache/{bnt_full,nonbnt_full}_14000_{nobary,bary}/cache.npz` (full vectors,
  120 feat at rebin20 — slice these) and `outputs/baryon_tension/vmim/cache/{bnt_full,nonbnt_full,
  bnt_580,nonbnt_460}/cache.npz` (the old fixed-cut caches).
- **Env:** everything is in the **jaxili** conda env (`/home/tersenov/anaconda3/envs/jaxili/bin/python`)
  — verified present: `sbi_lens`, `haiku` 0.0.16, `optax` 0.2.4, `distrax` 0.1.5,
  `tensorflow_probability` 0.24.0, `tarp`, `jaxili`. (`import tensorflow_probability.substrates.jax`
  BEFORE `sbi_lens` — there's a lazy-loader ordering bug; see `vmim_compress.py:71`.) **GPU 2** on titan
  (no scheduler; check `nvidia-smi` first, stagger multi-process CUDA inits — simultaneous JAX starts
  race with "no supported devices").
- **Tension/getdist** (if computing tension/contours) → **aname** env. Score benchmark numbers + the
  score pipeline: `docs/PLAN_score_bnt_tension_14000.md`.

## 7. Recommended first steps (pilot before scaling — this is the failure-avoidance order)

1. **Smoke the Stage-2 port:** build the sbi_lens RealNVP, train on the EXISTING score summaries (or on
   a by-cosmology-split raw cache for `bnt_full`), sample at `y_fid`, confirm a finite, constrained,
   roughly-on-truth null. No new physics — just prove the RealNVP Stage-2 runs and is sane.
2. **Fix Stage 1 split + drop noise + per-feature-standardize preproc;** retrain `bnt_full` and
   `nonbnt_full` compressors (single seed).
3. **Run the lossless-identity gate (§5.1):** does `bnt_full` ≈ `nonbnt_full`, both on truth, σ near
   Fisher floor, TARP/SBC OK? This is the make-or-break. If it passes, the pipeline is correct and you
   can move to the de-biasing cuts (bnt @460 etc.) and the comparison to the score. If it fails, debug
   in the order {Stage-2 family, split, preproc} — do not tune `summary_noise` to paper over it.
4. Only after the identity passes: per-cut compressors, the de-biasing-cut contours, and the
   head-to-head vs the score (does learned VMIM match or beat the Gaussian score, calibrated?).

## 8. Open questions / risks

- **Preproc for signed ill-conditioned data is the genuine unknown** — per-feature z-score is the safe
  default but may leave the flow to find the projection in a correlated space; regularized whitening
  drops noisy nulled directions but discards some signal. A/B by the calibrated identity oracle.
- **Over-confidence is the headline danger** (recipe lessons 5-7) — single-compressor amortization tends
  to over-confidence; the ensemble is the fix, but only if the direction is over-confident (SBC std high,
  marginals narrow). Diagnose first; for a *conservative* posterior the ensemble is the wrong tool.
- **summary_dim:** start d=8; the PS is near-Gaussian so d≈n_params may suffice (recipe lesson 4).
- The score result already stands — VMIM that can't be calibrated is a negative result worth recording
  (it would say "for this near-Gaussian PS the analytic score is the right tool"), not a blocker.
