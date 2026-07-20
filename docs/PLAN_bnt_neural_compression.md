# Plan: VMIM-MLP neural compression for the BNT data vectors (overnight build)

**Status:** building 2026-06-22 (overnight, unattended). Goal: close the gap the whitening fix left —
NPE recovered only ~half the Fisher BNT advantage (BNT-580/non-BNT-460 area 0.79; Fisher 0.37). The
remaining gain needs a better summary of the 240-feature vector. Based on AT's validated recipe
`/mnt/home/tersenov/software/cnn_sbi/NEURAL_SUMMARIZATION_RECIPE.md`.

## Recipe adopted (AT's, verbatim where it transfers)
Two-stage, trained **separately**:
1. **Stage 1 — VMIM compressor.** MLP `c(x)` (hidden 256,256, leaky_relu → summary_dim) trained to
   maximize `I(θ; c(x))` via its variational bound: minimize `L = −E[log q(θ | c(x))]` with a
   ConditionalRealNVP **companion** `q` (sbi_lens, n_layers 4, coupling [128,128] silu). Adam,
   piecewise-const LR ×0.7, batch 512, ~30k steps, single seed (41), keep **best-validation**,
   time-boxed. Companion discarded after Stage 1.
2. **Stage 2 — NDE on the *frozen* summaries.** Conditional flow `p(θ|y)`; pool **3 seeds**. Here =
   **jaxili NPE** on the ~8-D summary (the estimator we already trust), default flow for now.
3. **Gate (mandatory): TARP (+ SBC).** Tightness ≠ correctness — the recipe repeatedly saw summaries
   win FoM by becoming *over-confident* and failing coverage. Gate before trusting any width.

## Adaptations for this project
- **Preproc = whitening, NOT log1p-zscore.** Our cross/BNT C_ℓ are **signed** (log1p invalid), and we
  already proved full-rank Cholesky whitening fixes the BNT representation (null→truth, σ↓). So Stage-1
  preprocessing = mean-subtract + Cholesky-decorrelate (ridge), fitted on train, applied to val/obs.
  This conditions the MLP input exactly as the recipe's z-score does for positive stats, but signed-safe.
- **summary_dim ≈ 8** (6 params + a little headroom; recipe lesson 4).
- **Stage 2 = jaxili NPE** (reuse our trusted path) rather than sbi_lens; estimator-family sweep
  (RealNVP vs MAF, recipe lesson 1) deferred — note, don't block.
- Data is near-Gaussian PS, so the raw-high-D failure is milder than ℓ1 (recipe §5) — but VMIM to ≈8-D
  is still the clean reduction that should let the flow use the high-ℓ BNT modes it currently wastes.

## Configs (14000 deg², monopole-subtracted, NULL posteriors)
| tag | basis | cut | role | purpose |
|---|---|---|---|---|
| nonbnt_full | non-BNT | all ℓ≤1024 | null | oracle ref |
| bnt_full | BNT | all ℓ≤1024 | null | **oracle**: must ≈ nonbnt_full, null on truth |
| nonbnt_460 | non-BNT | all ℓ≤460 | null | payoff baseline (required cut) |
| bnt_580 | BNT | bin1 ℓ≤580, 2-4 full | null | **payoff**: area vs nonbnt_460 → toward Fisher 0.37? |

## Pipeline (3 components + orchestrator)
1. **Worker `--dump-cache <dir>`** (`run_npe_inference_auto_cross_ps_master.py`): build `(params,
   X_train, x_fid)` for a config (all the cut/rebin/grid logic we already have), save npz, exit before
   training. Keeps cache construction identical to the production data vector.
2. **`scripts/vmim_compress.py`** (ported from `cnn_sbi/.../vmim_from_cache.py`): load cache → whiten
   (fit on train) → train MLP+RealNVP companion (best-val, time-box) → write compressed
   `{train,val}` cache + compressed fiducial + history/meta.
3. **`scripts/npe_on_summary.py`**: load compressed cache → jaxili NPE on `(θ, y)` (N seeds) → sample
   at compressed fid → save pooled posterior + run TARP coverage.
4. **`scripts/run_vmim_overnight.py`**: orchestrate {dump → compress → NPE} for the 4 configs, then
   compute the oracle (bnt_full ≈ nonbnt_full, null S8→0.84), the payoff (bnt_580/nonbnt_460 σ & area
   vs Fisher 0.37 and whitening 0.79), TARP verdicts, a contour overlay, and write
   `outputs/baryon_tension/vmim/MORNING_SUMMARY.md`.

## Verification / back-pressure (so an unattended failure is loud, not silent)
- **Smoke gate FIRST:** run the full dump→compress→NPE chain on `bnt_full` with SHORT training
  (~2k Stage-1 steps, few Stage-2 epochs). Require a finite, constrained, roughly-on-truth null before
  launching the full overnight. If smoke fails, STOP and write the error to the summary.
- **Oracle:** summary-NPE `bnt_full` ≈ `nonbnt_full` (σ within scatter), null S8 ≈ 0.84.
- **TARP** coverage per config; flag over/under-confidence. SBC if time (rank-std vs uniform).
- Time-boxes on Stage 1 (`--max-minutes`) and a global wall cap; keep best-so-far; never present an
  ungated width as a result.

## Honest risks
- Signed-data preproc (whitening, untested *inside* a VMIM loop) — smoke catches gross failure.
- Companion / compressor non-convergence — best-val + NaN guard + time-box.
- **Over-confidence** (recipe lessons 5–7): the headline danger — a tighter BNT contour that fails
  TARP is NOT a win. The compressor deep-ensemble fix (recipe lesson 7) is the registered remedy if
  TARP shows mild over-confidence; noted, applied only if needed.
- Stage-2 family is a real lever (lesson 1); using jaxili default tonight, sweep later.
- Reaching Fisher 0.37 is not guaranteed; any move below whitening's 0.79 *with passing TARP* is a win.

## Morning deliverable
`outputs/baryon_tension/vmim/MORNING_SUMMARY.md`: oracle verdict, TARP verdicts, the
bnt_580/nonbnt_460 area ratio vs {whitening 0.79, Fisher 0.37}, a contour overlay PNG, and a clear
PASS/PARTIAL/FAIL with next step. All artifacts under `outputs/baryon_tension/vmim/`.
