# PLAN — VMIM compressed PS: baryon-tension vs ℓmax scale cut (14000 deg²)

Goal: reproduce the uncompressed `nsigma_vs_upper_cut` curve (which crosses 0.3σ at ℓmax≈480 for
non-BNT cut-all at 14000) but with the **VMIM-compressed** PS, for both:
- **non-BNT, cut ALL bins**: cuts = [c, c, c, c]
- **BNT, cut bin-1 only**: cuts = [c, 1024, 1024, 1024]
and check whether the de-biasing ℓmax is the same once properly extracted. Compressors are **retrained
per cut** (the data vector changes).

## Pipeline (reuse the validated P2f recipe + the score cut machinery)
Per (cut c, config):
1. **Slice** the full rebin=20 score cache to the cut: `keep = score_cut_utils.keep_indices(cuts)`;
   `X_c = X_full[:, keep]`, `x_fid_null_c`, `x_fid_biased_c` (biased = the `_bary` cache's x_fid).
2. **Compressor** (`vmim_compress.py --cuts ... --analytic-cov {bnt,nonbnt}`): ana_whiten with the
   **cut covariance** `build_score(cuts, bnt)['C']` + per-feature-relative clip (the P2e/f fix), H0/100,
   by-cosmology split. Output compressed `{theta, y_tr, y_va, y_fid (null), y_fid_biased}`.
3. **NDE** (`nde_realnvp_from_summary.py`): train K seeds, sample null at y_fid and biased at
   y_fid_biased, save **per-seed** null+biased (for tension scatter).
4. **Tension** (`aname` env): per seed, Gaussian 3-param `Q_DM` (Ωm,S8,w0) via
   `tension/estimators.q_dm_tension` → nσ; mean±std over seeds = the error bar (mirrors the paper's
   5-NPE-run bars).

## Grid / compute
- **Cuts**: ℓmax = 340…1020 step 40 (18 cuts), matching the reference x-axis.
- **Configs**: 2 (nonbnt cut-all, bnt bin-1) → 36 (cut,config) jobs.
- **Seeds**: K=3 NDE seeds per job (tension scatter); single compressor per job (cheap vs the full
  ensemble — the ensemble was for the full-vector calibration polish, not needed for the tension point).
- **GPUs 0,1,2,3**, packed (mem-fraction ~0.2 → ~4/GPU) via a Popen slot pool ⇒ ~16 concurrent.
  Estimate: 36 compressors (~2.5 min) + 36 drivers (3 seeds, ~5 min) ≈ 30–45 min wall.

## Gates / back-pressure (smoke before the full launch)
1. **Slice oracle**: `keep_indices([580,1024,1024,1024])` BNT slice reproduces the validated `bnt_580`
   columns (already PASS in the score work) — sanity that the cut machinery is wired right.
2. **Smoke one cut** (ℓmax≈480) both configs: compressor trains, null on-truth, biased shifted, Q_DM
   finite and sensible (non-BNT @480 should be near the 0.3σ crossing if the curves track the
   uncompressed; if compressed crosses much earlier, that's the result).
3. Null per cut ~on-truth; biased carries the shift; tension monotonic in ℓmax.

## Deliverable
`plots/nsigma_vs_upper_cut_compressed_14000.png` — compressed non-BNT (cut-all) + BNT (bin-1) nσ vs
ℓmax, with the 0.3σ line, overlaid (optionally) on the uncompressed reference for the 14000 panel.
Outputs under `outputs/baryon_tension/vmim_v2/scalecuts/`.
