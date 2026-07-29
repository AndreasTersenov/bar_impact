# Plan: adapt the NPE to BNT data vectors — easy steps first

**Status:** in progress (easy steps approved 2026-06-22). Target: 14000 deg², monopole-subtracted.
Scope deliberately small: try the cheap, principled normalization fix; **neural compression is the
intended next tier but is NOT to be implemented without an explicit go-ahead from AT.**

## Problem (established this session)

BNT carries a real **~3× constraining-power gain** at the required scale cut — non-BNT cuts all bins
to ℓ≤460, BNT only cuts bin-1 to ℓ≤580 and keeps bins 2-4 to ℓ≤1024 — verified by Fisher
(`scripts/diagnostics/fisher_bnt_vs_nonbnt.py`; oracle BNT-full = non-BNT-full to 1e-13; BNT-580 vs
non-BNT-460 = σ(Ωm) 0.58×, σ(S8) 0.63×, area 0.37×). **The NPE does not realize it** — BNT-580
contours ≈ non-BNT-460 (slightly wider), and the BNT null is off-truth (S8≈0.86 vs 0.84).

### Root cause (found in the code)
jaxili's NPE applies **per-feature z-scoring by default** (`compressor.Standardizer`, `(x−μ)/σ`). For
a BNT vector that is the wrong normalization: (1) it divides the nulled crosses by their tiny
noise-std, promoting them to unit-variance *noise* channels the flow must model; (2) it leaves BNT's
strong anti-correlations untouched. Fisher inverts the full covariance (decorrelates + down-weights
those directions), so it sees the gain; the z-scored flow can't.

## Design principle (encodes AT's standing concern)
**No variance-ordered truncation, ever** — PCA orders by variance, not information, so it drops
low-variance directions that can carry cosmological signal and inflates posteriors (AT's experience).
The easy step below keeps **every** dimension (invertible ⇒ removes no information; only decorrelates).

## The easy step (NOW): full-rank Cholesky whitening
`x → L⁻¹(x − μ)`, `L = chol(C_train + εI)`, ε a tiny ridge. Decorrelates and isotropizes the data
vector while keeping all dimensions (invertible, zero info loss). Targets exactly what z-scoring
leaves broken — the correlations — and, by orthogonalizing, lets the flow cleanly ignore the nulled
noise directions instead of fighting them mixed into the signal.
- Implement: a `--compress {none,whiten}` flag in `run_npe_inference_auto_cross_ps_master.py`.
  Fit `μ, L` on the **training** data vector (after cut/rebin); apply the identical transform to the
  training vectors **and** the observation; pass `z_score_x=False` to `NPE()` (no double-standardize).
- `none` = current behavior (regression-safe default).

### Cheap fallbacks if whitening alone doesn't fully close it (still "easy")
- Floored/regularized z-score (cap the std so nulled directions aren't amplified) — isolates whether
  the residual is the *noise channels* vs the *correlations*.
- Basic training hygiene: check loss convergence, epochs/lr/batch — rule out under-training.

## Verification / back-pressure
1. **Oracle (NPE analog of the Fisher oracle):** whitened **BNT-full** NPE ≈ **non-BNT-full** NPE
   (σ within run scatter) and the null returns to truth (S8 → 0.84). Both are invertible linear views
   of the same data, so a well-conditioned flow shouldn't care about the basis.
2. **Payoff:** only if the oracle passes — BNT-580 NPE σ(S8)/σ(Ωm) drop toward the Fisher ratios
   (0.63 / 0.58) vs non-BNT-460.
3. **Regression:** non-BNT NPE unchanged with `--compress whiten` vs the current z-score result
   (null still on truth, σ comparable).

## Implementation steps
1. Add `--compress {none,whiten}` to the worker: fit on training, apply to train+obs, persist `μ,L`
   with the run, `z_score_x=False` when active. Math in one small helper.
2. Smoke + **oracle** on 14000 full range: whitened BNT-full vs non-BNT-full, check null S8; plus the
   non-BNT regression. (A handful of NPE jobs on the existing GPU-pack harness.)
3. If the oracle passes: re-run BNT bin1-580 (+ non-BNT-460) and compare contours to the Fisher
   prediction.
4. Decide next tier with AT.

## Next tier — DO NOT implement without talking to AT first
**Neural compression** (an MLP compressor trained with a VMIM — variational mutual-information-
maximization — objective) to learn an optimal low-dim summary of the data vector. This is the intended
principled compression and can capture non-Gaussian information; it replaces, not supplements, the
classical linear routes. **Removed from scope entirely: PCA. Not pursued: MOPED.** Bring results from
the easy step back to AT before starting neural compression.
