# BNT data vectors and neural compression — methods notes (paper-relevant)

Status: distilled from the score/MOPED experiment (`outputs/score_experiment/`, plan
`~/.claude/plans/jolly-toasting-robin.md`, memory `bnt-npe-score-chase`) and the 2026-06-25
follow-up confirming the non-BNT control. Intended as raw material for the methods section.

## 1. Why the BNT data vector is unsuitable for direct neural posterior estimation

The summary statistics are the tomographic auto + cross angular power spectra. For the standard
(non-BNT) analysis this vector is high-dimensional (≈96 features at rebin=20; ≈240 masked / ≈980
full-sky at rebin=10) but **well-conditioned**, and a normalizing-flow NPE extracts the parameter
information from it without trouble.

The BNT transform is a fixed lower-triangular nulling operator `M` on the tomographic bins; on the
power-spectrum vector it acts as `C̃ = M C Mᵀ` per multipole. The nulling is, by design, a set of
near-cancellations, and this makes the **transformed vector ill-conditioned** in exactly the way a
density estimator handles worst:

- **Strong anti-correlations and near-degenerate ("nulled") directions.** The BNT cross-spectra are
  differences of correlated quantities; several linear combinations carry almost no variance. The
  covariance becomes ill-conditioned (the raw score `JᵀC⁻¹x` has condition number ~1e8).
- **Wide dynamic range and sign changes.** BNT cross-spectra are small and can be negative, so
  positivity/log preprocessing is invalid; their scale differs greatly from the autos.
- **Information in the low-S/N, high-ℓ modes.** The BNT FoM gain lives in many individually
  low-signal, correlated high-ℓ modes — precisely the directions a flow trained on a finite
  simulation grid struggles to weight correctly.

A flow asked to learn the optimal `JᵀC⁻¹` projection directly from this high-dimensional,
ill-conditioned input, with finite simulations and finite capacity, **under-learns the projection**:
it fails to down-weight the noise directions and to exploit the nulled structure, so the posterior
comes out **too wide (under-extracted)**. Critically this failure is *silent* — the posterior remains
*calibrated* (honestly uncertain), it is simply less informative than the data allow. The well-
conditioned non-BNT vector does not suffer this (see §3), which is why the problem is specific to BNT.

**The fix is to not make the flow discover the projection.** Hand it the projection analytically and
let it learn only a low-dimensional density.

## 2. Terminology: whitening vs MOPED / score compression (they are NOT the same)

- **MOPED ≡ score compression.** Same construction: the score `t = JᵀC⁻¹(x−μ)`, used in its
  parameter-MLE form `θ̂ = θ_fid + F⁻¹JᵀC⁻¹(x−μ)` (6 numbers, in parameter units). For a (near-)
  Gaussian likelihood these are *sufficient* statistics — the compression is **lossless** (verified
  `Fisher(t)=Fisher(x)` to ~1e-10). The `F⁻¹` (parameter-units) rotation is essential: the raw score
  `JᵀC⁻¹x` is itself ill-conditioned (cond ~1e8) and the NPE fails on it; the rotation conditions it.
- **Whitening is different and is NOT a compression.** It is a full-rank Cholesky decorrelation
  `L⁻¹(x−μ)` that keeps *all* ≈96 dimensions — it rotates/decorrelates but does not reduce dimension,
  so the flow must still find the projection in the full space.

**Result (14000 deg², rebin=20, 3-param FoM):**

| method | BNT FoM₃ | BNT/non-BNT FoM ratio | verdict |
|---|---|---|---|
| whitening (non-compressed) | 105k | **0.96** | under-extracts BNT — *no* advantage realized |
| score / MOPED (compressed) | 165k | **1.46** | advantage realized, calibrated |

So whitening and score behave **oppositely on the BNT vector**: whitening leaves BNT looking no
better than non-BNT (ratio ≈ 1); only the score/MOPED compression reveals the advantage. Whitening
still yields a *valid, calibrated* BNT posterior — it is simply under-extracted, so the benefit is
invisible. (The earlier VMIM *learned* neural compression failed differently — over-confident, off-
truth — and is unresolved; see `bnt-npe-score-chase`.)

## 3. The non-BNT control (compression acts only on the ill-conditioned vector)

Applying the *same* compression to the well-conditioned non-BNT vector reproduces the un-compressed
(whitening) posterior — confirmed across all six footprints at the fixed cut: FoM ratio
score/whiten = 1.00–1.08, σ ratios 0.98–1.04, both on-truth, contours coincide. So compression is a
**no-op on non-BNT** and does not generically tighten contours; it only rescues the BNT vector. This
is what licenses attributing the BNT FoM gain to real information rather than to the method.

## 4. What the score result establishes — and the honest caveats ("by construction")

**Claim that is supported:** under the correct treatment of the data vector (score/MOPED
compression), BNT outperforms non-BNT — ~1.5× tighter 3-param FoM at 14000 — with calibrated,
on-truth posteriors, and a control showing the compression is not the source of the gain. The
contours are genuinely constraining and lossless.

**Caveats to state, not hide:**
1. **Constructive, not discovery.** The score uses the Fisher's J and C, so "score reaches the Fisher
   FoM" is partly definitional. What is *earned* (not automatic): TARP/SBC calibration, the full
   posterior, and the non-BNT control. Frame it as *realizing* the Fisher-predicted advantage with a
   calibrated SBI pipeline (which whitening/raw NPE cannot), not as discovering a new advantage.
2. **Gaussian-sufficiency assumption.** Score is optimal only up to the Gaussian likelihood; genuine
   non-Gaussian gain would require an over-complete compression (untested). "Score ≈ Fisher" also
   says "no large non-Gaussian gain beyond the Gaussian bound."
3. **Realized 1.5× vs Fisher 2.5×.** The gap is a non-BNT-side effect (non-BNT NPE comes out slightly
   tighter than its Fisher), not BNT under-extraction (BNT NPE ≈ its Fisher).
4. **Scope.** This is a single fixed cut (BNT-580 / non-BNT-460), rebin=20 — a constraining-power
   methods result, **not** the production scale-cut tension analysis, and distinct from the baryon-
   *bias*-mitigation question (BNT-580 under baryonification is a separate, secondary robustness check).
