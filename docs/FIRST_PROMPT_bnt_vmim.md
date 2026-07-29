# First prompt for the new VMIM session (paste into a fresh Claude Code session)

---

We're re-implementing **VMIM neural compression for the BNT power-spectrum data vectors**, correctly
this time. A previous attempt in this repo produced super-uncalibrated, clearly-wrong posteriors; I've
already diagnosed why and written a detailed handoff.

**Read these first, in order, before doing anything:**
1. `docs/HANDOFF_bnt_vmim_reimplement.md` — the active handoff: the goal, the exact failure diagnosis
   (5 confirmed causes with file/line pointers), the correct two-stage recipe, the oracles/gates, the
   file map, and the recommended pilot-first steps.
2. `/home/tersenov/software/cnn_sbi/NEURAL_SUMMARIZATION_RECIPE.md` — the validated recipe this is based
   on (worked very well on wavelet ℓ1 in my other project).
3. `docs/NOTES_bnt_compression_for_paper.md` — why the BNT data vector is ill-conditioned/signed/noisy.
4. Skim `docs/PLAN_score_bnt_tension_14000.md` — the score/MOPED result that is the **benchmark to
   match** and the **safety net** (we already have a calibrated, on-truth, Fisher-floor BNT extraction
   "by construction"; VMIM is the learned, assumption-light alternative/cross-check).

**The goal in one sentence:** a properly-trained VMIM-MLP compressor + an expressive NDE that yields
*calibrated, on-truth* BNT posteriors, reproducing the lossless identity (full-BNT ≡ full-non-BNT) and
landing on the Fisher floor — so we can trust the BNT constraints from a learned summary, not just the
analytic score.

**Scope:** BNT power spectra ONLY, at **14000 deg² first**. Leave the non-BNT (cut-all) results alone —
they're well-understood and as-expected; use non-BNT only as the lossless-identity control. Each scale
cut needs its own compressor.

**The four things the previous attempt got wrong (fix all of them — details in the handoff §3-4):**
1. Stage-2 used `jaxili`, not the validated **sbi_lens RealNVP** (`build_flow`/`train_flow`) — port it.
2. **Train/val split was random → leaked the 7 realizations of each cosmology** across train/val (the
   grid is 2424 cosmologies × 7 reals). Split **by unique cosmology**.
3. An ad-hoc `--summary-noise` knob was tuned to hit truth — **remove it**; if over-confident, use the
   principled **compressor deep-ensemble** (recipe lesson 7).
4. Cholesky-whitening preproc on the ill-conditioned BNT cov amplifies noise — **A/B test** vs
   per-feature standardization. And **z-score θ for the RealNVP** (our H0≈67 breaks `log_prob` — handoff
   §4 "CRITICAL ADAPTATION").

**Non-negotiable gates (a tight contour that fails these is NOT a win — this is exactly how the last
attempt fooled itself with TARP-OK-but-wrong):**
- **Lossless identity (hard):** `bnt_full` null ≈ `nonbnt_full` null (same truth, same width). This is
  the make-or-break check the last attempt FAILED. If it doesn't hold, the pipeline is wrong.
- **Null on truth** (Ωm,S8,w0)=(0.26,0.84,−1.0); **σ near the Fisher floor** (handoff §2); **TARP-DRP
  plot** (look at the ECP-vs-α curve, not just the scalar) + **SBC** rank-std ≈ 0.289.

**Environment:** everything is in the **jaxili** conda env
(`/home/tersenov/anaconda3/envs/jaxili/bin/python`; `sbi_lens`/`haiku`/`optax`/`distrax`/`tfp`/`tarp`
all present — verified). **GPU 2** on titan (no scheduler; check `nvidia-smi`, stagger CUDA inits).
Tension/getdist in the `aname` env.

**How to proceed:** Do NOT start coding or launch training yet. First read the docs above, confirm you
understand the failure diagnosis (re-derive it from the code/results — don't just trust me), then write
a short plan to a file (`docs/PLAN_bnt_vmim_v2.md`) and get my sign-off. Bake the gates in as
back-pressure. Then pilot small first: smoke the sbi_lens RealNVP Stage-2 on existing summaries, fix the
Stage-1 split/preproc, and run the **lossless-identity gate on `bnt_full` vs `nonbnt_full`** — that one
check decides whether the pipeline is correct before any scaling. The score result already stands, so a
VMIM that can't be calibrated is a recordable negative result, not a blocker.
