# PLAN — Baryon-bias tension vs scale cut, with monopole-subtracted PS from ℓ=37

**Status:** draft for sign-off. Nothing is launched until the plan is approved and
Phase 1/2 (script review + refactor + regression gate) pass.

## 1. Goal & scientific framing

Reproduce the paper's "how aggressively must we cut small scales before baryonic
feedback is negligible" analysis, but on the **monopole-subtracted** power spectra,
which let the ℓ-floor drop from **100 → 37**.

The measurement is unchanged in spirit: for each footprint and each progressively
lower upper-ℓ cut, run NPE twice — a **null** posterior (`nobaryons` observation vs
`nobaryons`-trained model, centred on truth) and a **biased** posterior
(`baryonified` observation vs the same `nobaryons` model, shifted by feedback) — then
measure the **null↔biased tension in σ**. As the upper cut drops, the feedback-
contaminated high-ℓ leaves the data vector, the biased posterior walks back toward
truth, and the tension falls. We read off the cut at which it drops below threshold.

What's new at ℓ=37: the recovered low-ℓ band is baryon-*insensitive* but adds
constraining power. So at fixed upper cut it tightens the contours, which can *raise*
the σ-tension (a fixed feedback shift becomes more σ when the contour is smaller) even
though no new bias is added. Whether the required scale cut moves is exactly what this
quantifies. A clean side-result: **ℓ100-submean vs ℓ100-raw** at matched cut isolates
whether monopole subtraction changed anything at ℓ≥100 (should be ~no-op); **ℓ37 vs
ℓ100 submean** isolates what the recovered low-ℓ contributes.

## 2. Decisions locked (from interview, 2026-06-19)

| Decision | Choice |
|---|---|
| Footprints | **14000 first** (validate end-to-end), then all six (2000/5000/10000/28000/35000) |
| NPE repetitions | **5 runs per config**, tension reported as mean ± std across runs |
| Tension estimator | **Gaussian Q_DM headline** (reproduce paper), module structured so a non-Gaussian parameter-shift estimator can be dropped in later |
| Refactor ambition | **Heavy — extract a shared module**; the per-statistic scripts become thin configs over it |
| O1 — runs marginalise over | **Training seed only** (same baked observation, `random_seed = base + run`), as in the paper. Observation-noise scatter revisited only if a problem shows up. |
| O2 — module location | **`scripts/tension/`** package (confirmed). |
| O3 — upper-cut grid | **Keep the paper grid**: 340…1020 step 20 (35 points). |
| Organization | **First-class requirement.** Dedicated, browsable output tree + clear, readable, subdivided filenames (see §8). |
| Robustness | **NaN-loss auto-retry + per-posterior QA gate** required, since the sweep runs unattended in a loop (see §4.1). |

## 3. Current-state findings — bug & issue inventory

Found while tracing the existing scripts. These drive Phase 1/2.

**B1 — broken `utilities` import (correctness, silent).**
`compute_tension_statistics.py` line 26 was changed (Feb-4 commit) to
`import tensiometer.utilities as utilities5`, but line 95 still calls
`utilities.from_confidence_to_sigma(...)` → `NameError`, swallowed by the `try/except`
in `compute_tension` → the script would silently emit **empty** tables. The `_fullsky`
variant has the correct import. Fix in the shared module (one import, used everywhere).

**B2 — "runs" are not reproducibly seeded (methodology). [mechanism corrected in Phase 1]**
`--run N` only changes the output **filename** (NPE script lines 1007/1187), and
`--random-seed` (default 1, line 1108) controls **only the posterior-sampling key**, not
training. The actual run-to-run scatter (14000 @ ucut 520: nσ = 0.19/0.24/0.42) comes from
jaxili internals the script never seeds:
- `inference.append_simulations(params, data)` is called **without `key=`** → jaxili falls
  back to `jr.PRNGKey(np.random.randint(0,1000))` (npe.py:215) for the **train/val split**,
  and numpy's global RNG is not seeded beforehand → a different split every process.
- `inference.train(...)` is called **without `seed=`** → jaxili uses its default
  `seed=42` (npe.py:560) for flow init + batch order → **fixed across all runs**.
So the paper's multiplicity was driven by an *uncontrolled* split, with init pinned — not
reproducible. **Fix:** pass `key=jr.PRNGKey(base+run)` to `append_simulations` AND
`seed=base+run` to `train()`. Then each run is reproducible *and* genuinely independent.

**B2b — NaN-loss training failures + why the old retry "didn't work".**
jaxili NPE sometimes diverges (flow loss → NaN). A previous NaN-check + rerun was
unreliable and ended up needing manual diagnosis/reruns. Most likely cause: **a same-seed
retry reproduces the same NaN deterministically.** Since `--run` never changed the seed
(B2), the retry was effectively a no-op. Fixing B2 (`seed = base + run`) also fixes the
retry — on NaN, retry with a *fresh* seed. This becomes a first-class feature of the
sweep runner (§4.1), not an afterthought.

**B3 — what the 5 runs marginalise over. RESOLVED (O1): training seed only**, matching
the paper (same baked observation; `random_seed = base + run`). The fiducial files hold
~200 perms, so observation-noise scatter is available as a documented secondary check if
training-only proves inadequate — but it is out of scope for the headline run.

**B4 — hard-coded run-patching (smell).**
`compute_tension_statistics_fixed_nobaryons.py` always loads the nobaryons sample from
run3 for mask 5000, regardless of `--run`. This is a hand-patch around bad/outlier
runs. The new protocol should make this unnecessary and instead **detect & flag**
outlier runs (e.g. width or mean far from the run-median) rather than silently
hard-code one.

**B5 — lmax special-case is stale for submean.**
The old sweep used `lmax=1530` for the 14000 footprint (raw) and 1535 elsewhere. The
monopole-subtracted production is **1535 for all footprints** (→ nlb=4). Drop the
special-case; always `--lmax 1535` with `--subtract-mean`.

**B6 — dead/duplicated code.**
`get_sample_filename` in the PS tension script computes an `lmax` (1530/1535) that is
never used in the returned name. The five `compute_tension_statistics*` scripts are
~90% identical (same Q_DM core, CSV/pivot output); they differ only in (a) filename
builder and (b) sweep axes (PS: upper_cut×area; fullsky: ell_max×rebin±BNT; l1/peaks:
scales×area). Textbook consolidation target.

**B7 — `--batch-size` is silently ignored.** The script calls `inference.train(...,
batch_size=args.batch_size, ...)`, but jaxili's `train()` signature is
`train(training_batch_size=50, ...)` (npe.py:470). `batch_size` lands in `**kwargs` and is
never used → batch size is always the default 50. Fix: pass `training_batch_size=`.

### 3.1 Phase-1 design-lock facts (verified against jaxili source + NPE script)
- **NaN hook:** `metrics, density_estimator = inference.train(...)`; `metrics` carries
  `'train/loss'`,`'val/loss'`,`'test/loss'` (npe.py:511-512,578). Detect divergence by
  `np.isfinite` on these + on the drawn samples.
- **Reproducible per-run training** needs BOTH seeds: `append_simulations(key=PRNGKey(s))`
  (train/val split) and `train(seed=s)` (init+optim), with `s = base + run`.
- **Observation = perm mean.** Auto/cross fiducials are `np.mean(fid_full, axis=0)` over
  ~200 noisy perms (lines 548/613) → a fixed, ≈noiseless target, identical across runs.
  Justifies a tight QA null-on-truth tolerance and confirms O1 (training-only scatter).
- **Scale-cut indexing** under nlb=4 (lmax>1500) is handled in `load_and_process_*`
  (binned branch); the campaign's `l37-1024` submean endpoints already validate it.
- **Envs (three, do not mix):** NPE = `jaxili` (getdist 1.6.1); processing = `cosmostat_new`
  (pymaster); **tension = `aname`** = `/home/tersenov/anaconda3/envs/aname/bin/python`
  (tensiometer + getdist 1.4.3). Run `compute_tension.py` with the `aname` interpreter.
- **Regression gate (Stage-2) ALREADY GREEN:** `scripts/tension/estimators.py` reproduces
  the paper CSV nsigma to 1e-5 (520/14000=0.19025, 520/2000=0.01875, 520/5000=0.00549),
  confirming the B1 import fix and Q_DM parity. (NaNs warning-free; getdist "Removed no
  burn in" is benign.)

## 4. Target architecture — the shared module

A new self-contained package, **`scripts/tension/`** (importable by thin entrypoints;
**not** plugged into the half-finished/broken `src/bar_impact/` — see CLAUDE.md). Pure
functions, no classes-for-classes'-sake. Proposed layout:

```
scripts/tension/
  __init__.py
  paths.py       # SINGLE source of truth for the output tree + filename builders (§8).
                 #   Every read/write path derives from here — no ad-hoc string-building elsewhere.
  io.py          # safe load → MCSamples; supports legacy flat layout (old paper posteriors) + new tree
  estimators.py  # Q_DM (Gaussian) now; param-shift hook later. Single correct `utilities` import.
  qa.py          # per-posterior diagnostics gate (§4.1): NaN/degeneracy/prior-rail/null-on-truth/outlier
  sweep.py       # config-driven NPE runner (replaces submit_*_parameter_sweep_parallel.py):
                 #   GPU pool, seed=base+run, NaN-loss auto-retry (fresh seed), QA gate, resume, logging
  aggregate.py   # collect per-(config,run) σ → mean ± std tables, pivots; drop/flag QA-failed runs
  configs.py     # one declarative config per statistic: axes, filename builder, tags
  plots.py       # σ-vs-upper_cut curves (one line per footprint; error bands from the runs)
```

Thin entrypoints (keep the standalone-script convention):
`scripts/run_tension_sweep.py` (drives NPE via `sweep.py`) and
`scripts/compute_tension.py --statistic {ps,ps_fullsky,l1,peaks} [--submean --lower-cut 37 ...]`
(replaces all five `compute_tension_statistics*` scripts). Old scripts are kept until
the regression gate passes, then moved to `scripts/archive/`.

### 4.1 Robustness: NaN-loss auto-retry + posterior QA gate

The sweep runs unattended in a loop, so every posterior must be **earned**, not assumed.
Two layers:

**(a) NaN-loss auto-retry (during training).** Capture the flow's per-epoch loss from
jaxili. If any loss is NaN/Inf (or training raises), abort that attempt and **retry with a
fresh seed** (`base + run` → `base + run + K·offset`), up to `max_retries` (default 3). If
all attempts fail, mark the config **FAILED** (logged, never a silent empty/garbage file).
Exact jaxili loss-history hook is a Phase-1 item (the old check existed, so the loss is
reachable). Optional escalation on repeat failure (e.g. lower learning rate) — only if
fresh-seed retries prove insufficient.

**(b) Posterior QA gate (after sampling, before accept).** Cheap automated checks, each
emitting OK / FLAG(reason) to a per-run QA record:
- finite: no NaN/Inf in samples;
- not degenerate: per-param std neither ≈0 (collapsed) nor ≈ prior width (unconstrained →
  training likely failed);
- not prior-railing: posterior mass not piled at a prior edge beyond a threshold;
- null-on-truth: for `nobaryons_vs_nobaryons`, the mean is within a tolerance of truth
  (a biased *null* signals a failed fit, not physics);
- outlier-vs-siblings: across the 5 runs of a config, flag any run whose mean/width is far
  from the run-median (this is the principled replacement for the B4 `run3` hard-patch).

`aggregate.py` consumes the QA records: FLAGGED runs are excluded from the mean ± std (and
the count of excluded runs is reported), so a few bad fits can't move the headline tension.
Thresholds are config constants, tuned on the Phase-3 pilot and recorded here.

## 5. Phased plan (with verification oracles)

**Phase 0 — branch.** New branch off `bnt_inference` (e.g. `tension-submean-l37`).
All edits live there.

**Phase 1 — deep read + design lock (together, no big compute).**
Read in full: `run_npe_inference_auto_cross_ps_master.py` (esp. how the fiducial
observation/perm and noise are selected; how the scale cut maps to bandpower indices
under nlb=4), all five tension scripts, the sweep submit script, `load_tension_arrays.py`.
Resolve O1–O3 (§6). Produce a short design note appended here. **Oracle:** a written,
agreed bug list (B1–B6 confirmed/closed) + module API signed off.

**Phase 2 — build the module + regression gate (cheap, decisive).**
Implement `scripts/tension/` and the two entrypoints. Then the **back-pressure oracle**:
run the new `compute_tension.py --statistic ps` over the **existing paper raw-ℓ100
posteriors** and reproduce the existing `outputs/tension_analysis/tension_*_runN.csv`
to numerical tolerance. If we can't reproduce the old numbers, the refactor is wrong —
fix before touching the submean run. (This is why B1 matters: the *current* on-disk
script can't even produce them.)

**Phase 3 — pilot (14000, validate + time).**
Run the full submean/ℓ37 sweep for **14000 only**: upper_cut ∈ {340..1020 step 20} ×
{nobaryons, baryonified} × 5 seeded runs = 350 NPE trainings, with NaN-retry + QA gate
live. Then `compute_tension.py --statistic ps --submean --lower-cut 37`. **Oracles:**
(i) null posterior sits on truth at every cut; (ii) tension decreases
monotonically-ish as upper_cut drops; (iii) ℓ100-submean reproduces ℓ100-raw tension
(monopole no-op at ℓ≥100); (iv) per-config run scatter is sane (std/mean sensible);
(v) QA gate flags ≈0 runs on a healthy config, and any NaN-retry actually recovers.
Time one job, tune QA thresholds, → project the all-six cost. **Decision gate:** review
14000 numbers before launching the rest.

**Phase 4 — full sweep (all six footprints).**
Same sweep for 2000/5000/10000/28000/35000 (5 seeded runs each). Background, GPU pool,
skip-existing so it's resumable. Likely overnight.

**Phase 5 — tension tables + figures.**
`aggregate.py` → mean ± std σ pivots (rows=upper_cut, cols=footprint), full-6 and
3-param (Ωm,S₈,w₀). `plots.py` → σ-vs-upper_cut curves with run-scatter error bands,
one panel/line per footprint, ℓ37 vs the paper's ℓ100 overlaid. Write results into a
results section of this doc + memory.

## 6. Open questions — RESOLVED

- **O1** → training seed only (paper-matched). **O2** → `scripts/tension/` package.
  **O3** → keep the paper grid (340…1020 step 20). Remaining Phase-1 unknowns are
  mechanical, not decisions: the jaxili loss-history hook for NaN detection, and the
  exact fiducial-observation/perm selection in the NPE script (read, don't guess).

## 7. Cost estimate (refine after Phase 3 timing)

Per footprint: 35 cuts × 2 fiducials × 5 runs = **350 NPE trainings**. At an assumed
~2.5 min/training and G GPUs in parallel: 14000 pilot ≈ 350×2.5/G min (~5 h at G=3).
All six ≈ 2100 trainings (~22 GPU-h; one overnight with a 3–4 GPU pool). Tension
computation itself is seconds. Knobs if too slow: coarsen O3 grid, fewer epochs, fewer
runs — decided at the Phase-3 gate, not now.

## 8. Output organization & naming (first-class requirement)

Everything for this analysis lives under one dedicated, browsable root —
**`outputs/baryon_tension/`** — never scattered into the giant flat `outputs/samples/`.
A single `paths.py` is the only place that builds paths, so the tree can't drift.

**Campaign tag** = `{statistic}_{gauge}_{lmin}`, e.g. `ps_submean_l37` (this run) vs
`ps_raw_l100` (the paper). The tag disambiguates everything at a glance.

```
outputs/baryon_tension/
  README.md                         # what this tree is; regenerated index of campaigns
  ps_submean_l37/                    # one dir per campaign
    manifest.json                   # exact grid, seeds, code commit, date — reproducibility
    posteriors/
      mask_02000/  null/   posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_..._l37-{uc}_..._run{N}.npy
                   biased/ posterior_samples_ps_auto_cross_nobaryons_vs_baryonified_..._l37-{uc}_..._run{N}.npy
      mask_05000/  ...
      mask_14000/  ...                # zero-padded so footprints sort naturally
      ...
    qa/
      qa_report.csv                  # one row per (footprint, cut, fiducial, run): status, reason, mean, std, n_retries
      failures.log                   # configs that exhausted NaN-retries
    logs/                            # per-job NPE training logs
    tables/
      tension_3param_mean.csv  tension_3param_std.csv  tension_3param_pivot.csv
      tension_6param_mean.csv  tension_6param_std.csv  tension_6param_pivot.csv
    figures/
      sigma_vs_uppercut_3param.png   # σ vs upper-cut, one line per footprint, run-scatter band
      sigma_vs_uppercut_l37_vs_l100.png
```

Conventions: footprint dirs zero-padded (`mask_02000`…`mask_35000`); `null/` vs `biased/`
split so the two fiducial types are visually separate; the NPE script's descriptive
filename is kept verbatim (path carries the campaign/footprint context, filename carries
the cut/run). Legacy paper posteriors stay where they are in `outputs/samples/`; `io.py`
reads that flat layout for the regression gate only.

## 9. Build log

**2026-06-19 — Phase 2 module built + Stage-2 gate GREEN (steps 1–3 of the agreed order).**
`scripts/tension/` now has: `estimators.py`, `paths.py`, `io.py`, `configs.py`,
`aggregate.py`, `qa.py`, plus the entrypoint `scripts/compute_tension.py`. Verifications:
- `estimators.py` + `compute_tension.py --paper-raw` reproduce the published CSVs to
  **max |Δnσ| = 5e-6** over all 130 matched grid rows (3- and 6-param) — the tension stage
  is faithful. Run with the **`aname`** interpreter.
- `paths.py` filename builders match real on-disk files (submean campaign + legacy paper).
- `qa.py` catches NaN/collapsed single-posteriors and, crucially, **`flag_outlier_posteriors`
  flags a real bad fit** by width: the `outputs/samples` copy of the 14000 `l37-1024` null is
  a failed fit (σ(S8)=0.205, mean S8=0.697 off truth), while the good campaign copies in
  `outputs/diagnostics/lmin_compare/…` have σ(S8)=0.024 on truth. The MAD/width rule flags the
  bad one and keeps the 4 good siblings.

**Findings:**
- **Don't trust the flat `outputs/samples` submean `l37-1024` endpoints** — at least the 14000
  null is a silent failed fit. The good copies live under `lmin_compare/`. The Phase-3 sweep
  regenerates everything cleanly into the tree (5 seeded runs) with QA, so this is moot for the
  production run — but it's exactly the silent failure the QA gate exists to catch, and it
  explains the historical `_fixed_nobaryons` hand-patch.
- **QA threshold note:** absolute single-posterior thresholds (null-off-truth 5σ, unconstrained
  0.9·σ_uniform) did NOT catch that bad fit on their own (it was 0.7σ off, just under the width
  cut). The **cross-run width-outlier check is the workhorse**; absolute thresholds are a
  backstop for gross failures (NaN/collapse/full railing). Calibrate both on the pilot.

**Next (step 4, needs review before it runs):** the surgical worker edit
(`append_simulations(key=PRNGKey(base+run))` + `train(seed=base+run, training_batch_size=…)` +
non-finite-loss `sys.exit(42)`) and `sweep.py` (GPU pool, NaN-retry with bumped seed, QA gate,
resume, manifest). Diff to be shown before it touches the shared worker; nothing trains until
approved.

**2026-06-19 — Step 4 built (worker patch + sweep), validated dry, NOT yet trained.**
- Worker `run_npe_inference_auto_cross_ps_master.py` patched (3 changes, compiles): seeded
  train/val split `append_simulations(key=PRNGKey(--random-seed))`; `train(seed=--random-seed,
  training_batch_size=…)` (also fixes B7); non-finite-loss → `sys.exit(42)` without saving.
- `scripts/tension/sweep.py` + `scripts/run_tension_sweep.py`: GPU pool, per-run seed
  (base+run), NaN/crash retry with bumped seed, per-posterior QA → `qa/qa_report.csv`,
  manifest, resume (skip existing). `--dry-run` validated: 14000 pilot = 350 jobs
  (175 null + 175 biased), all-six = 2100; worker cmd + tree output paths correct.
- Also removed 25 stale nlb=1-era PS submean posteriors from `outputs/samples/` (list in
  `docs/removed_nlb1_submean_posteriors_2026-06-19.txt`); peak-counts submean (user's) and
  the good nlb=4 `lmin_compare/` copies untouched.
- **Next (Phase 3, needs go + GPUs):** 1-job smoke test → 14000 pilot on free GPUs.

**2026-06-19 — Smoke test PASSED (worker fix validated end-to-end).**
First smoke run exposed a jaxili bug: `train(seed=...)` double-passes seed to create_trainer
("multiple values for 'seed'") — removed it; per-run variation now comes from the seeded
train/val split (append_simulations key), which matches the paper's scatter mechanism (init
pinned at 42). Also gave each job a unique checkpoint dir (worker ckpt name omits run →
parallel runs would collide) + auto-cleanup. Re-run (14000, l37-1024, run1, GPU2):
- both jobs OK, QA OK, 0 failed; outputs in the tree; manifest + qa_report written.
- null σ(Om,S8,w0)=[0.013,0.024,0.076] on truth — matches the known-good campaign
  [0.013,0.024,0.080]; biased shows the feedback shift (w0 −1.19 vs null −1.02).
- end-to-end tension (single run, l37-1024): 3-param 2.03σ, 6-param 2.52σ.
- **timing ≈ 140 s/job** (incl. JAX init + data load + train + sample). One job per GPU
  (~34 GB). 14000 pilot = 350 jobs → ≈6.8 h on 2 GPUs. All-six ≈ 6×.
- Disk note: each job writes a ~33 MB `datavectors_npe_input_*` byproduct to its dir
  (~2.3 GB at 14000; regenerable) — add cleanup before the all-six run.

**2026-06-19 — Decision: binning/rebin (option a).** Investigated why the new submean
σ-vs-cut curve stair-steps (adjacent step-20 cuts byte-identical) while the paper's didn't.
Root cause: BOTH would under nlb=4+rebin=10, but the **paper posteriors used nlb=1 (per-ℓ,
lmax=1024) data** (10-ℓ effective bins → distinct step-20); the original nlb=1 grid is gone,
replaced by this session's nlb=4 grids. nlb=4 was adopted for the submean low-ℓ recovery
(nlb=1 ill-conditioned at low f_sky). nlb=4 × rebin=10 = 40-ℓ bins → ~half of step-20 cuts
coincide (17/34, any lmin). **DECISION (user, option a): accept the 40-ℓ binning** — keep
rebin=10, take the qualitative scale-cut trend; the current pilot stays valid (no re-run).
**For all-six: use a step-40 upper-cut grid** (340,380,…,1020 = 18 cuts) — identical info,
half the jobs. Combined with GPU packing (×2.8): all-six ≈ 7–8 h. Caveat noted: 40-ℓ bins are
4× coarser than the paper's 10-ℓ, so these tension magnitudes aren't a like-for-like match to
the published numbers (would need rebin≈3 for that — deferred, option b).

**2026-06-19 — 14000 PILOT COMPLETE (Phase-3 gate PASSED).**
330/330 OK, 0 QA-flagged, 0 real failures (the 2 in failures.csv are stale smoke-test cut-1024).
8 transient NaN/crash retries auto-recovered → retry+QA machinery validated. Throughput 2.8×
(packed, ~2.3 h). Curve (3-param Q_DM, mean±std/5 runs): 0.04σ@ℓ340 → ~2.2σ@ℓ1020, crosses 0.3σ
at ℓmax≈460–480. Plot: plots/nsigma_vs_upper_cut_masks.{png,pdf} (notebook style, live monitor).
**Next: all-six** — step-40 grid (18 cuts), packed, 6 areas × 5 runs = 1080 jobs ≈ 7.5 h;
relaunch monitor with all six areas (6-panel). Awaiting go.

**2026-06-20 — ALL-SIX RUN COMPLETE (overnight, unattended).**
Step-40 grid, packed (4/gpu, mem 0.15, GPUs 2+3). 900 jobs (5 remaining areas; 14000 skipped by
resume), **0 hard failures**, 0 retries needed. Finished ~05:22 UTC (~7 h, on estimate).
- **QA:** the sweep first reported 152 "collapsed" flags — investigated and confirmed FALSE
  POSITIVES: all in the two largest footprints (28000/35000), which constrain S8 to ~1.4–2% of
  the prior *legitimately* (on-truth, tight). The collapse threshold (2% of prior) was too
  aggressive. Fixed COLLAPSE_FRAC 0.02→0.005 in qa.py (a real collapse has σ≈0); re-assessed all
  1080 posteriors → **1080/1080 OK, clean report**. Flags never affected results (aggregation
  ignores QA flags; tension is computed on every run).
- **Deliverables:** full 3+6-param tension tables (all 6 footprints) in
  outputs/baryon_tension/ps_submean_l37/tables/; 6-panel deduped figure
  plots/nsigma_vs_upper_cut_masks.{png,pdf}; clean qa_report.csv; manifest.
- **Result:** tension rises monotonically with ℓmax and with footprint area. 0.3σ crossing scales
  with area — 2000:ℓ900, 5000:ℓ620, 10000:ℓ580, 14000:ℓ500, 28000:ℓ420, 35000:ℓ380. At ℓmax=1020:
  0.43σ(2000) → 3.56σ(35000). Bigger surveys need more aggressive small-scale cuts to keep
  baryon bias sub-0.3σ.
- Note: failures.csv still holds 2 stale smoke-test rows (cut 1024, outside the production grid) —
  harmless. Caveat from option (a) still stands: 40-ℓ bins are 4× coarser than the paper's 10-ℓ,
  so magnitudes aren't a like-for-like match to the published numbers (rebin≈3 = option b).

**2026-06-20 — FULL-SKY panel added (7th).** New healpy-pipeline campaign `ps_fullsky_l37`
(full-sky NPE worker `run_npe_inference_auto_cross_ps.py`, patched: seeded split + NaN gate +
`--run`). Threaded `pipeline="healpy"` through configs/paths/sweep (default "master" unchanged);
`--fullsky` on run_tension_sweep.py + compute_tension.py; `fullsky_campaign()`. Full sky is
per-ℓ (10-ℓ bins) vs masked nlb=4 (40-ℓ) — different estimator, NOT magnitude-comparable
(option-a caveat), same scale-cut trend. Grid EXTENDED to ℓmax=100 (24 step-40 cuts 100–1020)
because full sky is so constraining its 0.3σ crossing (~ℓ320) is below the masked grid's 340 floor.
240 jobs, 0 flagged, 0 failed. **0.3σ crossing sequence (monotonic in f_sky): 2000:ℓ900, 5000:ℓ620,
10000:ℓ580, 14000:ℓ500, 28000:ℓ420, 35000:ℓ380, full-sky:~ℓ320.** Deliverables:
`plots/nsigma_vs_upper_cut_with_fullsky.{png,pdf}` (7-panel, builder
`scripts/build_7panel_tension_plot.py`), full-sky tables in `outputs/baryon_tension/ps_fullsky_l37/`.
Live monitor `scripts/monitor_fullsky_7panel.py` (merges static masked + live full sky). Masked
panels start at ℓ340 (their crossings all ≥340) → small empty space left of 340 in shared-x.
