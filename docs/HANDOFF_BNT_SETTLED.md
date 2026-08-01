# HANDOFF — the BNT scale-cut thread is SETTLED; what remains and how to pick it up

**Written 2026-08-01.** Supersedes `HANDOFF_BNT_VMIM_RESTART.md`, whose premises were largely wrong
(see §6). Read alongside `PAPER_NOTES.md` §4 (the settled story), `PLAN_score_bnt_tension_14000.md`
(ADDENDUM 2026-07-31) and `paper/README.md`.

Git: branch `tension-submean-l37`, HEAD **`fac61e7`**, pushed, working tree clean.
Figures: **39 published in `paper/figures/`, 39 verify clean.**

---

## 1. THE RESULT (settled — do not re-derive)

**BNT localizes the baryon sensitivity to tomographic bin 1, so the scale cut can be applied to that
bin alone. Less information is discarded than by a uniform cut at the same ℓmax, and the contours
tighten by ~1.4×. It is information RETENTION, not better baryon control.**

All numbers at 14000 deg², rebin=20, matched ℓmax=460, 5 NDE seeds, **mean of per-seed FoM₃**.

| configuration | BNT | non-BNT | ratio |
|---|---|---|---|
| **no cut** (ℓmax 1024, 120 feat) | 4.326e5 | 4.420e5 | **0.979** |
| **cut @460, embedding net** (PAPER) | 2.565e5 | 1.826e5 | **1.405** |
| cut @460, MOPED (cross-check) | 1.646e5 | 1.120e5 | **1.470** |

**The decomposition — the cleanest statement of the result:**

```
              uncut      cut@460    cost of the cut
BNT          4.326e5     2.565e5        1.69x
non-BNT      4.420e5     1.826e5        2.42x
ratio of costs = 2.42/1.69 = 1.435   vs directly measured 1.405   (agree to 2%)
```

Both bases start from identical information (the oracle, ratio 0.979 — BNT is an invertible linear
map). The cut costs the standard analysis 2.42× and BNT 1.69×. **BNT does not add information; it
loses less when you cut.**

Per parameter at the cut: σ(Ωm) **1.15×**, σ(S₈) **1.24×**, σ(w₀) **1.04×** — concentrated where
lensing constrains, negligible in w₀.

**Bias at ℓmax=460 is asymmetric — state it honestly.** non-BNT 0.172 ± 0.027σ (safe); BNT
0.304 ± 0.091σ, i.e. *marginally at* the 0.3 threshold, tolerated on its error bar (mean − σ = 0.21).
BNT's own adopted cut is **420**. 460 is used because it is the adopted cut of the main PS analysis
(`ps_submean_l37`: 460 → 0.288σ, 500 → 0.413σ).

### 1.1 Why two extractors, and why that matters

MOPED assumes Gaussianity and needs the analytic covariance and local Jacobian. The embedding network
(16-dim MLP inside the flow, trained jointly under the NPE loss, fed the noise-whitened full data
vector) needs none of them. They share essentially no failure modes and agree to 4%.

**The ratio is also a diagnostic that extraction worked:**

| extractor | ratio | BNT r(Ωm,S₈) | det(R) |
|---|---|---|---|
| MOPED | 1.470 | −0.909 | 0.026 |
| embedding + **ana_whiten** | 1.405 | −0.919 | 0.026 |
| embedding + z-score | 1.026 | −0.884 | 0.109 (partial) |
| raw plain flow | 0.331 | **−0.025** | 0.995 (lost) |

**PREPROCESSING decides it.** z-scoring the raw features leaves the noise correlated and inflates
noise-dominated features to the scale of signal-carrying ones. Whitening by the analytic C⁻¹ᐟ² first
makes the noise isotropic so the ~6 signal directions stand out. Same data, same cut, same flow.

### 1.2 Why compression/whitening is needed at all (measured, alternatives excluded)

| explanation | verdict |
|---|---|
| ill-conditioning | **WRONG.** BNT correlation-matrix cond **8.3e2** vs non-BNT **4.4e3** — BNT is *better* conditioned. (`NOTES_bnt_compression_for_paper.md` quotes ~1e8 for the raw score; measured 1.2e4. **Do not repeat that claim.**) |
| dynamic range / sign changes | real (24×, 29/92 negative) but **irrelevant** — z-scoring removes both |
| dimension (92 vs 50) | **EXCLUDED** — raw non-BNT at **100 features** (rebin 10) keeps r = −0.946 and *improves* |
| **information dilution** | **SUPPORTED — this is it** |

Highest-S/N **10%** of features carry **64%** of the S₈ Fisher in the standard basis but only **5%**
after BNT; median per-feature S/N falls **73 → 6**; for w₀ only **39%** of BNT features reach S/N > 1.
Nulling cancels the dominant common mode and leaves differences, so the signal survives in small
residuals spread across many modes. A flow must then learn ~90 relative weights from 16,965 sims;
errors there damage the joint structure far more than the marginals — exactly the observed failure.
Reproduce: `scripts/diagnostics/why_compression_is_needed.py`.

---

## 2. TWO METHODS FINDINGS WORTH PUBLISHING IN THEIR OWN RIGHT

**(a) TARP and SBC cannot see a wrong degeneracy.** Every configuration above passes both —
*including the raw plain flow* whose posterior has r(Ωm,S₈) = −0.03 against the physical −0.9
(TARP dev 0.115, SBC 0.285/0.282/0.290). The best TARP score in the whole set (0.0325) belongs to an
embedding run with a doubled w₀ offset. Both tests average coverage over the prior and test marginal
rank uniformity per parameter, so neither sees joint structure nor local behaviour at the single
fiducial where every FoM is measured.

**(b) Rebinning does NOT cost realizable information, contrary to the Fisher.** MOPED at rebin
20/10/5/2/1 (matched @460, calibrated at every rung): Fisher predicts **2.1–2.6×** going to native;
the trained posteriors deliver **+3%** (BNT, peak at r5) and **+11%** (non-BNT, peak at r2), and
native is *worse* than r20 for BNT. NPE/Fisher falls **0.53 → 0.24** as binning refines — the
analytic covariance keeps promising more and the posterior declines to deliver. rebin=20 is
vindicated as production. → `outputs/diagnostics/score_rebin_ladder_fom.csv`.

---

## 3. WHAT WAS DESTROYED AND REBUILT THIS SESSION

### 3.1 `scripts/score_cut_utils.py` — rewritten from scratch
The original was 5144 B of 100% NUL, untracked. Rebuilt from five surviving call sites +
`PLAN_score_bnt_tension_14000.md` §1, with the covariance/Jacobian lifted verbatim from the surviving
`score_compress.py`. **Validated three ways:** `keep_indices` reproduces the in_dim of all 36 VMIM
sweep runs; the gate-1 slice oracle is bit-exact (`max|dx| = 0.00e+00`); and the Fisher ratio
reproduces the pre-crash 1.46 (measured 1.455). Run `python scripts/score_cut_utils.py` to re-verify.

### 3.2 The entire analytic-covariance cache was destroyed — and is now regenerated
**Every** `gaussian_cov_native_*.npy` (6), `cw_*.fits` (6) and `w_*.fits` (6) in
`scripts/diagnostics/cache_gaussian_cov/` carried the 3 MiB RAID0 stripe signature, ~20% zeroed.
**A zeroed covariance is INDEFINITE, not merely singular**, so numpy inverts it and returns
plausible nonsense — it gave a Fisher ratio of 1.377 where the truth is 1.455.

I initially dismissed that file's 21% zero fraction as block structure. **That is exactly the mistake
the forensics note warns about for large binaries.** Always check the stripe signature, not the
zero fraction.

- **Regenerated for area 14000** via `scripts/jz/rebuild_gaussian_cov.slurm` (4 minutes, not the 20 h
  allocated). Validated against the surviving rebinned cache to **3.8e-16**.
- The other five areas are **still destroyed**. Re-run the job with `AREA=<n>` if needed.
- `cov_rebinned_full_*.npz` survived (234 KB, under the stripe period) and are backed up read-only in
  `cache_gaussian_cov/_INTACT_PRECRASH_BACKUP/`. `score_cut_utils.analytic_cov_at()` falls back to
  them at rebin=20 if the native file is missing or fails its sanity check.
- Damaged originals quarantined in `cache_gaussian_cov/_DAMAGED_RAID0/`.

### 3.3 `scripts/score_bnt_tension_sweep.py` is 100% NUL — destroyed
Not on the previous handoff's lost list. **Treat that inventory as a floor, not a total.** Its
compression step is replaced by `scripts/score_compress_at_rebin.py`, gated bit-exactly against the
surviving rebin=20 caches for both arms.

### 3.4 A leaky early-stopping split, fixed
`nde_realnvp_from_summary.py` split train/val by a random permutation over **rows**. Each cosmology
has ~7 realizations, so validation shared cosmologies with training and early stopping never fired
honestly. Now **by cosmology**, with a leakage assert; `--split-rows` keeps the old behaviour as an
ablation. Damage scales with parameter count — mild for a 6-dim MOPED summary, material with an
embedding net.

---

## 4. MACHINERY — how to run any of this again

```bash
PYJ=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python   # contours, NDE, numpy
PYA=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python    # tables, tension, getdist 1.4.3
PYN=/lustre/fswork/projects/rech/nzu/ulx34io/envs/namaster2/bin/python # NaMaster 2.4 (PYTHONNOUSERSITE=1!)
```

| script | does |
|---|---|
| `scripts/score_cut_utils.py` | cut slicing + MOPED (`keep_indices`, `build_score`). Run bare to self-validate. |
| `scripts/score_compress_at_rebin.py` | builds the NDE input at any rebin. `--mode moped\|raw\|whiten\|ana_whiten` |
| `scripts/nde_realnvp.py` | `build_flow` and `build_flow_embedded` (the embedding net) |
| `scripts/nde_realnvp_from_summary.py` | trains K seeds, samples null+biased, TARP/SBC bundle. `--embed-dim` |
| `scripts/jz/score_rebin_ladder.slurm` | the driver for everything above. Env: `REBINS ARMS CUT COVK MODE EMBED_DIM EMBED_HIDDEN ANAW_CLIP ROOT SEEDS` |
| `scripts/jz/rebuild_gaussian_cov.slurm` | regenerate the analytic covariance (quarantines damaged inputs, validates against the intact reference) |
| `scripts/plot_posterior_overlay.py` | overlay posteriors from arbitrary dirs; `--seed-mode pooled\|single` |
| `scripts/diagnostics/why_compression_is_needed.py` | the dilution measurement |
| `scripts/diagnostics/fisher_fom_vs_rebin.py` | Fisher vs rebin |
| `scripts/paper/figures.py` | `publish` / `republish` / `manifest` / `verify` |

**Reproduce the paper figure:**
```bash
REBINS=20 ARMS="bnt nonbnt" CUT=460 COVK=hybrid MODE=ana_whiten EMBED_DIM=16 \
  EMBED_HIDDEN="256,256" ROOT=outputs/score_embed_anaw16 sbatch --export=ALL \
  scripts/jz/score_rebin_ladder.slurm
```
(~15 min on one A100. `qos_gpu_a100-dev`, 2 h cap, is plenty.)

**TARP needs two things:** `PYTHONPATH=$PWD/tarp/src` (the package is a source checkout, not
installed) and `deprecation` (now pip-installed into aname). Without the PYTHONPATH, `import tarp`
silently finds an empty namespace package.

---

## 5. STILL OPEN

| item | status |
|---|---|
| **~500 MB of `outputs/score_*` posteriors are NOT archived** | Twelve directories holding every trained posterior behind every number above. Gitignored by design. **Archive to `$STORE` with tar** (`scripts/jz/archive_science_to_store.slurm`, `--partition=archive`, 100k inode quota → tar not rsync). This session began by recovering from losing exactly this kind of file. |
| **The embedding exceeds the Gaussian Fisher by 1.25–1.36×** | With a w₀ offset that grows with the FoM (−0.027 MOPED → −0.044 embedding). Real non-Gaussian information or fiducial-local over-confidence — unresolved. **Common mode** (−0.0435 BNT vs −0.0436 non-BNT), so it cancels in the ratio and does not touch the headline. Decisive test available: the fiducial "observation" is the MEAN of 200 permutations while every training row is a single realization; evaluate at a single realization instead and see whether the posterior widens to the Fisher. One run. |
| **A pre-existing w₀ offset of ≈ −0.025 in every method**, MOPED included | Wider contours were masking it. Not baryon-related (these are null/nobaryons posteriors). |
| **`NOTES_bnt_compression_for_paper.md` contradicts the settled story** | It is the paper-facing methods doc and still leads with ill-conditioning + a "~1e8" condition number that measures 1.2e4. `PAPER_NOTES` §4 now says otherwise. **Reconcile before drafting methods.** |
| **Five of six areas still have a destroyed analytic covariance** | Only 14000 regenerated. |
| **TARP/SBC for the corrected NPE posteriors of the OTHER (non-BNT-thread) configs** | Unchanged from before; still the largest referee exposure outside this thread. |
| **Baryon-model dependence** | One baryonification prescription throughout. Needs new sims. |
| **2.22σ vs 1.43σ** | `PLAN_score_bnt_tension_14000.md`: the published 2.22σ PS headline at 14000 does not reproduce (1.43 ± 0.30). Unresolved; a scientific decision. |

---

## 6. WHERE THE PREVIOUS HANDOFF WAS WRONG

Recorded so the same time isn't spent twice:

- **"3 collapsed seeds need retraining"** — they were disk-damaged *pooled* files sitting on top of
  healthy per-seed posteriors. Two were fixable by re-concatenation; no GPU retraining was needed.
- **"the sweep is complete but the FoM comparison doesn't exist"** — it did exist, in
  `PLAN_score_bnt_tension_14000.md`'s DE-BIASING FoM CONTROL section.
- **"BNT controls baryons better"** (`PAPER_NOTES` §4, pre-correction) — it does not. BNT crosses
  0.3σ at a *lower* ℓmax (460) than non-BNT cut-all (620).
- **"the BNT vector is ill-conditioned"** — measured false; see §1.2.
- **The loss inventory was incomplete** — `score_bnt_tension_sweep.py` and the entire
  `cache_gaussian_cov/` binary set were destroyed and unlisted.

---

## 7. TRAPS

- **A zero-fraction test lies on large binaries.** Check for the RAID0 signature: ~512 KiB zero runs
  repeating every **3 MiB**. 21% zeros in a covariance file was damage, not block structure.
- **A zeroed covariance is INDEFINITE, not singular** — numpy inverts it happily and returns
  plausible nonsense. `score_cut_utils._assert_covariance_sane` now checks PSD *and* trace == Σeig.
- **A damaged `.npy` raises `ValueError`, not IOError**; a *partially* zeroed one loads fine and
  returns correct-shaped garbage. Guard both. Never `allow_pickle=True`.
- **`pkill -f <pattern>` matches the invoking shell's own command line** and kills the job you just
  launched. Cost two runs.
- **`--series "Label=dir:tag"` splits on the first `=`** — a LaTeX label containing `=` breaks it.
- **The ladder names output dirs by the cut**, so `CUT=1024` writes `..._c1024`, not `_c460`.
- **getdist 1.4.3 (aname) cannot fill contours** under matplotlib ≥3.8 → contour work in **jaxili**
  (getdist 1.6.1); table work in **aname** (jaxili has no pandas).
- **`import tarp` finds an empty namespace package** unless `PYTHONPATH=$PWD/tarp/src`.
- **Size GPU walltime from a live log's epoch rate.** `qos_gpu_a100-dev` (2 h) for short jobs — a
  20 h `-t3` request once bought a 34 h queue wait for a 35-minute job.
- **Don't poll running jobs continuously** (admin risk). Check once, then wait.
