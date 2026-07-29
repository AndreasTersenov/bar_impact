# PAPER FIGURE MAP — what exists, what it's worth, what's missing

**Built 2026-07-30** by an 8-agent survey (4 parallel inventories → cross-join → 2 opportunity
lenses → completeness critic; 409 tool calls). Every claim below that carries a number was
re-verified directly afterwards; the ones that were checked are marked ✔.

Scope: **9,550 figure files** across 32 directories, of which **1,907 are destroyed**. Those
collapse to **63 distinct figure concepts**. Provenance sidecars: **7 of 63**.

| state | n | meaning |
|---|---|---|
| READY | 7 | exists, corrected convention, has sidecars |
| NEEDS_REGEN | 24 | generator + inputs exist; needs running |
| NEEDS_NEW_SCRIPT | 10 | data exists, nothing builds it |
| BLOCKED_ON_COMPUTE | 9 | needs a GPU/CPU job first |
| LOST_UNRECOVERABLE | 5 | destroyed, inputs gone |
| SUPERSEDED | 8 | a later correction retired it |

---

## 1. Read this before touching anything

### 1.1 The paper's headline PS number is contested by this repo's own later analysis ✔

`nsigma_vs_lmax` reports **2.2183σ** at 14000 deg², ℓmax=1020. That is faithfully what
`ps_submean_l37/tables/tension_3param_agg.csv` contains, and it is the published number.

`docs/PLAN_score_bnt_tension_14000.md` lines 160-174 then say, in its own words:

> "**The published grey 2.22 does not reproduce** — it was a high draw of the raw-NPE's large
> run-to-run tension scatter (σ≈0.2–0.3) on top of off-truth nulls (S8≈0.85, not 0.84)."

A controlled re-run of the same configuration gives **1.43 ± 0.30** (per-seed 1.08, 1.39, 1.82),
and concludes "raw-rebin10's 2.22 is the suspect outlier" while the score-compressed extractor is
"binning-independent, on-truth, calibrated, low scatter."

This propagates into the **adopted cut**, which is derived from the same curve. At 14000 deg² the
adopted ℓmax is **460** from the raw r10 campaign, **~500–540** from VMIM-compressed, and **580**
from score-compressed — a spread of 120 in ℓ depending only on the extractor. The paper must pick
one and justify it. Nothing on disk resolves this; it is a scientific decision.

### 1.2 One command can silently destroy the paper's ground truth ✔

`scripts/monitor_tension_pilot.py:43` does `agg.to_csv(tdir / "tension_3param_agg.csv")`. That file
is the **only surviving record of the full 5-seed campaign** — the posteriors behind it are thinned
(only 36 of 108 cells still retain 5 matched pairs). Re-running that monitor would re-aggregate over
the thinned subset and **silently move every adopted ℓmax**, i.e. the paper's headline table.

**Defused:** 44 readable tension tables are archived read-only in
`outputs/baryon_tension/_TABLES_ARCHIVE_precrash_20260730/`. Never write into `*/tables/`; write
re-derivations to a new directory.

### 1.3 A damaged CSV reads as an empty table instead of raising ✔

38 CSVs are the correct size on disk and 100% NUL. `pandas.read_csv` on one returns
`shape=(0, 1), columns=['Unnamed: 0']` **without raising**. An existence-or-size check passes and the
figure panel comes out empty. Worst affected: **all 8 files** in `bnt_ps_bin1_fullsky_l37/tables/`,
and in `ps_submean_l37/tables/` the entire 6-param family plus both 3-param pivots. Any generator
reading a table must assert non-empty and assert expected column names.

### 1.4 The real generators for most `plots/` figures are damaged notebooks — not scripts

No prior audit covered `notebooks/` (478 files). `paper_plots.ipynb`, `paper_plots_dark.ipynb`,
`tomographic_maps_bnt.ipynb`, `compare_master_power_spectra.ipynb` and others write most of the
hand-curated `plots/` figures. All are partially NUL (9–47%) and **none parses as JSON**, so Jupyter
cannot open any of them. `jax_paper_plots.ipynb` is the only intact one.

Because the damage is *partial*, surviving code cells are recoverable by text extraction. This also
explains every "no generator found" gap in the earlier audits — and it means those figures can never
carry sidecars while their notebooks are unopenable.

### 1.5 Recovery routes that were wrongly written off

- **The `_dark` twin family is an intact vector route.** ✔ `nsigma_vs_upper_cut_masks.pdf` is dead
  (and so is its `.bak` — re-transferred together, so the backup bought nothing), but
  `nsigma_vs_upper_cut_masks_dark.pdf` is **intact at 33,052 B against the dead file's 33,049 B**.
  Same for `nsigma_vs_mask_area_all_stats_lmax1024_dark` ✔ and three more. Same-stem sibling tests
  excluded these by construction.
- **"Destroyed" ≠ "unrecoverable."** Three families declared lost have runnable generators and
  readable inputs: `contours_bnt_vs_nonbnt_14000_requiredcut`, `outputs/diagnostics/fullsky_baseline`,
  and the `triangle_*` set. Minutes of work, not losses.

### 1.6 Claims with no figure at all ✔

`outputs/diagnostics/lmin_compare/` **does not exist** — confirmed, and `find` for `withl*` returns
nothing — yet `HANDOFF_lmin_recovery_PS_vs_HOS` is marked DONE and **three headline claims cite it**
(the ×1.9/×3.8 low-ℓ FoM recovery, the HOS-vs-PS FoM ratios, the σ(S8) masked-vs-fullsky ladder).
Separately, `ps_submean_l37/figures/` exists and is **empty** inside a campaign marked DONE.

### 1.7 Smaller traps worth knowing

- **13 contour scripts document the wrong environment** and will crash as documented: they draw
  `filled=True` under `aname` (getdist 1.4.3 + mpl 3.11), which dies on the removed
  `QuadContourSet.tcolors`. Only `plot_contours_three_stats.py` names `jaxili`. Note the envs
  *partition* the generators: `jaxili` has no pandas and no tarp, so the table plotters must stay on
  `aname`. No doc records this split.
- **`n_seeds` means two different things** in the two sidecars held up as reference implementations:
  pre-damage (5 everywhere) in `nsigma_vs_lmax_values.csv`, post-damage (5/5/3/4/7/2) in
  `nsigma_vs_area_fullres_noref_values.csv`.
- **An undocumented hand exclusion** lives inside the reference provenance implementation:
  `MANUAL_EXCLUDE = {('Power spectrum', 28000): {5,6}}`, justified only in a commit message, with no
  sidecar field for it.
- **Three filename collisions with no provenance to disambiguate** —
  `nsigma_vs_upper_cut_with_fullsky.*` is written by three different scripts.
- **`masked_peaks_area_scaling.py:58` uses `allow_pickle=True`** on exactly the array class where
  damage is invisible to a zero-fraction check — the one place the structural test matters most is
  the one place it is disabled.
- **A nonstandard mask tag will silently thin 2000 deg²:** stale L1 splits across `masked_2001sqdeg`
  (3 pairs) and `masked_2002sqdeg` (17 pairs); every generator keys on AREA+1, so a regen drops 46 of
  66 files without warning.
- **The score/compressed family** (`..._score_14000`, `score_contours_debiased_14000`,
  `..._fullsky_optimal`, `..._compressed_14000`) is intact, sidecar-less, and per the docs is the
  most defensible BNT material in the repo.

---

## 2. Figure inventory by state

### READY  (7)

| figure | role | conv | prov | effort | blocked on / note |
|---|---|---|---|---|---|
| PS baryon-bias nsigma vs lmax, 6 masked panels (current) | CORE | CORRECTED | yes | DONE | outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv INTACT (108 rows, n=5, n_excluded=0). I re-derived all 6 adopted cuts (860/580/540 |
| PS baryon-bias nsigma vs lmax, 7 panels with full sky (current) | CORE | CORRECTED | yes | DONE | both ps_submean_l37 and ps_fullsky_l37 3param agg tables intact (fullsky: all 8 table files intact, 24 rows, n=5) |
| BNT bin-1 tension vs lmax, all areas, optimal binning rebin=40 | CORE | CORRECTED | yes | DONE | tables_r40/ intact (n=3). Posteriors at r40: adopted-cut pairs 3/3/2/2/0/3 - 28000 retains ZERO |
| Three-statistic contours PS/peaks/L1 at 14000 (null, biased, both) | CORE | CORRECTED | yes | DONE | PS 4 pairs, peaks 7, L1 4 at full resolution - all present and readable |
| Baryon-bias nsigma vs survey area, full resolution | CORE | CORRECTED | yes | DONE | present; the sidecar records post-damage counts |
| Archival pre-regeneration reference snapshots | DIAGNOSTIC | CORRECTED | **no** | DONE | n/a |
| Fisher constraining-power contours, full-resolution and baryon-safe re | SUPPORTING | FISHER | yes | DONE | ALL 27 input arrays verified reading cleanly from stage3_forecast (16965x1025, 16965x6150, 16965x5x40, 200x...). Recovered exactly after the crash: re |

### NEEDS_REGEN  (24)

| figure | role | conv | prov | effort | blocked on / note |
|---|---|---|---|---|---|
| Constraining power (sigma) vs survey area, 3 statistics | CORE | CORRECTED | **no** | MINUTES | only the sidecars - add the _values.csv/_provenance.json block already implemented in plot_nsigma_vs_area.py |
| Score/MOPED-compressed BNT bin-1 tension vs lmax at 14000 | CORE | CORRECTED | **no** | MINUTES | sidecars only |
| BNT Fisher error-budget ratio ladder at 14000 | CORE | FISHER | **no** | MINUTES | nothing technically - it will render today. An evidence-backed version needs the pymaster env built and the whole Gaussian covariance chain rebuilt. |
| HOS w0 degeneracy mechanism, publication figure | CORE | CORRECTED | **no** | MINUTES | nothing - rerun to restore the vector version and add sidecars |
| BNT vs non-BNT contour overlays at the optimal rebin | DIAGNOSTIC | CORRECTED | **no** | MINUTES | Clean env deadlock: it draws triangle_plot(filled=True) which needs getdist 1.6.1 (jaxili), but does `from tension import configs, estimators as E` an |
| All-mask and per-mask NPE posterior triangles | DIAGNOSTIC | CORRECTED | **no** | MINUTES | nothing on the data side; must run under jaxili (filled=True), not the aname the docstring implies |
| Full-sky healpy control and l100paper-vs-new triangles | DIAGNOSTIC | STALE | **no** | MINUTES | nothing on the data side; filled=True means jaxili, not aname |
| NPE bias contours and bias vs area | DIAGNOSTIC | CORRECTED | **no** | MINUTES | sidecars |
| TARP coverage plots | DIAGNOSTIC | MIXED | **no** | MINUTES | two stems (masked_14000sqdeg null and baryonified) are dead in BOTH pdf and png - those need their npz or a retrain |
| Per-run NPE posterior corner plots (flat outputs/plots namespace) | DIAGNOSTIC | MIXED | **no** | GPU_RERUN | n/a - these are training by-products |
| Masked peaks area scaling (Fisher) | DIAGNOSTIC | CORRECTED | **no** | HOURS_CPU | sidecars |
| VMIM v2 side-study figures (cut480, native480, phase p0-p2g) | DIAGNOSTIC | CORRECTED | **no** | MINUTES | confirm the p2f/p2g arm filenames |
| Stray top-level and out-of-scope figures | DIAGNOSTIC | MIXED | **no** | MINUTES | mixed; score_experiment/npe_whiten is NULL-ONLY (all 21 posteriors are nobaryons_vs_nobaryons) |
| BNT bin-1 tension vs lmax, full sky, rebin=60 | SUPPORTING | CORRECTED | **no** | MINUTES | needs sidecars plus either re-derivation or an explicit 'quoted historical result' label |
| FoM3 vs survey area, 3 statistics | SUPPORTING | CORRECTED | **no** | MINUTES | sidecars only |
| BNT vs non-BNT contours at the required cuts, 14000 | SUPPORTING | CORRECTED | **no** | MINUTES | nothing on the data side; must be run under jaxili, NOT the aname its docstring specifies |
| BNT whitening comparison contours | SUPPORTING | CORRECTED | **no** | MINUTES | sidecars |
| VMIM-compressed baryon tension vs lmax at 14000 | SUPPORTING | CORRECTED | **no** | MINUTES | sidecars |
| VMIM raw vs compressed posteriors | SUPPORTING | MIXED | **no** | MINUTES | confirm the p2f arm filenames resolve |
| Fisher FoM bar charts (standalone and combined probes) | SUPPORTING | FISHER | **no** | MINUTES | sidecars - a half-instrumented generator, fix once and all four figures comply |
| Gaussian covariance validation at 14000 | SUPPORTING | FISHER | **no** | HOURS_CPU | (1) no pymaster/cosmostat env exists on Jean Zay - only aname and jaxili are built; build with scripts/jz/build_envs.sh namaster pinning namaster 2.5. |
| Fisher gate contour and ladder replots | SUPPORTING | FISHER | **no** | MINUTES | sidecars only for the replots |
| L1 submean vs non-submean overlays, per footprint | SUPPORTING | CORRECTED | **no** | HOURS_CPU | identify or write the generator |
| Coarse-scale drop justification (L1 coarse contours, coarse_contours_s | SUPPORTING | CORRECTED | **no** | HOURS_CPU | generator identification; sidecars |

### NEEDS_NEW_SCRIPT  (10)

| figure | role | conv | prov | effort | blocked on / note |
|---|---|---|---|---|---|
| Starlet wavelet scale to multipole mapping | CORE | CORRECTED | **no** | MINUTES | the full generator needs the unbuilt cosmostat env, but a small new replot script reading the surviving starlet_transfer_data.npz can restore the figu |
| BNT Cl fractional-difference figures (bin localisation) | CORE | CORRECTED | **no** | HOURS_CPU | generator notebooks unopenable; needs a new script |
| Score vs non-BNT contours at the de-biasing cuts, 14000 | SUPPORTING | CORRECTED | **no** | HOURS_CPU | scripts/score_cut_utils.py must be REWRITTEN from docs/PLAN_vmim_scalecuts.md (build_score()/keep_indices()). The Fisher-floor ratio it returns is pri |
| Non-BNT score-vs-whiten control (score is a no-op on the well-conditio | SUPPORTING | CORRECTED | **no** | HOURS_CPU | no generator exists; the figure must be rebuilt from a new script |
| HOS w0 mechanism and origin diagnostics (fixedbin_l1) | SUPPORTING | CORRECTED | **no** | HOURS_CPU | regenerating needs scripts/diagnostics/fixedbin_l1_full.py, which imports pycs.astro.wl.hos_peaks_l1 + h5py + healpy -> the cosmostat_new/NaMaster env |
| PS vs HOS parameter degeneracy directions | SUPPORTING | CORRECTED | **no** | HOURS_CPU | no generator; a new script is needed to regenerate or sidecar it |
| Monopole / mean-leakage appendix (A2-A5) | SUPPORTING | CORRECTED | **no** | HOURS_CPU | no generator |
| l37 vs l100 posterior comparison per footprint (lmin-recovery evidence | SUPPORTING | CORRECTED | **no** | HOURS_CPU | no generator |
| Fisher vs NPE contours at 14000 (score experiment) | SUPPORTING | CORRECTED | **no** | GPU_RERUN | the destroyed posterior_summary must be regenerated (GPU) |
| Survey n(z) tomographic bins | SUPPORTING | UNKNOWN | **no** | MINUTES | nothing serious - jax_paper_plots.ipynb survives and is the recovery route |

### BLOCKED_ON_COMPUTE  (9)

| figure | role | conv | prov | effort | blocked on / note |
|---|---|---|---|---|---|
| Three-statistic contours, BARYON-SAFE variant | CORE | CORRECTED | **no** | GPU_RERUN | one L1 scales234-submean NPE run plus 3 peaks scales234-submean re-runs. scripts/jz/npe_hos_baryonsafe_14000.slurm exists (A100, ~20 trainings, 20 h). |
| Fisher BNT vs non-BNT contours at 14000 | DIAGNOSTIC | FISHER | **no** | HOURS_CPU | same pymaster env + covariance rebuild as gaussian_cov_validation |
| VMIM calibration diagnostics (TARP ECP curves, SBC rank histograms) | SUPPORTING | CORRECTED | **no** | GPU_RERUN | score_cut_utils.py rewrite, plus the `tarp` package which is NOT installed in jaxili (it imports only via the vendored repo copy at PYTHONPATH=<repo>/ |
| Moment responses and even/odd variance decomposition | SUPPORTING | CORRECTED | **no** | HOURS_CPU | cosmostat_new/pycs env must be BUILT, then a 40-worker starlet job to rebuild fixedbin_l1_full.npz |
| Per-footprint peaks tension, scales234 (baryon-safe higher-order cut) | SUPPORTING | CORRECTED | **no** | GPU_RERUN | 3 peaks scales234-submean NPE re-runs (A100). scripts/jz/npe_hos_baryonsafe_14000.slurm covers 14000. |
| Score adopted-cut six-footprint NPE contours and montage | SUPPORTING | CORRECTED | **no** | GPU_RERUN | those NPE-on-score runs must be redone (GPU), or the loaders need a guard plus an honest n-per-point |
| PS old-vs-new check and nlb condition number | SUPPORTING | CORRECTED | **no** | HOURS_CPU | build the namaster env (pin 2.5.2), then regenerate datavectors.npz |
| l37 vs l100 tension curve comparison (planned, never built) | SUPPORTING | CORRECTED | **no** | GPU_RERUN | a full l100-submean tension campaign (GPU NPE), then a new plotting script |
| Three-statistic baryon contours, BNT variant | UNCLEAR | STALE | **no** | GPU_RERUN | BNT peaks + L1 recomputed on submean maps (GPU NPE). Additionally requires the noise->mask->BNT ordering (--mask-correction), which HANDOFF_masked_pea |

### LOST_UNRECOVERABLE  (5)

| figure | role | conv | prov | effort | blocked on / note |
|---|---|---|---|---|---|
| Score vs raw NPE contours at the de-biasing cuts, 14000 | SUPPORTING | CORRECTED | **no** | HOURS_CPU | both the plotting script AND score_cut_utils.py must be rewritten from docs/PLAN_score_bnt_tension_14000.md |
| Score vs whitening contours at 14000 | SUPPORTING | CORRECTED | **no** | HOURS_CPU | no generator and no surviving image |
| MASTER low-ell ratio and variance validation | SUPPORTING | CORRECTED | **no** | IMPOSSIBLE | figure destroyed AND generator notebook destroyed AND the env its rebuild needs does not exist |
| Tomographic map panels (noiseless, noisy, BNT-transformed) | SUPPORTING | UNKNOWN | **no** | IMPOSSIBLE | the generator notebook is partially destroyed; only manual text-extraction of surviving code cells could recover the plotting logic |
| lmin-recovery comparison triangles (withl0 / withl2 family) | SUPPORTING | MIXED | **no** | GPU_RERUN | retraining the PS lmin=2 (and lmin=0) NPE runs, GPU |

### SUPERSEDED  (8)

| figure | role | conv | prov | effort | blocked on / note |
|---|---|---|---|---|---|
| PS nsigma-vs-cut, legacy hand-curated version | SUPERSEDED | CORRECTED | **no** | DONE | campaign table intact |
| PS nsigma-vs-cut with full sky, legacy version | SUPERSEDED | CORRECTED | **no** | DONE | tables intact |
| BNT bin-1 tension vs lmax, all areas, rebin=10 | SUPERSEDED | CORRECTED | yes | DONE | bnt_ps_bin1_submean_l37/tables/ intact (n=5); posteriors thinned - at the non-BNT adopted cuts only 5/4/2/1/3/5 pairs survive, and 14000 (the referenc |
| Three-statistic baryon contours, pre-crash set (unbiased/biased) | SUPERSEDED | STALE | **no** | DONE | pre-correction posteriors (l100, non-submean) survive in bulk |
| NPE vs Fisher constraining-power contours | SUPERSEDED | STALE | **no** | GPU_RERUN | the destroyed full-sky l100 PS posterior would need retraining |
| Baryonified parameter triangles (peaks/L1/BNT variants) | SUPERSEDED | STALE | **no** | DONE | deep pre-correction posterior sets survive (stale L1 scales234: ~22-23 runs/footprint; stale peaks: ~14-15) |
| nsigma vs mask area at lmax1024 (pre-correction) and the FAKE placehol | SUPERSEDED | STALE | **no** | DONE | pre-correction posteriors |
| Per-scale fractional difference (L1 and peak counts) | SUPPORTING | STALE | **no** | DONE | pre-correction (non-submean) HOS grids |
---

## 3. Results the paper could still have

Two independent lenses were run: data-driven (start from the inventory) and adversarial-referee.
They converged on one thing above all — **every existing figure shows bias *removal*, none shows the
information *sacrificed* to achieve it.**

### 3.1 Available now, data verified present (minutes to hours, no GPU)

| priority | result | question it answers |
|---|---|---|
| HIGH | **Cost of baryon safety**: σ and FoM₃ vs ℓmax, and FoM vs area at each footprint's *own* adopted cut | What does baryon safety cost, and does constraining power still improve with area once you may only use safe scales? |
| HIGH | **Null-on-truth accuracy** vs ℓmax per footprint | Does the nobaryons null actually sit on truth? If the null is itself biased, a difference-of-means tension compares two biased estimates. |
| HIGH | **Re-derive the tension table from surviving posteriors**, and restore the destroyed 6-param family | Are the adopted cuts stable under the post-damage subset, and does the cut depend on a 3- vs 6-parameter tolerance? |
| HIGH | **Per-parameter bias decomposition**, replacing the 3-param scalar | *Which* parameter does feedback bias, in what direction, by how much? |
| HIGH | **Step-20 refined curve at 14000** (the fine grid exists only there) | How much headroom does the adopted cut actually have? |
| MEDIUM | **Tolerance sensitivity**: the adopted ladder at 0.2σ / 0.3σ / 0.5σ | Would the headline table survive a referee who prefers a different tolerance — and where does 0.3 come from? |
| MEDIUM | **ℓmax_safe(A) scaling law** | Can a future survey read the cut off a formula instead of a table? |
| MEDIUM | **PS vs HOS degeneracy directions** in the corrected convention on real footprints | Is the opposite-sign Ωm–w₀ degeneracy real off the full-sky Fisher forecast? |

### 3.2 Needs compute

| priority | result | blocker |
|---|---|---|
| HIGH | **TARP/SBC calibration** of the corrected posteriors | No flow checkpoints survive for any corrected config (0 of 45 checkpoint dirs match submean/l37) → retrain. **Cannot be faked from the saved sample files** — they are 3000×6 draws at one fixed observation and cannot produce an ECP curve. |
| HIGH | **Baryon-model dependence** | Everything rests on ONE baryonification prescription. Needs new sims. If true suppression is 2× this one, every cut in the table is too permissive — and no sentence currently protects that reading. |
| MEDIUM | **Joint PS + HOS posterior** at the reference footprint | No joint posterior exists; training inputs do. Must share one noise realisation and mask or the gain is overstated by construction. |
| MEDIUM | **Non-Gaussian tension estimator** vs Gaussian Q_DM | `estimators.py` already has the dispatch hook. Caveat: non-Gaussian shift estimators need ≫3000 samples in 3D and return overconfident numbers when under-sampled. |
| MEDIUM | **Fisher-vs-NPE reconciliation** (Fisher says BNT buys 0.45–0.48 in area, NPE says 0.79) | Currently asserted, not measured. Analytic-Gaussian legs blocked: all 6 `gaussian_cov_native_*.npy` stripe-damaged, all 12 NaMaster workspaces damaged, pymaster env not built. |
| LOW | **Mask/apodization robustness** | Needs the pymaster env at pinned namaster 2.5.2; workspaces are version-specific. |

### 3.3 Two calibration cautions

- The score campaign's calibration is graded against **0.15**, while the **0.05/0.10** rule was the
  one used to gate VMIM. Adopting the looser threshold silently because it passes would be a real
  problem; if the strict rule is the standard, the score figure needs the deep-ensemble treatment.
- The quoted nσ has **no distribution over data realisations** (the observation is the 200-perm
  mean), so 0.3σ is a deterministic bias measure, not a detection significance. Bootstrapping perms
  against a fixed flow would produce a scatter too small by construction.

---

## 4. Provenance status

**7 of 63** figure concepts have sidecars. **48 of 53** generators never write them, and the
notebook-generated figures cannot until the notebooks are text-recovered.

The 7 compliant figures are also **not one reproducible build**: sidecar `git_commit` values are
`b91af33`, `701fba8`, `8ab6450`, `def9087` and — for the four Fisher figures — literally `unknown`,
spanning numpy 1.26.4 (jaxili) and 2.4.6 (aname). The Fisher sidecars also lack `mplstyle` and their
CSVs lack `n_seeds`.
