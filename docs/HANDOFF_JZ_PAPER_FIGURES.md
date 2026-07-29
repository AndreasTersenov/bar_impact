# HANDOFF — regenerating the paper figures on Jean Zay

**For:** a fresh session continuing the bar_impact paper corrections.
**From:** the JZ recovery session, 2026-07-29. Recovery itself is finished; this is
about rebuilding the figures the disk failure took.

Read `BAR_IMPACT_RECOVERY_STATUS.md` in `../recovery_handoff/` for how the data got
here. This file is what you need to *work*.

---

## 0. The rule that must not be dropped

**Every figure ships provenance sidecars.** `plot_nsigma_vs_area.py` now writes
`<figure>_values.csv` and `<figure>_provenance.json` next to the PDF/PNG. Replicate
that in every generator you touch — it is not optional bookkeeping.

Why: the pre-crash figures recorded no numbers, only pixels. When the regenerated
`nsigma_vs_area` differed slightly from the original we could only *reason* about the
cause, never measure it. Worse, reading values off a log tail led to a whole table
being mislabelled (peak-counts values attributed to the power spectrum) and only the
CSV caught it.

The sidecars must carry: per-point values, **`n_seeds`**, error-bar kind, git commit,
package versions, which mplstyle, and the standing caveats. `n_seeds` is the critical
column — disk damage means each point averages a *different subset* of runs than
pre-crash, and that is the dominant source of numerical drift.

Copy the block at the end of `scripts/diagnostics/plot_nsigma_vs_area.py`.

## 1. Environments — three, never mixed

| purpose | interpreter |
|---|---|
| NPE / jaxili / **getdist contours** | `/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python` |
| tension stats (tensiometer, getdist **1.4.3**) | `/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python` |
| power spectra (pymaster) | **NOT BUILT YET** — `bash scripts/jz/build_envs.sh namaster` |

The getdist version differs deliberately between jaxili (1.6.1) and aname (1.4.3);
recovered memory `bar-impact-tension-env` calls this out. Do not unify them.

**Which env for a contour plot: jaxili, not aname.** getdist 1.4.3 calls
`QuadContourSet.tcolors`, removed in matplotlib 3.8, so any *filled* contour under aname
dies with `AttributeError: 'QuadContourSet' object has no attribute 'tcolors'`. Line
contours are unaffected, which is why some getdist scripts run fine there. Anything that
only *plots* posteriors needs no tensiometer — use jaxili.

**Envs live under `nzu`, never `prk`.** prk's `$WORK` is at ~94% of its 500k *inode*
quota; a conda env is ~68k files. The failure presents as "Disk quota exceeded" while
`df` shows 1.5 TB free — it is inodes, not bytes. Also always use `matplotlib-base`,
not `matplotlib`, which drags in Qt6 (~1 GB, thousands of files) for GUI backends
nothing uses.

## 2. Data

`--data-dir` / `DD` → `/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast`
containing `new_grid/`, `fiducial/` (flat aggregates **plus** a real `cosmo_fiducial/`
subdir), and `grid/cosmo_params.npy`. All ~93 hardcoded Titan paths are already
repointed. Verified: 180/180 masked inputs + full-sky resolve.

**Never symlink `cosmo_fiducial` to its parent.** It is a distinct dataset (863
aggregates + 200 `perm_*` dirs), and a self-referential link loops under any
`-L`/`followlinks` walk.

## 3. Figure status

| figure | state |
|---|---|
| scaling vs area, 3 stats (`scaling_vs_area_all_stats`) | ✅ regenerated; slopes −0.46/−0.39/−0.42 reproduce `PAPER_NOTES §1` to 0.01; FoM error bars added |
| FoM vs area, single panel (`fom_vs_area_all_stats`) | ✅ new, for the referee. `_lowanchor` variant preserved |
| nσ vs area (`nsigma_vs_area_fullres_noref`) | ✅ regenerated, provenance sidecars, √A line dropped |
| PS bias vs ℓmax (`nsigma_vs_lmax`) | ✅ new `scripts/diagnostics/plot_nsigma_vs_lmax.py`; reproduces the surviving PNG exactly. `_with_fullsky` 7-panel variant too. Emits `_crossings.csv` |
| BNT bin-1 tension vs cut | ✅ regenerated both variants via `build_bnt_bin1_allareas_plot.py` (see §4 for the three bugs found in it) |
| **3-statistic baryon-safe contours** | 🟡 `scripts/diagnostics/plot_contours_three_stats.py` written — null/biased for PS, peaks, L1. **The `_BNT` variant is blocked, see §6** |

Always preserve the existing figure first — see `_reference_pre_regen/`.

## 4. Traps that cost real time here

- **Corrupt `.npy` raises `ValueError`, not an IO error.** numpy reads the mangled
  header as pickled data. Guards written pre-crash catch only
  `(FileNotFoundError, IndexError)` and let it through. Do **not** "fix" it with
  `allow_pickle=True` — that unpickles garbage. Skip and report the skip.
- **`aa.mplstyle` is a RECONSTRUCTION** at `styles/aa.mplstyle`. The original
  (`~/.claude/skills/figure-polish/`) died with the disk. Cosmetic differences from
  pre-crash figures are expected; data is unaffected.
- **Zero-fraction is not an integrity test for large binaries.** A 117 MB sparse
  covariance holds ~20% legitimate zeros; a lost RAID stripe removes ~17%. Only the
  3 MiB-periodic / 512 KiB-aligned signature separates them. This hid 19 damaged
  NaMaster files from three separate audits.
- **`pgrep -f` / `pkill -f` match your own command line.** Cost two separate detours
  (a killed shell mid-heredoc, and a "job is running" false positive that silently
  skipped a launch). See recovered memory `feedback_no_pkill_self_match`.
- **Broad `except` hides API breaks.** `q_dm_tension` returns NaN on any exception, so
  tensiometer relocating `from_confidence_to_sigma` surfaced as `nσ=nan` on every row
  rather than a failure. Fixed, but the pattern recurs in this codebase.

## 5. Science note — the √A line

`nsigma_vs_area` used to draw `∝√A`. That is right for a *single-parameter*
significance (fixed bias / σ ~ A^−1/2). This figure plots 3-parameter Gaussian Q_DM
mapped through χ²(3): **Q_DM ~ A**, not √A, and the χ²→nσ map is regime-dependent —
with Q ∝ A exactly, the measured slope is ~+1.2 near nσ≈0.5, ~+1.0 near nσ≈1.4, and
tends to +0.5 only asymptotically. `REFLINE=0` drops it; the kept version is
relabelled "(asymptotic)". Decide deliberately which goes in the paper.

**Thin points to watch:** PS at 35000 rests on **2 seeds**, at 10000 on **3**. Their
error bars are barely estimates. Check `_values.csv` before quoting them.

## 6. The blocked `_BNT` contour variant — needs a decision, not more work

`plots/contours_PS_peaks_L1_baryons_BNT.pdf` is 39% zeros. It cannot simply be
regenerated, because the BNT posteriors that survive for **peaks and L1** are in the
*pre-correction* convention:

```
posterior_samples_bnt_pc_..._bntbins1234_scales1234_noisy_s0.26_masked_14001sqdeg_new_normalization_npe.npy
                                                                              ^^^ no `submean`
```

They carry no `submean` token, and the BNT PS products alongside them are `l100-…`, i.e.
ℓmin = 100. Both are exactly what the last six months of work *corrected*: ℓmin 100→37 and
the submean subtraction. A BNT PS at ℓ≥37+submean (`bnt_ps_bin1_submean_l37/posteriors/`,
which does exist and is intact) overlaid on ℓ≥100 non-submean peaks and L1 would be three
statistics under two different analyses in one triangle.

So the options are:
1. **Drop the 3-statistic BNT contour** and let the BNT story rest on
   `nsigma_vs_lmax_bnt_bin1_allareas_optimal`, which is PS-only but current and honest.
2. **Rerun BNT peaks + L1** at ℓ≥37/submean. That is NPE training, not plotting — GPU
   work, and the only route that makes the figure defensible.
3. Ship it as an explicitly pre-correction figure. Not recommended:
   `PLAN_bnt_optimal_binning.md` shows the old BNT numbers materially overstated BNT's
   baryon mitigation.

`scripts/diagnostics/plot_contours_three_stats.py` builds the null/biased figure, which
has no such problem — all three statistics exist there in the corrected convention.

## 7. Still outstanding beyond figures

- **Covariance rebuild** — 19 NaMaster products destroyed, unrecoverable (damaged
  identically at source). Needs the pymaster env, then per-area
  `FISHER_AREA=$A .../cosmostat_new/bin/python scripts/diagnostics/fisher_gaussian_cov.py`
  as a `cpu_p1` job. Let it regenerate `w_*.fits`/`cw_*.fits`; the damaged ones must
  not be reused and workspaces are version-specific. Pin **namaster 2.5.2** (memory
  `bar-impact-namaster-venv`), and expect small numerical differences — the on-disk
  MASTER products came from an even older build.
- **Score caches: 48 regenerated and verified** (`bnt_full` reproduces the documented
  pre-crash shape `x[16965,120]` exactly). 4 stale `*_r10_*` caches remain corrupt —
  rebin=10, outside the validated rebin=20 path; delete or ignore.
- **`score_cut_utils.py` is corrupt and untracked** — confirmed unrecoverable. Needed
  for the VMIM per-cut sweep (`keep_indices`, `build_score`). Must be rewritten from
  `PLAN_vmim_scalecuts.md` before that work resumes. `run_vmim_pilot.py` is simply
  absent.
- **Permanently lost:** 34 notebooks, 533 `vmim_v2` intermediates,
  `jolly-toasting-robin.md`. Campaign *results* survived.
- **`$SCRATCH` purges after 30 idle days.** Memories are archived to `$WORK` and
  `$STORE`; the 83 GB science archive to `$STORE` was still running at handoff —
  check `du -sh /lustre/fsstor/projects/rech/prk/ulx34io/titan_recovery`.
- **git:** HEAD `def9087`, in sync with GitHub. ~235 uncommitted changes from this
  session (path repoints, the new style, provenance code, new scripts) — review and
  commit.
