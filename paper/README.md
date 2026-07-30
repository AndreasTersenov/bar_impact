# `paper/` — the clean room

Everything in here is **current, provenanced, and safe to cite.** Nothing else in the repo is.

For contrast: `outputs/` and `plots/` together hold **9,550 figure files** across 32
directories, **1,907 of them destroyed**, spanning three different analysis conventions, with
three separate cases of one filename being written by three different scripts. That tree is a
working area and a graveyard at once. `docs/PAPER_FIGURE_MAP.md` is the survey of it; this
directory is what survived the survey.

## Layout

    paper/figures/MANIFEST.md          the index — start here
    paper/figures/manifest.json        the same, machine-readable
    paper/figures/<slug>/
        figure.pdf                     vector, for LaTeX
        figure.png                     raster preview
        values.csv                     the plotted numbers
        provenance.json                how they were made
        meta.json                      published from where/when, sha256 per file, gate result
        README.md                      what it shows and its caveats, in prose
        (crossings.csv, covariance.csv where the generator emits them)

**One directory per figure**, with fixed filenames inside. A figure therefore cannot be
separated from its provenance by a copy, a move, or a tarball, and "is this figure complete?"
is answered by listing one directory. In LaTeX:

```latex
\includegraphics{paper/figures/ps_bias_vs_lmax/figure.pdf}
```

Slugs, not numbers, name the directories — figure order churns while a paper is being written,
and renumbering directories would break every `\includegraphics` and every git history link.
The intended position lives in `meta.json` as `paper_position` and is what `MANIFEST.md` sorts
by, so reordering is a one-field edit.

## These are copies, not symlinks

Deliberate, and the reasons are specific:

- `$SCRATCH` **purges after 30 idle days**, and symlink targets under `outputs/` would go with it.
- Symlinks do not survive archiving. This project has already been bitten by a self-referential
  link (`ln -s . cosmo_fiducial`) that looped under any `-L` walk.
- A symlink would let a regeneration in `outputs/` **silently change a figure the paper cites.**

The cost is duplication. `verify` is how that cost is paid back: it re-checks every published
file against the sha256 recorded at publish time, and against the source, so both
edited-in-place and source-drifted are detected instead of assumed away.

## The gate

`scripts/paper/figures.py publish` **refuses** a figure that is missing a valid vector PDF, a
non-empty `values.csv`, or a parseable `provenance.json`. This is the whole point. The
provenance rule has existed for some time and **7 of 63** figures followed it, because a rule
enforced by discipline is not enforced.

The empty-`values.csv` check is not paranoia: **38 CSVs in this repo are the correct size on
disk and 100% NUL**, and `pandas.read_csv` returns `shape=(0,1)` on them *without raising*. An
existence check passes and the figure panel comes out blank. Row count is checked, not presence.

Warnings (a missing `mplstyle`, an `unknown` git commit, no seed-count column) do **not** block
publication — the figure is still traceable — but they are recorded in `meta.json` and surfaced
in the `warns` column of `MANIFEST.md`, so a known-imperfect figure stays visible instead of
being quietly forgotten.

## Workflow

```bash
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python

$PY scripts/paper/figures.py publish outputs/plots/<dir>/<stem> \
      --slug my_figure --position 12 --title "..." --note "..."
$PY scripts/paper/figures.py manifest     # rebuild the index
$PY scripts/paper/figures.py verify       # re-gate everything, detect drift
$PY scripts/paper/figures.py list         # short status table
```

Run `verify` before every submission. **Never edit a file in `paper/figures/` in place** —
regenerate at the source and re-publish, so a figure and its recorded numbers can never
disagree.

## Not in here yet, and why

| what | why not | to fix |
|---|---|---|
| BNT bin-1 at rebin=10 | **SUPERSEDED.** `PLAN_bnt_optimal_binning.md` records that it substantially overstated BNT's mitigation — the low BNT tension was NPE under-extraction inflating the contours and hiding the bias. The rebin=40 version is published in its place. | nothing; it should stay out |
| 3-statistic contours at the baryon-safe cut | Blocked on compute. L1 at `scales234` was never rerun on zero-mean maps (305 exist, **zero** submean); peaks has one seed and its biased side at 14000 is destroyed. | GPU job `npe_hos_baryonsafe_14000.slurm` |
| Cost of baryon safety (σ, FoM at each adopted cut) | Not built yet. Inputs verified present. Highest-value gap: every published figure shows bias *removal*, none shows the information *sacrificed*. | minutes of scripting |
| TARP/SBC calibration | No flow checkpoints survive for any corrected config. **Cannot** be computed from the saved sample files — they are 3000×6 draws at one fixed observation and cannot produce an ECP curve. | GPU retrain |
| Everything generated by `notebooks/` | The notebooks are 9–47% NUL and **none parses as JSON**, so Jupyter cannot open them. Damage is partial, so code cells are text-recoverable. | extract cells to scripts, then add sidecars |

## One thing to know before citing figure 1

The published PS bias at 14000 deg² is **2.2183σ**, faithfully what the campaign table holds.
`docs/PLAN_score_bnt_tension_14000.md` then records that this number **"does not reproduce"** —
a controlled re-run gives **1.43 ± 0.30** and attributes 2.22 to a high draw of raw-NPE scatter
on off-truth nulls. The adopted cut comes off the same curve, and moves with the extractor: 460
(raw r10), ~500–540 (VMIM), 580 (score). This is unresolved and is a scientific decision, not a
bug. See `docs/PAPER_FIGURE_MAP.md` §1.1.
