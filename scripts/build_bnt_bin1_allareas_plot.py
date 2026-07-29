#!/usr/bin/env python3
"""Multi-panel BNT bin-1 vs non-BNT-cut-all baryon-tension figure (masked footprints).

One panel per footprint, overlaying:
  - non-BNT — cut all bins   (grey squares; from ps_submean_l37 — the standard analysis)
  - BNT — cut bin-1 only      (blue circles; from bnt_ps_bin1_submean_l37 — bins 2-4 full)

Both curves are the 3-param (Ωm,S₈,w₀) Gaussian Q_DM tension, mean±std over the 5 NPE
training seeds, on the same monopole-subtracted ℓ≥37, step-40 cut grid. This is the
single-panel monitor_bnt_bin1.py figure generalized to every footprint with data.

Reads the aggregated tables written by compute_tension.py (so it does NOT need tensiometer),
but still runs fine under the aname env (matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/build_bnt_bin1_allareas_plot.py
"""
import argparse
import os
import sys
from math import ceil

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
BNT_AGG = f"{REPO}/outputs/baryon_tension/bnt_ps_bin1_submean_l37/tables/tension_3param_agg.csv"
NONBNT_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_allareas.png"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_allareas.pdf"
ALL_AREAS = (2000, 5000, 10000, 14000, 28000, 35000)
THRESHOLD = 0.3
# Per-area "% extracted at no-cut" = BNT/non-BNT tension at ℓmax=1020 with BOTH SIDES AT
# rebin=40 (lossless target is 100%). Annotated per panel on the --rebin-note variant.
#
# NOT REPRODUCIBLE FROM THIS REPO. The non-BNT-at-r40 tension it divides by was never
# written as a table (the shipped non-BNT campaign is r10, its own optimum), so dividing
# the r40 BNT table by ps_submean_l37 gives 57/65/65/76/99/99 — a different quantity, the
# best-vs-best comparison, not the same-binning lossless check. These values are carried
# over verbatim from docs/PLAN_bnt_optimal_binning.md "FINAL RESULTS (2026-06-25)", which
# is their only surviving record. Recorded as such in the provenance sidecar.
EXTRACTED_R40 = {2000: 87, 5000: 85, 10000: 82, 14000: 76, 28000: 93, 35000: 93}
EXTRACTED_SOURCE = ("docs/PLAN_bnt_optimal_binning.md, FINAL RESULTS (2026-06-25) — "
                    "BNT/non-BNT tension at lmax=1020 with both at rebin=40")


def _apply_rcparams():
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 14


def _crossing(sub):
    """Lowest upper_cut whose mean nσ >= THRESHOLD, or None."""
    hit = sub[sub["mean"] >= THRESHOLD]["upper_cut"]
    return int(hit.min()) if len(hit) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--areas", type=int, nargs="*", default=list(ALL_AREAS))
    p.add_argument("--bnt-agg", default=BNT_AGG, help="BNT tension table (e.g. the tables_r40 one).")
    p.add_argument("--rebin-note", default=None, help="Annotate panels with rebin* + %% extracted.")
    p.add_argument("--out-png", default=OUT_PNG)
    p.add_argument("--out-pdf", default=OUT_PDF)
    args = p.parse_args()

    bnt = pd.read_csv(args.bnt_agg)
    nonbnt = pd.read_csv(NONBNT_AGG) if os.path.exists(NONBNT_AGG) else pd.DataFrame()

    areas = [a for a in args.areas if (bnt["area"] == a).any()]
    if not areas:
        print(f"No BNT bin-1 data in {BNT_AGG} for areas {args.areas}. Nothing to plot.")
        return
    n = len(areas)

    _apply_rcparams()
    ncols = min(n, 3)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), sharex=True)
    axes = list(axes.flatten()) if n > 1 else [axes]

    for ax, area in zip(axes, areas):
        if len(nonbnt):
            nb = nonbnt[nonbnt["area"] == area].sort_values("upper_cut")
            if len(nb):
                ax.errorbar(nb["upper_cut"], nb["mean"], yerr=nb["std"].fillna(0),
                            fmt="s", color="0.55", ms=5, elinewidth=1.2, capsize=3,
                            label="non-BNT — cut all bins")
        b = bnt[bnt["area"] == area].sort_values("upper_cut")
        ax.errorbar(b["upper_cut"], b["mean"], yerr=b["std"].fillna(0),
                    fmt="o", color="C0", ms=6, elinewidth=1.5, capsize=4,
                    label="BNT — cut bin-1 only (bins 2-4 full)")
        ax.axhline(THRESHOLD, color="r", linestyle="--", linewidth=1.4,
                   label=f"Threshold ({THRESHOLD})")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_title(rf"Area = {area} deg$^2$")
        ax.set_ylim(bottom=0)
        if args.rebin_note:
            ext = EXTRACTED_R40.get(area)
            note = f"{args.rebin_note}" + (f"\n{ext}% extracted @ no-cut" if ext else "")
            ax.text(0.04, 0.96, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color="0.35",
                    bbox=dict(boxstyle="round", fc="white", ec="0.85", alpha=0.8))
    for ax in axes[n:]:               # hide any unused grid cells
        ax.set_visible(False)

    # The legend lives outside the axes. In-panel it had nowhere safe to go: the rebin
    # note occupies every panel's upper-left (the shipped _optimal figure has the two
    # overlapping), and the curves rise left-to-right so the lower-right is data too.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=len(labels), fontsize=11, frameon=False)
    fig.supxlabel(r"Upper Cut ($\ell_{\mathrm{max}}$)", fontsize=15, y=0.03)
    fig.supylabel(r"Significance ($n_\sigma$)", fontsize=15, x=0.05)
    title = ("OPTIMAL-BNT (rebin 40) vs standard non-BNT — baryon tension vs scale cut"
             if args.rebin_note else
             "BNT bin-1-only vs non-BNT cut-all — baryon tension vs scale cut")
    fig.suptitle(title, fontsize=16)
    # Read the run count off the table rather than hardcoding it per variant: the two
    # tables genuinely differ (tables/ = 5 runs, tables_r40/ = 3, the overnight budget),
    # and a subtitle that states the wrong n is the mislabelling failure this project
    # has already been bitten by once.
    n_runs = sorted(bnt["n"].unique())
    n_txt = "/".join(str(int(v)) for v in n_runs)
    sub = (f"monopole-subtracted PS, ℓ≥37, step-40 | 3-param Q_DM, mean±std / {n_txt} runs | "
           "BNT at its optimal binning (no-cut % = how close to the lossless BNT=non-BNT identity)"
           if args.rebin_note else
           f"monopole-subtracted PS, ℓ≥37, step-40 | 3-param Q_DM, mean±std / {n_txt} runs")
    fig.text(0.5, 0.955, sub, ha="center", va="top", fontsize=9.5, color="0.4")
    plt.tight_layout(rect=(0.05, 0.04, 1, 0.90))   # room for suptitle + subtitle + legend

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(args.out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"rendered {n} panels -> {args.out_png}")
    crossings = []
    for area in areas:
        cb = _crossing(bnt[bnt["area"] == area].sort_values("upper_cut"))
        cg = (_crossing(nonbnt[nonbnt["area"] == area].sort_values("upper_cut"))
              if len(nonbnt) else None)
        crossings.append((area, cb, cg))
        print(f"  {area:>6} deg²: 0.3σ crossing — BNT bin-1 {cb or '>1020'}  |  "
              f"non-BNT cut-all {cg or '>1020'}")

    _write_provenance(args, areas, bnt, nonbnt, crossings)


def _write_provenance(args, areas, bnt, nonbnt, crossings):
    """Write <figure>_values.csv, _crossings.csv and _provenance.json beside the figure.

    Standing project rule (docs/HANDOFF_JZ_PAPER_FIGURES.md §0): a figure without its
    numbers cannot be compared against an earlier version by measurement, only argued
    about. n_seeds is carried per point because it varies between these two variants.
    """
    import csv
    import json
    import subprocess
    import datetime

    stem = os.path.splitext(args.out_pdf)[0]

    with open(stem + "_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "area_sqdeg", "upper_cut_lmax", "nsigma", "nsigma_err",
                    "n_seeds", "errbar_kind"])
        for label, df in (("BNT bin-1 only", bnt), ("non-BNT cut all", nonbnt)):
            if not len(df):
                continue
            for area in areas:
                s = df[df["area"] == area].sort_values("upper_cut")
                for _, r in s.iterrows():
                    w.writerow([label, int(area), int(r["upper_cut"]), f"{r['mean']:.6f}",
                                f"{(r['std'] if r['std'] == r['std'] else 0.0):.6f}",
                                int(r["n"]), "std"])

    with open(stem + "_crossings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["area_sqdeg", "lmax_at_0.3sigma_bnt_bin1", "lmax_at_0.3sigma_nonbnt_cutall"])
        for area, cb, cg in crossings:
            w.writerow([int(area), cb if cb else ">1020", cg if cg else ">1020"])

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "unavailable"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=REPO, stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except Exception:
        commit = "unknown"

    prov = {
        "figure": os.path.basename(stem),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "errbar": "std",
        "threshold_sigma": THRESHOLD,
        "n_seeds_bnt": sorted(int(v) for v in bnt["n"].unique()),
        "n_seeds_nonbnt": sorted(int(v) for v in nonbnt["n"].unique()) if len(nonbnt) else [],
        "source_tables": {"bnt": args.bnt_agg, "non_bnt": NONBNT_AGG},
        "derivation": "aggregated campaign tables (NOT recomputed from posteriors)",
        "estimator": "tensiometer gaussian_tension.Q_DM -> chi2.cdf -> from_confidence_to_sigma",
        "param_subset": [0, 1, 2],
        "param_names": ["Omega_m", "S8", "w0"],
        "lmin": 37,
        "rebin_note": args.rebin_note,
        "versions": {m: ver(m) for m in ("numpy", "pandas", "matplotlib")},
        "caveats": [
            "Crossings are the lowest grid cut with mean >= threshold (no interpolation), "
            "matching how this campaign has always reported them.",
        ],
    }
    if args.rebin_note:
        prov["extracted_pct_annotation"] = {
            "values": EXTRACTED_R40,
            "source": EXTRACTED_SOURCE,
            "reproducible_from_repo": False,
            "note": ("The non-BNT-at-rebin-40 tension these percentages divide by was never "
                     "saved as a table; the shipped non-BNT campaign is rebin=10. Dividing "
                     "the r40 BNT table by ps_submean_l37 yields 57/65/65/76/99/99, which is "
                     "the best-vs-best comparison, NOT this same-binning lossless check. "
                     "Treat the annotation as a quoted historical result."),
        }
        prov["caveats"].append(
            "At-cut comparison is best-vs-best: BNT at rebin=40 vs non-BNT at rebin=10 "
            "(different binning) — see docs/PLAN_bnt_optimal_binning.md Caveats.")
        prov["caveats"].append(
            "rebin=40 means 3-run means (overnight budget), noisier than the 5-run "
            "default-binning variant, and coarser cut resolution (visible staircase).")

    with open(stem + "_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"wrote {stem}_values.csv / _crossings.csv / _provenance.json")


if __name__ == "__main__":
    main()
