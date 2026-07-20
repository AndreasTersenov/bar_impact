#!/usr/bin/env python3
"""Single-render of the LIVE score-compressed baryon-tension plot at 14000 deg² (call repeatedly).

Recomputes the 3-param Q_DM tension from whatever score posteriors exist right now and renders one
PNG. Blue = score-BNT bin-1 (compressed); grey = score-nonBNT cut-all (compressed, appears once that
sweep runs). Faint dotted = the published RAW curves (the under-extracted blue / noisy grey) for
before/after context. Designed to be looped (e.g. every 60s) so the PNG updates live.

Run under aname (tensiometer + matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/live_score_plot.py
"""
import glob
import os
import re
import sys
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/mnt/home/tersenov/software/bar_impact"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from tension import estimators  # noqa: E402

AREA = 14000
THRESH = 0.3
P3 = (0, 1, 2)
OUT = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_score_14000_live.png"


def score_curve(tag):
    """Aggregate score posteriors under <tag>/area14000/posteriors into (cut, mean, std, n)."""
    pdir = f"{REPO}/outputs/baryon_tension/{tag}/area{AREA}/posteriors"
    rows = []
    for cdir in glob.glob(f"{pdir}/cut*"):
        cut = int(re.search(r"cut(\d+)", cdir).group(1))
        ns = []
        for f in sorted(glob.glob(f"{cdir}/null_run*.npy")):
            fb = f.replace("null", "biased")
            if not os.path.exists(fb):
                continue
            t = estimators.tension_sigma(np.load(f), np.load(fb), indices=P3, estimator="q_dm")
            if t["ok"]:
                ns.append(t["nsigma"])
        if ns:
            rows.append((cut, float(np.mean(ns)), float(np.std(ns)), len(ns)))
    return pd.DataFrame(sorted(rows), columns=["upper_cut", "mean", "std", "n"])


def raw_curve(tag):
    p = f"{REPO}/outputs/baryon_tension/{tag}/tables/tension_3param_agg.csv"
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["area"] == AREA].sort_values("upper_cut")


def main():
    blue = score_curve("bnt_ps_bin1_score_l37")
    grey = score_curve("ps_cutall_score_l37")
    raw_blue = raw_curve("bnt_ps_bin1_submean_l37")
    raw_grey = raw_curve("ps_submean_l37")

    plt.rcParams.update({"axes.labelsize": 14, "legend.fontsize": 10})
    fig, ax = plt.subplots(figsize=(8, 5.6))

    if len(raw_blue):
        ax.plot(raw_blue["upper_cut"], raw_blue["mean"], ":", color="C0", alpha=0.45, lw=1.5,
                label="raw-BNT bin-1 (uncompressed, under-extracted)")
    if len(raw_grey):
        ax.plot(raw_grey["upper_cut"], raw_grey["mean"], ":", color="0.55", alpha=0.55, lw=1.5,
                label="raw-nonBNT cut-all (uncompressed, noisy)")
    if len(grey):
        ax.errorbar(grey["upper_cut"], grey["mean"], yerr=grey["std"], fmt="s-", color="0.4",
                    ms=5, lw=1.5, elinewidth=1.0, capsize=3, label="score nonBNT — cut all bins")
    if len(blue):
        ax.errorbar(blue["upper_cut"], blue["mean"], yerr=blue["std"], fmt="o-", color="C0",
                    ms=6, lw=1.8, elinewidth=1.3, capsize=4,
                    label="score BNT — cut bin-1 only (bins 2-4 full)")
    ax.axhline(THRESH, color="r", ls="--", lw=1.3, label=f"Threshold ({THRESH}σ)")

    nb = int(blue["n"].sum()) if len(blue) else 0
    ng = int(grey["n"].sum()) if len(grey) else 0
    ax.set_title(f"Score-compressed baryon tension vs scale cut — {AREA} deg²", fontsize=14)
    ax.text(0.5, 1.005,
            f"live {datetime.now():%H:%M:%S} | score-BNT {nb}/90 · score-grey {ng}/90 posteriors | "
            f"calibrated, on-truth, binning-independent",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="0.4")
    ax.set_xlabel(r"Upper cut $\ell_{\max}$ (BNT bin-1 swept; bins 2-4 full)")
    ax.set_ylabel(r"Significance ($n_\sigma$), 3-param $Q_{\rm DM}$")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, ls=":")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{datetime.now():%H:%M:%S}] rendered {OUT}  (score-BNT {nb}/90, score-grey {ng}/90)")


if __name__ == "__main__":
    main()
