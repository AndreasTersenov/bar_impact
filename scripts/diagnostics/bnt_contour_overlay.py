#!/usr/bin/env python3
"""Indicative overlay corner plot: non-BNT vs BNT posteriors at a fixed scale cut.

Overlays the 3-param (Ωm,S₈,w₀) null and biased posteriors for non-BNT cut-all and BNT bin-1
at the SAME upper cut, so you can see directly whether the BNT tension is flat because the
bias is removed (null/biased overlap, contours same size) or because the contours bloat
(BNT contours much larger than non-BNT). Truth marked. Stacks the 5 training-seed runs for a
representative ensemble contour.

  /home/tersenov/anaconda3/envs/aname/bin/python scripts/diagnostics/bnt_contour_overlay.py --area fullsky --cut 1020
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/diagnostics/bnt_contour_overlay.py --area 14000  --cut 1020
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
from getdist import plots  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tension import configs, estimators as E  # noqa: E402

TRUTH = {"Omega_m": 0.26, "S8": 0.84, "w0": -1.0}
REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"


def stack(camp, role, area, cut, runs=(1, 2, 3, 4, 5)):
    parts = []
    for r in runs:
        p = camp.posterior_path(role, area, cut, r)
        if p.exists():
            parts.append(np.load(p))
    return np.concatenate(parts) if parts else None


def width(samples):
    return np.array([samples[:, i].std() for i in (0, 1, 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="fullsky", help="'fullsky' or an int sqdeg")
    ap.add_argument("--cut", type=int, default=1020)
    ap.add_argument("--rebin", type=int, default=10, help="BNT ℓ-rebin factor (match the sweep).")
    ap.add_argument("--nonbnt-rebin", type=int, default=None,
                    help="non-BNT rebin (default = --rebin). Use 10 to give non-BNT its own optimum "
                         "while BNT uses its coarser optimum — the fair best-vs-best comparison.")
    args = ap.parse_args()
    fullsky = (args.area == "fullsky")
    area = "fullsky" if fullsky else int(args.area)
    rb = args.rebin
    rbn = args.nonbnt_rebin if args.nonbnt_rebin else rb

    if fullsky:
        nonbnt = configs.fullsky_campaign(runs=(1, 2, 3, 4, 5), rebin=rbn)
        bnt = configs.fullsky_bnt_bin1_campaign(runs=(1, 2, 3, 4, 5), rebin=rb)
        atxt = f"full sky (healpy, BNT r{rb} / non-BNT r{rbn})"
    else:
        nonbnt = configs.submean_l37_campaign(runs=(1, 2, 3, 4, 5), rebin=rbn)
        bnt = configs.bnt_bin1_campaign(runs=(1, 2, 3, 4, 5), rebin=rb)
        atxt = f"{area} deg² (masked, BNT r{rb} / non-BNT r{rbn})"

    data = {
        "non-BNT null": stack(nonbnt, "null", area, args.cut),
        "non-BNT biased (baryons)": stack(nonbnt, "biased", area, args.cut),
        "BNT bin-1 null": stack(bnt, "null", area, args.cut),
        "BNT bin-1 biased (baryons)": stack(bnt, "biased", area, args.cut),
    }
    missing = [k for k, v in data.items() if v is None]
    if missing:
        print(f"MISSING posteriors: {missing}"); return

    # width readout (what drives the 'inflation' story)
    wn, wb = width(data["non-BNT null"]), width(data["BNT bin-1 null"])
    print(f"[{atxt}, ℓmax={args.cut}]  σ(null) per param  [Ωm, S8, w0]:")
    print(f"   non-BNT : {np.round(wn,4)}")
    print(f"   BNT     : {np.round(wb,4)}   ratio BNT/non-BNT = {np.round(wb/wn,2)}")

    colors = {"non-BNT null": "0.4", "non-BNT biased (baryons)": "red",
              "BNT bin-1 null": "C0", "BNT bin-1 biased (baryons)": "green"}
    # Opaque (alpha=1) fills occlude, so draw widest-first → narrowest ends up on top and all four
    # stay visible (e.g. the tight non-BNT contours nested inside the bloated full-sky BNT ones).
    order = sorted(data, key=lambda k: width(data[k]).sum(), reverse=True)
    mcs = [E.make_mcsamples(data[k], indices=(0, 1, 2), label=k) for k in order]
    g = plots.get_subplot_plotter(width_inch=8.5)
    g.settings.alpha_filled_add = 1.0
    g.settings.legend_fontsize = 12
    g.triangle_plot(
        mcs, params=["Omega_m", "S8", "w0"],
        filled=True,
        contour_colors=[colors[k] for k in order],
        legend_labels=order,
        markers=TRUTH,
    )
    g.fig.suptitle(f"non-BNT vs BNT bin-1 posteriors — {atxt}, ℓmax={args.cut}  "
                   f"(σ ratio BNT/non-BNT = {np.round(wb/wn,2)}; truth=dashed)",
                   fontsize=12, y=1.02)
    tag = "fullsky" if fullsky else f"{area}"
    rtag = f"r{rb}" if rb == rbn else f"BNTr{rb}_nonBNTr{rbn}"
    out = f"{REPO}/plots/bnt_contour_overlay_{tag}_l{args.cut}_{rtag}.png"
    g.export(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
