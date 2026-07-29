#!/usr/bin/env python3
"""Triangle (Om,S8,w0) at a single scale cut: null + biased for compressed PS non-BNT (cut-all)
and BNT (bin-1). Truth markers (0.26,0.84,-1.0); biased landing on the null/truth = de-biased.
"""
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/baryon_tension/vmim_v2/cut480")
    p.add_argument("--cut", type=int, default=480)
    p.add_argument("--out", default="outputs/baryon_tension/vmim_v2/cut480/contours_cut480.png")
    p.add_argument("--title", default="")
    p.add_argument("--biased-only", action="store_true", help="drop the null contours, biased filled")
    return p.parse_args()


def main():
    a = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots

    root = Path(a.root)
    specs = [  # (path, label, filled, color, ls)
        (root / f"nonbnt_c{a.cut}/nde/null_pooled_nonbnt.npy",        "non-BNT cut-all — null",   True,  "#1f77b4", "-"),
        (root / f"nonbnt_c{a.cut}/nde/null_biased_pooled_nonbnt.npy", "non-BNT cut-all — biased", False, "#1f77b4", "--"),
        (root / f"bnt_c{a.cut}/nde/null_pooled_bnt.npy",              "BNT bin-1 — null",         True,  "#2ca02c", "-"),
        (root / f"bnt_c{a.cut}/nde/null_biased_pooled_bnt.npy",       "BNT bin-1 — biased",       False, "#2ca02c", ":"),
    ]
    if a.biased_only:
        specs = [(p, lab.replace(" — biased", ""), True, c, "-") for p, lab, f, c, ls in specs
                 if "biased" in lab]
    names, labels = ["Om", "S8", "w0"], [r"\Omega_m", "S_8", "w_0"]
    mcs, filled, colors, line_args, rows = [], [], [], [], []
    for path, label, fill, color, ls in specs:
        if not path.exists():
            print("MISSING:", path); continue
        s = np.load(path)[:, :3]
        mcs.append(MCSamples(samples=s, names=names, labels=labels, label=label))
        filled.append(fill); colors.append(color)
        line_args.append({"lw": 2.2, "ls": ls, "color": color})
        rows.append((label, s.mean(0), s.std(0)))

    g = plots.get_subplot_plotter(width_inch=8.5)
    g.settings.legend_fontsize = 11
    g.settings.axes_labelsize = 15
    g.triangle_plot(mcs, filled=filled, contour_colors=colors, line_args=line_args,
                    markers={"Om": 0.26, "S8": 0.84, "w0": -1.0})
    if a.title:
        import matplotlib.pyplot as plt
        plt.suptitle(a.title, y=1.02)
    g.export(a.out)
    print(f"[plot] wrote {a.out}\n")
    for label, mu, sd in rows:
        print(f"  {label:26s} Om={mu[0]:.3f}±{sd[0]:.3f} S8={mu[1]:.3f}±{sd[1]:.3f} w0={mu[2]:.3f}±{sd[2]:.3f}")


if __name__ == "__main__":
    main()
