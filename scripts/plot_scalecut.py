#!/usr/bin/env python3
"""Plot nsigma (3-param Q_DM baryon tension) vs ℓmax upper cut for the VMIM-compressed PS:
non-BNT (cut-all) and BNT (bin-1 cut) at 14000 deg², with the 0.3σ de-biasing line.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/baryon_tension/vmim_v2/scalecuts/tension_agg.csv")
    p.add_argument("--out", default="plots/nsigma_vs_upper_cut_compressed_14000.png")
    return p.parse_args()


def main():
    a = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = defaultdict(list)
    with open(a.csv) as fh:
        for r in csv.DictReader(fh):
            data[r["config"]].append((int(r["upper_cut"]), float(r["nsigma_mean"]), float(r["nsigma_std"])))

    style = {"nonbnt": ("non-BNT, cut all bins", "#1f77b4", "o"),
             "bnt": ("BNT, cut bin-1 only", "#2ca02c", "s")}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cfg, pts in data.items():
        pts.sort()
        x = [p[0] for p in pts]; y = [p[1] for p in pts]; e = [p[2] for p in pts]
        lab, col, mk = style.get(cfg, (cfg, None, "o"))
        ax.errorbar(x, y, yerr=e, marker=mk, ms=5, lw=1.5, capsize=3, color=col, label=lab)
    ax.axhline(0.3, ls="--", color="red", lw=1.2, label="0.3σ (de-biasing threshold)")
    ax.set_xlabel(r"Upper cut  $\ell_{\max}$")
    ax.set_ylabel(r"Baryon tension  $n_\sigma$  (3-param $Q_{DM}$)")
    ax.set_title("VMIM-compressed PS — baryon tension vs scale cut (14000 deg²)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=140)
    print(f"[plot] wrote {a.out}")
    for cfg, pts in data.items():
        cross = next((p[0] for p in sorted(pts) if p[1] > 0.3), None)
        print(f"  {cfg}: 0.3σ crossing at ℓmax ~ {cross}")


if __name__ == "__main__":
    main()
