#!/usr/bin/env python3
"""BNT vs standard basis: fractional baryonic impact, one panel per tomographic bin (paper Figs. 4/5).

Replaces plots/Cl_fractional_difference_all_bins_bnt.pdf (power spectrum) and
plots/l1_fractional_difference_all_bins_bnt.pdf (l1-norm), carrying every correction applied to the
non-BNT figures:

  * MATCHED SHAPE NOISE, in both bases. Critically the BNT and standard suites reuse the SAME
    per-map seeds, so the two series differ ONLY by the nulling transform -- which is exactly what
    these figures claim to isolate. (The published versions had independent noise in both bases,
    so part of the visible BNT/standard difference was noise, not nulling.)
  * BAND = err(stat_bar)/<stat_DMO>, the error of a single measurement against a known model,
    rather than the scatter of a difference.
  * NO +5 OFFSET for the HOS; empty bins are masked instead.
  * TRUE BIN CENTRES -- the saved SNR centres for the HOS, and the real log-band centres for the
    power spectrum instead of a linear axis under a logarithmic binning.

BIN 1 IS EXPECTED TO COINCIDE. The BNT matrix is lower-triangular with a leading 1, so its first
row is the identity: BNT bin 1 IS standard bin 1. The two series lying exactly on top of each other
in the first panel is a correctness check, not a coincidence -- and the published figures show it
too.

  PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python
  $PY scripts/plot_bnt_frac_diff.py --stat ps
  $PY scripts/plot_bnt_frac_diff.py --stat l1 --scale 1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import shlex
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# --- style -----------------------------------------------------------------
# paper_v1, NOT aa.mplstyle. Several figures from the submitted version are being
# kept in the revision, so everything regenerated has to match THEM: sans-serif,
# tab10, and the label/tick/legend sizes read off the notebook that made them.
# aa.mplstyle (A&A house style) is still in styles/ and is the better baseline for
# a figure set built from scratch -- it is deliberately not used here.
STYLE = os.path.join(REPO, "styles", "paper_v1.mplstyle")
plt.style.use(STYLE)

FID = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/"
       "fiducial/cosmo_fiducial")
SKY_DEG2 = 41253.0
LMIN, LMAX, NBAND = 30, 1024, 10
SCALE_ARCMIN = ["10'", "20'", "40'", "80'", "coarse"]
YLAB = {"ps": r"$\langle \Delta C_\ell \rangle / \langle C_\ell \rangle$",
        "l1": r"$\langle \Delta \ell_1 \rangle / \langle \ell_1 \rangle$",
        "peaks": r"$\langle \Delta \mathrm{PC} \rangle / \langle \mathrm{PC} \rangle$"}


def ps_edges():
    """Fixed log-band edges: rounded and clamped, so the first band starts exactly at LMIN.

    (astype(int) truncation made the first edge 29 in the published code; harmless there only
    because it never subtracted the offset before indexing.)
    """
    e = np.round(np.logspace(np.log10(LMIN), np.log10(LMAX), NBAND + 1)).astype(int)
    e[0], e[-1] = LMIN, LMAX
    return e


def ps_rebin(arr):
    e = ps_edges()
    return np.stack([arr[:, e[i] - LMIN:e[i + 1] - LMIN].mean(1) for i in range(NBAND)], axis=1)


def ps_centres():
    e = ps_edges()
    return np.array([0.5 * (e[i] + e[i + 1] - 1) for i in range(NBAND)])


def load(stat, basis, kind, b, tag="_matchednoise"):
    pre = {"ps": "all_bnt_cls" if basis == "bnt" else "all_cls",
           "l1": "all_bnt_l1_norms" if basis == "bnt" else "all_l1_norms",
           "peaks": "all_bnt_peak_counts" if basis == "bnt" else "all_peak_counts"}[stat]
    suf = "" if stat == "ps" else "_new_normalization"
    p = f"{FID}/{pre}_fiducial_{kind}_bin{b}_noisy_s0.26{suf}{tag}.npy"
    if not os.path.exists(p):
        sys.exit(f"[fatal] missing {p}")
    a = np.load(p)
    if not np.isfinite(a).all():
        sys.exit(f"[fatal] {p} has non-finite entries")
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stat", default="ps", choices=["ps", "l1", "peaks"])
    p.add_argument("--scale", type=int, default=1, help="wavelet scale (1-indexed), HOS only")
    p.add_argument("--area", default="fullsky")
    p.add_argument("--ylim", default="curves")
    p.add_argument("--ypad-lo", type=float, default=0.18)
    p.add_argument("--ypad-hi", type=float, default=0.32)
    p.add_argument("--fill-alpha", type=float, default=0.28)
    p.add_argument("--edge-alpha", type=float, default=0.50)
    p.add_argument("--edge-lw", type=float, default=0.65)
    p.add_argument("--curve-lw", type=float, default=2.6,
                   help="curve weight. The style baseline is 1.2; 1.4 lifts the "
                        "curves just clear of their bands without reading as heavy "
                        "at A&A column width.")
    p.add_argument("--fig-width", type=float, default=16.0,
                   help="canvas width in inches. Matches plot_hos_frac_diff.py so the two "
                        "families print at the same text size; was 20, which made this "
                        "figure's text print ~0.6x the sibling's.")
    p.add_argument("--fig-height", type=float, default=3.2,
                   help="canvas height in inches.")
    p.add_argument("--outdir", default="outputs/plots/bnt_frac_diff")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    fsky = 1.0 if a.area == "fullsky" else float(a.area) / SKY_DEG2
    bscale = 1.0 / np.sqrt(fsky)
    area_lab = "full sky" if a.area == "fullsky" else f"{float(a.area):.0f} deg2"
    s = a.scale - 1

    # x axis
    if a.stat == "ps":
        x, xlab, cen_src = ps_centres(), r"$\ell$", "true log-band centres"
    else:
        stem = "all_bnt_l1_norms" if a.stat == "l1" else "all_bnt_peak_counts"
        cp = f"{FID}/{stem}_fiducial_bin1_matchednoise_bincentres.npy"
        c = np.load(cp)
        x = c[s] if c.ndim == 2 else c
        xlab, cen_src = "SNR", os.path.basename(cp)

    # colours/styles follow the published figures: BNT orange solid, standard blue dashed
    STATLAB = {"ps": "", "l1": r"$\ell_1$-norm, ", "peaks": "peaks, "}[a.stat]
    SER = [(f"{STATLAB}BNT", "bnt", "C1", "-"),
           (f"{STATLAB}no BNT" if a.stat != "ps" else "No BNT", "std", "C0", "--")]

    print(f"stat={a.stat}  scale={a.scale if a.stat!='ps' else '-'}  area={area_lab} "
          f"(band x{bscale:.3f})\n  x from: {cen_src}")

    rows, panels = [], []
    for b in (1, 2, 3, 4):
        per = []
        for lab, basis, col, ls in SER:
            bar = load(a.stat, basis, "baryonified", b)
            nob = load(a.stat, basis, "nobaryons", b)
            if a.stat == "ps":
                bar, nob = ps_rebin(bar[:, LMIN:]), ps_rebin(nob[:, LMIN:])
            else:
                bar, nob = bar[:, s, :], nob[:, s, :]
            mn = nob.mean(0)
            keep = mn > 0
            curve = np.full(mn.shape, np.nan); band = np.full(mn.shape, np.nan)
            curve[keep] = (bar - nob).mean(0)[keep] / mn[keep]
            band[keep] = bar.std(0)[keep] / mn[keep] * bscale
            per.append((lab, col, ls, curve, band))
            for j in np.where(keep)[0]:
                rows.append(dict(stat=a.stat, basis=basis, bin=b, x=float(x[j]),
                                 frac_diff=float(curve[j]), band=float(band[j]),
                                 n_realizations=int(bar.shape[0]), n_seeds=1))
        panels.append(per)

    # CANVAS WIDTH vs TEXT SIZE -- these look coupled but need not be.
    #
    # Every text size comes from styles/paper_v1.mplstyle in POINTS, so what reaches the page
    # depends on how far LaTeX scales the canvas. The sibling family (plot_hos_frac_diff.py)
    # is figsize=(12, 3), published 11.67 in. This one was (20, 4), published 19.69 in, so at
    # the same width in the paper its text printed at 11.67/19.69 = 0.59x the sibling's.
    #
    # Simply shrinking the canvas to 12 fixed the text but squared up the panels: four panels
    # across 12 in is 3.0 in each against the 5.0 in they had at 20. The fix is to scale the
    # TYPE with the width instead, so the two are decoupled: at REF_WIDTH the sizes are
    # paper_v1's own, and any wider canvas gets them enlarged by exactly the factor LaTeX will
    # shrink them by. The printed size is then invariant to fig_width, and the width is free
    # to be chosen for the panel shape alone. 16 in is the middle ground -- 4.0 in panels,
    # close to the original proportions -- with text printing as if it were a 12 in figure.
    REF_WIDTH = 12.0
    k = a.fig_width / REF_WIDTH
    scaled = {key: plt.rcParams[key] * k for key in
              ("font.size", "axes.labelsize", "axes.titlesize",
               "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    print(f"  canvas {a.fig_width} x {a.fig_height} in; type scaled x{k:.2f} "
          f"so it prints as at {REF_WIDTH} in")

    with plt.rc_context(scaled):
        fig, axes = plt.subplots(1, 4, figsize=(a.fig_width, a.fig_height), sharey=True)
        for b, ax, per in zip((1, 2, 3, 4), axes, panels):
            for lab, col, ls, curve, band in per:
                ax.fill_between(x, curve - band, curve + band, color=col, alpha=a.fill_alpha, lw=0)
                for edge in (curve - band, curve + band):
                    ax.plot(x, edge, color=col, lw=a.edge_lw, alpha=a.edge_alpha, zorder=3)
                ax.plot(x, curve, color=col, ls=ls, lw=a.curve_lw, zorder=5,
                        label=lab if b == 1 else None)
            ax.axhline(0, color="black", ls="--", lw=1)
            ax.set_xlabel(xlab)
            ax.set_title(f"Bin {b}")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel(YLAB[a.stat])
        if a.ylim in ("auto", "curves"):
            if a.ylim == "curves":
                lo = np.nanmin([np.nanmin(c) for per in panels for *_, c, _ in per])
                hi = np.nanmax([np.nanmax(c) for per in panels for *_, c, _ in per])
                r = hi - lo
                axes[0].set_ylim(lo - a.ypad_lo * r, hi + a.ypad_hi * r)
            else:
                lo = np.nanmin([np.nanmin(c - bd) for per in panels for *_, c, bd in per])
                hi = np.nanmax([np.nanmax(c + bd) for per in panels for *_, c, bd in per])
                axes[0].set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        elif a.ylim != "none":
            axes[0].set_ylim(*[float(v) for v in a.ylim.split(",")])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=4,
                   frameon=False)
        fig.tight_layout()

    outdir = os.path.join(REPO, a.outdir); os.makedirs(outdir, exist_ok=True)
    name = a.name or (f"bnt_frac_diff_{a.stat}"
                      + ("" if a.stat == "ps" else f"_scale{a.scale}")
                      + ("" if a.area == "fullsky" else f"_{a.area}"))
    base = os.path.join(outdir, name)
    for ext in ("pdf", "png"):
        plt.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200, transparent=True)
    plt.close("all")

    with open(f"{base}_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "figure": name, "generator": "scripts/plot_bnt_frac_diff.py",
        "command": shlex.join(sys.argv),   # quoted, so it round-trips through shlex.split "git_commit": commit,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mplstyle": "styles/paper_v1.mplstyle (matches the submitted version)",
        "statistic": a.stat,
        "scales_included": ("PS: 10 logarithmic bands, lmin=30, lmax=1024"
                            if a.stat == "ps" else
                            f"wavelet scale {a.scale} (~{SCALE_ARCMIN[s]})"),
        "x_axis_source": cen_src,
        "curve": "mean(stat_bar - stat_DMO)/mean(stat_DMO); empty DMO bins masked",
        "band": f"std(stat_bar)/mean(stat_DMO) x 1/sqrt(f_sky); area = {area_lab}",
        "noise_seeds": ("MATCHED between baryonified and DMO, AND shared between the BNT and "
                        "standard suites, so the two series differ only by the nulling transform "
                        "(jobs 533505/536722 standard, 536997/536998 BNT)"),
        "f_sky": fsky, "band_scale_applied": bscale,
        "caveats": [
            "Bin 1 is expected to coincide between the two bases: the BNT matrix's first row is "
            "the identity, so BNT bin 1 IS standard bin 1. Agreement there is a correctness check.",
            "No +5 offset in the HOS denominator; empty bins are masked instead.",
            "y-range is fitted to the CURVES; the near-empty end bins have bands far larger.",
            "Band scaled to survey area by 1/sqrt(f_sky) -- indicative for the HOS, which have no "
            "mode-counting result behind them.",
        ],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    blob = json.dumps(prov, indent=2)
    tmp = f"{base}_provenance.json.tmp"
    open(tmp, "w").write(blob); json.loads(open(tmp).read())
    os.replace(tmp, f"{base}_provenance.json")
    print(f"wrote {base}.pdf/.png + _values.csv + _provenance.json")


if __name__ == "__main__":
    main()
