#!/usr/bin/env python3
"""Triangle contours (Ωm, S8, w0) for the three summary statistics at one footprint.

Replaces the lost `plots/contours_PS_peaks_L1_baryons_{biased,unbiased}.pdf`. The shipped
`_biased.pdf` is 100% zeros and `_BNT.pdf` 39% zeros; the surviving `_unbiased.pdf` is
from Sept 2025, i.e. before the ℓmin=100→37 recovery and before the submean correction,
so it is a layout reference only — its contours are not the current analysis.

Three statistics, all in the corrected convention (monopole-subtracted, ℓ≥37 for the PS,
wavelet scales 1-4 for peaks/L1):

  null    nobaryons data vs nobaryons-trained model  — the unbiased baseline
  biased  baryonified data vs nobaryons-trained model — the same posterior with the
          baryonic bias in it; the offset from the truth marker IS the baryon bias

`--mode both` overlays the two per statistic (null filled, biased dashed), which is the
form that shows the bias directly rather than by comparing two figures.

Runs are used only in NULL/BIASED PAIRS. A run whose null or biased side is unreadable is
dropped from both, so the visible offset between the two is a like-for-like shift and not
partly a change of which seeds contribute. This matters here: at 14000 deg² the damage is
lopsided (L1 has 7 readable nulls but only 5 biased), so an unpaired figure would compare
different seed ensembles.

Pooling across NPE training seeds folds the seed-to-seed training scatter into the contour
width — that is deliberate (it is the honest width given we do not trust a single seed),
and `--single-run N` gives the single-seed version for comparison.

Run with the **jaxili** env, not aname:
  /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python \
      scripts/diagnostics/plot_contours_three_stats.py --area 14000 --mode both

aname pins getdist 1.4.3 for tensiometer, and 1.4.3 calls `QuadContourSet.tcolors`, which
matplotlib removed in 3.8 — it dies with AttributeError on any filled contour. jaxili has
getdist 1.6.1. This script computes no tension, so it does not need aname at all; the two
getdist versions are deliberately not unified (recovered memory `bar-impact-tension-env`).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMP = f"{REPO}/outputs/samples"
PSD = f"{REPO}/outputs/baryon_tension/ps_submean_l37/posteriors"

# Peaks/L1 were run under a footprint tag one degree off the PS mask area.
HOS_TAG = {2000: 2001, 5000: 5001, 10000: 10001, 14000: 14001, 28000: 28001, 35000: 35001}

TRUTH = {"Om": 0.26, "S8": 0.84, "w0": -1.0}
SIG_MAX = 0.08          # collapse guard on sigma(S8), same threshold as plot_nsigma_vs_area

# Okabe-Ito, matching plot_nsigma_vs_area.py so a reader can carry colours between figures.
STYLE = {"Power spectrum": "#0072B2", "Peak counts": "#D55E00", "L1 norm": "#009E73"}


def ps_globs(A):
    tail = (f"bins1234_l37-1020_r10_masked_{A}sqdeg_apod2.0_master_submean_"
            f"noisy_s0.26_run*.npy")
    base = f"{PSD}/mask_{A:05d}"
    return (f"{base}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_{tail}",
            f"{base}/biased/posterior_samples_ps_auto_cross_nobaryons_vs_baryonified_{tail}")


def hos_globs(prefix, A):
    tail = (f"bins1234_scales1234_noisy_s0.26_masked_{HOS_TAG[A]}sqdeg_submean_"
            f"new_normalization*_npe.npy")
    return (f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_nobaryons_{tail}",
            f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_baryonified_{tail}")


def run_index(fname):
    m = re.search(r"_run(\d+)", fname)
    return int(m.group(1)) if m else 1


def sigma_s8(a):
    return float(np.sqrt(np.cov(a[:, [0, 1, 2]], rowvar=False)[1, 1]))


def load_pairs(null_glob, biased_glob):
    """{run: (null, biased)} for runs readable and collapse-free on BOTH sides.

    Disk-failure-damaged .npy raise ValueError ("contains pickled data") because numpy
    reads the mangled header as a pickle stream — NOT an IOError. Guards written before
    the crash catch only (FileNotFoundError, IndexError) and let it through. Never
    "fix" this with allow_pickle=True: that unpickles garbage.
    """
    def index(pattern):
        out = {}
        for f in sorted(glob.glob(pattern)):
            try:
                out[run_index(f)] = np.load(f)
            except (FileNotFoundError, IndexError, ValueError, OSError):
                pass
        return out

    n, b = index(null_glob), index(biased_glob)
    pairs, dropped = {}, []
    for r in sorted(set(n) | set(b)):
        if r not in n or r not in b:
            dropped.append((r, "unreadable on one side"))
            continue
        s = max(sigma_s8(n[r]), sigma_s8(b[r]))
        if s >= SIG_MAX:
            dropped.append((r, f"collapsed sigma_S8={s:.3f}"))
            continue
        pairs[r] = (n[r], b[r])
    return pairs, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=int, default=14000, choices=sorted(HOS_TAG))
    ap.add_argument("--mode", default="both", choices=("both", "null", "biased"))
    ap.add_argument("--single-run", type=int, default=None,
                    help="Use only this run index instead of pooling all paired runs.")
    ap.add_argument("--width", type=float, default=6.0, help="figure width in inches")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    aa = f"{REPO}/styles/aa.mplstyle"
    if os.path.exists(aa):
        plt.style.use(aa)
    else:
        print(f"[warn] A&A style not found at {aa} — using matplotlib defaults")

    from getdist import MCSamples, plots

    stats = [("Power spectrum", ps_globs(args.area)),
             ("Peak counts", hos_globs("pc_", args.area)),
             ("L1 norm", hos_globs("", args.area))]

    names = ["Om", "S8", "w0"]
    labels = [r"\Omega_\mathrm{m}", "S_8", "w_0"]

    mcs, colors, filled, line_args, legend, rows, dropped_all = [], [], [], [], [], [], {}
    for label, (ng, bg) in stats:
        pairs, dropped = load_pairs(ng, bg)
        dropped_all[label] = dropped
        if args.single_run is not None:
            pairs = {r: v for r, v in pairs.items() if r == args.single_run}
        if not pairs:
            print(f"[warn] {label}: no usable null/biased pairs — omitted from the figure")
            continue
        runs = sorted(pairs)
        print(f"  {label:14s}: {len(runs)} paired run(s) {runs}"
              + (f"   dropped: {', '.join(f'r{r}({w})' for r, w in dropped)}" if dropped else ""))

        want = (("null", 0), ("biased", 1)) if args.mode == "both" else \
               ((args.mode, 0 if args.mode == "null" else 1),)
        for role, idx in want:
            s = np.concatenate([pairs[r][idx] for r in runs])[:, :3]
            tag = f"{label}" if args.mode != "both" else f"{label} — {role}"
            mcs.append(MCSamples(samples=s, names=names, labels=labels, label=tag))
            colors.append(STYLE[label])
            # Filled+solid vs open+dashed distinguishes null from biased, so it only
            # carries meaning in `both`. In a single-mode figure every series is the
            # same role, and dashing them all just makes the figure harder to read.
            solo = args.mode != "both"
            filled.append(solo or role == "null")
            line_args.append({"color": STYLE[label], "lw": 1.6,
                              "ls": "-" if (solo or role == "null") else "--"})
            legend.append(tag)
            rows.append({"statistic": label, "role": role, "n_runs": len(runs),
                         "runs": runs, "n_samples": int(s.shape[0]),
                         "mean": s.mean(0).tolist(), "std": s.std(0).tolist()})

    if not mcs:
        raise SystemExit("[fatal] nothing to plot — every statistic lost its pairs")

    g = plots.get_subplot_plotter(width_inch=args.width)
    g.settings.legend_fontsize = 8
    g.settings.axes_fontsize = 7
    g.settings.lab_fontsize = 9
    g.settings.alpha_filled_add = 0.55
    g.triangle_plot(mcs, names, filled=filled, contour_colors=colors,
                    contour_lws=[a["lw"] for a in line_args],
                    contour_ls=[a["ls"] for a in line_args],
                    legend_labels=legend, legend_loc="upper right",
                    markers=TRUTH)

    out = args.out or (f"{REPO}/outputs/plots/contours_three_stats/"
                       f"contours_PS_peaks_L1_{args.mode}_{args.area}")
    out = os.path.splitext(out)[0]
    if args.single_run is not None:
        out += f"_run{args.single_run}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    g.export(out + ".pdf")
    g.export(out + ".png")

    # ---- provenance -------------------------------------------------------
    # Standing rule (docs/HANDOFF_JZ_PAPER_FIGURES.md §0). For a contour figure the
    # "values" are the 1D marginals plus, critically, WHICH runs survived: the pooled
    # width depends on the seed ensemble, and the damage left a different ensemble than
    # the campaign originally had.
    with open(out + "_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "role", "parameter", "mean", "std", "truth",
                    "n_runs_pooled", "n_samples", "runs"])
        for r in rows:
            for i, p in enumerate(names):
                w.writerow([r["statistic"], r["role"], p, f"{r['mean'][i]:.6f}",
                            f"{r['std'][i]:.6f}", TRUTH[p], r["n_runs"], r["n_samples"],
                            " ".join(str(x) for x in r["runs"])])

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
        "figure": os.path.basename(out),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "area_sqdeg": args.area,
        "hos_footprint_tag": HOS_TAG[args.area],
        "mode": args.mode,
        "single_run": args.single_run,
        "truth": TRUTH,
        "param_names": ["Omega_m", "S8", "w0"],
        "collapse_guard_sigma_S8_max": SIG_MAX,
        "series": rows,
        "runs_dropped": {k: [[r, why] for r, why in v] for k, v in dropped_all.items()},
        "conventions": {
            "power_spectrum": "monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10",
            "peaks_l1": "wavelet scales 1-4, submean, new_normalization, noisy s=0.26",
        },
        "versions": {m: ver(m) for m in ("numpy", "scipy", "getdist", "matplotlib")},
        "mplstyle": aa if os.path.exists(aa) else "matplotlib defaults",
        "caveats": [
            "Runs are used only as null/biased PAIRS; a run unreadable on either side is "
            "dropped from both, so the null-to-biased offset is like-for-like. See "
            "runs_dropped for what the disk failure removed.",
            "Contours pool all surviving NPE training seeds, so their width includes the "
            "seed-to-seed training scatter, not just the posterior width of one seed. "
            "--single-run gives the single-seed version.",
            "aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from "
            "pre-crash figures are expected, data points are unaffected.",
            "The surviving pre-crash contours_PS_peaks_L1_baryons_unbiased.pdf (Sept 2025) "
            "predates the lmin 100->37 recovery and the submean correction, so it is NOT "
            "numerically comparable to this figure.",
        ],
    }
    with open(out + "_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)

    print(f"\nwrote {out}.pdf / .png")
    print(f"wrote {out}_values.csv / _provenance.json")
    for r in rows:
        m, s = r["mean"], r["std"]
        print(f"  {r['statistic']:14s} {r['role']:7s} n={r['n_runs']}  "
              f"Om {m[0]:.4f}±{s[0]:.4f}  S8 {m[1]:.4f}±{s[1]:.4f}  w0 {m[2]:.4f}±{s[2]:.4f}")


if __name__ == "__main__":
    main()
