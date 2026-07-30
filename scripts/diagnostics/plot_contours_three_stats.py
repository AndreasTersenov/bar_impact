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
import sys
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from tension.seeds import representative_seed  # noqa: E402

SAMP = f"{REPO}/outputs/samples"
PSD = f"{REPO}/outputs/baryon_tension/ps_submean_l37/posteriors"

# Peaks/L1 were run under a footprint tag one degree off the PS mask area.
HOS_TAG = {2000: 2001, 5000: 5001, 10000: 10001, 14000: 14001, 28000: 28001, 35000: 35001}

TRUTH = {"Om": 0.26, "S8": 0.84, "w0": -1.0}
SIG_MAX = 0.08          # collapse guard on sigma(S8), same threshold as plot_nsigma_vs_area

# Okabe-Ito, matching plot_nsigma_vs_area.py so a reader can carry colours between figures.
STYLE = {"Power spectrum": "#0072B2", "Peak counts": "#D55E00", "L1 norm": "#009E73"}


PS_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
THRESHOLD = 0.3         # the baryon-safety tolerance the campaign reports against


def required_ps_lmax(A, threshold=THRESHOLD):
    """Largest step-40 upper cut whose mean 3-param tension is still BELOW `threshold`.

    Not the same as the 0.3-sigma *crossing*. The crossing (reported by
    plot_nsigma_vs_lmax.py as `lmax_first_grid_cut_at_or_above`) is the first cut that
    FAILS; the cut you would actually adopt is the last one that PASSES. At 14000 deg^2
    those are 500 and 460 respectively — using the crossing would put a 0.41-sigma bias
    into a figure captioned "baryon-safe".

    Read from the campaign table rather than hardcoded, so it tracks the data.
    """
    rows = [(int(r["upper_cut"]), float(r["mean"]))
            for r in csv.DictReader(open(PS_AGG)) if int(r["area"]) == A]
    if not rows:
        raise SystemExit(f"[fatal] no rows for area {A} in {PS_AGG}")
    safe = [c for c, m in sorted(rows) if m < threshold]
    if not safe:
        raise SystemExit(f"[fatal] no cut at {A} deg^2 keeps the PS bias below "
                         f"{threshold} sigma — nothing is baryon-safe here")
    return max(safe)


def ps_globs(A, lmax):
    tail = (f"bins1234_l37-{lmax}_r10_masked_{A}sqdeg_apod2.0_master_submean_"
            f"noisy_s0.26_run*.npy")
    base = f"{PSD}/mask_{A:05d}"
    return (f"{base}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_{tail}",
            f"{base}/biased/posterior_samples_ps_auto_cross_nobaryons_vs_baryonified_{tail}")


def hos_globs(prefix, A, scales):
    tail = (f"bins1234_{scales}_noisy_s0.26_masked_{HOS_TAG[A]}sqdeg_submean_"
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
    ap.add_argument("--cut-mode", default="full", choices=("full", "baryon-safe"),
                    help="full: PS to lmax=1020, HOS keep all four detail scales. "
                         "baryon-safe: PS at the largest cut whose bias stays under "
                         "0.3 sigma (read from the campaign table), HOS drop the finest "
                         "wavelet scale (scales234).")
    ap.add_argument("--ps-lmax", type=int, default=None,
                    help="Override the PS upper cut. Use 400 to reproduce the ell-matched "
                         "pairing of the Fisher baryon-safe figure.")
    ap.add_argument("--hos-scales", default=None,
                    help="Override the HOS scale tag, e.g. scales234 or scales1234.")
    ap.add_argument("--seed-mode", default="pooled", choices=("pooled", "single"),
                    help="pooled: stack every matched seed. single: draw the single most "
                         "REPRESENTATIVE seed (scripts/tension/seeds.py), which is what a "
                         "real analysis reports. --single-run overrides this with an "
                         "explicit choice.")
    ap.add_argument("--width", type=float, default=6.0, help="figure width in inches")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # The two cuts are not the same KIND of cut — the PS drops multipoles, the HOS drop a
    # wavelet scale — so they are resolved independently and both recorded in the sidecar.
    if args.cut_mode == "baryon-safe":
        ps_lmax = args.ps_lmax or required_ps_lmax(args.area)
        hos_scales = args.hos_scales or "scales234"
        ps_cut_note = ("largest step-40 cut with mean tension < 0.3 sigma"
                       if not args.ps_lmax else "explicit --ps-lmax")
    else:
        ps_lmax = args.ps_lmax or 1020
        hos_scales = args.hos_scales or "scales1234"
        ps_cut_note = "no upper cut (full resolution)" if not args.ps_lmax else "explicit --ps-lmax"
    print(f"cut mode: {args.cut_mode}   PS lmax={ps_lmax} ({ps_cut_note})   HOS {hos_scales}")

    aa = f"{REPO}/styles/aa.mplstyle"
    if os.path.exists(aa):
        plt.style.use(aa)
    else:
        print(f"[warn] A&A style not found at {aa} — using matplotlib defaults")

    from getdist import MCSamples, plots

    stats = [("Power spectrum", ps_globs(args.area, ps_lmax)),
             ("Peak counts", hos_globs("pc_", args.area, hos_scales)),
             ("L1 norm", hos_globs("", args.area, hos_scales))]

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
            _per = [pairs[r][idx][:, :3] for r in runs]
            s = np.concatenate(_per)
            seed_choice = None
            if args.seed_mode == "single" and args.single_run is None:
                _i, _lbl, seed_choice = representative_seed(_per, runs)
                s = _per[_i]
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
                         "seed_shown": (seed_choice or {}).get("chosen_run"),
                         "seed_choice": seed_choice,
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

    # The cut goes in the filename: a baryon-safe figure and a full-resolution one are
    # different physics and must never land on the same path.
    _cut_tag = "" if args.cut_mode == "full" else f"_bsafe_l{ps_lmax}_{hos_scales}"
    _cut_tag += "_pooled" if args.seed_mode == "pooled" else "_single_seed"
    out = args.out or (f"{REPO}/outputs/plots/contours_three_stats/"
                       f"contours_PS_peaks_L1_{args.mode}_{args.area}{_cut_tag}")
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
        "seed_mode": args.seed_mode,
        "single_run": args.single_run,
        "truth": TRUTH,
        "param_names": ["Omega_m", "S8", "w0"],
        "collapse_guard_sigma_S8_max": SIG_MAX,
        "series": rows,
        "runs_dropped": {k: [[r, why] for r, why in v] for k, v in dropped_all.items()},
        "cut_mode": args.cut_mode,
        "cuts": {
            "power_spectrum_lmax": ps_lmax,
            "power_spectrum_lmax_chosen_by": ps_cut_note,
            "hos_scale_tag": hos_scales,
            "threshold_sigma": THRESHOLD,
        },
        "conventions": {
            "power_spectrum": (f"monopole-subtracted MASTER, lmin=37, lmax={ps_lmax}, "
                               f"rebin=10"),
            "peaks_l1": (f"wavelet {hos_scales}, submean, new_normalization, "
                         f"noisy s=0.26"),
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
    if args.cut_mode == "baryon-safe":
        prov["caveats"] += [
            "The PS cut and the HOS cut are different KINDS of cut — multipoles vs a "
            "wavelet scale — so they are not ell-matched to each other. Each is the cut "
            "that keeps its own statistic under 0.3 sigma. Pass --ps-lmax 400 to instead "
            "reproduce the ell-matched pairing used by the Fisher baryon-safe figure.",
            "The PS lmax is the largest step-40 cut still BELOW 0.3 sigma, not the "
            "crossing. The crossing is the first cut that fails (500 at 14000 deg^2); "
            "adopting it would put a 0.41-sigma bias in a 'baryon-safe' figure.",
        ]
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
