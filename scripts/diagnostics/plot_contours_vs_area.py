#!/usr/bin/env python3
"""Triangle contours (Ωm, S8, w0) for every survey area overlaid — one figure per statistic.

Shows how the posterior tightens with survey area, at FULL MAP RESOLUTION, separately for
each summary statistic. Companion to plot_contours_three_stats.py, which overlays the three
statistics at one area; this overlays the six areas for one statistic.

Full map resolution means:
  power spectrum : monopole-subtracted MASTER, ℓ = 37 .. 1020, rebin 10
  peaks / L1     : wavelet scales1234 — all four DETAIL scales, coarse/mass-sheet excluded —
                   on footprint-mean-subtracted ("submean") maps, new_normalization, σ_e=0.26

The submean requirement is not cosmetic. The pre-submean higher-order products spuriously
tighten the masked posteriors, which is precisely the error the ℓmin-recovery/submean campaign
corrected, so this script refuses to fall back to a non-submean file: the glob carries
`_submean_` and a missing file is reported, never silently substituted.

COLOUR. Survey area is an ORDERED quantity, so the six contours use a sequential single-hue
ramp (light = small area, dark = large), not six categorical hues — a categorical palette
would imply the areas are unrelated categories and a rainbow would imply a false ordering of
its own. The ramp is built from each statistic's own base hue (the Okabe-Ito colour that
statistic carries in every other figure), so a reader still recognises "this is the L1 figure"
at a glance while reading area off the lightness.

Contours are drawn UNFILLED. Six filled 2D contours over one another are unreadable however
they are coloured; lines keep all six legible.

  --statistic ps|peaks|l1     (default: all three, one figure each)
  --role      null|biased     null = nobaryons data (constraining power);
                              biased = baryonified data (the bias, uncorrected)
  --include-fullsky           add a 7th contour. PS only in practice — see the note below.

Run with the **jaxili** env (getdist 1.6.1; aname's 1.4.3 cannot draw contours under
matplotlib >= 3.8):
  /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python \
      scripts/diagnostics/plot_contours_vs_area.py --role null
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from tension.seeds import representative_seed  # noqa: E402

SAMP = f"{REPO}/outputs/samples"
PSD = f"{REPO}/outputs/baryon_tension/ps_submean_l37/posteriors"
PSD_FS = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/posteriors/fullsky"

AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
HOS_TAG = {2000: 2001, 5000: 5001, 10000: 10001, 14000: 14001, 28000: 28001, 35000: 35001}

TRUTH = {"Om": 0.26, "S8": 0.84, "w0": -1.0}
SIG_MAX = 0.08          # collapse guard on sigma(S8), same threshold as the other generators
LMAX = 1020             # full map resolution
HOS_SCALES = "scales1234"

# Base hue per statistic — matches plot_nsigma_vs_area.py / plot_contours_three_stats.py.
STAT = {
    "ps":    {"label": "Power spectrum", "hue": "#0072B2"},
    "peaks": {"label": "Peak counts",    "hue": "#D55E00"},
    "l1":    {"label": "L1 norm",        "hue": "#009E73"},
}


def area_ramp(hue, n):
    """n colours, light -> dark, through the statistic's own hue.

    Sequential because area is ordered. Endpoints are a near-white tint and a darkened
    shade of the same hue, so lightness alone encodes area and the hue stays recognisable.
    """
    base = np.array(to_rgb(hue))
    light = 1 - 0.18 * (1 - base)          # very light tint
    dark = 0.42 * base                      # darkened shade
    cmap = LinearSegmentedColormap.from_list("ramp", [light, base, dark])
    return [cmap(x) for x in np.linspace(0.12, 1.0, n)]


def run_index(fname):
    m = re.search(r"_run(\d+)", fname)
    return int(m.group(1)) if m else 1


def sigma_s8(a):
    return float(np.sqrt(np.cov(a[:, [0, 1, 2]], rowvar=False)[1, 1]))


def ps_glob(area, role):
    r = "nobaryons_vs_nobaryons" if role == "null" else "nobaryons_vs_baryonified"
    if area == "fullsky":
        # Full sky carries NO _submean_ and NO _master_ tag, and that is correct rather than
        # an omission: submean subtracts a FOOTPRINT mean, which only exists under a mask, and
        # the full-sky leg uses the healpy per-ell pipeline rather than MASTER decoupling.
        # Consequence: the full-sky contour is not magnitude-comparable to the masked ones.
        return (f"{PSD_FS}/{role}/posterior_samples_ps_auto_cross_{r}_bins1234_"
                f"l37-{LMAX}_r10_noisy_s0.26_run*_npe.npy")
    tail = (f"bins1234_l37-{LMAX}_r10_masked_{area}sqdeg_apod2.0_master_submean_"
            f"noisy_s0.26_run*.npy")
    return f"{PSD}/mask_{area:05d}/{role}/posterior_samples_ps_auto_cross_{r}_{tail}"


def hos_glob(prefix, area, role):
    r = "nobaryons_vs_nobaryons" if role == "null" else "nobaryons_vs_baryonified"
    # `_submean_` is REQUIRED in the pattern. A non-submean file is a different (superseded)
    # measurement, so a missing submean product must surface as missing, not be substituted.
    if area == "fullsky":
        return (f"{SAMP}/posterior_samples_{prefix}{r}_bins1234_{HOS_SCALES}_noisy_s0.26"
                f"_fullsky_submean_new_normalization*_npe.npy")
    return (f"{SAMP}/posterior_samples_{prefix}{r}_bins1234_{HOS_SCALES}_noisy_s0.26"
            f"_masked_{HOS_TAG[area]}sqdeg_submean_new_normalization*_npe.npy")


def globber(stat):
    return {"ps": lambda a, r: ps_glob(a, r),
            "peaks": lambda a, r: hos_glob("pc_", a, r),
            "l1": lambda a, r: hos_glob("", a, r)}[stat]


def load(pattern):
    """Pooled samples over readable, non-collapsed seeds. Returns (array, runs, dropped).

    A disk-damaged .npy raises ValueError ("contains pickled data") because numpy reads the
    mangled header as a pickle stream — NOT an IOError. Never "fix" that with
    allow_pickle=True, which unpickles garbage.
    """
    keep, runs, dropped = [], [], []
    for f in sorted(glob.glob(pattern)):
        r = run_index(f)
        try:
            a = np.load(f)
        except (FileNotFoundError, IndexError, ValueError, OSError) as e:
            dropped.append((r, f"unreadable ({type(e).__name__})"))
            continue
        s = sigma_s8(a)
        if s >= SIG_MAX:
            dropped.append((r, f"collapsed sigma_S8={s:.3f}"))
            continue
        keep.append(a[:, :3])
        runs.append(r)
    if not keep:
        return None, [], dropped, []

    # CENTRE-outlier guard. The sigma(S8) collapse guard above catches a seed whose posterior
    # is too WIDE; it cannot catch one whose posterior is the right width in the wrong PLACE.
    # Real case: peaks/biased at 28000 deg^2 run 8 returned w0 = -0.053 (essentially the prior
    # edge) while every other seed sat near -1.25, with a perfectly normal sigma(S8) = 0.013.
    # Pooling it inflated the covariance determinant ~200x, which showed up as a pooled/per-seed
    # FoM ratio of x14 where every other contour sits at ~1.1. plot_nsigma_vs_area.py's dual
    # mis-fit QA tests per-parameter WIDTH anomalies, so it would miss this too.
    keep, runs, dropped = _drop_centre_outliers(keep, runs, dropped)
    if not keep:
        return None, [], dropped, []
    # Per-seed FoM_3 alongside the pooled samples. Pooling folds NPE seed-to-seed training
    # scatter into the covariance, which LOWERS the FoM, so the pooled value describes the
    # drawn contour while the per-seed mean is what plot_fom_vs_area.py plots. Both are
    # recorded; comparing one against the other across figures would be an error.
    per_seed = [fom3(a) for a in keep]
    order = np.argsort(runs)
    keep = [keep[i] for i in order]
    runs = [runs[i] for i in order]
    per_seed = [per_seed[i] for i in order]
    return np.concatenate(keep), runs, dropped, per_seed, keep


CENTRE_MADZ = 5.0       # robust-z on the per-seed posterior centre, per parameter
CENTRE_REL = 1.0        # ...AND the offset must exceed this many posterior widths


def _drop_centre_outliers(keep, runs, dropped, madz=CENTRE_MADZ, rel=CENTRE_REL):
    """Drop a seed whose posterior CENTRE is a robust outlier in any of (Om, S8, w0).

    TWO conditions, both required, mirroring the dual-condition design of
    plot_nsigma_vs_area.py's QA:

      1. the centre is a robust outlier      |c - median| > madz * MAD
      2. AND the offset actually matters     |c - median| > rel * (typical posterior sigma)

    Condition 2 is not optional. Median/MAD alone is scale-free, so when the surviving seeds
    happen to cluster very tightly the MAD collapses and a physically trivial offset scores as
    a huge z. Measured here: at 35000 deg^2 the good seeds sit within w0 = -1.267 +/- 0.001, so
    seeds at -1.2435 and -1.2306 -- off by 0.024 and 0.037, i.e. 0.26 and 0.41 of the w0
    posterior width -- came out at 8 and 12 MAD and would have been discarded as outliers.
    They are ordinary scatter. Requiring a full posterior width as well keeps them and still
    removes the real failures by a wide margin (run 8 at 28000: w0 = -0.053, 25.6 posterior
    widths off; run 10 at 35000: 7.2 widths off).

    The threshold is set at 1.0 width deliberately conservatively, to discard only the
    unambiguous. Borderline cases exist and are NOT silently cut: at 35000 deg^2, biased peaks,
    runs 6 and 7 sit 0.5 and 0.8 widths from the median. They look anomalous only because the
    other seven seeds agree on w0 to within 0.002 while the posterior width is 0.09 -- a
    suspiciously tight cluster that may itself mean those seeds are not as independent as
    assumed. Deciding whether 6 and 7 are bad fits or the cluster is too tight is a judgement
    about the data, not a threshold choice, so both are kept and the fact is recorded here.

    Needs >= 4 seeds for a usable median; below that nothing is called an outlier.
    """
    if len(keep) < 4:
        return keep, runs, dropped
    cent = np.array([a.mean(0) for a in keep])                 # (nseed, 3)
    width = np.median(np.array([a.std(0) for a in keep]), 0)   # typical posterior sigma
    med = np.median(cent, 0)
    dev = np.abs(cent - med)
    mad = 1.4826 * np.median(dev, 0) + 1e-12
    bad = (dev > madz * mad) & (dev > rel * width)             # BOTH conditions
    ok = ~bad.any(1)
    if ok.all():
        return keep, runs, dropped
    for i, good in enumerate(ok):
        if not good:
            j = int(np.argmax(dev[i] / np.maximum(width, 1e-12)))
            pname = ("Om", "S8", "w0")[j]
            dropped.append((runs[i],
                            f"centre outlier: {pname}={cent[i][j]:.4f} vs median {med[j]:.4f}"
                            f" ({dev[i][j] / width[j]:.1f} posterior widths, "
                            f"{dev[i][j] / mad[j]:.0f} MAD)"))
    return ([a for a, g in zip(keep, ok) if g],
            [r for r, g in zip(runs, ok) if g], dropped)


def fom3(samples):
    """FoM_3 = 1/sqrt(det C_3) over (Omega_m, S8, w0) — same definition as plot_fom_vs_area.py."""
    c = np.cov(np.asarray(samples)[:, :3], rowvar=False)
    return float(1.0 / np.sqrt(np.linalg.det(c)))


def build(stat, role, include_fullsky, width, seed_mode):
    from getdist import MCSamples, plots

    gl = globber(stat)
    areas = list(AREAS) + (["fullsky"] if include_fullsky else [])
    names, labels = ["Om", "S8", "w0"], [r"\Omega_\mathrm{m}", "S_8", "w_0"]

    mcs, colors, legend, rows, dropped_all = [], [], [], [], {}
    ramp = area_ramp(STAT[stat]["hue"], len(areas))
    for col, area in zip(ramp, areas):
        pooled, runs, dropped, per_seed_fom, per_seed_arrays = load(gl(area, role))
        s = pooled
        seed_choice = None
        if pooled is not None and seed_mode == "single":
            i, run_lbl, seed_choice = representative_seed(per_seed_arrays, runs)
            s = per_seed_arrays[i]
        dropped_all[str(area)] = dropped
        if pooled is None:
            print(f"  [missing] {stat} {role} {area}: no usable posterior "
                  f"({gl(area, role)})")
            continue
        lbl = "full sky" if area == "fullsky" else f"{area} deg$^2$"
        mcs.append(MCSamples(samples=s, names=names, labels=labels, label=lbl))
        colors.append(col)
        legend.append(lbl)
        rows.append({"area": area, "n_seeds": len(runs), "runs": runs,
                     "n_samples": int(s.shape[0]),
                     "mean": s.mean(0).tolist(), "std": s.std(0).tolist(),
                     "fom3_pooled": fom3(pooled),
                     "seed_shown": (seed_choice or {}).get("chosen_run"),
                     "seed_choice": seed_choice,
                     "fom3_per_seed_mean": float(np.mean(per_seed_fom)),
                     "fom3_per_seed_std": float(np.std(per_seed_fom))})
        print(f"  {str(area):>8}: {len(runs)} seed(s) {runs}   "
              f"FoM3 pooled {rows[-1]['fom3_pooled']:.4g} / per-seed "
              f"{rows[-1]['fom3_per_seed_mean']:.4g}"
              + (f"   dropped {len(dropped)}" if dropped else ""))

    if not mcs:
        print(f"  [skip] {stat} {role}: nothing usable, no figure written")
        return None

    # Shared contour style, not the mplstyle sheet: plt.style.use() does not reach getdist's
    # own font sizes or fills. See scripts/paper_contour_style.py.
    import paper_contour_style as PCS
    g = plots.get_subplot_plotter(width_inch=width)
    _palette = PCS.apply(g)
    # Unfilled: six overlapping filled contours are unreadable at any palette.
    g.triangle_plot(mcs, names, filled=False, contour_colors=colors,
                    contour_lws=[1.4] * len(mcs),
                    legend_labels=legend, legend_loc="upper right", markers=TRUTH)

    tag = "" if not include_fullsky else "_with_fullsky"
    mode_tag = "_pooled" if seed_mode == "pooled" else "_single_seed"
    out = (f"{REPO}/outputs/plots/contours_vs_area/"
           f"contours_vs_area_{stat}_{role}_l37-{LMAX}{tag}{mode_tag}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    g.export(out + ".pdf")
    g.export(out + ".png")

    scales = ({"power_spectrum": f"monopole-subtracted MASTER, lmin=37, lmax={LMAX}, rebin=10"}
              if stat == "ps" else
              {"peaks_l1": (f"wavelet {HOS_SCALES} — all four detail scales, coarse/mass-sheet "
                            f"excluded — submean (footprint-mean-subtracted) maps, "
                            f"new_normalization, noisy sigma_e=0.26")})

    with open(out + "_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "role", "area_sqdeg", "parameter", "mean", "std", "truth",
                    "n_seeds", "n_samples", "fom3_pooled", "fom3_per_seed_mean",
                    "fom3_per_seed_std", "runs"])
        for r in rows:
            for i, p in enumerate(names):
                w.writerow([STAT[stat]["label"], role, r["area"], p, f"{r['mean'][i]:.6f}",
                            f"{r['std'][i]:.6f}", TRUTH[p], r["n_seeds"], r["n_samples"],
                            f"{r['fom3_pooled']:.6g}", f"{r['fom3_per_seed_mean']:.6g}",
                            f"{r['fom3_per_seed_std']:.6g}",
                            " ".join(map(str, r["runs"]))])

    def ver(m):
        try:
            return __import__(m).__version__
        except Exception:
            return "unavailable"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"

    aa = f"{REPO}/styles/paper_v1.mplstyle"
    json.dump({
        "figure": os.path.basename(out),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "statistic": STAT[stat]["label"],
        "role": role,
        "seed_mode": seed_mode,
        "role_meaning": ("null = nobaryons data vs nobaryons-trained model (constraining power)"
                         if role == "null" else
                         "biased = baryonified data vs nobaryons-trained model (bias, uncorrected)"),
        "areas_drawn": [r["area"] for r in rows],
        "cut": f"FULL map resolution — no scale cut applied (PS lmax={LMAX}, HOS {HOS_SCALES})",
        "scales_included": scales,
        "truth": TRUTH,
        "param_names": ["Omega_m", "S8", "w0"],
        "collapse_guard_sigma_S8_max": SIG_MAX,
        "fom3": {
            "definition": "FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)",
            "pooled_vs_per_seed": (
                "fom3_pooled comes from the pooled samples, i.e. the covariance the DRAWN "
                "contour represents; pooling across NPE training seeds folds training scatter "
                "into the covariance and so LOWERS the FoM. fom3_per_seed_mean is what "
                "plot_fom_vs_area.py and plot_scaling_vs_area.py plot. Do not compare a pooled "
                "value against a per-seed one across figures."),
            "per_area": {str(r["area"]): {"fom3_pooled": r["fom3_pooled"],
                                          "fom3_per_seed_mean": r["fom3_per_seed_mean"],
                                          "fom3_per_seed_std": r["fom3_per_seed_std"],
                                          "n_seeds": r["n_seeds"]} for r in rows},
        },
        "colour_encoding": ("sequential single-hue ramp on the statistic's own base hue; "
                            "light = small area, dark = large. Area is ordered, so a "
                            "categorical or rainbow palette would misrepresent it."),
        "series": rows,
        "runs_dropped": {k: [[r, why] for r, why in v] for k, v in dropped_all.items()},
        "versions": {m: ver(m) for m in ("numpy", "scipy", "getdist", "matplotlib")},
        "mplstyle": aa if os.path.exists(aa) else "matplotlib defaults",
        **PCS.provenance(_palette),
        "caveats": [
            "FULL RESOLUTION — no scale cut. For the power spectrum and, at large areas, the "
            "higher-order statistics, this is the regime where the baryon bias is significant; "
            "the 'biased' role therefore shows a posterior that is NOT baryon-safe by design.",
            "Contours pool all readable, non-collapsed seeds for the given role, so their width "
            "includes NPE seed-to-seed training scatter, not the posterior width of one seed. "
            "n_seeds per area is in the values CSV and varies with disk damage.",
            "Seeds are NOT matched between roles here (each role pools its own readable set), "
            "because these are single-role figures. Use plot_contours_three_stats.py, which "
            "pairs runs, for any null-to-biased comparison.",
            "Higher-order products are the SUBMEAN (footprint-mean-subtracted) ones with the "
            "corrected mask treatment. The pre-submean products spuriously tighten the masked "
            "posteriors; the glob requires '_submean_' so a missing file surfaces as missing "
            "rather than being silently substituted.",
            "styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this "
            "figure sits beside the figures kept verbatim from it.",
        ],
    }, open(out + "_provenance.json", "w"), indent=2)

    print(f"  wrote {os.path.relpath(out, REPO)}.pdf / .png / _values.csv / _provenance.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statistic", default="all", choices=("all", "ps", "peaks", "l1"))
    ap.add_argument("--role", default="null", choices=("null", "biased"))
    ap.add_argument("--include-fullsky", action="store_true")
    ap.add_argument("--seed-mode", default="pooled", choices=("pooled", "single"),
                    help="pooled: stack every surviving seed (contour includes training "
                         "scatter). single: draw the single most REPRESENTATIVE seed, which "
                         "is what a real analysis reports. See scripts/tension/seeds.py.")
    ap.add_argument("--width", type=float, default=6.0)
    args = ap.parse_args()

    aa = f"{REPO}/styles/paper_v1.mplstyle"
    if os.path.exists(aa):
        plt.style.use(aa)
    else:
        print(f"[warn] A&A style not found at {aa} — using matplotlib defaults")

    stats = ["ps", "peaks", "l1"] if args.statistic == "all" else [args.statistic]
    for s in stats:
        print(f"\n=== {STAT[s]['label']} — {args.role} — full resolution ===")
        build(s, args.role, args.include_fullsky, args.width, args.seed_mode)


if __name__ == "__main__":
    main()
