#!/usr/bin/env python3
"""Overlay null posteriors from arbitrary NDE output directories.

Built for the MOPED-vs-raw control, where the series live in different output trees and the existing
plotters are hardwired to the production tension directories. Each --series is
"Label=<dir>:<tag>", where <dir> holds nde/null_s*_<tag>.npy.

THE COMPARISON THIS WAS WRITTEN FOR: "best achievable under each method". Under MOPED compression the
better arm is BNT; under raw NPE the better arm is non-BNT, because raw catastrophically loses the
Omega_m-S8 degeneracy on the ill-conditioned BNT vector (r = -0.03 against the physical -0.9) while
handling the well-conditioned non-BNT vector fine. Overlaying the two answers whether the
compression machinery is worth its complexity relative to the naive route.

Run under jaxili (getdist 1.6.1); aname's getdist 1.4.3 cannot fill contours under matplotlib >=3.8.
Writes <out>_values.csv and <out>_provenance.json beside the figure.
"""
import argparse
import glob
import json
import os
import subprocess
import shlex
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
from getdist import MCSamples, plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
NAMES = ["Omega_m", "S8", "w0"]
LABELS = [r"\Omega_m", "S_8", "w_0"]
TRUTH = [0.26, 0.84, -1.0]


def load_null(d, tag):
    """Per-seed null posteriors, skipping the two disk-damage modes (unreadable header; zeroed
    interior that loads fine and returns a slab of zeros)."""
    good, runs, skipped = [], [], []
    for f in sorted(glob.glob(f"{d}/nde/null_s*_{tag}.npy")):
        run = os.path.basename(f).split("_s")[1].split("_")[0]
        try:
            a = np.load(f)
        except Exception as e:
            skipped.append((run, type(e).__name__)); continue
        if (a == 0).mean() > 0.5 or not np.isfinite(a).all():
            skipped.append((run, "zeroed/nonfinite")); continue
        good.append(a[:, :3]); runs.append(run)
    if not good:
        raise SystemExit(f"[fatal] no healthy null posteriors in {d} (tag={tag})")
    return good, runs, skipped


def stats(per_seed):
    """Seed-averaged covariance -> sigma, correlation matrix, FoM3. The correlation matrix is the
    diagnostic that matters here: a flow can get marginal widths right and still miss the parameter
    degeneracy entirely, which inflates the 3-param volume without widening any 1-D projection."""
    C = np.mean([np.cov(s, rowvar=False) for s in per_seed], axis=0)
    sig = np.sqrt(np.diag(C))
    R = C / np.outer(sig, sig)
    return sig, R, float(1.0 / np.sqrt(np.linalg.det(C)))


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help='"Label=<dir>:<tag>", repeatable (2-4 series)')
    ap.add_argument("--colors", default="C0,0.45,C3,C2")
    ap.add_argument("--seed-mode", choices=["pooled", "single"], default="pooled",
                    help="pooled = all surviving seeds concatenated; single = the representative "
                         "seed per series (tension.seeds, median-referenced on centre AND width)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    series = []
    for spec in a.series:
        label, rest = spec.split("=", 1)
        d, tag = rest.rsplit(":", 1)
        per, runs, sk = load_null(d, tag)
        sig_avg, R_avg, fom_avg = stats(per)      # seed-averaged: scatter-free constraining power
        if sk:
            print(f"[warn] {label}: skipped {len(sk)} damaged seed(s): {sk}")
        if a.seed_mode == "pooled":
            draw, seed_tag, diag = np.concatenate(per), f"{len(per)} seeds pooled", {"mode": "pooled"}
        else:
            sys.path.insert(0, os.path.join(REPO, "scripts"))
            from tension.seeds import representative_seed
            i, run, diag = representative_seed(per, runs)
            draw, seed_tag = per[i], f"seed {run}"
            diag["mode"] = "single"
        # the annotation must describe the contour actually DRAWN, not a different estimator
        sig, R, fom = stats([draw])
        series.append(dict(label=label, dir=d, tag=tag, per=per, runs=runs, skipped=sk,
                           sig=sig, R=R, fom=fom, sig_avg=sig_avg, fom_avg=fom_avg,
                           samples=draw, seed_tag=seed_tag, diag=diag))
        print(f"{label:26s} [{seed_tag}]  sigma={np.round(sig, 5)}  "
              f"r(Om,S8)={R[0,1]:+.3f}  det(R)={np.linalg.det(R):.4f}  "
              f"FoM3(drawn)={fom:.4e}  FoM3(seed-avg)={fom_avg:.4e}")

    if len(series) >= 2:
        r = series[0]["fom"] / series[1]["fom"]
        print(f"\nFoM3 ratio  {series[0]['label']} / {series[1]['label']} = {r:.3f}x")

    cols = a.colors.split(",")
    # getdist draws roots in order, so later ones land ON TOP. Draw the LARGEST contour first
    # (lowest FoM3) so the tightest posterior is never hidden underneath a looser one. The legend
    # and all reported numbers keep the user's --series order; only the z-order changes.
    order = sorted(range(len(series)), key=lambda i: series[i]["fom"], reverse=True)
    mcs = [MCSamples(samples=series[i]["samples"], names=NAMES, labels=LABELS,
                     label=series[i]["label"]) for i in order]
    draw_cols = [cols[i] for i in order]
    g = plots.get_subplot_plotter(width_inch=7.5)
    g.settings.legend_fontsize = 12
    g.settings.axes_labelsize = 15
    g.settings.axes_fontsize = 11
    g.triangle_plot(mcs, filled=True, contour_colors=draw_cols,
                    legend_labels=[series[i]["label"] for i in order], legend_loc="upper right")
    for i in range(3):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            ax.axvline(TRUTH[j], color="k", ls=":", lw=1, alpha=0.7)
            if i != j:
                ax.axhline(TRUTH[i], color="k", ls=":", lw=1, alpha=0.7)

    if a.title:
        g.fig.suptitle(a.title, fontsize=14, y=1.02)
    box = [rf"FoM$_3$ = {s['fom']:.2e}   $r(\Omega_m,S_8)$ = {s['R'][0,1]:+.2f}" for s in series]
    if len(series) >= 2:
        box.append(rf"ratio = $\mathbf{{{series[0]['fom']/series[1]['fom']:.2f}\times}}$")
    g.fig.text(0.62, 0.80, "\n".join(box), ha="center", va="center", fontsize=11,
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    if a.subtitle:
        g.fig.text(0.5, -0.01, a.subtitle, ha="center", fontsize=9, color="0.4")
    for ext in ("png", "pdf"):
        g.export(f"{a.out}.{ext}")

    hdr = ("label", "dir", "tag", "seed_mode", "seed_drawn", "n_seeds",
           "sigma_Om", "sigma_S8", "sigma_w0",
           "r_Om_S8", "r_Om_w0", "r_S8_w0", "det_R", "fom3_drawn", "fom3_seed_avg_cov")
    with open(f"{a.out}_values.csv", "w") as fh:
        fh.write(",".join(hdr) + "\n")
        for s in series:
            fh.write(",".join([s["label"].replace(",", ";"), s["dir"], s["tag"],
                               a.seed_mode, s["seed_tag"].replace(",", ";"), str(len(s["per"])),
                               *[f"{v:.6f}" for v in s["sig"]],
                               f"{s['R'][0,1]:.6f}", f"{s['R'][0,2]:.6f}", f"{s['R'][1,2]:.6f}",
                               f"{np.linalg.det(s['R']):.6f}", f"{s['fom']:.6e}",
                               f"{s['fom_avg']:.6e}"]) + "\n")

    prov = {
        "figure": os.path.basename(a.out),
        "generator": "scripts/plot_posterior_overlay.py",
        "command": shlex.join(sys.argv),
        "git_commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "mplstyle": "getdist default (no custom style sheet)",
        "area_deg2": 14000, "rebin": 20, "lmin": 37,
        "statistic": "tomographic auto+cross angular power spectra",
        "scales_included": "both arms at ell_max=460 (matched); BNT cuts bin-1 only, bins 2-4 to 1024",
        "seed_mode": a.seed_mode,
        "series": [{"label": s["label"], "dir": s["dir"], "n_seeds": len(s["per"]),
                    "seed_drawn": s["seed_tag"], "seed_selection": s["diag"],
                    "runs": s["runs"], "skipped_disk_damage": s["skipped"],
                    "sigma3": [float(v) for v in s["sig"]],
                    "corr_Om_S8": float(s["R"][0, 1]), "det_corr": float(np.linalg.det(s["R"])),
                    "fom3_drawn": s["fom"], "fom3_seed_avg_cov": s["fom_avg"]} for s in series],
        "posterior": "sigma/correlations/fom3_drawn describe exactly the samples plotted; "
                     "fom3_seed_avg_cov is the seed-averaged-covariance value, which removes "
                     "between-seed mean scatter and is the better constraining-power estimator",
        "caveats": [
            "Both series are calibrated and on-truth (SBC rank-std 0.28-0.29 vs the ideal 0.289; "
            "null means within 0.3 sigma of truth), so the FoM difference is a real difference in "
            "information, not one posterior being broken.",
            "The decisive quantity is r(Omega_m, S8). Weak lensing carries a physical degeneracy "
            "near -0.9; raw NPE on the ill-conditioned BNT vector returns -0.03, i.e. it loses the "
            "degeneracy structure entirely while keeping plausible (even tighter) marginals. That "
            "inflates the 3-param volume without widening any 1-D projection.",
            "SBC and TARP CANNOT see that failure: both test marginal rank uniformity per "
            "parameter, so a posterior with correct marginals and missing correlations passes both.",
            "MOPED is not free where it is not needed: on the well-conditioned non-BNT vector raw "
            "NPE is ~20% tighter than MOPED, presumably non-Gaussian information the "
            "Gaussian-optimal compression discards.",
        ],
    }
    with open(f"{a.out}_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"wrote {a.out}.png / .pdf / _values.csv / _provenance.json")


if __name__ == "__main__":
    main()
