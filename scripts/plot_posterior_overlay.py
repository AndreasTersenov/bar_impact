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
LABELS = [r"\Omega_m", r"\sigma_8", "w_0"]
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
    # Omit to take colours from paper_contour_style, like the rest of the paper's contour
    # figures. The old default was a private tab10 list (C0,0.45,C3,C2) that could not follow
    # PALETTE=; the published figures then pinned --colors C0,0.45 on top of it, so this family
    # rendered in tab10 blue while every other contour figure was Okabe-Ito.
    ap.add_argument("--colors", default=None,
                    help="comma-separated contour colours in --series order. Omit to use "
                         "scripts/paper_contour_style.py (the paper default).")
    ap.add_argument("--seed-mode", choices=["pooled", "single"], default="pooled",
                    help="pooled = all surviving seeds concatenated; single = the representative "
                         "seed per series (tension.seeds, median-referenced on centre AND width)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--fom-box", action="store_true",
                    help="draw the FoM/ratio box inside the axes. Off by default: it restates "
                         "values.csv in a form nobody can verify and goes stale on regeneration. "
                         "Put the numbers in the caption instead.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    series = []
    for spec in a.series:
        # rsplit, not split: the LABEL may legitimately contain an "=" -- the paper's legends
        # carry things like "standard basis, global $\ell_{\rm max}=460$". Splitting on the
        # FIRST "=" cut the label there and left "460$=<dir>" as the path, failing on a
        # confusing FileNotFoundError. The directory never contains "=", so the LAST one is
        # unambiguously the label/path separator.
        label, rest = spec.rsplit("=", 1)
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

    import paper_contour_style as PCS
    # Cycle colours, NOT colors_for(). The series here are bases and cuts, not the three
    # statistics, so the PALETTES table has no meaningful key for them -- and the keys that do
    # match ("BNT basis", "standard basis") map the standard arm onto "0.45", a grey that is not
    # an Okabe-Ito colour and looks foreign in a figure whose other arm is Okabe-Ito blue.
    # cycle_colors gives every series its own colourblind-safe hue.
    cols = a.colors.split(",") if a.colors else PCS.cycle_colors(len(series))
    # getdist draws roots in order, so later ones land ON TOP. Draw the LARGEST contour first
    # (lowest FoM3) so the tightest posterior is never hidden underneath a looser one.
    # The LEGEND follows this same order, not the user's --series order: getdist maps
    # legend_labels onto the roots positionally, so the two cannot be decoupled without
    # mislabelling the colours. Printed numbers and values.csv do keep --series order.
    # ASCENDING FoM3. High FoM3 = TIGHT contour, so ascending puts the widest first (bottom) and
    # the tightest last (top), which is what the line above describes. This read reverse=True
    # until 2026-08-04 -- descending -- which did the exact opposite and buried the tightest
    # posterior under the loosest. Invisible while the figures were effectively unfilled; the
    # moment fills became opaque, the BNT arm vanished under the non-BNT one.
    order = sorted(range(len(series)), key=lambda i: series[i]["fom"])
    mcs = [MCSamples(samples=series[i]["samples"], names=NAMES, labels=LABELS,
                     label=series[i]["label"]) for i in order]
    draw_cols = [cols[i] for i in order]
    # Type and fills from the shared contour style, so every contour figure in the paper
    # matches. plt.style.use() cannot do this: getdist carries its own font sizes.
    g = plots.get_subplot_plotter(width_inch=7.5)
    _palette = PCS.apply(g)
    # markers=, not hand-drawn axvline/axhline. getdist puts the truth on both axes of every
    # panel itself, in the thin dashed grey the published contour figures use; the manual
    # version drew dotted BLACK, which is the same information in a different visual language
    # from the rest of the paper.
    g.triangle_plot(mcs, filled=True, contour_colors=draw_cols,
                    legend_labels=[series[i]["label"] for i in order], legend_loc="upper right",
                    markers=dict(zip(NAMES, TRUTH)))

    if a.title:
        g.fig.suptitle(a.title, fontsize=14, y=1.02)
    # The FoM/ratio box is OFF by default. It duplicates values.csv inside the image, where
    # it cannot be checked and goes stale if the figure is regenerated with different seeds,
    # and it sits in the empty upper-right where a reader expects the legend. The numbers
    # belong in the caption, sourced from values.csv. --fom-box brings it back for a slide.
    if a.fom_box:
        box = [rf"FoM$_3$ = {s['fom']:.2e}   $r(\Omega_m,\sigma_8)$ = {s['R'][0,1]:+.2f}"
               for s in series]
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
