#!/usr/bin/env python3
"""BNT vs scale-cut vs all-scales triangle contours for a higher-order statistic (peaks or L1).

Replaces the notebook cells that made `plots/triangle_plot_subset_baryonified_{peaks_bnt,l1_BNT}.pdf`.
Those lived in `notebooks/inference/paper_plots{,_dark}.ipynb`, which the disk failure left 34% and
23% NUL respectively; the light `peaks_bnt.pdf` is 100% NUL and gone. The POSTERIOR SAMPLES are all
intact, so nothing needs retraining or re-running -- this is a pure replot, and it is a script rather
than a notebook so the next replot does not depend on recovering damaged JSON.

WHAT THE FIGURE SHOWS. One statistic, three treatments of the baryon problem, at the fiducial
cosmology with baryons injected into the data and a baryon-free model:
    all scales   -- keep everything; the contour is tight but biased (that is the problem)
    scale cut    -- drop the finest wavelet scale; less biased, wider
    BNT          -- the nulling transform instead of a cut

THE SCALE LABELS. Filenames are 1-indexed, internal indices are 0-indexed
(`submit_npe_inference_l1_parameter_sweep_parallel.py:60`):
    scales1234 = internal 0,1,2,3 = ALL FOUR scales
    scales234  = internal   1,2,3 = THREE scales, finest dropped
The original legend listed FOUR entries for the three-scale vector, [20',40',80',coarse]. The coarse
scale was not used in practice, so the label now reads [20',40',80'] -- three entries for three
scales. Do not reintroduce "coarse".

THE BNT ARM (`--bnt-arm`). The two original figures are NOT consistent with each other:
    peaks  plotted `bnt_pc_nobaryons_vs_baryonified` (run2)   -- baryonified, like the other two
    L1     plotted `nobaryons_vs_nobaryons`                    -- NO baryons injected
and both labelled it plainly "BNT". So the L1 figure appears to show BNT sitting on truth while the
scale cut does not, but it sits on truth because it has no baryons in it. Measured means (Om,S8,w0):
    L1 BNT, nobaryons    (plotted) : w0 = -1.022   <- lands on truth
    L1 BNT, baryonified  (not used): w0 = -0.825   <- biased, and in the OPPOSITE direction to blue
The notebook defined both, labelling the second "$\\ell_1^{BNT}$, baryonified".

`--bnt-arm` is therefore explicit and has NO default that hides the choice. `nobaryons` reproduces
the published L1 figure; `baryonified` gives the like-for-like comparison. Whichever is chosen is
recorded in the provenance and stated in the legend, so the figure can no longer be ambiguous.

Run with the JAXILI env: aname pins getdist 1.4.3, which calls `QuadContourSet.tcolors` (removed in
matplotlib 3.8) and dies on any filled contour.

  PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
  $PY scripts/plot_hos_bnt_triangle.py --stat peaks --bnt-arm baryonified
  $PY scripts/plot_hos_bnt_triangle.py --stat l1    --bnt-arm nobaryons
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMP = os.path.join(REPO, "outputs", "samples")

PAR_LABELS = [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
SUB = [0, 1, 2]
TRUTH = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])

# The finest wavelet scale is dropped by the cut; "coarse" was never used, so three scales.
CUT_SCALES = r"[20',40',80']"

# stat -> (file stem for each arm). All are `new_normalization`, noisy s=0.26, bins1234.
FILES = {
    "peaks": {
        "bnt_baryonified": "posterior_samples_bnt_pc_nobaryons_vs_baryonified_bntbins1234_"
                           "scales1234_noisy_s0.26_new_normalization_run2_npe.npy",
        "bnt_nobaryons":   "posterior_samples_bnt_pc_nobaryons_vs_nobaryons_bntbins1234_"
                           "scales1234_noisy_s0.26_new_normalization_npe.npy",
        "cut":             "posterior_samples_pc_nobaryons_vs_baryonified_bins1234_"
                           "scales234_noisy_s0.26_new_normalization_npe.npy",
        "all":             "posterior_samples_pc_nobaryons_vs_baryonified_bins1234_"
                           "scales1234_noisy_s0.26_new_normalization_npe.npy",
    },
    "l1": {
        "bnt_baryonified": "posterior_samples_nobaryons_vs_baryonified_bntbins1234_"
                           "scales1234_noisy_s0.26_new_normalization_npe.npy",
        "bnt_nobaryons":   "posterior_samples_nobaryons_vs_nobaryons_bntbins1234_"
                           "scales1234_noisy_s0.26_new_normalization_npe.npy",
        "cut":             "posterior_samples_nobaryons_vs_baryonified_bins1234_"
                           "scales234_noisy_s0.26_new_normalization_npe.npy",
        "all":             "posterior_samples_nobaryons_vs_baryonified_bins1234_"
                           "scales1234_noisy_s0.26_new_normalization_npe.npy",
    },
}
STAT_TEX = {"peaks": "peaks", "l1": r"$\ell_1$"}
STAT_BNT_TEX = {"peaks": "BNT peaks", "l1": r"$\ell_1^{BNT}$"}


def load(path):
    """Load a posterior, refusing anything with disk damage rather than plotting nonsense."""
    if not os.path.exists(path):
        sys.exit(f"[fatal] missing: {path}")
    raw = np.fromfile(path, dtype=np.uint8)
    zfrac = float((raw == 0).mean())
    a = np.load(path)
    if zfrac > 0.05 or not np.isfinite(a).all():
        sys.exit(f"[fatal] {os.path.basename(path)} looks disk-damaged "
                 f"(zeros={zfrac:.1%}, finite={np.isfinite(a).all()})")
    return a


def fom3(s):
    return float(1.0 / np.sqrt(np.linalg.det(np.cov(s[:, :3], rowvar=False))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stat", required=True, choices=["peaks", "l1"])
    p.add_argument("--bnt-arm", required=True, choices=["nobaryons", "baryonified"],
                   help="which BNT run to draw. No default: the two published figures disagreed on "
                        "this and the legend did not say which was used (see module docstring).")
    p.add_argument("--colors", default="0.55,#d62728,#1f77b4",
                   help="comma-separated getdist contour colors, in legend order BNT,cut,all")
    p.add_argument("--cut-scales", default=CUT_SCALES,
                   help="scale list shown in the legend for the cut arm")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--outdir", default="outputs/plots/hos_bnt_triangle")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    f = FILES[a.stat]
    bnt_key = f"bnt_{a.bnt_arm}"
    bnt_note = ("NO baryons injected -- not like-for-like with the other two"
                if a.bnt_arm == "nobaryons" else "baryonified, like-for-like with the other two")
    bnt_label = STAT_BNT_TEX[a.stat] + ("" if a.bnt_arm == "baryonified" else ", no baryons")

    series = [
        (bnt_label, f[bnt_key], bnt_key),
        (f"{STAT_TEX[a.stat]}, scales {a.cut_scales}", f["cut"], "cut"),
        (f"{STAT_TEX[a.stat]}, all scales", f["all"], "all"),
    ]
    colors = [c.strip() for c in a.colors.split(",")]
    if len(colors) != len(series):
        sys.exit(f"[fatal] need {len(series)} colors, got {len(colors)}")

    print(f"=== {a.stat} triangle | BNT arm = {a.bnt_arm} ({bnt_note}) ===")
    mcs, rows = [], []
    for (label, fname, key), col in zip(series, colors):
        s = load(os.path.join(SAMP, fname))
        mcs.append(MCSamples(samples=s[:, SUB], names=[PAR_LABELS[i] for i in SUB], label=label))
        m, sd = s[:, :3].mean(0), s[:, :3].std(0)
        # float() every numpy scalar: json cannot serialize np.float32, and the L1 posteriors are
        # stored float32 where the peaks ones are float64 -- so this only bites on some arms, which
        # is exactly the kind of thing that ships a truncated provenance file unnoticed.
        # n_seeds is 1 and stated rather than omitted: these runs predate the seed-pooling
        # convention, so each contour's width carries ONE seed's training scatter. The publish
        # gate warns when this column is absent, and it is right to -- an absent seed count
        # reads as "pooled" to anyone comparing against the current figures.
        rows.append(dict(arm=key, label=label, color=col, file=fname, n_seeds=1,
                         n_samples=int(s.shape[0]),
                         mean_Om=float(m[0]), mean_S8=float(m[1]), mean_w0=float(m[2]),
                         sigma_Om=float(sd[0]), sigma_S8=float(sd[1]), sigma_w0=float(sd[2]),
                         fom3=fom3(s)))
        print(f"  {label:34s} n={s.shape[0]:5d}  mean(Om,S8,w0)={np.round(m,4)}  "
              f"FoM3={rows[-1]['fom3']:.3e}")

    style = "dark_background" if a.dark else "default"
    with plt.style.context(style):
        g = plots.get_subplot_plotter(width_inch=6)
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add = 0.99
        g.settings.axes_fontsize = 14
        g.settings.lab_fontsize = 16
        g.settings.legend_fontsize = 14
        g.settings.title_limit_fontsize = 12
        g.triangle_plot(mcs, filled=True, contour_colors=colors,
                        legend_labels=[m.label for m in mcs],
                        markers={PAR_LABELS[i]: TRUTH[i] for i in SUB},
                        contour_lws=1.1,
                        marker_args={"color": "white" if a.dark else "black",
                                     "lw": 1.5 if a.dark else 1.0})
        outdir = os.path.join(REPO, a.outdir)
        os.makedirs(outdir, exist_ok=True)
        name = a.name or f"hos_bnt_{a.stat}_{a.bnt_arm}" + ("_dark" if a.dark else "")
        base = os.path.join(outdir, name)
        for ext in ("pdf", "png"):
            plt.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200,
                        transparent=bool(a.dark))
        plt.close("all")

    with open(f"{base}_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "figure": name,
        "generator": "scripts/plot_hos_bnt_triangle.py",
        "command": " ".join(sys.argv),
        "git_commit": commit,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mplstyle": ("matplotlib dark_background" if a.dark else "matplotlib default")
                    + " + getdist get_subplot_plotter(width_inch=6); no repo style sheet",
        "statistic": a.stat,
        "bnt_arm": a.bnt_arm,
        "bnt_arm_note": bnt_note,
        "scales_included": {
            "all": "scales1234 = internal 0,1,2,3 = all four wavelet scales",
            "cut": f"scales234 = internal 1,2,3 = three scales, finest dropped; legend "
                   f"{a.cut_scales}",
            "bnt": "bntbins1234 scales1234",
        },
        "legend_correction": "Original legend listed four entries [20',40',80',coarse] for the "
                             "three-scale cut vector. The coarse scale was not used; the label now "
                             "lists three.",
        "truth": {PAR_LABELS[i]: float(TRUTH[i]) for i in SUB},
        "fom_definition": "FoM_3 = 1/sqrt(det Cov(Omega_m, S8, w0))",
        "caveats": [
            f"BNT arm is the {a.bnt_arm} run ({bnt_note}).",
        ] + ([
            "The BNT contour has NO baryons injected, while the other two do. It therefore sits "
            "on the truth because it is baryon-free, NOT because BNT removed the bias. The "
            "baryonified BNT run exists and sits at w0 = -0.825, biased opposite to 'all "
            "scales'. Rerun with --bnt-arm baryonified for the like-for-like comparison."
        ] if a.bnt_arm == "nobaryons" else []) + [
            "OLD CONVENTION: these NPE runs predate the lmin=37 / monopole-subtraction / MASTER "
            "recovery. Do not overlay on current-convention figures without rerunning.",
            "Single NPE run per arm (not seed-pooled), so the contour width carries the "
            "seed-to-seed training scatter of one seed only.",
        ],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "series": rows,
    }
    # Serialize fully, THEN write, THEN read back. A json.dump straight to the final path leaves a
    # truncated file if encoding raises partway (it did: np.float32 on the L1 arms), and a partial
    # provenance beside a perfectly good PDF reads as a complete figure.
    blob = json.dumps(prov, indent=2)
    tmp = f"{base}_provenance.json.tmp"
    with open(tmp, "w") as fh:
        fh.write(blob)
    json.loads(open(tmp).read())
    os.replace(tmp, f"{base}_provenance.json")
    print(f"\nwrote {base}.pdf/.png + _values.csv + _provenance.json")


if __name__ == "__main__":
    main()
