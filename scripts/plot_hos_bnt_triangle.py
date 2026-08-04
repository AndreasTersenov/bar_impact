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
import shlex
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMP = os.path.join(REPO, "outputs", "samples")

# Column 1 is sigma_8. Verified against cosmo_params.npy, NOT taken from any script's label list:
# col1 spans 0.400-1.397 with fiducial 0.84, which is the CosmoGrid fiducial sigma_8 (alongside
# Om=0.26, w0=-1, H0=67.36, ns=0.9649, Ob=0.0493 -- the rest of the TRUTH row below). Reading that
# column as the OTHER parameter would imply sigma_8 spanning 0.313-2.347, which is unphysical, and
# would put the fiducial at 0.84*sqrt(0.26/0.3) = 0.782 rather than 0.84.
#
# Every generator in the repo used to mislabel this column; scripts/paper/fix_sigma8_labels.py
# relabelled all 48 of them on 2026-08-04. No conversion was applied anywhere, and none is wanted:
# the samples were always sigma_8 and only the labels were wrong.
# \Omega_\mathrm{m} (upright m) matches the published contour family.
PAR_LABELS = [r"$\Omega_\mathrm{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
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
        # _run1: the RETRAINED arm (job 570333, 2026-08-04). The original run of this name is
        # prior-dominated and is kept in NON_CONVERGED below so it cannot be selected by mistake.
        "bnt_baryonified": "posterior_samples_nobaryons_vs_baryonified_bntbins1234_"
                           "scales1234_noisy_s0.26_new_normalization_run1_npe.npy",
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


# Unusable runs. The posterior fills the prior in every parameter, so it is not a constraint at
# all -- it just looks like an enormously wide one, which is exactly the shape that would be
# mistaken for "BNT inflates the contours". Measured ranges for the l1 entry below:
#   Om 0.016-0.443, S8 0.434-1.224, w0 -1.37..-0.30    (prior-wide)
# against Om 0.216-0.301, S8 0.742-0.893 for the retrain that replaced it.
#
# CAUSE, established by job 570333. Not a training failure: the checkpoint name is built from
# simulation_type only, not fiducial_type (run_npe_inference.py:679,696), so ONE NDE served both
# arms that evening -- the good nobaryons contour at 19:47 and this one at 21:53. The training was
# fine; the OBSERVATION broke the evaluation. Standardised against the training grid, 588 of 589
# live features of the baryonified fiducial sit at |z| < 1.8, but one -- BNT bin4, finest scale,
# SNR bin 36, nonzero in 2 of 16,965 grid sims -- sits at |z| = 27. filter_zero_variance_bins then
# used an absolute floor only (1e-5), which that bin survives; the relative floor added in 3d92555
# (1e-6 x median variance) drops it. Retraining under the fixed filter (569 features, not 589)
# recovered sigma_S8 = 0.022 from 0.107.
#
# Listed by filename so the guard survives any refactor of FILES.
NON_CONVERGED = {
    "posterior_samples_nobaryons_vs_baryonified_bntbins1234_scales1234_noisy_s0.26"
    "_new_normalization_npe.npy":
        "l1 BNT baryonified, ORIGINAL run (2026-02-08): NPE evaluation degenerated; the posterior "
        "is prior-dominated. SUPERSEDED -- FILES now points at the _run1 retrain (job 570333), "
        "which converged to sigma_S8=0.022. This entry stays so the bad file cannot be reselected.",
}


def load(path):
    """Load a posterior, refusing disk damage AND known non-converged runs rather than plotting
    nonsense. A prior-wide posterior is the more dangerous of the two: it renders as a plausible
    figure."""
    _b = os.path.basename(path)
    if _b in NON_CONVERGED:
        sys.exit(f"[fatal] {_b}\n        {NON_CONVERGED[_b]}")
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
    # DEFAULT: take the colours from paper_contour_style, exactly as the published contour
    # figures do (contours_baryon_safe_biased_14000_pooled and the rest of that family call
    # PCS.colors_for). These labels are not in the PALETTES table -- they are three views of ONE
    # statistic, not three different statistics -- so they fall through to the palette's CYCLE
    # and come out as its first three hues, the same blue/orange/green the published figures use.
    #
    # Do NOT reinstate a hardcoded default here. This script carried one for a long time
    # (0.55,#d62728,#1f77b4) and that is precisely why it read as a foreign palette next to the
    # rest of the paper: a private colour list cannot follow PALETTE= or any future repalette.
    p.add_argument("--colors", default=None,
                   help="comma-separated getdist contour colors, in legend order BNT,cut,all. "
                        "Omit to use scripts/paper_contour_style.py (the paper default).")
    p.add_argument("--cut-scales", default=CUT_SCALES,
                   help="scale list shown in the legend for the cut arm")
    p.add_argument("--outline-bnt", action="store_true",
                   help="draw the BNT contour as an OUTLINE instead of a filled region. The BNT "
                        "arm is uncut, so its contour is ~10x wider per dimension than the others; "
                        "filled, it covers them and the comparison is unreadable. Unfilled, it "
                        "still shows its extent while the two cut/uncut standard contours stay "
                        "legible inside it.")
    p.add_argument("--fom-in-legend", action="store_true",
                   help="append FoM3 and its ratio to the BNT arm in each legend entry. Contour "
                        "AREA is what the eye compares and it is a poor guide to a 3-parameter "
                        "volume, so the ratio is stated rather than left to be estimated.")
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
    # Resolved here rather than inside the style block because the per-arm colour is recorded in
    # _values.csv, which is built before the figure is drawn.
    import paper_contour_style as PCS
    _palette = PCS.palette_name()
    if a.colors is None:
        colors = PCS.colors_for([lab for lab, _, _ in series], _palette)
    else:
        colors = [c.strip() for c in a.colors.split(",")]
    if len(colors) != len(series):
        sys.exit(f"[fatal] need {len(series)} colors, got {len(colors)}")

    print(f"=== {a.stat} triangle | BNT arm = {a.bnt_arm} ({bnt_note}) ===")
    # FoM of every arm first, so the legend can quote the ratio against the BNT arm.
    foms = {key: fom3(load(os.path.join(SAMP, fn))) for _, fn, key in series}
    bnt_fom = foms[bnt_key]
    mcs, rows = [], []
    for (label, fname, key), col in zip(series, colors):
        s = load(os.path.join(SAMP, fname))
        lab = label
        if a.fom_in_legend:
            lab = (f"{label}  (FoM$_3$={foms[key]:.1e})" if key == bnt_key
                   else f"{label}  ({foms[key]/bnt_fom:.0f}$\\times$ tighter)")
        mcs.append(MCSamples(samples=s[:, SUB], names=[PAR_LABELS[i] for i in SUB], label=lab))
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
        PCS.apply(g, _palette)
        g.settings.title_limit_fontsize = 12
        # filled accepts a per-sample list; the BNT arm is the one that swamps the panel.
        filled = ([i != 0 for i in range(len(mcs))] if a.outline_bnt else True)
        # No marker_args on the light figure: the published contour figures pass `markers` and
        # nothing else (plot_contours_three_stats.py:301), so the truth lines take getdist's own
        # thin dashed grey. Forcing solid black here made this figure read differently from the
        # rest even once the palette matched. Dark mode still needs an override to stay visible.
        mk = {"color": "white", "lw": 1.5} if a.dark else None
        g.triangle_plot(mcs, filled=filled, contour_colors=colors,
                        legend_labels=[m.label for m in mcs], legend_loc="upper right",
                        markers={PAR_LABELS[i]: TRUTH[i] for i in SUB},
                        contour_lws=1.1,
                        **({"marker_args": mk} if mk else {}))
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
        "command": shlex.join(sys.argv),   # quoted, so it round-trips through shlex.split
        "git_commit": commit,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mplstyle": ("matplotlib dark_background" if a.dark else "matplotlib default")
                    + " + getdist get_subplot_plotter(width_inch=6); no repo style sheet",
        # Records palette and type sizes, so a repalette is visible in the diff of every figure it
        # touches rather than only in the PNG. The published contour figures all carry this block;
        # this one did not, which is how its private palette went unnoticed.
        **PCS.provenance(_palette),
        "colors_source": "paper_contour_style.colors_for" if a.colors is None else "--colors override",
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
