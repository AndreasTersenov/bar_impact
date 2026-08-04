#!/usr/bin/env python
"""Full-sky PS-vs-HOS null triangle plots WITHOUT the monopole (PS ell>=2, not ell>=0).

The full-sky `withl0_FS_*` plots carried a PS "all low modes" probe starting at
ell=0. On the full sky that band includes the ell=0 mass-sheet monopole, so the
contour collapses (sigma(S8) ~ 0.006 -- an artifact, not real constraining
power). This rebuilds the two full-sky plots with the legitimate low-mode PS
starting at ell=2 (excludes ell=0,1): sigma(S8) ~ 0.011, which sits just inside
the ell=37 recovery (0.014) -- a real low-mode gain with no monopole leakage.

Probes (nobaryons_vs_nobaryons null, in Omega_m / S8 / w0):
  PS ell100 (paper) | PS ell37 (recovered) | PS ell2 (low modes, no monopole)
  | l1-norm | peaks [provisional]
Full-ell pairs PS upper-cut 1024 with HOS scales1234; baryon-safe pairs upper-cut
400 with scales234 (drops the finest, most baryon-sensitive wavelet scale).

Full-sky peaks at scales1234 do not exist, so the full-ell plot is 4-probe;
the baryon-safe peaks (scales234) do exist, so that plot is 5-probe.

Outputs: outputs/diagnostics/lmin_compare/ps_l0/withl2_FS_{fulll,baryonsafe}.png
The original withl0_FS_* plots are left untouched for side-by-side comparison.
"""
import glob

import matplotlib
import numpy as np

matplotlib.use("Agg")
from getdist import MCSamples, plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
OUTDIR = f"{REPO}/outputs/diagnostics/lmin_compare/ps_l0"

# Full-sky posteriors are spread across several directories; search in order.
SEARCH_DIRS = [
    f"{REPO}/outputs/diagnostics/fullsky_baseline/samples",   # PS l100/l37 at upper-cut 1024
    f"{REPO}/outputs/diagnostics/lmin_compare/fullsky_l400",  # PS l100/l37 at 400; peaks scales234
    f"{REPO}/outputs/diagnostics/lmin_compare/ps_l0",         # PS l2 (no-monopole low modes)
    f"{REPO}/outputs/samples",                                # l1-norm; peaks
]

NAMES = ["Om", "S8", "w0"]
LABELS = [r"\Omega_m", r"\sigma_8", "w_0"]
TRUTH = [0.26, 0.84, -1.0]

# Fixed colour per probe so the legend/contour mapping stays correct even when a
# probe (e.g. full-ell peaks) is missing and the plot has fewer curves.
LABEL_PS100 = r"PS $\ell$100 (paper)"
LABEL_PS37 = r"PS $\ell$37 (recovered)"
LABEL_PS2 = r"PS $\ell$2 (low modes, no monopole)"
LABEL_L1 = r"$\ell_1$-norm"
LABEL_PEAKS = "peaks [provisional]"
COLOR = {
    LABEL_PS100: "#bbbbbb",
    LABEL_PS37: "#2ca02c",
    LABEL_PS2: "#8B4513",
    LABEL_L1: "#1f77b4",
    LABEL_PEAKS: "#d62728",
}


def load_first(pattern):
    """Return (path, Om/S8/w0 samples) for the first match across SEARCH_DIRS."""
    for directory in SEARCH_DIRS:
        matches = sorted(glob.glob(f"{directory}/{pattern}"))
        if matches:
            return matches[0], np.load(matches[0])[:, :3]
    return None, None


def ps_pattern(lower_cut, upper_cut):
    return (
        f"posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_"
        f"bins1234_l{lower_cut}-{upper_cut}_r10_noisy_s0.26*.npy"
    )


def hos_pattern(prefix, scales):
    return (
        f"posterior_samples_{prefix}nobaryons_vs_nobaryons_"
        f"bins1234_scales{scales}_noisy_s0.26_new_normalization*npe.npy"
    )


def build(upper_cut, scales, regime, out_name):
    candidates = [
        (LABEL_PS100, ps_pattern(100, upper_cut)),
        (LABEL_PS37, ps_pattern(37, upper_cut)),
        (LABEL_PS2, ps_pattern(2, upper_cut)),
        (LABEL_L1, hos_pattern("", scales)),
        (LABEL_PEAKS, hos_pattern("pc_", scales)),
    ]
    print(f"[{regime}]  upper-cut {upper_cut}, scales{scales}")
    loaded = []
    for label, pattern in candidates:
        path, samples = load_first(pattern)
        if samples is None:
            print(f"    MISSING  {label}")
            continue
        print(
            f"    {label:34s} sigma(S8)={samples[:, 1].std():.4f}"
            f"  <- {path.split('/')[-1]}"
        )
        loaded.append((label, samples))

    mc = [
        MCSamples(samples=s, names=NAMES, labels=LABELS, label=label)
        for label, s in loaded
    ]
    colors = [COLOR[label] for label, _ in loaded]

    plotter = plots.get_subplot_plotter(width_inch=8.0)
    plotter.settings.alpha_filled_add = 0.45
    plotter.settings.legend_fontsize = 8
    plotter.triangle_plot(
        mc, filled=True, contour_colors=colors, markers=dict(zip(NAMES, TRUTH))
    )
    plotter.fig.suptitle(
        f"FULL-SKY (f_sky 1.00) {regime}; PS ℓ2 no-monopole; peaks provisional",
        fontsize=9,
    )
    out_path = f"{OUTDIR}/{out_name}"
    plotter.export(out_path)
    print(f"  -> {out_path}  ({len(mc)} probes)\n")


if __name__ == "__main__":
    build("1024", "1234", r"full-$\ell$", "withl2_FS_fulll.png")
    build("400", "234", "baryon-safe", "withl2_FS_baryonsafe.png")
