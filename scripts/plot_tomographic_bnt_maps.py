#!/usr/bin/env python3
"""Appendix map-level figures: standard vs BNT tomographic convergence, noiseless and noisy.

Replaces notebooks/tomographic_maps_bnt.ipynb, which the RAID0 failure left 18% NUL (two
clean stripe holes at the 2.5 MiB and 5 MiB offsets). Only the base64 image outputs were
destroyed; every code cell survived and is reproduced here, with one fix.

THE BUG IN THE PUBLISHED FIGURES. All four bins shared one absolute colour range (kappa in
[-0.01, 0.06] standard, [-0.005, 0.02] BNT). Lensing is CUMULATIVE, so every bin sits at a
different mean, and the mean is large compared with the fluctuations the figure is meant to
show. Measured on the fiducial map:

    bin   mean      sigma     mean/sigma
      1   0.00755   0.00275   2.75
      2   0.01638   0.00464   3.53
      3   0.03610   0.00737   4.90
      4   0.05445   0.00891   6.11

Bin 1's whole fluctuation field lived in the bottom quarter of the bar, where viridis has
little contrast. Bin 4's mean is SIX sigma, so its bright half ran past vmax=0.06 and CLIPPED
to a flat yellow wash -- that is the washed-out look, and it is why bin 4 appeared to have
less structure than bin 1 rather than more. The mean convergence carries no structural
information; it is only the cumulative lensing amplitude.

THE FIX, and only this one: subtract the per-panel mean, then give each ROW one shared
absolute range. Mean subtraction recovers the dynamic range and stops the clipping. Keeping
the range SHARED ACROSS A ROW is what preserves the physics the appendix is about:

  * Standard row -- sigma grows 0.0027 -> 0.0089 across the bins (3.3x). Rendered on a
    shared scale, the row visibly gains contrast left to right: the SAME structures, and
    they keep accumulating. That is the first claim of the appendix.
  * BNT row -- sigma goes 0.0027 -> 0.0017 (0.63x), essentially flat, and the patterns
    differ from bin to bin. Independent structures, no accumulation. Second claim.

Normalising each PANEL by its own sigma would also fix the dynamic range, but it divides out
exactly the accumulation the figure exists to show, so it is deliberately not done.

NO DISPLAY SMOOTHING. Worth knowing what this costs, because it is a real limitation rather
than an oversight. Per-pixel SNR of the noisy maps is 0.19 / 0.31 / 0.50 / 0.61 across the
standard bins and 0.19 / 0.11 / 0.10 / 0.07 across the BNT bins -- below 1 everywhere. At
NSIDE=512 the shape noise dominates every individual pixel, so in the noisy figure the
structure that survives in the high-z standard bins reads only as faint large-scale mottling,
not as crisp structure; the eye has to do the spatial averaging. The standard-vs-BNT contrast
at bin 4 is still a factor of 9 in SNR. Smoothing at FWHM ~35' would lift standard bin 4 to
SNR 1.55 against BNT's 0.18 and make it obvious, at the cost of a visibly smoothed map, and
was rejected as a display choice. If you ever reinstate it, --smooth-px is still wired up and
the caption must then declare it.

  PYTHONNOUSERSITE=1 <jaxili python> scripts/plot_tomographic_bnt_maps.py
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

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import ImageGrid  # noqa: E402
import healpy as hp  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# SIZE THE FIGURE TO ITS PRINTED SIZE. The paper puts these in a figure* (A&A double column,
# 180 mm = 7.087 in) at width=0.95\textwidth, so 6.73 in reaches the page. The first version
# of this script used figsize=(20, 5): LaTeX then scaled it by 0.33, and matplotlib's default
# 10 pt tick labels printed at ~3.5 pt against A&A's 8 pt floor. Sizing at the printed width
# means a point is a point -- what the style sheet says is what the reader sees.
#
# styles/paper_v1.mplstyle is the paper's own style (font 18 / titles 16 / labels 15 /
# ticks 14, sans-serif, tab10) and is NOT the A&A house style: the revision has to sit beside
# figures kept verbatim from the submitted version, so styles/aa.mplstyle is deliberately not
# used here. Those sizes are calibrated for a ~6.9 in figure, which is what the healthy
# siblings measure (bias_vs_area_three_stats 6.90 in, starlet_scale_ell 6.67 in).
FIG_W_IN = 6.73

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPLSTYLE = os.path.join(REPO, "styles", "paper_v1.mplstyle")
CG = "/lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast"

# Cosmologies this figure may be drawn from. Anything else needs the BNT caveat below checked
# again from scratch.
#
# WHY A NON-FIDUCIAL COSMOLOGY IS ALLOWED HERE. This is an illustrative figure, and a clumpier
# universe makes the structure it is illustrating easier to see. But BNT is a GEOMETRIC
# nulling: the matrix is built from the comoving distances to the source bins, so it is only
# correct for the cosmology it was derived in. The matrix is hardcoded in all seven bnt_*
# processing scripts and nothing in the repo recomputes it, so a cosmology with different
# Om / w0 / H0 would be nulled by the WRONG matrix and the "independent structures" claim
# would quietly degrade.
#
# cosmo_131643 is therefore chosen for near-fiducial GEOMETRY and high sigma_8:
#     Om 0.2574 (fiducial 0.26)   w0 -1.009 (-1.0)   H0 65.69 (67.36)   s8 1.0729 (0.84)
# sigma_8 sets the clumpiness and does NOT enter the distances, so the fiducial matrix stays
# valid. Verified empirically rather than assumed -- the standard/BNT variance ratio per bin,
# which is what "the nulling works" means in practice, is
#     fiducial       1.00  2.07  2.69  5.21
#     cosmo_131643   1.00  2.11  2.66  5.16
# i.e. unchanged. The accumulation (3.3x bin1->bin4) and the BNT flatness (0.63x) also match.
# If you add a cosmology here, rerun that check before trusting the figure.
COSMOLOGIES = {
    "fiducial":     (f"{CG}/fiducial/cosmo_fiducial/perm_0000/"
                     "projected_probes_maps_nobaryons512.h5",
                     dict(Om=0.2600, s8=0.8400, w0=-1.000, H0=67.36)),
    "cosmo_131643": (f"{CG}/grid/cosmo_131643/perm_0000/"
                     "projected_probes_maps_nobaryons512.h5",
                     dict(Om=0.2574, s8=1.0729, w0=-1.009, H0=65.69)),
}

# The BNT matrix used throughout the project. Row 0 is the identity on bin 1, so BNT bin 1
# and standard bin 1 are the SAME field -- the figure is not four independent transforms,
# and the noisy BNT bin 1 legitimately still shows whatever bin 1 shows.
BNT_MATRIX = np.array([
    [1.,         0.,          0.,        0.],
    [-1.,        1.,          0.,        0.],
    [0.4521097, -1.4521097,   1.,        0.],
    [0.,         0.25127807, -1.251278,  1.],
])

NSIDE = 512
SIGMA_E = 0.26
N_GAL = 6.75          # arcmin^-2
NOISE_SEED = 1234     # fixed so the noisy figure is reproducible


def main():
    p = argparse.ArgumentParser()
    # Patch geometry defaults to the published one: reso=10', xsize=200 -> 33.3 deg. The
    # large field is deliberate. NSIDE=512 is low resolution (6.87' pixels), so zooming in
    # buys pixel noise rather than detail; a wide field shows many structures instead.
    p.add_argument("--lon", type=float, default=180.0, help="patch centre longitude (deg)")
    p.add_argument("--lat", type=float, default=30.0, help="patch centre latitude (deg)")
    p.add_argument("--reso", type=float, default=10.0, help="arcmin per pixel")
    p.add_argument("--xsize", type=int, default=200, help="pixels per side")
    p.add_argument("--klim", type=float, default=2.0,
                   help="colour range = klim x (largest bin sigma in that row)")
    p.add_argument("--smooth-px", type=float, default=0.0,
                   help="Gaussian display smoothing in pixels. 0 = none (the default). "
                        "If set, say so in the caption.")
    # Default is the FIDUCIAL cosmology. cosmo_131643 (sigma_8 = 1.073) was tried and is a
    # valid alternative -- the nulling check in COSMOLOGIES passes -- but the fiducial patch
    # was preferred on how its structure reads, which is the only criterion that matters for
    # an illustrative figure.
    p.add_argument("--cosmo", default="fiducial", choices=sorted(COSMOLOGIES),
                   help="which CosmoGrid cosmology to draw the patch from. See COSMOLOGIES "
                        "for why only near-fiducial-geometry entries are allowed.")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="multiply the shape-noise amplitude by this. 1.0 = the Euclid-like "
                        "level the analysis actually uses. Below 1 is a VISUALISATION choice "
                        "and the caption must give the effective n_gal it corresponds to.")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--outdir", default="outputs/plots/tomographic_bnt_maps")
    a = p.parse_args()

    fid_path, cosmo_par = COSMOLOGIES[a.cosmo]
    deg = a.xsize * a.reso / 60.0
    # Noise amplitude scales as 1/sqrt(n_gal), so scaling the noise by f is exactly a survey
    # with n_gal / f^2. Quote that number rather than the scale factor -- it is the physically
    # meaningful statement and a reader can judge it.
    eff_ngal = N_GAL / a.noise_scale**2
    outdir = os.path.join(REPO, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f"cosmology {a.cosmo}  " + "  ".join(f"{k}={v}" for k, v in cosmo_par.items()))
    print(f"patch centre=({a.lon},{a.lat})  {deg:.1f} deg  {a.reso}'/px  "
          f"({a.xsize}x{a.xsize})  smoothing={a.smooth_px or 'none'}")
    print(f"noise scale {a.noise_scale:g}  ->  effective n_gal = {eff_ngal:.1f} /arcmin^2"
          + ("   [Euclid-like]" if a.noise_scale == 1.0 else "   [REDUCED for visibility]"))

    with h5py.File(fid_path, "r") as f:
        kgs = np.array([np.array(f[f"kg/stage3_lensing{i}"]) for i in (1, 2, 3, 4)],
                       dtype=np.float64)

    # Shape noise on the STANDARD maps, then BNT applied -- matching the paper text. BNT is
    # linear, so this is identical to adding BNT-mixed noise, which is the point: the mixing
    # is what inflates the noise while the differencing removes signal.
    rng = np.random.default_rng(NOISE_SEED)
    sigma_pix = a.noise_scale * SIGMA_E / np.sqrt(
        N_GAL * hp.nside2pixarea(NSIDE, degrees=True) * 3600)
    kgs_noisy = kgs + rng.normal(0.0, sigma_pix, size=kgs.shape)

    fields = {("standard", "noiseless"): kgs,
              ("standard", "noisy"):     kgs_noisy,
              ("bnt",      "noiseless"): BNT_MATRIX @ kgs,
              ("bnt",      "noisy"):     BNT_MATRIX @ kgs_noisy}

    def project(m):
        patch = np.asarray(hp.gnomview(m, rot=(a.lon, a.lat), reso=a.reso, xsize=a.xsize,
                                       return_projected_map=True, no_plot=True))
        return gaussian_filter(patch, a.smooth_px) if a.smooth_px else patch

    # Mean-subtracted patches. Everything downstream uses these.
    D = {k: np.array([(lambda x: x - x.mean())(project(v[i])) for i in range(4)])
         for k, v in fields.items()}

    # One shared limit per row, set by the widest bin in that row.
    LIM = {k: a.klim * max(D[k][i].std() for i in range(4)) for k in D}

    rows = []
    for (basis, noise), arrs in D.items():
        for i in range(4):
            clean_sigma = float(D[(basis, "noiseless")][i].std())
            total_sigma = float(arrs[i].std())
            noise_sigma = float(np.sqrt(max(total_sigma**2 - clean_sigma**2, 0.0)))
            rows.append(dict(
                basis=basis, noise=noise, bin=i + 1,
                # n_seeds is 1 and stated rather than omitted. This is ONE realisation of one
                # patch, not an average, and the publish gate is right to ask: an absent seed
                # count reads as "pooled" beside the seed-pooled contour figures.
                n_seeds=1, realisation="perm_0000",
                sigma=total_sigma, sigma_signal=clean_sigma, sigma_noise=noise_sigma,
                snr=(clean_sigma / noise_sigma) if noise_sigma > 0 else float("inf"),
                colour_limit=float(LIM[(basis, noise)]),
                min=float(arrs[i].min()), max=float(arrs[i].max())))

    print("\nsigma of the mean-subtracted field (the accumulation the figure shows):")
    for basis in ("standard", "bnt"):
        s = [r["sigma"] for r in rows if r["basis"] == basis and r["noise"] == "noiseless"]
        print(f"  {basis:9s} " + "  ".join(f"b{i+1}={v:.5f}" for i, v in enumerate(s))
              + f"   -> bin4/bin1 = {s[3]/s[0]:.2f}x")
    print("\nper-pixel SNR once shape noise is added:")
    for basis in ("standard", "bnt"):
        s = [r["snr"] for r in rows if r["basis"] == basis and r["noise"] == "noisy"]
        print(f"  {basis:9s} " + "  ".join(f"b{i+1}={v:4.2f}" for i, v in enumerate(s)))

    # One single-row PDF per (basis, noise), reusing the four filenames the paper already
    # \includegraphics, so the .tex needs no change.
    NAMES = {("standard", "noiseless"): "noiseless_tomographic_maps_flatsky",
             ("bnt",      "noiseless"): "noiseless_bnt_transformed_maps_flatsky",
             ("standard", "noisy"):     "noisy_tomographic_maps_flatsky",
             ("bnt",      "noisy"):     "noisy_bnt_transformed_maps_flatsky"}
    extent = [-deg / 2, deg / 2, -deg / 2, deg / 2]
    # LAYOUT, solved rather than guessed. The panels are square, so the figure height is not
    # free: it follows from how wide each panel ends up once the margins and the colorbar are
    # taken out of FIG_W_IN. Getting this wrong is what left a tall colorbar beside short maps
    # in the first place -- ImageGrid then centres the square panels in whatever vertical slot
    # it was given, and any excess shows up as a gap above and below them.
    # M_B has to hold the x tick labels AND the shared "deg" label below them; at 0.175 the
    # two collided once the solved height came out short.
    # M_L likewise holds the y tick labels AND the rotated "deg" beside them.
    M_L, M_R, M_B, M_T = 0.075, 0.115, 0.260, 0.130   # figure fractions
    AX_PAD, CB_PAD, CB_FRAC = 0.13, 0.13, 0.045       # inches, inches, fraction of a panel
    rect_w, rect_h = 1.0 - M_L - M_R, 1.0 - M_B - M_T
    # rect_w * FIG_W = 4*w + 3*AX_PAD + CB_PAD + CB_FRAC*w
    panel_in = (rect_w * FIG_W_IN - 3 * AX_PAD - CB_PAD) / (4.0 + CB_FRAC)
    fig_h = panel_in / rect_h        # so the square panel exactly fills the rect's height
    GRID_RECT = [M_L, M_B, rect_w, rect_h]
    # M_R is the widest of the four margins on purpose: it carries the colorbar's tick labels
    # AND its rotated axis label, which sit OUTSIDE the rect. Too small and both are clipped.

    # TYPE SIZES ARE SCALED DOWN FROM paper_v1 FOR THIS FIGURE, deliberately. The style's
    # 18/16/15/14 pt is calibrated for figures whose panels are several inches across
    # (bias_vs_area_three_stats is 6.90 x 5.65 in for a handful of panels). Four square maps
    # across a 6.73 in figure* gives ~1.4 in per panel, and at that size the style's type
    # swamps the data -- rendered once at full paper_v1 sizes, the titles and tick labels took
    # more area than the maps. The sizes below stay above the A&A floor (tick 8 / label 9 /
    # annotation 8 pt), so they are legible at print size; they are simply proportionate to a
    # dense panel row. Family, colour cycle and the vector/font-embedding settings still come
    # from paper_v1, so the figure remains of a piece with the rest of the paper.
    DENSE = {"axes.titlesize": 9, "axes.labelsize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 9,
             # paper_v1 sets savefig.bbox: tight, which retrims the canvas on save and so
             # silently overrides figsize -- the figure came out 173.8 mm against the 171 mm
             # asked for. Exact width is the whole point here, so turn it off and let
             # constrained_layout do the fitting instead.
             "savefig.bbox": "standard"}
    # Plot in units of 1e-3 so the colorbar ticks are short 2-digit numbers instead of
    # matplotlib's floating "1e-2" offset box, which collides with the top panel.
    SCALE, SCALE_TEX = 1e3, r"10^{-3}"

    with plt.style.context([MPLSTYLE, DENSE]):
        for (basis, noise), stem in NAMES.items():
            lim = LIM[(basis, noise)] * SCALE
            fig = plt.figure(figsize=(FIG_W_IN, fig_h))
            # ImageGrid rather than subplots + fig.colorbar. With square panels
            # (aspect='equal') the Axes box is shrunk to fit the image, but fig.colorbar sizes
            # itself to the axes' ALLOCATED slot, so the bar ended up taller than the maps.
            # ImageGrid ties the colorbar to the image height by construction. The rect leaves
            # room for the titles and the shared axis labels, which ImageGrid does not
            # auto-fit the way constrained_layout would.
            grid = ImageGrid(fig, GRID_RECT,
                             nrows_ncols=(1, 4), axes_pad=AX_PAD,
                             share_all=True, label_mode="L",
                             cbar_location="right", cbar_mode="single",
                             cbar_size=f"{CB_FRAC * 100:g}%", cbar_pad=CB_PAD)
            for i, ax in enumerate(grid):
                # rasterized: the panel is a 200x200 image, and leaving it vector bloats the
                # PDF with a quarter-million rectangles. Text, axes and ticks stay vector.
                im = ax.imshow(D[(basis, noise)][i] * SCALE, origin="lower", extent=extent,
                               cmap=a.cmap, vmin=-lim, vmax=lim, interpolation="nearest",
                               rasterized=True)
                ax.set_title(("Bin " if basis == "standard" else "BNT bin ") + str(i + 1))
                # 4 ticks/axis: at ~1.4 in per panel the default density collides.
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
            # One shared pair of axis labels rather than "deg" repeated under all four panels.
            fig.supxlabel("deg", fontsize=DENSE["axes.labelsize"], y=0.02)
            fig.supylabel("deg", fontsize=DENSE["axes.labelsize"], x=0.010)
            sym = (r"\kappa" if basis == "standard" else r"\kappa_\mathrm{BNT}")
            lbl = rf"$({sym} - \langle {sym} \rangle)\;/\;{SCALE_TEX}$"
            grid.cbar_axes[0].colorbar(im, label=lbl)
            base = os.path.join(outdir, stem)
            # No bbox_inches="tight": it retrims the canvas and would undo the exact printed
            # width this figure is sized for. constrained_layout already fits the elements.
            fig.savefig(f"{base}.pdf", transparent=True, dpi=300)
            fig.savefig(f"{base}.png", transparent=True, dpi=300)
            plt.close(fig)
            print(f"  wrote {stem}.pdf/.png   (colour range +/-{lim:.3g}e-3)")

    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "figure": "PER_FIGURE",   # replaced per stem below
        "generator": "scripts/plot_tomographic_bnt_maps.py",
        "command": shlex.join(sys.argv),
        "git_commit": commit,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "replaces": "notebooks/tomographic_maps_bnt.ipynb (18% NUL after the RAID0 failure; "
                    "code cells intact, image outputs destroyed)",
        "mplstyle": "styles/paper_v1.mplstyle, with the type sizes scaled down for a dense "
                    "4-panel row (titles/labels 9 pt, ticks 8 pt) and savefig.bbox forced to "
                    "'standard'. An earlier version of this file claimed the style sheet "
                    "'would change nothing visible here' and ran on matplotlib defaults -- "
                    "that was wrong: it sets every text size in the figure.",
        "figure_sizing": {
            "width_mm": 170.9, "width_in": FIG_W_IN,
            "rule": "A&A figure* is 180 mm; the paper includes these at width=0.95\\textwidth, "
                    "so 0.95 x 180 = 171 mm reaches the page. Sizing the PDF at that width "
                    "makes LaTeX scale it 1.00, so 8 pt tick text prints as 8 pt.",
            "previous_bug": "figsize=(20, 5) -> LaTeX scaled by 0.33 -> default 10 pt tick "
                            "labels printed at ~3.5 pt, against the A&A 8 pt floor.",
            "note": "figure-polish's check_figure.py FLAGS 170.9 mm as not matching a nominal "
                    "A&A column. That is expected: it assumes width=\\textwidth. If the .tex "
                    "ever switches to width=\\textwidth, set FIG_W_IN = 7.087.",
            "type_sizes_pt": {"title": 9, "axis_label": 9, "tick_label": 8},
            "aa_minima_pt": {"tick_labels": 8, "axis_labels": 9, "annotations": 8},
        },
        "grayscale_safe": "viridis luminance rises monotonically over 0.084-0.870, so the "
                          "panels keep their ordering in A&A's grayscale print; no information "
                          "is carried by hue alone.",
        "conventions": {
            "scales_included": "NOT APPLICABLE -- this is a map-level figure. No wavelet "
                               "decomposition, no multipole cut and no scale selection is "
                               "applied; every panel is the full projected convergence field "
                               "at the native NSIDE=512 resolution.",
            "quantity_plotted": "kappa - <kappa> per panel (the per-panel mean is removed; "
                                "see colour_scale.why_mean_subtracted)",
            "tomographic_bins": "stage3_lensing1..4, increasing in source redshift",
        },
        "input_map": fid_path,
        "simulation": f"CosmoGridV1 stage3_forecast, {a.cosmo}, perm_0000, nobaryons, "
                      f"NSIDE={NSIDE}",
        "cosmology": {"name": a.cosmo, **cosmo_par,
                      "why": ("fiducial" if a.cosmo == "fiducial" else
                              "Illustrative: sigma_8 = 1.073 (1.28x fiducial) makes the "
                              "structure clearer. Om/w0/H0 are within ~1-2.5% of fiducial, so "
                              "the hardcoded BNT matrix -- which depends on comoving distances "
                              "and NOT on sigma_8 -- remains valid. Verified: standard/BNT "
                              "variance ratio per bin is 1.00/2.11/2.66/5.16 here against "
                              "1.00/2.07/2.69/5.21 for fiducial.")},
        "patch": {"centre_lonlat_deg": [a.lon, a.lat], "size_deg": deg,
                  "reso_arcmin_per_pixel": a.reso, "pixels": a.xsize,
                  "projection": "healpy gnomview (gnomonic)"},
        "patch_selection": "Centres scanned on a 30x30 deg grid, ranked by peak/sigma of the "
                           "bin-1 fluctuation field. (180,30) scores 59.1 (skew 10.4); the "
                           "notebook's (90,90) is the north pole and scores 19.4.",
        "colour_scale": {
            "quantity": "kappa - <kappa>, per panel",
            "range_per_row": {f"{b}_{n}": float(v) for (b, n), v in LIM.items()},
            "rule": f"+/- {a.klim} x (largest bin sigma in that row)",
            "why_mean_subtracted": "Lensing is cumulative, so each bin has a different mean; "
                                   "on a shared absolute scale bin 1 was squeezed into the "
                                   "dark end and bin 4 clipped past vmax. The mean carries no "
                                   "structural information.",
            "why_shared_per_row": "Preserves the amplitude ratio between bins, which IS the "
                                  "accumulation (standard sigma grows 3.3x bin1->bin4) and "
                                  "its absence under BNT (0.63x). Per-panel normalisation "
                                  "would divide exactly that out.",
        },
        "display_smoothing": {"gaussian_sigma_px": a.smooth_px,
                              "applied": bool(a.smooth_px)},
        "noise": {"sigma_e": SIGMA_E, "n_gal_arcmin2_nominal": N_GAL, "nside": NSIDE,
                  "seed": NOISE_SEED, "scale_factor": a.noise_scale,
                  "effective_n_gal_arcmin2": eff_ngal,
                  "order": "noise added to the STANDARD maps, then BNT applied",
                  "note": ("Euclid-like, as used in the analysis." if a.noise_scale == 1.0
                           else f"REDUCED by {a.noise_scale:g}x for visibility, equivalent to "
                                f"n_gal = {eff_ngal:.1f}/arcmin^2 rather than {N_GAL}. The "
                                f"ANALYSIS uses the full Euclid-like noise; this affects the "
                                f"figure only and must be stated in the caption.")},
        "bnt_matrix": BNT_MATRIX.tolist(),
        "caveats": [
            "BNT row 0 is the identity on bin 1, so 'BNT bin 1' and standard 'Bin 1' are the "
            "SAME field. Not four independent transforms.",
            "NO display smoothing. Per-pixel SNR is below 1 in every bin of the noisy maps "
            "(standard 0.19/0.31/0.50/0.61, BNT 0.19/0.11/0.10/0.07), so surviving structure "
            "in the high-z standard bins reads as faint large-scale mottling rather than "
            "crisp structure. The standard-vs-BNT SNR ratio at bin 4 is still ~9x.",
            "The colour range differs between the noiseless and the noisy figure (each row "
            "sets its own), so the two are NOT directly comparable in amplitude. They are "
            "comparable in texture and in within-row contrast.",
            "One realisation, one patch. Illustrative, not a statistical statement.",
        ],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "healpy": hp.__version__, "matplotlib": matplotlib.__version__},
    }

    # PER-FIGURE sidecars. scripts/paper/figures.py publish resolves <stem>_values.csv and
    # <stem>_provenance.json beside the PDF, so a single combined pair for all four figures
    # would fail the publish gate ("no _values.csv"). Each figure also deserves provenance
    # describing only what it shows -- a reader opening one slug should not have to filter
    # three other figures' rows out of the table.
    for (basis, noise), stem in NAMES.items():
        base = os.path.join(outdir, stem)
        mine = [r for r in rows if r["basis"] == basis and r["noise"] == noise]
        with open(f"{base}_values.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mine[0]))
            w.writeheader()
            w.writerows(mine)
        pf = dict(prov)
        pf["figure"] = stem
        pf["shows"] = (f"{'standard' if basis == 'standard' else 'BNT'} basis, "
                       f"{noise}, tomographic bins 1-4")
        pf["colour_range"] = [-float(LIM[(basis, noise)]), float(LIM[(basis, noise)])]
        pf["series"] = mine
        # The companion figures, so a slug is traceable to the set it belongs to.
        pf["companion_figures"] = {f"{b}_{n}": s for (b, n), s in NAMES.items()
                                   if (b, n) != (basis, noise)}
        blob = json.dumps(pf, indent=2)
        tmp = f"{base}_provenance.json.tmp"
        with open(tmp, "w") as fh:
            fh.write(blob)
        json.loads(open(tmp).read())          # never leave a truncated sidecar beside a good PDF
        os.replace(tmp, f"{base}_provenance.json")

    # Combined table too -- convenient for quoting the accumulation and SNR numbers in the text.
    with open(os.path.join(outdir, "tomographic_bnt_maps_values.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {outdir}/  (4 figures, each + _values.csv + _provenance.json)")


if __name__ == "__main__":
    main()
