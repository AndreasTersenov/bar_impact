r"""Measure the multipole response W_j(ell)^2 of each starlet scale, for the paper appendix.

WHAT THIS ESTABLISHES. The L1-norm and peak counts are computed on the undecimated isotropic
starlet transform (CMRStarlet, nscale=5 -> 4 wavelet/detail bands + 1 coarse band). Each scale j
is a fixed isotropic linear filter W_j(ell) in harmonic space, so "wavelet scale 2" corresponds to
a definite band of multipoles. This script measures that correspondence, which is what licenses
every scale-cut statement in the paper: dropping the finest wavelet drops a specific, measured
ell range rather than an abstract "small scales".

TWO METHODS, and the default is the exact one.

  --method delta  (DEFAULT, deterministic, no sampling noise)
      A single-pixel map is WHITE: for a delta at n-hat, a_lm = Omega_pix Y*_lm(n-hat), so
      sum_m |a_lm|^2 = Omega_pix^2 (2l+1)/(4pi) and C_ell = Omega_pix^2/(4pi), independent of ell.
      Push it through the transform and take the ratio of output to input C_ell per multipole:

          W_j(ell)^2 = C_ell[coef_j] / C_ell[input]

      Both sides are deterministic, so the ratio carries no Monte Carlo error at any ell. Taking
      the ratio -- rather than assuming the input is exactly flat -- also divides out the pixel
      window, which does bite near the Nyquist multipole.

      Several delta positions are averaged, not for noise (there is none) but because one
      position probes any residual pixelisation anisotropy in a single direction. The spread
      across positions is saved as `delta_spread`: for an exactly isotropic filter it is zero,
      so it is a genuine systematic check rather than decoration.

  --method noise  (the original: NREAL white-noise maps, kept for cross-validation)
      Correct in expectation but noisy where there are few modes: the fractional error on C_ell
      goes as sqrt(2 / ((2l+1) NREAL)), ~10% at ell=2 for NREAL=40. That is exactly the low-ell
      scatter on the coarse band in the original figure -- an artefact of the estimator, not
      structure in the filter. The original hid it by binning in log-ell; the delta method
      removes the need to bin at all.

  --method both   runs both and reports their maximum fractional disagreement, which is the
      validation that the deterministic method measures the same filter.

Run with cosmostat on the path (jaxili has healpy + numpy; pycs is a source checkout):

  PYTHONNOUSERSITE=1 PYTHONPATH=/lustre/fswork/projects/rech/nzu/ulx34io/cosmostat_src \
    /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python \
    scripts/diagnostics/starlet_scale_ell.py [--method delta|noise|both]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pycs.sparsity.mrs.mrs_starlet import CMRStarlet  # noqa: E402

NSIDE = 512
NSCALE = 5
LMAX = 1535                       # 3*nside - 1, the map's Nyquist multipole
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "outputs", "diagnostics", "starlet_scale_ell")

LABELS = [f"wavelet {j}" for j in range(NSCALE - 1)] + ["coarse"]
# Okabe-Ito, matching scripts/paper_contour_style.py so the appendix figure sits with the rest.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "unknown"


def transform(m):
    """One starlet transform of map `m`; returns its nscale coefficient maps."""
    C = CMRStarlet()
    C.init_starlet(NSIDE, nscale=NSCALE)
    C.transform(m)
    return [C.coef[j] for j in range(NSCALE)]


def measure_delta(ndelta):
    """Exact W_j(ell)^2 from single-pixel inputs. Returns (W2, fractional spread)."""
    npix = hp.nside2npix(NSIDE)
    # Spread the positions over the sphere rather than clustering them.
    pix = [int(round(k * npix / ndelta)) % npix for k in range(ndelta)]
    per = []
    for i, p in enumerate(pix):
        m = np.zeros(npix)
        m[p] = 1.0
        cl_in = hp.anafast(m, lmax=LMAX)
        coefs = transform(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            per.append(np.array([hp.anafast(c, lmax=LMAX) / cl_in for c in coefs]))
        print(f"  delta {i+1}/{ndelta} (pixel {p})", flush=True)
    per = np.array(per)                      # (ndelta, NSCALE, LMAX+1)
    W2 = per.mean(0)
    spread = per.std(0) / np.maximum(np.abs(W2), 1e-300)
    return W2, spread


def measure_noise(nreal, seed=1):
    """W_j(ell)^2 by Monte Carlo over white-noise maps. Kept for cross-validation."""
    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(seed)
    acc = np.zeros((NSCALE, LMAX + 1))
    for r in range(nreal):
        wn = rng.normal(0, 1, npix)
        for j, c in enumerate(transform(wn)):
            acc[j] += hp.anafast(c, lmax=LMAX)
        if (r + 1) % 10 == 0:
            print(f"  {r+1}/{nreal} realizations", flush=True)
    return acc / nreal


def ranges(W2n):
    """Peak and half-power multipole range per scale, from the UNBINNED response."""
    out = []
    for j in range(NSCALE):
        w = W2n[j]
        pk = int(np.argmax(w))
        above = np.where(w >= 0.5)[0]
        lo, hi = (int(above.min()), int(above.max())) if len(above) else (0, 0)
        out.append(dict(scale=j, type=("coarse" if j == NSCALE - 1 else f"wav{j}"),
                        ell_peak=pk, ell_half_lo=lo, ell_half_hi=hi))
    return out



def make_figure(W2, tbl, a):
    """Draw and save. Split out so --replot can reuse it without recomputing the response.

    Type is sized to the CANVAS, not fixed: at one A&A column (width 7) a 5-entry single-row
    legend and the two-line annotation both overflow the axes, which is what made the narrow
    variant unusable. Everything below scales with `a.width` against the 14-inch double-column
    reference.
    """
    import numpy as np
    plt.style.use(os.path.join(REPO, "styles", "paper_v1.mplstyle"))
    fig, ax = plt.subplots(figsize=(a.width, a.width * a.aspect))

    # TYPE FOLLOWS THE CANVAS WIDTH, keyed so that the default reproduces paper_v1's own
    # sizes exactly (labels 15, ticks 14, legend 13 at W=7.0). Previously only the legend and
    # the annotation scaled while the axis labels and ticks were inherited fixed, so a
    # narrowed render kept full-size axis type and overflowed. Now every text element moves
    # together and --width alone is enough to retarget the figure.
    scale = a.width / 7.0                     # 1.0 at the repo's one-column canvas
    lab_fs = 15.0 * scale
    tick_fs = 14.0 * scale
    leg_fs = 13.0 * scale
    ncol = a.legend_ncol or (5 if a.width >= 11.0 else 3)
    rows = int(np.ceil(5 / ncol))
    # Headroom for however many legend rows there are, so it never sits on a curve.
    ax.set_ylim(0, 1.0 + 0.16 * rows + 0.06)

    for j in range(NSCALE):
        ax.semilogx(ell_axis()[1:], W2[j][1:], color=COLORS[j], lw=2.4 * max(0.75, scale),
                    label=LABELS[j])

    # The shaded band over wav0's range and its "dropped by the baryon-safe cut" annotation
    # were removed: this figure's job is to report the measured scale-to-multipole mapping,
    # and which scales a given analysis discards is an assertion about the cut, made in the
    # text and in the scale-cut figures. Carrying it here also dated the figure to one
    # particular cut (scales234). The measurement itself is unchanged.

    ax.set_xlim(2, LMAX)
    ax.set_xlabel(r"multipole $\ell$", fontsize=lab_fs)
    ax.set_ylabel(r"normalised response $W_j(\ell)^2/\max$", fontsize=lab_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.legend(frameon=False, ncol=ncol, loc="upper center", fontsize=leg_fs,
              columnspacing=1.1, handlelength=1.5, borderaxespad=0.2)
    ax.grid(alpha=0.25, ls=":")
    if a.top_axis:
        secax = ax.secondary_xaxis("top",
                                   functions=(lambda l: 10800.0 / np.clip(l, 1e-6, None),
                                              lambda t: 10800.0 / np.clip(t, 1e-6, None)))
        secax.set_xlabel(r"angular scale $10800/\ell$ [arcmin]", fontsize=lab_fs)
        secax.tick_params(labelsize=tick_fs)
    fig.tight_layout()
    stem = os.path.join(OUT, "starlet_scale_ell" + (f"_{a.tag}" if a.tag else ""))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    print(f"figure: {stem}.png/.pdf  ({a.width:.1f}x{a.width*a.aspect:.1f} in, "
          f"legend {ncol} col, fs {leg_fs:.0f})")


def ell_axis():
    return np.arange(LMAX + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("delta", "noise", "both"), default="delta")
    ap.add_argument("--ndelta", type=int, default=8)
    ap.add_argument("--nreal", type=int, default=40)
    # Repo canvas convention (see plot_fom_vs_area / plot_scaling_vs_area): W=7.0 is one
    # A&A column, W=14.0 is two. Both are 2x the physical width because paper_v1 fonts are
    # ~2x the A&A ones. The default is one column at 0.62 aspect -- the published choice.
    # Text size and legend columns follow the width (see make_figure); the figure was
    # first drafted at 9.0 with 14-inch type, which overflowed the axes when narrowed.
    ap.add_argument("--width", type=float, default=7.0,
                    help="canvas width in inches: 7.0 single column, 14.0 double column")
    ap.add_argument("--aspect", type=float, default=0.62, help="height / width")
    ap.add_argument("--replot", default=None, metavar="NPZ",
                    help="rebuild the figure from a saved *_data.npz instead of recomputing. "
                         "The response does not depend on the canvas, so layout iteration "
                         "should not cost 8 spherical-harmonic transforms each time.")
    ap.add_argument("--legend-ncol", type=int, default=0,
                    help="0 = choose by width: one row when there is room, two when there is "
                         "not. A 5-entry single row does not fit one A&A column.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--top-axis", action="store_true",
                    help="add the 10800/ell angular-scale axis on top. Off for the "
                         "paper: the conversion is conventional and belongs in the "
                         "caption, qualified.")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.replot:
        z = np.load(a.replot, allow_pickle=True)
        W2, raw = z["W2"], z["raw"]
        spread = z["delta_spread"] if "delta_spread" in z else None
        validation, tbl = {}, ranges(W2)
        print(f"[replot] figure only, from {a.replot}")
        make_figure(W2, tbl, a)
        return

    W2d = spread = W2_noise = None
    if a.method in ("delta", "both"):
        print(f"[delta] exact response from {a.ndelta} single-pixel inputs")
        W2d, spread = measure_delta(a.ndelta)
    if a.method in ("noise", "both"):
        print(f"[noise] Monte Carlo over {a.nreal} white-noise maps")
        W2_noise = measure_noise(a.nreal)

    raw = W2d if W2d is not None else W2_noise
    W2 = raw / raw.max(axis=1, keepdims=True)          # peak-normalised per scale

    validation = {}
    if a.method == "both":
        # Compare only where the response is measurable; in the far tails both are ~0 and the
        # ratio is dominated by each estimator's own floor.
        n2 = W2_noise / W2_noise.max(axis=1, keepdims=True)
        m = W2 > 0.05
        rel = np.abs(n2[m] - W2[m]) / W2[m]
        validation = dict(max_rel_diff=float(rel.max()), median_rel_diff=float(np.median(rel)))
        print(f"\n[validate] |noise-delta|/delta where response>5%: "
              f"max {rel.max():.3%}, median {np.median(rel):.3%}")
    if spread is not None:
        m = W2 > 0.05
        validation.update(isotropy_max=float(spread[m].max()),
                          isotropy_median=float(np.median(spread[m])))
        print(f"[isotropy] delta-position spread where response>5%: "
              f"max {spread[m].max():.3%}, median {np.median(spread[m]):.3%}")

    tbl = ranges(W2)
    print(f"\n{'scale':>6} {'type':>7} {'ell_peak':>9} {'half_lo':>9} {'half_hi':>9}")
    for r in tbl:
        print(f"{r['scale']:>6} {r['type']:>7} {r['ell_peak']:>9} "
              f"{r['ell_half_lo']:>9} {r['ell_half_hi']:>9}")

    ell = np.arange(LMAX + 1)
    stem = os.path.join(OUT, "starlet_scale_ell" + (f"_{a.tag}" if a.tag else ""))
    np.savez(stem + "_data.npz", ell=ell, W2=W2, raw=raw,
             delta_spread=(spread if spread is not None else np.zeros(1)),
             method=a.method, ndelta=a.ndelta, nreal=a.nreal, table=json.dumps(tbl))

    make_figure(W2, tbl, a)

    # GIT_COMMIT is exported by the SLURM wrapper: compute nodes do not reliably have git on
    # PATH, and a silent "unknown" here is exactly what the publish gate flags.
    commit = os.environ.get("GIT_COMMIT", "")
    if not commit:
        try:
            commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                             text=True).strip()
        except Exception:
            commit = "unknown"
    json.dump(dict(figure=os.path.basename(stem),
                   generator="scripts/diagnostics/starlet_scale_ell.py",
                   git_commit=commit,
                   generated_utc=datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
                   nside=NSIDE, nscale=NSCALE, lmax=LMAX, method=a.method,
                   ndelta=a.ndelta, nreal=a.nreal,
                   estimator=("single-pixel (white) input, W^2 = C_ell[coef]/C_ell[input]; "
                              "deterministic, no Monte Carlo error"
                              if a.method != "noise" else
                              "Monte Carlo over white-noise maps"),
                   validation=validation, table=tbl,
                   # Rendered into the published README by scripts/paper/figures.py. The
                   # measured bands ARE the result of this figure, so they belong where a
                   # reader will see them, not only in the CSV.
                   readme_table=dict(
                       title="Measured multipole coverage per starlet scale",
                       intro=f"nside {NSIDE}, nscale={NSCALE}. Half-power = the multipole "
                             f"range over which the response is at least 50% of its peak. "
                             f"Bands OVERLAP; these are where each dominates, not sharp "
                             f"windows.",
                       columns=["scale", "type", "ell at peak", "ell half-power range",
                                "role in the analysis"],
                       rows=[[r["scale"], r["type"], r["ell_peak"],
                              f"{r['ell_half_lo']} - {r['ell_half_hi']}", role]
                             for r, role in zip(tbl, [
                                 "dropped by the baryon-safe cut (scales234)",
                                 "kept", "kept",
                                 "kept; its half-power edge sets the PS floor l>=37",
                                 "excluded throughout; the only band carrying the monopole"])],
                       footnote="Wavelet 0 is RESOLUTION-limited: it peaks near the Nyquist "
                                "multipole 3*nside-1 = 1535, so its band moves with map "
                                "resolution while the others do not. Angular-scale labels "
                                "(~10 arcmin for wavelet 0, doubling thereafter) are dyadic "
                                "smoothing-scale names and are NOT 10800/ell_peak -- a starlet "
                                "band peaks at roughly half the multipole its label suggests.",
                   ),
                   versions={m: _ver(m) for m in ("numpy", "healpy", "matplotlib")},
                   mplstyle="styles/paper_v1.mplstyle",
                   figure_inches=[a.width, round(a.width * a.aspect, 3)],
                   scales_included={
                       "note": "This figure MEASURES the scale-to-multipole mapping; it is not "
                               "computed on a scale-cut data vector.",
                       "starlet": f"nscale={NSCALE} -> wavelets 0-{NSCALE-2} plus coarse",
                       "nside": NSIDE,
                       "ell_range_measured": [0, LMAX],
                       "analysis_cut_shown": "shading marks the band removed by scales234 "
                                             "(wavelet 0), i.e. ell >= its half-power edge",
                   },
                   caveats=[
                       "Measured at nside 512 with nscale=5. Wavelet 0 is RESOLUTION-limited, "
                       "peaking near the Nyquist multipole 3*nside-1=1535, so its band moves "
                       "with map resolution; the other bands do not.",
                       "Half-power ranges summarise bands that OVERLAP substantially. "
                       "'scales234 covers ell 36-336' is where those bands dominate, not a "
                       "sharp window.",
                       "Measured on the FULL SPHERE. A mask couples multipoles, so the "
                       "effective coverage of a masked map is broader than the table implies.",
                       "theta ~ 10800/ell is a convention and is NOT how the scale labels were "
                       "assigned: a starlet band peaks at roughly half the multipole its "
                       "nominal scale size suggests. The measured ell ranges are authoritative.",
                   ]),
              open(stem + "_provenance.json", "w"), indent=2)
    with open(stem + "_values.csv", "w") as fh:
        # n_seeds is how many independent measurements were averaged. For the delta method that
        # is the number of PIXEL POSITIONS, not training seeds -- they average out pixelisation
        # anisotropy, not noise, since the measurement is deterministic. Recorded because the
        # publish gate is right that a values file should say what its ensemble was.
        fh.write("scale,type,ell_peak,ell_half_lo,ell_half_hi,n_seeds,ensemble\n")
        n = a.ndelta if a.method != "noise" else a.nreal
        kind = "delta positions" if a.method != "noise" else "white-noise realisations"
        for r in tbl:
            fh.write(f"{r['scale']},{r['type']},{r['ell_peak']},"
                     f"{r['ell_half_lo']},{r['ell_half_hi']},{n},{kind}\n")
    print(f"\nwrote {stem}.png/.pdf + _data.npz + _provenance.json + _values.csv")


if __name__ == "__main__":
    main()
