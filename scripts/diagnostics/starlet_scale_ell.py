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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("delta", "noise", "both"), default="delta")
    ap.add_argument("--ndelta", type=int, default=8)
    ap.add_argument("--nreal", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

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

    # ---- figure ----------------------------------------------------------------
    plt.style.use(os.path.join(REPO, "styles", "paper_v1.mplstyle"))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for j in range(NSCALE):
        # No binning: the deterministic response is already smooth, and binning would only
        # blur the band edges that the whole figure exists to show.
        ax.semilogx(ell[1:], W2[j][1:], color=COLORS[j], lw=2.4, label=LABELS[j])

    # The analysis keeps wavelets 1-3 ("scales234") and drops wavelet 0. Shade what that means
    # in multipole terms -- this is the point of the figure.
    w0 = next(r for r in tbl if r["type"] == "wav0")
    ax.axvspan(w0["ell_half_lo"], LMAX, color="0.85", alpha=0.55, zorder=0)
    ax.text(np.sqrt(w0["ell_half_lo"] * LMAX), 0.50, "dropped by the\nbaryon-safe cut",
            ha="center", va="center", fontsize=12, color="0.35")

    ax.set_xlim(2, LMAX)
    # Headroom for the legend: the bands peak at 1.0 right across the panel, so an in-axes
    # legend at any y below this sits on top of a curve.
    ax.set_ylim(0, 1.30)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"normalised response $W_j(\ell)^2/\max$")
    ax.legend(frameon=False, ncol=5, loc="upper center", fontsize=12,
          columnspacing=1.2, handlelength=1.6, borderaxespad=0.2)
    ax.grid(alpha=0.25, ls=":")
    secax = ax.secondary_xaxis("top",
                               functions=(lambda l: 10800.0 / np.clip(l, 1e-6, None),
                                          lambda t: 10800.0 / np.clip(t, 1e-6, None)))
    secax.set_xlabel(r"angular scale $10800/\ell$ [arcmin]")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)

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
                   validation=validation, table=tbl),
              open(stem + "_provenance.json", "w"), indent=2)
    with open(stem + "_values.csv", "w") as fh:
        fh.write("scale,type,ell_peak,ell_half_lo,ell_half_hi\n")
        for r in tbl:
            fh.write(f"{r['scale']},{r['type']},{r['ell_peak']},"
                     f"{r['ell_half_lo']},{r['ell_half_hi']}\n")
    print(f"\nwrote {stem}.png/.pdf + _data.npz + _provenance.json + _values.csv")


if __name__ == "__main__":
    main()
