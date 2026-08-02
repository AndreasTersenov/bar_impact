#!/usr/bin/env python3
"""Fig. 2 -- fractional impact of baryons on the auto power spectra, with the survey error band.

REFEREE POINT THIS ADDRESSES. The referee asked whether the error band is err(PS_bar)/PS_DMO or
err(PS_bar - PS_DMO)/PS_DMO, and said it should be the former. It was the latter:

    std_fractional_diff_bin1 = np.std((fid_bar_bin1 - fid_nobar_bin1) / fid_nobar_bin1, axis=0)

`--band survey` (the default here) is the referee's requested form: the scatter of the baryonified
spectrum itself, over the mean DMO spectrum, i.e. the STATISTICAL ERROR OF THE SURVEY. `--band diff`
reproduces the published band for comparison.

WHY THE CHANGE ALSO SETTLES THE REFEREE'S SECOND POINT. They asked whether the same shape-noise
realization is used for baryons and DMO, since cosmic variance should otherwise cancel. It is NOT --
`power_spectrum_processing.py:91` seeds each worker from os.urandom(4), so the two runs got
independent noise, and it is not even reproducible across reruns. Measured confirmation: the
per-ell correlation between the baryonified and DMO realizations tracks f^2 (f = signal fraction),
the signature of shared simulations with independent noise, against f for shared noise. Control:
correlating noisy vs noiseless DMO (certainly the same sims) tracks f almost exactly, which both
validates that model and shows realization indices do correspond across files.

The survey band needs no subtraction, so nothing has to cancel and the seeds do not matter for it.
The residual wiggle in the MEAN curve does still come from the unmatched noise, and cannot be
post-processed away -- removing it needs the spectra regenerated with matched seeds.

WHAT THIS FIGURE IS, AND HOW IT DIFFERS FROM Fig. B.1. Here the denominator is the NOISY DMO
spectrum, so this shows the baryonic shift relative to what is actually measured. Fig. B.1 uses
noiseless maps, so its denominator is the signal alone. The same physics therefore appears ~34x
larger in B.1 at high ell, where these maps are ~97% shape noise:

    bin 1, ell ~900-1000 : this figure -0.0044   /f (f=0.029)  =  -0.152   vs B.1's -0.165

That is not a discrepancy, but it must be stated in the caption or it reads as one.

  PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python
  $PY scripts/plot_ps_frac_diff.py                 # referee's band
  $PY scripts/plot_ps_frac_diff.py --band diff     # the published band, for comparison
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FID = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/"
       "fiducial/cosmo_fiducial")
LMIN = 30
NBIN = 10
LMAX = 1024


def load(kind, b, tag=""):
    """tag='' is the published (independent-noise) suite; tag='_matchednoise' is the regenerated
    one where the SAME shape-noise realization was added to the baryonified and DMO maps."""
    p = f"{FID}/all_cls_fiducial_{kind}_bin{b}_noisy_s0.26{tag}.npy"
    raw = np.fromfile(p, dtype=np.uint8)
    z = (raw == 0)
    run = best = 0
    for v in z:
        run = run + 1 if v else 0
        best = max(best, run)
    a = np.load(p)
    # Stripe signature, not zero fraction: a RAID0 loss leaves >=512 KiB of contiguous NULs.
    # Zero FRACTION is not a damage test for these files and has produced false alarms here.
    if best >= 524288 or not np.isfinite(a).all():
        sys.exit(f"[fatal] {p} looks disk-damaged (longest zero run {best} B)")
    return a


def rebin(arr, new_size=NBIN):
    """The published binning, reproduced verbatim so the curve is unchanged.

    NOTE the log_bins values are used as INDICES into an array already sliced at LMIN, so each bin
    covers ell = index + LMIN. `ell_centres()` below reports where the bins actually sit; the
    published x-axis was np.arange(30, 1024, 100), which is linear and does not match. Exposed as
    --ell-axis rather than silently corrected, because it changes where the points are drawn.
    """
    log_bins = np.logspace(np.log10(LMIN), np.log10(LMAX), new_size + 1).astype(int)
    out = np.zeros((arr.shape[0], new_size))
    for i in range(new_size):
        out[:, i] = arr[:, log_bins[i]:log_bins[i + 1]].mean(axis=1)
    return out


def ell_centres(ncol, new_size=NBIN):
    log_bins = np.logspace(np.log10(LMIN), np.log10(LMAX), new_size + 1).astype(int)
    c = []
    for i in range(new_size):
        lo = min(log_bins[i], ncol) + LMIN
        hi = min(log_bins[i + 1], ncol) + LMIN
        c.append(0.5 * (lo + hi))
    return np.array(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", default="survey", choices=["survey", "diff"],
                   help="survey = err(PS_bar)/PS_DMO (referee's request, default). "
                        "diff = err(PS_bar-PS_DMO)/PS_DMO (as published).")
    p.add_argument("--ell-axis", default="published", choices=["published", "centres"],
                   help="published = np.arange(30,1024,100) as in the paper. "
                        "centres = where the log bins actually sit.")
    p.add_argument("--ylim", default="published",
                   help="'published' = (-0.05, 0.05) as in the paper; 'auto' = fit the curves and "
                        "bands, which is the only way the shape is actually visible since the "
                        "curves reach only -0.016; or 'lo,hi'.")
    p.add_argument("--noise", default="published", choices=["published", "matched"],
                   help="published = independent shape-noise seeds between the two runs (the "
                        "original suite). matched = the regenerated suite where the SAME noise "
                        "realization was added to both, so cosmic variance AND shape noise cancel "
                        "in the difference (job 533505).")
    p.add_argument("--outdir", default="outputs/plots/ps_frac_diff")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    styles = ["-", "--", "-.", ":"]
    colors = ["C0", "C1", "C2", "C3"]

    curves, bands, rows = [], [], []
    ncol = None
    for b in (1, 2, 3, 4):
        tag = "" if a.noise == "published" else "_matchednoise"
        bar = rebin(load("baryonified", b, tag)[:, LMIN:])
        nob = rebin(load("nobaryons", b, tag)[:, LMIN:])
        ncol = load("nobaryons", b, tag).shape[1] - LMIN
        mean_nob = nob.mean(0)
        curve = (bar - nob).mean(0) / mean_nob          # unchanged from the published figure
        if a.band == "survey":
            band = bar.std(0) / mean_nob                # err(PS_bar) / PS_DMO
        else:
            band = ((bar - nob) / nob).std(0)           # the published band
        curves.append(curve); bands.append(band)
        for j in range(NBIN):
            rows.append(dict(bin=b, ell_index=j, frac_diff=float(curve[j]),
                             band=float(band[j]), n_realizations=int(bar.shape[0])))

    # Cumulative significance of the model-data mismatch. The per-band comparison of curve against
    # band is what the eye reads, and it UNDERSTATES the impact: baryons push every multipole the
    # same way while noise is random, so the mismatch accumulates coherently. Recorded here because
    # the caption has to state it -- without it the figure reads as "baryons are harmless", which is
    # the opposite of the paper's conclusion. Computed per-ell (not on the plotted bands) with a
    # Gaussian sigma_l = C_l sqrt(2/((2l+1) f_sky)).
    tag = "" if a.noise == "published" else "_matchednoise"
    ell = np.arange(LMAX + 1)
    cumsn = {}
    for fsky, lab in ((1.0, "full_sky"), (14000 / 41253.0, "14000deg2")):
        per_bin = {}
        for b in (1, 2, 3, 4):
            mb = load("baryonified", b, tag).mean(0)
            mn = load("nobaryons", b, tag).mean(0)
            sig = mn * np.sqrt(2.0 / ((2 * ell + 1) * fsky))
            m = ell >= LMIN
            r = np.zeros(LMAX + 1)
            r[m] = ((mb - mn)[m] / sig[m]) ** 2
            c = np.sqrt(np.cumsum(r))
            per_bin[f"bin{b}"] = {str(lm): round(float(c[lm]), 2)
                                  for lm in (200, 400, 460, 600, 800, 1000)}
        cumsn[lab] = per_bin
    print("\ncumulative S/N of the mismatch (sqrt(sum (dC/sigma)^2)), full sky:")
    for b in (1, 2, 3, 4):
        print(f"  bin {b}: " + "  ".join(f"lmax{k}={v}" for k, v in
                                         cumsn['full_sky'][f'bin{b}'].items()))

    ells_published = np.arange(LMIN, LMAX, 100)
    ells_true = ell_centres(ncol)
    x = ells_published if a.ell_axis == "published" else ells_true

    print(f"band = {a.band}   ell-axis = {a.ell_axis}   noise = {a.noise}")
    print(f"  published x : {ells_published}")
    print(f"  true centres: {np.round(ells_true).astype(int)}")
    for i, b in enumerate((1, 2, 3, 4)):
        print(f"  bin {b}: frac_diff[last]={curves[i][-1]:+.4f}  band[last]={bands[i][-1]:.4f}")

    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.figure(figsize=(6, 3.5))
    for i, b in enumerate((1, 2, 3, 4)):
        plt.fill_between(x, curves[i] - bands[i], curves[i] + bands[i], color=colors[i], alpha=0.3)
        plt.plot(x, curves[i], label=f"bin {b}", ls=styles[i], color=colors[i])
    plt.axhline(0, color="black", linestyle="--", lw=1)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\langle \Delta C_\ell \rangle / \langle C_\ell \rangle$")
    plt.legend()
    if a.ylim == "published":
        plt.ylim(-0.05, 0.05)
    elif a.ylim == "auto":
        lo = min((c - b).min() for c, b in zip(curves, bands))
        hi = max((c + b).max() for c, b in zip(curves, bands))
        pad = 0.08 * (hi - lo)
        plt.ylim(lo - pad, hi + pad)
    else:
        plt.ylim(*[float(v) for v in a.ylim.split(",")])

    outdir = os.path.join(REPO, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    name = a.name or f"ps_frac_diff_band_{a.band}" + ("" if a.ell_axis == "published" else "_ellfix")
    base = os.path.join(outdir, name)
    for ext in ("pdf", "png"):
        plt.savefig(f"{base}.{ext}", transparent=True, bbox_inches="tight", dpi=200)
    plt.close("all")

    with open(f"{base}_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "figure": name,
        "generator": "scripts/plot_ps_frac_diff.py",
        "command": " ".join(sys.argv),
        "git_commit": commit,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mplstyle": "matplotlib default; no repo style sheet",
        "statistic": "tomographic auto angular power spectra, 4 bins",
        "source_data": f"{FID}/all_cls_fiducial_{{baryonified,nobaryons}}_bin{{1..4}}_noisy_s0.26.npy",
        "n_realizations": 200,
        "noise_seeds": ("INDEPENDENT between baryonified and DMO (original suite)"
                        if a.noise == "published" else
                        "MATCHED: the same shape-noise realization added to both maps "
                        "(scripts/ps_matched_noise_processing.py, job 533505)"),
        "curve": "mean(C_bar - C_DMO) / mean(C_DMO), both NOISY -- unchanged from the published figure",
        "band": ("std(C_bar) / mean(C_DMO) -- the survey statistical error; the referee's requested "
                 "err(PS_bar)/PS_DMO"
                 if a.band == "survey" else
                 "std((C_bar - C_DMO)/C_DMO) -- as published"),
        "binning": f"{NBIN} logarithmic bands, lmin={LMIN}, lmax={LMAX}",
        "ell_axis": ("np.arange(30,1024,100), as published" if a.ell_axis == "published"
                     else "true log-bin centres"),
        "ell_axis_published": [int(v) for v in ells_published],
        "ell_axis_true_centres": [float(v) for v in ells_true],
        "cumulative_SN_of_mismatch": cumsn,
        "band_area": "full sky (the simulations' own footprint); NOT rescaled to 14000 deg2",
        "caveats": [
            "THE PER-BAND COMPARISON UNDERSTATES THE IMPACT. The curve sits below the band at most "
            "ell, but the baryonic shift is COHERENT while the noise is random, so it accumulates "
            "as sqrt(N). See cumulative_SN_of_mismatch: it reaches ~10 sigma (full sky) / ~6 sigma "
            "(14000 deg2) by lmax=1000 in the high-z bins. The caption MUST say this, or the "
            "figure argues against the paper's own conclusion.",
            "The denominator is the NOISY DMO spectrum, so this is the baryonic shift relative to "
            "the measured spectrum. Fig. B.1 divides by the signal alone, so the same physics "
            "appears up to ~34x larger there at high ell. State this in the caption.",
            "Shape-noise realizations are INDEPENDENT between the baryonified and DMO runs "
            "(power_spectrum_processing.py:91 seeds from os.urandom). The survey band is unaffected "
            "since it involves no subtraction, but the residual wiggle in the mean curve is caused "
            "by this and can only be removed by regenerating with matched seeds.",
            "The published x-axis is linear while the binning is logarithmic; --ell-axis centres "
            "shows where the bins actually sit. Default reproduces the published positions.",
        ],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    blob = json.dumps(prov, indent=2)
    tmp = f"{base}_provenance.json.tmp"
    with open(tmp, "w") as fh:
        fh.write(blob)
    json.loads(open(tmp).read())
    os.replace(tmp, f"{base}_provenance.json")
    print(f"wrote {base}.pdf/.png + _values.csv + _provenance.json")


if __name__ == "__main__":
    main()
