#!/usr/bin/env python3
"""Fractional impact of baryons on the starlet l1-norm / peak counts, per wavelet scale.

Replaces the notebook cells behind plots/frac_diff_L1_scales_v2.pdf and frac_diff_pc_scales_v2.pdf,
applying the same corrections that Fig. 2 (the power spectrum) received, plus two that are specific
to the HOS.

WHAT CHANGED FROM THE PUBLISHED FIGURES

  1. BAND = err(stat_bar) / <stat_DMO>, not err(bar - DMO)/<DMO>. The referee's point on Fig. 2
     applies unchanged: the NDE is trained on DMO, so DMO is the MODEL and the baryonified vector is
     the OBSERVATION. The uncertainty that belongs on the figure is the error of a single
     measurement. (For a model with no error of its own the two forms are algebraically identical;
     they differ only when the second is computed empirically from paired realizations, where the
     shared simulations cancel part of the variance.)

  2. MATCHED SHAPE NOISE. The published summaries were built with os.urandom-seeded noise drawn
     independently for the two variants. These read the regenerated suite (tag `matchednoise`),
     where the same noise realization was added to both maps. Measured effect on the l1-norm:
     correlation between the baryonified and DMO realizations 0.027 -> 0.975.

  3. NO +5 IN THE DENOMINATOR. The published figure plotted <d stat> / (<stat> + 5). That offset
     contributes 0.00% in the core bins and 100% in the tails, where <l1> is identically zero, so
     the wings of the published figure are `d stat / 5` rather than a fractional difference. Here
     the empty bins are MASKED instead -- which is what the paper's own text describes ("the
     absolute signal quickly becomes identically zero at the lowest SNRs where the starlet
     l1-norm is empty"). Bins that are small but real, including the nu >~ 6 turnover the text
     discusses, are kept and get an honest band.

  4. TRUE SNR BIN CENTRES. get_wtl1_sphere returns the centres -- midpoints of nbins+1 edges -- but
     the old pipeline discarded them and the figure rebuilt them as np.linspace(min, max, nbins).
     For 40 bins on [-10, 10] the truth is -9.75..9.75 at spacing 0.5, not -10..10 at 0.513. These
     are read from the *_bincentres.npy written by the regeneration.

PEAKS BEHAVE DIFFERENTLY FROM L1 UNDER MATCHED NOISE, and it is not a bug: peak counts are
discrete, so identical noise still moves pixels across a threshold or a bin edge. Measured
correlation after matching is ~0.65 for peaks against ~0.97 for l1, so the peaks bands stay wider.

  PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python
  $PY scripts/plot_hos_frac_diff.py --stat l1
  $PY scripts/plot_hos_frac_diff.py --stat peaks --area 14000
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
# --- A&A house style -------------------------------------------------------
# Load the style sheet rather than scattering rcParams: it fixes font family and
# every text size at the journal minimum, the Okabe-Ito colour cycle, inward ticks
# on all four sides, and Type-42 embedded fonts. Restored from the figure-polish
# skill (private repo dotfiles-claude); styles/aa.mplstyle carries the provenance.
STYLE = os.path.join(REPO, "styles", "aa.mplstyle")
plt.style.use(STYLE)
# A&A printed widths. Size the figure to the column it will occupy -- letting the
# journal scale it down shrinks the text below the legibility minimum.
AA_W = {"single": 3.465, "intermediate": 4.724, "double": 7.087}

FID = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/"
       "fiducial/cosmo_fiducial")
SKY_DEG2 = 41253.0
STEM = {"l1": "all_l1_norms", "peaks": "all_peak_counts"}
YLAB = {"l1": r"$\langle \Delta \ell_1 \rangle / \langle \ell_1 \rangle$",
        "peaks": r"$\langle \Delta N_{\rm peaks} \rangle / \langle N_{\rm peaks} \rangle$"}
# The first four starlet bands are ~[10', 20', 40', 80']; the fifth is the coarse scale.
SCALE_ARCMIN = ["10'", "20'", "40'", "80'", "coarse"]


def load(stat, kind, b, tag):
    p = f"{FID}/{STEM[stat]}_fiducial_{kind}_bin{b}_noisy_s0.26_new_normalization{tag}.npy"
    if not os.path.exists(p):
        sys.exit(f"[fatal] missing {p}\n        (run scripts/jz/hos_matched_noise.slurm first)")
    a = np.load(p)
    if not np.isfinite(a).all():
        sys.exit(f"[fatal] {p} has non-finite entries")
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stat", default="l1", choices=["l1", "peaks"])
    p.add_argument("--noise", default="matched", choices=["matched", "published"])
    p.add_argument("--scales", default="1,2,3", help="1-indexed wavelet scales to panel")
    p.add_argument("--area", default="fullsky",
                   help="'fullsky' or a survey area in deg2. Only the BAND scales, by "
                        "1/sqrt(f_sky). For the HOS this is cruder than for the power spectrum -- "
                        "there is no mode-counting result behind it, only the fact that the "
                        "statistic is a sum over the map, so the number of independent patches "
                        "scales with f_sky. Treat it as indicative.")
    p.add_argument("--min-frac", type=float, default=0.0,
                   help="mask bins whose <stat_DMO> is below this fraction of the scale's peak. "
                        "0 masks only exactly-empty bins, which is what the paper's text describes.")
    p.add_argument("--palette", default="tab10")
    p.add_argument("--fill-alpha", type=float, default=0.28)
    p.add_argument("--edge-alpha", type=float, default=0.50)
    p.add_argument("--edge-lw", type=float, default=0.45)
    p.add_argument("--curve-lw", type=float, default=1.4,
                   help="curve weight. The style baseline is 1.2; 1.4 lifts the "
                        "curves just clear of their bands without reading as heavy "
                        "at A&A column width.")
    p.add_argument("--ylim", default="curves",
                   help="'curves' (default) fits the CURVES with a margin and lets the tail bands "
                        "clip -- the bands in the near-empty end bins reach +-14, so fitting them "
                        "compresses the entire signal region to invisibility. 'auto' fits "
                        "curve+-band. 'none' leaves matplotlib's choice. Or 'lo,hi'.")
    p.add_argument("--ypad-lo", type=float, default=0.18,
                   help="bottom margin, as a fraction of the curve range (--ylim curves)")
    p.add_argument("--ypad-hi", type=float, default=0.32,
                   help="top margin, as a fraction of the curve range. Larger than the bottom by "
                        "default: the curves are almost entirely negative, so symmetric padding "
                        "leaves the zero line crowded against the top of the frame.")
    p.add_argument("--outdir", default="outputs/plots/hos_frac_diff")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    tag = "_matchednoise" if a.noise == "matched" else ""
    scales = [int(v) - 1 for v in a.scales.split(",")]
    fsky = 1.0 if a.area == "fullsky" else float(a.area) / SKY_DEG2
    band_scale = 1.0 / np.sqrt(fsky)
    area_lab = "full sky" if a.area == "fullsky" else f"{float(a.area):.0f} deg2"

    # True bin centres, written by the regeneration. Never reconstruct these.
    cpath = f"{FID}/{STEM[a.stat]}_fiducial_bin1{tag}_bincentres.npy"
    if tag and os.path.exists(cpath):
        cen_all = np.load(cpath)
        centres = (lambda s: cen_all[s]) if cen_all.ndim == 2 else (lambda s: cen_all)
        cen_src = os.path.basename(cpath)
    else:
        n = 40 if a.stat == "l1" else 30
        lo, hi = (-10.0, 10.0) if a.stat == "l1" else (-2.0, 10.0)
        e = np.linspace(lo, hi, n + 1)
        c = 0.5 * (e[:-1] + e[1:])
        centres = lambda s: c                                            # noqa: E731
        cen_src = f"reconstructed midpoints of linspace({lo},{hi},{n+1})"

    styles = ["-", "--", "-.", ":"]
    colors = ["C0", "C1", "C2", "C3"] if a.palette == "tab10" else \
        ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

    print(f"stat={a.stat}  noise={a.noise}  scales={[s+1 for s in scales]}  area={area_lab} "
          f"(band x{band_scale:.3f})\n  bin centres from: {cen_src}")

    rows, panel = [], []
    for s in scales:
        per_bin = []
        for i, b in enumerate((1, 2, 3, 4)):
            bar = load(a.stat, "baryonified", b, tag)[:, s, :]
            nob = load(a.stat, "nobaryons", b, tag)[:, s, :]
            mn = nob.mean(0)
            keep = mn > a.min_frac * mn.max() if a.min_frac > 0 else mn > 0
            curve = np.full(mn.shape, np.nan)
            band = np.full(mn.shape, np.nan)
            curve[keep] = (bar - nob).mean(0)[keep] / mn[keep]
            band[keep] = bar.std(0)[keep] / mn[keep] * band_scale
            per_bin.append((curve, band, keep))
            for j in np.where(keep)[0]:
                rows.append(dict(stat=a.stat, scale=s + 1, bin=b, snr=float(centres(s)[j]),
                                 frac_diff=float(curve[j]), band=float(band[j]),
                                 n_realizations=int(bar.shape[0]), n_seeds=1))
        panel.append(per_bin)
        nk = per_bin[0][2].sum()
        print(f"  scale {s+1} ({SCALE_ARCMIN[s]}): {nk}/{len(per_bin[0][2])} bins kept")

    fig, axes = plt.subplots(1, len(scales), figsize=(AA_W["double"], 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s, per_bin in zip(axes, scales, panel):
        x = centres(s)
        for i, b in enumerate((1, 2, 3, 4)):
            curve, band, keep = per_bin[i]
            ax.fill_between(x, curve - band, curve + band, color=colors[i], alpha=a.fill_alpha,
                            lw=0)
            for edge in (curve - band, curve + band):
                ax.plot(x, edge, color=colors[i], lw=a.edge_lw, alpha=a.edge_alpha, zorder=3)
            ax.plot(x, curve, color=colors[i], ls=styles[i], lw=a.curve_lw, zorder=5,
                    label=f"bin {b}" if ax is axes[0] else None)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_xlabel("SNR")
        ax.text(0.04, 0.94, f"Scale {s+1}", transform=ax.transAxes,
                ha="left", va="top")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(YLAB[a.stat])
    if a.ylim in ("auto", "curves"):
        if a.ylim == "curves":
            lo = np.nanmin([np.nanmin(c) for pb in panel for c, _, _ in pb])
            hi = np.nanmax([np.nanmax(c) for pb in panel for c, _, _ in pb])
            rng = hi - lo
            axes[0].set_ylim(lo - a.ypad_lo * rng, hi + a.ypad_hi * rng)
        else:
            lo = np.nanmin([np.nanmin(c - b) for pb in panel for c, b, _ in pb])
            hi = np.nanmax([np.nanmax(c + b) for pb in panel for c, b, _ in pb])
            pad = 0.08 * (hi - lo)
            axes[0].set_ylim(lo - pad, hi + pad)
    elif a.ylim != "none":
        axes[0].set_ylim(*[float(v) for v in a.ylim.split(",")])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4)

    outdir = os.path.join(REPO, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    name = a.name or (f"hos_frac_diff_{a.stat}_{a.noise}"
                      + ("" if a.area == "fullsky" else f"_{a.area}"))
    base = os.path.join(outdir, name)
    for ext in ("pdf", "png"):
        plt.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200, transparent=True)
    plt.close("all")

    with open(f"{base}_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "figure": name, "generator": "scripts/plot_hos_frac_diff.py",
        "command": " ".join(sys.argv), "git_commit": commit,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mplstyle": "styles/aa.mplstyle (A&A; restored original)",
        "statistic": ("starlet l1-norm" if a.stat == "l1" else "starlet peak counts"),
        "scales_included": {f"scale {s+1}": SCALE_ARCMIN[s] for s in scales},
        "snr_binning": ("l1: 40 bins over [-10,10] (the PAPER's range; the old code used [-13,13], "
                        "so these vectors are not interchangeable with the NPE inference inputs)"
                        if a.stat == "l1" else
                        "peaks: 31 edges over [-2,10] -> 30 counts. NB the paper says 40 bins; the "
                        "code and the data both say 30, so the text needs correcting."),
        "bin_centres_source": cen_src,
        "curve": "mean(stat_bar - stat_DMO) / mean(stat_DMO), empty DMO bins masked",
        "band": f"std(stat_bar)/mean(stat_DMO) x 1/sqrt(f_sky); area = {area_lab}",
        "noise_seeds": ("MATCHED between baryonified and DMO (job 536722)" if a.noise == "matched"
                        else "INDEPENDENT (published suite)"),
        "f_sky": fsky, "band_scale_applied": band_scale,
        "caveats": [
            "No +5 offset in the denominator. The published figure used <stat>+5, which is 100% of "
            "the denominator in the empty tails; those bins are masked here instead.",
            "Peak counts are DISCRETE, so matched noise cancels less for them than for the l1-norm "
            "(correlation ~0.65 vs ~0.97) and their bands stay wider. Not a defect.",
            "The band is scaled to survey area by 1/sqrt(f_sky), which for a HOS has no "
            "mode-counting justification -- indicative only.",
        ],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    blob = json.dumps(prov, indent=2)
    tmp = f"{base}_provenance.json.tmp"
    open(tmp, "w").write(blob); json.loads(open(tmp).read()); os.replace(tmp, f"{base}_provenance.json")
    print(f"wrote {base}.pdf/.png + _values.csv + _provenance.json")


if __name__ == "__main__":
    main()
