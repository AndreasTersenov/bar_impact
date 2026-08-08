#!/usr/bin/env python3
"""Noiseless fractional impact of baryons on C_ell: standard bins vs BNT bins.

Reproduces the pre-disk-failure version of this figure from the rebuilt noiseless spectra
(SLURM job 597132 for the baryonified halves; the nobaryons halves are the 2025-10 originals).

BAND. The 200 baryonified and 200 nobaryons realisations are the SAME 200 CosmoGridV1 perms in
the same order, so C_bar - C_dmo is a paired difference and its sample variance is far smaller
than either spectrum's. The band is the standard error on the mean of that paired difference,
std(C_bar - C_dmo)/sqrt(200) / <C_dmo> -- i.e. how well 200 realisations pin down the mean
suppression. Taking the two sets as independent would inflate it by roughly an order of
magnitude and would be wrong here.

No shape noise anywhere in this figure.

  PYTHONNOUSERSITE=1 <jaxili python> scripts/plot_ps_frac_diff_noiseless.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/"
       "fiducial/cosmo_fiducial")
LMIN_FIT = 200   # window quoted in the job's verification table, echoed into the values file


def load(src: str, pre: str, kind: str, b: int) -> np.ndarray:
    return np.load(os.path.join(src, f"{pre}_fiducial_{kind}_bin{b}.npy"))


def curve(src: str, pre: str, b: int):
    dmo, bar = load(src, pre, "nobaryons", b), load(src, pre, "baryonified", b)
    if dmo.shape != bar.shape:
        sys.exit(f"[fatal] shape mismatch {pre} bin{b}: {dmo.shape} vs {bar.shape}")
    n = dmo.shape[0]
    mean_dmo = dmo.mean(0)
    frac = bar.mean(0) / mean_dmo - 1.0
    err = (bar - dmo).std(0, ddof=1) / np.sqrt(n) / mean_dmo
    return frac, err, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmin", type=int, default=2, help="first ell plotted; 0 and 1 carry no signal")
    p.add_argument("--src", default=None,
                   help="directory holding the 16 .npy. Defaults to the cluster tree, and falls "
                        "back to this script's own directory so the transferred bundle runs "
                        "as-is off the cluster.")
    p.add_argument("--style", default=os.path.join(REPO, "styles", "paper_v1.mplstyle"))
    p.add_argument("--outdir", default="outputs/plots/ps_frac_diff_noiseless")
    p.add_argument("--name", default="ps_frac_diff_noiseless")
    a = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = a.src or (SRC if os.path.isdir(SRC) else here)
    probe = os.path.join(src, "all_cls_fiducial_nobaryons_bin1.npy")
    if not os.path.exists(probe):
        sys.exit(f"[fatal] no spectra in {src} (looked for {os.path.basename(probe)}); pass --src")

    if os.path.exists(a.style):
        plt.style.use(a.style)
    ell = np.arange(1025)
    sl = slice(a.lmin, None)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="k", ls="--", lw=0.9, zorder=1)

    rows = []
    for pre, ls, tag in (("all_cls", "-", "Bin"), ("all_bnt_cls", ":", "BNT bin")):
        for b in (1, 2, 3, 4):
            frac, err, n = curve(src, pre, b)
            col = f"C{b - 1}"
            ax.plot(ell[sl], frac[sl], ls=ls, color=col, lw=1.8, label=f"{tag} {b}", zorder=3)
            ax.fill_between(ell[sl], (frac - err)[sl], (frac + err)[sl],
                            color=col, alpha=0.30, lw=0, zorder=2)
            w = slice(LMIN_FIT, 1000)
            rows.append(dict(statistic="BNT" if "bnt" in pre else "standard", bin=b, n_real=n,
                             frac_at_ell1000=float(frac[1000]),
                             mean_frac_200_1000=float(frac[w].mean()),
                             err_at_ell1000=float(err[1000])))

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\langle \Delta C_\ell \rangle / \langle C_\ell \rangle$")
    ax.set_xlim(a.lmin, 1024)
    ax.legend(ncol=2, loc="lower left", frameon=True)
    fig.tight_layout()

    # In the repo, write under it. In the transferred bundle there is no repo above the script,
    # so fall back to the working directory rather than inventing an outputs/ tree beside it.
    base = REPO if os.path.isdir(os.path.join(REPO, "scripts")) else os.getcwd()
    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(base, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, a.name)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)

    with open(f"{stem}_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:  # no git off-cluster (the transferred bundle) -- degrade rather than crash
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                         text=True).strip()
        if subprocess.check_output(["git", "-C", REPO, "status", "--porcelain",
                                    "scripts/plot_ps_frac_diff_noiseless.py"], text=True).strip():
            commit += " (generator dirty/untracked at run time)"
    except Exception:
        commit = "unknown (run outside the repo)"
    json.dump({
        "generator": "scripts/plot_ps_frac_diff_noiseless.py",
        "command": shlex.join(sys.argv),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "mplstyle": a.style if os.path.exists(a.style) else "matplotlib default (style sheet absent)",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "caveats": [
            "noiseless -- no shape noise anywhere; row scatter is sample variance only",
            "BNT bin 1 is standard bin 1 bit-for-bit (first BNT row is the identity); "
            "the two curves overlapping exactly is a consistency check, not a duplicate",
            "baryonified halves rebuilt post-crash from the read-only CosmoGridV1 release "
            "(job 597132); nobaryons halves are the 2025-10 originals",
        ],
        "source_dir": src,
        "noise": "none -- noiseless spectra",
        "n_realisations": rows[0]["n_real"],
        "band": ("standard error on the mean of the PAIRED difference, "
                 "std(C_bar - C_dmo)/sqrt(N) / <C_dmo>; the two sets share realisations"),
        "scales_included": {"lmin_plotted": a.lmin, "lmax": 1024,
                           "note": "full multipole range, no scale cut; noiseless product is "
                                   "lmax=1024 only -- never mix with the *_lmax2048 variants"},
        "baryonified_provenance": "SLURM job 597132, scripts/jz/ps_noiseless_baryonified.slurm",
        "nobaryons_provenance": "originals, 2025-10-07 (standard) / 2025-10-09 (BNT)",
    }, open(f"{stem}_provenance.json", "w"), indent=2)

    for r in rows:
        print(f"  {r['statistic']:8s} bin{r['bin']}  dC/C(1000)={r['frac_at_ell1000']:+.4f}  "
              f"mean(200-1000)={r['mean_frac_200_1000']:+.4f}  err(1000)={r['err_at_ell1000']:.2e}")
    print(f"\nwrote {stem}.pdf/.png + _values.csv + _provenance.json")


if __name__ == "__main__":
    main()
