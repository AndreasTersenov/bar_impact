#!/usr/bin/env python3
"""Audit the inputs for the baryon-safe FoM table (the paper's Table 3), and compute what exists.

WHAT THE TABLE IS. For each summary statistic, at ITS OWN required scale cut -- the cut where the
baryonic bias is still below 0.3 sigma -- how much cosmological information is left? One column per
survey area, values as FoM_3 = 1/sqrt(det Cov(Omega_m, sigma_8, w_0)), plus the ratio to the PS
baseline. The submitted version quoted 2000 / 5000 / 14000 / full sky.

WHY IT HAS TO BE REBUILT RATHER THAN LOOKED UP. Every number in the submitted table predates the
corrections this branch carries: the ell>=37 recovery, the submean (footprint-mean-subtracted) HOS
products, and the disk failure that thinned the seed ensembles. None of its values can be reused,
and the numbers here will differ.

THE CUT IS NOT A CONSTANT.
  PS   -- per AREA, the largest step-40 upper cut whose MEAN 3-param tension is still < 0.3 sigma.
           Read from the campaign table, never hardcoded. Note this is the last cut that PASSES,
           not the first that fails; at 14000 those are 460 and 500, and using the crossing would
           put a 0.41-sigma bias inside a column headed "baryon-safe".
  HOS  -- scales234 (the finest wavelet scale dropped). Whether that is ACTUALLY safe at every
           area is a separate question this script reports on rather than assumes.

WHAT IT REPORTS, per (statistic, area):
  the adopted cut, how many usable NULL seeds back it, FoM_3 pooled over those seeds, and a
  seed-to-seed error. Missing combinations are named as missing -- the point of the audit is the
  gaps, so nothing is silently skipped.

ERROR BARS. Two different numbers, and they answer different questions:
  fom3_pooled   -- FoM of the posterior formed by stacking every surviving seed. Its width includes
                   NPE training scatter, so it is the CONSERVATIVE constraint.
  fom3_mean/std -- mean and spread of the per-seed FoMs. The std is the honest error bar for a
                   table, because it says how much the answer moves if you retrain.
A single-seed entry gets no error bar, and that is reported rather than shown as +/- 0.

  PYTHONNOUSERSITE=1 <jaxili python> scripts/paper/audit_baryon_safe_fom.py [--areas ...] [--csv OUT]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMP = f"{REPO}/outputs/samples"
PSD = f"{REPO}/outputs/baryon_tension/ps_submean_l37/posteriors"
PSD_FS = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/posteriors/fullsky"
PS_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
PS_AGG_FS = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/tables/tension_3param_agg.csv"

AREAS = [2000, 5000, 10000, 14000, 28000, 35000, "fullsky"]
HOS_TAG = {2000: 2001, 5000: 5001, 10000: 10001, 14000: 14001, 28000: 28001, 35000: 35001}
THRESHOLD = 0.3
SIG_MAX = 0.08          # prior-collapse guard, same threshold as every other generator
HOS_SCALES = "scales234"


def required_ps_lmax(A, threshold=THRESHOLD):
    """Largest step-40 upper cut whose MEAN 3-param tension is still below `threshold`."""
    path = PS_AGG_FS if A == "fullsky" else PS_AGG
    if not os.path.exists(path):
        return None, f"no campaign table at {os.path.relpath(path, REPO)}"
    rows = [(int(r["upper_cut"]), float(r["mean"])) for r in csv.DictReader(open(path))
            if (r["area"] == "fullsky" if A == "fullsky" else int(r["area"]) == A)]
    if not rows:
        return None, f"no rows for area {A}"
    safe = [c for c, m in sorted(rows) if m < threshold]
    if not safe:
        return None, f"NOTHING baryon-safe: every cut at {A} exceeds {threshold} sigma"
    return max(safe), None


def ps_null_glob(A, lmax):
    if A == "fullsky":
        return (f"{PSD_FS}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_"
                f"bins1234_l37-{lmax}_r10_noisy_s0.26_run*_npe.npy")
    return (f"{PSD}/mask_{A:05d}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_"
            f"bins1234_l37-{lmax}_r10_masked_{A}sqdeg_apod2.0_master_submean_noisy_s0.26_run*.npy")


def hos_null_globs(prefix, A, scales=HOS_SCALES):
    """Candidate patterns, best first. Full sky may legitimately fall back to the non-submean
    product for a DETAIL-ONLY scale set: a starlet detail coefficient is a difference of
    successively smoothed maps, so a constant cancels identically. Never for a set containing
    the coarse scale."""
    if A == "fullsky":
        out = [f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_{scales}"
               f"_noisy_s0.26_fullsky_submean_new_normalization*_npe.npy"]
        if "5" not in scales.replace("scales", ""):
            out.append(f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_{scales}"
                       f"_noisy_s0.26_new_normalization*_npe.npy")
        return out
    return [f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_{scales}"
            f"_noisy_s0.26_masked_{HOS_TAG[A]}sqdeg_submean_new_normalization*_npe.npy"]


def fom3(a):
    return float(1.0 / np.sqrt(np.linalg.det(np.cov(a[:, :3], rowvar=False))))


def load_seeds(pattern):
    """Return (per_seed_arrays, runs, dropped). A disk-damaged .npy raises ValueError because
    numpy reads the mangled header as a pickle stream -- never 'fix' that with allow_pickle."""
    keep, runs, dropped = [], [], []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_run(\d+)", f)
        r = int(m.group(1)) if m else 1
        try:
            a = np.load(f)
        except (FileNotFoundError, IndexError, ValueError, OSError) as e:
            dropped.append((r, type(e).__name__))
            continue
        s = float(np.sqrt(np.cov(a[:, :3], rowvar=False)[1, 1]))
        if s >= SIG_MAX:
            dropped.append((r, f"collapsed sigma={s:.3f}"))
            continue
        keep.append(a[:, :3])
        runs.append(r)
    return keep, runs, dropped


def measure(patterns):
    for p in patterns:
        keep, runs, dropped = load_seeds(p)
        if keep:
            per = [fom3(a) for a in keep]
            pooled = fom3(np.concatenate(keep))
            return dict(ok=True, n=len(keep), runs=runs, dropped=dropped,
                        pooled=pooled, mean=float(np.mean(per)),
                        std=float(np.std(per, ddof=1)) if len(per) > 1 else None,
                        pattern=p)
    return dict(ok=False, n=0, runs=[], dropped=[], pattern=patterns[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--csv", default=None)
    # AREA=LMAX, repeatable. The rule required_ps_lmax() implements is "last cut whose MEAN bias
    # is under 0.3 sigma". A cut whose mean is above 0.3 but whose ERROR BAR still reaches it is a
    # defensible alternative and a DIFFERENT rule, so it has to be stated explicitly rather than
    # folded into the function. Live case: full sky, where lmax 340 gives 0.358 +/- 0.099 -- the
    # mean fails, the 1-sigma interval does not. Whichever is used must be said in the caption.
    ap.add_argument("--ps-lmax", action="append", default=[], metavar="AREA=LMAX",
                    help="override the PS cut for one area, e.g. --ps-lmax fullsky=340")
    a = ap.parse_args()
    areas = [x if x == "fullsky" else int(x) for x in a.areas] if a.areas else AREAS

    stats = [("Power spectrum", "ps"), ("Peak counts", "pc_"), ("L1 norm", "")]
    rows = []
    print("=" * 100)
    print("BARYON-SAFE FoM AUDIT  --  FoM_3 = 1/sqrt(det Cov(Om, sigma_8, w0)), NULL role")
    print("=" * 100)
    override = {}
    for spec in a.ps_lmax:
        k, _, v = spec.partition("=")
        override[k if k == "fullsky" else int(k)] = int(v)

    for A in areas:
        if A in override:
            lmax, why = override[A], None
            auto, _ = required_ps_lmax(A)
            print(f"\n[override] {A}: PS lmax {lmax} (rule would give {auto}) "
                  f"-- justified by the error bar, not the mean; say so in the caption")
        else:
            lmax, why = required_ps_lmax(A)
        print(f"\n### {A} deg^2" if A != "fullsky" else "\n### full sky")
        print(f"    PS baryon-safe lmax : {lmax if lmax else 'N/A -- ' + str(why)}")
        print(f"    HOS scale set       : {HOS_SCALES}")
        base = None
        for name, pref in stats:
            if pref == "ps":
                if lmax is None:
                    print(f"    {name:16s} SKIPPED ({why})")
                    rows.append(dict(statistic=name, area=A, cut="n/a", status=f"no safe cut: {why}"))
                    continue
                r = measure([ps_null_glob(A, lmax)])
                cut = f"l37-{lmax}"
            else:
                r = measure(hos_null_globs(pref, A))
                cut = HOS_SCALES
            if not r["ok"]:
                print(f"    {name:16s} MISSING   ({cut})   no usable posterior")
                print(f"                     looked for {os.path.relpath(r['pattern'], REPO)}")
                rows.append(dict(statistic=name, area=A, cut=cut, status="MISSING",
                                 n_seeds=0, fom3_pooled="", fom3_mean="", fom3_std="", ratio=""))
                continue
            if pref == "ps":
                base = r["mean"]
            ratio = (r["mean"] / base) if base else None
            err = f" +/- {r['std']:.3e}" if r["std"] is not None else "  (1 seed: no error bar)"
            print(f"    {name:16s} {cut:10s} n={r['n']:2d}  "
                  f"FoM3 pooled={r['pooled']:.4e}  per-seed mean={r['mean']:.4e}{err}"
                  + (f"   x{ratio:.2f} vs PS" if ratio else ""))
            if r["dropped"]:
                print(f"                     dropped {len(r['dropped'])}: {r['dropped'][:4]}")
            rows.append(dict(statistic=name, area=A, cut=cut, status="ok", n_seeds=r["n"],
                             runs=" ".join(map(str, r["runs"])),
                             fom3_pooled=f"{r['pooled']:.6e}", fom3_mean=f"{r['mean']:.6e}",
                             fom3_std=(f"{r['std']:.6e}" if r["std"] is not None else ""),
                             ratio=(f"{ratio:.3f}" if ratio else "")))

    miss = [r for r in rows if r.get("status") != "ok"]
    thin = [r for r in rows if r.get("status") == "ok" and r.get("n_seeds", 0) < 3]
    print("\n" + "=" * 100)
    print(f"SUMMARY: {len(rows) - len(miss)} of {len(rows)} cells have a usable posterior")
    if miss:
        print(f"\nMISSING ({len(miss)}) -- these need runs:")
        for r in miss:
            print(f"   {r['statistic']:16s} {str(r['area']):8s} {r.get('cut','')}  {r['status']}")
    if thin:
        print(f"\nTHIN ({len(thin)}) -- present but <3 seeds, so the error bar is weak or absent:")
        for r in thin:
            print(f"   {r['statistic']:16s} {str(r['area']):8s} n_seeds={r['n_seeds']}")
    if a.csv:
        keys = ["statistic", "area", "cut", "status", "n_seeds", "runs",
                "fom3_pooled", "fom3_mean", "fom3_std", "ratio"]
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
