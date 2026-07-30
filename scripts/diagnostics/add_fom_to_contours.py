#!/usr/bin/env python3
"""Compute FoM_3 for every contour in the contour figures, and record it in the sidecars.

FoM_3 = 1 / sqrt(det C_3), where C_3 is the 3x3 covariance of (Omega_m, S8, w0). Same
definition as plot_fom_vs_area.py / plot_scaling_vs_area.py, so numbers are comparable across
figures.

TWO FoM VALUES ARE REPORTED PER CONTOUR, and the difference is not cosmetic:

  fom3_pooled     computed from the pooled samples — i.e. from exactly the covariance the
                  DRAWN CONTOUR represents. Pooling across NPE training seeds folds the
                  seed-to-seed training scatter into the covariance, which inflates it and so
                  LOWERS the FoM. This is the honest figure-of-merit *of the plotted contour*.
  fom3_per_seed   mean +/- std of the per-seed FoM_3. This is what plot_fom_vs_area.py and
                  plot_scaling_vs_area.py plot, so it is the number to quote when comparing
                  against those figures.

Reporting only one would invite a false comparison: pooled is always the smaller of the two,
and by a factor that grows with seed scatter, so a pooled value set beside a per-seed value
from another figure would look like a physical difference when it is a bookkeeping one.

Why this exists as a backfill rather than living only in the generators: the contour figures
take 15-60 minutes each under getdist, and FoM depends only on the samples, not on the render.
This recomputes from the EXACT posteriors each figure's provenance records it used (statistic,
area, role, cut, and the run list), writes a <figure>_fom.csv, and injects a "fom3" block into
the existing <figure>_provenance.json. Re-publish afterwards with `figures.py publish --force`
so paper/figures/ picks up the enriched sidecars and the recorded sha256 stays consistent.
The generators also compute it natively now, so new figures do not need this pass.

  /lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python \
      scripts/diagnostics/add_fom_to_contours.py [--dry-run]
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P3 = [0, 1, 2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # main() is __main__-guarded, so import is safe
    return m


CVA = load_module("scripts/diagnostics/plot_contours_vs_area.py", "cva")
C3S = load_module("scripts/diagnostics/plot_contours_three_stats.py", "c3s")

STAT_KEY = {"Power spectrum": "ps", "Peak counts": "peaks", "L1 norm": "l1"}


def fom3(samples):
    c = np.cov(np.asarray(samples)[:, P3], rowvar=False)
    return float(1.0 / np.sqrt(np.linalg.det(c)))


def sigmas(samples):
    c = np.cov(np.asarray(samples)[:, P3], rowvar=False)
    return [float(np.sqrt(c[i, i])) for i in range(3)]


def run_index(f):
    m = re.search(r"_run(\d+)", f)
    return int(m.group(1)) if m else 1


def arrays_for(pattern, wanted_runs):
    """Load exactly the runs the figure used, so the FoM matches the drawn contour."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        r = run_index(f)
        if wanted_runs and r not in wanted_runs:
            continue
        try:
            out[r] = np.load(f)
        except Exception:
            pass                     # damaged .npy raises ValueError, not an IOError
    return out


def contours_vs_area_jobs(prov):
    """(label, pattern, runs) per contour for a contours_vs_area figure."""
    stat = STAT_KEY[prov["statistic"]]
    role = prov["role"]
    gl = CVA.globber(stat)
    for s in prov["series"]:
        area = s["area"]
        area = area if area == "fullsky" else int(area)
        yield f"{area}", gl(area, role), set(s["runs"])


def three_stats_jobs(prov):
    """(label, pattern, runs) per contour for a contours_three_stats figure."""
    area = int(prov["area_sqdeg"])
    lmax = (prov.get("cuts") or {}).get("power_spectrum_lmax", 1020)
    scales = (prov.get("cuts") or {}).get("hos_scale_tag", "scales1234")
    for s in prov["series"]:
        stat, role = s["statistic"], s["role"]
        if stat == "Power spectrum":
            pats = C3S.ps_globs(area, lmax)
        elif stat == "Peak counts":
            pats = C3S.hos_globs("pc_", area, scales)
        else:
            pats = C3S.hos_globs("", area, scales)
        yield f"{stat} / {role}", pats[0 if role == "null" else 1], set(s["runs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    provs = (sorted(glob.glob(f"{REPO}/outputs/plots/contours_vs_area/*_provenance.json"))
             + sorted(glob.glob(f"{REPO}/outputs/plots/contours_three_stats/*_provenance.json")))
    print(f"{len(provs)} contour figure(s) found\n")

    for pf in provs:
        prov = json.load(open(pf))
        stem = pf[:-len("_provenance.json")]
        name = os.path.basename(stem)
        jobs = (contours_vs_area_jobs(prov) if "statistic" in prov and "role" in prov
                else three_stats_jobs(prov))
        rows = []
        for label, pattern, runs in jobs:
            arrs = arrays_for(pattern, runs)
            if not arrs:
                print(f"  [warn] {name}: no arrays for {label}")
                continue
            per = [fom3(a) for a in arrs.values()]
            pooled = np.concatenate([a for _, a in sorted(arrs.items())])
            sg = sigmas(pooled)
            rows.append({
                "contour": label,
                "n_seeds": len(arrs),
                "runs": sorted(arrs),
                "fom3_pooled": fom3(pooled),
                "fom3_per_seed_mean": float(np.mean(per)),
                "fom3_per_seed_std": float(np.std(per)),
                "sigma_Om_pooled": sg[0], "sigma_S8_pooled": sg[1], "sigma_w0_pooled": sg[2],
            })
        if not rows:
            continue

        print(f"  {name}")
        for r in rows:
            ratio = r["fom3_per_seed_mean"] / r["fom3_pooled"]
            print(f"     {r['contour']:26s} n={r['n_seeds']:2d}  "
                  f"pooled {r['fom3_pooled']:11.4g}   per-seed {r['fom3_per_seed_mean']:11.4g}"
                  f" +/- {r['fom3_per_seed_std']:9.3g}   (x{ratio:.2f})")

        if args.dry_run:
            continue

        with open(stem + "_fom.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            cols = ["contour", "n_seeds", "fom3_pooled", "fom3_per_seed_mean",
                    "fom3_per_seed_std", "sigma_Om_pooled", "sigma_S8_pooled",
                    "sigma_w0_pooled", "runs"]
            w.writerow(cols)
            for r in rows:
                w.writerow([r["contour"], r["n_seeds"], f"{r['fom3_pooled']:.6g}",
                            f"{r['fom3_per_seed_mean']:.6g}", f"{r['fom3_per_seed_std']:.6g}",
                            f"{r['sigma_Om_pooled']:.6g}", f"{r['sigma_S8_pooled']:.6g}",
                            f"{r['sigma_w0_pooled']:.6g}", " ".join(map(str, r["runs"]))])

        prov["fom3"] = {
            "definition": "FoM_3 = 1/sqrt(det C_3), C_3 = covariance of (Omega_m, S8, w0)",
            "pooled_vs_per_seed": (
                "fom3_pooled is computed from the pooled samples, i.e. from the covariance the "
                "DRAWN contour represents; pooling across NPE training seeds folds training "
                "scatter into the covariance and therefore LOWERS the FoM. fom3_per_seed_mean "
                "is the mean of the per-seed FoM and is what plot_fom_vs_area.py and "
                "plot_scaling_vs_area.py plot, so it is the value to use when comparing against "
                "those figures. Do not compare a pooled value against a per-seed one."),
            "per_contour": rows,
        }
        json.dump(prov, open(pf, "w"), indent=2)
        print(f"     -> wrote {os.path.basename(stem)}_fom.csv and updated provenance\n")


if __name__ == "__main__":
    main()
