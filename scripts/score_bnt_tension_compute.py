#!/usr/bin/env python3
"""Compute the score-BNT bin-1 tension curve from the saved null/biased score-NPE posteriors.

For each cut c and seed s, loads posteriors/cut<c>/{null,biased}_run<s>.npy (each (n_samples, 6))
and computes the Gaussian Q_DM 3-param (Ωm,S₈,w₀) tension via scripts/tension/estimators.py — the
SAME estimator as the raw-NPE campaign, so the curves are directly comparable. Aggregates over
seeds to mean±std and writes long+agg CSVs in the campaign schema
(area,upper_cut,mean,std,n,n_total,n_excluded), so build_bnt_bin1_allareas_plot.py can overlay it
on the standard non-BNT grey curve. Also folds in the per-cut TARP/SBC calibration verdicts.

Run with the tensiometer env:
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/score_bnt_tension_compute.py --area 14000
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = "/mnt/home/tersenov/software/bar_impact"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from tension import aggregate, estimators   # noqa: E402

P3 = (0, 1, 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--area", type=int, default=14000)
    p.add_argument("--tag", default="bnt_ps_bin1_score_l37")
    p.add_argument("--min-runs", type=int, default=1)
    return p.parse_args()


def collect(pdir):
    """One record per (cut, seed) with the 3-param and 6-param Q_DM nσ."""
    rec3, rec6 = [], []
    for cdir in sorted(glob.glob(os.path.join(pdir, "cut*")), key=lambda d: int(re.search(r"cut(\d+)", d).group(1))):
        cut = int(re.search(r"cut(\d+)", cdir).group(1))
        for fn in sorted(glob.glob(os.path.join(cdir, "null_run*.npy"))):
            seed = int(re.search(r"null_run(\d+)", fn).group(1))
            fb = os.path.join(cdir, f"biased_run{seed}.npy")
            if not os.path.exists(fb):
                continue
            null = np.load(fn)
            bias = np.load(fb)
            t3 = estimators.tension_sigma(null, bias, indices=P3, estimator="q_dm")
            t6 = estimators.tension_sigma(null, bias, indices=None, estimator="q_dm")
            r3 = {"area": None, "upper_cut": cut, "run": seed, "nsigma": t3["nsigma"],
                  "Q_DM": t3["Q_DM"], "dofs": t3["dofs"], "ok": t3["ok"]}
            r6 = {"area": None, "upper_cut": cut, "run": seed, "nsigma": t6["nsigma"],
                  "Q_DM": t6["Q_DM"], "dofs": t6["dofs"], "ok": t6["ok"]}
            rec3.append(r3); rec6.append(r6)
    return rec3, rec6


def calibration_table(pdir):
    rows = []
    for fn in glob.glob(os.path.join(pdir, "cut*", "calibration_run*.json")):
        cut = int(re.search(r"cut(\d+)", fn).group(1))
        with open(fn) as fh:
            c = json.load(fh)
        rows.append({"upper_cut": cut, "tarp": c.get("tarp_verdict"),
                     "tarp_dev": c.get("tarp_max_abs_dev"), "sbc": c.get("sbc_verdict"),
                     "sbc_rank_std": c.get("sbc_rank_std")})
    return pd.DataFrame(sorted(rows, key=lambda r: r["upper_cut"]))


def write_tables(records, out_dir, stem, area, min_runs):
    os.makedirs(out_dir, exist_ok=True)
    for r in records:
        r["area"] = area
    long_df = pd.DataFrame(records)
    long_df.to_csv(os.path.join(out_dir, f"{stem}_long.csv"), index=False, float_format="%.6f")
    agg = aggregate.aggregate_runs(long_df, ("area", "upper_cut"), value="nsigma", min_runs=min_runs)
    agg.to_csv(os.path.join(out_dir, f"{stem}_agg.csv"), index=False, float_format="%.5f")
    return agg


def main():
    a = parse_args()
    root = f"{REPO}/outputs/baryon_tension/{a.tag}/area{a.area}"
    pdir = os.path.join(root, "posteriors")
    tdir = os.path.join(root, "tables")
    rec3, rec6 = collect(pdir)
    if not rec3:
        print(f"No posteriors under {pdir}. Nothing to compute."); return
    agg3 = write_tables(rec3, tdir, "tension_3param", a.area, a.min_runs)
    write_tables(rec6, tdir, "tension_6param", a.area, a.min_runs)

    cal = calibration_table(pdir)
    if len(cal):
        cal.to_csv(os.path.join(tdir, "calibration.csv"), index=False)

    print(f"Tables -> {tdir}")
    print("\n3-param score-BNT tension nσ (mean±std over seeds):")
    for _, r in agg3.sort_values("upper_cut").iterrows():
        c = cal[cal["upper_cut"] == r["upper_cut"]] if len(cal) else pd.DataFrame()
        ct = ""
        if len(c):
            ct = f"   TARP={c['tarp'].iloc[0]} SBC={c['sbc'].iloc[0]}"
        sd = r["std"] if pd.notna(r["std"]) else 0.0
        print(f"  cut {int(r['upper_cut']):4d}: {r['mean']:.3f} ± {sd:.3f}  (n={int(r['n'])}){ct}")


if __name__ == "__main__":
    main()
