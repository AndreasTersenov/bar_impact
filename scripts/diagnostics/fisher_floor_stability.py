#!/usr/bin/env python3
"""Is the Fisher-floor FoM3 ratio stable across local-Jacobian settings?

WHY THIS EXISTS — and a retraction. On 2026-07-31 I measured this ratio swinging 1.38 -> 2.41 across
(order, bandwidth) and reported the Fisher floor as untrustworthy, attributing the swing to a shallow
local w0 derivative. That measurement was made against the RAID0-DESTROYED analytic covariance
(gaussian_cov_native_14000.npy, ~20% zeroed, indefinite), which was itself producing nonsense — it
gave BNT460/nonBNT580 = 1.377 where the pre-crash documented value is 1.46. So the reported
instability conflated a genuine Jacobian question with a corrupt covariance, and had to be redone.

This re-measures it with the covariance restored from the intact rebinned cache (which reproduces
the documented 1.46 to 1.455). fisher_local_jacobian's own docstring makes stability across
(order, h) the trustworthiness criterion:

    "the gradient -> Fisher sigma must PLATEAU as h shrinks and across order. order2 should be flat
     in h (curvature removed); order1 should converge to order2 as h->0. If the ... ratio is stable
     across (order, h), it is trustworthy."

So this is the script that decides whether the Fisher floor may be quoted in a figure annotation at
all, or whether only the realized (posterior-measured) ratio is defensible.

  FISHER_AREA=14000 FISHER_REBIN=20 python scripts/diagnostics/fisher_floor_stability.py
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CFGS = {"BNT@460": ([460, 1024, 1024, 1024], True),
        "BNT@580": ([580, 1024, 1024, 1024], True),
        "nonBNT@460": ([460] * 4, False),
        "nonBNT@580": ([580] * 4, False)}
RATIOS = [("BNT@460", "nonBNT@580", "de-biasing pair"),
          ("BNT@580", "nonBNT@580", "matched pair"),
          ("BNT@580", "nonBNT@460", "Fisher-methods pair")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/diagnostics/fisher_floor_stability.csv")
    ap.add_argument("--orders", default="order2,order1_free,order1_anchored")
    ap.add_argument("--bws", default="0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--covk", default="hybrid")
    a = ap.parse_args()

    import score_cut_utils as S
    import fisher_hybrid_cov as H
    orders = [o for o in a.orders.split(",") if o]
    bws = [float(x) for x in a.bws.split(",") if x]
    print(f"=== Fisher-floor stability (AREA={H.AREA} REBIN={H.REBIN} covk={a.covk}) ===\n", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    hdr = f"{'order':17s}{'bw':>6s}" + "".join(f"{n:>13s}" for n, _, _ in RATIOS)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for order in orders:
        for bw in bws:
            try:
                fom = {}
                for name, (cuts, bnt) in CFGS.items():
                    d = S.build_score(cuts, bnt=bnt, covk=a.covk, order=order, bw=bw)
                    fom[name] = d["fom3"]
                cells, rec = [], {"order": order, "bw": bw}
                for num, den, _ in RATIOS:
                    r = fom[num] / fom[den]
                    cells.append(f"{r:13.3f}")
                    rec[f"{num}/{den}"] = r
                for name in CFGS:
                    rec[f"fom3_{name}"] = fom[name]
                rows.append(rec)
                print(f"{order:17s}{bw:6.2f}" + "".join(cells), flush=True)
            except Exception as e:
                print(f"{order:17s}{bw:6.2f}   FAIL {type(e).__name__}: {e}", flush=True)

    if rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out}")

        print("\n=== verdict ===")
        for num, den, label in RATIOS:
            v = np.array([r[f"{num}/{den}"] for r in rows])
            spread = v.max() / v.min()
            o2 = [r for r in rows if r["order"] == "order2"]
            v2 = np.array([r[f"{num}/{den}"] for r in o2]) if o2 else np.array([])
            s2 = (v2.max() / v2.min()) if v2.size else np.nan
            print(f"  {num}/{den} ({label}):")
            print(f"      all settings : {v.min():.3f} - {v.max():.3f}  (spread {spread:.2f}x)")
            if v2.size:
                print(f"      order2 only  : {v2.min():.3f} - {v2.max():.3f}  (spread {s2:.2f}x)")
            verdict = ("STABLE - quotable" if spread < 1.15 else
                       "order2 stable, order1 not - quote order2 only" if s2 < 1.15 else
                       "UNSTABLE - do not quote the Fisher floor as a ceiling")
            print(f"      -> {verdict}")


if __name__ == "__main__":
    main()
