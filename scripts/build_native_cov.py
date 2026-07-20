#!/usr/bin/env python3
"""Build the native (un-rebinned, nlb=4) analytic Gaussian covariance at a per-bin scale cut, matching
a native dump (master worker --rebin 1). Slices the native C_ANA (3830x3830 = 10 spectra x NBPW) to the
cut bins per spectrum (BNT min rule via per_spectrum_uppers). No SSC term (Gaussian only) — sufficient
for ana_whiten noise-whitening. Run with the jaxili interpreter (numpy only)."""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("FISHER_AREA", "14000")
sys.path.insert(0, "scripts")
import score_cut_utils as SC          # noqa: E402  (adds scripts/diagnostics to sys.path)
import fisher_hybrid_cov as H         # noqa: E402


def native_cov(cuts, bnt):
    ups = H.per_spectrum_uppers(list(cuts))               # per-spectrum upper ℓ (BNT min rule)
    Cbase = SC.C_ANA_BNT if bnt else H.C_ANA              # native 3830x3830
    lo = max(0, int((H.LOWER - H.ELL_OFFSET) / H.ELL_PER_BIN))
    keep = []
    for s, up in enumerate(ups):
        hi = min(H.NBPW, int((up - H.ELL_OFFSET) / H.ELL_PER_BIN))
        keep.extend(range(s * H.NBPW + lo, s * H.NBPW + hi))
    keep = np.asarray(keep, dtype=int)
    return Cbase[np.ix_(keep, keep)], keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", required=True, help="4 comma ints, e.g. 480,480,480,480 / 480,1024,1024,1024")
    ap.add_argument("--bnt", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-cache", default=None, help="dump cache.npz to verify dim alignment")
    a = ap.parse_args()
    cuts = [int(c) for c in a.cuts.split(",")]
    C, keep = native_cov(cuts, a.bnt)
    np.save(a.out, C)
    print(f"[native-cov] cuts={cuts} bnt={a.bnt} -> C {C.shape} saved {a.out}")
    if a.check_cache:
        x = np.load(a.check_cache)["x"]
        ok = x.shape[1] == C.shape[0]
        ev = np.linalg.eigvalsh(C)
        print(f"  dump x dim {x.shape[1]} vs cov dim {C.shape[0]} -> {'MATCH' if ok else 'MISMATCH'}")
        print(f"  cov eig range [{ev.min():.2e}, {ev.max():.2e}]  cond {ev.max()/max(ev.min(),1e-300):.2e}")
