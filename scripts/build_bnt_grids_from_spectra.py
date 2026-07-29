#!/usr/bin/env python3
"""Build BNT power-spectrum grids by transforming the existing (submean) auto+cross grids.

Validated shortcut (see docs/BNT_on_spectra.md): on a single shared mask, the BNT transform
commutes with mask + MASTER decouple + bandpower bin, so the BNT-transformed spectra are exactly
C~(l) = M C(l) M^T applied to the already-produced spectra — no map reprocessing. The numerical
oracle (scripts/diagnostics/bnt_on_spectra_oracle.py) confirmed this to 1e-11 (roundoff).

This reads, per area and per simulation type, the 4 auto grids + 1 cross grid (6 pairs), assembles
the 4x4 tomographic matrix C(l), applies M C M^T, and writes BNT autos + BNT crosses under the
filenames the worker's `--bnt` path expects (all_cls_ -> all_bnt_cls_, all_cross_cls_ ->
all_bnt_cross_cls_). Pure numpy; run with any interpreter that has numpy.

  /home/tersenov/anaconda3/envs/aname/bin/python scripts/build_bnt_grids_from_spectra.py
"""
import os
import sys

import numpy as np

DD = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
M = np.array([[1., 0., 0., 0.],
              [-1., 1., 0., 0.],
              [0.4521097, -1.4521097, 1., 0.],
              [0., 0.25127807, -1.251278, 1.]])
PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
TAIL = "apod2.0_master_submean_noisy_s0.26_lmax1535"

# (subdir, sim_type, kind):  grid exists only for nobaryons; fiducials for both.
JOBS = [("new_grid", "nobaryons", "grid")] + \
       [("fiducial", t, "fiducial") for t in ("nobaryons", "baryonified")]


def auto_name(kind, sim, b, area, bnt=False):
    pre = "all_bnt_cls" if bnt else "all_cls"
    return f"{pre}_{kind}_{sim}_bin{b}_masked_{area}sqdeg_{TAIL}.npy"


def cross_name(kind, sim, area, bnt=False):
    pre = "all_bnt_cross_cls" if bnt else "all_cross_cls"
    return f"{pre}_{kind}_{sim}_bins1234_masked_{area}sqdeg_{TAIL}.npy"


def assemble_C(autos, cross, nell):
    """autos: list of 4 (nrow,nell); cross: (nrow, 6*nell) -> C (nrow,nell,4,4)."""
    nrow = autos[0].shape[0]
    C = np.zeros((nrow, nell, 4, 4))
    for b in range(4):
        C[:, :, b, b] = autos[b]
    for k, (i, j) in enumerate(PAIRS):
        blk = cross[:, k * nell:(k + 1) * nell]
        C[:, :, i - 1, j - 1] = blk
        C[:, :, j - 1, i - 1] = blk
    return C


def main():
    Minv = np.linalg.inv(M)
    worst_roundtrip = 0.0
    for area in AREAS:
        for subdir, sim, kind in JOBS:
            d = os.path.join(DD, subdir)
            apaths = [os.path.join(d, auto_name(kind, sim, b, area)) for b in (1, 2, 3, 4)]
            cpath = os.path.join(d, cross_name(kind, sim, area))
            missing = [p for p in apaths + [cpath] if not os.path.exists(p)]
            if missing:
                print(f"  SKIP {sim:11s} {kind:8s} area={area}: missing {len(missing)} input(s) "
                      f"e.g. {os.path.basename(missing[0])}")
                continue
            autos = [np.load(p, allow_pickle=True) for p in apaths]
            cross = np.load(cpath, allow_pickle=True)
            nell = autos[0].shape[1]
            C = assemble_C(autos, cross, nell)
            Ct = np.einsum("ai,peij,bj->peab", M, C, M)          # M C M^T

            # round-trip check: M^-1 C~ M^-T == C
            back = np.einsum("ai,peij,bj->peab", Minv, Ct, Minv)
            rt = np.abs(back - C).max() / (np.abs(C).max() + 1e-30)
            worst_roundtrip = max(worst_roundtrip, rt)

            # write BNT autos (diagonal) + BNT crosses (off-diagonal, PAIRS order)
            for b in range(4):
                np.save(os.path.join(d, auto_name(kind, sim, b + 1, area, bnt=True)),
                        Ct[:, :, b, b])
            cblocks = [Ct[:, :, i - 1, j - 1] for (i, j) in PAIRS]
            np.save(os.path.join(d, cross_name(kind, sim, area, bnt=True)),
                    np.concatenate(cblocks, axis=1))
            print(f"  OK   {sim:11s} {kind:8s} area={area:5d}  nrow={C.shape[0]:5d} nell={nell} "
                  f"roundtrip={rt:.1e}")

    print(f"\n[build] worst round-trip residual (M^-1 C~ M^-T vs C) = {worst_roundtrip:.2e}")

    # sanity: reproduce the bin-1 baryon localization from the WRITTEN fiducial BNT grids (14000)
    print("[build] verify baryon localization from written BNT fiducials (14000, high l):")
    a = 14000
    fb = {t: [np.load(os.path.join(DD, "fiducial", auto_name("fiducial", t, b, a, bnt=True)),
                      allow_pickle=True).mean(0) for b in (1, 2, 3, 4)]
          for t in ("nobaryons", "baryonified")}
    nell = fb["nobaryons"][0].shape[0]
    ell = 2 + 4 * np.arange(nell) + 1.5
    hi = (ell >= 600) & (ell < 1024)
    ratios = [np.nanmedian(fb["baryonified"][b][hi] / fb["nobaryons"][b][hi]) for b in range(4)]
    print("   BNT-auto baryon ratio (bary/nobary):",
          "  ".join(f"bin{b+1}={ratios[b]:.4f}" for b in range(4)),
          "  -> expect bin1~0.995, bins2-4~0.999")


if __name__ == "__main__":
    main()
