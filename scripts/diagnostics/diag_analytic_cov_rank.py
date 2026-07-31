#!/usr/bin/env python3
"""Why does covk='analytic' go singular, and does it block a native-resolution MOPED run?

Context. score_cut_utils.build_score(..., covk='analytic') raises LinAlgError('Singular matrix') at
rebin=20 with only 60-96 features, while covk='hybrid' (analytic + top-3 eigenmodes of
C_sample - C_analytic) inverts fine. That matters beyond a config flag: a native (un-rebinned) MOPED
run CANNOT use the hybrid covariance, because its low-rank correction is estimated from 200
permutations and at ~2500 features that difference is pure estimation noise. Native therefore
*requires* a usable analytic covariance. If the analytic path is structurally rank-deficient, native
MOPED is blocked until it is regularized; if the deficiency is a handful of numerically-tiny modes,
a ridge fixes it and the native run is viable.

This script does not fix anything. It measures, at rebin=20:
  1. the native 3830x3830 C_ANA: rank, eigenvalue spectrum, how many modes are at/below noise;
  2. the same for the BNT-transformed native covariance (T (x) I) C (T (x) I)^T;
  3. the cut+rebinned analytic C at the four production configs, with the rank deficit localised;
  4. what the hybrid actually adds, i.e. whether its top-3 modes coincide with the null directions;
  5. the smallest ridge (relative to median diag) that restores a usable condition number.

Run under jaxili (numpy only):
  FISHER_AREA=14000 FISHER_REBIN=20 python scripts/diagnostics/diag_analytic_cov_rank.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # scripts/ for score_cut_utils

import fisher_hybrid_cov as H          # noqa: E402
import fisher_local_jacobian as L      # noqa: E402
import score_cut_utils as S            # noqa: E402

CFGS = [("nonBNT@460", [460] * 4, False), ("nonBNT@580", [580] * 4, False),
        ("BNT@460", [460, 1024, 1024, 1024], True), ("BNT@580", [580, 1024, 1024, 1024], True)]


def spectrum(C, label, tol_rel=1e-12):
    """Eigen-spectrum summary. Rank is derived from the eigenvalues rather than np.linalg.matrix_rank
    — the matrix is symmetric PSD by construction, so a second SVD would double the cost of a
    3830^3 decomposition for no extra information."""
    ev = np.linalg.eigvalsh(C)
    ev_sorted = np.sort(ev)
    tol = tol_rel * ev.max()
    rank = int((np.abs(ev) > tol).sum())          # numpy's matrix_rank convention, from eigenvalues
    n_small = int((ev <= tol).sum())
    n_neg = int((ev < 0).sum())
    cond = ev.max() / max(ev[ev > tol].min(), 1e-300) if (ev > tol).any() else np.inf
    print(f"  {label:22s} dim={C.shape[0]:5d} rank={rank:5d} "
          f"eig[min,max]=[{ev.min():.3e},{ev.max():.3e}] "
          f"n(<=1e-12*max)={n_small:4d} n(<0)={n_neg:4d} cond={cond:.3e}", flush=True)
    print(f"      5 smallest: {np.array2string(ev_sorted[:5], precision=3)}", flush=True)
    return ev


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-native", action="store_true",
                    help="skip steps [1]-[2] (two 3830^3 eigendecompositions, minutes each). The "
                         "cut-level steps [3]-[5] are what actually gate the native-MOPED decision.")
    a = ap.parse_args()

    print(f"=== analytic covariance rank diagnosis (AREA={H.AREA}, REBIN={H.REBIN}) ===\n", flush=True)
    C_bnt = S._build_C_ANA_BNT()

    if a.skip_native:
        print("[1][2] skipped (--skip-native)\n", flush=True)
    else:
        print("[1] native analytic Gaussian covariance, non-BNT basis", flush=True)
        spectrum(H.C_ANA, "C_ANA (native)")
        print(f"      exact-zero entries: {float((H.C_ANA == 0).mean()):.3f} "
              "(block structure, not damage)\n", flush=True)

        print("[2] native analytic covariance, BNT basis  (T (x) I) C (T (x) I)^T", flush=True)
        spectrum(C_bnt, "C_ANA_BNT (native)")
        print(flush=True)

    print("[3] cut+rebinned analytic C at the production configs", flush=True)
    for name, cuts, bnt in CFGS:
        R = H.cut_rebin_R(H.per_spectrum_uppers(list(cuts)))
        Cana = R @ (C_bnt if bnt else H.C_ANA) @ R.T
        ev = spectrum(Cana, name)
        # localise: which spectrum blocks do the null directions live in?
        w, V = np.linalg.eigh(Cana)
        tol = 1e-12 * w.max()
        null = V[:, w <= tol]
        if null.shape[1]:
            nb = [S.n_bands(u) for u in H.per_spectrum_uppers(list(cuts))]
            edges = np.cumsum([0] + nb)
            weight = [float((null[edges[s]:edges[s + 1]] ** 2).sum()) for s in range(10)]
            top = np.argsort(weight)[::-1][:4]
            print("      null-space weight by spectrum: "
                  + ", ".join(f"{H.SPECTRA[s]}:{weight[s]/max(sum(weight),1e-300):.2f}" for s in top))
    print()

    print("[4] what the hybrid adds (does it plug the null space?)")
    for name, cuts, bnt in CFGS:
        R = H.cut_rebin_R(H.per_spectrum_uppers(list(cuts)))
        Cana = R @ (C_bnt if bnt else H.C_ANA) @ R.T
        fa, fc, nell = L.load_set("fiducial", "nobaryons", bnt)
        perms = L.datavector(fa, fc, nell, list(cuts))
        Csamp = np.cov(perms, rowvar=False)
        D = Csamp - Cana
        ev, V = np.linalg.eigh(D)
        idx = np.argsort(ev)[::-1][:3]
        Chyb = Cana + (V[:, idx] * ev[idx]) @ V[:, idx].T
        ea, eh = np.linalg.eigvalsh(Cana), np.linalg.eigvalsh(Chyb)
        print(f"  {name:12s} analytic min-eig={ea.min():.3e} cond={ea.max()/max(ea.min(),1e-300):.2e}"
              f"  ->  hybrid min-eig={eh.min():.3e} cond={eh.max()/max(eh.min(),1e-300):.2e}"
              f"  | top-3 D eigs={np.array2string(ev[idx], precision=2)}")
        print(f"      sample min-eig={np.linalg.eigvalsh(Csamp).min():.3e} "
              f"(200 perms, {Csamp.shape[0]} feat)")
    print()

    print("[5] smallest relative ridge restoring cond < 1e10 on the analytic C")
    for name, cuts, bnt in CFGS:
        R = H.cut_rebin_R(H.per_spectrum_uppers(list(cuts)))
        Cana = R @ (C_bnt if bnt else H.C_ANA) @ R.T
        med = np.median(np.diag(Cana))
        got = None
        for r in [0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
            Cr = Cana + r * med * np.eye(Cana.shape[0])
            ev = np.linalg.eigvalsh(Cr)
            if ev.min() > 0 and ev.max() / ev.min() < 1e10:
                got = r
                break
        print(f"  {name:12s} ridge_rel={got}  (median diag={med:.3e})")


if __name__ == "__main__":
    main()
