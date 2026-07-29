"""Fisher audit — noise-averaged Jacobian (the methodologically-correct fix).

The grid has ~7 realizations per cosmology. The current J is fit on the noisy per-realization C_ell,
so shape noise leaks into the derivative (Phase 0: low-R^2 high-ell modes drive the BNT gain). Here we
average the ~7 realizations per cosmology FIRST (reduces noise in the signal by ~sqrt(7)), fit J on the
~2424 cleaned cosmologies, and keep the noisy 200-perm covariance for C. This is J = d(signal)/dtheta,
C = covariance-with-noise.

Discriminating test: if the high-ell modes carry a REAL (just noisily-estimated) derivative, averaging
raises their R^2 and the BNT gain stabilizes; if their derivative is genuinely ~0, R^2 stays low and the
gain stays an artifact. Reports the BNT-580/non-BNT-460 result under noisy-J vs averaged-J, with R^2 gating.
numpy only.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import load_set, datavector, DD, NPERM  # noqa: E402

PARAMS = np.load(f"{DD}/grid/cosmo_params.npy", allow_pickle=True).astype(float)


def jac_and_r2(th, dv):
    thc = th - th.mean(0)
    dvc = dv - dv.mean(0)
    J, *_ = np.linalg.lstsq(thc, dvc, rcond=None)
    r2 = 1.0 - ((dvc - thc @ J) ** 2).sum(0) / ((dvc ** 2).sum(0) + 1e-300)
    return J, r2


def build(bin_cuts, bnt):
    ga, gc, nell = load_set("new_grid", "nobaryons", bnt)
    fa, fc, _ = load_set("fiducial", "nobaryons", bnt)
    dv_g = datavector(ga, gc, nell, bin_cuts)
    dv_f = datavector(fa, fc, nell, bin_cuts)
    ok = np.isfinite(dv_g).all(1) & np.isfinite(PARAMS).all(1)
    dv_g, th = dv_g[ok], PARAMS[ok]
    C = np.cov(dv_f, rowvar=False)
    # noisy Jacobian (per-realization)
    J_noisy, r2_noisy = jac_and_r2(th, dv_g)
    # noise-averaged Jacobian (mean over ~7 realizations per cosmology)
    keys = np.round(th, 8)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    nc = uniq.shape[0]
    dv_avg = np.zeros((nc, dv_g.shape[1]))
    np.add.at(dv_avg, inv, dv_g)
    dv_avg /= np.bincount(inv)[:, None]
    J_avg, r2_avg = jac_and_r2(uniq, dv_avg)
    return C, (J_noisy, r2_noisy), (J_avg, r2_avg)


def fsig(J, C, mask=None):
    if mask is not None:
        J, C = J[:, mask], C[np.ix_(mask, mask)]
    nf = J.shape[1]
    h = (NPERM - nf - 2) / (NPERM - 1)
    if h <= 0:
        return None
    return np.sqrt(np.diag(np.linalg.inv(J @ (h * np.linalg.inv(C)) @ J.T)))


def main():
    cfgs = {"bnt_580": ([580, 1024, 1024, 1024], True),
            "nonbnt_460": ([460, 460, 460, 460], False)}
    D = {t: build(c, b) for t, (c, b) in cfgs.items()}

    print("=== R² of the Jacobian: noisy (per-realization) vs noise-averaged (~7/cosmo) ===")
    for t in cfgs:
        C, (Jn, r2n), (Ja, r2a) = D[t]
        print(f"{t}: median R²  noisy={np.median(r2n):.3f} -> averaged={np.median(r2a):.3f} | "
              f"frac>0.3 {np.mean(r2n>0.3):.2f} -> {np.mean(r2a>0.3):.2f} | "
              f"frac<0.05 {np.mean(r2n<0.05):.2f} -> {np.mean(r2a<0.05):.2f}")

    for which, idx in [("NOISY-J (current)", 0), ("NOISE-AVERAGED-J", 1)]:
        print(f"\n=== {which}: BNT-580 / non-BNT-460 ===")
        Jb, r2b = D["bnt_580"][1 + idx]
        Cb = D["bnt_580"][0]
        Jc, r2c = D["nonbnt_460"][1 + idx]
        Cc = D["nonbnt_460"][0]
        for g in [0.0, 0.1, 0.3, 0.5]:
            mb, mc = r2b > g, r2c > g
            sb, sc = fsig(Jb, Cb, mb), fsig(Jc, Cc, mc)
            if sb is None or sc is None:
                print(f"  R²>{g}: n_feat too high for Hartlap"); continue
            print(f"  R²>{g}: BNT nf={mb.sum():>3} σS8={sb[1]:.4f} | nB nf={mc.sum():>3} "
                  f"σS8={sc[1]:.4f} | area ratio {(sb[0]*sb[1])/(sc[0]*sc[1]):.3f}")
    print("\n(NPE gives ~0.79. If averaged-J keeps a low ratio that's STABLE under gating, the BNT")
    print(" gain is real; if averaging raises R² but the ratio rises to ~0.8, the gain was noise.)")


if __name__ == "__main__":
    main()
