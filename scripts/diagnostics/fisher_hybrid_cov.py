#!/usr/bin/env python3
"""Phase II/III close-out: the BNT-580/non-BNT-460 Fisher ratio under an ESTIMATION-NOISE-FREE
covariance, resolving the Hartlap(0.50) vs Percival(0.72) ambiguity.

The 200-perm sample covariance forces a finite-N correction (Hartlap/Percival) that penalizes the
higher-dimensional BNT vector (96 feat, m1=1.77) far more than non-BNT (50 feat, m1=1.24) -- so the
ratio depends on the correction, which is exactly the ambiguity we want gone. Phase I gives the cure:
an analytic mask-aware Gaussian covariance (validated to 1-2% on the diagonal) carries NO estimation
noise, and the non-Gaussian excess is a low-rank SSC+cNG correction we can add from the sims.

Covariances compared (local J from Phase II; F = J^T Cinv J; NO Hartlap/Percival on the analytic/
hybrid -- they are not 200-sample estimates):
  - analytic Gaussian (pure): clean, but omits the ~25-30% SSC+cNG (too tight in absolute terms; the
    RATIO is the clean quantity since both configs miss the same physics).
  - hybrid = analytic + top-k eigenmodes of (C_sample - C_analytic): the production-grade TRUE
    covariance (full non-Gaussianity, negligible estimation noise).
BNT propagation is exact: BNT bandpowers are linear combos of the originals, so Cov(C~) = (T x I)
Cov(C) (T x I)^T with T the per-bandpower 10x10 map induced by the 4x4 BNT matrix M (C~ = M C M^T).

Oracle: BNT-full == non-BNT-full under the analytic covariance too. Run with cosmostat_new or jaxili
python (numpy only; loads the Phase-I cached analytic cov).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import load_set, datavector, LOWER, ELL_OFFSET, ELL_PER_BIN, REBIN, PAIRS  # noqa: E402
import fisher_local_jacobian as L  # noqa: E402

NBPW = 383
AREA = int(os.environ.get("FISHER_AREA", "14000"))   # footprint selector (env-threaded for the rollout)
C_ANA = np.load(os.path.join(HERE, "cache_gaussian_cov", f"gaussian_cov_native_{AREA}.npy"))
M = np.array([[1., 0., 0., 0.],
              [-1., 1., 0., 0.],
              [0.4521097, -1.4521097, 1., 0.],
              [0., 0.25127807, -1.251278, 1.]])
# native-cov spectrum order (must match fisher_gaussian_cov.SPECTRA = AUTOS + PAIRS)
SPECTRA = [(1, 1), (2, 2), (3, 3), (4, 4)] + PAIRS


def bnt_spectra_operator():
    """10x10 T: maps the original 10-spectrum bandpower vector to the BNT one (C~ = M C M^T)."""
    T = np.zeros((10, 10))
    for col, (i, j) in enumerate(SPECTRA):                 # original spectrum (i,j) as a unit input
        Cmat = np.zeros((4, 4))
        Cmat[i - 1, j - 1] = 1.0
        Cmat[j - 1, i - 1] = 1.0
        Ct = M @ Cmat @ M.T
        for row, (a, b) in enumerate(SPECTRA):
            T[row, col] = Ct[a - 1, b - 1]
    return T


def cut_rebin_R(upper_per_spectrum):
    """Block-diagonal R mapping native [10 spectra x 383] to the cut+rebinned data vector,
    matching datavector(): autos cut at their bin's upper; cross at min(bin_i, bin_j)."""
    rows = []
    cols_total = 10 * NBPW
    blocks = []
    for s, (i, j) in enumerate(SPECTRA):
        upper = upper_per_spectrum[s]
        lo = max(0, int((LOWER - ELL_OFFSET) / ELL_PER_BIN))
        hi = min(NBPW, int((upper - ELL_OFFSET) / ELL_PER_BIN))
        n = (hi - lo) // REBIN
        Rb = np.zeros((n, NBPW))
        for k in range(n):
            Rb[k, lo + k * REBIN: lo + (k + 1) * REBIN] = 1.0 / REBIN
        blocks.append(Rb)
    nrow = sum(b.shape[0] for b in blocks)
    R = np.zeros((nrow, cols_total))
    r0 = 0
    for s, Rb in enumerate(blocks):
        R[r0:r0 + Rb.shape[0], s * NBPW:(s + 1) * NBPW] = Rb
        r0 += Rb.shape[0]
    return R


def per_spectrum_uppers(bin_cuts):
    ups = list(bin_cuts[:4])
    for (i, j) in PAIRS:
        ups.append(min(bin_cuts[i - 1], bin_cuts[j - 1]))
    return ups                                              # length 10, in SPECTRA order


def area_from_F(F):
    cov = np.linalg.inv(F)
    return np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]), np.sqrt(np.diag(cov))


def fisher(J, C):
    return J.T @ np.linalg.inv(C) @ J


def main():
    Tspec = bnt_spectra_operator()
    Tfull = np.kron(Tspec, np.eye(NBPW))                    # 3830 x 3830 (spectrum-major ordering)
    C_ana_bnt = Tfull @ C_ANA @ Tfull.T                     # exact BNT native analytic covariance

    cfgs = {
        "nonbnt_full": ([1024, 1024, 1024, 1024], False),
        "bnt_full":    ([1024, 1024, 1024, 1024], True),
        "nonbnt_460":  ([460, 460, 460, 460], False),
        "bnt_580":     ([580, 1024, 1024, 1024], True),
    }
    LC = {k: L.build_config(cuts, bnt) for k, (cuts, bnt) in cfgs.items()}

    def analytic_cov(name, bin_cuts, bnt):
        R = cut_rebin_R(per_spectrum_uppers(bin_cuts))
        base = C_ana_bnt if bnt else C_ANA
        return R @ base @ R.T

    def sample_cov(bin_cuts, bnt):
        fa, fc, nell = load_set("fiducial", "nobaryons", bnt)
        dv = datavector(fa, fc, nell, bin_cuts)
        return np.cov(dv, rowvar=False)

    def Jof(name, order, h):
        J, _ = L.local_jacobian(LC[name]["grid_avg"], LC[name]["ucos"], LC[name]["fid_mean"], order, h)
        return J                                            # (nfeat, 6) dDV/dtheta, unwhitened

    # ---- oracle: BNT-full == non-BNT-full under the analytic covariance ----
    Cf_non = analytic_cov("nonbnt_full", cfgs["nonbnt_full"][0], False)
    Cf_bnt = analytic_cov("bnt_full", cfgs["bnt_full"][0], True)
    s_non, _ = area_from_F(fisher(Jof("nonbnt_full", "order2", 0.75), Cf_non))
    s_bnt, _ = area_from_F(fisher(Jof("bnt_full", "order2", 0.75), Cf_bnt))
    print(f"=== ORACLE (analytic cov)  BNT-full vs non-BNT-full area: {s_non:.4e} vs {s_bnt:.4e}  "
          f"rel diff {abs(s_bnt-s_non)/s_non:.1e} ===\n")

    print("Covariance         | order2,h=0.75            | order1_free,h=1.0")
    print("                   | nonBNT460  BNT580  ratio | nonBNT460  BNT580  ratio")
    for cov_kind in ["analytic", "hybrid_k3", "hybrid_k5"]:
        cells = []
        for order, h in [("order2", 0.75), ("order1_free", 1.0)]:
            Cn = analytic_cov("nonbnt_460", cfgs["nonbnt_460"][0], False)
            Cb = analytic_cov("bnt_580", cfgs["bnt_580"][0], True)
            if cov_kind.startswith("hybrid"):
                k = int(cov_kind.split("k")[1])
                for (C0, cuts, bnt) in [(Cn, cfgs["nonbnt_460"][0], False),
                                        (Cb, cfgs["bnt_580"][0], True)]:
                    Cs = sample_cov(cuts, bnt)
                    Dn = Cs - C0
                    ev, V = np.linalg.eigh(Dn)
                    idx = np.argsort(ev)[::-1][:k]
                    low = (V[:, idx] * ev[idx]) @ V[:, idx].T
                    if bnt:
                        Cb = C0 + low
                    else:
                        Cn = C0 + low
            an_n, _ = area_from_F(fisher(Jof("nonbnt_460", order, h), Cn))
            an_b, _ = area_from_F(fisher(Jof("bnt_580", order, h), Cb))
            cells.append((an_n, an_b, an_b / an_n))
        c0, c1 = cells
        print(f"{cov_kind:18s} | {c0[0]:.3e} {c0[1]:.3e} {c0[2]:.3f} | "
              f"{c1[0]:.3e} {c1[1]:.3e} {c1[2]:.3f}")

    print("\nReference ratios: global-lstsq=0.37 | local-J Hartlap=0.50 | local-J Percival=0.72 | NPE=0.79")
    print("Analytic/hybrid carry NO estimation-noise penalty; the hybrid adds the real SSC+cNG.")


if __name__ == "__main__":
    main()
