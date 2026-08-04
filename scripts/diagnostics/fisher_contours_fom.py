#!/usr/bin/env python3
"""Phase III deliverable: overlaid Fisher contours (Om, S8, w0) for BNT-580 vs non-BNT-460, and the
3-parameter FoM. Production setup = local order-2 Jacobian at the fiducial (Phase II) + estimation-
noise-free covariance (Phase I analytic Gaussian, and the hybrid = analytic + low-rank SSC/cNG).
BNT covariance is propagated exactly: Cov(C~) = (T x I) Cov(C) (T x I)^T.

FoM3 = 1/sqrt(det C3) over the MARGINALIZED 3x3 (Om,S8,w0) block of inv(F) (higher = tighter).
Triangle plot: 1D Gaussians on the diagonal, 1-sigma(68.3%)/2-sigma(95.4%) ellipses off-diagonal,
centered on the fiducial. numpy + matplotlib (run with cosmostat_new python)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fisher_local_jacobian as L          # noqa: E402  (build_config, local_jacobian, FID)
import fisher_hybrid_cov as H              # noqa: E402  (C_ANA, bnt operator, cut_rebin_R, ...)

NAMES = ["Om", "S8", "w0", "H0", "ns", "Ob"]
P3 = [0, 1, 2]                              # Om, S8, w0
FID = L.FID
OUT = "outputs/diagnostics/fisher_cov"
os.makedirs(OUT, exist_ok=True)

Tfull = np.kron(H.bnt_spectra_operator(), np.eye(H.NBPW))
C_ANA_BNT = Tfull @ H.C_ANA @ Tfull.T

CFG = {"nonbnt_460": ([460, 460, 460, 460], False),
       "bnt_580":    ([580, 1024, 1024, 1024], True)}
LC = {k: L.build_config(cuts, bnt) for k, (cuts, bnt) in CFG.items()}


def analytic_native(bnt):
    return C_ANA_BNT if bnt else H.C_ANA


def cov_for(name, cov_kind, order="order2", h=0.75):
    cuts, bnt = CFG[name]
    R = H.cut_rebin_R(H.per_spectrum_uppers(cuts))
    C = R @ analytic_native(bnt) @ R.T
    if cov_kind.startswith("hybrid"):
        k = int(cov_kind.split("k")[1])
        fa, fc, nell = L.load_set("fiducial", "nobaryons", bnt)
        dv = L.datavector(fa, fc, nell, cuts)
        D = np.cov(dv, rowvar=False) - C
        ev, V = np.linalg.eigh(D)
        idx = np.argsort(ev)[::-1][:k]
        C = C + (V[:, idx] * ev[idx]) @ V[:, idx].T
    J, _ = L.local_jacobian(LC[name]["grid_avg"], LC[name]["ucos"], LC[name]["fid_mean"], order, h)
    F = J.T @ np.linalg.inv(C) @ J
    return np.linalg.inv(F)                 # full 6x6 parameter covariance


def fom3(cov6):
    c3 = cov6[np.ix_(P3, P3)]
    return 1.0 / np.sqrt(np.linalg.det(c3)), np.sqrt(np.diag(cov6))[P3]


def area2(cov6, i, j):
    s = cov6[np.ix_([i, j], [i, j])]
    return np.pi * 1.515 ** 2 * np.sqrt(np.linalg.det(s))   # 68.3% 2D ellipse area


# ----------------------------- FoM table -----------------------------
print("=== 3-parameter Fisher FoM (Om, S8, w0), 14000 deg^2, local order-2 J ===\n")
print(f"{'cov':10s} {'config':12s} {'sig(Om)':>9}{'sig(S8)':>9}{'sig(w0)':>9}"
      f"{'A(Om,S8)':>11}{'FoM3':>11}")
COVS = {}
for cov_kind in ["analytic", "hybrid_k3", "hybrid_k5"]:
    fom = {}
    for name in CFG:
        cov6 = cov_for(name, cov_kind)
        COVS[(cov_kind, name)] = cov6
        f3, sig = fom3(cov6)
        fom[name] = f3
        print(f"{cov_kind:10s} {name:12s} {sig[0]:>9.4f}{sig[1]:>9.4f}{sig[2]:>9.4f}"
              f"{area2(cov6,0,1):>11.3e}{f3:>11.3e}")
    print(f"{'':10s} {'-> BNT/nonBNT ':12s} "
          f"area(Om,S8) x{area2(COVS[(cov_kind,'bnt_580')],0,1)/area2(COVS[(cov_kind,'nonbnt_460')],0,1):.3f}"
          f"   FoM3 x{fom['bnt_580']/fom['nonbnt_460']:.2f}\n")

# ----------------------------- triangle contour plot -----------------------------
PLOT_COV = "hybrid_k3"
cov_non = COVS[(PLOT_COV, "nonbnt_460")]
cov_bnt = COVS[(PLOT_COV, "bnt_580")]
styles = [("non-BNT  (cut all bins @ ℓ460)", cov_non, "#7f7f7f"),
          ("BNT  (bin1 @ ℓ580, bins2-4 full)", cov_bnt, "#c0392b")]
labels = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$"]

fig, axes = plt.subplots(3, 3, figsize=(8.2, 8.0))
for a in range(3):
    for b in range(3):
        ax = axes[a, b]
        if b > a:
            ax.axis("off"); continue
        ia, ib = P3[a], P3[b]
        if a == b:                                          # 1D marginal Gaussian
            for _, cov, col in styles:
                s = np.sqrt(cov[ia, ia]); x = np.linspace(-4 * s, 4 * s, 200)
                ax.plot(FID[ia] + x, np.exp(-0.5 * (x / s) ** 2), color=col, lw=1.8)
            ax.set_yticks([])
        else:                                               # 2D ellipses (1sigma, 2sigma)
            for _, cov, col in styles:
                sub = cov[np.ix_([ib, ia], [ib, ia])]       # x=param b, y=param a
                w, V = np.linalg.eigh(sub)
                ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
                for nsig, lw, al in [(2.486, 1.0, 0.25), (1.515, 1.9, 0.9)]:
                    ax.add_patch(Ellipse((FID[ib], FID[ia]), 2 * nsig * np.sqrt(w[1]),
                                         2 * nsig * np.sqrt(w[0]), angle=ang, fill=False,
                                         edgecolor=col, lw=lw, alpha=al))
            ax.plot(FID[ib], FID[ia], "k+", ms=7, mew=1.2)
            ax.relim(); ax.autoscale_view()
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=8)
        if a == 2:
            ax.set_xlabel(labels[b], fontsize=12)
        else:
            ax.set_xticklabels([])
        if b == 0 and a != 0:
            ax.set_ylabel(labels[a], fontsize=12)
        elif b == 0:
            ax.set_ylabel("")
axes[0, 0].legend([plt.Line2D([], [], color=c, lw=2) for _, _, c in styles],
                  [s[0] for s in styles], loc="upper left",
                  bbox_to_anchor=(1.05, 1.0), fontsize=9, frameon=False)
fom_n, _ = fom3(cov_non); fom_b, _ = fom3(cov_bnt)
fig.suptitle("Proper Fisher 68%/95% contours — BNT vs non-BNT  (14000 deg$^2$, auto+cross PS)\n"
             f"local order-2 J + hybrid covariance (analytic+SSC/cNG);  "
             f"FoM3 ratio BNT/non-BNT = {fom_b/fom_n:.1f}$\\times$", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
p = f"{OUT}/fisher_contours_bnt_vs_nonbnt_14000.png"
fig.savefig(p, dpi=140); print("saved", p)
