#!/usr/bin/env python3
"""Proper Fisher triangle (Om, S8, w0): BNT-580 vs non-BNT-460, rendered with getdist for
publication-quality shared-range filled contours. Production setup = local order-2 Jacobian
(Phase II) + hybrid covariance (analytic Gaussian + low-rank SSC/cNG, Phase I/III); BNT covariance
propagated exactly via (T x I) Cov(C) (T x I)^T. Run with jaxili or aname python."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fisher_local_jacobian as L          # noqa: E402
import fisher_hybrid_cov as H              # noqa: E402
from getdist import plots
from getdist.gaussian_mixtures import GaussianND

P3 = [0, 1, 2]
FID3 = L.FID[P3]
NAMES3 = ["Om", "S8", "w0"]
LABELS3 = [r"\Omega_m", r"\sigma_8", "w_0"]
OUT = "outputs/diagnostics/fisher_cov"

Tfull = np.kron(H.bnt_spectra_operator(), np.eye(H.NBPW))
C_ANA_BNT = Tfull @ H.C_ANA @ Tfull.T
CFG = {"nonbnt_460": ([460, 460, 460, 460], False),
       "bnt_580":    ([580, 1024, 1024, 1024], True)}
LC = {k: L.build_config(cuts, bnt) for k, (cuts, bnt) in CFG.items()}


def cov3(name, k=3, order="order2", h=0.75):
    cuts, bnt = CFG[name]
    R = H.cut_rebin_R(H.per_spectrum_uppers(cuts))
    C = R @ (C_ANA_BNT if bnt else H.C_ANA) @ R.T
    fa, fc, nell = L.load_set("fiducial", "nobaryons", bnt)
    dv = L.datavector(fa, fc, nell, cuts)
    D = np.cov(dv, rowvar=False) - C
    ev, V = np.linalg.eigh(D)
    idx = np.argsort(ev)[::-1][:k]
    C = C + (V[:, idx] * ev[idx]) @ V[:, idx].T
    J, _ = L.local_jacobian(LC[name]["grid_avg"], LC[name]["ucos"], LC[name]["fid_mean"], order, h)
    cov6 = np.linalg.inv(J.T @ np.linalg.inv(C) @ J)
    return cov6[np.ix_(P3, P3)]


g_non = GaussianND(FID3, cov3("nonbnt_460"), names=NAMES3, labels=LABELS3,
                   label="non-BNT  (cut all bins @ $\\ell$460)")
g_bnt = GaussianND(FID3, cov3("bnt_580"), names=NAMES3, labels=LABELS3,
                   label="BNT  (bin1 @ $\\ell$580, bins 2-4 full)")

p = plots.get_subplot_plotter(width_inch=7.0)
p.settings.alpha_filled_add = 0.5
p.settings.legend_fontsize = 11
p.settings.axes_fontsize = 11
p.settings.axes_labelsize = 14
p.triangle_plot([g_non, g_bnt], NAMES3, filled=True,
                contour_colors=["#7f7f7f", "#c0392b"],
                legend_labels=[g_non.label, g_bnt.label], legend_loc="upper right")
p.fig.suptitle("Proper Fisher 68%/95% contours — 14000 deg$^2$ auto+cross PS  "
               "(local order-2 J + hybrid cov)", fontsize=11, y=1.02)
out = f"{OUT}/fisher_contours_bnt_vs_nonbnt_14000.png"
p.export(out)
print("saved", out)
