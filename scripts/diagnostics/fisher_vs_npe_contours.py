#!/usr/bin/env python3
"""Phase-B figure: NPE-on-score posteriors (filled) vs the Fisher ellipses (dashed) for BNT-580 and
non-BNT-460, 14000 deg2. Shows the realized SBI constraint against the Fisher target. getdist."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fisher_local_jacobian as L
import fisher_hybrid_cov as H
from getdist import plots
from getdist.mcsamples import MCSamples
from getdist.gaussian_mixtures import GaussianND

P3 = [0, 1, 2]
NAMES3, LAB3 = ["Om", "S8", "w0"], [r"\Omega_m", "S_8", "w_0"]
FID3 = L.FID[P3]
SCORE = "outputs/score_experiment/npe_score"
OUT = "outputs/score_experiment"

Tfull = np.kron(H.bnt_spectra_operator(), np.eye(H.NBPW))
C_ANA_BNT = Tfull @ H.C_ANA @ Tfull.T
CFG = {"nonbnt_460": ([460, 460, 460, 460], False), "bnt_580": ([580, 1024, 1024, 1024], True)}
LC = {k: L.build_config(c, b) for k, (c, b) in CFG.items()}


def fisher_cov3(name, k=3):
    cuts, bnt = CFG[name]
    R = H.cut_rebin_R(H.per_spectrum_uppers(cuts))
    C = R @ (C_ANA_BNT if bnt else H.C_ANA) @ R.T
    fa, fc, nell = L.load_set("fiducial", "nobaryons", bnt)
    D = np.cov(L.datavector(fa, fc, nell, cuts), rowvar=False) - C
    ev, V = np.linalg.eigh(D); idx = np.argsort(ev)[::-1][:k]
    C = C + (V[:, idx] * ev[idx]) @ V[:, idx].T
    J, _ = L.local_jacobian(LC[name]["grid_avg"], LC[name]["ucos"], LC[name]["fid_mean"], "order2", 0.75)
    cov6 = np.linalg.inv(J.T @ np.linalg.inv(C) @ J)
    return cov6[np.ix_(P3, P3)]


def mc(name, tag, label):
    s = np.load(f"{SCORE}/posterior_summary_{tag}.npy")[:, P3]
    return MCSamples(samples=s, names=NAMES3, labels=LAB3, label=label)


npe_non = mc("nonbnt_460", "nonbnt_460_14000_mle", "NPE-score non-BNT")
npe_bnt = mc("bnt_580", "bnt_580_14000_mle", "NPE-score BNT")
fis_non = GaussianND(FID3, fisher_cov3("nonbnt_460"), names=NAMES3, labels=LAB3, label="Fisher non-BNT")
fis_bnt = GaussianND(FID3, fisher_cov3("bnt_580"), names=NAMES3, labels=LAB3, label="Fisher BNT")

p = plots.get_subplot_plotter(width_inch=7.5)
p.settings.legend_fontsize = 10
p.triangle_plot(
    [fis_non, fis_bnt, npe_non, npe_bnt], NAMES3,
    filled=[False, False, True, True],
    contour_ls=["--", "--", "-", "-"],
    contour_colors=["#7f7f7f", "#c0392b", "#7f7f7f", "#c0392b"],
    legend_labels=["Fisher non-BNT", "Fisher BNT", "NPE-score non-BNT", "NPE-score BNT"],
    legend_loc="upper right")
p.fig.suptitle("NPE-on-score (filled) vs Fisher (dashed) — 14000 deg$^2$ auto+cross PS\n"
               "BNT-580 (red) vs non-BNT-460 (grey); NPE calibrated (TARP/SBC)", fontsize=10.5, y=1.02)
out = f"{OUT}/fisher_vs_npe_contours_14000.png"
p.export(out)
print("saved", out)
