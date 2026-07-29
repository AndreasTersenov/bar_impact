#!/usr/bin/env python3
"""Phase I validation figure: analytic NaMaster Gaussian covariance vs the 200-perm sim covariance.
(A) per-bandpower variance ratio analytic/sample vs ell (diagonal validated ~1 at high ell).
(B) rebinned correlation matrices, analytic vs sample (the sim's extra off-diagonal = non-Gaussian).
(C) the excess D=C_samp-C_ana: diagonal excess + the coherent leading (SSC) eigenvector.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import load_set, datavector, ELL_OFFSET, ELL_PER_BIN  # noqa: E402
import fisher_gaussian_cov as G  # noqa: E402

AREA = 14000
NBPW = 383
EFF = ELL_OFFSET + ELL_PER_BIN * (np.arange(NBPW) + 0.5)
OUT = "outputs/diagnostics/fisher_cov"
os.makedirs(OUT, exist_ok=True)

C = np.load(os.path.join(HERE, "cache_gaussian_cov", f"gaussian_cov_native_{AREA}.npy"))
fa, fc, _ = load_set("fiducial", "nobaryons", bnt=False)
pdata = {}
for k, (i, _) in enumerate(G.AUTOS):
    pdata[(i, i)] = fa[k]
for k, (i, j) in enumerate(G.PAIRS):
    pdata[(i, j)] = fc[:, k * NBPW:(k + 1) * NBPW]

fig = plt.figure(figsize=(13, 4.2))

# (A) diagonal validation
axA = fig.add_subplot(1, 3, 1)
for a, (i, j) in enumerate(G.SPECTRA):
    ana = np.diag(C)[a * NBPW:(a + 1) * NBPW]
    samp = pdata[(min(i, j), max(i, j))].var(0, ddof=1)
    axA.plot(EFF, ana / (samp + 1e-300), lw=0.8, alpha=0.7)
axA.axhline(1.0, color="k", ls="--", lw=1)
axA.axhspan(0.9, 1.1, color="grey", alpha=0.15)
axA.set_xscale("log"); axA.set_xlim(30, 1535); axA.set_ylim(0.7, 1.2)
axA.set_xlabel(r"$\ell$"); axA.set_ylabel("var ratio  analytic / sim")
axA.set_title("(A) diagonal validated\n(10 spectra; band = 10%)")

# (B) rebinned correlation matrices
R = G.build_full_R(NBPW, upper=1024)
C_ana = R @ C @ R.T
dv = datavector(fa, fc, NBPW, [1024, 1024, 1024, 1024])
C_samp = np.cov(dv, rowvar=False)
corr_a = C_ana / np.sqrt(np.outer(np.diag(C_ana), np.diag(C_ana)))
corr_s = C_samp / np.sqrt(np.outer(np.diag(C_samp), np.diag(C_samp)))
axB = fig.add_subplot(1, 3, 2)
n = corr_a.shape[0]
combo = np.tril(corr_s, -1) + np.triu(corr_a, 1) + np.eye(n)
im = axB.imshow(combo, vmin=-0.3, vmax=0.3, cmap="RdBu_r")
axB.set_title("(B) correlation matrix\nupper=analytic  lower=sim")
axB.set_xlabel("feature"); axB.set_ylabel("feature")
fig.colorbar(im, ax=axB, fraction=0.046)

# (C) excess: diagonal + leading eigenvector
D = C_samp - C_ana
s = 1.0 / np.sqrt(np.diag(C_samp))
Dn = (D * s[:, None]) * s[None, :]
evals, evecs = np.linalg.eigh(Dn)
lead = evecs[:, np.argmax(evals)]
if np.mean(lead) < 0:
    lead = -lead
axC = fig.add_subplot(1, 3, 3)
axC.plot(np.diag(D) / np.diag(C_samp), label=r"diag excess $D_{ii}/C^{\rm sim}_{ii}$", lw=1.2)
axC.plot(lead, label="leading eigvec (coherent=SSC)", lw=1.2)
axC.axhline(0, color="k", lw=0.6)
axC.set_xlabel("feature"); axC.set_ylim(-0.2, 0.6)
axC.set_title("(C) non-Gaussian excess\n~23% extra var, coherent SSC mode")
axC.legend(fontsize=7, loc="upper right")

fig.tight_layout()
p = os.path.join(OUT, f"gaussian_cov_validation_{AREA}.png")
fig.savefig(p, dpi=130)
print("saved", p)
