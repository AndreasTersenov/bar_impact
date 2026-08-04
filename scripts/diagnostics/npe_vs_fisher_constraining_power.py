#!/usr/bin/env python
"""
Does the NPE reproduce the full-sky Fisher constraining-power picture?

Compares the existing full-sky nobaryons_vs_nobaryons NPE posteriors (noisy s0.26) against the Fisher
covariances from fisher_constraining_power.py, for the probes where matched NPE samples already exist
on disk (NO retraining): PS l100-1024, PS l100-400, l1 scales1234, l1 scales234.

For each probe we compare the (Om, S8, w0) marginals: sigma's, the (Om,w0) correlation (the degeneracy
flip), and 2D areas. The decisive test is whether the l1-over-PS RATIO and the w0-flip that the Fisher
predicts also show up in the NPE.

CAVEATS: the PS Fisher uses coarse bandpowers while the NPE PS uses r10 (finer) -> the PS absolute
widths need not match exactly; the l1 binning is closer. The NPE posterior is the ground truth; the
Fisher is the Gaussian/linear-response forecast we are validating. Compare RATIOS and DIRECTIONS, not
just absolute sigma's.

Run: python scripts/diagnostics/npe_vs_fisher_constraining_power.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/constraining_power"
SAMP = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples"
FID3 = np.array([0.26, 0.84, -1.0])           # fiducial truth (Om, S8, w0)
covs = np.load(f"{OUT}/fisher_covs.npz")

# (label, fisher cov key, NPE sample filename)
PROBES = [
    ("PS l100-1024", "PS_l100-1024_paper",
     "posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l100-1024_r10_noisy_s0.26_npe.npy"),
    ("PS l100-400", "PS_l100-400",
     "posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l100-400_r10_noisy_s0.26_npe.npy"),
    ("l1 scales1234", "l1_scales1234",
     "posterior_samples_nobaryons_vs_nobaryons_bins1234_scales1234_noisy_s0.26_new_normalization_npe.npy"),
    ("l1 scales234", "l1_scales234",
     "posterior_samples_nobaryons_vs_nobaryons_bins1234_scales234_noisy_s0.26_new_normalization_npe.npy"),
    ("peaks scales123", "peaks_scales123",
     "posterior_samples_pc_nobaryons_vs_nobaryons_bins1234_scales123_noisy_s0.26_new_normalization_npe.npy"),
]


def stats3(cov3):
    d = np.sqrt(np.diag(cov3))
    area = lambda i, j: np.pi * np.sqrt(np.linalg.det(cov3[np.ix_([i, j], [i, j])]))
    return dict(sOm=d[0], sS8=d[1], sw0=d[2], r_Omw0=cov3[0, 2] / (d[0] * d[2]),
                A_OmS8=area(0, 1), A_Omw0=area(0, 2))


rows = []
for label, fkey, sfile in PROBES:
    covF = covs[fkey][np.ix_([0, 1, 2], [0, 1, 2])]       # Fisher (Om,S8,w0) marginal
    samp = np.load(os.path.join(SAMP, sfile))[:, [0, 1, 2]]
    covN = np.cov(samp, rowvar=False)
    rows.append((label, stats3(covF), stats3(covN), covF, covN, samp))

# -------------------------------------- table --------------------------------------------------
print("\n" + "=" * 96)
print("NPE vs FISHER — full-sky nobaryons_vs_nobaryons, noisy s0.26  (F=Fisher, N=NPE posterior)")
print("=" * 96)
h = f"{'probe':<15}{'sig(S8) F/N':>18}{'sig(w0) F/N':>18}{'r(Om,w0) F/N':>18}{'A(Om,w0) F/N':>22}"
print(h); print("-" * 96)
for label, F, N, *_ in rows:
    print(f"{label:<15}{F['sS8']:.4f}/{N['sS8']:.4f}     {F['sw0']:.4f}/{N['sw0']:.4f}     "
          f"{F['r_Omw0']:+.2f}/{N['r_Omw0']:+.2f}      {F['A_Omw0']:.2e}/{N['A_Omw0']:.2e}")
print("-" * 96)

# l1-over-PS ratios (the headline): is l1's advantage reproduced?
def get(label):
    return next((F, N) for lab, F, N, *_ in rows if lab == label)
(Fps, Nps), (Fl1, Nl1) = get("PS l100-1024"), get("l1 scales1234")
print("\nl1 scales1234 vs PS l100-1024 (full-ell regime) — does NPE reproduce the Fisher ratio?")
for key, nm in [("sS8", "sig(S8)"), ("sw0", "sig(w0)"), ("A_Omw0", "(Om,w0) area")]:
    print(f"  {nm:13s}  l1/PS:  Fisher x{Fl1[key]/Fps[key]:.2f}   NPE x{Nl1[key]/Nps[key]:.2f}")
print(f"  w0 degeneracy sign:  PS Fisher {Fps['r_Omw0']:+.2f} / NPE {Nps['r_Omw0']:+.2f} ;"
      f"  l1 Fisher {Fl1['r_Omw0']:+.2f} / NPE {Nl1['r_Omw0']:+.2f}  (flip reproduced if signs match)")

Fpk, Npk = get("peaks scales123")
print("\npeaks scales123 vs PS l100-1024 — does NPE reproduce the Fisher ratio?")
for key, nm in [("sS8", "sig(S8)"), ("sw0", "sig(w0)"), ("A_Omw0", "(Om,w0) area")]:
    print(f"  {nm:13s}  peaks/PS:  Fisher x{Fpk[key]/Fps[key]:.2f}   NPE x{Npk[key]/Nps[key]:.2f}")
print(f"  w0 degeneracy sign:  peaks Fisher {Fpk['r_Omw0']:+.2f} / NPE {Npk['r_Omw0']:+.2f}")

# -------------------------------------- contour overlay ----------------------------------------
def ell(ax, cov, ctr, i, j, color, ls, label):
    s = cov[np.ix_([i, j], [i, j])]
    w, V = np.linalg.eigh(s); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(ctr[[i, j]], 2 * 1.517 * np.sqrt(w[1]), 2 * 1.517 * np.sqrt(w[0]),
                         angle=ang, fill=False, edgecolor=color, lw=2.0, ls=ls, label=label))

pairs = [(0, 1, r"$\Omega_m$", r"$\sigma_8$"), (0, 2, r"$\Omega_m$", r"$w_0$")]
show = {"PS l100-1024": "#4d4d4d", "l1 scales1234": "#1f77b4", "peaks scales123": "#2ca02c"}
fig, axc = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (i, j, xl, yl) in zip(axc, pairs):
    for label, F, N, covF, covN, samp in rows:
        if label not in show:
            continue
        c = show[label]
        ax.scatter(samp[:, i], samp[:, j], s=2, color=c, alpha=0.05, zorder=1)
        ell(ax, covN, samp.mean(0), i, j, c, "-", f"{label} (NPE)")
        ell(ax, covF, FID3, i, j, c, "--", f"{label} (Fisher)")
    ax.axvline(FID3[i], color="k", lw=0.4, ls=":"); ax.axhline(FID3[j], color="k", lw=0.4, ls=":")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
axc[0].legend(loc="best", frameon=False, fontsize=7)
fig.suptitle("NPE (solid, +samples) vs Fisher (dashed) — full-sky 68% contours; PS l100-1024, l1 "
             "scales1234, peaks scales123 (noisy s0.26). PS/l1 agree; Fisher under-states peaks.", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/npe_vs_fisher_contours.{ext}", dpi=160, bbox_inches="tight")
print(f"\nwrote {OUT}/npe_vs_fisher_contours.{{png,pdf}}")
