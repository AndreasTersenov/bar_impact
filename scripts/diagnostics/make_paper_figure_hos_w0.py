#!/usr/bin/env python
"""
PUBLICATION FIGURE: why the wavelet HOS (l1-norm) have an opposite-sign w0 degeneracy to the PS.

Story (NOISY regime = the actual analysis): restrict the l1-norm to its VARIANCE (the two-point
information it shares with the power spectrum) and the Omega_m-w0 degeneracy matches the PS; add the
higher MOMENTS (skewness, kurtosis = the non-Gaussian information) and the degeneracy rotates to the
opposite sign.

Two panels:
  (a) Omega_m-w0 Fisher correlation as moments are added: PS, variance, +skew, +skew+kurt.
  (b) the Omega_m-w0 degeneracy DIRECTION (correlation ellipses) rotating from PS-like to flipped.

------------------------------------------------------------------------------------------------
HOW TO REMAKE / RESTYLE (read me before editing for the paper):
  * Run with the jaxili (or any numpy+matplotlib) python:
        python scripts/diagnostics/make_paper_figure_hos_w0.py
    Outputs: outputs/diagnostics/paper/fig_hos_w0_mechanism.{pdf,png}
  * INPUT DATA (already on disk; regenerate only if lost):
      - moments: outputs/diagnostics/moments_w0/moments.npz  (NOISY var/skew/kurt per detail scale
        per tomo, 350 grid + 195 fid). Remake: cosmostat_new venv:
            python scripts/diagnostics/moments_w0_reprocess.py
      - PS l100-400 grid/fid auto+cross Cls in CosmoGridV1/stage3_forecast (full-sky, noisy s0.26).
  * TO MATCH THE PAPER FORMAT: edit ONLY the STYLE block below (figsize, fonts, colors, dpi) and the
    panel titles/labels. The physics/Fisher code does not need to change. For a single-column figure
    set ONE_COLUMN=True. To switch to actual-size (not normalized) ellipses use ellipse_mode="cov".
  * The numbers it prints (and writes to the caption stub) are the values quoted in the paper text.
------------------------------------------------------------------------------------------------
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ============================== STYLE (edit for the paper format) ==============================
ONE_COLUMN   = False
FIGSIZE      = (7.0, 3.3) if ONE_COLUMN else (11.0, 4.2)
DPI          = 300
FONT         = 11
COL_PS       = "#4d4d4d"   # power spectrum (2-pt)
COL_VAR      = "#1f77b4"   # l1 variance (= 2-pt info in l1)
COL_VARSKEW  = "#ff7f0e"   # + skewness
COL_FULL     = "#d62728"   # + skewness + kurtosis (full)
ellipse_mode = "corr"      # "corr" = normalized direction ; "cov" = actual marginal size
plt.rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 1,
                     "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 2,
                     "axes.titlesize": FONT, "figure.dpi": DPI})
# ==============================================================================================

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
OUT  = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/paper"
os.makedirs(OUT, exist_ok=True)
PIDX = [0, 1, 2]  # Omega_m, S8, w0


# ----------------------------- Fisher machinery (do not edit for restyling) -------------------
def _fisher_cov(grid_vec, params, fid_vec):
    """6-param Fisher covariance from a linear Jacobian (lstsq) + Hartlap-corrected sample cov."""
    v = fid_vec.var(0); keep = v > v.max() * 1e-10
    grid_vec, fid_vec = grid_vec[:, keep], fid_vec[:, keep]
    n_fid, n_dat = fid_vec.shape
    X = np.column_stack([np.ones(len(params)), params - params.mean(0)])
    coef, *_ = np.linalg.lstsq(X, grid_vec, rcond=None)
    J = coef[1:].T
    C = np.cov(fid_vec, rowvar=False)
    Cinv = (np.linalg.inv(C) * ((n_fid - n_dat - 2) / (n_fid - 1))) if n_dat < n_fid - 2 \
        else np.diag(1.0 / np.diag(C))
    return np.linalg.inv(J.T @ Cinv @ J)


def moment_cov(M, moment_indices):
    """Fisher cov from the l1 SNR-field moments [var, skew, kurt] selected by moment_indices."""
    G, F, par = M["G"], M["F"], M["gparams"]            # G:(n,4tomo,3scale,3moment)
    g = G[:, :, :, moment_indices].reshape(len(par), -1)
    f = F[:, :, :, moment_indices].reshape(len(F), -1)
    return _fisher_cov(g, par, f)


def ps_cov():
    """Power spectrum l100-400 (auto+cross, full-sky, noisy s0.26) Fisher cov."""
    par = np.load(f"{BASE}/grid/cosmo_params.npy")
    LO, HI, RB = 100, 400, 15
    rebin = lambda a: a[..., :(a.shape[-1] // RB) * RB].reshape(*a.shape[:-1], a.shape[-1] // RB, RB).mean(-1)

    def vec(kind):
        d = "new_grid" if kind == "grid" else "fiducial/cosmo_fiducial"
        t = "grid" if kind == "grid" else "fiducial"
        autos = [rebin(np.load(f"{BASE}/{d}/all_cls_{t}_nobaryons_bin{b}_noisy_s0.26.npy")[:, LO:HI])
                 for b in [1, 2, 3, 4]]
        cr = np.load(f"{BASE}/{d}/all_cross_cls_{t}_nobaryons_bins1234_noisy_s0.26.npy").reshape(-1, 6, 1025)[:, :, LO:HI]
        return np.concatenate(autos + [rebin(cr[:, p, :]) for p in range(6)], 1)

    g, f = vec("grid"), vec("fid")
    v = f.var(0); k = v > v.max() * 1e-10; g, f = g[:, k], f[:, k]
    X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
    coef, *_ = np.linalg.lstsq(X, g, rcond=None); J = coef[1:].T
    C = np.cov(f, rowvar=False)
    return np.linalg.inv(J.T @ np.diag(1.0 / np.diag(C)) @ J)


def corr_omw0(cov):
    return cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2])


# ----------------------------------------- build ---------------------------------------------
M = np.load("/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/moments_w0/moments.npz")
covs = {
    "PS (2-pt)":       (ps_cov(),                       COL_PS),
    "variance":        (moment_cov(M, [0]),             COL_VAR),
    "+ skewness":      (moment_cov(M, [0, 1]),          COL_VARSKEW),
    "+ skew + kurt":   (moment_cov(M, [0, 1, 2]),       COL_FULL),
}
vals = {k: corr_omw0(c) for k, (c, _) in covs.items()}
print("Omega_m-w0 correlation:", {k: round(v, 2) for k, v in vals.items()})


# ----------------------------------------- plot ----------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE)

# (a) progression bar
labels = list(covs.keys()); cols = [covs[k][1] for k in labels]
axL.axhspan(0, 1, color=COL_FULL, alpha=0.05, zorder=0)   # "flipped" region (light)
axL.axhspan(-1, 0, color=COL_PS, alpha=0.06, zorder=0)    # "PS-like" region (light)
axL.bar(range(len(labels)), [vals[k] for k in labels], color=cols, width=0.66, zorder=3)
axL.axhline(0, color="k", lw=0.8, zorder=2)
for i, k in enumerate(labels):
    v = vals[k]; axL.text(i, v + (0.04 if v > 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=FONT - 2, zorder=4)
axL.set_xticks(range(len(labels)))
axL.set_xticklabels(["PS\n(2-pt)", "l1\nvariance", "l1\n+skew", "l1\n+skew\n+kurt"])
axL.set_ylim(-1, 1); axL.set_ylabel(r"$\Omega_m$--$w_0$ correlation")
axL.text(0.97, 0.93, "flipped", transform=axL.transAxes, color=COL_FULL, fontsize=FONT - 2, ha="right")
axL.text(0.97, 0.05, "PS-like", transform=axL.transAxes, color=COL_PS, fontsize=FONT - 2, ha="right")
axL.set_title(r"(a) adding non-Gaussian moments rotates $w_0$")

# (b) degeneracy-direction ellipses
def draw(ax, cov, color, label):
    s = cov[np.ix_([0, 2], [0, 2])]
    if ellipse_mode == "corr":
        d = np.sqrt(np.diag(s)); s = s / np.outer(d, d)
    w, V = np.linalg.eigh(s); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    scale = 1.0 if ellipse_mode == "corr" else 1.0
    ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1]) * scale, 2 * np.sqrt(w[0]) * scale,
                         angle=ang, fill=False, edgecolor=color, lw=2.0, label=label))

for k, (c, col) in covs.items():
    draw(axR, c, col, k)
lim = 2.0
axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim); axR.set_aspect("equal")
axR.axhline(0, color="k", lw=0.4); axR.axvline(0, color="k", lw=0.4)
axR.set_xlabel(r"$\Delta\Omega_m$ (normalized)"); axR.set_ylabel(r"$\Delta w_0$ (normalized)")
axR.legend(loc="upper right", frameon=False)
axR.set_title(r"(b) $\Omega_m$--$w_0$ degeneracy direction")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_hos_w0_mechanism.{ext}", bbox_inches="tight")
print(f"wrote {OUT}/fig_hos_w0_mechanism.{{pdf,png}}")

# caption stub for the paper
with open(f"{OUT}/fig_hos_w0_mechanism_caption.txt", "w") as fh:
    fh.write(
        "Origin of the opposite-sign w0 degeneracy between the wavelet l1-norm and the power "
        "spectrum (noisy analysis). (a) The Omega_m-w0 correlation when the l1-norm is restricted to "
        f"its variance ({vals['variance']:+.2f}) matches the power spectrum ({vals['PS (2-pt)']:+.2f}); "
        "adding the skewness and kurtosis (the non-Gaussian information) rotates it to the opposite "
        f"sign ({vals['+ skew + kurt']:+.2f}). (b) The corresponding Omega_m-w0 degeneracy direction "
        "rotating from PS-like (anti-diagonal) to flipped (diagonal).\n")
print(f"wrote caption stub {OUT}/fig_hos_w0_mechanism_caption.txt")
