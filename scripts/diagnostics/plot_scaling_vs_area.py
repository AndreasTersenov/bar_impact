#!/usr/bin/env python3
"""Constraining-power scaling vs mask area — PS / peaks / L1, submean, no scale cuts.
Publication figure (A&A profile). σ(S8) and FoM3 of the nobaryons posterior vs masked area, seed-averaged.
PS = ℓmin=37 low-ℓ-recovered range (l37-1020); HOS = scales1234 (coarse dropped). See docs/scaling_vs_area_submean.md."""
import numpy as np, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.stats import linregress

plt.style.use("/home/tersenov/.claude/skills/figure-polish/style/aa.mplstyle")

D = "outputs/samples"
PSD = "outputs/baryon_tension/ps_submean_l37/posteriors"
P3 = [0, 1, 2]  # Om, S8, w0

def met(f):
    a = np.load(f); c = np.cov(a[:, P3], rowvar=False)
    return np.sqrt(c[1, 1]), 1.0 / np.sqrt(np.linalg.det(c))

def avg(files):
    S, F = [], []
    for f in files:
        try: s, fm = met(f)
        except (FileNotFoundError, IndexError): continue
        if s < 0.08: S.append(s); F.append(fm)        # drop any residual prior-collapsed seed
    return (np.mean(S), np.std(S), np.mean(F), len(S)) if S else (np.nan, np.nan, np.nan, 0)

HOS = [2001, 5001, 10001, 14001, 28001, 35001]
PS  = [2000, 5000, 10000, 14000, 28000, 35000]

def hf(prefix, A):
    base = f"{D}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_scales1234_noisy_s0.26_masked_{A}sqdeg_submean_new_normalization_npe.npy"
    runs = [f"{D}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_scales1234_noisy_s0.26_masked_{A}sqdeg_submean_new_normalization_run{r}_npe.npy" for r in (2, 3, 4, 5)]
    return [base] + runs

def pf(A):
    return glob.glob(f"{PSD}/mask_{A:05d}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l37-1020_r10_masked_{A}sqdeg_apod2.0_master_submean_noisy_s0.26*.npy")

# (label, areas, metrics, Okabe-Ito color, linestyle, marker)
SERIES = [
    ("Power spectrum", PS,  [avg(pf(A))      for A in PS],  "#0072B2", "-",  "o"),
    ("Peak counts",    HOS, [avg(hf("pc_", A)) for A in HOS], "#D55E00", "--", "s"),
    ("L1 norm",        HOS, [avg(hf("",   A)) for A in HOS], "#009E73", "-.", "^"),
]

W = 7.087  # A&A double-column, inches
fig, ax = plt.subplots(1, 2, figsize=(W, 0.46 * W))

for j, (name, A, m, c, ls, mk) in enumerate(SERIES):
    A = np.array(A, float)
    s8  = np.array([x[0] for x in m]); es = np.array([x[1] for x in m])
    fom = np.array([x[2] for x in m])
    ss = linregress(np.log(A), np.log(s8)).slope
    sf = linregress(np.log(A), np.log(fom)).slope
    ax[0].errorbar(A, s8, yerr=es, color=c, ls=ls, marker=mk, ms=4.5, lw=1.4,
                   capsize=2, elinewidth=0.9, label=rf"{name} ($\alpha={ss:+.2f}$)")
    ax[1].plot(A, fom, color=c, ls=ls, marker=mk, ms=4.5, lw=1.4,
               label=rf"{name} ($\alpha={sf:+.2f}$)")

# reference slopes (anchored at 14000), neutral gray dotted
Aref = np.array([1.7e3, 4.2e4])
ax[0].plot(Aref, 0.0135 * (Aref / 14000.) ** -0.5, color="0.45", ls=":", lw=1.0, zorder=0, label=r"$A^{-1/2}$")
ax[1].plot(Aref, 1.05e5 * (Aref / 14000.) ** +1.5, color="0.45", ls=":", lw=1.0, zorder=0, label=r"$A^{+3/2}$")

ax[0].set_ylabel(r"$\sigma(S_8)$")
ax[1].set_ylabel(r"$\mathrm{FoM}_3\,(\Omega_\mathrm{m},S_8,w_0)$")
for k, a in enumerate(ax):
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"mask area $\,[\mathrm{deg}^2]$")
    a.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    a.tick_params(which="both", direction="in", top=True, right=True)
    a.legend(frameon=False, fontsize=8, handlelength=2.2)
    a.text(0.04, 0.06, f"({'ab'[k]})", transform=a.transAxes, fontsize=9, va="bottom")

fig.tight_layout(pad=0.4, w_pad=1.2)
out = "outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out + ".pdf / .png")
for name, A, m, c, ls, mk in SERIES:
    A = np.array(A, float); s8 = np.array([x[0] for x in m])
    print(f"  {name:14s} slope sig(S8) = {linregress(np.log(A), np.log(s8)).slope:+.2f}  (nseed {[x[3] for x in m]})")
