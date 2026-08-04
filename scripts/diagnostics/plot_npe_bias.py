"""Contour figures from the mean-subtracted masked-PS NPE posteriors (ell 37-1024).
Per mask: (Omega_m, S8) contours for the null case (nobaryons-vs-nobaryons) and the
baryon-bias case (nobaryons-vs-baryonified). Plus constraint/bias vs survey area."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SAMP = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/npe_prod"
MASKS = [2000, 5000, 10000, 14000, 28000, 35000]
PIDX = {"Om": 0, "S8": 1}


def load(mask, fid):
    f = (f"{SAMP}/posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_"
         f"l37-1024_r10_masked_{mask}sqdeg_apod2.0_master_submean_noisy_s0.26.npy")
    return np.load(f)


def contour2d(ax, x, y, color, label, levels=(0.95, 0.68)):
    """Filled 68% + 95% line HPD contours from samples via KDE."""
    k = gaussian_kde(np.vstack([x, y]))
    xi = np.linspace(x.min(), x.max(), 140)
    yi = np.linspace(y.min(), y.max(), 140)
    X, Y = np.meshgrid(xi, yi)
    Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Zs = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(Zs) / Zs.sum()
    lv = [Zs[np.searchsorted(cum, L)] for L in levels]   # [95%, 68%] density levels
    ax.contourf(X, Y, Z, levels=[lv[1], Z.max()], colors=[color], alpha=0.30)
    ax.contour(X, Y, Z, levels=sorted(lv), colors=color, linewidths=1.6)
    ax.plot([], [], color=color, lw=3, label=label)


# ---- Figure 1: per-mask Om-S8 contours, null vs baryon-biased ------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
bias = {}
for ax, mask in zip(axes.flat, MASKS):
    nb = load(mask, "nobaryons")
    by = load(mask, "baryonified")
    contour2d(ax, nb[:, 0], nb[:, 1], "#2980b9", "nobaryons (null)")
    contour2d(ax, by[:, 0], by[:, 1], "#c0392b", "baryonified (biased)")
    # mark the null mean as the recovered-truth reference
    ax.plot(nb[:, 0].mean(), nb[:, 1].mean(), "kx", ms=8)
    dS8 = by[:, 1].mean() - nb[:, 1].mean()
    bias[mask] = dict(sig_S8_null=float(nb[:, 1].std()),
                      sig_S8_bar=float(by[:, 1].std()),
                      dS8=float(dS8), dS8_sigma=float(dS8 / nb[:, 1].std()))
    ax.set_title(rf"{mask} deg$^2$   ($\\Delta \sigma_8$={dS8:+.3f} = {dS8/nb[:,1].std():+.1f}$\\sigma$)",
                 fontsize=11)
    ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$\sigma_8$")
    if mask == MASKS[0]:
        ax.legend(fontsize=9, loc="upper right")
fig.suptitle("Masked power-spectrum NPE ($\\ell$ 37-1024, mean-subtracted): baryonic bias by survey area",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/npe_bias_contours.png", dpi=140)
print("wrote npe_bias_contours.png")

# ---- Figure 2: constraint + bias vs survey area --------------------------
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
areas = MASKS
a1.plot(areas, [bias[m]["sig_S8_null"] for m in areas], "o-", color="#2980b9", label="null")
a1.plot(areas, [bias[m]["sig_S8_bar"] for m in areas], "s--", color="#c0392b", label="baryonified")
a1.set_xscale("log"); a1.set_xlabel("survey area [deg$^2$]"); a1.set_ylabel(r"$\sigma(\sigma_8)$")
a1.set_title("Constraint vs area (tightens as area grows)"); a1.legend(); a1.grid(alpha=0.3)
a2.plot(areas, [bias[m]["dS8_sigma"] for m in areas], "D-", color="#8e44ad")
a2.axhline(0, color="k", lw=0.8); a2.axhline(-1, color="grey", ls=":")
a2.set_xscale("log"); a2.set_xlabel("survey area [deg$^2$]")
a2.set_ylabel(r"$\sigma_8$ bias  [$\sigma$]"); a2.set_title(r"Baryonic bias on $\sigma_8$ vs area")
a2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{OUT}/npe_bias_vs_area.png", dpi=140)
print("wrote npe_bias_vs_area.png")

print("\n=== summary ===")
for m in MASKS:
    b = bias[m]
    print(f"  {m:>5} deg2: sig(S8) null={b['sig_S8_null']:.3f} bar={b['sig_S8_bar']:.3f} "
          f"| dS8={b['dS8']:+.3f} ({b['dS8_sigma']:+.1f} sigma)")
