"""Measure the exact ell-space response of each starlet scale (nscale=5: 4 wavelet + coarse)
at nside 512, by pushing white-noise maps through CMRStarlet and taking anafast of each scale.
For white noise (flat C_ell), the per-scale C_ell IS the squared transfer function W_j(ell)^2.

Publication-quality, low-noise version: many realizations + log-ell binning. Saves the
transfer-function data (npz), a clean figure, and prints the per-scale ell ranges.
Tells us the precise ell coverage of each wavelet scale and the coarse scale -> sets the
power-spectrum ell range that is scale-matched to the higher-order statistics.
"""
import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycs.sparsity.mrs.mrs_starlet import CMRStarlet

NSIDE = 512
NSCALE = 5
LMAX = 1535
NREAL = 40
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "outputs", "diagnostics", "starlet_scale_ell")
os.makedirs(OUT, exist_ok=True)

# Wong colour-blind-safe palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
LABELS = [f"wavelet {j}" for j in range(NSCALE - 1)] + ["coarse"]

npix = hp.nside2npix(NSIDE)
ell = np.arange(LMAX + 1)
acc = np.zeros((NSCALE, LMAX + 1))
rng = np.random.default_rng(1)
for r in range(NREAL):
    wn = rng.normal(0, 1, npix)
    C = CMRStarlet()
    C.init_starlet(NSIDE, nscale=NSCALE)
    C.transform(wn)
    for j in range(NSCALE):
        acc[j] += hp.anafast(C.coef[j], lmax=LMAX)
    if (r + 1) % 10 == 0:
        print(f"  {r+1}/{NREAL} realizations", flush=True)
acc /= NREAL                                   # W_j(ell)^2 up to a common flat normalization
W2 = acc / acc.max(axis=1, keepdims=True)      # peak-normalised per scale

# log-ell binning for a smooth display curve
edges = np.unique(np.round(np.logspace(np.log10(2), np.log10(LMAX), 60)).astype(int))
centers = np.sqrt(edges[:-1] * edges[1:])
W2_binned = np.zeros((NSCALE, len(centers)))
for j in range(NSCALE):
    for b in range(len(centers)):
        sel = (ell >= edges[b]) & (ell < edges[b + 1])
        W2_binned[j, b] = W2[j, sel].mean() if sel.any() else np.nan

# per-scale ell ranges from the (unbinned) transfer function
print(f"\n{'scale':>6} {'type':>7} {'ell_peak':>9} {'ell_lo(half)':>13} {'ell_hi(half)':>13}")
table = []
for j in range(NSCALE):
    w = W2[j]
    pk = int(np.argmax(w))
    above = np.where(w >= 0.5)[0]
    lo, hi = (int(above.min()), int(above.max())) if len(above) else (0, 0)
    typ = "coarse" if j == NSCALE - 1 else f"wav{j}"
    table.append((j, typ, pk, lo, hi))
    print(f"{j:>6} {typ:>7} {pk:>9} {lo:>13} {hi:>13}")

np.savez(os.path.join(OUT, "starlet_transfer_data.npz"),
         ell=ell, W2=W2, centers=centers, W2_binned=W2_binned,
         table=np.array(table, dtype=object))

# ---- clean figure ---------------------------------------------------------
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for j in range(NSCALE):
    ax.semilogx(centers, W2_binned[j], color=COLORS[j], lw=2.2, label=LABELS[j])
ax.axvspan(2, 30, color="0.85", alpha=0.5, zorder=0)
ax.text(6.5, 0.9, "coarse /\nmean regime", fontsize=9, color="0.35", ha="center")
ax.set_xlim(2, LMAX)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"multipole $\ell$")
ax.set_ylabel(r"normalised response $W_j(\ell)^2 / \max$")
ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center")

# top axis: angular scale (arcmin) ~ 10800/ell
secax = ax.secondary_xaxis("top", functions=(lambda l: 10800.0 / np.clip(l, 1e-6, None),
                                             lambda a: 10800.0 / np.clip(a, 1e-6, None)))
secax.set_xlabel("angular scale  $10800/\\ell$  [arcmin]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "starlet_scale_ell_clean.png"), dpi=200)
fig.savefig(os.path.join(OUT, "starlet_scale_ell_clean.pdf"))
print("\nwrote", os.path.join(OUT, "starlet_scale_ell_clean.png"), "(+ .pdf, + data npz)")
