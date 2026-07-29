#!/usr/bin/env python
"""
Is the NEW masked power-spectrum approach different from the OLD one at ell>100? (definitive check)

Reconciles the three candidate "new vs old" differences for the MASTER masked PS, at the 14000 sqdeg
paper footprint, nobaryons grid (16965 cosmologies):

  (1) lmax 1530 (old) vs 1535 (new): both nlb=4, both on disk -> direct per-bandpower ratio vs ell.
  (2) mean-subtraction (submean) vs raw: from the 5000 gate datavectors -> frac diff vs ell.
  (3) nlb=1 vs nlb=4 and pymaster-version: cited (documented elsewhere), not recomputed here.

Bandpower effective ell is linear: ell_k = 3.5 + 4k (nlb=4, lmin=2). Produces a 2-panel figure and
prints the headline numbers.

Run (plain numpy env): python scripts/diagnostics/ps_old_vs_new_check.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NG = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/new_grid"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/ps_old_vs_new"
os.makedirs(OUT, exist_ok=True)
base = "all_cls_grid_nobaryons_bin{b}_masked_14000sqdeg_apod2.0_master_noisy_s0.26"
crossbase = "all_cross_cls_grid_nobaryons_bins1234_masked_14000sqdeg_apod2.0_master_noisy_s0.26"


def ratio_vs_ell(a30, a35):
    nb = min(a30.shape[1], a35.shape[1])
    ell = 3.5 + 4 * np.arange(nb)
    med = np.median(a35[:, :nb] / a30[:, :nb], axis=0)
    scat = (np.percentile(a35[:, :nb] / a30[:, :nb], 84, 0) -
            np.percentile(a35[:, :nb] / a30[:, :nb], 16, 0)) / 2
    return ell, med, scat


print("=" * 78)
print("(1) lmax 1530 (old) vs 1535 (new), 14000 sqdeg, nlb=4 — median ratio 1535/1530")
print("=" * 78)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
for b in [1, 2, 3, 4]:
    p = f"{NG}/{base.format(b=b)}"
    if not (os.path.exists(p + "_lmax1530.npy") and os.path.exists(p + "_lmax1535.npy")):
        print(f"  bin{b}: one lmax missing, skip"); continue
    ell, med, scat = ratio_vs_ell(np.load(p + "_lmax1530.npy"), np.load(p + "_lmax1535.npy"))
    axA.plot(ell, med, lw=1.0, label=f"auto bin{b}")
    m = (ell >= 100) & (ell <= 1024)
    print(f"  auto bin{b} (100<=ell<=1024): median ratio in [{med[m].min():.4f},{med[m].max():.4f}], "
          f"mean|ratio-1|={np.abs(med[m]-1).mean():.2e}, #bands|ratio-1|>2%={int((np.abs(med[m]-1)>0.02).sum())}")
# cross spectra
if os.path.exists(f"{NG}/{crossbase}_lmax1530.npy") and os.path.exists(f"{NG}/{crossbase}_lmax1535.npy"):
    c30 = np.load(f"{NG}/{crossbase}_lmax1530.npy"); c35 = np.load(f"{NG}/{crossbase}_lmax1535.npy")
    nb30, nb35 = c30.shape[1] // 6, c35.shape[1] // 6
    c30 = c30.reshape(-1, 6, nb30); c35 = c35.reshape(-1, 6, nb35)
    ell, med, _ = ratio_vs_ell(c30[:, 0, :], c35[:, 0, :])
    axA.plot(ell, med, lw=1.0, ls="--", label="cross 1-2")
    m = (ell >= 100) & (ell <= 1024)
    print(f"  cross 1-2 (100<=ell<=1024): median ratio in [{med[m].min():.4f},{med[m].max():.4f}], "
          f"mean|ratio-1|={np.abs(med[m]-1).mean():.2e}")
axA.axhspan(0.99, 1.01, color="green", alpha=0.12, label="±1%")
axA.axvline(1024, color="k", ls=":", lw=0.8); axA.axvline(400, color="purple", ls=":", lw=0.8)
axA.set_xlabel(r"$\ell$"); axA.set_ylabel("median  Cl(lmax1535) / Cl(lmax1530)")
axA.set_xlim(0, 1100); axA.set_ylim(0.7, 1.75); axA.legend(fontsize=7, ncol=2)
axA.set_title("(1) old (lmax1530) vs new (lmax1535): diverge at low-l, agree >1% above l~400")
axA.text(405, 1.6, "l=400", color="purple", fontsize=7); axA.text(1028, 1.6, "l=1024", fontsize=7)

print("\n" + "=" * 78)
print("(2) mean-subtraction vs raw (5000 gate fiducial) — max |frac diff| across 10 spectra vs ell")
print("=" * 78)
g = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fisher_gate_5000_l37/datavectors.npz"
if os.path.exists(g):
    d = np.load(g); ells = d["ells"]
    raw = d["fid_raw"].reshape(200, -1, len(ells)).mean(0)
    sub = d["fid_sub"].reshape(200, -1, len(ells)).mean(0)
    maxfrac = (np.abs(sub - raw) / (np.abs(raw) + 1e-30)).max(0)
    axB.semilogy(ells, maxfrac, lw=1.2, color="C3")
    axB.axhline(0.01, color="green", ls="--", lw=0.8); axB.axvline(100, color="k", ls=":", lw=0.8)
    axB.set_xlabel(r"$\ell$"); axB.set_ylabel("max |(submean - raw)/raw|  over spectra")
    axB.set_title("(2) mean-subtraction is a no-op above ell~100"); axB.set_xlim(0, 600)
    for lc in [50, 100, 150]:
        mm = ells >= lc
        print(f"  ell>={lc}: max frac diff = {maxfrac[mm].max():.2e}")
    axB.text(105, maxfrac.max()*0.5, "l=100", fontsize=7)
fig.suptitle("Masked PS: NEW vs OLD at ell>100 (14000 footprint). Bulk agreement ~1%; differences "
             "are the top edge band & low-ell monopole only.", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/ps_old_vs_new_check.png", dpi=160, bbox_inches="tight")
fig.savefig(f"{OUT}/ps_old_vs_new_check.pdf", bbox_inches="tight")
print(f"\nwrote {OUT}/ps_old_vs_new_check.{{png,pdf}}")
