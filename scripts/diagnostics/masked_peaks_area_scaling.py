#!/usr/bin/env python
"""
DIAGNOSTIC: does masked peak-count constraining power scale with sky area?

Symptom under investigation: masked peak-count posteriors are anomalously tight
(FoM degrades only ~1.3x under masking vs ~2.4x for l1, ~3x for PS) — unphysical,
since less sky should mean less information. Full-sky peaks and all l1/PS are fine.

This computes the *data-level* Gaussian Fisher constraining power (no NPE) as a
function of mask area, for peaks vs l1 (control), reusing the verified recipe from
fisher_constraining_power.py (local Jacobian, Hartlap-corrected fiducial-perm cov).

Why this is decisive: Fisher info F is a property of the DATA VECTOR. Physically,
F ∝ N_modes ∝ area, so marginal sigma(param) ∝ area^-0.5 (log-log slope -0.5) and
FoM6 ∝ area^+3. If peaks show much shallower slopes than l1 here, the spurious
"information" is in the masked peak data vector itself (e.g. fixed-geometry boundary
peaks), NOT an NPE artifact. If peaks scale normally here but the NPE posteriors are
tight, the problem is NPE overconfidence instead.

Outputs printed table + outputs/diagnostics/masked_peaks/area_scaling.png.
Runs in any numpy env (~20s). No reprocessing.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
GRID = f"{BASE}/grid"
FID = f"{BASE}/fiducial/cosmo_fiducial"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/masked_peaks"
os.makedirs(OUT, exist_ok=True)

PN = ["Om", "S8", "w0", "H0", "ns", "Ob"]
FID_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
LOCAL_BW = 1.0
TOTAL_SKY = 41252.96125
AREAS = [2001, 5001, 10001, 14001, 28001, 35001]  # canonical _new_normalization sweep
params = np.load(f"{GRID}/cosmo_params.npy")  # (16965, 6)


def _rebin(a, target=5):
    """Rebin last axis (SNR bins) to `target` bins by averaging."""
    nb = a.shape[-1]
    k = nb // target
    a = a[..., : k * target]
    return a.reshape(*a.shape[:-1], target, k).mean(-1)


def hos_vec(prefix, kind, area, scales, target_bins=5):
    """Load masked (area) or full-sky (area=None) l1/peaks, select scales, rebin, concat bins."""
    d, t = (GRID, "grid") if kind == "grid" else (FID, "fiducial")
    mask = f"_masked_{area}sqdeg" if area is not None else ""
    out = []
    for b in [1, 2, 3, 4]:
        p = f"{d}/all_{prefix}_{t}_nobaryons_bin{b}{mask}_noisy_s0.26_new_normalization.npy"
        a = np.asarray(np.load(p, allow_pickle=True), float)[:, scales, :]
        out.append(_rebin(a, target_bins).reshape(a.shape[0], -1))
    return np.concatenate(out, axis=1)


def fisher_cov(grid_vec, fid_vec, par=params):
    """6-param cov from a LOCAL linear Jacobian + Hartlap sample cov (whitened features)."""
    s = fid_vec.std(0)
    keep = s > 0
    grid_vec = grid_vec[:, keep] / s[keep]
    fid_vec = fid_vec[:, keep] / s[keep]
    n_fid, n_dat = fid_vec.shape
    fid_mean = fid_vec.mean(0)
    ps = par.std(0)
    w = np.exp(-0.5 * (((par - FID_PARAMS) / ps) ** 2).sum(1) / LOCAL_BW ** 2)
    sw = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq((par - FID_PARAMS) * sw, (grid_vec - fid_mean) * sw, rcond=None)
    J = sol.T
    C = np.cov(fid_vec, rowvar=False)
    hartlap = (n_fid - n_dat - 2) / (n_fid - 1)
    Cinv = np.linalg.inv(C) * hartlap
    cov = np.linalg.inv(J.T @ Cinv @ J)
    return cov, {"n_feat": n_dat, "hartlap": hartlap}


def fom(cov):
    d = np.sqrt(np.diag(cov))
    return {
        "sig_Om": d[0], "sig_S8": d[1], "sig_w0": d[2],
        "FoM6": 1.0 / np.sqrt(np.linalg.det(cov)),
    }


def slope(areas, vals):
    """Power-law exponent: log(vals) = slope*log(area) + c."""
    return np.polyfit(np.log(np.array(areas)), np.log(np.array(vals)), 1)[0]


SCALE_SETS = {"scales234": [1, 2, 3], "scales1234": [0, 1, 2, 3]}
PROBES = [("peak_counts", 6), ("l1_norms", 8)]  # (prefix, snr_rebin so 30/40 -> 5 bins)

print(f"{'='*92}\nMASKED PEAK-COUNT AREA SCALING — data-level Fisher (local Jacobian)\n{'='*92}")
print("Physical expectation: sigma(param) ∝ area^-0.5  (slope -0.5);  FoM6 ∝ area^+3 (slope +3).\n")

curves = {}  # (prefix, scaleset) -> dict(area-> fom)
for prefix, rebin in PROBES:
    for sname, scales in SCALE_SETS.items():
        rows = {}
        for area in AREAS:
            g = hos_vec(prefix, "grid", area, scales)
            f = hos_vec(prefix, "fid", area, scales)
            cov, info = fisher_cov(g, f)
            rows[area] = {**fom(cov), **info}
        curves[(prefix, sname)] = rows
        ar = AREAS
        sS8 = [rows[a]["sig_S8"] for a in ar]
        sw0 = [rows[a]["sig_w0"] for a in ar]
        f6 = [rows[a]["FoM6"] for a in ar]
        print(f"--- {prefix:11s} {sname:10s}  (nfeat={rows[AREAS[0]]['n_feat']}, "
              f"hartlap≈{rows[AREAS[0]]['hartlap']:.2f}) ---")
        print(f"  {'area':>7}{'f_sky':>8}{'sig(S8)':>10}{'sig(w0)':>10}{'FoM6':>12}")
        for a in ar:
            print(f"  {a:>7}{a/TOTAL_SKY:>8.3f}{rows[a]['sig_S8']:>10.4f}"
                  f"{rows[a]['sig_w0']:>10.4f}{rows[a]['FoM6']:>12.3e}")
        print(f"  SLOPES vs area:  sig(S8) {slope(ar,sS8):+.2f}  sig(w0) {slope(ar,sw0):+.2f}"
              f"  FoM6 {slope(ar,f6):+.2f}   (expect -0.5, -0.5, +3.0)\n")

# ---- figure: sigma(S8) and FoM6 vs area, peaks vs l1, both scale sets ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
styles = {("peak_counts", "scales234"): ("#2ca02c", "-", "o"),
          ("peak_counts", "scales1234"): ("#2ca02c", "--", "s"),
          ("l1_norms", "scales234"): ("#1f77b4", "-", "o"),
          ("l1_norms", "scales1234"): ("#1f77b4", "--", "s")}
for (prefix, sname), rows in curves.items():
    col, ls, mk = styles[(prefix, sname)]
    lab = f"{'peaks' if prefix=='peak_counts' else 'l1'} {sname}"
    axes[0].loglog(AREAS, [rows[a]["sig_S8"] for a in AREAS], color=col, ls=ls, marker=mk, label=lab)
    axes[1].loglog(AREAS, [rows[a]["FoM6"] for a in AREAS], color=col, ls=ls, marker=mk, label=lab)
# reference slopes anchored at the smallest area
a0 = AREAS[0]
for ax, ref_slope, anch_key in [(axes[0], -0.5, "sig_S8"), (axes[1], 3.0, "FoM6")]:
    y0 = curves[("l1_norms", "scales234")][a0][anch_key]
    ax.loglog(AREAS, [y0 * (a / a0) ** ref_slope for a in AREAS], "k:", lw=1,
              label=f"area^{ref_slope:+.1f} (physical)")
axes[0].set_xlabel("mask area [sq deg]"); axes[0].set_ylabel(r"$\sigma(S_8)$")
axes[0].set_title("Constraint vs area (lower=better; physical: area$^{-0.5}$)")
axes[1].set_xlabel("mask area [sq deg]"); axes[1].set_ylabel("FoM6")
axes[1].set_title("6-param FoM vs area (higher=better; physical: area$^{+3}$)")
for ax in axes:
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.suptitle("Masked peaks vs l1: does data-level Fisher constraining power scale with sky area?")
fig.tight_layout()
fig.savefig(f"{OUT}/area_scaling.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT}/area_scaling.png")
