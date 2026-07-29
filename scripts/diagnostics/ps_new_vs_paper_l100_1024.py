#!/usr/bin/env python
"""
Validation: does the masked PS at the PAPER cut (ell 100-1024, 14000 footprint) give consistent
constraints across processing versions, and match the published paper posterior?

The user's worry: the bug-fixed (mean-subtracted, current-pymaster, lmax1535/nlb4) pipeline must, in the
ell>100 range the paper already used, reproduce the paper (same or TIGHTER ok; LOOSER = worrying).
Mean-subtraction is a verified no-op above ell~100, so at this cut "new vs old" reduces to the two
on-disk processing versions: lmax1530 vs lmax1535 (both nlb=4 RAW). We Fisher-forecast both (a
validated proxy for the PS NPE) at ell100-1024 and put the PUBLISHED NPE posterior width next to them.

Same binning for both versions -> the only difference is the data. Local Jacobian + Hartlap cov.
Run (numpy env): python scripts/diagnostics/ps_new_vs_paper_l100_1024.py
"""
import os, glob
import numpy as np

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
SAMP = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples"
FID_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
PN = ["Om", "S8", "w0", "H0", "ns", "Ob"]
params = np.load(f"{BASE}/grid/cosmo_params.npy")
LO, HI = 100, 1024
TARGET_BANDS = 10                      # per spectrum after rebin -> ~100 features (Hartlap ~0.5)


def load_masked(kind, lmaxtag):
    """auto(4)+cross(6) masked 14000 Cls, cut to [LO,HI], rebinned. kind in {grid,fid}."""
    d, t = ("new_grid", "grid") if kind == "grid" else ("fiducial", "fiducial")
    suf = f"_masked_14000sqdeg_apod2.0_master_noisy_s0.26_{lmaxtag}.npy"
    autos = [np.load(f"{BASE}/{d}/all_cls_{t}_nobaryons_bin{b}{suf}") for b in [1, 2, 3, 4]]
    nb = autos[0].shape[1]
    cross = np.load(f"{BASE}/{d}/all_cross_cls_{t}_nobaryons_bins1234{suf}").reshape(-1, 6, nb)
    ell = 3.5 + 4 * np.arange(nb)
    keep = (ell >= LO) & (ell <= HI)
    nk = keep.sum()
    rb = max(1, nk // TARGET_BANDS)
    def proc(cl):
        c = cl[:, keep]
        n = c.shape[1] // rb
        return c[:, :n * rb].reshape(c.shape[0], n, rb).mean(2)
    blocks = [proc(a) for a in autos] + [proc(cross[:, p, :]) for p in range(6)]
    return np.concatenate(blocks, 1)


def fisher_cov(grid_vec, fid_vec, par=params):
    s = fid_vec.std(0); k = s > 0
    g, f = grid_vec[:, k] / s[k], fid_vec[:, k] / s[k]
    n_fid, n_dat = f.shape
    ps = par.std(0); d2 = (((par - FID_PARAMS) / ps) ** 2).sum(1)
    w = np.exp(-0.5 * d2); sw = np.sqrt(w)[:, None]
    J = np.linalg.lstsq((par - FID_PARAMS) * sw, (g - f.mean(0)) * sw, rcond=None)[0].T
    C = np.cov(f, rowvar=False)
    Cinv = np.linalg.inv(C) * ((n_fid - n_dat - 2) / (n_fid - 1))
    cov = np.linalg.inv(J.T @ Cinv @ J)
    return cov, n_dat, (n_fid - n_dat - 2) / (n_fid - 1)


def stats(cov):
    d = np.sqrt(np.diag(cov))
    A = lambda i, j: np.pi * np.sqrt(np.linalg.det(cov[np.ix_([i, j], [i, j])]))
    return dict(sOm=d[0], sS8=d[1], sw0=d[2], A_OmS8=A(0, 1), A_Omw0=A(0, 2),
                r_Omw0=cov[0, 2] / (d[0] * d[2]), FoM6=1 / np.sqrt(np.linalg.det(cov)))


print("=" * 84)
print("Masked PS @ paper cut ell100-1024, 14000 footprint — Fisher across processing versions")
print("=" * 84)
res = {}
for tag in ["lmax1530", "lmax1535"]:
    cov, nfeat, hart = fisher_cov(load_masked("grid", tag), load_masked("fid", tag))
    res[tag] = stats(cov)
    print(f"  {tag}: nfeat={nfeat} hartlap={hart:.2f}  "
          f"sig(Om)={res[tag]['sOm']:.4f} sig(S8)={res[tag]['sS8']:.4f} sig(w0)={res[tag]['sw0']:.4f}  "
          f"A(Om,w0)={res[tag]['A_Omw0']:.2e}  r(Om,w0)={res[tag]['r_Omw0']:+.2f}")
print(f"\n  ratio 1535/1530:  sig(S8) x{res['lmax1535']['sS8']/res['lmax1530']['sS8']:.3f}  "
      f"sig(w0) x{res['lmax1535']['sw0']/res['lmax1530']['sw0']:.3f}  "
      f"A(Om,w0) x{res['lmax1535']['A_Omw0']/res['lmax1530']['A_Omw0']:.3f}  "
      f"(>1 => new looser; <1 => new tighter)")

# ---- published paper NPE posterior width (anchor) ----
paper = sorted(glob.glob(f"{SAMP}/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
                         f"l100-1000_r10_masked_14000sqdeg_apod2.0_master_noisy_s0.26*.npy"))
print("\n" + "=" * 84)
print(f"Published PAPER NPE posterior (l100-1000 r10, masked 14000): {len(paper)} runs")
print("=" * 84)
if paper:
    sig = np.array([np.load(p)[:, [0, 1, 2]].std(0) for p in paper])
    m, sd = sig.mean(0), sig.std(0)
    print(f"  paper NPE sig(Om)={m[0]:.4f}±{sd[0]:.4f}  sig(S8)={m[1]:.4f}±{sd[1]:.4f}  "
          f"sig(w0)={m[2]:.4f}±{sd[2]:.4f}   (mean±run-scatter over {len(paper)} runs)")
    print("  (Fisher uses coarse bandpowers vs the NPE's r10, so absolute sigmas differ; the")
    print("   version-to-version Fisher RATIO above is the clean 'new vs old' statement.)")
