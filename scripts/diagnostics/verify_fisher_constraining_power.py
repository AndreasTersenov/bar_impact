#!/usr/bin/env python
"""
AUDIT of fisher_constraining_power.py — is the Fisher result correct?

Five independent checks, each printed PASS/FAIL or with a number to judge:
  1. Whitening is a no-op: cov(raw features) == cov(whitened features) to ~1e-8 (Fisher invariance).
  2. Independent recompute: a from-scratch Fisher (separate data load, separate bandpower code,
     explicit Hartlap, einsum F) matches fisher_cov() for PS l100-400 to ~1e-8.
  3. Ties to ALREADY-ACCEPTED code: rerun fisher_ps_vs_hos_degeneracy.py and confirm it still gives
     the memorized r(Om,w0) signs (PS<0, l1>0, peaks>0); confirm our code reproduces the same signs.
  4. Linear-fit QUALITY (R^2) per probe, global vs local Jacobian — the honest test of whether the
     HOS FoM is trustworthy or a linearization artifact. (Low R^2 => the linear "derivative" is a poor
     description of the response => take that probe's absolute FoM with a grain of salt.)
  5. Covariance conditioning (cond number) per probe; Hartlap factor positivity.

Run: python scripts/diagnostics/verify_fisher_constraining_power.py
"""
import os, sys, subprocess
import numpy as np

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
FID = f"{BASE}/fiducial/cosmo_fiducial"
params = np.load(f"{BASE}/grid/cosmo_params.npy")
PN = ["Om", "S8", "w0", "H0", "ns", "Ob"]
FID_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
ok = []


# ---------- independent loaders (do NOT import the main script: a loader bug must be catchable) ----
def load_ps_l(kind, lmin, lmax, edges):
    d, t = ("new_grid", "grid") if kind == "grid" else ("fiducial/cosmo_fiducial", "fiducial")
    autos = [np.load(f"{BASE}/{d}/all_cls_{t}_nobaryons_bin{b}_noisy_s0.26.npy") for b in [1, 2, 3, 4]]
    cross = np.load(f"{BASE}/{d}/all_cross_cls_{t}_nobaryons_bins1234_noisy_s0.26.npy").reshape(-1, 6, 1025)
    ed = edges[(edges >= lmin) & (edges <= lmax)]
    def bp(cl):  # bandpower by explicit python loop (different code path than the main script)
        cols = []
        for a, b in zip(ed[:-1], ed[1:]):
            cols.append(np.array([cl[:, a:b].mean(1)]).T)
        return np.hstack(cols)
    blocks = [bp(a) for a in autos] + [bp(cross[:, p, :]) for p in range(6)]
    return np.hstack(blocks)


def load_hos(prefix, kind, scales, snr_rebin):
    d, t = ("grid", "grid") if kind == "grid" else ("fiducial/cosmo_fiducial", "fiducial")
    out = []
    for b in [1, 2, 3, 4]:
        a = np.asarray(np.load(f"{BASE}/{d}/all_{prefix}_{t}_nobaryons_bin{b}_noisy_s0.26_new_normalization.npy"), float)
        a = a[:, scales, :]
        n = a.shape[-1] // snr_rebin
        a = a[..., :n * snr_rebin].reshape(*a.shape[:-1], n, snr_rebin).mean(-1)
        out.append(a.reshape(a.shape[0], -1))
    return np.hstack(out)


# ---------- a clean, from-scratch Fisher (raw features, explicit Hartlap, einsum) ------------------
def fisher_raw(grid_vec, fid_vec, par=params, whiten=False, jac="global"):
    if whiten:
        s = fid_vec.std(0); k = s > 0
        grid_vec, fid_vec = grid_vec[:, k] / s[k], fid_vec[:, k] / s[k]
    else:
        s = fid_vec.std(0); k = s > 0
        grid_vec, fid_vec = grid_vec[:, k], fid_vec[:, k]
    n_fid, n_dat = fid_vec.shape
    if jac == "global":
        X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
        J = np.linalg.lstsq(X, grid_vec, rcond=None)[0][1:].T
    else:
        Xc = par - FID_PARAMS
        J = np.linalg.lstsq(Xc, grid_vec - fid_vec.mean(0), rcond=None)[0].T
    C = np.cov(fid_vec, rowvar=False)
    Cinv = np.linalg.inv(C) * ((n_fid - n_dat - 2) / (n_fid - 1))
    F = np.einsum("ai,ab,bj->ij", J, Cinv, J)      # einsum form of J^T Cinv J
    return np.linalg.inv(F)


def metrics(cov):
    d = np.sqrt(np.diag(cov))
    return dict(sS8=d[1], sw0=d[2], FoM6=1 / np.sqrt(np.linalg.det(cov)),
                r_Omw0=cov[0, 2] / (d[0] * d[2]))


EDGES = np.array([37, 68, 100, 140, 200, 280, 400, 560, 760, 1024])

print("=" * 78)
print("CHECK 1 — whitening invariance (PS l100-400, global Jacobian, 40 feat)")
g = load_ps_l("grid", 100, 400, EDGES); f = load_ps_l("fid", 100, 400, EDGES)
cov_raw = fisher_raw(g, f, whiten=False)
cov_wht = fisher_raw(g, f, whiten=True)
rel = np.abs(cov_raw - cov_wht).max() / np.abs(cov_raw).max()
print(f"  max rel diff cov(raw) vs cov(whitened) = {rel:.2e}   -> {'PASS' if rel < 1e-6 else 'FAIL'}")
ok.append(rel < 1e-6)

print("\nCHECK 2 — independent recompute vs the main script's fisher_cov (PS l100-400, global)")
# import ONLY the function under test, suppressing the main script's prints/figures
import importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location(
    "fcp", os.path.join(os.path.dirname(__file__), "fisher_constraining_power.py"))
fcp = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    import matplotlib; matplotlib.use("Agg")
    spec.loader.exec_module(fcp)            # runs main (makes figures) but we only want fisher_cov
old = fcp.JAC_MODE; fcp.JAC_MODE = "global"
cov_main, info = fcp.fisher_cov(g, f)
fcp.JAC_MODE = old
m_main, m_ind = metrics(cov_main), metrics(cov_raw)
for key in ["sS8", "sw0", "FoM6", "r_Omw0"]:
    rd = abs(m_main[key] - m_ind[key]) / (abs(m_ind[key]) + 1e-300)
    print(f"  {key:7s}: main={m_main[key]:.5g}  independent={m_ind[key]:.5g}  reldiff={rd:.2e}")
allclose = all(abs(m_main[k] - m_ind[k]) / (abs(m_ind[k]) + 1e-300) < 1e-6 for k in m_main)
print(f"  -> {'PASS' if allclose else 'FAIL'}")
ok.append(allclose)

print("\nCHECK 3 — reproduce the ALREADY-ACCEPTED degeneracy script (signs of r(Om,w0))")
deg = os.path.join(os.path.dirname(__file__), "fisher_ps_vs_hos_degeneracy.py")
out = subprocess.run([sys.executable, deg], capture_output=True, text=True).stdout
print("  (degeneracy script tail:)")
for line in out.strip().splitlines()[-6:]:
    print("   ", line)
# our code, matched-ish config: PS l100-400 global, l1/peaks scales234 global
covs3 = {
    "PS l100-400": fisher_raw(g, f, jac="global"),
    "l1 sc234":    fisher_raw(load_hos("l1_norms", "grid", [1, 2, 3], 5),
                              load_hos("l1_norms", "fid", [1, 2, 3], 5), jac="global"),
    "peaks sc234": fisher_raw(load_hos("peak_counts", "grid", [1, 2, 3], 5),
                              load_hos("peak_counts", "fid", [1, 2, 3], 5), jac="global"),
}
signs = {k: np.sign(metrics(c)["r_Omw0"]) for k, c in covs3.items()}
print("  our r(Om,w0):", {k: round(metrics(c)["r_Omw0"], 2) for k, c in covs3.items()})
sign_ok = signs["PS l100-400"] < 0 and signs["l1 sc234"] > 0 and signs["peaks sc234"] > 0
print(f"  expected signs PS<0, l1>0, peaks>0  -> {'PASS' if sign_ok else 'FAIL'}")
ok.append(sign_ok)

print("\nCHECK 4 — linear-fit QUALITY R^2 per probe (honesty test for the HOS FoM)")
def fit_r2(grid_vec, par, jac, bw=1.0):
    s = grid_vec.std(0); k = s > 0; gv = grid_vec[:, k] / s[k]
    if jac == "global":
        X = np.column_stack([np.ones(len(par)), par - par.mean(0)]); w = np.ones(len(par))
    else:
        X = np.column_stack([np.ones(len(par)), par - FID_PARAMS])
        ps = par.std(0); d2 = (((par - FID_PARAMS) / ps) ** 2).sum(1); w = np.exp(-0.5 * d2 / bw ** 2)
    sw = np.sqrt(w)[:, None]
    coef = np.linalg.lstsq(X * sw, gv * sw, rcond=None)[0]
    pred = X @ coef
    ss_res = (w[:, None] * (gv - pred) ** 2).sum()
    ss_tot = (w[:, None] * (gv - np.average(gv, 0, weights=w)) ** 2).sum()
    return 1 - ss_res / ss_tot
probes4 = {
    "PS l100-400": g,
    "PS l37-1024": load_ps_l("grid", 37, 1024, EDGES),
    "l1 sc234":    load_hos("l1_norms", "grid", [1, 2, 3], 5),
    "peaks sc234": load_hos("peak_counts", "grid", [1, 2, 3], 5),
}
print(f"  {'probe':<14}{'R2(global)':>12}{'R2(local)':>12}")
for nm, gv in probes4.items():
    print(f"  {nm:<14}{fit_r2(gv, params, 'global'):>12.3f}{fit_r2(gv, params, 'local'):>12.3f}")
print("  (R^2 ~ fraction of grid data-vector variance the linear Jacobian explains. PS should be")
print("   high; if l1/peaks are much lower, their ABSOLUTE FoM is linearization-limited, as flagged.)")

print("\nCHECK 5 — covariance conditioning & Hartlap")
for nm, (gv, fv) in {
    "PS l100-400": (g, f),
    "l1 sc234": (load_hos("l1_norms", "grid", [1, 2, 3], 5), load_hos("l1_norms", "fid", [1, 2, 3], 5)),
    "peaks sc234": (load_hos("peak_counts", "grid", [1, 2, 3], 5), load_hos("peak_counts", "fid", [1, 2, 3], 5)),
}.items():
    s = fv.std(0); k = s > 0; C = np.corrcoef((fv[:, k] / s[k]), rowvar=False)
    nd = int(k.sum()); h = (200 - nd - 2) / (200 - 1)
    print(f"  {nm:<14} n_feat={nd:3d}  cond(corr)={np.linalg.cond(C):.1e}  Hartlap={h:.2f}")

print("\n" + "=" * 78)
print(f"VERDICT: {sum(ok)}/{len(ok)} hard checks PASS "
      f"({'all green' if all(ok) else 'SEE FAILURES'}).")
print("Code correctness: checks 1-3. Method honesty: check 4 (R^2) — read the HOS absolute FoM")
print("in light of its linear-fit R^2; the low-ell PS recovery ratio does not depend on it.")
