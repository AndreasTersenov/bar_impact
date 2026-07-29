#!/usr/bin/env python
"""
PS vs HOS CONSTRAINING POWER (full-sky, noisy s0.26) — Gaussian Fisher forecast.

Science questions:
  (1) How much does recovering the low-ell band (l37-100) tighten the power spectrum,
      i.e. PS l37-1024 vs the paper's PS l100-1024?
  (2) How much do the higher-order statistics (l1-norm, peak counts, scales234) add over
      the PS, on a (roughly) matched footprint?
  (3) Does PS + HOS combined beat either alone, and by how much?

Method: a linear Jacobian of the grid data vector on the cosmological parameters (lstsq over
the full 16965-cosmology grid) + a Hartlap-corrected sample covariance from the 200 fiducial
perms. Fisher F = J^T Cinv J ; parameter covariance = inv(F). We report the marginalized
FoM: sigma(Om), sigma(S8), sigma(w0), the (Om,S8) and (Om,w0) 1-sigma ellipse areas, and a
6-parameter FoM = 1/sqrt(det(param_cov)).

CAVEATS (read before over-interpreting — this is NOT a clean bound on the HOS):
  * The COVARIANCE is the empirical fiducial-perm covariance, so it already contains the HOS
    non-Gaussian VARIANCE (it is not a Gaussian-covariance approximation). What is approximate
    is (a) the Gaussian LIKELIHOOD shape and (b) the JACOBIAN: a linear response fit over the
    grid. The Jacobian is the dominant approximation and it can over- OR under-state a probe's
    sensitivity -- so do NOT read the Fisher HOS FoM as a lower (or upper) bound on the NPE.
  * The HOS gain is JACOBIAN-SENSITIVE. l1's 6-param FoM lead over PS l100 is ~x70 with the
    global/anchored (whole-grid) linear Jacobian but ~x17 with a LOCAL (fiducial-neighbourhood)
    derivative -- the global linearization inflates a steeply-nonlinear statistic. See the
    ROBUSTNESS block. The low-ell PS recovery gain (l37 vs l100) is robust across modes.
  * 200 fiducial perms may UNDER-estimate the non-Gaussian covariance tails of l1/peaks, which
    would make the HOS look optimistically tight. Treat absolute HOS FoM with caution; the
    ground truth is the NPE.
  * Covariance invertibility: 200 fiducial perms => keep n_features < ~190. We bin to keep
    every probe (and the PS+HOS combined) below that; the Hartlap factor is printed per probe.
    The combined PS+HOS rows have Hartlap ~0.25 (150 feat) -- usable but the precision matrix
    is noisy; trim features for production numbers.
  * Full-sky (data ready, no reprocess). The paper analysis is MASKED; the masked low-ell PS
    needs the gated 6-mask nlb=4+submean production (see memory nlb4-submean-gate-passed).

Single, consistent binning is used everywhere so probe-to-probe deltas are clean (not a
binning artifact). Edit the CONFIG block to vary it.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ================================= CONFIG (edit to taste) =====================================
BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
GRID_PS = f"{BASE}/new_grid"          # full-sky healpy PS grid
GRID_HOS = f"{BASE}/grid"             # l1 / peaks grid
FID = f"{BASE}/fiducial/cosmo_fiducial"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/constraining_power"
os.makedirs(OUT, exist_ok=True)

PN = ["Om", "S8", "w0", "H0", "ns", "Ob"]
FID_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])  # repo Fisher fiducial cosmology
# JAC_MODE: how the linear response (Jacobian) is estimated from the grid.
#   "anchored" -> repo-consistent: slope through the fiducial (lstsq of grid-fid on params-fid).
#   "global"   -> free OLS slope with intercept (whole-grid best linear fit).
#   "local"    -> Gaussian-kernel-weighted slope anchored at the fiducial (LOCAL_BW in whitened
#                 param std units) -> tests whether the global linearization overstates a nonlinear
#                 (e.g. l1) response.
JAC_MODE = "local"          # HEADLINE: local derivative at the fiducial (honest for nonlinear HOS)
LOCAL_BW = 1.0              # Gaussian kernel bandwidth in whitened param-std units
# Bandpower edges shared by every PS cut: fine at low ell (isolate the l37-100 recovery band),
# coarse at high ell (the PS is smooth and mode-rich there, so binning preserves Gaussian info).
PS_EDGES = np.array([37, 68, 100, 140, 200, 280, 400, 560, 760, 1024])
# Fair scale pairing with the PS regime: the full-ell PS (to l1024) keeps its small scales, so the
# full-ell HOS keep their SMALLEST wavelet scale too (scales1234 = idx 0,1,2,3); the baryon-safe PS
# (lmax 400) drops small scales, so the baryon-safe HOS drop the finest wavelet scale (scales234 =
# idx 1,2,3). idx 0 = finest/highest-ell band (pycs scale_arcmin = 2^(i+1)); idx 4 (coarse) always
# excluded (mass-sheet). Naming: "scales{i+1...}" so idx[0,1,2,3]->scales1234, idx[1,2,3]->scales234.
HOS_SCALES_FULL = [0, 1, 2, 3]        # full-ell-fair HOS (includes smallest wavelet scale)
HOS_SCALES_BSAFE = [1, 2, 3]          # baryon-safe HOS (drops smallest wavelet scale)
HOS_SCALES = HOS_SCALES_BSAFE         # hos_vec default (overridden per-probe below)
HOS_SNR_REBIN = 8                     # 40 SNR bins -> 5 per scale
# Coarser binning used ONLY for the PS+HOS COMBINED vectors, so the joint covariance keeps a
# healthy Hartlap factor (>~0.5). The matched-coarse standalones are reported alongside so the
# "gain from adding HOS" is computed at identical binning, not confounded by feature count.
PS_EDGES_COARSE = np.array([37, 100, 200, 400, 760, 1024])   # l37 -> 5 bands/spec ; l100 -> 4
HOS_SNR_REBIN_COARSE = 10             # 40 -> 4 SNR bins ; 4 tomo x 3 scale x 4 = 48 features
# ==============================================================================================

params = np.load(f"{BASE}/grid/cosmo_params.npy")   # (16965, 6) = [Om, S8, w0, H0, ns, Ob]


# --------------------------------- data-vector builders --------------------------------------
def _bandpower(cl, edges):
    """Average Cl (n, n_ell) over bandpower edges; column index == multipole ell."""
    return np.stack([cl[:, lo:hi].mean(1) for lo, hi in zip(edges[:-1], edges[1:])], axis=1)


def ps_vec(kind, lmin, lmax, edge_set=None):
    """Full-sky auto+cross PS bandpowers for the cut [lmin, lmax]. kind in {'grid','fid'}."""
    es = PS_EDGES if edge_set is None else edge_set
    edges = es[(es >= lmin) & (es <= lmax)]
    d, t = (GRID_PS, "grid") if kind == "grid" else (FID, "fiducial")
    autos = [np.load(f"{d}/all_cls_{t}_nobaryons_bin{b}_noisy_s0.26.npy") for b in [1, 2, 3, 4]]
    cross = np.load(f"{d}/all_cross_cls_{t}_nobaryons_bins1234_noisy_s0.26.npy")
    cross = cross.reshape(cross.shape[0], 6, 1025)
    blocks = [_bandpower(a, edges) for a in autos] + [_bandpower(cross[:, p, :], edges) for p in range(6)]
    return np.concatenate(blocks, axis=1)


def hos_vec(prefix, kind, scales=HOS_SCALES, snr_rebin=HOS_SNR_REBIN):
    """l1 / peaks SNR histograms, selected scales, rebinned. kind in {'grid','fid'}."""
    d, t = (GRID_HOS, "grid") if kind == "grid" else (FID, "fiducial")
    out = []
    for b in [1, 2, 3, 4]:
        a = np.load(f"{d}/all_{prefix}_{t}_nobaryons_bin{b}_noisy_s0.26_new_normalization.npy")
        a = np.asarray(a, float)[:, scales, :]               # (n, n_scale, 40)
        n = a.shape[-1] // snr_rebin
        a = a[..., :n * snr_rebin].reshape(*a.shape[:-1], n, snr_rebin).mean(-1)
        out.append(a.reshape(a.shape[0], -1))
    return np.concatenate(out, axis=1)


# ------------------------------------- Fisher core -------------------------------------------
def fisher_cov(grid_vec, fid_vec, par=params):
    """6-param parameter covariance from a linear Jacobian (lstsq) + Hartlap sample cov.

    Whitening: each feature is normalized by its fiducial std before building J and C. This is
    Fisher-INVARIANT (F = J^T Cinv J is unchanged by a per-feature rescale) but it conditions C
    and makes the dead-column cut scale-independent -- essential for combined PS+HOS vectors,
    where the tiny-amplitude PS Cls would otherwise be wiped out by a variance floor set by the
    much larger HOS features.

    Returns (cov6x6, info) where info records n_features and the Hartlap factor for transparency.
    """
    s = fid_vec.std(0)
    keep = s > 0                                              # drop only truly dead columns
    grid_vec = grid_vec[:, keep] / s[keep]
    fid_vec = fid_vec[:, keep] / s[keep]
    n_fid, n_dat = fid_vec.shape
    fid_mean = fid_vec.mean(0)                                # whitened fiducial data mean

    if JAC_MODE == "global":
        X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
        coef, *_ = np.linalg.lstsq(X, grid_vec, rcond=None)
        J = coef[1:].T                                        # (n_dat, 6)
    else:  # "anchored" or "local": slope through the fiducial (no intercept), repo-consistent
        Xc = par - FID_PARAMS
        Yc = grid_vec - fid_mean
        if JAC_MODE == "local":
            ps = par.std(0)
            d2 = (((par - FID_PARAMS) / ps) ** 2).sum(1)      # whitened param distance^2
            w = np.exp(-0.5 * d2 / LOCAL_BW ** 2)
            sw = np.sqrt(w)[:, None]
            sol, *_ = np.linalg.lstsq(Xc * sw, Yc * sw, rcond=None)
        else:
            sol, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
        J = sol.T                                             # (n_dat, 6)
    C = np.cov(fid_vec, rowvar=False)
    if n_dat < n_fid - 2:
        hartlap = (n_fid - n_dat - 2) / (n_fid - 1)
        Cinv = np.linalg.inv(C) * hartlap
        mode = "full+Hartlap"
    else:
        hartlap = float("nan")
        Cinv = np.diag(1.0 / np.diag(C))
        mode = "DIAG (n_feat too large!)"
    cov = np.linalg.inv(J.T @ Cinv @ J)
    return cov, {"n_feat": int(n_dat), "hartlap": hartlap, "mode": mode}


def fom(cov):
    """Marginalized FoM metrics from a 6x6 parameter covariance."""
    d = np.sqrt(np.diag(cov))
    def area(i, j):
        s = cov[np.ix_([i, j], [i, j])]
        return float(np.pi * np.sqrt(np.linalg.det(s)))
    return {
        "sig_Om": float(d[0]), "sig_S8": float(d[1]), "sig_w0": float(d[2]),
        "area_Om_S8": area(0, 1), "area_Om_w0": area(0, 2),
        "corr_Om_w0": float(cov[0, 2] / (d[0] * d[2])),
        "FoM6": float(1.0 / np.sqrt(np.linalg.det(cov))),
    }


# ------------------------- local-Jacobian effective sample size ------------------------------
# Report how many grid cosmologies effectively enter the LOCAL derivative, so the bandwidth is
# transparent (N_eff = (sum w)^2 / sum w^2 for the Gaussian kernel in whitened param-std units).
_ps = params.std(0)
_w = np.exp(-0.5 * (((params - FID_PARAMS) / _ps) ** 2).sum(1) / LOCAL_BW ** 2)
N_EFF_LOCAL = (_w.sum() ** 2) / (_w ** 2).sum()
print(f"\nJacobian mode = '{JAC_MODE}'.  LOCAL kernel bw={LOCAL_BW} -> N_eff = {N_EFF_LOCAL:.0f} "
      f"of {len(params)} grid cosmologies feed the derivative.")


# ----------------------------------- assemble probes -----------------------------------------
def cat(*vs):
    return np.concatenate(vs, axis=1)

# ---- (A) STANDALONE single-probe FoM: rich binning, healthy Hartlap (each well below 198 feat).
ps_cuts = {
    "PS l100-1024 (paper)":      (100, 1024),
    "PS l37-1024 (recovered)":   (37, 1024),
    "PS l37-400 (HOS-l-matched)": (37, 400),
    "PS l37-280 (HOS-l-tight)":  (37, 280),
    "PS l100-400":               (100, 400),
}
PSg = {k: ps_vec("grid", lo, hi) for k, (lo, hi) in ps_cuts.items()}
PSf = {k: ps_vec("fid", lo, hi) for k, (lo, hi) in ps_cuts.items()}
# HOS in both scale sets: scales1234 (full-ell-fair) and scales234 (baryon-safe)
def _hos(prefix, scales):
    return (hos_vec(prefix, "grid", scales=scales), hos_vec(prefix, "fid", scales=scales))
L1full, L1bs = _hos("l1_norms", HOS_SCALES_FULL), _hos("l1_norms", HOS_SCALES_BSAFE)
PKfull, PKbs = _hos("peak_counts", HOS_SCALES_FULL), _hos("peak_counts", HOS_SCALES_BSAFE)

standalone = {k: (PSg[k], PSf[k]) for k in ps_cuts}
standalone["l1 scales1234"] = L1full          # full-ell-fair (with smallest scale)
standalone["peaks scales1234"] = PKfull
standalone["l1 scales234"] = L1bs             # baryon-safe (smallest scale dropped)
standalone["peaks scales234"] = PKbs
standalone["peaks scales123"] = _hos("peak_counts", [0, 1, 2])   # matches on-disk full-sky pc NPE

# ---- (B) COMBINED PS+HOS: coarse matched binning so the JOINT covariance keeps Hartlap >~0.5.
#      Full-ell regime -> HOS scales1234 (fair vs the l1024 PS). Coarse standalones included so the
#      "gain from adding HOS" is at matched binning.
def ps_vec_c(kind, lo, hi):
    return ps_vec(kind, lo, hi, edge_set=PS_EDGES_COARSE)
def hos_vec_c(prefix):
    return (hos_vec(prefix, "grid", scales=HOS_SCALES_FULL, snr_rebin=HOS_SNR_REBIN_COARSE),
            hos_vec(prefix, "fid", scales=HOS_SCALES_FULL, snr_rebin=HOS_SNR_REBIN_COARSE))

PSc = {nm: (ps_vec_c("grid", lo, hi), ps_vec_c("fid", lo, hi))
       for nm, (lo, hi) in [("PS l100-1024", (100, 1024)), ("PS l37-1024", (37, 1024))]}
L1c = hos_vec_c("l1_norms")
PKc = hos_vec_c("peak_counts")
combined = {
    "PS l100-1024 [coarse]": PSc["PS l100-1024"],
    "PS l37-1024 [coarse]":  PSc["PS l37-1024"],
    "l1 sc1234 [coarse]":    L1c,
    "peaks sc1234 [coarse]": PKc,
    "PS l100 + l1":   (cat(PSc["PS l100-1024"][0], L1c[0]), cat(PSc["PS l100-1024"][1], L1c[1])),
    "PS l37 + l1":    (cat(PSc["PS l37-1024"][0], L1c[0]),  cat(PSc["PS l37-1024"][1], L1c[1])),
    "PS l100 + peaks": (cat(PSc["PS l100-1024"][0], PKc[0]), cat(PSc["PS l100-1024"][1], PKc[1])),
    "PS l37 + peaks":  (cat(PSc["PS l37-1024"][0], PKc[0]),  cat(PSc["PS l37-1024"][1], PKc[1])),
}

results, covs = {}, {}
for name, (g, f) in {**standalone, **combined}.items():
    cov, info = fisher_cov(g, f)
    covs[name] = cov
    results[name] = {**fom(cov), **info}


# ------------------------------------- print tables ------------------------------------------
def _print_table(title, names):
    hdr = f"{'probe':<28}{'nfeat':>6}{'hart':>6}  {'sig(Om)':>8}{'sig(S8)':>8}{'sig(w0)':>8}" \
          f"{'A(Om,S8)':>10}{'A(Om,w0)':>10}{'r(Om,w0)':>9}{'FoM6':>11}"
    print("\n" + "=" * len(hdr)); print(title); print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for k in names:
        r = results[k]
        hh = "  -- " if np.isnan(r["hartlap"]) else f"{r['hartlap']:.2f}"
        print(f"{k:<28}{r['n_feat']:>6}{hh:>6}  {r['sig_Om']:>8.4f}{r['sig_S8']:>8.4f}{r['sig_w0']:>8.4f}"
              f"{r['area_Om_S8']:>10.2e}{r['area_Om_w0']:>10.2e}{r['corr_Om_w0']:>+9.2f}{r['FoM6']:>11.3e}")
    print("-" * len(hdr))

_print_table("(A) STANDALONE  (full-sky, noisy s0.26, Fisher; rich binning)", list(standalone.keys()))
_print_table("(B) COMBINED PS+HOS  (coarse matched binning, joint covariance)", list(combined.keys()))

def ratio(a, b, key):
    return results[a][key] / results[b][key]
print("\nKEY RATIOS (smaller sigma / larger FoM = better):")
print(f"  low-ell recovery (PS l37 vs l100):   sig(S8) x{ratio('PS l37-1024 (recovered)','PS l100-1024 (paper)','sig_S8'):.3f}"
      f"   sig(w0) x{ratio('PS l37-1024 (recovered)','PS l100-1024 (paper)','sig_w0'):.3f}"
      f"   FoM6 x{ratio('PS l37-1024 (recovered)','PS l100-1024 (paper)','FoM6'):.2f}")
print("  -- full-ell regime: HOS use scales1234 (fair vs the l1024 PS) --")
print(f"  l1 sc1234 over PS l100  (FoM6):      x{ratio('l1 scales1234','PS l100-1024 (paper)','FoM6'):.2f}")
print(f"  peaks sc1234 over PS l100 (FoM6):    x{ratio('peaks scales1234','PS l100-1024 (paper)','FoM6'):.2f}")
print("  -- baryon-safe regime: HOS use scales234 vs PS l100-400 --")
print(f"  l1 sc234 over PS l100-400 (FoM6):    x{ratio('l1 scales234','PS l100-400','FoM6'):.2f}")
print(f"  peaks sc234 over PS l100-400 (FoM6): x{ratio('peaks scales234','PS l100-400','FoM6'):.2f}")
print("  -- combined gains at MATCHED coarse binning (Hartlap-clean): --")
print(f"  PS l37 + l1 over PS l37 (FoM6):      x{ratio('PS l37 + l1','PS l37-1024 [coarse]','FoM6'):.2f}")
print(f"  PS l37 + peaks over PS l37 (FoM6):   x{ratio('PS l37 + peaks','PS l37-1024 [coarse]','FoM6'):.2f}")
print(f"  PS l100 + peaks over PS l100 (FoM6): x{ratio('PS l100 + peaks','PS l100-1024 [coarse]','FoM6'):.2f}")


# --------------------- robustness: Jacobian mode (global / anchored / local) -----------------
def _fom6_under_mode(grid_vec, fid_vec, mode):
    global JAC_MODE
    old = JAC_MODE; JAC_MODE = mode
    try:
        cov, _ = fisher_cov(grid_vec, fid_vec)
    finally:
        JAC_MODE = old
    return fom(cov)["FoM6"]

print("\nROBUSTNESS — FoM6 under each Jacobian mode (headline = local; does the ranking survive?):")
print(f"  {'probe':<26}{'global':>12}{'anchored':>12}{'local(bw=%.1f)' % LOCAL_BW:>14}")
for nm in ["PS l100-1024 (paper)", "PS l37-1024 (recovered)", "l1 scales1234", "peaks scales1234"]:
    g, f = standalone[nm]
    row = [_fom6_under_mode(g, f, m) for m in ("global", "anchored", "local")]
    print(f"  {nm:<26}{row[0]:>12.2e}{row[1]:>12.2e}{row[2]:>14.2e}")
_l1, _ps100 = standalone['l1 scales1234'], standalone['PS l100-1024 (paper)']
print(f"  (FoM6 ratio l1/PS-l100:  global x{_fom6_under_mode(*_l1,'global')/_fom6_under_mode(*_ps100,'global'):.1f}"
      f"  anchored x{_fom6_under_mode(*_l1,'anchored')/_fom6_under_mode(*_ps100,'anchored'):.1f}"
      f"  local x{_fom6_under_mode(*_l1,'local')/_fom6_under_mode(*_ps100,'local'):.1f})")

# names used by the figures below
probes = {**standalone, **combined}


# --------------------------- bar figure (A): STANDALONE, rich binning ------------------------
metrics = [("FoM6", "6-parameter FoM  (1/√det C)", False),
           ("sig_S8", r"$\sigma(S_8)$", True),
           ("sig_w0", r"$\sigma(w_0)$", True),
           ("area_Om_w0", r"$(\Omega_m,w_0)$ 1$\sigma$ area", True)]
bar_order = ["PS l100-1024 (paper)", "PS l37-1024 (recovered)", "PS l37-280 (HOS-l-tight)",
             "l1 scales1234", "peaks scales1234"]
short = {"PS l100-1024 (paper)": "PS\nl100", "PS l37-1024 (recovered)": "PS\nl37",
         "PS l37-280 (HOS-l-tight)": "PS\nl37-280", "l1 scales1234": "l1\nsc1234",
         "peaks scales1234": "peaks\nsc1234"}
colors = ["#4d4d4d", "#7f7f7f", "#bcbd22", "#1f77b4", "#2ca02c"]

def _bars(fig_path, order, sub, title_suffix):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, title, smaller_better) in zip(axes.ravel(), metrics):
        vals = [results[k][key] for k in order]
        ax.bar(range(len(order)), vals, color=[colors[i % len(colors)] for i in range(len(order))],
               width=0.7, zorder=3)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([sub[k] for k in order], fontsize=8)
        ax.set_title(title + ("   (lower = better)" if smaller_better else "   (higher = better)"), fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2g}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("PS vs HOS constraining power (full-sky, noisy s0.26, Fisher, LOCAL Jacobian). "
                 + title_suffix, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{fig_path}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{fig_path}.pdf", bbox_inches="tight")
    plt.close(fig)

_bars(f"{OUT}/fisher_fom_bars", bar_order,
      short, "Standalone probes, rich binning. HOS gain is Jacobian-sensitive — see robustness.")
print(f"\nwrote {OUT}/fisher_fom_bars.{{png,pdf}}")

# --------------------------- bar figure (B): COMBINED gain, coarse matched binning -----------
comb_order = ["PS l37-1024 [coarse]", "l1 sc1234 [coarse]", "peaks sc1234 [coarse]",
              "PS l37 + l1", "PS l37 + peaks", "PS l100 + peaks"]
comb_short = {"PS l37-1024 [coarse]": "PS l37", "l1 sc1234 [coarse]": "l1", "peaks sc1234 [coarse]": "peaks",
              "PS l37 + l1": "PS l37\n+l1", "PS l37 + peaks": "PS l37\n+peaks",
              "PS l100 + peaks": "PS l100\n+peaks"}
_bars(f"{OUT}/fisher_combined_bars", comb_order, comb_short,
      "Combined PS+HOS at MATCHED coarse binning (Hartlap~0.5) — does HOS add over PS?")
print(f"wrote {OUT}/fisher_combined_bars.{{png,pdf}}")


# ------------------------------- contour (ellipse) figures -----------------------------------
def ellipse(ax, cov, i, j, color, label, lw=2.0, ls="-"):
    s = cov[np.ix_([i, j], [i, j])]
    w, V = np.linalg.eigh(s)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * 1.52 * np.sqrt(w[1]), 2 * 1.52 * np.sqrt(w[0]),  # 1.52 = 68% (2D)
                         angle=ang, fill=False, edgecolor=color, lw=lw, ls=ls, label=label))

pairs = [(0, 1, r"$\Delta\Omega_m$", r"$\Delta S_8$"), (0, 2, r"$\Delta\Omega_m$", r"$\Delta w_0$")]

def contour_fig(cmp_probes, fname, suptitle):
    fig2, axc = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (i, j, xl, yl) in zip(axc, pairs):
        for name, col, ls in cmp_probes:
            ellipse(ax, covs[name], i, j, col, name, ls=ls)
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.relim(); ax.autoscale_view(); ax.set_aspect("auto")
    axc[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig2.suptitle(suptitle, fontsize=11); fig2.tight_layout()
    fig2.savefig(f"{OUT}/{fname}.png", dpi=160, bbox_inches="tight")
    fig2.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight"); plt.close(fig2)
    print(f"wrote {OUT}/{fname}.{{png,pdf}}")

# (i) full-ell regime: PS to l1024, HOS scales1234 (fair — keeps the smallest wavelet scale)
contour_fig([("PS l100-1024 (paper)", "#4d4d4d", "-"), ("PS l37-1024 (recovered)", "#7f7f7f", "--"),
             ("l1 scales1234", "#1f77b4", "-"), ("peaks scales1234", "#2ca02c", "-")],
            "fisher_contours",
            "Fisher 68% contours — FULL-ell regime (PS to $\\ell{=}1024$; HOS scales1234, smallest "
            "scale kept; full-sky, noisy)")

# (ii) BARYON-SAFE regime: PS lmax=400, HOS scales234 (already excludes the finest/smallest scale)
contour_fig([("PS l100-400", "#4d4d4d", "-"), ("PS l37-400 (HOS-l-matched)", "#7f7f7f", "--"),
             ("l1 scales234", "#1f77b4", "-"), ("peaks scales234", "#2ca02c", "-")],
            "fisher_contours_baryon_safe",
            "Fisher 68% contours — BARYON-SAFE regime (PS $\\ell_{max}{=}400$; HOS scales234, "
            "smallest scale dropped; full-sky, noisy)")


# ------------------------------------- save results ------------------------------------------
np.savez(f"{OUT}/fisher_covs.npz", **{k.replace(" ", "_").replace("(", "").replace(")", ""): v
                                       for k, v in covs.items()}, param_names=np.array(PN))
with open(f"{OUT}/fisher_fom_table.json", "w") as fh:
    json.dump({"config": {"JAC_MODE": JAC_MODE, "LOCAL_BW": LOCAL_BW, "N_eff_local": float(N_EFF_LOCAL),
                          "PS_EDGES": PS_EDGES.tolist(), "PS_EDGES_COARSE": PS_EDGES_COARSE.tolist(),
                          "HOS_SCALES_FULL": HOS_SCALES_FULL, "HOS_SCALES_BSAFE": HOS_SCALES_BSAFE,
                          "HOS_SNR_REBIN": HOS_SNR_REBIN,
                          "HOS_SNR_REBIN_COARSE": HOS_SNR_REBIN_COARSE,
                          "n_fid": 200, "n_grid": len(params)},
               "results": results}, fh, indent=2)
print(f"wrote {OUT}/fisher_fom_table.json  and  fisher_covs.npz")

# ------------------------------------- provenance --------------------------------------------
# Standing rule (docs/HANDOFF_JZ_PAPER_FIGURES.md §0): each figure gets _values.csv and
# _provenance.json beside it. fisher_fom_table.json already holds the config and every
# probe's FoM, but it is keyed by probe across ALL regimes — it does not record which four
# probes a given contour figure actually drew, and the figure is what gets cited.
import csv as _csv, subprocess as _sub, datetime as _dt

_FIGURE_PROBES = {
    "fisher_contours": ["PS l100-1024 (paper)", "PS l37-1024 (recovered)",
                        "l1 scales1234", "peaks scales1234"],
    "fisher_contours_baryon_safe": ["PS l100-400", "PS l37-400 (HOS-l-matched)",
                                    "l1 scales234", "peaks scales234"],
}


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "unavailable"


try:
    _commit = _sub.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        stderr=_sub.DEVNULL, text=True).strip()
except Exception:
    _commit = "unknown"

for _fig, _probes in _FIGURE_PROBES.items():
    _stem = f"{OUT}/{_fig}"
    _keys = ("sig_Om", "sig_S8", "sig_w0", "area_Om_S8", "area_Om_w0", "corr_Om_w0",
             "FoM6", "n_feat", "hartlap")
    with open(_stem + "_values.csv", "w", newline="") as _fh:
        _w = _csv.writer(_fh)
        _w.writerow(["probe"] + list(_keys))
        for _p in _probes:
            _r = results.get(_p)
            if _r is None:
                print(f"[warn] {_fig}: probe {_p!r} absent from results")
                continue
            _w.writerow([_p] + [f"{_r[k]:.10g}" if isinstance(_r[k], float) else _r[k]
                                for k in _keys])
    # The ellipses are what the reader sees, so ship the 3x3 (Om,S8,w0) sub-covariance each
    # one was drawn from, not only the scalar summaries.
    with open(_stem + "_covariance.csv", "w", newline="") as _fh:
        _w = _csv.writer(_fh)
        _w.writerow(["probe", "row", "Om", "S8", "w0"])
        for _p in _probes:
            if _p not in covs:
                continue
            _c = np.asarray(covs[_p])[:3, :3]
            for _i, _nm in enumerate(("Om", "S8", "w0")):
                _w.writerow([_p, _nm] + [f"{v:.10g}" for v in _c[_i]])

    with open(_stem + "_provenance.json", "w") as _fh:
        json.dump({
            "figure": _fig,
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _commit,
            "probes_drawn": _probes,
            "contour_level": "68% 2D (1.52 sigma semi-axis scaling)",
            "regime": ("baryon-safe: PS lmax=400, HOS drop the finest wavelet scale "
                       "(scales234)" if "baryon_safe" in _fig else
                       "full-ell: PS to lmax=1024, HOS keep all four scales (scales1234)"),
            "footprint": "FULL SKY (healpy), noisy sigma_e=0.26 — NOT the masked paper analysis",
            "method": ("Gaussian Fisher: linear Jacobian over the 16965-cosmology grid + "
                       "Hartlap-corrected sample covariance from 200 fiducial perms"),
            "jacobian_mode": JAC_MODE,
            "local_bw": LOCAL_BW,
            "fiducial_cosmology": dict(zip(PN, FID_PARAMS.tolist())),
            "ps_edges": PS_EDGES.tolist(),
            "hos_scales_full": HOS_SCALES_FULL,
            "hos_scales_baryon_safe": HOS_SCALES_BSAFE,
            "versions": {m: _ver(m) for m in ("numpy", "scipy", "matplotlib")},
            "caveats": [
                "FISHER, not NPE. The Jacobian is a linear response fit and is the dominant "
                "approximation; it can over- OR under-state a probe's sensitivity, so do not "
                "read the HOS FoM as a bound on the NPE. See the module docstring.",
                "The HOS gain is jacobian-sensitive: l1's 6-param FoM lead over PS l100 is "
                "~x70 with a global linear Jacobian but ~x17 with the local derivative used "
                "here (JAC_MODE='local').",
                "200 fiducial perms may under-estimate the non-Gaussian covariance tails of "
                "l1/peaks, which would make the HOS look optimistically tight.",
                "FULL-SKY, not the masked paper footprint.",
                "Regenerated after the RAID0 disk failure destroyed both the .pdf and .png "
                "(100% zeros) and fisher_covs.npz. Every input .npy was verified readable "
                "first; fisher_fom_table_PRECRASH_REFERENCE.json is the surviving pre-crash "
                "table, kept for numerical comparison.",
            ],
        }, _fh, indent=2)
    print(f"wrote {_stem}_values.csv / _covariance.csv / _provenance.json")
