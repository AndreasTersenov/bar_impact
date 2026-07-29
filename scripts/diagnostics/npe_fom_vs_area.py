#!/usr/bin/env python3
"""Six-footprint summary: BNT-580/non-BNT-460 3-param FoM (Om,S8,w0) vs survey area, for the calibrated
score-compression NPE and the Fisher (hybrid). Reads the rollout outputs:
  NPE      : outputs/score_experiment/npe_score/posterior_summary_{tag}_{A}_mle.npy
  Fisher   : outputs/score_experiment/score/score_cache_{tag}_{A}_hybrid.npz  (F_x = full hybrid Fisher)
Plots the FoM3 ratio and per-config sigma(S8) vs area; prints the table. Run with jaxili python."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P3 = [0, 1, 2]
AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
NPE = "outputs/score_experiment/npe_score"
SC = "outputs/score_experiment/score"
OUT = "outputs/score_experiment"
CFG = ["nonbnt_460", "bnt_580"]


def fom3(c3):
    return 1.0 / np.sqrt(np.linalg.det(c3))


def npe_cov(tag, A):
    p = f"{NPE}/posterior_summary_{tag}_{A}_mle.npy"
    if not os.path.exists(p):
        return None
    return np.cov(np.load(p)[:, P3], rowvar=False), np.load(p)[:, P3]


def fisher_cov(tag, A):
    # 14000 hybrid cache may have no COVK suffix (built before tagging); try both.
    for nm in (f"score_cache_{tag}_{A}_hybrid.npz", f"score_cache_{tag}_{A}.npz"):
        p = f"{SC}/{nm}"
        if os.path.exists(p):
            return np.linalg.inv(np.load(p)["F_x"])[np.ix_(P3, P3)]
    return None


def boot_ratio(sn, sb, nboot=200):
    """bootstrap the BNT/non-BNT FoM3 ratio from pooled posterior samples."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(nboot):
        cn = np.cov(sn[rng.integers(0, len(sn), len(sn))].T)
        cb = np.cov(sb[rng.integers(0, len(sb), len(sb))].T)
        out.append(fom3(cb) / fom3(cn))
    return np.std(out)


rows = []
for A in AREAS:
    nn, sn = npe_cov("nonbnt_460", A) or (None, None)
    nb, sb = npe_cov("bnt_580", A) or (None, None)
    fn, fb = fisher_cov("nonbnt_460", A), fisher_cov("bnt_580", A)
    if nn is None or nb is None:
        print(f"A={A}: NPE incomplete, skipping"); continue
    npe_ratio = fom3(nb) / fom3(nn)
    fis_ratio = fom3(fb) / fom3(fn) if (fn is not None and fb is not None) else np.nan
    err = boot_ratio(sn, sb)
    rows.append((A, npe_ratio, err, fis_ratio,
                 np.sqrt(nn[1, 1]), np.sqrt(nb[1, 1])))
    print(f"A={A:5d}  NPE FoM3 ratio={npe_ratio:.2f}±{err:.2f}  Fisher ratio={fis_ratio:.2f}  "
          f"sigS8 nonBNT={np.sqrt(nn[1,1]):.4f} BNT={np.sqrt(nb[1,1]):.4f}")

if rows:
    A = np.array([r[0] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    npe_r = np.array([r[1] for r in rows]); npe_e = np.array([r[2] for r in rows])
    fis_r = np.array([r[3] for r in rows])
    ax[0].errorbar(A, npe_r, yerr=npe_e, fmt="o-", color="#c0392b", lw=2, ms=7,
                   label="NPE-score (calibrated)", capsize=3)
    ax[0].plot(A, fis_r, "s--", color="#7f7f7f", lw=1.8, ms=6, label="Fisher (hybrid, idealized)")
    ax[0].axhline(1.0, color="k", lw=0.6)
    ax[0].set_xscale("log"); ax[0].set_xlabel("survey area [deg$^2$]")
    ax[0].set_ylabel("BNT-580 / non-BNT-460  FoM3 ratio")
    ax[0].set_title("3-param BNT advantage vs area"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    sn = np.array([r[4] for r in rows]); sbv = np.array([r[5] for r in rows])
    ax[1].plot(A, sn, "o-", color="#7f7f7f", lw=2, label="non-BNT-460")
    ax[1].plot(A, sbv, "o-", color="#c0392b", lw=2, label="BNT-580")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("survey area [deg$^2$]")
    ax[1].set_ylabel(r"NPE $\sigma(S_8)$"); ax[1].set_title("Realized $\\sigma(S_8)$ vs area")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Score-compression NPE across six footprints (calibrated) vs Fisher", fontsize=11)
    fig.tight_layout()
    p = f"{OUT}/npe_fom_vs_area.png"
    fig.savefig(p, dpi=140); print("saved", p)
