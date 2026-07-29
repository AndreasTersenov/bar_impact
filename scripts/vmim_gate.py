#!/usr/bin/env python3
"""P2 gate — the make-or-break checks for the VMIM v2 pipeline (BNT only; non-BNT is the control).

Consumes the Stage-2 outputs of two arms (bnt_full, nonbnt_full) and runs, in order:
  1. LOSSLESS IDENTITY (hard): bnt_full null posterior ~ nonbnt_full null posterior (same truth, width
     within scatter). The check the prior attempt FAILED.
  2. NULL ON TRUTH: each arm's null mean within ~0.5 sigma of (Om,S8,w0)=(0.26,0.84,-1.0).
  3. SIGMA vs FISHER FLOOR: realized sigma(Om,S8,w0) vs sqrt(diag(F^-1)) from score_cut_utils.build_score
     at the FULL cut (network-independent). Collapsed (<~0.7x) = over-confident; inflated (>~1.5x) = under.
  4. TARP-DRP + SBC: ECP-vs-alpha curve + rank histograms on held-out val cosmologies (plots rendered).

Params are [Om, S8, w0, H0, ns, Ob] (S8 = 0.84 at fiducial; h0 in the posterior is /100 but the first
three — the ones we judge — are physical and identical across spaces). Writes verdict.json +
GATE_REPORT.md + plots. Run with the jaxili interpreter.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

TRUTH3 = np.array([0.26, 0.84, -1.0])
NAMES3 = ["Om", "S8", "w0"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bnt-dir", required=True, help="Stage-2 out dir for bnt_full")
    p.add_argument("--nonbnt-dir", required=True, help="Stage-2 out dir for nonbnt_full")
    p.add_argument("--bnt-tag", default="bnt_full")
    p.add_argument("--nonbnt-tag", default="nonbnt_full")
    p.add_argument("--out", required=True)
    p.add_argument("--area", default="14000")
    return p.parse_args()


def fisher_floor(area, bnt):
    """sqrt(diag(F^-1)) for [Om,S8,w0,...] at the full cut — network-independent floor."""
    os.environ.setdefault("FISHER_AREA", str(area))
    import sys
    sys.path.insert(0, "scripts")
    import score_cut_utils as S
    d = S.build_score(S.FULL_CUTS, bnt=bnt, covk="hybrid")
    Finv = np.linalg.inv(d["F"])
    return np.sqrt(np.diag(Finv))[:3]


def load_arm(d, tag):
    null = np.load(Path(d) / f"null_pooled_{tag}.npy")
    summ = json.loads((Path(d) / f"summary_{tag}.json").read_text())
    tarp_s = np.load(Path(d) / f"tarp_samples_{tag}.npy")     # (n_draws, n_points, 6)
    tarp_t = np.load(Path(d) / f"tarp_theta_{tag}.npy")       # (n_points, 6)
    return {"null": null, "summary": summ, "tarp_s": tarp_s, "tarp_t": tarp_t}


def run_tarp(tarp_s, tarp_t):
    from tarp import get_tarp_coverage
    ecp, alpha = get_tarp_coverage(tarp_s, tarp_t, references="random", num_bootstrap=200, norm=True)
    ecp_m = ecp.mean(0) if ecp.ndim == 2 else ecp
    ecp_sd = ecp.std(0) if ecp.ndim == 2 else np.zeros_like(ecp)
    max_dev = float(np.max(np.abs(ecp_m - alpha)))
    net_bias = float(np.mean(ecp_m - alpha))
    return alpha, ecp_m, ecp_sd, max_dev, net_bias


def sbc_ranks(tarp_s, tarp_t):
    """Rank of truth within posterior draws, per param. Uniform => std ~0.289."""
    # tarp_s: (n_draws, n_points, 6) ; tarp_t: (n_points, 6)
    nd = tarp_s.shape[0]
    ranks = (tarp_s < tarp_t[None, :, :]).mean(0)            # (n_points, 6) in [0,1]
    return ranks, ranks.std(0)


def main():
    a = parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bnt = load_arm(a.bnt_dir, a.bnt_tag)
    non = load_arm(a.nonbnt_dir, a.nonbnt_tag)
    report = {"gates": {}}

    # ---- Gate 1: lossless identity (bnt null ~ nonbnt null) ----
    mb, sb = bnt["null"].mean(0)[:3], bnt["null"].std(0)[:3]
    mn, sn = non["null"].mean(0)[:3], non["null"].std(0)[:3]
    dmean_sig = np.abs(mb - mn) / (0.5 * (sb + sn))
    sratio = sb / sn
    g1_pass = bool(np.all(dmean_sig < 0.3) and np.all((sratio > 0.85) & (sratio < 1.18)))
    report["gates"]["1_identity"] = {
        "bnt_mean": mb.tolist(), "nonbnt_mean": mn.tolist(),
        "bnt_sigma": sb.tolist(), "nonbnt_sigma": sn.tolist(),
        "dmean_in_sigma": dmean_sig.tolist(), "sigma_ratio_bnt_over_nonbnt": sratio.tolist(),
        "PASS": g1_pass}

    # ---- Gate 2: null on truth ----
    g2 = {}
    for tag, m, s in [(a.bnt_tag, mb, sb), (a.nonbnt_tag, mn, sn)]:
        bias = np.abs(m - TRUTH3) / s
        g2[tag] = {"mean": m.tolist(), "bias_in_sigma": bias.tolist(), "PASS": bool(np.all(bias < 0.5))}
    g2_pass = all(v["PASS"] for v in g2.values())
    report["gates"]["2_on_truth"] = {**g2, "PASS": g2_pass}

    # ---- Gate 3: sigma vs Fisher floor ----
    floor_bnt = fisher_floor(a.area, bnt=True)
    floor_non = fisher_floor(a.area, bnt=False)
    g3 = {}
    for tag, s, fl in [(a.bnt_tag, sb, floor_bnt), (a.nonbnt_tag, sn, floor_non)]:
        ratio = s / fl
        g3[tag] = {"sigma": s.tolist(), "fisher_floor": fl.tolist(), "ratio": ratio.tolist(),
                   "PASS": bool(np.all((ratio > 0.7) & (ratio < 1.6)))}
    g3_pass = all(v["PASS"] for v in g3.values())
    report["gates"]["3_fisher_floor"] = {**g3, "PASS": g3_pass}

    # ---- Gate 4: TARP-DRP + SBC ----
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4.2))
    fig2, ax2 = plt.subplots(2, 3, figsize=(11, 6))
    g4 = {}
    for col, (tag, arm) in enumerate([(a.bnt_tag, bnt), (a.nonbnt_tag, non)]):
        alpha, ecp_m, ecp_sd, max_dev, net_bias = run_tarp(arm["tarp_s"], arm["tarp_t"])
        ranks, rstd = sbc_ranks(arm["tarp_s"], arm["tarp_t"])
        worst = float(np.max(rstd[:3]))                       # judge on the 3 params we report
        tarp_ok = max_dev <= 0.10
        sbc_ok = bool(np.all((rstd[:3] > 0.24) & (rstd[:3] < 0.34)))
        g4[tag] = {"tarp_max_dev": max_dev, "tarp_net_bias": net_bias, "sbc_rank_std": rstd.tolist(),
                   "tarp_ok": bool(tarp_ok), "sbc_ok": sbc_ok, "PASS": bool(tarp_ok and sbc_ok)}
        ax1[col].plot([0, 1], [0, 1], "k--", lw=1)
        ax1[col].fill_between(alpha, ecp_m - ecp_sd, ecp_m + ecp_sd, alpha=0.3)
        ax1[col].plot(alpha, ecp_m, lw=1.6)
        ax1[col].set_title(f"{tag}  max|ecp-α|={max_dev:.3f}")
        ax1[col].set_xlabel("credibility α"); ax1[col].set_ylabel("expected coverage ECP")
        for k in range(3):
            ax2[col, k].hist(ranks[:, k], bins=20, range=(0, 1), color="C%d" % col, alpha=0.8)
            ax2[col, k].axhline(ranks.shape[0] / 20, color="k", ls="--", lw=0.8)
            ax2[col, k].set_title(f"{tag} {NAMES3[k]} rank-std={rstd[k]:.3f}")
    g4_pass = all(v["PASS"] for v in g4.values())
    report["gates"]["4_tarp_sbc"] = {**g4, "PASS": g4_pass}
    fig1.tight_layout(); fig1.savefig(out / "tarp_ecp.png", dpi=130)
    fig2.tight_layout(); fig2.savefig(out / "sbc_hist.png", dpi=130)

    # ---- verdict ----
    primary = g1_pass and g2_pass            # the make-or-break pair
    report["PRIMARY_PASS"] = bool(primary)
    report["ALL_PASS"] = bool(primary and g3_pass and g4_pass)
    (out / "verdict.json").write_text(json.dumps(report, indent=2))

    # ---- human report ----
    L = ["# P2 GATE REPORT — VMIM v2 (BNT, 14000, full vector)\n",
         f"PRIMARY (identity + on-truth): {'PASS ✅' if primary else 'FAIL ❌'}",
         f"ALL (incl. floor + TARP/SBC): {'PASS ✅' if report['ALL_PASS'] else 'FAIL ❌'}\n",
         "## Gate 1 — lossless identity (bnt_full null ≈ nonbnt_full null)",
         f"  param:        {NAMES3}",
         f"  bnt   mean:   {np.round(mb,4).tolist()}  σ={np.round(sb,4).tolist()}",
         f"  nonbnt mean:  {np.round(mn,4).tolist()}  σ={np.round(sn,4).tolist()}",
         f"  |Δmean|/σ:    {np.round(dmean_sig,3).tolist()} (need <0.3)",
         f"  σ_bnt/σ_non:  {np.round(sratio,3).tolist()} (need 0.85–1.18)  -> {'PASS' if g1_pass else 'FAIL'}\n",
         "## Gate 2 — null on truth (0.26, 0.84, -1.0)"]
    for tag in (a.bnt_tag, a.nonbnt_tag):
        v = report["gates"]["2_on_truth"][tag]
        L.append(f"  {tag:12s} bias/σ = {np.round(v['bias_in_sigma'],3).tolist()} -> {'PASS' if v['PASS'] else 'FAIL'}")
    L.append("\n## Gate 3 — σ vs Fisher floor (σ/floor; 0.7–1.6 ok)")
    for tag in (a.bnt_tag, a.nonbnt_tag):
        v = report["gates"]["3_fisher_floor"][tag]
        L.append(f"  {tag:12s} σ={np.round(v['sigma'],4).tolist()} floor={np.round(v['fisher_floor'],4).tolist()} "
                 f"ratio={np.round(v['ratio'],2).tolist()} -> {'PASS' if v['PASS'] else 'FAIL'}")
    L.append("\n## Gate 4 — TARP + SBC")
    for tag in (a.bnt_tag, a.nonbnt_tag):
        v = report["gates"]["4_tarp_sbc"][tag]
        L.append(f"  {tag:12s} TARP max|ecp-α|={v['tarp_max_dev']:.3f} bias={v['tarp_net_bias']:+.3f} "
                 f"SBC rank-std(3)={np.round(v['sbc_rank_std'][:3],3).tolist()} -> {'PASS' if v['PASS'] else 'FAIL'}")
    L.append("\nPlots: tarp_ecp.png (ECP-vs-α), sbc_hist.png (rank histograms).")
    (out / "GATE_REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
