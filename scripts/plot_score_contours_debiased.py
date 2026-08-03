#!/usr/bin/env python3
"""Constraining-power plot: score-compressed null posteriors at a scale cut, BNT vs non-BNT.

Two ways to pair the cuts, and they answer different questions:

  * DE-BIASING pair (--bnt-cut 460 --nonbnt-cut 580): each analysis at the cut IT needs to be
    unbiased (<0.3 sigma). Answers "given each must be unbiased, who keeps more information?"
  * MATCHED pair (--bnt-cut 580 --nonbnt-cut 580): both at the same ell_max. Answers "at identical
    scales, what does cutting in the BNT basis buy?" BNT at 580 sits at 0.367 +/- 0.076 sigma, i.e.
    above the 0.3 nominal but tolerated on its error bar (mean - sigma = 0.291).

`--ref-cut` adds a third, dashed, unfilled contour for the same arm as --bnt-cut, which is how you
see what moving BNT's own cut actually buys.

Run under jaxili (getdist 1.6.1). NOT aname: its getdist 1.4.3 cannot draw filled contours under
matplotlib >= 3.8.
    /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python scripts/plot_score_contours_debiased.py

Every run writes <out>_values.csv and <out>_provenance.json beside the figure (repo rule: a figure
without its sidecars is unfinished).
"""
import argparse
import glob
import json
import os
import subprocess
import shlex
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
from getdist import MCSamples, plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
NAMES = ["Omega_m", "S8", "w0"]
LABELS = [r"\Omega_m", "S_8", "w_0"]
TRUTH = [0.26, 0.84, -1.0]
P3 = [0, 1, 2]
ARM = {"bnt": "bnt_ps_bin1_score_l37", "nonbnt": "ps_cutall_score_l37"}
# tension_3param_agg.csv, mean +/- std over 5 seeds, read at call time
TENSION_CSV = "{repo}/outputs/baryon_tension/{tag}/area14000/tables/tension_3param_agg.csv"


def fisher_fom3(cuts, bnt, order="order2", bw=0.75):
    """3-param Fisher-floor FoM = 1/sqrt(det(inv(F)[:3,:3])).

    STABILITY (re-measured 2026-07-31 after the covariance was restored): stable. The de-biasing
    ratio spans 1.319-1.455 across (order, bw), a 1.10x spread, and order2 alone spans 1.09x —
    within fisher_local_jacobian's plateau criterion. An earlier report of a 1.38->2.41 swing was
    made against the RAID0-destroyed analytic covariance and was an artifact of that corruption, not
    of the Jacobian. Quote order2 and state the ~10% systematic; see
    outputs/diagnostics/fisher_floor_stability.csv.
    """
    os.environ.setdefault("FISHER_AREA", "14000")
    os.environ.setdefault("FISHER_REBIN", "20")
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import score_cut_utils as S
    return S.build_score(cuts, bnt=bnt, order=order, bw=bw)["fom3"]


def load_null(arm, cut):
    """Per-seed null posteriors, skipping seeds destroyed by the disk failure.

    Guards for BOTH damage modes seen in this repo: a mangled .npy header raises ValueError (numpy
    reads it as a pickle stream — never 'fix' that with allow_pickle=True), and a file whose
    interior was zeroed loads fine but returns a slab of zeros.
    """
    d = f"{REPO}/outputs/baryon_tension/{ARM[arm]}/area14000/posteriors/cut{cut}"
    good, runs, skipped = [], [], []
    for f in sorted(glob.glob(f"{d}/null_run*.npy")):
        run = os.path.basename(f).replace("null_run", "").replace(".npy", "")
        try:
            a = np.load(f)
        except Exception as e:
            skipped.append((run, type(e).__name__)); continue
        if (a == 0).mean() > 0.5 or not np.isfinite(a).all():
            skipped.append((run, "zeroed/nonfinite")); continue
        good.append(a[:, :3]); runs.append(run)
    if not good:
        raise SystemExit(f"[fatal] no healthy null posteriors for {arm} @ cut{cut} ({d})")
    return good, runs, skipped


def select(per_seed, runs, mode):
    """Return (samples_to_draw, label_suffix, diagnostics) for the chosen seed convention.

    pooled = every surviving seed concatenated. single = the representative seed chosen by
    tension.seeds.representative_seed (median-referenced, centre AND width), because a survey
    trains one density estimator and quotes ITS posterior.
    """
    if mode == "pooled":
        return np.concatenate(per_seed), f"{len(per_seed)} seeds pooled", {"seed_mode": "pooled"}
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from tension.seeds import representative_seed
    i, run, diag = representative_seed(per_seed, runs)
    diag["seed_mode"] = "single"
    return per_seed[i], f"seed {run}", diag


def fom3_of(samples):
    """3-param FoM = 1/sqrt(det Cov) of exactly the samples being drawn."""
    cov = np.cov(samples, rowvar=False)
    return 1.0 / np.sqrt(np.linalg.det(cov)), np.sqrt(np.diag(cov))


def fom3_seed_avg(per_seed):
    """FoM from the seed-AVERAGED covariance — constraining power with between-seed mean scatter
    removed. Kept alongside the drawn value so the two conventions stay visible and comparable."""
    cov = np.mean([np.cov(s, rowvar=False) for s in per_seed], axis=0)
    return 1.0 / np.sqrt(np.linalg.det(cov))


def tension_at(arm, cut):
    """(mean, std) 3-param Q_DM tension at this cut, from the intact agg table."""
    p = TENSION_CSV.format(repo=REPO, tag=ARM[arm])
    try:
        rows = [l.strip().split(",") for l in open(p).read().splitlines()[1:] if l.strip()]
    except OSError:
        return None, None
    for r in rows:
        if len(r) >= 4 and int(r[1]) == cut:
            return float(r[2]), float(r[3])
    return None, None


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bnt-cut", type=int, default=460)
    ap.add_argument("--nonbnt-cut", type=int, default=580)
    ap.add_argument("--ref-cut", type=int, default=None,
                    help="optional dashed BNT reference contour at another cut")
    ap.add_argument("--no-fisher", action="store_true", help="skip the (slow, unstable) Fisher floor")
    ap.add_argument("--seed-mode", choices=["pooled", "single"], default="pooled",
                    help="pooled = all surviving seeds concatenated; single = representative seed")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    matched = args.bnt_cut == args.nonbnt_cut
    sfx = "_pooled" if args.seed_mode == "pooled" else "_single_seed"
    out = args.out or (f"{REPO}/plots/score_contours_"
                       f"{'matched' if matched else 'debiased'}_{args.nonbnt_cut}_14000{sfx}")

    bnt_seeds, bnt_runs, sk_b = load_null("bnt", args.bnt_cut)
    non_seeds, non_runs, sk_n = load_null("nonbnt", args.nonbnt_cut)
    for arm, cut, sk in [("BNT", args.bnt_cut, sk_b), ("nonBNT", args.nonbnt_cut, sk_n)]:
        if sk:
            print(f"[warn] {arm}@{cut}: skipped {len(sk)} damaged seed(s): "
                  + ", ".join(f"run{r}({w})" for r, w in sk))

    bnt, tag_b, diag_b = select(bnt_seeds, bnt_runs, args.seed_mode)
    non, tag_n, diag_n = select(non_seeds, non_runs, args.seed_mode)
    fb, sb = fom3_of(bnt)
    fn, sn = fom3_of(non)
    fb_avg, fn_avg = fom3_seed_avg(bnt_seeds), fom3_seed_avg(non_seeds)
    tb, tbs = tension_at("bnt", args.bnt_cut)
    tn, tns = tension_at("nonbnt", args.nonbnt_cut)
    print(f"[seed-mode={args.seed_mode}] BNT->{tag_b}  nonBNT->{tag_n}")
    print(f"BNT bin-1  @ {args.bnt_cut} (bins 2-4 full) [{len(bnt_seeds)} healthy]: "
          f"sigma={np.round(sb,4)} FoM3(drawn)={fb:.4e} tension={tb}+/-{tbs}")
    print(f"nonBNT cut-all @ {args.nonbnt_cut}        [{len(non_seeds)} healthy]: "
          f"sigma={np.round(sn,4)} FoM3(drawn)={fn:.4e} tension={tn}+/-{tns}")
    print(f"  realized 3-param FoM ratio BNT/non = {fb/fn:.3f}x "
          f"(seed-averaged-cov convention: {fb_avg/fn_avg:.3f}x)")

    ffb = ffn = None
    if not args.no_fisher:
        ffb = fisher_fom3([args.bnt_cut, 1024, 1024, 1024], True)
        ffn = fisher_fom3([args.nonbnt_cut] * 4, False)
        print(f"  Fisher-floor ratio (order2, bw=0.75; UNSTABLE across settings) = {ffb/ffn:.3f}x")

    ref = ref_seeds = None
    fr = diag_r = None
    if args.ref_cut is not None:
        ref_seeds, ref_runs, sk_r = load_null("bnt", args.ref_cut)
        if sk_r:
            print(f"[warn] BNT@{args.ref_cut} (ref): skipped {len(sk_r)} damaged seed(s)")
        ref, tag_r, diag_r = select(ref_seeds, ref_runs, args.seed_mode)
        fr, sr = fom3_of(ref)
        print(f"  ref BNT@{args.ref_cut} ({tag_r}): FoM3={fr:.4e}  "
              f"-> BNT@{args.bnt_cut}/BNT@{args.ref_cut} = {fb/fr:.3f}x")

    # ---- plot ----
    lb = rf"BNT bin-1 cut ($\ell_{{\max}}={args.bnt_cut}$, bins 2–4 full)"
    ln = rf"non-BNT cut-all ($\ell_{{\max}}={args.nonbnt_cut}$)"
    s_bnt = MCSamples(samples=bnt, names=NAMES, labels=LABELS, label=lb)
    s_non = MCSamples(samples=non, names=NAMES, labels=LABELS, label=ln)
    sets, cols, labs, filled = [s_non, s_bnt], ["0.5", "C0"], [ln, lb], [True, True]
    if ref is not None:
        lr = rf"BNT bin-1 ($\ell_{{\max}}={args.ref_cut}$), reference"
        sets.append(MCSamples(samples=ref, names=NAMES, labels=LABELS, label=lr))
        cols.append("C3"); labs.append(lr); filled.append(False)

    import paper_contour_style as PCS
    g = plots.get_subplot_plotter(width_inch=7.5)
    _palette = PCS.apply(g)
    g.triangle_plot(sets, filled=filled, contour_colors=cols,
                    legend_labels=labs, legend_loc="upper right")
    for i in range(3):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            ax.axvline(TRUTH[j], color="k", ls=":", lw=1, alpha=0.7)
            if i != j:
                ax.axhline(TRUTH[i], color="k", ls=":", lw=1, alpha=0.7)

    title = args.title or (
        rf"Null contours at a MATCHED scale cut ($\ell_{{\max}}={args.nonbnt_cut}$) — 14000 deg$^2$"
        if matched else "Null contours at the de-biasing scale cut — 14000 deg²")
    g.fig.suptitle(title, fontsize=14, y=1.02)

    box = ["BNT 3-param FoM advantage:", rf"$\mathbf{{{fb/fn:.2f}\times}}$ (realized, calibrated)"]
    if ffb is not None:
        box.append(rf"${ffb/ffn:.2f}\times$ (Fisher floor, indicative)")
    if fr is not None:
        box.append(rf"BNT {args.ref_cut}$\to${args.bnt_cut}: ${fb/fr:.2f}\times$")
    g.fig.text(0.62, 0.78, "\n".join(box), ha="center", va="center", fontsize=11,
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    tline = ""
    if tb is not None and tn is not None:
        tline = (rf"baryon tension: BNT {tb:.2f}$\pm${tbs:.2f}$\sigma$, "
                 rf"non-BNT {tn:.2f}$\pm${tns:.2f}$\sigma$ | ")
    seedline = (f"{len(bnt_seeds)}/{len(non_seeds)} seeds pooled" if args.seed_mode == "pooled"
                else f"single representative seed (BNT {tag_b.split()[-1]}, "
                     f"non-BNT {tag_n.split()[-1]})")
    g.fig.text(0.5, -0.01, tline + f"score-compressed calibrated nulls (nobaryons), {seedline}"
               " | dotted = truth", ha="center", fontsize=9, color="0.4")
    for ext in ("png", "pdf"):
        g.export(f"{out}.{ext}")

    # ---- sidecars ----
    rows = [("config", "ell_max", "seed_mode", "seed_label", "n_seeds",
             "sigma_Om", "sigma_S8", "sigma_w0", "fom3_drawn", "fom3_seed_avg_cov",
             "tension_mean", "tension_std", "n_features")]
    import score_cut_utils as S  # already imported above if fisher ran; cheap either way
    rows.append(("BNT_bin1", args.bnt_cut, args.seed_mode, tag_b, len(bnt_seeds),
                 *[f"{v:.6f}" for v in sb], f"{fb:.6e}", f"{fb_avg:.6e}",
                 tb, tbs, S.keep_indices([args.bnt_cut, 1024, 1024, 1024]).size))
    rows.append(("nonBNT_cutall", args.nonbnt_cut, args.seed_mode, tag_n, len(non_seeds),
                 *[f"{v:.6f}" for v in sn], f"{fn:.6e}", f"{fn_avg:.6e}",
                 tn, tns, S.keep_indices([args.nonbnt_cut] * 4).size))
    if ref is not None:
        tr, trs = tension_at("bnt", args.ref_cut)
        rows.append(("BNT_bin1_ref", args.ref_cut, args.seed_mode, tag_r, len(ref_seeds),
                     *[f"{v:.6f}" for v in sr], f"{fr:.6e}", f"{fom3_seed_avg(ref_seeds):.6e}",
                     tr, trs, S.keep_indices([args.ref_cut, 1024, 1024, 1024]).size))
    with open(f"{out}_values.csv", "w") as fh:
        for r in rows:
            fh.write(",".join("" if v is None else str(v) for v in r) + "\n")

    import getdist as _gd
    prov = {
        "figure": os.path.basename(out),
        "generator": "scripts/plot_score_contours_debiased.py",
        "command": shlex.join(sys.argv),
        "git_commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "getdist": _gd.__version__, "matplotlib": matplotlib.__version__},
        "mplstyle": "scripts/paper_contour_style.py (getdist settings; no rcParams sheet)",
        **PCS.provenance(_palette),
        "area_deg2": 14000,
        "statistic": "tomographic auto+cross angular power spectra",
        "compression": "score / MOPED, quasi-MLE form theta_hat = FID + (x-x_fid) @ C^-1 J F^-1",
        "covariance": "hybrid = analytic Gaussian + top-3 SSC/cNG eigenmodes",
        "jacobian": "local order-2 at the fiducial, bandwidth 0.75",
        "rebin": 20,
        "lmin": 37,
        "cuts": {"BNT_bin1": [args.bnt_cut, 1024, 1024, 1024],
                 "nonBNT_cutall": [args.nonbnt_cut] * 4,
                 **({"BNT_bin1_ref": [args.ref_cut, 1024, 1024, 1024]} if ref is not None else {})},
        "scales_included": (f"BNT: bin-1 to ell<={args.bnt_cut}, bins 2-4 to ell<=1024; "
                            f"non-BNT: all bins to ell<={args.nonbnt_cut}"),
        "cut_pairing": "matched" if matched else "de-biasing",
        "n_features": {"BNT_bin1": int(S.keep_indices([args.bnt_cut, 1024, 1024, 1024]).size),
                       "nonBNT_cutall": int(S.keep_indices([args.nonbnt_cut] * 4).size),
                       "full_vector": 120},
        "seed_mode": args.seed_mode,
        "posterior": ("all surviving NDE seeds concatenated" if args.seed_mode == "pooled" else
                      "the single representative NDE seed per arm (tension.seeds)"),
        "seed_selection": {"BNT": diag_b, "nonBNT": diag_n,
                           **({"BNT_ref": diag_r} if diag_r is not None else {})},
        "seeds_healthy": {"BNT": len(bnt_seeds), "nonBNT": len(non_seeds)},
        "seeds_skipped_disk_damage": {"BNT": sk_b, "nonBNT": sk_n},
        "fom3_realized": {
            "convention": "fom3_drawn is 1/sqrt(det Cov) of exactly the samples plotted; "
                          "fom3_seed_avg_cov removes between-seed mean scatter. They differ by "
                          "0.7-1.8% here, so conclusions are insensitive to the choice.",
            "BNT": fb, "nonBNT": fn, "ratio": fb / fn,
            "BNT_seed_avg_cov": fb_avg, "nonBNT_seed_avg_cov": fn_avg,
            "ratio_seed_avg_cov": fb_avg / fn_avg},
        "fom3_fisher_floor": ({"BNT": ffb, "nonBNT": ffn, "ratio": ffb / ffn,
                               "settings": "order2, bw=0.75; stable to ~10% across (order, bw) "
                                           "once the analytic covariance is the restored one — see "
                                           "outputs/diagnostics/fisher_floor_stability.csv"}
                              if ffb is not None else "not computed for this figure (--no-fisher)"),
        "tension_3param_QDM": {"BNT": [tb, tbs], "nonBNT": [tn, tns],
                               "source": "tension_3param_agg.csv (5 seeds, TARP/SBC OK per cut)"},
        "caveats": [
            "The analytic covariance behind the MOPED weights comes from the INTACT rebinned cache "
            "cov_rebinned_full_14000.npz, not from gaussian_cov_native_14000.npy: every native "
            "covariance and NaMaster workspace in cache_gaussian_cov/ was destroyed by the RAID0 "
            "failure (3 MiB stripe signature, ~20% zeroed). The substitution is exact at rebin=20 "
            "because a cut keeps whole leading bands and BNT commutes with the ell-rebin.",
            f"Scale cuts are quantised to 80 in ell by the rebin=20 floor division, so "
            f"ell_max={args.nonbnt_cut} selects the same columns as its degenerate partner "
            f"({args.nonbnt_cut - 40} or {args.nonbnt_cut + 40}); the effective ell_max of the "
            "retained vector is lower than the label suggests. Quote the lower member of a pair.",
            "This is a CONSTRAINING-POWER comparison, not baryon mitigation. BNT bin-1 crosses "
            "0.3 sigma at a LOWER ell_max (460) than non-BNT cut-all (620), so cutting only bin 1 "
            "does not control baryons better; its advantage is retaining more of the vector at "
            "equal unbiasedness.",
            "fom3_drawn describes exactly the samples plotted; fom3_seed_avg_cov removes "
            "between-seed mean scatter. They differ by 0.7-1.8% here.",
        ] + ([f"BNT@{args.bnt_cut} runs on {len(bnt_seeds)} of 5 seeds; the rest were destroyed by "
              "the disk failure and are listed in seeds_skipped_disk_damage."]
             if len(bnt_seeds) < 5 else [])
          + ([f"non-BNT@{args.nonbnt_cut} runs on {len(non_seeds)} of 5 seeds; see "
              "seeds_skipped_disk_damage."] if len(non_seeds) < 5 else []),
    }
    with open(f"{out}_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"wrote {out}.png / .pdf / _values.csv / _provenance.json")


if __name__ == "__main__":
    main()
