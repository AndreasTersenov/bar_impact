#!/usr/bin/env python3
"""Is the embedding's excess over the Gaussian Fisher real information, or over-confidence from
evaluating at an atypically clean "observation"?

THE ISSUE. The fiducial observation every FoM is measured at is `x_fid = perms_null.mean(0)` --
the MEAN of 200 independent fiducial realizations (score_compress_at_rebin.py:104). Every row the
flow trains on is a SINGLE realization. So the conditioning value at evaluation time carries
sqrt(200) ~ 14x less noise than anything in the training distribution: the NPE is being asked for
p(theta | y) at a y it never saw and that no real survey could produce.

An NPE returns a posterior width that is a learned function of y. Off-distribution -- at an
implausibly clean y -- there is no guarantee that width is calibrated, and the natural failure
direction is over-confidence. That would inflate every FoM measured this way. It is a candidate
explanation for the embedding sitting 1.25-1.36x ABOVE the Gaussian Fisher, which is otherwise
awkward: the Fisher is the Gaussian bound, and beating it demands genuine non-Gaussian information.

THE TEST. Take the ALREADY-TRAINED flows and re-evaluate them at single fiducial realizations --
in-distribution, exactly what one survey delivers -- instead of at the mean of 200. Nothing is
retrained; only the conditioning value changes. So this isolates the evaluation point from every
other difference, including training noise.

  If the excess is a fiducial-mean artifact : FoM at single realizations falls toward the Fisher.
  If the excess is real information         : FoM stays put; only its scatter grows.

Both arms are run, because the claim that this is common-mode (and therefore cancels in the BNT
ratio) is itself part of what needs checking.

TWO GATES, both must pass before any number here is believed:
  A. the recomputed ana_whiten preprocessing reproduces the stored y_fid in compressed.npz
  B. re-sampling at the stored y_fid reproduces the stored per-seed posterior null_s<sd>.npy

Gate B is what makes this a controlled experiment rather than a reimplementation with a different
answer. Run under jaxili.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

H0_IDX = 3


def fom3(s):
    """FoM_3 = 1/sqrt(det Cov(Om, S8, w0)) -- the convention used by every other figure here."""
    return float(1.0 / np.sqrt(np.linalg.det(np.cov(s[:, :3], rowvar=False))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/score_embed_anaw16",
                   help="run root holding <arm>_r20_c460/{comp,nde}")
    p.add_argument("--arms", default="bnt,nonbnt")
    p.add_argument("--cut", default="460")
    p.add_argument("--seeds", default="41,42,43,44,45")
    p.add_argument("--n-obs", type=int, default=20, help="single realizations to evaluate at")
    p.add_argument("--num-samples", type=int, default=0,
                   help="0 = infer from the stored posterior. It MUST match the original run or "
                        "gate B fails on shape: the sampler consumes the PRNG key per draw, so a "
                        "different count is a different sequence, not a longer one.")
    p.add_argument("--nde-layers", type=int, default=4)
    p.add_argument("--nde-hidden", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--embed-hidden", default="256,256")
    p.add_argument("--out", default="outputs/diagnostics/fiducial_mean_overconfidence.json")
    p.add_argument("--gpu", default="0")
    p.add_argument("--mem-fraction", default="0.3")
    return p.parse_args()


def main():
    a = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", a.mem_fraction)

    import jax
    import jax.numpy as jnp
    import nde_realnvp as N
    import score_cut_utils as S
    import fisher_local_jacobian as L
    import fisher_hybrid_cov as H

    seeds = [int(s) for s in a.seeds.split(",")]
    eh = tuple(int(v) for v in a.embed_hidden.split(",") if v.strip())
    results = {"root": a.root, "n_obs": a.n_obs, "seeds": seeds, "arms": {}}

    for arm in a.arms.split(","):
        bnt = arm == "bnt"
        cuts = [int(a.cut), 1024, 1024, 1024] if bnt else [int(a.cut)] * 4
        rd = os.path.join(a.root, f"{arm}_r20_c460")
        comp = np.load(os.path.join(rd, "comp", "compressed.npz"))
        qa = json.load(open(os.path.join(rd, "comp", "compress_qa.json")))
        dim = int(qa["summary_dim"])
        fisher = float(qa["fisher_fom3"])
        print(f"\n{'='*78}\n=== {arm}  cuts={cuts}  dim={dim}  Fisher FoM3={fisher:.4e}\n{'='*78}",
              flush=True)

        # ---- rebuild the ana_whiten preprocessing EXACTLY as score_compress_at_rebin.py did ----
        ga, gc, nell = L.load_set("new_grid", "nobaryons", bnt)
        x = L.datavector(ga, gc, nell, cuts)
        theta = L.PARAMS.copy()
        ok = np.isfinite(x).all(1) & np.isfinite(theta).all(1)
        x = x[ok]                                   # same row drop as the compress step
        fa, fc, _ = L.load_set("fiducial", "nobaryons", bnt)
        perms = L.datavector(fa, fc, nell, cuts)    # (200, nfeat) -- the single realizations
        x_fid = perms.mean(0)

        d = S.build_score(cuts, bnt=bnt, covk=qa["covk"])
        ev, V = np.linalg.eigh(d["C"])
        evf = np.maximum(ev, 1e-4 * ev.max())
        Wh = (V / np.sqrt(evf)) @ V.T
        mu = x.mean(0)
        sw = ((x - mu) @ Wh).std(0)
        sw[sw < 1e-12] = 1.0
        CLIP = float(os.environ.get("ANAW_CLIP", "5"))

        def ap(A):
            Z = ((np.atleast_2d(A) - mu) @ Wh) / sw
            return np.clip(Z, -CLIP, CLIP) if CLIP > 0 else Z

        # ---- GATE A: preprocessing reproduces the stored summary ----
        y_fid_ref = comp["y_fid"].astype(np.float32)
        y_fid_new = ap(x_fid)[0].astype(np.float32)
        gA = float(np.max(np.abs(y_fid_new - y_fid_ref)))
        print(f"[gate A] recomputed y_fid vs stored: max|dy| = {gA:.3e}", flush=True)
        if not gA < 1e-5:
            print("[gate A] FAIL - preprocessing does not match the run; numbers would be "
                  "meaningless", flush=True)
            sys.exit(1)
        print("[gate A] PASS", flush=True)

        # ---- load the trained flows ----
        nfp, nfs = N.build_flow_embedded(6, n_layers=a.nde_layers, hidden=a.nde_hidden,
                                         embed_dim=a.embed_dim, embed_hidden=eh)

        def draw(params, y_obs, key, m):
            yb = jnp.broadcast_to(jnp.asarray(y_obs).reshape(1, dim), (m, dim))
            s = np.asarray(nfs.apply(params, key, yb, m))
            return s[np.all(np.isfinite(s), 1)]

        flows = {}
        for sd in seeds:
            with open(os.path.join(rd, "nde", f"ckpt_s{sd}", "params_flow_best.pkl"), "rb") as fh:
                flows[sd] = pickle.load(fh)

        # The draw count is part of the experiment, not a free knob -- recover it from the run
        # itself rather than trusting a default that happened to differ (the original used 3000).
        n_samp = a.num_samples
        if n_samp <= 0:
            ref0 = os.path.join(rd, "nde", f"null_s{seeds[0]}_{arm}.npy")
            n_samp = int(np.load(ref0).shape[0])
            print(f"[setup] num_samples inferred from {os.path.basename(ref0)}: {n_samp}", flush=True)

        # ---- GATE B: re-sampling at the stored y_fid reproduces the stored posterior ----
        gB = {}
        for sd in seeds:
            ref_p = os.path.join(rd, "nde", f"null_s{sd}_{arm}.npy")
            if not os.path.exists(ref_p):
                continue
            ref = np.load(ref_p)
            new = draw(flows[sd], y_fid_ref, jax.random.PRNGKey(sd + 7), n_samp)
            if new.shape != ref.shape:
                print(f"[gate B] seed {sd}: shape {new.shape} vs {ref.shape}", flush=True)
                gB[sd] = float("inf")
            else:
                gB[sd] = float(np.max(np.abs(new - ref)))
            print(f"[gate B] seed {sd}: max|ds| = {gB[sd]:.3e}   "
                  f"FoM3 new={fom3(new):.4e} ref={fom3(ref):.4e}", flush=True)
        worst = max(gB.values()) if gB else float("inf")
        if not worst < 1e-4:
            print(f"[gate B] FAIL (worst {worst:.3e}) - cannot reproduce the run's own posterior",
                  flush=True)
            sys.exit(1)
        print("[gate B] PASS - this is the same flow, same sampler, same draws", flush=True)

        # ---- the measurement: mean-of-200 observation vs single realizations ----
        fom_mean = [fom3(draw(flows[sd], y_fid_ref, jax.random.PRNGKey(sd + 7), n_samp))
                    for sd in seeds]
        w0_mean = [float(draw(flows[sd], y_fid_ref, jax.random.PRNGKey(sd + 7),
                              n_samp)[:, 2].mean()) for sd in seeds]
        print(f"\n[mean-of-200] per-seed FoM3 = {[f'{v:.3e}' for v in fom_mean]}", flush=True)
        print(f"[mean-of-200] mean FoM3 = {np.mean(fom_mean):.4e}   "
              f"/Fisher = {np.mean(fom_mean)/fisher:.3f}", flush=True)

        per_obs = []
        rng = np.random.RandomState(0)
        idx = rng.choice(len(perms), size=min(a.n_obs, len(perms)), replace=False)
        for k in idx:
            y_k = ap(perms[k])[0].astype(np.float32)
            fs, w0s = [], []
            for sd in seeds:
                s = draw(flows[sd], y_k, jax.random.PRNGKey(sd + 7), n_samp)
                fs.append(fom3(s)); w0s.append(float(s[:, 2].mean()))
            per_obs.append({"perm": int(k), "fom3_per_seed": fs,
                            "fom3_mean": float(np.mean(fs)), "w0_mean": float(np.mean(w0s))})
            print(f"  perm {k:3d}: mean FoM3 = {np.mean(fs):.4e}  /Fisher = "
                  f"{np.mean(fs)/fisher:.3f}   <w0> = {np.mean(w0s):+.4f}", flush=True)

        fo = np.array([r["fom3_mean"] for r in per_obs])
        w0 = np.array([r["w0_mean"] for r in per_obs])
        res = {
            "fisher_fom3": fisher,
            "fom3_at_mean_of_200": float(np.mean(fom_mean)),
            "ratio_at_mean_of_200": float(np.mean(fom_mean) / fisher),
            "w0_at_mean_of_200": float(np.mean(w0_mean)),
            "fom3_single_median": float(np.median(fo)),
            "fom3_single_p16": float(np.percentile(fo, 16)),
            "fom3_single_p84": float(np.percentile(fo, 84)),
            "ratio_single_median": float(np.median(fo) / fisher),
            "w0_single_median": float(np.median(w0)),
            "w0_single_std": float(np.std(w0)),
            "shrink_factor": float(np.mean(fom_mean) / np.median(fo)),
            "per_obs": per_obs,
            "gate_A_max_dy": gA, "gate_B_max_ds": gB,
        }
        results["arms"][arm] = res
        print(f"\n--- {arm} SUMMARY ---")
        print(f"  Fisher FoM3                : {fisher:.4e}")
        print(f"  at mean-of-200 (published) : {res['fom3_at_mean_of_200']:.4e}  "
              f"({res['ratio_at_mean_of_200']:.3f}x Fisher)")
        print(f"  at single realization      : {res['fom3_single_median']:.4e}  "
              f"[{res['fom3_single_p16']:.3e}, {res['fom3_single_p84']:.3e}]  "
              f"({res['ratio_single_median']:.3f}x Fisher)")
        print(f"  shrink factor              : {res['shrink_factor']:.3f}")
        print(f"  <w0>  mean-obs {res['w0_at_mean_of_200']:+.4f}  vs single "
              f"{res['w0_single_median']:+.4f} +/- {res['w0_single_std']:.4f}", flush=True)

    # ---- the ratio, which is what the paper actually quotes ----
    if "bnt" in results["arms"] and "nonbnt" in results["arms"]:
        b, nb = results["arms"]["bnt"], results["arms"]["nonbnt"]
        results["ratio_bnt_over_nonbnt"] = {
            "at_mean_of_200": b["fom3_at_mean_of_200"] / nb["fom3_at_mean_of_200"],
            "at_single_realization": b["fom3_single_median"] / nb["fom3_single_median"],
        }
        print(f"\n{'='*78}\n=== BNT / non-BNT FoM3 ratio -- the published number ===")
        print(f"  at mean-of-200 (published) : "
              f"{results['ratio_bnt_over_nonbnt']['at_mean_of_200']:.3f}")
        print(f"  at single realization      : "
              f"{results['ratio_bnt_over_nonbnt']['at_single_realization']:.3f}")
        print("  (if these agree, the effect is common-mode and the headline is untouched)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
