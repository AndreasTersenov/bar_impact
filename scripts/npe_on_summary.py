#!/usr/bin/env python3
"""Stage 2 — jaxili NPE on the frozen VMIM summaries.

Loads a compressed.npz (from vmim_compress.py), trains jaxili NPE on (theta, y) for N seeds,
samples the posterior at the compressed fiducial y_fid, pools the seeds, saves the posterior, and
runs a bounded TARP coverage check (the recipe's mandatory gate: tightness != correctness).

Run with the jaxili interpreter.
"""
import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compressed", required=True)
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--tag", required=True, help="config tag for filenames")
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--tarp-points", type=int, default=200)
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()


def main():
    a = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    import jax
    import jax.numpy as jnp
    from jax import random
    from jaxili.inference import NPE
    try:
        from tarp import get_tarp_coverage
    except Exception:
        get_tarp_coverage = None

    z = np.load(a.compressed)
    theta = np.concatenate([z["theta_tr"], z["theta_va"]]).astype(np.float32)
    y = np.concatenate([z["y_tr"], z["y_va"]]).astype(np.float32)
    y_fid = z["y_fid"].astype(np.float32).reshape(1, -1)
    print(f"[npe] theta={theta.shape} y={y.shape} y_fid={y_fid.shape}", flush=True)

    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    pooled, per_seed_post = [], []
    for sd in seeds:
        inf = NPE()
        inf = inf.append_simulations(jnp.asarray(theta), jnp.asarray(y),
                                     key=random.PRNGKey(sd))
        metrics, _ = inf.train(checkpoint_path=os.path.join(a.out, f"ckpt_{a.tag}_s{sd}"),
                               num_epochs=a.epochs, training_batch_size=a.batch_size,
                               learning_rate=a.lr)
        tl, vl = float(metrics.get("train/loss", np.nan)), float(metrics.get("val/loss", np.nan))
        if not (np.isfinite(tl) and np.isfinite(vl)):
            print(f"[npe] seed {sd}: non-finite loss — skipping", flush=True); continue
        post = inf.build_posterior()
        k = random.PRNGKey(sd + 7)
        s = np.asarray(post.sample(x=jnp.asarray(y_fid[0]), num_samples=a.num_samples,
                                   key=k))
        pooled.append(s)
        per_seed_post.append((sd, post, inf))
        print(f"[npe] seed {sd}: val_loss={vl:.4f} S8={s[:,1].mean():.4f}±{s[:,1].std():.4f}",
              flush=True)

    if not pooled:
        print("[npe] no usable seeds — FAIL", flush=True); sys.exit(3)
    samples = np.concatenate(pooled)
    out_npy = os.path.join(a.out, f"posterior_summary_{a.tag}.npy")
    np.save(out_npy, samples)
    summ = {"tag": a.tag, "n_seeds": len(pooled), "n_samples": int(samples.shape[0]),
            "mean": samples.mean(0).tolist(), "std": samples.std(0).tolist()}

    # bounded TARP coverage on held-out points (best-effort)
    tarp = {"ran": False}
    if get_tarp_coverage is not None and per_seed_post:
        try:
            sd, post, inf = per_seed_post[0]
            ntest = min(a.tarp_points, theta.shape[0] // 5)
            ridx = np.random.default_rng(0).choice(theta.shape[0], size=ntest, replace=False)
            th_test, y_test = theta[ridx], y[ridx]
            draws = 100
            samp = np.stack([np.asarray(post.sample(x=jnp.asarray(y_test[i]), num_samples=draws,
                            key=random.PRNGKey(1000 + i))) for i in range(ntest)], axis=1)
            ecp, alpha = get_tarp_coverage(samp, th_test, references="random", metric="euclidean")
            bias = float(np.max(np.abs(ecp - alpha)))
            tarp = {"ran": True, "max_abs_dev": bias,
                    "verdict": "OK" if bias < 0.15 else ("OVERCONF" if (ecp - alpha).mean() < 0 else "UNDERCONF")}
            print(f"[npe] TARP max|ecp-alpha|={bias:.3f} -> {tarp['verdict']}", flush=True)
        except Exception as e:
            tarp = {"ran": False, "error": str(e)[:200]}
            print(f"[npe] TARP failed: {e}", flush=True)
    summ["tarp"] = tarp

    # SBC rank statistics (recipe gate): ranks of true theta within the posterior should be uniform
    # (rank-std ~0.289). >0.32 => over-confident (ranks pile at 0/1); <0.25 => under-confident.
    sbc = {"ran": False}
    if per_seed_post:
        try:
            sd, post, inf = per_seed_post[0]
            ntest = min(a.tarp_points, theta.shape[0] // 5)
            ridx = np.random.default_rng(1).choice(theta.shape[0], size=ntest, replace=False)
            ranks = []
            for i, j in enumerate(ridx):
                d = np.asarray(post.sample(x=jnp.asarray(y[j]), num_samples=200,
                                           key=random.PRNGKey(2000 + i)))
                ranks.append((d < theta[j]).mean(axis=0))   # per-param rank in [0,1]
            ranks = np.array(ranks)
            rstd = ranks.std(axis=0).tolist()
            worst = float(max(rstd))
            sbc = {"ran": True, "rank_std": rstd, "rank_mean": ranks.mean(0).tolist(),
                   "verdict": "OK" if worst < 0.32 and min(rstd) > 0.24 else
                              ("OVERCONF" if worst >= 0.32 else "UNDERCONF")}
            print(f"[npe] SBC rank-std={[f'{r:.3f}' for r in rstd]} (uniform~0.289) -> {sbc['verdict']}",
                  flush=True)
        except Exception as e:
            sbc = {"ran": False, "error": str(e)[:200]}
    summ["sbc"] = sbc
    with open(os.path.join(a.out, f"summary_{a.tag}.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    print(f"[npe] wrote {out_npy} (pooled {len(pooled)} seeds)")
    print("NPE OK")


if __name__ == "__main__":
    main()
