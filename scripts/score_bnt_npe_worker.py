#!/usr/bin/env python3
"""GPU worker for the score-BNT tension sweep: trains the NPE on pre-built score summaries and
samples the null AND biased posteriors. One long-lived process per GPU slot loops over a partition
of (cut, seed) jobs, so the (slow) jaxili/jax import is paid once per slot, not once per job.

Per job: load summaries/cut<c>.npz (theta, That[16965x6], t_null[6], t_biased[6]); train NPE on
(theta, That) for one seed (NaN-retry with a seed bump); sample the SAME trained model at t_null ->
null posterior and at t_biased -> biased posterior; save both as (n_samples, 6) arrays. On the
designated TARP seed per cut, also run TARP + SBC and write a verdict json (the calibration gate).

No seed pooling (the sweep needs per-seed tension for the error bars). Run with jaxili python.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jobs", required=True, help="comma list of cut:seed, e.g. 460:41,700:42")
    p.add_argument("--tarp-seeds", default="", help="comma list of cut:seed to also TARP/SBC")
    p.add_argument("--gpu", default="0")
    p.add_argument("--mem-fraction", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--tarp-points", type=int, default=200)
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(a.mem_fraction)
    import jax
    import jax.numpy as jnp
    from jax import random
    from jaxili.inference import NPE
    try:
        from tarp import get_tarp_coverage
    except Exception:
        get_tarp_coverage = None

    jobs = [tuple(int(v) for v in j.split(":")) for j in a.jobs.split(",") if j]
    tarp_jobs = {tuple(int(v) for v in j.split(":")) for j in a.tarp_seeds.split(",") if j}
    print(f"[worker gpu{a.gpu}] {len(jobs)} jobs; jax sees {jax.devices()}", flush=True)

    def train_one(theta, y, seed, ckpt):
        """Train NPE; on non-finite loss, bump the seed and retry. Returns (post, inf, used_seed, vl)."""
        for att in range(a.max_retries + 1):
            sd = seed + 1000 * att
            shutil.rmtree(ckpt, ignore_errors=True)               # fresh per attempt (orbax appends)
            inf = NPE().append_simulations(jnp.asarray(theta), jnp.asarray(y), key=random.PRNGKey(sd))
            metrics, _ = inf.train(checkpoint_path=ckpt,
                                   num_epochs=a.epochs, training_batch_size=a.batch_size,
                                   learning_rate=a.lr)
            vl = float(metrics.get("val/loss", np.nan))
            if np.isfinite(vl):
                return inf.build_posterior(), inf, sd, vl
            print(f"[worker gpu{a.gpu}] non-finite loss seed {sd} -> retry", flush=True)
        return None

    for (cut, seed) in jobs:
        cdir = os.path.join(a.out_dir, f"cut{cut}")
        os.makedirs(cdir, exist_ok=True)
        f_null = os.path.join(cdir, f"null_run{seed}.npy")
        f_bias = os.path.join(cdir, f"biased_run{seed}.npy")
        if os.path.exists(f_null) and os.path.exists(f_bias):
            print(f"[worker gpu{a.gpu}] cut{cut} seed{seed} exists -> skip", flush=True)
            continue
        z = np.load(os.path.join(a.summaries_dir, f"cut{cut}.npz"))
        theta = z["theta"].astype(np.float32)
        y = z["That"].astype(np.float32)
        t_null = z["t_null"].astype(np.float32).reshape(-1)
        t_bias = z["t_biased"].astype(np.float32).reshape(-1)

        ckpt = os.path.abspath(os.path.join(a.out_dir, f"_ckpt_cut{cut}_s{seed}_gpu{a.gpu}"))
        res = train_one(theta, y, seed, ckpt)
        if res is None:
            print(f"[worker gpu{a.gpu}] cut{cut} seed{seed} FAILED (NaN exhausted)", flush=True)
            shutil.rmtree(ckpt, ignore_errors=True)
            continue
        post, inf, used_sd, vl = res
        s_null = np.asarray(post.sample(x=jnp.asarray(t_null), num_samples=a.num_samples,
                                        key=random.PRNGKey(used_sd + 7)))
        s_bias = np.asarray(post.sample(x=jnp.asarray(t_bias), num_samples=a.num_samples,
                                        key=random.PRNGKey(used_sd + 9)))
        np.save(f_null, s_null)
        np.save(f_bias, s_bias)
        print(f"[worker gpu{a.gpu}] cut{cut} seed{seed} OK val={vl:.3f} "
              f"S8_null={s_null[:,1].mean():.4f}±{s_null[:,1].std():.4f} "
              f"S8_bias={s_bias[:,1].mean():.4f}", flush=True)

        if (cut, seed) in tarp_jobs:
            cal = {"cut": cut, "seed": int(used_sd)}
            try:
                ntest = min(a.tarp_points, theta.shape[0] // 5)
                ridx = np.random.default_rng(0).choice(theta.shape[0], size=ntest, replace=False)
                th_t, y_t = theta[ridx], y[ridx]
                if get_tarp_coverage is not None:
                    samp = np.stack([np.asarray(post.sample(x=jnp.asarray(y_t[i]), num_samples=100,
                                    key=random.PRNGKey(1000 + i))) for i in range(ntest)], axis=1)
                    ecp, alpha = get_tarp_coverage(samp, th_t, references="random", metric="euclidean")
                    bias = float(np.max(np.abs(ecp - alpha)))
                    cal["tarp_max_abs_dev"] = bias
                    cal["tarp_verdict"] = ("OK" if bias < 0.15 else
                                           ("OVERCONF" if (ecp - alpha).mean() < 0 else "UNDERCONF"))
                ranks = []
                for i, jdx in enumerate(ridx):
                    d = np.asarray(post.sample(x=jnp.asarray(y[jdx]), num_samples=200,
                                               key=random.PRNGKey(2000 + i)))
                    ranks.append((d < theta[jdx]).mean(axis=0))
                rstd = np.array(ranks).std(axis=0).tolist()
                worst, best = float(max(rstd)), float(min(rstd))
                cal["sbc_rank_std"] = rstd
                cal["sbc_verdict"] = ("OK" if worst < 0.32 and best > 0.24 else
                                      ("OVERCONF" if worst >= 0.32 else "UNDERCONF"))
                print(f"[worker gpu{a.gpu}] cut{cut} TARP {cal.get('tarp_verdict')} "
                      f"({cal.get('tarp_max_abs_dev', float('nan')):.3f}) "
                      f"SBC {cal['sbc_verdict']} rstd={[f'{r:.3f}' for r in rstd]}", flush=True)
            except Exception as e:
                cal["error"] = str(e)[:200]
                print(f"[worker gpu{a.gpu}] cut{cut} TARP/SBC failed: {e}", flush=True)
            with open(os.path.join(cdir, f"calibration_run{seed}.json"), "w") as fh:
                json.dump(cal, fh, indent=2)
        shutil.rmtree(ckpt, ignore_errors=True)                   # checkpoint is regenerable; keep disk clean

    print(f"[worker gpu{a.gpu}] DONE", flush=True)


if __name__ == "__main__":
    main()
