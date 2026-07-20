#!/usr/bin/env python3
"""Stage 2 — sbi_lens RealNVP NDE on the frozen VMIM summaries (scripts/nde_realnvp.py).

Loads a Stage-1 compressed.npz (theta_tr,y_tr,theta_va,y_va,y_fid; theta already H0/100 -> raw, O(1)),
trains the ported sbi_lens ConditionalRealNVP for N seeds on (theta_tr, y_tr), and produces:
  * null posterior at y_fid, pooled over seeds  -> null_pooled.npy + per_seed
  * a TARP/SBC bundle on the HELD-OUT val cosmologies (y_va,theta_va, unseen by compressor AND NDE):
    tarp_samples.npy (n_draws, n_points, 6), tarp_theta.npy (n_points, 6)
  * summary.json (null mean/std, n_seeds)

Raw theta throughout (no z-score) — the /100 at Stage-1 load already put it in the N(0.5,0.05) base
regime, matching the cnn_sbi reference. Run with the jaxili interpreter; set --gpu.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compressed", required=True, help="dir with compressed.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--nde-layers", type=int, default=4)
    p.add_argument("--nde-hidden", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--lr-init", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=20000, help="null draws per seed")
    p.add_argument("--tarp-points", type=int, default=400, help="held-out val points for TARP/SBC")
    p.add_argument("--tarp-draws", type=int, default=200, help="posterior draws per TARP point per seed")
    p.add_argument("--gpu", default="0")
    p.add_argument("--mem-fraction", default="0.3")
    p.add_argument("--save-per-seed", action="store_true",
                   help="also save per-seed null/biased arrays (for tension scatter, not just pooled)")
    p.add_argument("--no-tarp", action="store_true", help="skip the TARP/SBC bundle (faster sweeps)")
    return p.parse_args()


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = a.mem_fraction
    import sys
    sys.path.insert(0, "scripts")
    import jax
    import jax.numpy as jnp
    import nde_realnvp as N

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(Path(a.compressed) / "compressed.npz")
    theta_tr = z["theta_tr"].astype(np.float32)
    y_tr = z["y_tr"].astype(np.float32)
    theta_va = z["theta_va"].astype(np.float32)
    y_va = z["y_va"].astype(np.float32)
    y_fid = z["y_fid"].astype(np.float32)
    y_fid_biased = z["y_fid_biased"].astype(np.float32) if "y_fid_biased" in z.files else None
    dim = y_tr.shape[1]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    print(f"[nde] {a.tag} theta_tr={theta_tr.shape} y_tr={y_tr.shape} dim={dim} seeds={seeds}", flush=True)

    # internal 90/10 split for train_flow early stopping (deterministic seed 0), like the reference
    n = len(theta_tr)
    pidx = np.random.RandomState(0).permutation(n)
    nval = max(1, n // 10)
    vi, ti = pidx[:nval], pidx[nval:]
    dtr = {"theta": theta_tr[ti], "x": y_tr[ti]}
    dva = {"theta": theta_tr[vi], "x": y_tr[vi]}

    # ---- train N seeds; collect samplers ----
    samplers = []
    for sd in seeds:
        np.random.seed(sd)
        nfp, nfs = N.build_flow(6, n_layers=a.nde_layers, hidden=a.nde_hidden)
        params = N.train_flow(jax.random.PRNGKey(sd), nfp, dtr, dva, n_cosmo=6, summary_dim=dim,
                              total_steps=a.total_steps, batch_size=a.batch_size,
                              save_every=a.save_every, save_dir=out / f"ckpt_s{sd}",
                              lr_init=a.lr_init, end_lr=a.lr_end, grad_clip=1.0, weight_decay=1e-4,
                              patience=a.patience)
        samplers.append((sd, params, nfs))
        print(f"[nde] {a.tag} seed {sd} trained", flush=True)

    def draw(params, nfs, y_obs, key, m):
        yb = jnp.broadcast_to(jnp.asarray(y_obs).reshape(1, dim), (m, dim))
        s = nfs.apply(params, key, yb, m)
        return np.asarray(s)

    # ---- null posterior at y_fid (pooled, + optional per-seed) ----
    pooled, per_seed = [], {}
    for sd, params, nfs in samplers:
        s = draw(params, nfs, y_fid, jax.random.PRNGKey(sd + 7), a.num_samples)
        s = s[np.all(np.isfinite(s), 1)]
        per_seed[str(sd)] = [float(v) for v in s.mean(0)]
        pooled.append(s)
        if a.save_per_seed:
            np.save(out / f"null_s{sd}_{a.tag}.npy", s)
    null = np.concatenate(pooled, 0)
    np.save(out / f"null_pooled_{a.tag}.npy", null)

    # ---- biased posterior at the baryonified fiducial (if provided) ----
    if y_fid_biased is not None:
        bpool = []
        for sd, params, nfs in samplers:
            s = draw(params, nfs, y_fid_biased, jax.random.PRNGKey(sd + 17), a.num_samples)
            s = s[np.all(np.isfinite(s), 1)]
            bpool.append(s)
            if a.save_per_seed:
                np.save(out / f"biased_s{sd}_{a.tag}.npy", s)
        biased = np.concatenate(bpool, 0)
        np.save(out / f"null_biased_pooled_{a.tag}.npy", biased)
        bm = biased.mean(0)
        print(f"[nde] {a.tag} BIASED Om={bm[0]:.3f} S8={bm[1]:.3f} w0={bm[2]:.3f}", flush=True)

    # ---- TARP/SBC bundle on held-out val cosmologies (pooled posterior per point) ----
    if not a.no_tarp:
        ntp = min(a.tarp_points, len(theta_va))
        ridx = np.random.default_rng(0).choice(len(theta_va), size=ntp, replace=False)
        th_t, y_t = theta_va[ridx], y_va[ridx]
        samps = np.full((ntp, a.tarp_draws * len(seeds), 6), np.nan, np.float32)
        for i in range(ntp):
            chunks = []
            for j, (sd, params, nfs) in enumerate(samplers):
                chunks.append(draw(params, nfs, y_t[i], jax.random.PRNGKey(1000 + sd * 9973 + i), a.tarp_draws))
            samps[i] = np.concatenate(chunks, 0)
        # store as (n_draws, n_points, 6) for tarp.get_tarp_coverage
        np.save(out / f"tarp_samples_{a.tag}.npy", np.transpose(samps, (1, 0, 2)))
        np.save(out / f"tarp_theta_{a.tag}.npy", th_t)

    summ = {"tag": a.tag, "dim": int(dim), "n_seeds": len(seeds), "n_null": int(null.shape[0]),
            "null_mean": [float(v) for v in null.mean(0)], "null_std": [float(v) for v in null.std(0)],
            "per_seed_mean": per_seed, "tarp_points": int(ntp), "tarp_draws_per_point": int(a.tarp_draws * len(seeds))}
    (out / f"summary_{a.tag}.json").write_text(json.dumps(summ, indent=2))
    m, s = null.mean(0), null.std(0)
    print(f"[nde] {a.tag} NULL Om={m[0]:.3f}±{s[0]:.3f} s8={m[1]:.3f}±{s[1]:.3f} w0={m[2]:.3f}±{s[2]:.3f}",
          flush=True)
    print("NDE OK")


if __name__ == "__main__":
    main()
