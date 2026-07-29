#!/usr/bin/env python3
"""Stage 2 (recipe-faithful) — sbi_lens ConditionalRealNVP on a FROZEN VMIM summary.

Replaces the jaxili-default MAF (recipe lesson 1: RealNVP beats MAF ~30% on compressed summaries).
Fits q(θ|y) with the sbi_lens RealNVP (n_layers=4, coupling [128,128] silu), θ min-max normalized to
[0,1] (the flow's base is N(0.5, 0.05) — expects ~[0,1], NOT z-score), summary z-scored, adamw +
cosine LR + grad-clip + early stopping. Pools `--seeds` NDE seeds. SBC + saves the pooled posterior.

Input: compressed.npz (from vmim_compress.py). Run with the jaxili interpreter.
"""
import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compressed", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--nf-hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--tarp-points", type=int, default=150)
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()


def main():
    a = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import optax
    from functools import partial
    import tensorflow_probability.substrates.jax as _tfpj  # noqa: F401
    from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP

    z = np.load(a.compressed)
    th_tr, y_tr = z["theta_tr"].astype(np.float64), z["y_tr"].astype(np.float32)
    th_va, y_va = z["theta_va"].astype(np.float64), z["y_va"].astype(np.float32)
    y_fid = z["y_fid"].astype(np.float32).reshape(1, -1)
    d, sdim = th_tr.shape[1], y_tr.shape[1]

    # theta -> [0,1] (min-max on train, small margin); summary -> z-score
    lo, hi = th_tr.min(0), th_tr.max(0)
    rng_ = (hi - lo); rng_[rng_ == 0] = 1.0
    nθ = lambda t: ((t - lo) / rng_).astype(np.float32)
    uθ = lambda t: t * rng_ + lo
    ymu, ysd = y_tr.mean(0), y_tr.std(0) + 1e-8
    ny = lambda x: ((x - ymu) / ysd).astype(np.float32)
    Xtr, Ytr = nθ(th_tr), ny(y_tr)
    Xva, Yva = nθ(th_va), ny(y_va)
    Yfid = ny(y_fid)
    print(f"[rnvp] {a.tag}: train {Xtr.shape} val {Xva.shape} d={d} sdim={sdim}", flush=True)

    bij = partial(AffineCoupling, layers=[a.nf_hidden, a.nf_hidden], activation=jax.nn.silu)

    def make(y):
        return ConditionalRealNVP(d, n_layers=a.n_layers, bijector_fn=bij)(y)

    logp = hk.without_apply_rng(hk.transform(lambda th, y: make(y).log_prob(th)))
    # sample like the cnn_sbi reference: NF()(y).sample(n, seed=...) with y broadcast to (n, sdim),
    # rng via hk.next_rng_key() (so this transform KEEPS its rng).
    samp = hk.transform(lambda y, n: make(y).sample(n, seed=hk.next_rng_key()))

    def draw(p, y_vec, n, key):
        yb = jnp.broadcast_to(jnp.asarray(y_vec).reshape(1, sdim), (n, sdim))
        return np.asarray(samp.apply(p, key, yb, n))

    sched = optax.cosine_decay_schedule(a.lr, a.steps, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))

    def loss_fn(p, th, y):
        return -jnp.mean(logp.apply(p, th, y))

    @jax.jit
    def update(p, o, th, y):
        l, g = jax.value_and_grad(loss_fn)(p, th, y)
        u, o = opt.update(g, o, p)
        return l, optax.apply_updates(p, u), o

    @jax.jit
    def vloss(p, th, y):
        return loss_fn(p, th, y)

    pooled, per_seed = [], []
    for sd in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.default_rng(sd)
        key = jax.random.PRNGKey(sd)
        params = logp.init(key, jnp.asarray(Xtr[:2]), jnp.asarray(Ytr[:2]))
        opt_state = opt.init(params)
        Xt, Yt = jnp.asarray(Xtr), jnp.asarray(Ytr)
        Xv, Yv = jnp.asarray(Xva), jnp.asarray(Yva)
        best, best_p, wait = np.inf, None, 0
        for step in range(1, a.steps + 1):
            idx = rng.integers(0, Xtr.shape[0], size=a.batch_size)
            l, params, opt_state = update(params, opt_state, Xt[idx], Yt[idx])
            if step % a.val_every == 0:
                vl = float(np.mean([float(vloss(params, Xv[i:i+4096], Yv[i:i+4096]))
                                    for i in range(0, Xv.shape[0], 4096)]))
                if np.isfinite(vl) and vl < best - 1e-4:
                    best, best_p, wait = vl, jax.tree.map(np.asarray, params), 0
                else:
                    wait += 1
                if wait >= a.patience:
                    break
        if best_p is None:
            print(f"[rnvp] seed {sd}: no best — skip", flush=True); continue
        s = uθ(draw(best_p, Yfid[0], a.num_samples, jax.random.PRNGKey(sd + 9)))
        pooled.append(s); per_seed.append((sd, best_p))
        print(f"[rnvp] seed {sd}: val={best:.3f} S8={s[:,1].mean():.4f}±{s[:,1].std():.4f}", flush=True)

    if not pooled:
        print("[rnvp] FAIL"); return
    samples = np.concatenate(pooled)
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/posterior_rnvp_{a.tag}.npy", samples)
    summ = {"tag": a.tag, "mean": samples.mean(0).tolist(), "std": samples.std(0).tolist(),
            "n_seeds": len(pooled)}

    # SBC on the first seed's flow
    sbc = {"ran": False}
    try:
        sd, p = per_seed[0]
        nt = min(a.tarp_points, Xva.shape[0])
        ranks = []
        for i in range(nt):
            dr = uθ(draw(p, Yva[i], 200, jax.random.PRNGKey(3000 + i)))
            ranks.append((dr < th_va[i]).mean(0))
        rstd = np.array(ranks).std(0).tolist()
        w = float(max(rstd))
        sbc = {"ran": True, "rank_std": rstd,
               "verdict": "OK" if w < 0.32 and min(rstd) > 0.24 else
                          ("OVERCONF" if w >= 0.32 else "UNDERCONF")}
        print(f"[rnvp] SBC rank-std={[f'{r:.3f}' for r in rstd]} -> {sbc['verdict']}", flush=True)
    except Exception as e:
        sbc = {"ran": False, "error": str(e)[:200]}
    summ["sbc"] = sbc
    json.dump(summ, open(f"{a.out}/summary_rnvp_{a.tag}.json", "w"), indent=2)
    print(f"[rnvp] {a.tag} DONE S8={samples[:,1].mean():.4f}±{samples[:,1].std():.4f} "
          f"Om={samples[:,0].mean():.4f}±{samples[:,0].std():.4f} sbc={sbc.get('verdict','?')}")


if __name__ == "__main__":
    main()
