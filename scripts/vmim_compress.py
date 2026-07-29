#!/usr/bin/env python3
"""Stage 1 — VMIM compressor for the BNT power-spectrum data vectors (corrected recipe).

A faithful port of cnn_sbi/scripts/sbi/vmim_from_cache.py (AT's validated ℓ1-norm recipe). The
compressor (MLP 256,256 leaky_relu → summary_dim) and the VMIM loss (min L = −E[log q(θ|c(x))] with a
ConditionalRealNVP companion, discarded after) are identical to the reference. Only what the BNT data
forces is changed, and the three deviations that broke the prior attempt are fixed:

  * preproc      : reference is `log1p-zscore`; BNT C_l are SIGNED so log1p is invalid. Default here is
                   per-feature z-score (NO log) + clip + min-variance mask — the reference minus the
                   log1p. `--preproc whiten|pca_whiten` kept for A/B (whiten = the prior, broken, full
                   Cholesky; it amplifies the ill-conditioned nulled directions).
  * split        : the reference consumes a PRE-SPLIT train/val cache (no leakage). Our score cache is
                   one file, so we split internally — BY COSMOLOGY (all realizations of a cosmology to
                   the same side). `--split random` reproduces the prior leakage bug for A/B only.
  * H0 scaling   : reference feeds θ RAW because its h0 is already /100. We do `θ[:,3] /= 100` at load,
                   then raw θ everywhere (no z-score). First three params (Ωm,σ8,w0) stay physical.
  * summary-noise: removed (default 0) — it was a tuning knob that slid the null. Flag kept for ablation.

Input  : <cache.npz> with arrays theta (N,6), x (N,nfeat), x_fid (nfeat,)   [RAW, from --dump-cache]
Output : <out>/compressed.npz with theta_tr,y_tr,theta_va,y_va,y_fid + preproc params + split/QA info.

Run with the jaxili interpreter.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="cache.npz from worker --dump-cache (nobaryons = the model)")
    p.add_argument("--biased-cache", default=None,
                   help="optional baryonified cache.npz; ONLY its x_fid is used (the biased observation). "
                        "Compressed with the same nobaryons-fit preproc + trained MLP -> y_fid_biased.")
    p.add_argument("--out", required=True, help="output dir for compressed.npz")
    p.add_argument("--summary-dim", type=int, default=8)
    p.add_argument("--hidden", default="256,256")
    p.add_argument("--nf-layers", type=int, default=4)
    p.add_argument("--nf-hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=41, help="network init + batch RNG")
    p.add_argument("--split-seed", type=int, default=None,
                   help="train/val split seed; defaults to --seed. Fix this ACROSS a compressor "
                        "ensemble (vary only --seed) so members share the held-out val set.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--max-minutes", type=float, default=30.0)
    p.add_argument("--preproc", choices=["zscore", "whiten", "pca_whiten", "ana_whiten"],
                   default="zscore")
    p.add_argument("--split", choices=["cosmology", "random"], default="cosmology")
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--min-feature-variance", type=float, default=1e-8,
                   help="RELATIVE: drop features with variance < this * median(variance) "
                        "(scale-invariant; catches dead/constant bins, not low-amplitude C_l).")
    p.add_argument("--ridge-rel", type=float, default=1e-10, help="whiten/pca_whiten ridge")
    p.add_argument("--pca-var", type=float, default=0.999, help="pca_whiten cumulative-variance keep")
    p.add_argument("--analytic-cov", choices=["bnt", "nonbnt"], default=None,
                   help="REQUIRED for preproc=ana_whiten: which analytic cov to noise-whiten by "
                        "(IMNN-style; isotropizes noise so the MLP learns only the J projection).")
    p.add_argument("--ana-ridge", type=float, default=1e-4,
                   help="ana_whiten eigenvalue floor as a fraction of max eigenvalue (caps the "
                        "amplification of near-nulled directions).")
    p.add_argument("--cov-npz", default=None,
                   help="ana_whiten: load the covariance from this .npy/.npz (key 'C' if npz) instead "
                        "of build_score — e.g. a native (un-rebinned) cov matching a pre-cut cache.")
    p.add_argument("--area", default="14000", help="FISHER_AREA for the analytic cov (ana_whiten).")
    p.add_argument("--cuts", default=None,
                   help="optional per-bin ℓmax cut, 4 comma ints (e.g. '480,480,480,480' cut-all; "
                        "'480,1024,1024,1024' BNT bin-1). Slices the full cache via "
                        "score_cut_utils.keep_indices; ana_whiten cov rebuilt at the cut.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="0 = plain adam (reference)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="global-norm clip; 0 disables")
    p.add_argument("--summary-noise", type=float, default=0.0,
                   help="ABLATION ONLY. Gaussian noise on c(x) during training. 0 = off (default).")
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()


# ------------------------------------------------------------------ train/val split
def split_by_cosmology(theta, val_frac, seed):
    """Hold out whole cosmologies (all realizations together). No row leakage across the boundary."""
    keys = np.round(theta, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(val_frac * len(uniq)))
    val_cosmo = set(perm[:n_val].tolist())
    is_val = np.fromiter((c in val_cosmo for c in inv), dtype=bool, count=len(inv))
    return np.where(~is_val)[0], np.where(is_val)[0], len(uniq), n_val


def split_random(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    return perm[n_val:], perm[:n_val], n, n_val


# ------------------------------------------------------------------ preprocessing (fit on train)
def fit_preproc(X, kind, clip, min_var, ridge_rel, pca_var, cov=None, ana_ridge=1e-4):
    """Return a dict describing the transform fit on train X, plus a callable `apply`."""
    if kind == "ana_whiten":
        # IMNN-style: whiten by the (regularized) ANALYTIC covariance so noise is isotropic and the
        # MLP only has to find the parameter-responsive directions. Center by the train mean.
        mu = X.mean(0)
        ev, V = np.linalg.eigh(cov)
        floor = ana_ridge * ev.max()
        evf = np.maximum(ev, floor)
        W = (V / np.sqrt(evf)) @ V.T                  # symmetric C^{-1/2} (eigenvalue-floored)
        # PER-FEATURE-RELATIVE clip: after C^{-1/2} the parameter-sensitive directions carry the
        # cosmology-variation signal so their std >> 1 (here up to ~4.6). An ABSOLUTE clip (right for
        # unit-variance z-score) would lop that signal and bias S8 — so clip each whitened feature at
        # ±clip * its OWN std (clips genuine outliers, preserves signal).
        sw = ((X - mu) @ W).std(0)
        sw[sw < 1e-12] = 1.0

        def apply(A):
            # z-score the whitened features to UNIT std before the MLP. Equivalent to the old
            # per-feature ±clip*sw clipping (same signal preservation), but bounds the network input
            # to ±clip instead of ±clip*sw — at aggressive cuts sw reaches ~16, and ±80 inputs NaN'd
            # the companion. Information content is unchanged (an invertible per-feature rescale).
            Z = ((np.atleast_2d(A) - mu) @ W) / sw
            return np.clip(Z, -clip, clip) if (clip and clip > 0) else Z
        info = {"kind": kind, "mu": mu, "W": W, "sw": sw, "ana_ridge": ana_ridge,
                "cond_raw": float(ev.max() / max(ev.min(), 1e-300)),
                "cond_floored": float(evf.max() / evf.min()),
                "whitened_std_range": [float(sw.min()), float(sw.max())], "clip": clip}
        return info, apply

    if kind == "zscore":
        mean = X.mean(0)
        std = X.std(0)
        var = std ** 2                           # RELATIVE mask: drop dead bins, scale-invariant
        mask = var >= min_var * max(np.median(var), 1e-300)
        std_safe = std.copy(); std_safe[std_safe < 1e-12] = 1.0

        def apply(A):
            Z = (np.atleast_2d(A) - mean) / std_safe
            if clip and clip > 0:
                Z = np.clip(Z, -clip, clip)
            return Z[:, mask]
        info = {"kind": kind, "mean": mean, "std": std_safe, "mask": mask, "clip": clip}
        return info, apply

    if kind in ("whiten", "pca_whiten"):
        mu = X.mean(0)
        C = np.cov(X, rowvar=False)
        ridge = ridge_rel * np.median(np.diag(C))
        if kind == "whiten":
            L = np.linalg.cholesky(C + ridge * np.eye(C.shape[0]))
            W = np.linalg.inv(L).T                # (x-mu) @ W  == L^{-1}(x-mu)

            def apply(A):
                Z = (np.atleast_2d(A) - mu) @ W
                return np.clip(Z, -clip, clip) if (clip and clip > 0) else Z
            info = {"kind": kind, "mu": mu, "W": W, "clip": clip}
            return info, apply

        # pca_whiten: regularized/truncated whitening — drop the noisy low-variance directions
        ev, V = np.linalg.eigh(C + ridge * np.eye(C.shape[0]))
        order = np.argsort(ev)[::-1]
        ev, V = ev[order], V[:, order]
        keep = np.searchsorted(np.cumsum(ev) / ev.sum(), pca_var) + 1
        Wk = V[:, :keep] / np.sqrt(ev[:keep])     # (x-mu) @ Wk -> whitened k-dim

        def apply(A):
            Z = (np.atleast_2d(A) - mu) @ Wk
            return np.clip(Z, -clip, clip) if (clip and clip > 0) else Z
        info = {"kind": kind, "mu": mu, "Wk": Wk, "keep": int(keep), "clip": clip}
        return info, apply

    raise ValueError(kind)


def main():
    a = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import optax
    from functools import partial
    import tensorflow_probability.substrates.jax as _tfpj  # noqa: F401 (materialize tfp before sbi_lens)
    from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP

    t0 = time.time()
    deadline = t0 + a.max_minutes * 60.0
    z = np.load(a.cache)
    theta = z["theta"].astype(np.float64).copy()
    theta[:, 3] /= 100.0                                  # H0 -> h0 (O(1) for the N(0.5,0.05) base)
    theta = theta.astype(np.float32)
    X = z["x"].astype(np.float64)
    x_fid = z["x_fid"].astype(np.float64)

    # ---- optional scale cut: slice the full vector to keep_indices(cuts) ----
    cuts = None
    keep = None
    if a.cuts:
        cuts = [int(v) for v in a.cuts.split(",")]
        os.environ.setdefault("FISHER_AREA", str(a.area))
        sys.path.insert(0, "scripts")
        import score_cut_utils as SC
        keep = SC.keep_indices(cuts)
        X = X[:, keep]
        x_fid = x_fid[keep]
        print(f"[vmim] cut {cuts} -> kept {keep.size}/{z['x'].shape[1]} features", flush=True)

    n, nfeat = X.shape
    print(f"[vmim] cache theta={theta.shape} x={X.shape} fid={x_fid.shape}; H0/100 applied", flush=True)

    # ---- split (by cosmology by default; assert no leakage) ----
    split_seed = a.split_seed if a.split_seed is not None else a.seed
    if a.split == "cosmology":
        tr_idx, va_idx, n_cosmo, n_val_cosmo = split_by_cosmology(theta, a.val_frac, split_seed)
        tr_keys = {tuple(r) for r in np.round(theta[tr_idx], 6)}
        va_keys = {tuple(r) for r in np.round(theta[va_idx], 6)}
        leak = len(tr_keys & va_keys)
        assert leak == 0, f"cosmology leakage: {leak} shared cosmologies between train and val"
        print(f"[vmim] split=cosmology  {n_cosmo} cosmologies -> {n_val_cosmo} val; leakage={leak} "
              f"(train rows {tr_idx.size}, val rows {va_idx.size})", flush=True)
    else:
        tr_idx, va_idx, n_cosmo, n_val_cosmo = split_random(n, a.val_frac, split_seed)
        leak = -1
        print(f"[vmim] split=random (ABLATION — leaks realizations) train {tr_idx.size} val {va_idx.size}",
              flush=True)

    # ---- analytic covariance for ana_whiten (the 120-dim full-vector hybrid C from the score code) ----
    cov = None
    if a.preproc == "ana_whiten":
        if a.cov_npz:                                     # native / precomputed cov supplied directly
            z = np.load(a.cov_npz)
            cov = z["C"] if hasattr(z, "files") and "C" in getattr(z, "files", []) else np.asarray(z)
            print(f"[vmim] ana_whiten cov from {a.cov_npz} shape={cov.shape}", flush=True)
        else:
            if a.analytic_cov is None:
                print("[vmim] --analytic-cov {bnt,nonbnt} required for ana_whiten — FAIL", flush=True)
                sys.exit(2)
            os.environ.setdefault("FISHER_AREA", str(a.area))
            sys.path.insert(0, "scripts")
            import score_cut_utils as SC
            cov = SC.build_score(cuts if cuts else SC.FULL_CUTS,
                                 bnt=(a.analytic_cov == "bnt"), covk="hybrid")["C"]
        assert cov.shape[0] == nfeat, f"ana_whiten cov dim {cov.shape[0]} != x dim {nfeat}"

    # ---- preprocessing fit on TRAIN, applied to val + fid ----
    info, apply = fit_preproc(X[tr_idx], a.preproc, a.clip_value, a.min_feature_variance,
                              a.ridge_rel, a.pca_var, cov=cov, ana_ridge=a.ana_ridge)
    x_tr = apply(X[tr_idx]).astype(np.float32)
    x_va = apply(X[va_idx]).astype(np.float32)
    x_fid_p = apply(x_fid).astype(np.float32)
    in_dim = x_tr.shape[1]
    theta_tr, theta_va = theta[tr_idx], theta[va_idx]
    print(f"[vmim] preproc={a.preproc} nfeat {nfeat} -> in_dim {in_dim}; "
          f"x_tr finite={np.isfinite(x_tr).all()} std~[{x_tr.std(0).min():.2f},{x_tr.std(0).max():.2f}]",
          flush=True)

    # ---- compressor MLP (reference CompressorMLP, stateless) + RealNVP companion (raw theta) ----
    hidden = tuple(int(h) for h in a.hidden.split(","))

    def mlp_fn(x):
        net = x
        for w in hidden:
            net = jax.nn.leaky_relu(hk.Linear(w)(net))
        return hk.Linear(a.summary_dim)(net)

    mlp = hk.without_apply_rng(hk.transform(mlp_fn))
    bij = partial(AffineCoupling, layers=[a.nf_hidden, a.nf_hidden], activation=jax.nn.silu)
    nf_factory = partial(ConditionalRealNVP, n_layers=a.nf_layers, bijector_fn=bij)
    n_cosmo_p = theta.shape[1]
    nf = hk.without_apply_rng(hk.transform(
        lambda th, y: nf_factory(n_cosmo_p)(y).log_prob(th).squeeze()))

    key = jax.random.PRNGKey(a.seed)
    params = {"c": mlp.init(key, jnp.zeros((1, in_dim), jnp.float32)),
              "nf": nf.init(key, jnp.zeros((1, n_cosmo_p), jnp.float32),
                            jnp.zeros((1, a.summary_dim), jnp.float32))}
    print(f"[vmim] params: {sum(x.size for x in jax.tree.leaves(params)):,}", flush=True)

    sched_steps = a.steps - a.steps // 3
    lr = optax.piecewise_constant_schedule(
        a.lr, {int(sched_steps * f): 0.7 for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})
    opt_parts = []
    if a.grad_clip > 0:
        opt_parts.append(optax.clip_by_global_norm(a.grad_clip))
    opt_parts.append(optax.adamw(lr, weight_decay=a.weight_decay) if a.weight_decay > 0
                     else optax.adam(lr))
    opt = optax.chain(*opt_parts)
    opt_state = opt.init(params)

    def loss_fn(p, th, x, key):
        y = mlp.apply(p["c"], x)
        if a.summary_noise > 0:
            y = y + a.summary_noise * jax.random.normal(key, y.shape)
        return -jnp.mean(nf.apply(p["nf"], th, y))

    @jax.jit
    def update(p, o, th, x, key):
        loss, g = jax.value_and_grad(loss_fn)(p, th, x, key)
        u, o = opt.update(g, o, p)
        return loss, optax.apply_updates(p, u), o

    @jax.jit
    def eval_loss(p, th, x):  # noise-free companion NLL on val
        return -jnp.mean(nf.apply(p["nf"], th, mlp.apply(p["c"], x)))

    xt, tt = jnp.asarray(x_tr), jnp.asarray(theta_tr)
    xv, tv = jnp.asarray(x_va), jnp.asarray(theta_va)
    best_val, best_params, best_step, nonfinite, hist = np.inf, None, 0, 0, []
    rng = np.random.default_rng(a.seed)
    train_key = jax.random.PRNGKey(a.seed + 1)
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, x_tr.shape[0], size=a.batch_size)
        train_key, sk = jax.random.split(train_key)
        loss, params, opt_state = update(params, opt_state, tt[idx], xt[idx], sk)
        if not np.isfinite(float(loss)):
            nonfinite += 1
            if nonfinite > 20:
                print("[vmim] too many non-finite losses — stopping", flush=True); break
        if step % a.val_every == 0 or step == a.steps:
            vl = float(np.mean([float(eval_loss(params, tv[i:i + 4096], xv[i:i + 4096]))
                                for i in range(0, xv.shape[0], 4096)]))
            hist.append((step, float(loss), vl))
            flag = ""
            if np.isfinite(vl) and vl < best_val:
                best_val, best_params, best_step = vl, jax.tree.map(np.asarray, params), step
                flag = " *best"
            print(f"[vmim] step {step}/{a.steps} train {float(loss):.4f} val {vl:.4f} "
                  f"({time.time()-t0:.0f}s){flag}", flush=True)
            if time.time() > deadline:
                print(f"[vmim] TIME-BOX at step {step} — keeping best", flush=True); break

    if best_params is None:
        print("[vmim] NO usable checkpoint — FAIL", flush=True); sys.exit(3)
    print(f"[vmim] best val {best_val:.4f} @ step {best_step}", flush=True)

    @jax.jit
    def compress(x):
        return mlp.apply(best_params["c"], x)

    def comp_np(x, bs=8192):
        return np.concatenate([np.asarray(compress(jnp.asarray(x[i:i + bs])))
                               for i in range(0, x.shape[0], bs)]).astype(np.float32)

    y_tr, y_va = comp_np(x_tr), comp_np(x_va)
    y_fid = comp_np(x_fid_p)[0]
    extra = {}
    if a.biased_cache:                                    # baryonified fiducial -> biased summary
        x_fid_bias = np.load(a.biased_cache)["x_fid"].astype(np.float64)
        if keep is not None:
            x_fid_bias = x_fid_bias[keep]
        extra["y_fid_biased"] = comp_np(apply(x_fid_bias).astype(np.float32))[0]
        print(f"[vmim] biased fiducial compressed from {a.biased_cache}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    # preproc params saved opaquely (kind-dependent keys) for exact re-application downstream
    preproc_save = {f"preproc_{k}": np.asarray(v) for k, v in info.items() if k != "kind"}
    np.savez(os.path.join(a.out, "compressed.npz"),
             theta_tr=theta_tr, y_tr=y_tr, theta_va=theta_va, y_va=y_va, y_fid=y_fid,
             summary_dim=a.summary_dim, preproc_kind=a.preproc, in_dim=in_dim,
             best_val=best_val, best_step=best_step, **preproc_save, **extra)
    qa = {"preproc": a.preproc, "split": a.split, "n_cosmo": int(n_cosmo),
          "n_val_cosmo": int(n_val_cosmo), "cosmology_leakage": int(leak),
          "n_train": int(tr_idx.size), "n_val": int(va_idx.size), "in_dim": int(in_dim),
          "summary_dim": int(a.summary_dim), "summary_noise": a.summary_noise,
          "y_tr_finite": bool(np.isfinite(y_tr).all()), "y_fid_finite": bool(np.isfinite(y_fid).all()),
          "y_tr_per_dim_std": y_tr.std(0).tolist(), "min_dim_std": float(y_tr.std(0).min()),
          "best_val": float(best_val), "best_step": int(best_step),
          "h0_div100": True, "wall_s": time.time() - t0}
    with open(os.path.join(a.out, "vmim_qa.json"), "w") as fh:
        json.dump({"qa": qa, "hist": hist}, fh, indent=2)
    print(f"[vmim] wrote {a.out}/compressed.npz  y_tr={y_tr.shape} y_fid={y_fid.shape} "
          f"min_dim_std={y_tr.std(0).min():.3f}", flush=True)
    print("BUILD OK")


if __name__ == "__main__":
    main()
