#!/usr/bin/env python3
"""P0 verification for the Stage-2 RealNVP port (scripts/nde_realnvp.py).

Three independent checks (run any subset via --checks):
  equiv  : port-equivalence vs cnn_sbi. Same seed + same tiny (theta,y) -> the ported train_flow
           best-val must equal cnn_sbi's npe_cnn_nbody_tomo.train_flow best-val to ~machine
           precision (proves the copy added no drift).
  oracle : linear-Gaussian toy with a KNOWN analytic posterior. Train the flow on (theta, y=A theta+eps),
           sample at a fixed y_obs, and check the posterior mean/cov match the analytic Gaussian
           (proves flow + sampling are correct end-to-end, cache-independent).
  smoke  : real bnt_full_14000_nobary vector, H0/100, by-cosmology split, per-feature z-scored 120-D x
           fed raw to the flow; sample at x_fid; the null must be finite, constrained, ~on-truth in
           (Omega_m, sigma_8, w_0).

Writes one json per check under --out. jaxili interpreter; set --gpu.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

TRUTH = np.array([0.26, 0.84, -1.0, 0.6736, 0.9649, 0.0493])  # post-/100 (coincides with cnn_sbi FIDUCIAL)
SCORE_CACHE = "outputs/score_experiment/cache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="2")
    p.add_argument("--checks", default="equiv,oracle,smoke")
    p.add_argument("--out", default="outputs/baryon_tension/vmim_v2/p0")
    p.add_argument("--mem-fraction", default="0.3")
    return p.parse_args()


def _by_cosmology_split(theta, val_frac=0.1, seed=0):
    """Hold out whole cosmologies (all realizations together). Returns (tr_idx, va_idx)."""
    keys = np.round(theta, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(val_frac * len(uniq)))
    val_cosmo = set(perm[:n_val].tolist())
    is_val = np.array([c in val_cosmo for c in inv])
    return np.where(~is_val)[0], np.where(is_val)[0], len(uniq), n_val


# ----------------------------------------------------------------------------- equiv
def check_equiv(out):
    """Train the ported and cnn_sbi train_flow on identical tiny data; compare best-val."""
    import sys
    import jax
    import nde_realnvp as port

    rng = np.random.default_rng(123)
    n, d_y = 2000, 8
    theta = rng.normal(0.5, 0.1, size=(n, 6)).astype(np.float32)
    A = rng.normal(0, 1, size=(d_y, 6)).astype(np.float32)
    y = (theta @ A.T + 0.1 * rng.normal(size=(n, d_y))).astype(np.float32)
    nval = n // 10
    dtr = {"theta": theta[nval:], "x": y[nval:]}
    dva = {"theta": theta[:nval], "x": y[:nval]}
    kw = dict(n_cosmo=6, summary_dim=d_y, total_steps=400, batch_size=128, save_every=50,
              lr_init=1e-3, end_lr=1e-5, grad_clip=1.0, weight_decay=1e-4, patience=1000)

    # ported
    np.random.seed(7)
    nfp, _ = port.build_flow(6, n_layers=4, hidden=128)
    p_params = port.train_flow(jax.random.PRNGKey(0), nfp, dtr, dva,
                               save_dir=Path(out) / "equiv_port", **kw)
    port_best = json.loads((Path(out) / "equiv_port" / "flow_training_summary.json").read_text())["best_val_loss"]

    # cnn_sbi reference (wandb disabled)
    sys.path.insert(0, "/home/tersenov/software/cnn_sbi/scripts/sbi")
    import wandb
    wandb.init(mode="disabled")
    import npe_cnn_nbody_tomo as ref
    np.random.seed(7)
    nfr, _ = ref.build_flow(6, n_layers=4, hidden=128)
    ref.train_flow(jax.random.PRNGKey(0), nfr, dtr, dva,
                   save_dir=Path(out) / "equiv_ref", lr_schedule_fn=None, **kw)
    ref_best = json.loads((Path(out) / "equiv_ref" / "flow_training_summary.json").read_text())["best_val_loss"]

    delta = abs(port_best - ref_best)
    res = {"check": "equiv", "port_best_val": port_best, "ref_best_val": ref_best,
           "abs_delta": delta, "PASS": bool(delta < 1e-5)}
    print(f"[equiv] port={port_best:.8f} ref={ref_best:.8f} |Δ|={delta:.2e} -> {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ----------------------------------------------------------------------------- oracle
def check_oracle(out):
    """Linear-Gaussian: theta ~ N(mu0,S0), y|theta ~ N(A theta, Sn). Flow posterior vs analytic."""
    import jax
    import nde_realnvp as port

    rng = np.random.default_rng(321)
    d_t, d_y, n = 6, 8, 40000
    mu0 = np.array([0.30, 0.80, -1.0, 0.67, 0.96, 0.05])
    s0 = np.array([0.10, 0.12, 0.20, 0.05, 0.05, 0.01])
    S0 = np.diag(s0 ** 2)
    A = rng.normal(0, 0.8, size=(d_y, d_t))
    sn = 0.15 * np.ones(d_y)
    Sn = np.diag(sn ** 2)

    theta = rng.normal(mu0, s0, size=(n, d_t))
    y = theta @ A.T + rng.normal(0, sn, size=(n, d_y))
    theta = theta.astype(np.float32); y = y.astype(np.float32)
    nval = n // 10
    dtr = {"theta": theta[nval:], "x": y[nval:]}
    dva = {"theta": theta[:nval], "x": y[:nval]}

    # analytic posterior at a fixed y_obs (use the prior mean's noiseless image)
    y_obs = (mu0 @ A.T).astype(np.float32)
    S0i, Sni = np.linalg.inv(S0), np.linalg.inv(Sn)
    Spost = np.linalg.inv(S0i + A.T @ Sni @ A)
    mupost = Spost @ (S0i @ mu0 + A.T @ Sni @ y_obs)

    np.random.seed(11)
    nfp, nfs = port.build_flow(d_t, n_layers=4, hidden=128)
    params = port.train_flow(jax.random.PRNGKey(1), nfp, dtr, dva, n_cosmo=d_t, summary_dim=d_y,
                             total_steps=8000, batch_size=256, save_every=200, lr_init=1e-3,
                             end_lr=1e-5, grad_clip=1.0, weight_decay=1e-4, patience=30,
                             save_dir=Path(out) / "oracle_flow")
    samp = port.sample_posterior(jax.random.PRNGKey(99), nfs, params, y_obs, 20000)

    mu_f, S_f = samp.mean(0), np.cov(samp, rowvar=False)
    sd_an, sd_f = np.sqrt(np.diag(Spost)), np.sqrt(np.diag(S_f))
    mean_z = np.abs(mu_f - mupost) / sd_an          # mean error in units of analytic sigma
    sd_ratio = sd_f / sd_an
    res = {"check": "oracle", "n_samples": int(samp.shape[0]),
           "analytic_mean": mupost.tolist(), "flow_mean": mu_f.tolist(),
           "analytic_sigma": sd_an.tolist(), "flow_sigma": sd_f.tolist(),
           "mean_err_in_sigma": mean_z.tolist(), "sigma_ratio": sd_ratio.tolist(),
           "PASS": bool(np.all(mean_z < 0.25) and np.all((sd_ratio > 0.75) & (sd_ratio < 1.35)))}
    print(f"[oracle] max mean-err {mean_z.max():.3f}σ ; sigma-ratio [{sd_ratio.min():.2f},{sd_ratio.max():.2f}]"
          f" -> {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ----------------------------------------------------------------------------- smoke
def check_smoke(out):
    """Real bnt_full vector, H0/100, by-cosmo split, z-scored 120-D x fed raw; null at x_fid."""
    import jax
    import nde_realnvp as port

    z = np.load(f"{SCORE_CACHE}/bnt_full_14000_nobary/cache.npz")
    theta = z["theta"].astype(np.float64).copy()
    theta[:, 3] /= 100.0                                       # H0 -> h0
    X = z["x"].astype(np.float64)
    x_fid = z["x_fid"].astype(np.float64)

    tr_idx, va_idx, n_cosmo, n_val = _by_cosmology_split(theta, val_frac=0.1, seed=0)
    mean, std = X[tr_idx].mean(0), X[tr_idx].std(0)
    std[std < 1e-12] = 1.0
    zc = lambda a: np.clip((np.atleast_2d(a) - mean) / std, -5, 5)
    x_tr, x_va = zc(X[tr_idx]).astype(np.float32), zc(X[va_idx]).astype(np.float32)
    x_fid_z = zc(x_fid).astype(np.float32)
    dtr = {"theta": theta[tr_idx].astype(np.float32), "x": x_tr}
    dva = {"theta": theta[va_idx].astype(np.float32), "x": x_va}

    np.random.seed(5)
    nfp, nfs = port.build_flow(6, n_layers=4, hidden=128)
    params = port.train_flow(jax.random.PRNGKey(2), nfp, dtr, dva, n_cosmo=6, summary_dim=X.shape[1],
                             total_steps=8000, batch_size=256, save_every=200, lr_init=1e-3,
                             end_lr=1e-5, grad_clip=1.0, weight_decay=1e-4, patience=25,
                             save_dir=Path(out) / "smoke_flow")
    samp = port.sample_posterior(jax.random.PRNGKey(77), nfs, params, x_fid_z[0], 20000)
    mu, sd = samp.mean(0), samp.std(0)
    names = ["Om", "s8", "w0", "h0", "ns", "Ob"]
    on_truth = np.abs(mu[:3] - TRUTH[:3]) / sd[:3]            # |bias| in sigma for the 3 we care about
    res = {"check": "smoke", "n_cosmo": int(n_cosmo), "n_val_cosmo": int(n_val),
           "n_samples": int(samp.shape[0]), "null_mean": dict(zip(names, mu.tolist())),
           "null_std": dict(zip(names, sd.tolist())), "bias_in_sigma_Om_s8_w0": on_truth.tolist(),
           "finite": bool(np.all(np.isfinite(samp))), "constrained": bool(np.all(sd[:3] < np.array([0.1, 0.15, 0.4]))),
           "PASS": bool(np.all(np.isfinite(samp)) and np.all(on_truth < 1.0)
                        and np.all(sd[:3] < np.array([0.1, 0.15, 0.4])))}
    print(f"[smoke] null Om={mu[0]:.3f}±{sd[0]:.3f} s8={mu[1]:.3f}±{sd[1]:.3f} w0={mu[2]:.3f}±{sd[2]:.3f}"
          f" | bias(σ)={np.round(on_truth,2).tolist()} -> {'PASS' if res['PASS'] else 'FAIL'}")
    return res


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = a.mem_fraction
    import sys
    sys.path.insert(0, "scripts")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fns = {"equiv": check_equiv, "oracle": check_oracle, "smoke": check_smoke}
    for c in [c.strip() for c in a.checks.split(",") if c.strip()]:
        res = fns[c](str(out))
        (out / f"{c}.json").write_text(json.dumps(res, indent=2))
        print(f"[p0] wrote {out}/{c}.json")


if __name__ == "__main__":
    main()
