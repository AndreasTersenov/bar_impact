#!/usr/bin/env python3
"""Stage-2 NDE: sbi_lens ConditionalRealNVP via a faithful port of the cnn_sbi
`build_flow` / `train_flow` / `sample_posterior` (npe_cnn_nbody_tomo.py).

This is a VERBATIM port of AT's validated cnn_sbi reference, with ONLY wandb stripped (the
logging calls have no effect on the math). Keeping it byte-faithful is what lets P0's
port-equivalence gate assert the copy introduced no drift. The flow itself is sbi_lens's
`ConditionalRealNVP` (base N(0.5, 0.05)); cnn_sbi overrides its default bijector to the plain
`AffineCoupling` (shift+scale) with n_layers=4, hidden=128, silu.

Run with the jaxili interpreter. `import tensorflow_probability.substrates.jax` BEFORE sbi_lens
(lazy-loader ordering bug) — done at import time below.
"""
from __future__ import annotations

import json
import pickle
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as _tfpj  # noqa: F401  (materialize tfp before sbi_lens)
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP


# =============================================================================
# Normalizing Flow (Conditional RealNVP) — JAX / Haiku  [port of npe_cnn_nbody_tomo.build_flow]
# =============================================================================
def build_flow(n_cosmo_params: int, n_layers: int, hidden: int):
    """Build conditional RealNVP for NPE:  p(theta | y).  Returns (nf_logp, nf_sample)."""
    bijector_fn = partial(AffineCoupling, layers=[hidden] * 2, activation=jax.nn.silu)
    NF_factory = partial(ConditionalRealNVP, n_layers=n_layers, bijector_fn=bijector_fn)

    class NF(hk.Module):
        def __call__(self, y):
            return NF_factory(n_cosmo_params)(y)

    @hk.transform
    def nf_log_prob(theta, y):
        return NF()(y).log_prob(theta).squeeze()

    @hk.transform
    def nf_sample(y, n_samples):
        return NF()(y).sample(n_samples, seed=hk.next_rng_key())

    nf_logp = hk.without_apply_rng(nf_log_prob)
    return nf_logp, nf_sample


def build_flow_embedded(n_cosmo_params: int, n_layers: int, hidden: int,
                        embed_dim: int = 16, embed_hidden=(256, 256)):
    """Conditional RealNVP with a learned EMBEDDING NETWORK on the conditioning input.

    WHY THIS EXISTS. Fed the rebin=20 BNT data vector raw, the plain flow above fails: it returns
    r(Omega_m, S8) = -0.03 where the physical lensing degeneracy is -0.9, inflating the 3-param
    volume 3.6x, while passing SBC and TARP (which test marginal coverage only). Measured cause is
    information DILUTION — nulling cancels the dominant common mode, so the signal that the standard
    basis concentrates in a few high-S/N bandpowers is spread across ~90 individually weak ones
    (top-10% of features carry 64% of the S8 Fisher in the standard basis, 5% after BNT). The flow's
    conditioner has to learn ~90 relative weights from a finite sim suite, and does not.

    Two known remedies: compress first (MOPED), or bin coarsely enough to re-concentrate the signal
    (raw BNT at rebin 40 recovers r = -0.93 and matches MOPED's FoM). Both have costs — MOPED is
    Gaussian-optimal and measurably lossy where the flow can already cope (on non-BNT, raw NPE beats
    MOPED 1.39e5 vs 1.11e5), and coarse binning starves the standard vector (non-BNT drops to 20
    features at rebin 40).

    An embedding network is the third option and the most natural one: give the density estimator a
    dedicated feature extractor and train it JOINTLY with the flow under the same NPE loss. It needs
    no covariance, so it is not restricted to Gaussian-optimal projections, and it is part of the
    density estimator rather than a separate analysis stage that has to be justified on its own.

    The embedding is a plain MLP; nothing here is BNT-specific. Set embed_dim >= n_cosmo_params —
    there is no reason to bottleneck below the number of parameters being inferred.
    """
    bijector_fn = partial(AffineCoupling, layers=[hidden] * 2, activation=jax.nn.silu)
    NF_factory = partial(ConditionalRealNVP, n_layers=n_layers, bijector_fn=bijector_fn)

    class NF(hk.Module):
        def __call__(self, y):
            h = hk.nets.MLP(list(embed_hidden) + [embed_dim],
                            activation=jax.nn.silu, name="embedding")(y)
            return NF_factory(n_cosmo_params)(h)

    @hk.transform
    def nf_log_prob(theta, y):
        return NF()(y).log_prob(theta).squeeze()

    @hk.transform
    def nf_sample(y, n_samples):
        return NF()(y).sample(n_samples, seed=hk.next_rng_key())

    return hk.without_apply_rng(nf_log_prob), nf_sample


def make_update_fn(nf_logp, optimizer):
    """JIT-compiled training update step."""
    def loss_fn(params, theta_batch, y_batch):
        return -jnp.mean(nf_logp.apply(params, theta_batch, y_batch))

    @jax.jit
    def update(params, opt_state, theta_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, theta_batch, y_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    return update


def train_flow(
    rng_key: jax.Array,
    nf_logp,
    dataset_train: Dict[str, np.ndarray],
    dataset_val: Dict[str, np.ndarray],
    n_cosmo: int,
    summary_dim: int,
    total_steps: int,
    batch_size: int,
    save_every: int,
    save_dir: Path,
    lr_init: float,
    end_lr: float,
    grad_clip: float = 1.0,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> hk.Params:
    """Train the conditional normalizing flow with early stopping (best-val kept).

    Faithful port of npe_cnn_nbody_tomo.train_flow with wandb removed. Batching uses the GLOBAL
    numpy RNG (np.random.randint), exactly as the reference — seed np.random externally for
    bit-for-bit reproducibility.
    """
    key_init, _ = jax.random.split(rng_key)

    theta_dummy = 0.5 * jnp.zeros([1, n_cosmo])
    y_dummy = jnp.zeros([1, summary_dim])
    params = nf_logp.init(key_init, theta_dummy, y_dummy)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Flow parameters: {n_params:,}")

    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr_init, decay_steps=total_steps, alpha=end_lr / max(lr_init, 1e-12),
    )
    opt_parts = []
    if grad_clip > 0:
        opt_parts.append(optax.clip_by_global_norm(grad_clip))
    opt_parts.append(optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay))
    optimizer = optax.chain(*opt_parts)
    opt_state = optimizer.init(params)
    update = make_update_fn(nf_logp, optimizer)

    theta_train, x_train = dataset_train["theta"], dataset_train["x"]
    theta_val, x_val = dataset_val["theta"], dataset_val["x"]
    n_train, n_val = len(theta_train), len(theta_val)

    batch_losses: list[float] = []
    val_losses: list[float] = []
    val_steps: list[int] = []

    best_val_loss = float("inf")
    best_step = 0
    best_params = params
    patience_counter = 0
    val_batch_size = min(512, n_val)

    for step in range(1, total_steps + 1):
        idx = np.random.randint(0, n_train, batch_size)
        loss, params, opt_state = update(params, opt_state, theta_train[idx], x_train[idx])
        batch_losses.append(float(loss))

        if step % 100 == 0:
            print(f"  Step {step:6d} | train loss {loss:.4f}")

        if step % save_every == 0 or step == total_steps:
            save_dir.mkdir(parents=True, exist_ok=True)
            vidx = np.random.randint(0, n_val, val_batch_size)
            val_l = float(-jnp.mean(nf_logp.apply(params, theta_val[vidx], x_val[vidx])))
            val_losses.append(val_l)
            val_steps.append(step)

            improved = ""
            if val_l < best_val_loss:
                best_val_loss = val_l
                best_step = step
                best_params = params
                patience_counter = 0
                improved = " ***"
                with open(save_dir / "params_flow_best.pkl", "wb") as f:
                    pickle.dump(params, f)
            else:
                patience_counter += 1

            print(f"  Saved @ step {step}. Val loss = {val_l:.4f}{improved}"
                  f"  (best = {best_val_loss:.4f}, patience = {patience_counter})")

            if patience > 0 and patience_counter >= patience:
                print(f"  Early stopping at step {step} (no val improvement for {patience} checks)")
                break

    np.save(save_dir / "loss_train.npy", np.array(batch_losses))
    np.save(save_dir / "loss_val.npy", np.array(val_losses))
    np.save(save_dir / "loss_val_steps.npy", np.array(val_steps))
    summary = {
        "best_val_loss": float(best_val_loss), "best_step": int(best_step),
        "final_step": int(step), "best_at_final_step": bool(best_step == step),
        "total_steps_requested": int(total_steps), "save_every": int(save_every),
        "patience": int(patience), "n_val_checks": int(len(val_losses)),
    }
    (save_dir / "flow_training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Best validation loss: {best_val_loss:.4f} @ step {best_step}")
    if best_step == step:
        print("  WARNING: best val at final step — flow may be underconverged.")
    return best_params


def sample_posterior(rng_key, nf_sample, flow_params, summary_obs, n_samples):
    """Draw posterior samples via the trained NPE flow (NaN rows dropped)."""
    summary_dim = summary_obs.shape[-1]
    y_obs = jnp.asarray(summary_obs).reshape(1, summary_dim)
    y_cond = jnp.repeat(y_obs, repeats=n_samples, axis=0)
    samples = nf_sample.apply(flow_params, rng_key, y_cond, n_samples)
    samples = samples[~jnp.any(jnp.isnan(samples), axis=-1)]
    return np.array(samples)
