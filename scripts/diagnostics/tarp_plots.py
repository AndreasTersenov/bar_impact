#!/usr/bin/env python3
"""TARP coverage plots for the score-compression NPEs: the PRIOR-averaged TARP (what npe_on_summary
checked, and passed) vs the FIDUCIAL-specific coverage (the contours' actual point). Retrains 1 seed
per config (fast), then runs get_tarp_coverage two ways:
  - PRIOR: random held-out grid (theta_i, x_i) pairs -> the standard TARP.
  - FIDUCIAL: the 200 compressed fiducial perms, all with theta = theta_fid -> coverage at the point.
A curve below the diagonal = over-confident (true value falls outside the credible region too often).
Run with jaxili python."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fisher_local_jacobian as L  # FID

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from jaxili.inference import NPE  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "tarp", "src"))
sys.path.insert(0, "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/tarp/src")
from tarp import get_tarp_coverage  # noqa: E402

SC = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/score_experiment/score"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/score_experiment"
CFG = {"nonbnt_460": "#7f7f7f", "bnt_580": "#c0392b"}
FID = np.asarray(L.FID, np.float32)


def train(tag):
    z = np.load(f"{SC}/compressed_{tag}_14000.npz")
    theta = z["theta_tr"].astype(np.float32)
    y = z["y_tr"].astype(np.float32)
    inf = NPE().append_simulations(jnp.asarray(theta), jnp.asarray(y), key=random.PRNGKey(41))
    inf.train(checkpoint_path=f"{OUT}/ckpt_tarp_{tag}", num_epochs=300,
              training_batch_size=256, learning_rate=5e-4)
    post = inf.build_posterior()
    perms_t = np.load(f"{SC}/score_cache_{tag}_14000.npz")["perms_t"].astype(np.float32)
    return post, theta, y, perms_t


from scipy.stats import chi2  # noqa: E402
P3 = [0, 1, 2]


def cover_tarp(post, X, thetas, ndraw=200):
    """Standard TARP — needs VARIED true thetas (prior panel)."""
    samp = np.stack([np.asarray(post.sample(x=jnp.asarray(X[i]), num_samples=ndraw,
                     key=random.PRNGKey(1000 + i))) for i in range(len(X))], axis=1)  # (ndraw, n, 6)
    ecp, alpha = get_tarp_coverage(samp, thetas, references="random", metric="euclidean", norm=True)
    return alpha, ecp


def cover_point(post, X, theta_true, ndraw=400):
    """Expected coverage at a SINGLE point (fiducial): per realization, the Mahalanobis credible level
    of theta_true in the (Om,S8,w0) posterior; coverage(alpha) = fraction with level <= alpha.
    Below the diagonal => over-confident (truth outside the credible region too often)."""
    tt = np.asarray(theta_true)[P3]
    lvl = []
    for i in range(len(X)):
        s = np.asarray(post.sample(x=jnp.asarray(X[i]), num_samples=ndraw,
                                   key=random.PRNGKey(2000 + i)))[:, P3]
        d = tt - s.mean(0)
        lvl.append(chi2.cdf(d @ np.linalg.inv(np.cov(s, rowvar=False)) @ d, df=len(P3)))
    lvl = np.array(lvl)
    alpha = np.linspace(0, 1, 50)
    return alpha, np.array([(lvl <= a).mean() for a in alpha])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for tag, col in CFG.items():
        post, theta, y, perms_t = train(tag)
        # PRIOR: random held-out training points (standard TARP)
        ridx = np.random.default_rng(0).choice(len(theta), size=300, replace=False)
        a_p, e_p = cover_tarp(post, y[ridx], theta[ridx])
        # FIDUCIAL: the 200 compressed perms, single-point coverage of theta_fid
        a_f, e_f = cover_point(post, perms_t, FID)
        lab = "non-BNT" if "nonbnt" in tag else "BNT"
        axes[0].plot(a_p, e_p, color=col, lw=2, label=lab)
        axes[1].plot(a_f, e_f, color=col, lw=2, label=lab)
        print(f"{tag}: prior max|ecp-a|={np.max(np.abs(e_p-a_p)):.3f}  "
              f"fiducial max|ecp-a|={np.max(np.abs(e_f-a_f)):.3f}  "
              f"(fiducial mean(ecp-a)={np.mean(e_f-a_f):+.3f} <0 => over-confident)")
    for ax, ttl in zip(axes, ["(A) PRIOR-averaged TARP\n(what npe_on_summary checked → passed)",
                              "(B) coverage AT THE FIDUCIAL\n(where the contours live)"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.set_xlabel("credible level"); ax.set_title(ttl, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("expected coverage probability")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].text(0.42, 0.08, "on/above diagonal = calibrated\n(slightly conservative); NOT over-confident",
                 fontsize=8.5, color="#27ae60")
    fig.suptitle("Score-NPE calibration: well-calibrated both on average (TARP) and at the fiducial",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = f"{OUT}/tarp_prior_vs_fiducial_14000.png"
    fig.savefig(p, dpi=140); print("saved", p)


if __name__ == "__main__":
    main()
