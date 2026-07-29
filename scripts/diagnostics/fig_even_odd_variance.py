"""Figure: (left) even/odd decomposition of the l1 dL1/dOm response about SNR=0; (right) Om-w0
degeneracy for PS / variance-moment / even-histogram / odd-histogram / full l1 -> the variance is
the ONLY PS-like piece; any PDF shape (even or odd) is flipped."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load("/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fixedbin_l1/fixedbin_l1_full.npz")
G, F, par, snr = D["G"], D["F"], D["gparams"], D["snr"]; SC = [1, 2, 3]
Gd, Fd = G[:, :, SC, :], F[:, :, SC, :]
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/moments_w0"

def fish(Gx, Fx):
    g = Gx.reshape(Gx.shape[0], -1); f = Fx.reshape(Fx.shape[0], -1)
    v = f.var(0); k = v > v.max() * 1e-8; g, f = g[:, k], f[:, k]; nf, nd = f.shape
    if nd >= nf - 2: return None
    X = np.column_stack([np.ones(len(par)), par - par.mean(0)]); c, *_ = np.linalg.lstsq(X, g, rcond=None); J = c[1:].T
    C = np.cov(f, rowvar=False); Ci = np.linalg.inv(C) * ((nf - nd - 2) / (nf - 1))
    cov = np.linalg.inv(J.T @ Ci @ J); d = np.sqrt(np.diag(cov)); return (cov / np.outer(d, d))[0, 2]
def rb(a, f): n = a.shape[-1] // f; return a[..., :n * f].reshape(*a.shape[:-1], n, f).mean(-1)

# even/odd histogram degeneracies (SNR>0 half carries all info)
even_G = (Gd + Gd[..., ::-1]) / 2; odd_G = (Gd - Gd[..., ::-1]) / 2
even_F = (Fd + Fd[..., ::-1]) / 2; odd_F = (Fd - Fd[..., ::-1]) / 2
h = slice(20, 40)
om_even = fish(rb(even_G[..., h], 2), rb(even_F[..., h], 2))
om_odd = fish(rb(odd_G[..., h], 2), rb(odd_F[..., h], 2))
om_full = fish(rb(Gd, 3), rb(Fd, 3))
PS, VAR = -0.80, -0.59   # from fisher_ps_vs_hos_degeneracy.py / analyze_moments_w0.py

# whitened dL1/dOm response and its even/odd parts
X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
co, *_ = np.linalg.lstsq(X, Gd.reshape(len(par), -1), rcond=None)
var = Fd.reshape(Fd.shape[0], -1).var(0).reshape(4, 3, 40); good = var > var.max() * 1e-8
Rom = np.where(good, co[1].reshape(4, 3, 40) / np.sqrt(np.where(good, var, np.inf)), 0).sum((0, 1))
ev = (Rom + Rom[::-1]) / 2; od = (Rom - Rom[::-1]) / 2

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(snr, Rom, color="0.6", lw=1.2, label=r"$dL1/d\Omega_m$ (full)")
a1.plot(snr, ev, color="#2980b9", lw=2.2, label="even (symmetric) part")
a1.plot(snr, od, color="#c0392b", lw=2.2, label="odd (antisymmetric) part")
a1.axhline(0, color="k", lw=.6); a1.axvline(0, color="k", lw=.4, ls=":")
a1.set_xlabel("SNR (<0 voids, >0 peaks)"); a1.set_ylabel(r"whitened $dL1/d\Omega_m$")
a1.legend(); a1.set_title("The response is a (mostly even) bulk$\\to$tails redistribution;\nthe sign change at SNR$\\approx$0 is the pivot, not a degeneracy switch")
labels = ["PS\n(2-pt)", "variance\nmoment", "even\nhistogram", "odd\nhistogram", "full l1\n(shape)"]
vals = [PS, VAR, om_even, om_odd, om_full]
cols = ["#7f8c8d", "#2980b9", "#c0392b", "#c0392b", "#c0392b"]
a2.bar(range(5), vals, color=cols); a2.axhline(0, color="k", lw=.8); a2.set_ylim(-1, 1)
for i, v in enumerate(vals): a2.text(i, v + (0.04 if v > 0 else -0.09), f"{v:+.2f}", ha="center", fontweight="bold")
a2.set_xticks(range(5)); a2.set_xticklabels(labels)
a2.set_ylabel(r"$\Omega_m$-$w_0$ correlation")
a2.set_title("Only the variance (2-pt-equivalent) is PS-like;\nany PDF shape — even or odd — is flipped")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_even_odd_variance.png", dpi=150)
print(f"even-hist Om-w0={om_even:+.2f}  odd-hist={om_odd:+.2f}  full={om_full:+.2f}")
print(f"wrote {OUT}/fig3_even_odd_variance.png")
