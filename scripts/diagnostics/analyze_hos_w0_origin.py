"""Where does the HOS w0 degeneracy direction come from? Decompose the fixed-bin l1 (detail scales)
response by SNR (structure type). The Om-w0 posterior correlation sign = -sign(F_Om,w0), and
F_Om,w0 ~ sum_bin R_Om(bin) R_w0(bin)/var(bin). Bins where R_Om and R_w0 are ANTI-aligned push
toward the (flipped) positive Om-w0; aligned bins push PS-like negative. We plot that per-SNR
contribution and run a per-SNR-region Fisher."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load("/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fixedbin_l1/fixedbin_l1_full.npz")
G, F, par, snr = D["G"], D["F"], D["gparams"], D["snr"]   # G(n,4,5,40) F(m,4,5,40)
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fixedbin_l1"
SC = [1, 2, 3]; PN = ["Om", "S8", "w0", "H0", "ns", "Ob"]
Gd, Fd = G[:, :, SC, :], F[:, :, SC, :]                  # detail scales

def responses(Gx):
    """dD/dtheta for every feature via lstsq; returns dict param->R over flattened features."""
    flat = Gx.reshape(Gx.shape[0], -1)
    X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
    coef, *_ = np.linalg.lstsq(X, flat, rcond=None)
    return {PN[i]: coef[1 + i] for i in range(6)}, flat

def fisher_corr(Gx, Fx):
    flat_g = Gx.reshape(Gx.shape[0], -1); flat_f = Fx.reshape(Fx.shape[0], -1)
    v = flat_f.var(0); keep = v > v.max() * 1e-8
    flat_g, flat_f = flat_g[:, keep], flat_f[:, keep]
    nf, nd = flat_f.shape
    if nd >= nf - 2: return None, nd
    X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
    coef, *_ = np.linalg.lstsq(X, flat_g, rcond=None); J = coef[1:].T
    C = np.cov(flat_f, rowvar=False); Cinv = np.linalg.inv(C) * ((nf - nd - 2) / (nf - 1))
    cov = np.linalg.inv(J.T @ Cinv @ J); d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d), nd

def rebin_snr(Gx, f):  # rebin the last axis (SNR) by f
    n = Gx.shape[-1] // f
    return Gx[..., :n * f].reshape(*Gx.shape[:-1], n, f).mean(-1)

# ---- per-SNR contribution to F_Om,w0 (diagonal approx), summed over tomo+detail scale -----
R, _ = responses(Gd)
var = Fd.reshape(Fd.shape[0], -1).var(0).reshape(4, len(SC), 40)
good = var > var.max() * 1e-8                            # mask empty (zero-variance) SNR bins
Rom = R["Om"].reshape(4, len(SC), 40); Rw0 = R["w0"].reshape(4, len(SC), 40); Rs8 = R["S8"].reshape(4, len(SC), 40)
safe = np.where(good, var, np.inf)                       # -> contribution 0 where empty
whiten = lambda Rx: np.where(good, Rx / np.sqrt(safe), 0.0)
contrib = np.where(good, Rom * Rw0 / safe, 0.0)          # diagonal F_Om,w0 per (tomo,scale,snr)
contrib_snr = contrib.sum((0, 1))                        # sum over tomo+scale -> per SNR
print("Per-SNR diagonal F_Om,w0 contribution (NEGATIVE => pushes toward flipped +Om-w0 degeneracy):")
print(f"  total sum = {contrib_snr.sum():+.3e}  (=> posterior Om-w0 sign ~ {-np.sign(contrib_snr.sum()):+.0f})")
for lo, hi, name in [(-13, -2, "troughs/voids"), (-2, 0, "neg bulk"), (0, 2, "pos bulk"), (2, 5, "peaks"), (5, 13, "rare peaks")]:
    m = (snr >= lo) & (snr < hi)
    print(f"  SNR[{lo:+3d},{hi:+3d}] {name:14s}: {contrib_snr[m].sum():+.3e}")

# ---- per-SNR-region full 6-param Fisher Om-w0 (rebin SNR x2 so covariances invert) ----
Gr, Fr = rebin_snr(Gd, 2), rebin_snr(Fd, 2); snr_r = rebin_snr(snr[None], 2)[0]
print("\nPer-SNR-region full Fisher (detail scales, all tomo, SNR rebinned x2):  Om-w0 / S8-w0 / Om-S8")
for lo, hi, name in [(-13, 13, "ALL"), (-13, -2, "troughs/voids"), (-2, 0, "neg bulk"),
                     (0, 2, "pos bulk"), (2, 5, "peaks"), (5, 13, "rare peaks")]:
    m = (snr_r >= lo) & (snr_r < hi)
    corr, nd = fisher_corr(Gr[:, :, :, m], Fr[:, :, :, m])
    if corr is None: print(f"  {name:14s} (nfeat {nd:3d}) SKIP"); continue
    print(f"  {name:14s} (nfeat {nd:3d}):  {corr[0,2]:+.2f}   {corr[1,2]:+.2f}   {corr[0,1]:+.2f}")

# ---- figure: whitened responses + the F_Om,w0 contribution vs SNR ----
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for nm, Rx, c in [("dL1/dOm", whiten(Rom), "#2980b9"), ("dL1/dS8", whiten(Rs8), "#27ae60"), ("dL1/dw0", whiten(Rw0), "#c0392b")]:
    a1.plot(snr, Rx.sum((0, 1)), label=nm, color=c, lw=1.8)
a1.axhline(0, color="k", lw=.6); a1.legend(); a1.set_ylabel("whitened response (sum tomo+scale)")
a1.set_title("l1 detail-scale parameter responses vs SNR (which structures carry each parameter)")
a2.bar(snr, contrib_snr, width=(snr[1]-snr[0])*0.9,
       color=np.where(contrib_snr < 0, "#c0392b", "#7f8c8d"))
a2.axhline(0, color="k", lw=.6); a2.set_xlabel("SNR (negative=voids, positive=peaks)")
a2.set_ylabel(r"$F_{\Omega_m,w_0}$ per SNR  (red<0 = flips to +degeneracy)")
a2.set_title("Where the w0 degeneracy direction is set")
fig.tight_layout(); fig.savefig(f"{OUT}/hos_w0_origin.png", dpi=140); print(f"\nwrote {OUT}/hos_w0_origin.png")
