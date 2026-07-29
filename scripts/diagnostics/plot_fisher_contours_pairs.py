"""Fisher confidence ellipses for arbitrary parameter pairs (here Omega_m-w0 and S8-w0),
per ell band, from the saved gate covariances. Rows = parameter pairs, columns = ell bands."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "outputs", "diagnostics", "fisher_gate")
with open(os.path.join(OUT, "fisher_gate_summary.json")) as f:
    R = json.load(f)["results"]

PIDX = {"Omega_m": 0, "S8": 1, "w0": 2, "H0": 3, "ns": 4, "Omega_b": 5}
CENTRE = {"Omega_m": 0.30, "S8": 0.90, "w0": -1.00}
TEX = {"Omega_m": r"$\Omega_m$", "S8": r"$S_8$", "w0": r"$w_0$"}
PAIRS = [("Omega_m", "w0"), ("S8", "w0")]
BANDS = ["30-100", "30-1024"]
ARMS = ["fullsky", "masked_raw", "masked_sub"]
STYLE = {"masked_raw": ("#c0392b", "masked (raw)"),
         "masked_sub": ("#2980b9", "masked (mean-subtracted)"),
         "fullsky": ("#555555", "full-sky")}
LEVELS = {1: 2.30, 2: 6.17}  # chi^2(2 dof): 68.3%, 95.4%


def draw(ax, cov2, centre, color, label):
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for nsig, chi2 in LEVELS.items():
        e = Ellipse(centre, 2 * np.sqrt(chi2 * vals[0]), 2 * np.sqrt(chi2 * vals[1]),
                    angle=angle, fill=(nsig == 1),
                    facecolor=color if nsig == 1 else "none", edgecolor=color,
                    alpha=0.25 if nsig == 1 else 1.0, lw=2,
                    label=label if nsig == 1 else None)
        ax.add_patch(e)


fig, axes = plt.subplots(len(PAIRS), len(BANDS), figsize=(13, 11))
for r, (px, py) in enumerate(PAIRS):
    ix, iy = PIDX[px], PIDX[py]
    centre = (CENTRE[px], CENTRE[py])
    for c, band in enumerate(BANDS):
        ax = axes[r, c]
        for arm in ARMS:
            res = R.get(f"{arm}|{band}", {})
            if "param_cov" not in res:
                continue
            cov = np.array(res["param_cov"])
            cov2 = cov[np.ix_([ix, iy], [ix, iy])]
            color, label = STYLE[arm]
            draw(ax, cov2, centre, color, label)
        ax.plot(*centre, "k+", ms=10)
        ax.set_xlabel(TEX[px]); ax.set_ylabel(TEX[py])
        ax.set_title(f"{TEX[px]}-{TEX[py]}   ($\\ell$ {band})")
        ax.margins(0.3); ax.autoscale_view()
        if r == 0 and c == len(BANDS) - 1:
            ax.legend(fontsize=9, loc="upper right")

fig.suptitle("Fisher contours vs $w_0$: monopole subtraction removes the false low-$\\ell$ "
             "tightening\n(filled = 1$\\sigma$, open = 2$\\sigma$; 300 grid + 200 fid)", fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, "fisher_gate_contours_w0.png")
fig.savefig(path, dpi=140)
print("wrote", path)
