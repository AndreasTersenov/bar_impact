"""Draw the Fisher (Omega_m, S8) confidence ellipses for masked-raw / masked-submean /
full-sky, per ell band, from the saved gate covariances. Centered at the fiducial; all arms
share the centre so the comparison is purely about ellipse size/orientation."""
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
    S = json.load(f)
R = S["results"]

# Fiducial centre (Omega_m, S8) -- standard CosmoGrid fiducial.
CENTRE = (0.30, 0.90)
ARMS = ["fullsky", "masked_raw", "masked_sub"]
STYLE = {"masked_raw": ("#c0392b", "masked (raw)"),
         "masked_sub": ("#2980b9", "masked (mean-subtracted)"),
         "fullsky": ("#555555", "full-sky")}
# chi^2(2 dof): 68.3% -> 2.30, 95.4% -> 6.17
LEVELS = {1: 2.30, 2: 6.17}
BANDS = ["30-100", "30-1024"]


def ellipse(ax, cov2, color, label, lw=2):
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for nsig, chi2 in LEVELS.items():
        w = 2 * np.sqrt(chi2 * vals[0])
        h = 2 * np.sqrt(chi2 * vals[1])
        e = Ellipse(CENTRE, w, h, angle=angle, fill=(nsig == 1),
                    facecolor=color if nsig == 1 else "none",
                    edgecolor=color, alpha=0.25 if nsig == 1 else 1.0,
                    lw=lw, label=label if nsig == 1 else None)
        ax.add_patch(e)


fig, axes = plt.subplots(1, len(BANDS), figsize=(13, 6))
for ax, band in zip(axes, BANDS):
    for arm in ARMS:
        res = R.get(f"{arm}|{band}", {})
        if "param_cov" not in res:
            continue
        cov2 = np.array(res["param_cov"])[np.ix_([0, 1], [0, 1])]
        color, label = STYLE[arm]
        ellipse(ax, cov2, color, label)
    ax.plot(*CENTRE, "k+", ms=10)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\sigma_8$")
    ax.set_title(f"$\\ell$ {band}")
    ax.legend(fontsize=9, loc="upper right")
    ax.autoscale_view()
    ax.margins(0.25)

fig.suptitle("Fisher (Omega_m, S8) contours: monopole subtraction removes the false low-$\\ell$ "
             "tightening\n(filled = 1sigma, open = 2sigma; 300 grid + 200 fid)", fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, "fisher_gate_contours.png")
fig.savefig(path, dpi=140)
print("wrote", path)
