#!/usr/bin/env python3
"""Phase II/III money plot: BNT-580 / non-BNT-460 area ratio under every Fisher input choice,
grouped by what each choice means. Settles the ambiguity: the TRUE-information ratio (estimation-
noise-free, validated covariance, local J) is ~0.45-0.48; the 0.72 is a finite-200-sim penalty.
Numbers from fisher_local_jacobian.py / fisher_hybrid_cov.py (14000, masked nlb=4/lmax1535)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/diagnostics/fisher_cov/bnt_ratio_ladder_14000.png"
rows = [   # (label, ratio, group)
    ("raw sample cov (no de-bias)",        0.346, "biased"),
    ("global lstsq-over-prior J",          0.370, "biased"),
    ("analytic Gaussian cov",              0.484, "true"),
    ("hybrid k=3 (+ SSC/cNG)",             0.455, "true"),
    ("hybrid k=5 (+ SSC/cNG)",             0.439, "true"),
    ("local J + Hartlap (sim cov)",        0.502, "finite"),
    ("local J + Percival (sim cov)",       0.716, "finite"),
]
gcol = {"biased": "#c0392b", "true": "#27ae60", "finite": "#e67e22"}
gname = {"biased": "biased estimator\n(over-counts BNT)",
         "true": "TRUE information\n(estimation-noise-free)",
         "finite": "200-sim sample-cov\n(finite-N penalty)"}

fig, ax = plt.subplots(figsize=(9.2, 4.6))
y = list(range(len(rows)))[::-1]
for yi, (lab, r, g) in zip(y, rows):
    ax.barh(yi, r, color=gcol[g], height=0.62, zorder=3)
    ax.text(r + 0.008, yi, f"{r:.3f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.axvline(0.79, color="#2980b9", lw=2, ls="--", zorder=4)
ax.text(0.79, len(rows) - 0.4, "  NPE = 0.79", color="#2980b9", fontsize=10, va="top")
ax.axvspan(0.44, 0.50, color="#27ae60", alpha=0.10, zorder=0)
ax.set_xlim(0, 0.92); ax.set_xlabel("BNT-580 / non-BNT-460   1$\\sigma$ area ratio   (<1 = BNT tighter)")
ax.set_title("Proper Fisher: the BNT constraining-power advantage (14000 deg$^2$, auto+cross PS)\n"
             "true-information ratio $\\approx$0.45–0.48  →  BNT $\\sim$2$\\times$ tighter; NPE (0.79) under-extracts",
             fontsize=10.5)
# group legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=gcol[k], label=gname[k]) for k in ["biased", "true", "finite"]],
          loc="lower right", fontsize=8, frameon=True)
ax.grid(axis="x", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("saved", OUT)
