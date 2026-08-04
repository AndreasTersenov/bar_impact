"""Plot the Fisher-gate result: how masked raw / masked submean / full-sky constraints
compare per ell band. Highlights (a) the raw low-ell anomaly (masked tighter than full-sky)
and (b) its removal by mean subtraction, with the 100-1024 control unchanged."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "outputs", "diagnostics", "fisher_gate")
with open(os.path.join(OUT, "fisher_gate_summary.json")) as f:
    S = json.load(f)
R = S["results"]
bands = ["30-100", "100-1024", "30-1024"]
arms = ["masked_raw", "masked_sub", "fullsky"]
colors = {"masked_raw": "#c0392b", "masked_sub": "#2980b9", "fullsky": "#7f8c8d"}
labels = {"masked_raw": "masked (raw)", "masked_sub": "masked (mean-subtracted)", "fullsky": "full-sky"}

sig = {a: [R[f"{a}|{b}"]["sigmas"][1] for b in bands] for a in arms}   # sigma(S8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: sigma(S8) grouped bars (log scale)
x = np.arange(len(bands)); w = 0.26
for i, a in enumerate(arms):
    ax1.bar(x + (i - 1) * w, sig[a], w, label=labels[a], color=colors[a])
ax1.set_yscale("log")
ax1.set_xticks(x); ax1.set_xticklabels([f"$\\ell$ {b}" for b in bands])
ax1.set_ylabel(r"$\sigma(\sigma_8)$  (Fisher)")
ax1.set_title("Constraint per band — smaller = tighter")
ax1.legend(fontsize=9)
ax1.grid(axis="y", ls=":", alpha=0.5)

# Panel 2: anomaly ratio sigma_masked / sigma_fullsky (physical region > 1)
ax2.axhspan(0, 1, color="#c0392b", alpha=0.08)
ax2.axhline(1.0, color="k", lw=1)
for a in ["masked_raw", "masked_sub"]:
    ratio = [sig[a][k] / sig["fullsky"][k] for k in range(len(bands))]
    ax2.plot(x, ratio, "o-", color=colors[a], label=labels[a], ms=9, lw=2)
    for k, r in enumerate(ratio):
        ax2.annotate(f"{r:.2f}", (x[k], r), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels([f"$\\ell$ {b}" for b in bands])
ax2.set_ylabel(r"$\sigma(\sigma_8)_{\rm masked} / \sigma(\sigma_8)_{\rm full\text{-}sky}$")
ax2.set_title("Below 1 = impossible 'masking tightens' anomaly")
ax2.text(0.02, 0.06, "shaded: unphysical (masked tighter than full-sky)",
         transform=ax2.transAxes, fontsize=8, color="#c0392b")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(axis="y", ls=":", alpha=0.5)

fig.suptitle("Fisher gate: monopole subtraction removes the false low-$\\ell$ information "
             "(300 grid + 200 fid)", fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, "fisher_gate_result.png")
fig.savefig(path, dpi=140)
print("wrote", path)
