"""getdist triangle plots (Omega_m, S8, w0) for the mean-subtracted masked-PS NPE posteriors.
Per mask: null (nobaryons) vs baryon (baryonified). Plus an all-masks overlay (null case)."""
import os
import numpy as np
from getdist import plots, MCSamples

SAMP = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/npe_prod/triangles"
os.makedirs(OUT, exist_ok=True)
MASKS = [2000, 5000, 10000, 14000, 28000, 35000]
NAMES = ["Om", "S8", "w0"]
LABELS = [r"\Omega_m", r"\sigma_8", "w_0"]


def mcs(mask, fid, label):
    f = (f"{SAMP}/posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_"
         f"l37-1024_r10_masked_{mask}sqdeg_apod2.0_master_submean_noisy_s0.26.npy")
    s = np.load(f)[:, [0, 1, 2]]
    return MCSamples(samples=s, names=NAMES, labels=LABELS, label=label)


# fiducial truth ~ the well-constrained, unbiased 35000 null posterior mean
t = np.load(f"{SAMP}/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
            f"l37-1024_r10_masked_35000sqdeg_apod2.0_master_submean_noisy_s0.26.npy")[:, [0, 1, 2]]
truth = {"Om": float(t[:, 0].mean()), "S8": float(t[:, 1].mean()), "w0": float(t[:, 2].mean())}
print("fiducial (~truth, from 35000 null):", {k: round(v, 3) for k, v in truth.items()})

# ---- per-mask triangles: null vs baryon ----------------------------------
for m in MASKS:
    nb = mcs(m, "nobaryons", f"{m} deg$^2$ — nobaryons (null)")
    by = mcs(m, "baryonified", f"{m} deg$^2$ — baryonified")
    g = plots.get_subplot_plotter(width_inch=6.5)
    g.settings.alpha_filled_add = 0.6
    g.settings.title_limit_fontsize = 10
    g.triangle_plot([nb, by], NAMES, filled=True,
                    contour_colors=["#2980b9", "#c0392b"], markers=truth,
                    legend_loc="upper right")
    g.export(f"{OUT}/triangle_mask{m}.png")
    print("wrote triangle_mask", m)

# ---- all-masks overlay (null case) ---------------------------------------
allm = [mcs(m, "nobaryons", f"{m} deg$^2$") for m in MASKS]
g = plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(allm, NAMES, filled=False, markers=truth, legend_loc="upper right")
g.export(f"{OUT}/triangle_allmasks_null.png")
print("wrote triangle_allmasks_null")

# overlay for the baryon case too (so the bias-vs-area is visible)
allb = [mcs(m, "baryonified", f"{m} deg$^2$") for m in MASKS]
g = plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(allb, NAMES, filled=False, markers=truth, legend_loc="upper right")
g.export(f"{OUT}/triangle_allmasks_baryon.png")
print("wrote triangle_allmasks_baryon")
