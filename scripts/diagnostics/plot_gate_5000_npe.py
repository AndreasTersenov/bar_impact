"""NPE gate comparison, 5000 deg^2, nlb=4. Per fiducial (null, baryon):
 - paper l100        : published nlb=4-raw masked result (reference)
 - new l100 submean  : reproduces the paper above ell 100 (validation)
 - new l37 submean   : the recovered low-ell-inclusive result
 - full-sky l37      : healpy reference; masked MUST be looser than this
Truth markers at the CosmoGrid fiducial."""
import os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
from getdist import plots, MCSamples

SAMP = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples"
GATE = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/gate_nlb4/samples"
FS = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fullsky_baseline/samples"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/gate_nlb4"
NAMES = ["Om", "S8", "w0"]; LABELS = [r"\Omega_m", r"\sigma_8", "w_0"]
truth = {"Om": 0.26, "S8": 0.84, "w0": -1.0}


def mcs(path, lab):
    if not os.path.exists(path):
        print("  MISSING", path); return None
    s = np.load(path)[:, [0, 1, 2]]
    print(f"  {lab:24s} S8={s[:,1].mean():.4f}±{s[:,1].std():.4f}  "
          f"Om={s[:,0].mean():.4f}±{s[:,0].std():.4f}  w0={s[:,2].mean():.4f}±{s[:,2].std():.4f}")
    return MCSamples(samples=s, names=NAMES, labels=LABELS, label=lab)


for fid, title in [("nobaryons", "null (nobaryons vs nobaryons)"),
                   ("baryonified", "baryon (nobaryons vs baryonified)")]:
    print(f"\n=== {fid} ===")
    runs = [
        (f"{SAMP}/posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_l100-1024_r10_masked_5000sqdeg_apod2.0_master_noisy_s0.26.npy",
         "paper l100 (nlb=4 raw)", "#000000"),
        (f"{GATE}/posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_l100-1024_r10_masked_5000sqdeg_apod2.0_master_submean_noisy_s0.26.npy",
         "new l100 submean", "#27ae60"),
        (f"{GATE}/posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_l37-1024_r10_masked_5000sqdeg_apod2.0_master_submean_noisy_s0.26.npy",
         "new l37 submean (recovery)", "#c0392b"),
        (f"{FS}/posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_l37-1024_r10_noisy_s0.26_npe.npy",
         "full-sky l37 (reference)", "#8e44ad"),
    ]
    samps, cols = [], []
    for path, lab, col in runs:
        m = mcs(path, lab)
        if m is not None:
            samps.append(m); cols.append(col)
    if len(samps) < 2:
        print("  not enough to plot"); continue
    g = plots.get_subplot_plotter(width_inch=7.5)
    g.settings.alpha_filled_add = 0.5
    g.triangle_plot(samps, NAMES, filled=True, contour_colors=cols,
                    markers=truth, legend_loc="upper right")
    g.fig.suptitle(f"NPE gate — 5000 deg$^2$, nlb=4 — {title}", y=1.02, fontsize=12)
    out = f"{OUT}/gate_5000_npe_{fid}.png"
    g.export(out); print("  wrote", out)
