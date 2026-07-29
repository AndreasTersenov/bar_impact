"""Clean full-sky comparison, ALL within the paper's own healpy pipeline.
 - paper l100  : the published _npe file in outputs/samples (reference)
 - new  l100   : healpy rerun at the paper cut (reproducibility check)
 - new  l30    : healpy rerun extending to low-ell (the true full-sky low-ell gain)
Per fiducial (null, baryon): triangle of the three; report sigma(S8) ladder."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
from getdist import plots, MCSamples

PAPER = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples"
CTRL = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fullsky_baseline/samples"
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fullsky_baseline"
NAMES = ["Om", "S8", "w0"]; LABELS = [r"\Omega_m", "S_8", "w_0"]
truth = {"Om": 0.26, "S8": 0.84, "w0": -1.0}   # CosmoGrid fiducial

def fname(fid, lo):
    return (f"posterior_samples_ps_auto_cross_nobaryons_vs_{fid}_bins1234_"
            f"l{lo}-1024_r10_noisy_s0.26_npe.npy")

def mcs(path, lab):
    s = np.load(path)[:, [0, 1, 2]]
    return MCSamples(samples=s, names=NAMES, labels=LABELS, label=lab), s

for fid, title in [("nobaryons", "null (nobaryons vs nobaryons)"),
                   ("baryonified", "baryon (nobaryons vs baryonified)")]:
    runs = [
        (f"{PAPER}/{fname(fid,100)}", "l100 paper (published)", "#000000"),
        (f"{CTRL}/{fname(fid,100)}", "l100 new (healpy rerun)", "#27ae60"),
        (f"{CTRL}/{fname(fid,30)}",  "l30 new (healpy, low-l)", "#c0392b"),
    ]
    samps, cols = [], []
    print(f"\n=== {fid} ===")
    for path, lab, col in runs:
        if not os.path.exists(path):
            print(f"  MISSING {path}"); continue
        m, s = mcs(path, lab); samps.append(m); cols.append(col)
        print(f"  {lab:26s} S8={s[:,1].mean():.4f}±{s[:,1].std():.4f}  "
              f"Om={s[:,0].mean():.4f}±{s[:,0].std():.4f}  w0={s[:,2].mean():.4f}±{s[:,2].std():.4f}")
    if len(samps) < 2:
        print("  not enough samples to plot yet"); continue
    g = plots.get_subplot_plotter(width_inch=7)
    g.settings.alpha_filled_add = 0.5
    g.triangle_plot(samps, NAMES, filled=True, contour_colors=cols,
                    markers=truth, legend_loc="upper right")
    g.fig.suptitle(f"Full-sky, healpy pipeline only — {title}", y=1.02, fontsize=12)
    out = f"{OUT}/triangle_fullsky_healpy_control_{fid}.png"
    g.export(out); print("  wrote", out)
