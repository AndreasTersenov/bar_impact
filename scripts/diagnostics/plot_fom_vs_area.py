#!/usr/bin/env python3
"""FoM3(Om, S8, w0) vs mask area — PS / peaks / L1, submean, no scale cuts.

Single-panel standalone version of panel (b) of plot_scaling_vs_area.py, made for the
referee response rather than the paper. Same inputs, same seed-averaging, same QA cut;
only the presentation differs — one panel, decluttered axes, per-point seed error bars.

Run with the jaxili interpreter:
  /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python \
      scripts/diagnostics/plot_fom_vs_area.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator
from scipy.stats import linregress

_AA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "styles", "paper_v1.mplstyle")
if os.path.exists(_AA):
    plt.style.use(_AA)
else:
    print(f"[warn] A&A style not found at {_AA} — using matplotlib defaults")

D = "outputs/samples"
PSD = "outputs/baryon_tension/ps_submean_l37/posteriors"
P3 = [0, 1, 2]  # Om, S8, w0

HOS = [2001, 5001, 10001, 14001, 28001, 35001]
PS = [2000, 5000, 10000, 14000, 28000, 35000]


def met(f):
    a = np.load(f)
    c = np.cov(a[:, P3], rowvar=False)
    return np.sqrt(c[1, 1]), 1.0 / np.sqrt(np.linalg.det(c))


def avg(files):
    """Seed-average. ValueError/OSError catch disk-failure-damaged .npy (numpy reads a
    mangled header as pickled data); skips are reported so a thinned average is visible."""
    S, F, skipped = [], [], 0
    for f in files:
        try:
            s, fm = met(f)
        except (FileNotFoundError, IndexError, ValueError, OSError):
            skipped += 1
            continue
        if s < 0.08:                      # drop prior-collapsed seeds
            S.append(s)
            F.append(fm)
    if skipped:
        print(f"    [skip] {skipped}/{len(files)} unreadable (disk-failure damage)")
    if not S:
        return np.nan, np.nan, 0
    return np.mean(F), np.std(F), len(S)


def hf(prefix, A):
    stem = (f"{D}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_scales1234"
            f"_noisy_s0.26_masked_{A}sqdeg_submean_new_normalization")
    return [f"{stem}_npe.npy"] + [f"{stem}_run{r}_npe.npy" for r in (2, 3, 4, 5)]


def pf(A):
    return glob.glob(f"{PSD}/mask_{A:05d}/null/posterior_samples_ps_auto_cross_"
                     f"nobaryons_vs_nobaryons_bins1234_l37-1020_r10_masked_{A}sqdeg"
                     f"_apod2.0_master_submean_noisy_s0.26*.npy")


# Okabe-Ito, matching plot_scaling_vs_area.py so the two figures stay consistent.
SERIES = [
    ("Power spectrum", PS,  [avg(pf(A)) for A in PS],          "#0072B2", "-",  "o"),
    ("Peak counts",    HOS, [avg(hf("pc_", A)) for A in HOS],  "#D55E00", "--", "s"),
    ("L1 norm",        HOS, [avg(hf("", A)) for A in HOS],     "#009E73", "-.", "^"),
]

# One set of stroke weights across the line figures. Same values as plot_scaling_vs_area.py
# and plot_nsigma_vs_area.py so the family reads as one figure set.
LW, MS, ELW = 2.2, 7.0, 1.6
CAPSIZE, CAPTHICK = 6.0, 2.0

W = 7.0    # submitted-style canvas; paper_v1 fonts are ~2x the A&A ones
fig, ax = plt.subplots(figsize=(W, W * 0.95))

ps_anchor = None
for name, A, m, c, ls, mk in SERIES:
    A = np.array(A, float)
    fom = np.array([x[0] for x in m])
    ef = np.array([x[1] for x in m])
    slope = linregress(np.log(A), np.log(fom)).slope
    # Stroke weights shared with plot_scaling_vs_area.py and plot_nsigma_vs_area.py, which were
    # thickened first; this figure was left at the old lw 1.3 / ms 4.0 / elinewidth 0.9 /
    # capsize 2 and read as a different, fainter figure beside them.
    # Do NOT add markeredgewidth here: it overrides capthick and silently erases every cap.
    ax.errorbar(A, fom, yerr=ef, color=c, ls=ls, marker=mk, ms=MS, lw=LW,
                capsize=CAPSIZE, capthick=CAPTHICK, elinewidth=ELW,
                label=rf"{name} ($\alpha={slope:+.2f}$)")
    if name == "Power spectrum":
        ps_anchor = float(fom[list(A).index(14000.0)])

# Ideal-scaling guide, anchored ON the power-spectrum curve at 14000 deg^2 rather
# than at a fixed offset below it. Pinning it to the data means the line crosses PS
# at the anchor and fans away at both ends, so the gap IS the slope difference
# (measured alpha ~ +1.3 vs the ideal +1.5) instead of an arbitrary vertical offset.
# Anchor is read from the series, so it stays correct if the inputs change.
#
# GUIDE_SCALE drops the guide below the data by a documented factor. The default 1.0 is
# the pinned version above. GUIDE_SCALE=0.24 reproduces the earlier "lowanchor" look,
# where the line sits clear of every series and reads purely as a slope reference — that
# variant existed only as an orphan .pdf/.png with no sidecars, so it was unpublishable;
# it is now a reproducible option. The factor is COSMETIC: it shifts the line vertically
# and changes nothing about the slope, which is the only thing the guide asserts.
GUIDE_SCALE = float(os.environ.get("GUIDE_SCALE", "1.0"))

Aref = np.array([1.7e3, 4.2e4])
ax.plot(Aref, GUIDE_SCALE * ps_anchor * (Aref / 14000.) ** 1.5, color="0.45", ls=":",
        lw=1.0, zorder=0, label=r"$A^{+3/2}$ (ideal)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"mask area $[\mathrm{deg}^2]$")
ax.set_ylabel(r"$\mathrm{FoM}_3\,(\Omega_\mathrm{m},\,\sigma_8,\,w_0)$")

# --- decluttered axes -------------------------------------------------------
# The data spans only ~1.25 decades (2000-35000), so matplotlib's default log
# locator puts majors at 1e3/1e4 and then labels minors to fill the gap — which is
# what made the old ticks collide. Label the surveyed areas explicitly instead, in
# plain numbers, and silence every minor label.
ax.set_xlim(1.6e3, 4.6e4)
ax.xaxis.set_major_locator(FixedLocator([2000, 5000, 10000, 20000, 40000]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
ax.xaxis.set_minor_formatter(NullFormatter())

# y: decade majors only, minors unlabelled
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))
ax.yaxis.set_minor_formatter(NullFormatter())

ax.tick_params(which="major", length=4)
ax.tick_params(which="minor", length=2)
ax.legend(frameon=False, loc="upper left",
          borderaxespad=0.4, labelspacing=0.35)

fig.tight_layout(pad=0.3)
out = "outputs/plots/submean_masked_peaks/fom_vs_area_all_stats"
if GUIDE_SCALE != 1.0:
    out += "_lowanchor"      # never overwrite the pinned-guide version
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)
print("wrote", out + ".pdf / .png")

# ---- provenance -----------------------------------------------------------
# Standing rule (docs/HANDOFF_JZ_PAPER_FIGURES.md section 0). n_seeds is the critical
# column: disk damage means each point averages a different subset of runs than it did
# pre-crash, AND hf() enumerates only runs 1-5 by construction, so a point can be capped
# at 5 even at footprints where 10 posteriors exist. Both must be legible from the CSV.
import csv as _csv, json as _json, subprocess as _sub, datetime as _dt

with open(out + "_values.csv", "w", newline="") as _fh:
    _w = _csv.writer(_fh)
    _w.writerow(["statistic", "area_sqdeg", "fom3", "fom3_err", "n_seeds", "errbar_kind"])
    for _name, _A, _m, _c, _ls, _mk in SERIES:
        for _a, _x in zip(_A, _m):
            _w.writerow([_name, int(_a), f"{_x[0]:.6g}", f"{_x[1]:.6g}", int(_x[2]), "std"])


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "unavailable"


try:
    _commit = _sub.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=os.path.dirname(os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__)))),
                                stderr=_sub.DEVNULL, text=True).strip()
except Exception:
    _commit = "unknown"

_json.dump({
    "figure": os.path.basename(out),
    "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    "git_commit": _commit,
    "quantity": "FoM_3 = 1/sqrt(det C) over the (Omega_m, S8, w0) sub-covariance",
    "guide_scale": GUIDE_SCALE,
    "guide_note": ("A^+3/2 reference line = GUIDE_SCALE x (measured PS FoM3 at 14000) x "
                   "(A/14000)^1.5. GUIDE_SCALE is a COSMETIC vertical offset; only the "
                   "slope of this line carries meaning."),
    "scales_included": {
        "power_spectrum": "monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10",
        "peaks_l1": "wavelet scales1234 (four detail scales; coarse/mass-sheet excluded), "
                    "submean, new_normalization, noisy sigma_e=0.26",
    },
    "errbar": "std over seeds",
    "arm": "NULL only (nobaryons vs nobaryons) — constraining power, not bias",
    "cut": "full resolution: PS lmin=37 lmax=1020 r10; HOS scales1234 submean",
    "versions": {m: _ver(m) for m in ("numpy", "scipy", "matplotlib")},
    "mplstyle": _AA if os.path.exists(_AA) else "matplotlib defaults",
    "caveats": [
        "styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this "
            "figure sits beside the figures kept verbatim from it.",
        "Damaged posteriors are skipped (see [skip] lines in the run log), so n_seeds "
        "differs from the original campaign and each point averages a different subset.",
        "KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks "
        "and L1 are capped at 5 seeds even where 10 posteriors exist. Widening it would "
        "change the plotted values; n_seeds records what was actually used.",
        "The A^+3/2 guide is anchored on the MEASURED PS value at 14000 deg^2, read from the "
        "series rather than hardcoded, so it stays correct if the inputs change.",
        "FULL RESOLUTION — the regime where all three statistics are baryon-BIASED. This is "
        "constraining power available in principle, not at a baryon-safe cut.",
    ],
}, open(out + "_provenance.json", "w"), indent=2)
print("wrote", out + "_values.csv / _provenance.json")

print(f"\n  {'statistic':16s}" + "".join(f"{a:>11d}" for a in PS))
for name, A, m, c, ls, mk in SERIES:
    print(f"  {name:16s}" + "".join(f"{x[0]:>11.3g}" for x in m))
    print(f"  {'  +/- (seed)':16s}" + "".join(f"{x[1]:>11.3g}" for x in m))
    print(f"  {'  nseed':16s}" + "".join(f"{x[2]:>11d}" for x in m))
