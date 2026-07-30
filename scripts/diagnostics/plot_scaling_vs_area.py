#!/usr/bin/env python3
"""Constraining-power scaling vs mask area — PS / peaks / L1, submean, no scale cuts.
Publication figure (A&A profile). σ(S8) and FoM3 of the nobaryons posterior vs masked area, seed-averaged.
PS = ℓmin=37 low-ℓ-recovered range (l37-1020); HOS = scales1234 (coarse dropped). See docs/scaling_vs_area_submean.md."""
import numpy as np, glob, matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.stats import linregress

_AA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "styles", "aa.mplstyle")
if os.path.exists(_AA):
    plt.style.use(_AA)
else:
    print(f"[warn] A&A style not found at {_AA} — using matplotlib defaults")

D = "outputs/samples"
PSD = "outputs/baryon_tension/ps_submean_l37/posteriors"
P3 = [0, 1, 2]  # Om, S8, w0

def met(f):
    a = np.load(f); c = np.cov(a[:, P3], rowvar=False)
    return np.sqrt(c[1, 1]), 1.0 / np.sqrt(np.linalg.det(c))

def avg(files):
    # ValueError/OSError added after the disk failure: a stripe-damaged .npy makes
    # numpy read the header as pickled data and raise ValueError, which the original
    # (FileNotFoundError, IndexError) clause let through. Skipping them keeps the seed
    # average honest; the skip count is printed so a reduced seed count is visible
    # rather than silent.
    S, F, skipped = [], [], 0
    for f in files:
        try:
            s, fm = met(f)
        except (FileNotFoundError, IndexError, ValueError, OSError):
            skipped += 1
            continue
        if s < 0.08: S.append(s); F.append(fm)        # drop any residual prior-collapsed seed
    if skipped:
        print(f"    [skip] {skipped}/{len(files)} unreadable (disk-failure damage)")
    # index 3 is the seed scatter of FoM3, so panel (b) can carry error bars like panel (a).
    return ((np.mean(S), np.std(S), np.mean(F), np.std(F), len(S))
            if S else (np.nan, np.nan, np.nan, np.nan, 0))

HOS = [2001, 5001, 10001, 14001, 28001, 35001]
PS  = [2000, 5000, 10000, 14000, 28000, 35000]

def hf(prefix, A):
    base = f"{D}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_scales1234_noisy_s0.26_masked_{A}sqdeg_submean_new_normalization_npe.npy"
    runs = [f"{D}/posterior_samples_{prefix}nobaryons_vs_nobaryons_bins1234_scales1234_noisy_s0.26_masked_{A}sqdeg_submean_new_normalization_run{r}_npe.npy" for r in (2, 3, 4, 5)]
    return [base] + runs

def pf(A):
    return glob.glob(f"{PSD}/mask_{A:05d}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l37-1020_r10_masked_{A}sqdeg_apod2.0_master_submean_noisy_s0.26*.npy")

# (label, areas, metrics, Okabe-Ito color, linestyle, marker)
SERIES = [
    ("Power spectrum", PS,  [avg(pf(A))      for A in PS],  "#0072B2", "-",  "o"),
    ("Peak counts",    HOS, [avg(hf("pc_", A)) for A in HOS], "#D55E00", "--", "s"),
    ("L1 norm",        HOS, [avg(hf("",   A)) for A in HOS], "#009E73", "-.", "^"),
]

W = 7.087  # A&A double-column, inches
fig, ax = plt.subplots(1, 2, figsize=(W, 0.46 * W))

for j, (name, A, m, c, ls, mk) in enumerate(SERIES):
    A = np.array(A, float)
    s8  = np.array([x[0] for x in m]); es = np.array([x[1] for x in m])
    fom = np.array([x[2] for x in m]); ef = np.array([x[3] for x in m])
    ss = linregress(np.log(A), np.log(s8)).slope
    sf = linregress(np.log(A), np.log(fom)).slope
    ax[0].errorbar(A, s8, yerr=es, color=c, ls=ls, marker=mk, ms=4.5, lw=1.4,
                   capsize=2, elinewidth=0.9, label=rf"{name} ($\alpha={ss:+.2f}$)")
    ax[1].errorbar(A, fom, yerr=ef, color=c, ls=ls, marker=mk, ms=4.5, lw=1.4,
                   capsize=2, elinewidth=0.9, label=rf"{name} ($\alpha={sf:+.2f}$)")

# reference slopes (anchored at 14000), neutral gray dotted
Aref = np.array([1.7e3, 4.2e4])
ax[0].plot(Aref, 0.0135 * (Aref / 14000.) ** -0.5, color="0.45", ls=":", lw=1.0, zorder=0, label=r"$A^{-1/2}$")
ax[1].plot(Aref, 1.05e5 * (Aref / 14000.) ** +1.5, color="0.45", ls=":", lw=1.0, zorder=0, label=r"$A^{+3/2}$")

ax[0].set_ylabel(r"$\sigma(S_8)$")
ax[1].set_ylabel(r"$\mathrm{FoM}_3\,(\Omega_\mathrm{m},S_8,w_0)$")
for k, a in enumerate(ax):
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"mask area $\,[\mathrm{deg}^2]$")
    a.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    a.tick_params(which="both", direction="in", top=True, right=True)
    a.legend(frameon=False, fontsize=8, handlelength=2.2)
    a.text(0.04, 0.06, f"({'ab'[k]})", transform=a.transAxes, fontsize=9, va="bottom")

fig.tight_layout(pad=0.4, w_pad=1.2)
out = "outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("wrote", out + ".pdf / .png")

# ---- provenance -----------------------------------------------------------
# Standing rule (docs/HANDOFF_JZ_PAPER_FIGURES.md section 0). The fitted slopes are the
# figure's headline, so they are written down rather than read off the legend.
import csv as _csv, json as _json, subprocess as _sub, datetime as _dt

_slopes = {}
with open(out + "_values.csv", "w", newline="") as _fh:
    _w = _csv.writer(_fh)
    _w.writerow(["statistic", "area_sqdeg", "sigma_S8", "sigma_S8_err",
                 "fom3", "fom3_err", "n_seeds", "errbar_kind"])
    for _name, _A, _m, _c, _ls, _mk in SERIES:
        _Aa = np.array(_A, float)
        _s8 = np.array([x[0] for x in _m]); _fm = np.array([x[2] for x in _m])
        _slopes[_name] = {
            "sigma_S8_loglog_slope": float(linregress(np.log(_Aa), np.log(_s8)).slope),
            "fom3_loglog_slope": float(linregress(np.log(_Aa), np.log(_fm)).slope),
        }
        for _a, _x in zip(_A, _m):
            _w.writerow([_name, int(_a), f"{_x[0]:.6g}", f"{_x[1]:.6g}",
                         f"{_x[2]:.6g}", f"{_x[3]:.6g}", int(_x[4]), "std"])


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
    "panels": {"a": "sigma(S8) vs area", "b": "FoM_3 vs area"},
    "fitted_loglog_slopes": _slopes,
    "errbar": "std over seeds",
    "arm": "NULL only (nobaryons vs nobaryons) — constraining power, not bias",
    "scales_included": {
        "power_spectrum": "monopole-subtracted MASTER, lmin=37, lmax=1020, rebin=10",
        "peaks_l1": "wavelet scales1234 (four detail scales; coarse/mass-sheet excluded), "
                    "submean, new_normalization, noisy sigma_e=0.26",
    },
    "presentation_todo": ("If this figure is used anywhere (paper or referee response) the "
                          "x-axis tick labels need decluttering first — the log locator "
                          "currently overlaps them. plot_fom_vs_area.py already solves this "
                          "with a FixedLocator at 2k/5k/10k/20k/40k plus NullFormatter on the "
                          "minors; port that here."),
    "cut": "full resolution: PS lmin=37 lmax=1020 r10; HOS scales1234 submean",
    "versions": {m: _ver(m) for m in ("numpy", "scipy", "matplotlib")},
    "mplstyle": _AA if os.path.exists(_AA) else "matplotlib defaults",
    "caveats": [
        "aa.mplstyle is a post-disk-failure RECONSTRUCTION; cosmetic differences from "
        "pre-crash figures are expected, the data points are unaffected.",
        "Damaged posteriors are skipped, so n_seeds differs from the original campaign and "
        "each point averages a different subset.",
        "KNOWN LIMITATION: the higher-order file list is enumerated as runs 1-5, so peaks and "
        "L1 are capped at 5 seeds even where 10 posteriors exist. n_seeds records what was used.",
        "The two reference guides are HARDCODED anchors (0.0135 and 1.05e5 at 14000 deg^2) "
        "chosen pre-crash; unlike plot_fom_vs_area.py they are NOT read from the series, so "
        "they may no longer sit on the data. They are guides to the SLOPE, not fits.",
        "FULL RESOLUTION — the regime where all three statistics are baryon-BIASED.",
    ],
}, open(out + "_provenance.json", "w"), indent=2)
print("wrote", out + "_values.csv / _provenance.json")

for name, A, m, c, ls, mk in SERIES:
    A = np.array(A, float); s8 = np.array([x[0] for x in m])
    print(f"  {name:14s} slope sig(S8) = {linregress(np.log(A), np.log(s8)).slope:+.2f}  (nseed {[x[4] for x in m]})")
