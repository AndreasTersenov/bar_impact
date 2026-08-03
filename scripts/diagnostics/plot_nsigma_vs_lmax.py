#!/usr/bin/env python3
"""Baryon-tension significance vs the power-spectrum upper scale cut, per footprint.

One panel per survey area: the Gaussian Q_DM tension (3-param subset Ωm, S8, w0) between
the null (nobaryons-fiducial) and biased (baryonified-fiducial) posteriors, seed-averaged
over the NPE training seeds, as a function of the upper multipole cut ℓmax. A dashed line
marks the 0.3σ tolerance; where each curve crosses it is the largest ℓmax that stays
baryon-safe, which is the number the paper quotes.

Power spectrum only, monopole-subtracted MASTER, ℓmin = 37, step-40 cut grid.

Unlike `plot_nsigma_vs_area.py` this reads the campaign's **aggregated** table rather than
recomputing from posteriors. That is deliberate: `ps_submean_l37/tables/tension_3param_agg.csv`
survived the disk failure complete and unflagged (6 areas × 18 cuts, n = 5/5 on every row,
zero exclusions), so the aggregate reproduces the pre-crash figure exactly, while
re-deriving it would silently re-average over whatever posteriors happen to be readable.
The 6-param tables of the same campaign are zero-length casualties; they are not used here.

  FULLSKY=1   append the full-sky panel (healpy 10-ℓ bins, not magnitude-comparable to
              the masked NaMaster nlb=4 — same trend only; see build_7panel_tension_plot.py)
  SUBTITLE=0  drop the diagnostic subtitle for the paper version
  ERRBAR      "std" (default, per-seed spread — what the pre-crash figure drew)
              | "sem" (std/√n, uncertainty on the mean)

Run with the tensiometer env (aname):
  /lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python \
      scripts/diagnostics/plot_nsigma_vs_lmax.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_AA = os.path.join(REPO, "styles", "paper_v1.mplstyle")
if os.path.exists(_AA):
    plt.style.use(_AA)
else:
    print(f"[warn] A&A style not found at {_AA} — using matplotlib defaults")

MASKED = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
FULLSKY = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/tables/tension_3param_agg.csv"

AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
THRESHOLD = 0.3
ERRBAR = os.environ.get("ERRBAR", "std")
WANT_FULLSKY = os.environ.get("FULLSKY", "0") == "1"
WANT_SUBTITLE = os.environ.get("SUBTITLE", "1") != "0"


def read_agg(path, area_is_str=False):
    """Return {area: (cuts, mean, std, n)} from an aggregated tension table.

    Raises on a zero-length or header-only file rather than returning an empty curve —
    several tables in this campaign are zero-byte disk-failure casualties, and a silently
    empty panel is exactly the failure mode this figure must not have.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        raise ValueError(f"{path} has no data rows (zero-length table — disk-failure casualty?)")
    out = {}
    for r in rows:
        a = r["area"] if area_is_str else int(r["area"])
        out.setdefault(a, []).append(
            (int(r["upper_cut"]), float(r["mean"]), float(r["std"]), int(r["n"])))
    for a, v in out.items():
        v.sort()
        c, m, s, n = (np.array(x) for x in zip(*v))
        out[a] = (c, m, s, n)
    return out


def crossing(cuts, mean, thr=THRESHOLD):
    """ℓmax at which the curve first reaches `thr`, linearly interpolated.

    Returns (interpolated_cut, first_grid_cut_at_or_above) or (nan, nan) if never reached.
    The interpolation is on the FIRST upcrossing; these curves are non-monotonic in places
    (e.g. 2000 deg² dips at 940), so a last-crossing or a global search would give a
    different — and less conservative — answer.
    """
    above = np.where(mean >= thr)[0]
    if not len(above):
        return np.nan, np.nan
    i = int(above[0])
    grid = float(cuts[i])
    if i == 0:
        return grid, grid
    x0, x1 = float(cuts[i - 1]), float(cuts[i])
    y0, y1 = float(mean[i - 1]), float(mean[i])
    interp = x0 + (x1 - x0) * (thr - y0) / (y1 - y0) if y1 != y0 else x1
    return interp, grid


masked = read_agg(MASKED)
missing = [a for a in AREAS if a not in masked]
if missing:
    raise SystemExit(f"[fatal] no rows for area(s) {missing} in {MASKED}")

curves = [(f"{a} deg$^2$", a, *masked[a]) for a in AREAS]
if WANT_FULLSKY:
    fs = read_agg(FULLSKY, area_is_str=True)
    curves.append(("Full sky", "fullsky", *fs["fullsky"]))

n = len(curves)
fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 5.8), sharex=not WANT_FULLSKY)
axes = np.atleast_1d(axes)

RESULT = {}
for ax, (label, area, cuts, mean, std, nseed) in zip(axes, curves):
    err = std / np.sqrt(nseed) if ERRBAR == "sem" else std
    ax.errorbar(cuts, mean, yerr=err, fmt="o", ms=3.2, elinewidth=0.9,
                capsize=2, color="C0", zorder=3)
    ax.axhline(THRESHOLD, color="crimson", ls="--", lw=1.0, zorder=2)
    ax.set_title(rf"Area = {label}" if area != "fullsky" else "Full sky", pad=4)
    ax.grid(True, alpha=0.25, ls=":")
    ax.set_ylim(0, None)
    RESULT[area] = (cuts, mean, err, nseed)

    xi, xg = crossing(cuts, mean)
    if xi == xi:
        ax.axvline(xi, color="crimson", ls=":", lw=0.8, alpha=0.6, zorder=1)

axes[0].set_ylabel(r"baryon tension $\,n_\sigma\;(\Omega_\mathrm{m},S_8,w_0)$")
fig.supxlabel(r"upper scale cut $\,\ell_\mathrm{max}$", y=0.02)

# Label the threshold once, on the leftmost panel, rather than a legend box per panel.
axes[0].text(0.04, THRESHOLD, rf"${THRESHOLD}\sigma$", transform=axes[0].get_yaxis_transform(),
             color="crimson", fontsize=11, va="bottom", ha="left")

if WANT_SUBTITLE:
    what = "6 footprints + full sky" if WANT_FULLSKY else "6 footprints"
    # ℓ must go through mathtext — the A&A style's font has no U+2113 glyph.
    extra = r"  (full sky = healpy 10-$\ell$ bins; masked = NaMaster nlb=4)" if WANT_FULLSKY else ""
    fig.text(0.5, 0.995,
             rf"{what} | monopole-subtracted PS, $\ell\geq37$ | step-40 | "
             rf"3-param $Q_\mathrm{{DM}}$, mean$\pm$" + ("s.e.m." if ERRBAR == "sem" else "std")
             + rf"/{int(RESULT[AREAS[0]][3][0])} runs{extra}",
             ha="center", va="top", fontsize=11, color="0.4")

fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 0.94 if WANT_SUBTITLE else 1.0))

out = f"{REPO}/outputs/plots/ps_submean_l37/nsigma_vs_lmax"
if WANT_FULLSKY:
    out += "_with_fullsky"      # never overwrite the masked-only version
if ERRBAR == "sem":
    out += "_sem"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)

# ---- provenance -----------------------------------------------------------
# Same rule as plot_nsigma_vs_area.py: the plotted numbers and the environment land next
# to the figure, so a regenerated figure can be compared against an earlier one by
# measurement rather than by argument. n_seeds is carried per point because the disk
# failure means a point can silently average a different subset of runs than it did
# pre-crash — here every row is a full n=5, which is itself worth recording.
import json as _json, subprocess as _sub, datetime as _dt

with open(out + "_values.csv", "w", newline="") as _fh:
    _w = csv.writer(_fh)
    _w.writerow(["footprint", "upper_cut_lmax", "nsigma", "nsigma_err", "n_seeds", "errbar_kind"])
    for _a, (_c, _m, _e, _n) in RESULT.items():
        for _ci, _mi, _ei, _ni in zip(_c, _m, _e, _n):
            _w.writerow([_a, int(_ci), f"{_mi:.6f}", f"{_ei:.6f}", int(_ni), ERRBAR])

# The threshold crossings are the paper's headline numbers, so they get their own table
# rather than being read off the figure.
#
# Three different numbers, easy to confuse, so all three are written down:
#   lmax_largest_safe_grid_cut  — the cut you would ADOPT: last grid cut still under
#                                 threshold. This is the analysis choice.
#   lmax_first_grid_cut_at_or_above — the crossing: first cut that FAILS. One step past
#                                 the adoptable cut, and 0.41 sigma at 14000 deg^2.
#   lmax_at_0.3sigma_interp     — where the curve crosses between grid points.
# Using the crossing as the cut would put a failing bias into a "baryon-safe" figure.
_cross = []
with open(out + "_crossings.csv", "w", newline="") as _fh:
    _w = csv.writer(_fh)
    _w.writerow(["footprint", "lmax_largest_safe_grid_cut", "nsigma_at_largest_safe",
                 "lmax_first_grid_cut_at_or_above", "lmax_at_0.3sigma_interp",
                 "nsigma_at_lmax_max", "lmax_max"])
    for _a, (_c, _m, _e, _n) in RESULT.items():
        _xi, _xg = crossing(_c, _m)
        _safe = [(ci, mi) for ci, mi in zip(_c, _m) if mi < THRESHOLD]
        _sc, _sm = (max(_safe, key=lambda t: t[0]) if _safe else (float("nan"),) * 2)
        _cross.append((_a, _xi, _xg, _sc))
        _w.writerow([_a,
                     int(_sc) if _sc == _sc else "none safe",
                     f"{_sm:.4f}" if _sm == _sm else "n/a",
                     int(_xg) if _xg == _xg else "not reached",
                     f"{_xi:.1f}" if _xi == _xi else "not reached",
                     f"{_m[-1]:.4f}", int(_c[-1])])


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "unavailable"


try:
    _commit = _sub.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=REPO, stderr=_sub.DEVNULL, text=True).strip()
except Exception:
    _commit = "unknown"

_prov = {
    "figure": os.path.basename(out),
    "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    "git_commit": _commit,
    "errbar": ERRBAR,
    "threshold_sigma": THRESHOLD,
    "fullsky_panel": WANT_FULLSKY,
    "source_tables": [MASKED] + ([FULLSKY] if WANT_FULLSKY else []),
    "derivation": "aggregated campaign table (NOT recomputed from posteriors)",
    "estimator": "tensiometer gaussian_tension.Q_DM -> chi2.cdf -> from_confidence_to_sigma",
    "param_subset": [0, 1, 2],
    "param_names": ["Omega_m", "S8", "w0"],
    "lmin": 37,
    "scales_included": {
        "power_spectrum": ("monopole-subtracted MASTER, lmin=37, rebin=10; upper cut varies "
                           "along the x-axis over the step-40 grid 340..1020"),
    },
    "versions": {m: _ver(m) for m in ("numpy", "scipy", "getdist", "tensiometer", "matplotlib")},
    "mplstyle": _AA if os.path.exists(_AA) else "matplotlib defaults",
    "caveats": [
        "styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this "
            "figure sits beside the figures kept verbatim from it.",
        "Read from the campaign's aggregated table, which survived intact (n=5/5 on every "
        "row, 0 exclusions) — so unlike nsigma_vs_area these points are NOT re-averaged "
        "over a damaged subset and should reproduce the pre-crash figure exactly.",
        "The 6-param tables of this campaign are zero-length disk-failure casualties; only "
        "the 3-param subset is available.",
        "Threshold crossings are first-upcrossing linear interpolations; these curves are "
        "non-monotonic in places (2000 deg2 dips at lmax 940), so a global or last crossing "
        "would give a larger, less conservative lmax.",
    ],
}
if WANT_FULLSKY:
    _prov["caveats"].append(
        "Full sky uses the healpy 10-ell-bin pipeline vs the masked NaMaster nlb=4 (40-ell); "
        "the trend is comparable, the magnitude is not.")

with open(out + "_provenance.json", "w") as _fh:
    _json.dump(_prov, _fh, indent=2)

print(f"wrote {out}.pdf / .png")
print(f"wrote {out}_values.csv / _crossings.csv / _provenance.json")
print(f"\n{THRESHOLD}σ threshold (ℓmax) — ADOPT the 'safe' column, not the crossing:")
print(f"  {'footprint':>12}  {'safe':>6}  {'crossing':>8}  {'interp':>7}")
for _a, _xi, _xg, _sc in _cross:
    _lbl = "full sky" if _a == "fullsky" else f"{_a} deg²"
    if _xi != _xi:
        print(f"  {_lbl:>12}: never reaches {THRESHOLD}σ")
        continue
    print(f"  {_lbl:>12}  {int(_sc) if _sc == _sc else 'none':>6}  {int(_xg):>8}  {_xi:7.1f}")
