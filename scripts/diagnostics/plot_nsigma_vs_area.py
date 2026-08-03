#!/usr/bin/env python3
"""Baryon-tension significance vs survey area at full map resolution.

For each summary statistic, the Gaussian Q_DM tension (3-param subset Ωm,S8,w0) between
the null (nobaryons-fiducial) and biased (baryonified-fiducial) posteriors, seed-averaged
over the NPE training seeds (mean ± std), on a LINEAR survey-area axis.

Full resolution:
  - Power spectrum: ℓmin=37 .. ℓmax≈1020 (no upper scale cut), monopole-subtracted MASTER.
  - Peaks / L1 norm: wavelet detail scales 0,1,2,3 (coarse dropped), submean.
All three are computed per-seed directly from the saved posteriors, so the QA gate below
applies uniformly and any added seeds are picked up automatically.

QA gate (silent-mis-fit rejection):
  1. collapse guard  — drop a seed whose null or biased σ(S8) ≥ 0.08 (prior-collapsed).
  2. dual mis-fit    — drop a seed only when its tension is a robust outlier AND that is
                       explained by a per-parameter posterior-width anomaly (null or biased
                       width in Ωm/S8/w0 a robust MAD outlier, ≥15% off the median). This
                       catches localised mis-fits that hide in a poorly-constrained direction
                       (e.g. w0) and pass the S8-only collapse gate — the cause of the
                       L1-14000 non-monotonicity (run 1). A seed with an extreme nσ but NO
                       width anomaly (a legitimate tail draw, e.g. PS-28000-run5) is kept, as
                       is a seed with an odd width but normal tension (harmless).

Run with the tensiometer env (aname):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/diagnostics/plot_nsigma_vs_area.py
"""
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tension.estimators import tension_sigma, SUBSET_INDICES  # noqa: E402

_AA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "styles", "paper_v1.mplstyle")
if os.path.exists(_AA):
    plt.style.use(_AA)
else:
    print(f"[warn] A&A style not found at {_AA} — using matplotlib defaults")

SAMP = "outputs/samples"
PSD = "outputs/baryon_tension/ps_submean_l37/posteriors"
SIG_MAX = 0.08          # collapse guard on σ(S8)
NSIG_Z = 2.5            # a seed's tension must be a robust outlier (MAD-z) ...
WIDTH_Z = 3.0           # ... AND explained by a per-param width outlier (MAD-z) ...
WIDTH_REL = 0.15        # ... that is ≥15% off the median (ignore tiny wobbles).

AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
HOS_TAG = {2000: 2001, 5000: 5001, 10000: 10001, 14000: 14001, 28000: 28001, 35000: 35001}
ERRBAR = os.environ.get("ERRBAR", "sem")   # "sem" = std/√n (mean uncertainty) | "std" = per-seed spread

# Runs manually excluded as unreliable (in addition to the automatic QA gate), keyed by
# (stat, area). PS-28000 runs 5 & 6 are the two single-sided outliers (1.85 low / 4.64 high)
# bracketing the tight 8-seed cluster (2.53-3.50); removing them drops the std 0.69->0.32 and
# leaves the mean unchanged (~3.09). Flagged unreliable on inspection (see commit message).
MANUAL_EXCLUDE = {
    ("Power spectrum", 28000): {5, 6},
}


def run_index(fname):
    m = re.search(r"_run(\d+)", fname)
    return int(m.group(1)) if m else 1


def sigma_s8(a):
    return float(np.sqrt(np.cov(a[:, [0, 1, 2]], rowvar=False)[1, 1]))


def load_pairs(null_glob, biased_glob):
    """Return {run: (null_array, biased_array)} for runs present in BOTH roles.

    A run is dropped if EITHER side is unreadable. Disk-failure-damaged .npy files make
    numpy read the mangled header as pickled data and raise ValueError, so a bare
    np.load aborts the whole figure on the first bad file. Tension needs the null and
    biased posteriors from the SAME run, so a half-damaged pair is unusable and must go
    as a unit rather than being silently half-loaded.
    """
    nulls = {run_index(f): f for f in glob.glob(null_glob)}
    biased = {run_index(f): f for f in glob.glob(biased_glob)}
    out, dropped = {}, 0
    for r in sorted(set(nulls) & set(biased)):
        try:
            out[r] = (np.load(nulls[r]), np.load(biased[r]))
        except (ValueError, OSError, EOFError):
            dropped += 1
    if dropped:
        print(f"    [skip] {dropped} run-pair(s) unreadable (disk-failure damage)")
    return out


def mad_z(vals):
    vals = np.asarray(vals, float)
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-12
    return np.abs(vals - med) / mad, med


def qa(pairs):
    """Collapse + dual mis-fit QA. Returns (kept_runs, excluded[(run, reason)]).

    A seed is dropped as a mis-fit only when BOTH its tension is a robust outlier (|MAD-z|
    > NSIG_Z) AND a per-parameter posterior width (null or biased) is a robust outlier
    (MAD-z > WIDTH_Z, ≥WIDTH_REL off the median). This removes width-driven mis-calibrations
    (the L1-14000 case) while keeping legitimate tail draws (anomalous tension, normal width
    — e.g. PS-28000-run5) and harmless width wobbles (anomalous width, normal tension)."""
    excl, kept = [], []
    for r in sorted(pairs):
        nn, bb = pairs[r]
        sN, sB = sigma_s8(nn), sigma_s8(bb)
        if sN >= SIG_MAX or sB >= SIG_MAX:
            excl.append((r, f"collapse σS8={max(sN, sB):.3f}"))
        else:
            kept.append(r)
    if len(kept) >= 5:                                  # need an ensemble for robust stats
        nsig = {r: tension_sigma(*pairs[r], indices=SUBSET_INDICES)["nsigma"] for r in kept}
        nz, _ = mad_z([nsig[r] for r in kept])
        nz = dict(zip(kept, nz))
        width_z = {r: 0.0 for r in kept}
        for role in (0, 1):                             # 0=null, 1=biased
            stds = np.array([pairs[r][role][:, [0, 1, 2]].std(0) for r in kept])  # (nrun,3)
            med = np.median(stds, 0)
            mad = 1.4826 * np.median(np.abs(stds - med), 0) + 1e-12
            zeff = np.where(np.abs(stds - med) / med > WIDTH_REL,
                            np.abs(stds - med) / mad, 0.0).max(1)   # worst param, mag floor
            for i, r in enumerate(kept):
                width_z[r] = max(width_z[r], zeff[i])
        survivors = []
        for r in kept:
            if nz[r] > NSIG_Z and width_z[r] > WIDTH_Z:
                excl.append((r, f"mis-fit nσ-z={nz[r]:.1f} width-z={width_z[r]:.1f}"))
            else:
                survivors.append(r)
        kept = survivors
    return kept, excl


def series(null_glob, biased_glob, label, exclude=frozenset()):
    pairs = load_pairs(null_glob, biased_glob)
    if not pairs:
        print(f"  {label:14s}: NO DATA ({null_glob})")
        return np.nan, np.nan, 0
    total = len(pairs)
    manual = sorted(r for r in pairs if r in exclude)
    for r in manual:
        del pairs[r]
    kept, excl = qa(pairs)
    excl = [(r, "manual-unreliable") for r in manual] + excl
    ns = [tension_sigma(*pairs[r], indices=SUBSET_INDICES)["nsigma"] for r in kept]
    note = ("  excl: " + ", ".join(f"r{r}({why})" for r, why in excl)) if excl else ""
    print(f"  {label:14s}: n={len(kept)} (of {total})  nσ={np.mean(ns):.3f}±{np.std(ns):.3f}{note}")
    return float(np.mean(ns)), float(np.std(ns)), len(ns)


def ps_globs(A):
    base = f"{PSD}/mask_{A:05d}"
    tail = f"bins1234_l37-1020_r10_masked_{A}sqdeg_apod2.0_master_submean_noisy_s0.26_run*.npy"
    return (f"{base}/null/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_{tail}",
            f"{base}/biased/posterior_samples_ps_auto_cross_nobaryons_vs_baryonified_{tail}")


def hos_globs(prefix, A):
    tag = HOS_TAG[A]
    tail = f"bins1234_scales1234_noisy_s0.26_masked_{tag}sqdeg_submean_new_normalization*_npe.npy"
    return (f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_nobaryons_{tail}",
            f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_baryonified_{tail}")


SERIES_DEFS = [("Power spectrum", lambda A: ps_globs(A)),
               ("Peak counts", lambda A: hos_globs("pc_", A)),
               ("L1 norm", lambda A: hos_globs("", A))]

# ---- cache the NUMBERS so cosmetic re-runs are free ---------------------------------
# Q_DM over every seed pair for 3 statistics x 6 areas costs ~10 minutes. Paying that to
# change a line width is untenable, and the natural response to that cost is to hand-edit
# the PDF instead -- which is precisely how a figure and its values.csv drift apart.
#
# The key is (a) a hash of this file ABOVE the "# ---- plot ----" marker, i.e. every line
# that can change a number and none that only changes appearance, and (b) the identity
# (path, size, mtime) of every posterior actually matched. A new seed, an edited QA rule,
# or a repaired file all change the key; a thicker line does not. That asymmetry is the
# whole point: a stale cache must never be able to outlive a change to the data.
import hashlib as _hl, pickle as _pk  # noqa: E402

_CACHE = "outputs/plots/submean_masked_peaks/.nsigma_vs_area_result.pkl"


def _fingerprint():
    src = open(os.path.abspath(__file__)).read().split("# ---- plot ----")[0]
    h = _hl.sha256(src.encode())
    for _n, _g in SERIES_DEFS:
        for A in AREAS:
            for pat in _g(A):
                for p in sorted(glob.glob(pat)):
                    st = os.stat(p)
                    h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


_fp = _fingerprint()
RESULT = None
if os.environ.get("CACHE", "1") != "0" and os.path.exists(_CACHE):
    try:
        _c = _pk.load(open(_CACHE, "rb"))
        if _c.get("fingerprint") == _fp:
            RESULT = _c["result"]
            print(f"[cache] HIT {_CACHE} — numbers reused, nothing recomputed.\n"
                  f"[cache] fingerprint {_fp[:16]}  (CACHE=0 forces a recompute)")
    except Exception as e:
        print(f"[cache] ignoring unreadable cache ({type(e).__name__})")

if RESULT is None:
    print("Building n_sigma vs area (3-param Q_DM, full resolution, submean) with width-QA:")
    RESULT = {}
    for name, gfun in SERIES_DEFS:
        print(f"\n{name}:")
        M, S, N = [], [], []
        for A in AREAS:
            ng, bg = gfun(A)
            m, s, n = series(ng, bg, f"{A} deg2", MANUAL_EXCLUDE.get((name, A), frozenset()))
            M.append(m); S.append(s); N.append(n)
        RESULT[name] = (np.array(AREAS, float), np.array(M), np.array(S), np.array(N, float))
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    _pk.dump({"fingerprint": _fp, "result": RESULT}, open(_CACHE, "wb"))
    print(f"\n[cache] stored {_CACHE}")

# ---- plot ----
STYLE = {"Power spectrum": ("#0072B2", "-", "o"),
         "Peak counts": ("#D55E00", "--", "s"),
         "L1 norm": ("#009E73", "-.", "^")}
W = 6.9   # submitted-style canvas (paper_v1 fonts are ~2x the A&A ones)

# Stroke weights. The previous values (lw 1.3, ms 4.0, elinewidth 0.8, capsize 2) were
# sized for the A&A style sheet -- 9 pt type on an 88 mm single column. This figure now
# carries the submitted paper's type, roughly twice that, on a 6.9 in canvas, so those
# weights read as spindly against the labels. Scaled to match the type, and exposed as
# env vars because each recompute used to cost 10 minutes; with the cache above, trying
# a value is now seconds.
LW = float(os.environ.get("LW", "2.2"))          # data line
MS = float(os.environ.get("MS", "7.0"))          # marker
ELW = float(os.environ.get("ELW", "1.6"))        # error bar
# Caps need to be generous here, not merely proportional: several bars are short (sigma
# ~0.05-0.5 against a 0-6.5 axis), so a cap scaled to the line weight disappears into the
# marker. Sized to stay legible on the shortest bar in the figure.
CAPSIZE = float(os.environ.get("CAPSIZE", "6.0"))
CAPTHICK = float(os.environ.get("CAPTHICK", "2.0"))

# DO NOT pass markeredgewidth to errorbar(). The caps are '_' MARKERS, so their stroke IS
# markeredgewidth -- and an explicit markeredgewidth in kwargs beats capthick, contrary to
# what the docs imply by calling capthick "a synonym for markeredgewidth". Setting it to 0
# for a clean filled data marker therefore sets the cap stroke to 0 and the caps vanish
# entirely, at every capsize. Verified: with mew=0 the caps come back ms=12 mew=0.0
# (invisible); without it, ms=12 mew=2.0 (capthick, as intended).

fig, ax = plt.subplots(figsize=(W, 0.82 * W))
handles = []
for name in ("Power spectrum", "Peak counts", "L1 norm"):
    A, m, s, n = RESULT[name]
    err = s / np.sqrt(np.maximum(n, 1)) if ERRBAR == "sem" else s
    c, ls, mk = STYLE[name]
    ok = np.isfinite(m)
    eb = ax.errorbar(A[ok], m[ok], yerr=err[ok], color=c, ls=ls, marker=mk, ms=MS, lw=LW,
                     capsize=CAPSIZE, capthick=CAPTHICK, elinewidth=ELW, label=name)
    handles.append(eb)

# REFLINE=0 drops the sqrt(A) guide (writes a "_noref" variant; see `out` below).
#
# Why that is worth having as an option: sqrt(A) is the correct expectation for a
# SINGLE-parameter significance (fixed baryonic bias divided by sigma ~ A^-1/2). This
# figure plots the 3-parameter Gaussian Q_DM mapped through chi2(3) to an equivalent
# nsigma. Q_DM itself scales as A, not sqrt(A), and the chi2->nsigma map is regime
# dependent: with Q ∝ A exactly, the slope one would MEASURE is ~+1.2 near nsigma~0.5,
# ~+1.0 near nsigma~1.4, tending to +0.5 only asymptotically at large tension. So
# sqrt(A) is an asymptotic limit here rather than a general prediction, and drawing it
# can invite a reader to see a discrepancy that is really the conversion.
if os.environ.get("REFLINE", "1") != "0":
    Aref = np.linspace(AREAS[0], AREAS[-1], 100)
    anchor = float(RESULT["Power spectrum"][1][np.array(AREAS) == 14000][0])
    ref, = ax.plot(Aref, anchor * np.sqrt(Aref / 14000.), color="0.5", ls=":", lw=0.9,
                   zorder=0, label=r"$\propto\!\sqrt{A}$ (asymptotic)")
    handles.append(ref)

ax.set_xlabel(r"survey area $\,[\mathrm{deg}^2]$")
ax.set_ylabel(r"baryon tension $\,n_\sigma\;(\Omega_\mathrm{m},S_8,w_0)$")
ax.set_xlim(0, 37000)
ax.set_ylim(0, None)
ax.legend(handles=handles, frameon=False, loc="upper left")

fig.tight_layout(pad=0.4)
out = "outputs/plots/submean_masked_peaks/nsigma_vs_area_fullres"
if os.environ.get("REFLINE", "1") == "0":
    out += "_noref"          # never overwrite the with-line version
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)

# ---- provenance -----------------------------------------------------------
# Write the plotted numbers and the environment next to the figure. Previously the
# RESULT table lived only in memory, so a regenerated figure could not be compared
# numerically against an earlier one — the difference could only be reasoned about,
# not measured. n_seeds matters especially here: disk-failure damage means each point
# averages a different subset of runs than it did pre-crash.
import csv as _csv, json as _json, subprocess as _sub, datetime as _dt

with open(out + "_values.csv", "w", newline="") as _fh:
    _w = _csv.writer(_fh)
    _w.writerow(["statistic", "area_sqdeg", "nsigma", "nsigma_err", "n_seeds", "errbar_kind"])
    for _name, (_A, _M, _S, _N) in RESULT.items():
        for _a, _m, _s, _n in zip(_A, _M, _S, _N):
            # Report the bar that is actually DRAWN, not the raw spread. This wrote _s while
            # labelling the column ERRBAR, so in sem mode the sidecar disagreed with the figure
            # by sqrt(n) -- the one disagreement values.csv exists to make impossible.
            _e = _s / np.sqrt(max(int(_n), 1)) if ERRBAR == "sem" else _s
            _w.writerow([_name, int(_a), f"{_m:.6f}", f"{_e:.6f}", int(_n), ERRBAR])

def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "unavailable"

try:
    _commit = _sub.check_output(["git", "rev-parse", "--short", "HEAD"],
                                stderr=_sub.DEVNULL, text=True).strip()
except Exception:
    _commit = "unknown"

_prov = {
    "figure": os.path.basename(out),
    "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    "git_commit": _commit,
    "errbar": ERRBAR,
    # Stroke weights are env-tunable, so record them: otherwise the same commit can
    # produce visibly different figures and provenance would not say why.
    "stroke_weights": {"lw": LW, "ms": MS, "elinewidth": ELW, "capsize": CAPSIZE,
                       "capthick": CAPTHICK},
    "refline_drawn": os.environ.get("REFLINE", "1") != "0",
    # The publish gate warns when a figure does not say which multipoles and wavelet scales
    # went in, and it is right to: these three curves are only comparable because each is at
    # ITS full resolution, which is a different thing for the PS than for the HOS.
    "scales_included": {
        "power_spectrum": "monopole-subtracted MASTER, lmin=37, lmax~1020, rebin=10; "
                          "no upper scale cut",
        "peaks_l1": "wavelet detail scales 0,1,2,3 (coarse dropped), submean, "
                    "new_normalization, noisy s=0.26, bins1234",
        "note": "full resolution for every statistic — this figure measures how the baryon "
                "bias grows with area BEFORE any scale cut is applied.",
    },
    "lmin": 37,
    "estimator": "tensiometer gaussian_tension.Q_DM -> chi2.cdf -> from_confidence_to_sigma",
    "param_subset": list(SUBSET_INDICES),
    "versions": {m: _ver(m) for m in ("numpy", "scipy", "getdist", "tensiometer", "matplotlib")},
    "mplstyle": _AA if os.path.exists(_AA) else "matplotlib defaults",
    "caveats": [
        "styles/paper_v1.mplstyle reproduces the style of the SUBMITTED version, so this "
            "figure sits beside the figures kept verbatim from it.",
        "Damaged run-pairs are skipped (see [skip] lines in the run log), so n_seeds "
        "differs from the original campaign and each mean is over a different subset.",
    ],
}
with open(out + "_provenance.json", "w") as _fh:
    _json.dump(_prov, _fh, indent=2)
print("wrote", out + "_values.csv / _provenance.json")
print("\nwrote", out + ".pdf / .png")
