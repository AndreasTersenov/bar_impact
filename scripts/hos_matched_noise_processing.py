#!/usr/bin/env python3
"""Recompute the fiducial starlet l1-norm AND peak counts with the shape noise MATCHED between the
baryonified and DMO maps, and save the SNR bin centres alongside them.

WHY. The same three defects the referee's Fig. 2 comment exposed in the power spectra are present
in the HOS:

  1. UNMATCHED NOISE. l1_norm_processing.py:46 and peak_counts_processing.py:35 both seed each
     worker from os.urandom(4), and the baryonified and DMO runs were separate, so their shape
     noise is independent -- and not reproducible on a rerun. The simulations WERE already shared
     (both variants live in the same perm_XXXX directory, from the same shell permutation), so only
     the noise ever needed fixing.

  2. THE SNR AXIS WAS RECONSTRUCTED, WRONGLY. get_wtl1_sphere RETURNS the bin centres --
     `bins = 0.5*(thresholds[:-1] + thresholds[1:])` over nbins+1 edges -- but the processing script
     saved only the l1 values and threw the centres away. The figure then rebuilt them as
     np.linspace(min, max, nbins), which has the wrong endpoints AND the wrong spacing: for 40 bins
     on [-10, 10] the true centres are -9.75..9.75 at spacing 0.5, not -10..10 at 0.513. This script
     SAVES the centres, so nothing has to reconstruct them again.

  3. THE PUBLISHED FIGURE DIVIDED BY (<l1> + 5). That offset is negligible in the core bins and
     100% of the denominator in the tails, where <l1> is identically zero -- so the wings of the
     published figure plot dl1/5, not a fractional difference. Nothing here needs the offset; the
     figure masks empty bins instead.

RANGE CHANGE, DELIBERATE. The paper states the l1-norm uses 40 bins over nu in [-10, +10]; the code
default was [-13, +13]. Regenerating is the moment to make them agree, and the paper's range is
adopted. NOTE this changes the l1 data vector, so these summaries are NOT interchangeable with the
ones the NPE inference was trained on (which used [-13, 13]).

Peaks keep nbins=31 EDGES over [-2, +10] -> 30 counts, matching the shape on disk. (The paper says
"40 linearly spaced bins" for peaks; the code and the data both say 30. The text needs fixing.)

HOW THE STATISTIC IS COMPUTED. `pycs.astro.wl.hos_peaks_l1` cannot be imported: it does
`from pycs.sparsity.sparse2d.starlet import *` at module level, and that module raises NameError on
`class MRStarlet(pysparse.MRStarlet)` when the pysap C++ bindings are absent -- despite printing a
message claiming it falls back to Python. The SPHERICAL transform we actually need (CMRStarlet in
pycs.sparsity.mrs.mrs_starlet) has a genuine pure-Python path and imports fine.

So rather than retype the two functions, this extracts their ORIGINAL SOURCE from the repo file by
AST and executes it against the imports they need. The statistic is therefore the upstream code
verbatim, not a reimplementation, and `--check-source` prints the extracted text for eyeballing.

  PYTHONPATH=<cosmostat_src> python scripts/hos_matched_noise_processing.py --nproc 40
  ... --nperm 2 --bins 1 --nproc 2     # smoke test
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from functools import partial

import h5py
import healpy as hp
import numpy as np

SRC = "/lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial"
OUT = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/"
       "fiducial/cosmo_fiducial")
COSMOSTAT = os.environ.get("COSMOSTAT_SRC",
                           "/lustre/fswork/projects/rech/nzu/ulx34io/cosmostat_src")
NSIDE, NSCALES, SIGMA_E, NGAL, NOISE_STD = 512, 5, 0.26, 6.75, 0.0146
L1_NBINS, L1_MIN, L1_MAX = 40, -10.0, 10.0      # paper's range (was -13,13 in the code)
PK_NBINS, PK_MIN, PK_MAX = 31, -2.0, 10.0       # 31 EDGES -> 30 counts
SEED_BASE = 20260802                            # same rule as the power-spectrum regeneration

# BNT nulling matrix, verbatim from bnt_power_spectrum_processing.py:17. The published BNT HOS
# pipeline applies it to the four NOISY maps and then computes the statistic per BNT bin.
BNT_MATRIX = np.array([[1.0, 0.0, 0.0, 0.0],
                       [-1.0, 1.0, 0.0, 0.0],
                       [0.4521097, -1.4521097, 1.0, 0.0],
                       [0.0, 0.25127807, -1.251278, 1.0]])


def load_upstream():
    """Pull get_wtl1_sphere / get_wtpeaks_sphere out of the upstream file as source, and exec them.

    Importing the module is impossible without the pysap bindings (see the header). Extracting the
    two functions by AST keeps the algorithm byte-identical to upstream while sidestepping the
    unrelated 2-D starlet import that breaks.
    """
    path = os.path.join(COSMOSTAT, "pycs/astro/wl/hos_peaks_l1.py")
    if not os.path.exists(path):
        sys.exit(f"[fatal] cosmostat source not found at {path}; set COSMOSTAT_SRC")
    text = open(path).read()
    tree = ast.parse(text)
    # Extract EVERY top-level function, not just the two entry points: get_wtpeaks_sphere calls
    # get_peaks_sphere, and those helpers call others. A def only compiles at exec time, so the
    # ones that would need the broken 2-D starlet are harmless unless actually called.
    want = {"get_wtl1_sphere", "get_wtpeaks_sphere"}
    src = {n.name: ast.get_source_segment(text, n) for n in tree.body
           if isinstance(n, ast.FunctionDef)}
    missing = want - set(src)
    if missing:
        sys.exit(f"[fatal] could not extract {missing} from {path}")

    from scipy import ndimage                                     # noqa: F401
    from scipy.special import erf                                 # noqa: F401
    from numpy import linalg as LA                                # noqa: F401
    from pycs.sparsity.mrs.mrs_starlet import CMRStarlet          # the pure-python spherical WT

    ns = {"np": np, "hp": hp, "ndimage": ndimage, "erf": erf, "LA": LA,
          "CMRStarlet": CMRStarlet}
    for name, body in src.items():
        try:
            exec(compile(body, f"<upstream:{name}>", "exec"), ns)
        except Exception:                                          # noqa: BLE001
            pass          # a helper we do not need; only a real call would matter
    for name in want:
        if name not in ns:
            sys.exit(f"[fatal] {name} failed to define")
    return ns["get_wtl1_sphere"], ns["get_wtpeaks_sphere"], {k: src[k] for k in want}


def noise_map(seed):
    """Same formula as the published add_shape_noise, but from an EXPLICIT seed."""
    npix = hp.nside2npix(NSIDE)
    pix_arcmin2 = hp.nside2pixarea(NSIDE, degrees=True) * 3600
    sigma_pix = SIGMA_E / np.sqrt(NGAL * pix_arcmin2)
    return np.random.default_rng(seed).normal(0.0, sigma_pix, npix)


# Set per worker by _init_worker. The AST-extracted functions are not module attributes, so they
# cannot be pickled and sent to the pool -- each worker builds its own instead.
_L1F = _PKF = None


def _init_worker():
    global _L1F, _PKF
    _L1F, _PKF, _ = load_upstream()


def _stats(m, l1f, pkf):
    l1b, l1 = l1f(m, nscales=NSCALES, nbins=L1_NBINS, min_snr=L1_MIN, max_snr=L1_MAX,
                  noise_std=NOISE_STD)
    pk, pkb = pkf(m, nscales=NSCALES, noise_std=NOISE_STD, nbins=PK_NBINS, Min=PK_MIN,
                  Max=PK_MAX, verbose=False)
    return np.asarray(l1), np.asarray(pk), np.asarray(l1b), np.asarray(pkb)


def one_perm(p, bins, bnt=False):
    """With bnt=True the four tomographic maps are read together, the SAME per-bin noise as the
    non-BNT suite is added to each, and BNT_MATRIX mixes them before the statistic is taken."""
    l1f, pkf = _L1F, _PKF
    d = f"{SRC}/perm_{p:04d}"
    fb, fn = (f"{d}/projected_probes_maps_baryonified512.h5",
              f"{d}/projected_probes_maps_nobaryons512.h5")
    if not (os.path.exists(fb) and os.path.exists(fn)):
        return p, None, f"missing maps in {d}"
    out = {}
    try:
        with h5py.File(fb, "r") as hb, h5py.File(fn, "r") as hn:
            if bnt:
                noises = [noise_map(SEED_BASE + 1000 * p + (i + 1)) for i in range(4)]
                for kind, h in (("baryonified", hb), ("nobaryons", hn)):
                    maps = np.array([np.array(h[f"kg/stage3_lensing{i+1}"], dtype=np.float64)
                                     + noises[i] for i in range(4)])
                    maps = BNT_MATRIX @ maps
                    for b in bins:
                        l1, pk, l1b, pkb = _stats(maps[b - 1], l1f, pkf)
                        out[(b, kind, "l1")] = l1
                        out[(b, kind, "pk")] = pk
                        out[(b, "l1_bins")] = l1b
                        out[(b, "pk_bins")] = pkb
                return p, out, None
            for b in bins:
                key = f"kg/stage3_lensing{b}"
                kg = {"baryonified": np.array(hb[key], dtype=np.float64),
                      "nobaryons": np.array(hn[key], dtype=np.float64)}
                n = noise_map(SEED_BASE + 1000 * p + b)   # ONE noise array, added to both
                for kind in ("baryonified", "nobaryons"):
                    m = kg[kind] + n
                    l1b, l1 = l1f(m, nscales=NSCALES, nbins=L1_NBINS, min_snr=L1_MIN,
                                  max_snr=L1_MAX, noise_std=NOISE_STD)
                    pk, pkb = pkf(m, nscales=NSCALES, noise_std=NOISE_STD, nbins=PK_NBINS,
                                  Min=PK_MIN, Max=PK_MAX, verbose=False)
                    out[(b, kind, "l1")] = np.asarray(l1)
                    out[(b, kind, "pk")] = np.asarray(pk)
                    out[(b, "l1_bins")] = np.asarray(l1b)
                    out[(b, "pk_bins")] = np.asarray(pkb)
    except Exception as e:                                        # noqa: BLE001
        return p, None, f"{type(e).__name__}: {e}"
    return p, out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--bins", default="1,2,3,4")
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--bnt", action="store_true",
                    help="apply the BNT nulling to the four noisy maps before the statistic")
    ap.add_argument("--tag", default="matchednoise")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--check-source", action="store_true",
                    help="print the extracted upstream source and exit")
    a = ap.parse_args()
    bins = [int(v) for v in a.bins.split(",")]

    l1f, pkf, src = load_upstream()
    if a.check_source:
        for k, v in src.items():
            print("=" * 78); print(v)
        return
    print(f"src   : {SRC}\nout   : {a.outdir}")
    print(f"perms : {a.nperm}   bins: {bins}   nproc: {a.nproc}")
    print(f"l1    : nbins={L1_NBINS} range=[{L1_MIN},{L1_MAX}]  (paper's range)")
    print(f"peaks : nbins={PK_NBINS} edges -> {PK_NBINS-1} counts, range=[{PK_MIN},{PK_MAX}]")
    print(f"seed  : {SEED_BASE} + 1000*perm + bin  (shared by both variants)", flush=True)
    os.makedirs(a.outdir, exist_ok=True)

    import multiprocessing as mp
    fn = partial(one_perm, bins=bins, bnt=a.bnt)
    res, fails = {}, []
    with mp.Pool(processes=a.nproc, initializer=_init_worker) as pool:
        for i, (p, out, err) in enumerate(pool.imap_unordered(fn, range(a.nperm)), 1):
            if err:
                fails.append((p, err)); print(f"[FAIL] perm {p}: {err}", flush=True)
            else:
                res[p] = out
            if i % 10 == 0 or i == a.nperm:
                print(f"  {i}/{a.nperm} done ({len(fails)} failed)", flush=True)
    if not res:
        sys.exit("[fatal] nothing processed")

    order = sorted(res)
    for b in bins:
        for stat, nb in (("l1", L1_NBINS), ("pk", PK_NBINS - 1)):
            for kind in ("baryonified", "nobaryons"):
                arr = np.stack([res[p][(b, kind, stat)] for p in order])
                nm = (("all_bnt_l1_norms" if a.bnt else "all_l1_norms") if stat == "l1"
                      else ("all_bnt_peak_counts" if a.bnt else "all_peak_counts"))
                path = (f"{a.outdir}/{nm}_fiducial_{kind}_bin{b}_noisy_s{SIGMA_E:.2f}"
                        f"_new_normalization_{a.tag}.npy")
                np.save(path, arr)
                print(f"wrote {path}  {arr.shape}", flush=True)
            # THE BIN CENTRES -- the thing the original pipeline computed and discarded.
            bn = res[order[0]][(b, f"{stat}_bins")]
            pre = ("all_bnt_" if a.bnt else "all_") + ("l1_norms" if stat == "l1"
                                                        else "peak_counts")
            bpath = f"{a.outdir}/{pre}_fiducial_bin{b}_{a.tag}_bincentres.npy"
            np.save(bpath, bn)
            print(f"wrote {bpath}  {bn.shape}", flush=True)

    meta = {"generator": "scripts/hos_matched_noise_processing.py",
            "purpose": "l1-norm + peak counts with shape noise MATCHED between baryonified and DMO",
            "source": SRC, "cosmostat_src": COSMOSTAT,
            "statistic_code": "upstream get_wtl1_sphere / get_wtpeaks_sphere, extracted by AST",
            "n_perms_ok": len(order), "perms_used": order, "bins": bins,
            "l1": {"nbins": L1_NBINS, "min_snr": L1_MIN, "max_snr": L1_MAX,
                   "note": "paper's range; the code default was [-13,13] and the NPE inference "
                           "used that, so these vectors are NOT interchangeable with it"},
            "peaks": {"nbins_edges": PK_NBINS, "counts": PK_NBINS - 1,
                      "min": PK_MIN, "max": PK_MAX,
                      "note": "the paper says 40 bins; code and data both say 30"},
            "nside": NSIDE, "nscales": NSCALES, "sigma_e": SIGMA_E, "galaxy_density": NGAL,
            "noise_std": NOISE_STD,
            "seed_rule": f"np.random.default_rng({SEED_BASE} + 1000*perm + bin)",
            "noise_shared_between_variants": True, "bnt": bool(a.bnt),
            "bin_centres_saved": True,
            "failures": fails}
    mpath = f"{a.outdir}/all_{'bnt_' if a.bnt else ''}hos_fiducial_{a.tag}_manifest.json"
    with open(mpath, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {mpath}")
    if fails:
        print(f"WARNING: {len(fails)} permutations failed; arrays hold only the {len(order)} "
              f"that succeeded.")


if __name__ == "__main__":
    main()
