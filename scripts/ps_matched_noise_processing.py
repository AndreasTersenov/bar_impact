#!/usr/bin/env python3
"""Recompute the fiducial auto power spectra with the SHAPE NOISE MATCHED between the baryonified
and DMO maps, so that both cosmic variance AND shape noise cancel in the difference.

WHY. The referee on Fig. 2: "the cosmic variance should cancel almost exactly as I assume you're
using the same simulation with and without baryons. You should do the same for the shape noise."
They are right, and it was not being done. `power_spectrum_processing.py:91` seeds each
multiprocessing worker from os.urandom(4), and the baryonified and DMO spectra were produced in
separate runs, so their noise realizations are independent -- and not even reproducible on a rerun.

Measured consequence, on the published spectra: the per-ell correlation between the baryonified and
DMO realizations tracks f^2 (f = signal fraction), the signature of shared simulations with
INDEPENDENT noise, rather than the ~f expected when the noise is shared too. Control: correlating
noisy against noiseless DMO (certainly the same sims) tracks f almost exactly, confirming both the
model and that realization indices correspond across files.

The effect on the plotted band follows sqrt(2(1-f^2)):  where shape noise dominates (f -> 0) the
difference DOUBLES the noise variance and the band is inflated by sqrt(2); where signal dominates
(f -> 1) the shared ICs cancel cosmic variance and the band collapses. Both distortions vanish once
the noise is shared as well.

WHAT THIS DOES DIFFERENTLY. Both maps for a given permutation are read together and the SAME noise
array is added to each. The noise is drawn from a seed derived deterministically from (perm, bin),
so the run is exactly reproducible and any later regeneration reproduces it bit-for-bit -- the
opposite of seeding from os.urandom.

Everything else is byte-identical to power_spectrum_processing.py: the same add_shape_noise formula,
the same hp.map2alm -> hp.alm2cl at lmax=1024, the same nside, sigma_e and galaxy density. The only
change is WHICH noise array is used, so any difference in the output is attributable to the seeding
and nothing else.

Source maps are the read-only shared dataset (no download, no quota cost):
    /lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_XXXX/
        projected_probes_maps_baryonified512.h5
        projected_probes_maps_nobaryons512.h5
Both come from the same shell permutation, so the ICs are shared -- exactly as the referee assumed.

  python scripts/ps_matched_noise_processing.py --nproc 20            # all 200 perms, 4 bins
  python scripts/ps_matched_noise_processing.py --nperm 4 --nproc 4   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

import h5py
import healpy as hp
import numpy as np

SRC = ("/lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial")
OUT = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/"
       "fiducial/cosmo_fiducial")
NSIDE = 512
LMAX = 1024
SIGMA_E = 0.26
NGAL = 6.75
SEED_BASE = 20260802     # fixed, so this is reproducible forever; change only to make a new suite


def noise_map(seed, sigma_e=SIGMA_E, galaxy_density=NGAL, nside=NSIDE):
    """The shape-noise map. Same formula as power_spectrum_processing.add_shape_noise, but the
    realization is drawn from an EXPLICIT seed instead of the ambient global RNG."""
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    return np.random.default_rng(seed).normal(loc=0.0, scale=sigma_pix, size=npix)


def get_power_spectrum(m, lmax=LMAX):
    """Verbatim from power_spectrum_processing.py so the pipeline is otherwise unchanged."""
    return hp.alm2cl(hp.map2alm(m, lmax=lmax))


def one_perm(p, bins, lmax):
    """Process one permutation: for each bin, add the SAME noise map to both variants."""
    d = f"{SRC}/perm_{p:04d}"
    fb = f"{d}/projected_probes_maps_baryonified512.h5"
    fn = f"{d}/projected_probes_maps_nobaryons512.h5"
    if not (os.path.exists(fb) and os.path.exists(fn)):
        return p, None, f"missing maps in {d}"
    out = {}
    try:
        with h5py.File(fb, "r") as hb, h5py.File(fn, "r") as hn:
            for b in bins:
                key = f"kg/stage3_lensing{b}"
                kg_bar = np.array(hb[key], dtype=np.float64)
                kg_dmo = np.array(hn[key], dtype=np.float64)
                # THE POINT: one noise array, added to both. Seed depends only on (perm, bin),
                # never on wall-clock or OS entropy, so this is reproducible.
                n = noise_map(SEED_BASE + 1000 * p + b)
                out[(b, "baryonified")] = get_power_spectrum(kg_bar + n, lmax)
                out[(b, "nobaryons")] = get_power_spectrum(kg_dmo + n, lmax)
    except Exception as e:                                    # noqa: BLE001
        return p, None, f"{type(e).__name__}: {e}"
    return p, out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--bins", default="1,2,3,4")
    ap.add_argument("--nproc", type=int, default=10)
    ap.add_argument("--lmax", type=int, default=LMAX)
    ap.add_argument("--tag", default="matchednoise",
                    help="output suffix: all_cls_fiducial_<kind>_bin<b>_noisy_s0.26_<tag>.npy")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args()
    bins = [int(v) for v in a.bins.split(",")]

    print(f"src   : {SRC}")
    print(f"out   : {a.outdir}")
    print(f"perms : {a.nperm}   bins: {bins}   lmax: {a.lmax}   nproc: {a.nproc}")
    print(f"seed  : {SEED_BASE} + 1000*perm + bin   (deterministic, reproducible)", flush=True)
    os.makedirs(a.outdir, exist_ok=True)

    import multiprocessing as mp
    # NOTE: no pool initializer reseeding the RNG. That is exactly what broke the original --
    # every worker reseeded from os.urandom, so noise depended on which worker got the task.
    fn = partial(one_perm, bins=bins, lmax=a.lmax)
    results, failures = {}, []
    with mp.Pool(processes=a.nproc) as pool:
        for i, (p, out, err) in enumerate(pool.imap_unordered(fn, range(a.nperm)), 1):
            if err:
                failures.append((p, err))
                print(f"[FAIL] perm {p}: {err}", flush=True)
            else:
                results[p] = out
            if i % 10 == 0 or i == a.nperm:
                print(f"  {i}/{a.nperm} done ({len(failures)} failed)", flush=True)

    if not results:
        sys.exit("[fatal] nothing processed")

    order = sorted(results)
    nl = a.lmax + 1
    for b in bins:
        for kind in ("baryonified", "nobaryons"):
            arr = np.zeros((len(order), nl))
            for i, p in enumerate(order):
                arr[i] = results[p][(b, kind)][:nl]
            path = f"{a.outdir}/all_cls_fiducial_{kind}_bin{b}_noisy_s{SIGMA_E:.2f}_{a.tag}.npy"
            np.save(path, arr)
            print(f"wrote {path}  {arr.shape}", flush=True)

    meta = {
        "generator": "scripts/ps_matched_noise_processing.py",
        "purpose": "shape noise MATCHED between baryonified and DMO (referee request on Fig. 2)",
        "source": SRC,
        "n_perms_requested": a.nperm, "n_perms_ok": len(order),
        "perms_used": order, "bins": bins, "lmax": a.lmax,
        "nside": NSIDE, "sigma_e": SIGMA_E, "galaxy_density": NGAL,
        "seed_rule": f"np.random.default_rng({SEED_BASE} + 1000*perm + bin)",
        "noise_shared_between_variants": True,
        "failures": failures,
        "differs_from_published_only_by": ("the noise realization; add_shape_noise formula, "
                                           "map2alm/alm2cl, nside, lmax, sigma_e, n_gal identical"),
    }
    mpath = f"{a.outdir}/all_cls_fiducial_{a.tag}_manifest.json"
    with open(mpath, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {mpath}")
    if failures:
        print(f"WARNING: {len(failures)} permutations failed; arrays contain only the {len(order)} "
              f"that succeeded. Do not compare against the published 200-realization spectra "
              f"without accounting for that.")


if __name__ == "__main__":
    main()
