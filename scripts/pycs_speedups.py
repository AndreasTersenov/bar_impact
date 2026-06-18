"""In-repo performance patches for the pycs spherical wavelet pipeline.

Both patches target hot spots found by benchmarking the L1 / peak-count
processing (nside=512, nscales=5, single-threaded -- the production config):

1. ``map2alm`` refinement iterations.
   The starlet forward transform calls ``pycs...mrs_tools.map2alm`` which
   defaults to ``iter=3``. healpy's ``map2alm`` with ``iter=3`` runs ~7
   spherical-harmonic transforms (1 + 2*iter) instead of 1, so the forward
   SHT dominated the whole transform (2.84 s of 4.6 s per map). Dropping to
   ``iter=1`` cuts the forward SHT to ~1.2 s. Measured impact on the L1 norms
   of going all the way to ``iter=0`` was median 1.7e-4 / max 4e-3 -- far below
   sample variance -- so ``iter=1`` is a conservative, safe default.
   Net: ~1.8x on L1, ~1.3x on peaks.

2. ``hp.get_all_neighbours`` recomputation.
   ``get_wtpeaks_sphere`` -> ``get_peaks_sphere`` rebuilds
   ``hp.get_all_neighbours(nside, arange(npix))`` for every scale of every map
   (~0.36 s x nscales). The neighbour table depends only on ``nside``, so we
   cache it. Net: removes ~1.8 s/map from peak counts.

Both are monkeypatches on the already-imported ``pycs`` / ``healpy`` module
objects. On Linux (fork-based multiprocessing) forked Pool workers inherit the
patches, provided :func:`enable` runs in the parent process before the Pool is
created -- which is what calling it at module import time guarantees.

Usage (after importing pycs in a processing script)::

    import pycs_speedups
    pycs_speedups.enable(starlet_iter=1)

Idempotent: calling :func:`enable` more than once is a no-op after the first.
Originals are stashed on the wrappers as ``_pycs_speedups_orig`` for debugging.
"""

import functools

import numpy as np
import healpy as hp
import pycs.sparsity.mrs.mrs_starlet as _mrs_starlet

_APPLIED = {"starlet_iter": None, "neighbour_cache": False}


def _patch_starlet_iter(n_iter):
    """Make the starlet transform's map2alm default to ``iter=n_iter``."""
    if _APPLIED["starlet_iter"] is not None:
        return
    orig = _mrs_starlet.map2alm  # pycs...mrs_tools.map2alm, re-exported here

    @functools.wraps(orig)
    def map2alm_fast(maps, lmax=None, iter=n_iter):
        return orig(maps, lmax=lmax, iter=iter)

    map2alm_fast._pycs_speedups_orig = orig
    # wt_phi_filter_trans() resolves ``map2alm`` from the mrs_starlet module
    # globals at call time, so patching the name here is sufficient.
    _mrs_starlet.map2alm = map2alm_fast
    _APPLIED["starlet_iter"] = n_iter


def _patch_neighbour_cache():
    """Cache hp.get_all_neighbours for the full-sky pixel-index call pattern."""
    if _APPLIED["neighbour_cache"]:
        return
    orig = hp.get_all_neighbours
    cache = {}

    @functools.wraps(orig)
    def get_all_neighbours_cached(nside, theta, phi=None, nest=False, lonlat=False):
        # Only the exact pattern get_peaks_sphere uses is cacheable:
        #   get_all_neighbours(nside, np.arange(npix))   [pixel-index mode]
        # Detect it cheaply (size + integer dtype + endpoints) to avoid an
        # O(npix) array compare on every call.
        if phi is None and not lonlat and isinstance(theta, np.ndarray):
            npix = hp.nside2npix(nside)
            if (
                theta.ndim == 1
                and theta.size == npix
                and theta.dtype.kind in "iu"
                and theta[0] == 0
                and theta[-1] == npix - 1
            ):
                key = (int(nside), bool(nest))
                if key not in cache:
                    cache[key] = orig(nside, theta, nest=nest)
                return cache[key]
        return orig(nside, theta, phi=phi, nest=nest, lonlat=lonlat)

    get_all_neighbours_cached._pycs_speedups_orig = orig
    # hos_peaks_l1 calls the healpy global (it does ``import healpy as hp``),
    # so patching healpy.get_all_neighbours reaches it.
    hp.get_all_neighbours = get_all_neighbours_cached
    _APPLIED["neighbour_cache"] = True


def enable(starlet_iter=1, cache_neighbours=True):
    """Apply the pycs spherical-wavelet speedups. Idempotent.

    Parameters
    ----------
    starlet_iter : int
        ``iter`` value forced for the starlet forward ``map2alm``. 1 is a
        conservative default (3 SHTs); 0 is the fastest (1 SHT) with a measured
        <0.4% effect on the L1 norms.
    cache_neighbours : bool
        Cache ``hp.get_all_neighbours`` per nside (helps peak counts only).

    Returns
    -------
    dict
        Snapshot of which patches are active.
    """
    _patch_starlet_iter(starlet_iter)
    if cache_neighbours:
        _patch_neighbour_cache()
    return dict(_APPLIED)
