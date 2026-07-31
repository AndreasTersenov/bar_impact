#!/usr/bin/env python3
"""Low-resolution smoke test of the NaMaster code path in fisher_gaussian_cov.py.

WHY: the real rebuild is a ~20 h job at nside=512 / lmax=1535, and the NaMaster it now runs against
is a freshly installed 2.4 rather than whatever the pre-crash work used. An API or signature
mismatch that only surfaces after the covariance-coupling step would waste the whole allocation.
This exercises EVERY NaMaster call the real script makes — mask, field, bin, MCM workspace,
covariance workspace (write AND read-back), and gaussian_covariance — at nside=64, in seconds.

It deliberately writes its workspaces to a scratch cache dir so it cannot touch the real (and
partly quarantined) cache_gaussian_cov/ contents.

  PYTHONNOUSERSITE=1 FISHER_AREA=14000 \
    /lustre/fswork/projects/rech/nzu/ulx34io/envs/namaster2/bin/python \
    scripts/diagnostics/smoke_gaussian_cov_api.py
"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import fisher_gaussian_cov as G
    import pymaster as nmt

    print(f"[smoke] pymaster {nmt.__version__}")
    print(f"[smoke] production geometry would be NSIDE={G.NSIDE} LMAX={G.LMAX} NLB={G.NLB}")

    # shrink the geometry and redirect the cache; the module reads these as globals inside functions
    scratch = tempfile.mkdtemp(prefix="smoke_gcov_")
    G.NSIDE, G.LMAX, G.NLB, G.CACHE = 64, 191, 4, scratch
    print(f"[smoke] running at NSIDE={G.NSIDE} LMAX={G.LMAX}, cache -> {scratch}\n")

    t0 = time.time()
    mask, fsky = G.create_apodized_mask(G.NSIDE, G.AREA, G.CENTER, G.APOD_TYPE, G.APOD_DEG)
    print(f"[1] mask OK  npix={mask.size} f_sky={fsky:.4f} "
          f"nonzero={float((mask > 0).mean()):.4f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    b, w, cw = G.build_workspaces(mask)
    eff_ell = b.get_effective_ells()
    nbpw = len(eff_ell)
    print(f"[2] workspaces OK  n_bandpowers={nbpw} "
          f"(eff_ell {eff_ell[0]:.1f}..{eff_ell[-1]:.1f}) ({time.time()-t0:.1f}s)")

    # read-back path: the production run caches and re-reads these, so exercise it
    t0 = time.time()
    b2, w2, cw2 = G.build_workspaces(mask)
    print(f"[3] cache read-back OK ({time.time()-t0:.1f}s)")

    # gaussian_covariance with the exact argument shape assemble_native_cov uses
    t0 = time.time()
    flat = np.ones(G.LMAX + 1) * 1e-10
    covar = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [flat], [flat], [flat], [flat], w, w)
    blk = np.asarray(covar).reshape(nbpw, nbpw)
    sym = float(np.abs(blk - blk.T).max() / max(np.abs(blk).max(), 1e-300))
    print(f"[4] gaussian_covariance OK  block {blk.shape} rel-asymmetry={sym:.2e} "
          f"({time.time()-t0:.1f}s)")

    # unbin_to_per_ell, used to turn measured bandpowers into per-ell theory
    per_ell = G.unbin_to_per_ell(b, np.ones(nbpw) * 1e-10)
    print(f"[5] unbin_to_per_ell OK  len={len(per_ell)} finite={np.isfinite(per_ell).all()}")

    # a 2-spectrum assembly, to check the block bookkeeping and PSD-ness of the result
    t0 = time.time()
    theory = {(i, j): flat for i in (1, 2, 3, 4) for j in (1, 2, 3, 4) if i <= j}
    C = G.assemble_native_cov(b, w, cw, theory, nbpw)
    ev = np.linalg.eigvalsh(C)
    psd = ev.min() > -1e-10 * abs(ev.max())
    print(f"[6] assemble_native_cov OK  {C.shape}  PSD={psd}  "
          f"trace==sum(eig): {np.isclose(np.trace(C), ev.sum())} ({time.time()-t0:.1f}s)")

    print("\n[smoke] ALL NAMASTER CALLS PASS — the 2.4 API matches what the script expects.")
    print(f"[smoke] scratch cache left at {scratch} (safe to delete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
