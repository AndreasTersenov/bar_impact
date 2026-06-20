"""Loading posteriors off disk for the tension stage.

Thin and dumb on purpose: resolve a path (via a campaign), load the array, return None if
it's missing so the caller can record a gap and continue. The MCSamples wrapping and the
tension math live in estimators.py.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_posterior(path: Path) -> Optional[np.ndarray]:
    """Load a posterior-sample .npy as an (n_samples, n_params) array, or None if absent.

    Returns None on a read error too (e.g. a file caught mid-write during a live snapshot),
    so the caller treats it as not-yet-available rather than crashing.
    """
    if not Path(path).exists():
        return None
    try:
        return np.load(path)
    except (ValueError, OSError, EOFError):
        return None


def load_pair(campaign, area: int, upper_cut: int, run: Optional[int] = None
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (null_samples, biased_samples) for one coordinate, each None if missing."""
    from . import paths
    null = load_posterior(campaign.posterior_path(paths.NULL, area, upper_cut, run))
    biased = load_posterior(campaign.posterior_path(paths.BIASED, area, upper_cut, run))
    return null, biased
