"""Shared tension-collection logic: loop the campaign grid × runs, compute Q_DM tension.

Used by both the one-shot entrypoint (compute_tension.py) and the live monitor, so the two
can never drift. Read-only on posteriors — safe to run while the sweep is still producing.
"""
from typing import Tuple

from . import io
from .estimators import SUBSET_INDICES, tension_sigma


def collect_records(camp) -> Tuple[list, list, int]:
    """Return (records_full, records_subset, n_missing) over the whole grid × runs.

    Each record is one (area, upper_cut, run) tension result. Coordinates whose null or
    biased posterior is missing are skipped and counted (n_missing) — so a partial,
    mid-sweep run just yields fewer records, never an error.
    """
    rec_full, rec_sub, n_missing = [], [], 0
    for area, upper_cut in camp.coords():
        for run in camp.runs:
            null, biased = io.load_pair(camp, area, upper_cut, run)
            if null is None or biased is None:
                n_missing += 1
                continue
            base = {"area": area, "upper_cut": upper_cut, "run": -1 if run is None else run}
            rec_full.append({**base, **tension_sigma(null, biased, indices=None)})
            rec_sub.append({**base, **tension_sigma(null, biased, indices=SUBSET_INDICES)})
    return rec_full, rec_sub, n_missing
