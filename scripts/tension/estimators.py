"""Tension estimators between a null and a biased posterior.

Headline estimator is tensiometer's Gaussian ``Q_DM`` (difference of means under a
Gaussian approximation of both posteriors), exactly as the paper used it. ``Q_DM`` is
converted to a probability-to-exceed via the chi^2 CDF and then to an equivalent number
of sigma. The public entry point is :func:`tension_sigma`, which dispatches on an
``estimator`` name so a non-Gaussian parameter-shift estimator can be slotted in later
without touching callers.

NOTE on the historical bug this fixes: ``compute_tension_statistics.py`` imported
``tensiometer.utilities as utilities5`` but called ``utilities.from_confidence_to_sigma``
-> NameError, swallowed by a try/except -> silently empty tables. Here the import name
and the call site agree.
"""
from typing import Optional, Sequence

import numpy as np
import scipy.stats
from getdist import MCSamples

import tensiometer.utilities as utilities
from tensiometer import gaussian_tension

# Cosmological parameters carried in the posterior columns, in order.
PARAM_NAMES = ["Omega_m", "S8", "w0", "H0", "ns", "Omega_b"]
PARAM_LABELS = [r"\Omega_m", "S_8", "w_0", "H_0", "n_s", r"\Omega_b"]
# The 3-parameter subset used for the headline tension.
SUBSET_INDICES = (0, 1, 2)
TRUTH = (0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493)


def make_mcsamples(
    samples: np.ndarray,
    indices: Optional[Sequence[int]] = None,
    label: str = "",
) -> MCSamples:
    """Wrap a (n_samples, n_params) array as getdist MCSamples.

    If ``indices`` is given, only those parameter columns are kept (e.g. the 3-param
    subset). Names/labels are taken from the module-level metadata.
    """
    if indices is not None:
        indices = list(indices)
        samples = samples[:, indices]
        names = [PARAM_NAMES[i] for i in indices]
        labels = [PARAM_LABELS[i] for i in indices]
    else:
        n = samples.shape[1]
        names = PARAM_NAMES[:n]
        labels = PARAM_LABELS[:n]
    return MCSamples(samples=samples, names=names, labels=labels, label=label)


def q_dm_tension(null: MCSamples, biased: MCSamples):
    """Gaussian Q_DM tension between two posteriors.

    Returns a dict with the raw statistic, its degrees of freedom, the probability
    P = chi2.cdf(Q_DM, dofs), and the equivalent ``nsigma``. Returns NaNs (not an
    exception) if tensiometer fails, so a sweep can record a flagged row and continue.
    """
    try:
        q_dm, dofs = gaussian_tension.Q_DM(null, biased)
        p = float(scipy.stats.chi2.cdf(q_dm, dofs))
        nsigma = float(utilities.from_confidence_to_sigma(p))
        return {"Q_DM": float(q_dm), "dofs": int(dofs), "P": p, "nsigma": nsigma, "ok": True}
    except Exception as exc:  # noqa: BLE001 - intentionally broad; report, don't crash the sweep
        return {"Q_DM": np.nan, "dofs": 0, "P": np.nan, "nsigma": np.nan,
                "ok": False, "error": repr(exc)}


def tension_sigma(
    null_samples: np.ndarray,
    biased_samples: np.ndarray,
    indices: Optional[Sequence[int]] = None,
    estimator: str = "q_dm",
):
    """Top-level tension between a null and a biased posterior array.

    ``indices`` selects the parameter subset (None = all 6). ``estimator`` dispatches;
    only ``"q_dm"`` exists today, with a clear hook for a non-Gaussian estimator later.
    """
    null = make_mcsamples(null_samples, indices, label="null")
    biased = make_mcsamples(biased_samples, indices, label="biased")
    if estimator == "q_dm":
        return q_dm_tension(null, biased)
    raise ValueError(f"unknown estimator {estimator!r} (have: 'q_dm')")
