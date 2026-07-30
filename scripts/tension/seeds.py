"""Pick the most REPRESENTATIVE NPE training seed from an ensemble of posteriors.

Why this exists. A pooled contour stacks every seed's samples, so its width includes the
seed-to-seed training scatter on top of the statistical width. Measured on this project that
inflation is only 1-5% in sigma, so pooling is not distorting — but a pooled contour is still
not the object a real analysis reports. A survey trains ONE density estimator and quotes ITS
posterior, and the field convention is to show a representative (or seed-averaged) posterior
with the training scatter reported separately. This module supplies the "representative" half.

THE CHOICE. "Closest to the mean" is ambiguous for a posterior, because a posterior has both a
location and a shape, and a seed can match the ensemble on one while missing badly on the
other. So the score has two dimensionless terms and the winner minimises their sum:

    centre term   max_p |mu_p - median(mu_p)| / w_p        (offset in posterior widths)
    width  term   max_p |ln(sigma_p / median(sigma_p))|    (log width ratio, so 2x too wide
                                                            and 2x too narrow score equally)

Both use the MEDIAN across seeds, not the mean, so one broken seed cannot drag the reference
it is being judged against — the same reasoning as the centre-outlier guard in
plot_contours_vs_area.py, and it matters here for the same reason: this project has seeds that
land at a prior edge with a perfectly normal width.

Worst-parameter (max) rather than a sum over parameters, because a seed that is representative
in Omega_m and S8 but a full width off in w0 is not representative; averaging would hide that.

Run the outlier guards BEFORE calling this. It picks the most typical member of whatever it is
given, so a contaminated ensemble yields a "representative" seed drawn from a contaminated
distribution.
"""
import numpy as np


def representative_seed(arrays, runs=None, params=(0, 1, 2)):
    """Return (index, run_label, diagnostics) of the most representative posterior.

    `arrays` is a list of (n_samples, n_params) sample arrays, one per seed.
    `runs` optionally labels them; defaults to positional indices.

    diagnostics carries the per-seed score breakdown so the choice is auditable rather than
    an unexplained pick — it goes into the figure's provenance.
    """
    if not arrays:
        raise ValueError("no posteriors given")
    runs = list(runs) if runs is not None else list(range(len(arrays)))
    if len(arrays) == 1:
        return 0, runs[0], {"note": "single seed available; nothing to choose",
                            "per_seed": [{"run": runs[0], "score": 0.0}]}

    mu = np.array([np.asarray(a)[:, params].mean(0) for a in arrays])      # (nseed, npar)
    sd = np.array([np.asarray(a)[:, params].std(0) for a in arrays])
    med_mu, med_sd = np.median(mu, 0), np.median(sd, 0)
    w = np.maximum(med_sd, 1e-12)

    centre = np.abs(mu - med_mu) / w                       # in posterior widths
    width = np.abs(np.log(np.maximum(sd, 1e-30) / w))      # log ratio
    score = centre.max(1) + width.max(1)
    i = int(np.argmin(score))

    diag = {
        "criterion": ("argmin over seeds of [max_p |mu_p - median mu_p| / width_p] + "
                      "[max_p |ln(sigma_p / median sigma_p)|]"),
        "chosen_run": runs[i],
        "per_seed": [{"run": runs[k],
                      "centre_offset_widths": float(centre[k].max()),
                      "log_width_ratio": float(width[k].max()),
                      "score": float(score[k])} for k in range(len(arrays))],
    }
    return i, runs[i], diag
