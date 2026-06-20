"""Shared library for the baryon-bias tension-vs-scale-cut analysis.

Stages:
  estimators.py  -- Gaussian Q_DM tension between two posteriors (paper's estimator).
  paths.py       -- single source of truth for the output tree + filename builders.
  io.py          -- load posterior .npy -> MCSamples (new tree + legacy flat layout).
  qa.py          -- per-posterior diagnostics gate (NaN / degeneracy / null-on-truth / outliers).
  sweep.py       -- config-driven NPE runner (seeding, NaN-retry, QA, resume).
  aggregate.py   -- per-(config, run) sigma -> mean +/- std tables + pivots.
  configs.py     -- one declarative config per statistic (ps, ps_fullsky, l1, peaks).
  plots.py       -- sigma-vs-cut curves with run-scatter error bands.

Design note: this is a standalone package under scripts/, intentionally NOT plugged into
the half-finished src/bar_impact/ (see CLAUDE.md). It imports third-party libraries
directly, matching the repo's "scripts are the source of truth" convention.
"""
