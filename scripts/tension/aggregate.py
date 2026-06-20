"""Aggregate per-(coordinate, run) tension records into mean ± std tables.

Input is a long-form list of dicts, one per (area, upper_cut, run, param-space), each with
a `nsigma` (+ Q_DM, dofs, P) and an `ok` flag. Output is:
  - a long-form DataFrame (everything, including failed/flagged rows for the record), and
  - per-coordinate mean/std/n over the GOOD runs, plus a pivot (rows=upper_cut, cols=area).

QA flagging (qa.py) sets `ok=False` on bad runs; those are excluded from mean/std but kept
in the long-form table so nothing is silently dropped.
"""
from typing import Sequence

import numpy as np
import pandas as pd


def to_long(records: Sequence[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def aggregate_runs(
    long_df: pd.DataFrame,
    coord_cols: Sequence[str] = ("area", "upper_cut"),
    value: str = "nsigma",
    min_runs: int = 1,
) -> pd.DataFrame:
    """Mean/std/n of `value` per coordinate over rows with ok==True.

    Coordinates with fewer than `min_runs` good runs are dropped — used for interim
    snapshots so the curve only shows fully-sampled cuts (no misleading partial-n points).
    """
    good = long_df[long_df.get("ok", True) == True]  # noqa: E712 - explicit, NaN-safe
    grouped = good.groupby(list(coord_cols))[value]
    out = grouped.agg(mean="mean", std="std", n="count").reset_index()
    n_total = long_df.groupby(list(coord_cols))[value].size().rename("n_total").reset_index()
    out = out.merge(n_total, on=list(coord_cols), how="right")
    out["n"] = out["n"].fillna(0).astype(int)
    out["n_excluded"] = out["n_total"] - out["n"]
    if min_runs > 1:
        out = out[out["n"] >= min_runs].reset_index(drop=True)
    return out


def pivot(agg_df: pd.DataFrame, index: str = "upper_cut", columns: str = "area",
          values: str = "mean") -> pd.DataFrame:
    return agg_df.pivot(index=index, columns=columns, values=values)
