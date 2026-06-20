"""σ-vs-upper-cut figure in the paper_plots.ipynb style (one panel per footprint).

Renders the aggregated tension (mean ± std over runs) as the notebook cell does: a row of
panels, errorbars, a dashed threshold line at 0.3σ. Adapts the panel count to the footprints
that actually have data, so it works for the single-footprint pilot and the full six.
"""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _apply_rcparams():
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14


def plot_nsigma_vs_cut(
    agg_df,
    areas: Sequence[int],
    out_pdf: str,
    out_png: Optional[str] = None,
    threshold: float = 0.3,
    subtitle: str = "",
    dedup: bool = True,
) -> int:
    """Plot mean±std nσ vs upper-cut, one panel per footprint with data.

    `agg_df` has columns area, upper_cut, mean, std, n (from aggregate.aggregate_runs).
    With `dedup` (default), consecutive cuts that yield a byte-identical posterior (same
    (mean,std) — i.e. the same data vector under the coarse binning) are collapsed to one
    point at the lowest such cut, giving a clean curve. No-op when cuts are all distinct
    (e.g. the step-40 grid). Returns the number of panels drawn (0 if no data yet).
    """
    _apply_rcparams()
    areas_with_data = [a for a in areas if (agg_df["area"] == a).any()]
    n = len(areas_with_data)
    if n == 0:
        return 0

    fig, axes = plt.subplots(1, n, figsize=(max(5.0, 3.4 * n), 4.2), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        area = areas_with_data[i]
        sub = agg_df[agg_df["area"] == area].sort_values("upper_cut")
        if dedup and len(sub) > 1:
            # keep a row unless its (mean,std) equals the previous row's (identical fit)
            changed = sub[["mean", "std"]].round(9).diff().abs().sum(axis=1) > 1e-9
            changed.iloc[0] = True
            sub = sub[changed]
        ax.errorbar(sub["upper_cut"], sub["mean"], yerr=sub["std"],
                    fmt="o", color="C0", elinewidth=1.5, markersize=6, capsize=4)
        ax.axhline(y=threshold, color="r", linestyle="--", linewidth=1.5,
                   label=f"Threshold ({threshold})")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_title("Full sky" if area == "fullsky" else rf"Area = {area} deg$^2$")
        if i == 0:
            ax.set_ylabel(r"Significance ($n_\sigma$)")

    fig.supxlabel(r"Upper Cut ($\ell_{\mathrm{max}}$)", fontsize=15, y=0.02)
    if subtitle:
        fig.text(0.5, 0.97, subtitle, ha="center", va="top", fontsize=10, color="0.4")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.18, wspace=0.2)

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return n
