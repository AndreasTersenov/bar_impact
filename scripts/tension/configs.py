"""Declarative campaign specs: everything needed to LOCATE the posteriors for one run.

A `PSCampaign` bundles the analysis settings (gauge, ℓ-floor, footprints, cut grid, runs,
layout) and knows how to turn a (role, area, upper_cut, run) coordinate into a path via
`paths.py`. Two layouts:

  - "tree": the new organized output (outputs/baryon_tension/<tag>/…), used by the sweep.
  - "flat": the legacy paper posteriors in outputs/samples, used only by the regression gate.

Other statistics (l1, peaks, ps_fullsky) get their own campaign classes later; the tension
loop in compute_tension.py only needs `.coords()` and `.posterior_path()`.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

from . import paths

# The paper's scale-cut grid: fix the ℓ-floor, walk the upper cut down.
PAPER_UPPER_CUTS = tuple(range(340, 1021, 20))   # 340, 360, …, 1020 (35 values)
ALL_AREAS = (2000, 5000, 10000, 14000, 28000, 35000)


@dataclass
class PSCampaign:
    """PS auto+cross campaign. `submean`+`lmin` define the gauge; `layout` the on-disk home."""
    lmin: int = 37
    submean: bool = True
    layout: str = "tree"                       # "tree" (new) | "flat" (legacy paper)
    rebin: int = 10
    areas: Sequence[int] = ALL_AREAS
    upper_cuts: Sequence[int] = PAPER_UPPER_CUTS
    runs: Sequence[Optional[int]] = (None,)     # (None,) = single base run; e.g. (1,2,3,4,5)
    statistic: str = "ps"
    pipeline: str = "master"                    # "master" (masked NaMaster) | "healpy" (full-sky)

    @property
    def gauge(self) -> str:
        return "submean" if self.submean else "raw"

    @property
    def tag(self) -> str:
        if self.pipeline == "healpy":
            return f"{self.statistic}_fullsky_l{self.lmin}"
        return paths.campaign_tag(self.statistic, self.gauge, self.lmin)

    def coords(self):
        """Yield (area, upper_cut) over the full grid."""
        for area in self.areas:
            for upper_cut in self.upper_cuts:
                yield area, upper_cut

    def posterior_path(self, role: str, area, upper_cut: int, run: Optional[int] = None):
        if self.pipeline == "healpy":
            return paths.fullsky_posterior_path(
                self.tag, role=role, lower=self.lmin, upper=upper_cut, run=run, rebin=self.rebin)
        if self.layout == "tree":
            return paths.ps_posterior_path(
                self.tag, role=role, lower=self.lmin, upper=upper_cut, area=area,
                run=run, rebin=self.rebin, submean=self.submean,
            )
        if self.layout == "flat":
            return paths.legacy_ps_posterior_path(
                fiducial=paths.FID_BY_ROLE[role], lower=self.lmin, upper=upper_cut,
                area=area, run=run, rebin=self.rebin,
            )
        raise ValueError(f"unknown layout {self.layout!r} (have 'tree','flat')")


# Convenience constructors for the two cases we use immediately.
def submean_l37_campaign(**overrides) -> PSCampaign:
    """The new monopole-subtracted, ℓ≥37 campaign (tree layout)."""
    defaults = dict(lmin=37, submean=True, layout="tree")
    defaults.update(overrides)
    return PSCampaign(**defaults)


def paper_raw_l100_campaign(**overrides) -> PSCampaign:
    """The published raw, ℓ≥100 analysis (flat layout) — for the regression gate."""
    defaults = dict(lmin=100, submean=False, layout="flat")
    defaults.update(overrides)
    return PSCampaign(**defaults)


def fullsky_campaign(**overrides) -> PSCampaign:
    """Full-sky (healpy pipeline) ℓ≥37 campaign — the f_sky→1 endpoint.

    Single pseudo-footprint "fullsky" (no mask). Healpy is per-ℓ (≈10-ℓ bins after rebin),
    a different/finer estimator than the masked NaMaster nlb=4 (40-ℓ) panels — see the
    plan's option-(a) caveat: same scale-cut trend, not magnitude-comparable.
    """
    defaults = dict(lmin=37, pipeline="healpy", areas=("fullsky",))
    defaults.update(overrides)
    return PSCampaign(**defaults)
