"""Single source of truth for the baryon-tension output tree and posterior filenames.

Every read/write in this analysis derives its path from here, so the layout cannot drift.
Tree (see docs/PLAN_tension_submean_l37.md §8):

    outputs/baryon_tension/<campaign>/
        manifest.json
        posteriors/mask_<AAAAA>/<role>/<posterior>.npy     # role in {null, biased}
        qa/   logs/   tables/   figures/

Campaign tag = "<statistic>_<gauge>_l<lmin>", e.g. "ps_submean_l37".
Footprint dirs are zero-padded to 5 digits so they sort naturally.

The posterior FILENAME is reproduced bit-for-bit from
run_npe_inference_auto_cross_ps_master.py (so the worker's --samples-dir output lands
where io.py later looks). Verified against on-disk files in tests/test_paths.py.
"""
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO = Path("/mnt/home/tersenov/software/bar_impact")
ROOT = REPO / "outputs" / "baryon_tension"
LEGACY_SAMPLES = REPO / "outputs" / "samples"  # paper posteriors (flat, raw, lmin=100)

# Posterior roles and the fiducial-type token each maps to in the filename.
NULL = "null"        # nobaryons observation vs nobaryons model  -> on truth
BIASED = "biased"    # baryonified observation vs nobaryons model -> feedback-shifted
FID_BY_ROLE = {NULL: "nobaryons", BIASED: "baryonified"}
ROLES = (NULL, BIASED)


# --------------------------------------------------------------------------- tree

def campaign_tag(statistic: str = "ps", gauge: str = "submean", lmin: int = 37) -> str:
    return f"{statistic}_{gauge}_l{lmin}"


def campaign_dir(tag: str) -> Path:
    return ROOT / tag


def area_dirname(area) -> str:
    """Sortable per-footprint dir name; the full-sky sentinel "fullsky" maps to 'fullsky'."""
    return "fullsky" if area == "fullsky" else f"mask_{int(area):05d}"


def posteriors_dir(tag: str, area, role: str) -> Path:
    assert role in ROLES, f"role must be one of {ROLES}, got {role!r}"
    return campaign_dir(tag) / "posteriors" / area_dirname(area) / role


def qa_dir(tag: str) -> Path:
    return campaign_dir(tag) / "qa"


def logs_dir(tag: str) -> Path:
    return campaign_dir(tag) / "logs"


def tables_dir(tag: str) -> Path:
    return campaign_dir(tag) / "tables"


def figures_dir(tag: str) -> Path:
    return campaign_dir(tag) / "figures"


def manifest_path(tag: str) -> Path:
    return campaign_dir(tag) / "manifest.json"


def ensure_campaign_tree(tag: str, areas: Iterable[float]) -> None:
    """Create every directory the campaign writes into."""
    for d in (qa_dir(tag), logs_dir(tag), tables_dir(tag), figures_dir(tag)):
        d.mkdir(parents=True, exist_ok=True)
    for area in areas:
        for role in ROLES:
            posteriors_dir(tag, area, role).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- posterior names

def ps_posterior_filename(
    *,
    fiducial: str,
    lower: int,
    upper: int,
    area: float,
    rebin: int = 10,
    apod: float = 2.0,
    noise: float = 0.26,
    submean: bool = True,
    run: Optional[int] = None,
    bins: str = "1234",
    sim: str = "nobaryons",
    bnt: bool = False,
    cuts: Optional[Sequence[int]] = None,
) -> str:
    """Reproduce the worker's PS auto+cross posterior filename exactly.

    Worker assembly (run_npe_inference_auto_cross_ps_master.py):
      posterior_samples_[bnt_]ps_auto_cross_{sim}_vs_{fid}_bins{bins}_l{lo}-{cut_desc}[_r{rebin}]
        _masked_{int(area)}sqdeg_apod{apod}_master[_submean]_noisy_s{noise:.2f}[_run{N}].npy

    `cut_desc` is `{up}` for a uniform cut, or `{c1}_{c2}_{c3}_{c4}` when the per-bin `cuts`
    differ (BNT bin-1 sweep) — matching the worker's `len(set(upper_cuts)) == 1` branch.
    """
    bnt_prefix = "bnt_" if bnt else ""
    if cuts is not None and len(set(cuts)) > 1:
        cut_desc = "_".join(str(c) for c in cuts)
    else:
        cut_desc = str(upper)
    name = (f"posterior_samples_{bnt_prefix}ps_auto_cross_{sim}_vs_{fiducial}"
            f"_bins{bins}_l{lower}-{cut_desc}")
    if rebin and rebin > 1:
        name += f"_r{rebin}"
    name += f"_masked_{int(area)}sqdeg_apod{apod}_master"
    if submean:
        name += "_submean"
    name += f"_noisy_s{noise:.2f}"
    if run is not None:
        name += f"_run{run}"
    return name + ".npy"


def ps_posterior_path(
    tag: str, *, role: str, lower: int, upper: int, area: float,
    run: Optional[int] = None, **kw,
) -> Path:
    """Full path to a campaign posterior, inside the organized tree."""
    filename = ps_posterior_filename(
        fiducial=FID_BY_ROLE[role], lower=lower, upper=upper, area=area, run=run, **kw,
    )
    return posteriors_dir(tag, area, role) / filename


def fullsky_posterior_filename(
    *,
    fiducial: str,
    lower: int,
    upper: int,
    rebin: int = 10,
    noise: float = 0.26,
    run: Optional[int] = None,
    bins: str = "1234",
    sim: str = "nobaryons",
    bnt: bool = False,
    cuts: Optional[Sequence[int]] = None,
) -> str:
    """Reproduce the healpy full-sky worker's PS auto+cross posterior filename.

    run_npe_inference_auto_cross_ps.py (no mask): posterior_samples_[bnt_]ps_auto_cross_
      {sim}_vs_{fid}_bins{bins}_l{lo}-{cut_desc}[_r{rebin}]_noisy_s{n:.2f}[_run{N}]_npe.npy
      (trailing _npe; no mask/submean tags for full sky).

    `cut_desc` is `{up}` for a uniform cut, or the per-bin cuts joined with HYPHENS when they
    differ (BNT bin-1 sweep) — the healpy worker uses '-'.join (line 958), NOT the master
    worker's '_'.join. Matching it exactly is what lets the sweep's resume-check and the tension
    loader find the file.
    """
    bnt_prefix = "bnt_" if bnt else ""
    if cuts is not None and len(set(cuts)) > 1:
        cut_desc = "-".join(str(c) for c in cuts)
    else:
        cut_desc = str(upper)
    name = (f"posterior_samples_{bnt_prefix}ps_auto_cross_{sim}_vs_{fiducial}"
            f"_bins{bins}_l{lower}-{cut_desc}")
    if rebin and rebin > 1:
        name += f"_r{rebin}"
    name += f"_noisy_s{noise:.2f}"
    if run is not None:
        name += f"_run{run}"
    return name + "_npe.npy"


def fullsky_posterior_path(tag: str, *, role: str, lower: int, upper: int,
                           run: Optional[int] = None, **kw) -> Path:
    filename = fullsky_posterior_filename(
        fiducial=FID_BY_ROLE[role], lower=lower, upper=upper, run=run, **kw)
    return posteriors_dir(tag, "fullsky", role) / filename


def legacy_ps_posterior_path(
    *,
    fiducial: str,
    lower: int,
    upper: int,
    area: float,
    rebin: int = 10,
    apod: float = 2.0,
    noise: float = 0.26,
    run: Optional[int] = None,
    bins: str = "1234",
    sim: str = "nobaryons",
) -> Path:
    """Path to a PAPER (raw, flat-layout) posterior in outputs/samples.

    Same as the campaign name but without the `_submean` tag and not nested in the tree
    — used only by the Stage-2 regression gate to reproduce the published numbers.
    """
    name = f"posterior_samples_ps_auto_cross_{sim}_vs_{fiducial}_bins{bins}_l{lower}-{upper}"
    if rebin and rebin > 1:
        name += f"_r{rebin}"
    name += f"_masked_{int(area)}sqdeg_apod{apod}_master_noisy_s{noise:.2f}"
    if run is not None:
        name += f"_run{run}"
    return LEGACY_SAMPLES / (name + ".npy")
