#!/usr/bin/env python3
"""Regenerate every published figure that has not yet been rebuilt on the current style.

WHY A DRIVER. `figures.py republish` re-COPIES from a recorded source; it never re-runs the
generator. The sigma_8 relabel and the shared-palette work live in the generators, so the
sources have to be rebuilt first. Of the 59 published figures, 23 record the exact command that
made them, 12 can have theirs reconstructed from structured provenance fields, and 9 are driven
by environment variables rather than CLI flags. This encodes all three cases in one place.

THREE GROUPS:

  A. RECORDED COMMAND (23). Replayed verbatim. Two provenance formats exist: newer figures use
     shlex.join(sys.argv), which round-trips through shlex.split; older ones used " ".join,
     which loses quoting so a --series value with spaces comes back as several tokens. They are
     told apart exactly -- a shlex-quoted string is the one that survives a split/join round
     trip -- rather than guessed at.

  B. THREE-STATS FAMILY (12). plot_contours_three_stats.py records mode / area_sqdeg / cut_mode
     / seed_mode / single_run / cuts, which is everything its CLI needs. --ps-lmax is passed
     ONLY when provenance says the cut was chosen explicitly; when it was chosen by the
     0.3-sigma rule, passing it would pin a number that should be re-derived.

     Their output stems will CHANGE: the published ones predate the _pooled/_single_seed
     suffix, so re-running writes e.g. ..._bsafe_l460_scales234_pooled where the recorded
     source was ..._bsafe_l460_scales234. The slug is what the paper cites, so the slug is kept
     and re-pointed at the new stem.

  C. ENV-VAR FIGURES (9). These generators take no CLI flags for the variant; they switch on
     environment variables and append a suffix to the output name (REFLINE=0 -> _noref,
     FULLSKY=1 -> _with_fullsky, GUIDE_SCALE=0.24 -> _lowanchor). Read out of the generators
     rather than guessed.

Skips anything already regenerated today, so it is safe to re-run.

  PYTHONNOUSERSITE=1 <jaxili python> scripts/paper/regen_remaining_figures.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
THREE = "scripts/diagnostics/plot_contours_three_stats.py"

# Replayed commands are stripped of these BEFORE running. The whole point of the replay is to
# pick up the generators' new defaults; an option recorded in 2026-08-03 that PINS the old
# behaviour would silently defeat that. --colors is the live case: the flagship overlays record
# "--colors C0,0.45", which would override the shared Okabe-Ito palette with the tab10 pair the
# migration exists to remove.
STRIP_OPTS_WITH_VALUE = {"--colors"}
STRIP_FLAGS = {"--paper"}

# Slugs whose recorded command must NOT be replayed as-is, because the figure's CONTENT has
# been superseded rather than merely restyled.
OVERRIDE = {
    # The recorded command draws the l1 BNT arm from the nobaryons run, because at the time no
    # usable baryonified one existed -- the original was prior-dominated. Job 570333 retrained
    # it (sigma_S8 0.107 -> 0.022), so the figure can finally be like-for-like: all three arms
    # baryonified. --outline-bnt keeps the uncut BNT contour from burying the two cut ones.
    "hos_bnt_l1_vs_scale_cut": [
        "scripts/plot_hos_bnt_triangle.py", "--stat", "l1", "--bnt-arm", "baryonified",
        "--outline-bnt", "--name", "hos_bnt_l1_vs_scale_cut"],
    "hos_bnt_peaks_vs_scale_cut": [
        "scripts/plot_hos_bnt_triangle.py", "--stat", "peaks", "--bnt-arm", "baryonified",
        "--outline-bnt", "--name", "hos_bnt_peaks_vs_scale_cut"],
}

# Group C: slug -> (script, env overrides). Suffixes are appended by the generators themselves.
ENV_FIGURES = {
    "ps_bias_vs_lmax":              ("scripts/diagnostics/plot_nsigma_vs_lmax.py", {"FULLSKY": "0"}),
    "ps_bias_vs_lmax_with_fullsky": ("scripts/diagnostics/plot_nsigma_vs_lmax.py", {"FULLSKY": "1"}),
    "bias_vs_area_three_stats":     ("scripts/diagnostics/plot_nsigma_vs_area.py", {"REFLINE": "0"}),
    "fom_vs_area":                  ("scripts/diagnostics/plot_fom_vs_area.py", {}),
    "fom_vs_area_low_guide":        ("scripts/diagnostics/plot_fom_vs_area.py", {"GUIDE_SCALE": "0.24"}),
    "constraining_power_vs_area":   ("scripts/diagnostics/plot_scaling_vs_area.py", {}),
    # These two write BOTH fisher_contours and fisher_contours_baryon_safe in one run, so the
    # second slug is intentionally mapped to the same invocation rather than run twice.
    "fisher_constraining_power_full":        ("scripts/diagnostics/fisher_constraining_power.py", {}),
    "fisher_constraining_power_baryon_safe": ("scripts/diagnostics/fisher_constraining_power.py", {}),
}

# Figures that legitimately need no rebuild, listed so nobody "fixes" their absence later.
#   bnt_bin1_vs_cut_optimal -- an n_sigma vs lmax line plot from plot_score_bnt_tension_14000.py.
#     Its axes are "Upper cut lmax" and "Baryon tension n_sigma"; it has NO parameter axis, so
#     the sigma_8 relabel does not reach it and the palette work does not apply. An earlier
#     version of this table pointed it at build_bnt_bin1_allareas_plot.py, which is a different
#     generator writing to plots/ rather than outputs/plots/bnt_ps_bin1_submean_l37/ -- it ran,
#     reported success, and left the published source untouched.
NO_REBUILD_NEEDED = {"bnt_bin1_vs_cut_optimal"}


def argv_from_command(cmd: str):
    try:
        toks = shlex.split(cmd)
        if shlex.join(toks) == cmd:
            return toks
    except ValueError:
        pass
    argv = []
    for chunk in re.split(r" (?=--[a-zA-Z])", cmd.strip()):
        if chunk.startswith("--"):
            head, _, tail = chunk.partition(" ")
            argv.append(head)
            if tail:
                argv.append(tail)
        else:
            argv.extend(chunk.split())
    return argv


def strip_opts(argv):
    """Drop options that would pin superseded behaviour on a replayed command."""
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in STRIP_FLAGS:
            continue
        if tok in STRIP_OPTS_WITH_VALUE:
            skip = True
            continue
        if any(tok.startswith(o + "=") for o in STRIP_OPTS_WITH_VALUE):
            continue
        out.append(tok)
    return out


def three_stats_argv(p):
    a = [THREE, "--mode", str(p["mode"]), "--area", str(p["area_sqdeg"]),
         "--cut-mode", str(p["cut_mode"]), "--seed-mode", str(p["seed_mode"])]
    if p.get("single_run") is not None:
        a += ["--single-run", str(p["single_run"])]
    cuts = p.get("cuts") or {}
    if cuts.get("power_spectrum_lmax_chosen_by") == "explicit --ps-lmax":
        a += ["--ps-lmax", str(cuts["power_spectrum_lmax"])]
    if cuts.get("hos_scale_tag"):
        a += ["--hos-scales", cuts["hos_scale_tag"]]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    jobs, skipped, unresolved = [], [], []
    seen_env_cmd = set()
    for pj in sorted(glob.glob(os.path.join(REPO, "paper/figures/*/provenance.json"))):
        slug = os.path.basename(os.path.dirname(pj))
        if a.only and a.only not in slug:
            continue
        try:
            p = json.load(open(pj))
        except Exception:
            unresolved.append((slug, "unreadable provenance"))
            continue
        if (p.get("generated_utc") or "").startswith("2026-08-04"):
            skipped.append(slug)
            continue

        gen = p.get("generator") or ""
        if slug in NO_REBUILD_NEEDED:
            skipped.append(f"{slug} (no parameter axis; nothing to restyle)")
            continue
        if slug in ENV_FIGURES:
            script, env = ENV_FIGURES[slug]
            key = (script, tuple(sorted(env.items())))
            if key in seen_env_cmd:      # fisher writes two figures per run
                skipped.append(f"{slug} (same run as a sibling)")
                continue
            seen_env_cmd.add(key)
            jobs.append((slug, [script], env))
        elif slug in OVERRIDE:
            jobs.append((slug, list(OVERRIDE[slug]), {}))
        elif p.get("command"):
            jobs.append((slug, strip_opts(argv_from_command(p["command"])), {}))
        elif "mode" in p and "cut_mode" in p:
            jobs.append((slug, three_stats_argv(p), {}))
        else:
            unresolved.append((slug, f"no command, no recognised fields (generator={gen!r})"))

    print(f"{len(jobs)} to regenerate | {len(skipped)} already current | {len(unresolved)} unresolved\n")
    for slug, argv, env in jobs:
        e = " ".join(f"{k}={v}" for k, v in env.items())
        print(f"  {slug:52s} {e + ' ' if e else ''}{' '.join(argv[:6])}")
    if unresolved:
        print("\nUNRESOLVED (left alone, reported rather than guessed):")
        for s, why in unresolved:
            print(f"  {s:52s} {why}")
    if a.dry_run:
        return

    fails = []
    for i, (slug, argv, env) in enumerate(jobs, 1):
        argv = [x for x in argv if x != "--paper"]
        if argv and argv[0].endswith(".py"):
            argv = [PY, "-u"] + argv
        runenv = dict(os.environ, **env)
        print(f"\n[{i}/{len(jobs)}] {slug}")
        r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, env=runenv)
        if r.returncode != 0:
            fails.append(slug)
            tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            print(f"    FAILED rc={r.returncode}: {tail[:220]}")
        else:
            print("    ok")

    print(f"\ndone: {len(jobs) - len(fails)} ok, {len(fails)} failed")
    if fails:
        print("failed:", ", ".join(fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
