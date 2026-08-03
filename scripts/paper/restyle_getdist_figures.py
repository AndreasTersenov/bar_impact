#!/usr/bin/env python3
"""Re-run the getdist corner-plot generators so their output picks up the A&A style.

WHY A DRIVER. `figures.py republish` re-copies each figure from its recorded SOURCE path; it does
not re-run the generator. A style change lives in the generator, so the source files themselves have
to be rebuilt first. Every published figure records the exact command that made it, so this reads
those back rather than retyping long --series arguments and getting one subtly wrong.

WHAT CHANGED IN THE GENERATORS. Per the figure-polish checklist for corner plots: load the style
BEFORE constructing the plotter (getdist reads font family and sizes from rcParams at construction
time, so loading it afterwards does nothing), and set the plotter width to the target column --
A&A single, 3.465 in, for these 3-parameter posteriors. getdist's own contour colours are left
alone; the checklist is explicit that its defaults are tuned for overlapping posteriors and that
overriding them usually makes things worse.

Run under jaxili: aname pins getdist 1.4.3, which calls QuadContourSet.tcolors, removed in
matplotlib 3.8, and dies on any filled contour.

  PYTHONNOUSERSITE=1 <jaxili python> scripts/paper/restyle_getdist_figures.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = ("plot_posterior_overlay", "plot_score_contours_debiased")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on the slug")
    a = ap.parse_args()

    figdir = os.path.join(REPO, "paper", "figures")
    jobs = []
    for slug in sorted(os.listdir(figdir)):
        pj = os.path.join(figdir, slug, "provenance.json")
        if not os.path.exists(pj):
            continue
        try:
            p = json.load(open(pj))
        except Exception:
            continue
        gen = p.get("generator", "")
        if not any(t in gen for t in TARGETS):
            continue
        if a.only and a.only not in slug:
            continue
        cmd = p.get("command")
        if not cmd:
            print(f"[skip] {slug}: no command recorded")
            continue
        jobs.append((slug, cmd))

    print(f"{len(jobs)} getdist figures to rebuild\n")
    fails = []
    for i, (slug, cmd) in enumerate(jobs, 1):
        # Two provenance formats exist. Newer figures record shlex.join(sys.argv), which
        # round-trips through shlex.split. Older ones recorded " ".join(sys.argv), which drops the
        # quoting, so a --series value with spaces comes back as several tokens; those need
        # splitting on option boundaries instead. Distinguish them exactly rather than by guessing:
        # a shlex-quoted string is the one that survives a split/join round-trip.
        try:
            toks = shlex.split(cmd)
            quoted = shlex.join(toks) == cmd
        except ValueError:
            toks, quoted = [], False
        if quoted:
            argv = toks
        else:
            argv = []
            for chunk in re.split(r" (?=--[a-zA-Z])", cmd.strip()):
                if chunk.startswith("--"):
                    head, _, tail = chunk.partition(" ")
                    argv.append(head)
                    if tail:
                        argv.append(tail)
                else:
                    argv.extend(chunk.split())
        # --paper was added and then reverted during a style detour; drop it if a stale
        # provenance still carries it.
        argv = [x for x in argv if x != "--paper"]
        if argv and argv[0].endswith(".py"):
            argv = [sys.executable] + argv
        print(f"[{i}/{len(jobs)}] {slug}")
        if a.dry_run:
            print("   ", " ".join(shlex.quote(x) for x in argv)[:200])
            continue
        r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(slug)
            print(f"    FAILED rc={r.returncode}")
            print("    " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:200])
        else:
            print("    ok")
    print(f"\ndone: {len(jobs) - len(fails)} ok, {len(fails)} failed")
    if fails:
        print("failed:", ", ".join(fails))
        sys.exit(1)
    print("Now run:  scripts/paper/figures.py republish")


if __name__ == "__main__":
    main()
