#!/usr/bin/env python3
r"""Relabel the S-subscript-8 spelling to \sigma_8 across the repo.

The posteriors were always sigma_8; only the labels lied. Ran once on 2026-08-04: 83 edits
across 48 files. Kept because the reasoning below is the audit trail for that change, and
because a stray mislabel can reappear in any new generator.

WHY. Column 1 of cosmo_params.npy and of every posterior_samples_*.npy is sigma_8, verified
against the DATA rather than against any script's label list:

    col1 spans 0.400-1.397 with fiducial 0.84 -- the CosmoGrid fiducial sigma_8, sitting
    alongside Om=0.26, w0=-1, H0=67.36, ns=0.9649, Ob=0.0493, i.e. the rest of that TRUTH row.
    Read as the other parameter instead, the implied sigma_8 would span 0.313-2.347, which is
    unphysical, and the fiducial would be 0.84*sqrt(0.26/0.3) = 0.782, not 0.84.

No generator anywhere applies the sqrt(Om/0.3) conversion -- the samples are plotted raw. So
this was purely a labelling error and the fix is a relabel, NOT a conversion. No posterior,
FoM, tension value or scale cut changes.

SCOPE. A tokenize pass confirmed ZERO NAME tokens carry the misspelling, i.e. every occurrence
sat inside a string or a comment and none was a Python identifier. Internal keys spelled
without the underscore ("S8" dict keys, sig_S8 / sigma_S8 variables) are deliberately LEFT
ALONE: they never render, and renaming them would touch lookup logic for no visible gain.

THREE CASES, because a blind replace produces broken output in two of them:
  1. the string is exactly the misspelling -> a bare getdist label (siblings in the same list
     are raw LaTeX like r"\Omega_m"). Becomes r"\sigma_8"; plain "sigma_8" would render as
     literal text rather than as the symbol.
  2. the string contains $ or a backslash -> LaTeX. Gets \sigma_8, and an r prefix if it lacks
     one: a non-raw "$\sigma_8$" makes \s an invalid escape (SyntaxWarning on 3.12).
  3. anything else -- docstrings, print() prose -> plain "sigma_8", no backslash.

TWO SELF-PROTECTIONS, both learned the hard way. The first run walked scripts/ including this
file and rewrote its own docstring and its own replace() calls into no-ops:
  * this file is skipped by path, and
  * the search token is BUILT AT RUNTIME so the literal never appears in this source at all.
Do not "simplify" TOKEN back to a literal.

Usage:
    python3 scripts/paper/fix_sigma8_labels.py --dry-run     # review every edit first
    python3 scripts/paper/fix_sigma8_labels.py
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import tokenize

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELF = os.path.abspath(__file__)

# Built, not written literally -- see TWO SELF-PROTECTIONS above.
TOKEN = "S" + "_8"
LATEX = "\\sigma_8"
PROSE = "sigma_8"


def classify(tok_text: str) -> str:
    """Return the replacement mode for one STRING token: 'bare' | 'latex' | 'prose'."""
    i = 0
    while i < len(tok_text) and tok_text[i] not in "\"'":
        i += 1
    quoted = tok_text[i:]
    for q in ('"""', "'''", '"', "'"):
        if quoted.startswith(q) and quoted.endswith(q) and len(quoted) >= 2 * len(q):
            content = quoted[len(q):-len(q)]
            triple = len(q) == 3
            break
    else:
        return "prose"
    if content.strip() == TOKEN and not triple:
        return "bare"
    if "$" in content or "\\" in content:
        return "latex"
    return "prose"


def needs_raw(tok_text: str) -> bool:
    i = 0
    while i < len(tok_text) and tok_text[i] not in "\"'":
        i += 1
    return "r" not in tok_text[:i].lower()


def rewrite(path: str):
    src = open(path, encoding="utf-8").read()
    if TOKEN not in src:
        return None, []
    lines = src.splitlines(keepends=True)
    edits = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception as e:
        return None, [("TOKENIZE-FAIL", str(e))]
    for tok in toks:
        if TOKEN not in tok.string:
            continue
        if tok.type == tokenize.COMMENT:
            new = tok.string.replace(TOKEN, PROSE)
        elif tok.type == tokenize.STRING:
            mode = classify(tok.string)
            new = tok.string.replace(TOKEN, LATEX if mode in ("bare", "latex") else PROSE)
            if mode in ("bare", "latex") and needs_raw(new):
                new = "r" + new
        else:
            continue
        if new != tok.string:
            edits.append((tok.start, tok.end, tok.string, new))
    if not edits:
        return None, []
    # Apply from the END so earlier positions stay valid. Multi-line tokens are rebuilt across
    # their whole span rather than assumed single-line.
    out = lines[:]
    for (srow, scol), (erow, ecol), old, new in sorted(edits, key=lambda e: e[0], reverse=True):
        srow -= 1
        erow -= 1
        if srow == erow:
            out[srow] = out[srow][:scol] + new + out[srow][ecol:]
        else:
            out[srow:erow + 1] = [out[srow][:scol] + new + out[erow][ecol:]]
    return "".join(out), edits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default="scripts")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(REPO, a.root, "**", "*.py"), recursive=True))
    n_files = n_edits = 0
    failed = []
    for f in files:
        if os.path.abspath(f) == SELF:
            continue
        new_src, edits = rewrite(f)
        if edits and edits[0][0] == "TOKENIZE-FAIL":
            failed.append((f, edits[0][1]))
            continue
        if not new_src:
            continue
        rel = os.path.relpath(f, REPO)
        n_files += 1
        n_edits += len(edits)
        print(f"\n{rel}  ({len(edits)} edit{'s' if len(edits) != 1 else ''})")
        for (srow, _), _, old, new in edits:
            o = old if len(old) < 78 else old[:75] + "..."
            n = new if len(new) < 78 else new[:75] + "..."
            print(f"   L{srow}: {o}\n        -> {n}")
        if not a.dry_run:
            # compile() before writing: a broken escape or a mangled multi-line token must not
            # reach disk. This is the whole reason the raw-prefix rule above exists.
            try:
                compile(new_src, f, "exec")
            except SyntaxError as e:
                failed.append((rel, f"would not compile: {e}"))
                print(f"   [SKIPPED] {e}")
                continue
            open(f, "w", encoding="utf-8").write(new_src)

    print(f"\n{'DRY RUN: ' if a.dry_run else ''}{n_edits} edits across {n_files} files")
    if failed:
        print("\nFAILED:")
        for f, why in failed:
            print(f"   {f}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
