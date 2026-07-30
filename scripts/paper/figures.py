#!/usr/bin/env python3
"""Curate paper/figures/ — the one place a figure and its provenance live together.

WHY THIS EXISTS. outputs/ and plots/ hold 9550 figure files across 32 directories, 1907 of
them destroyed, in three different analysis conventions, with three separate cases of one
filename written by three different scripts. Nothing in that tree tells a reader which files
are current, which are superseded, or where a number came from. docs/PAPER_FIGURE_MAP.md is
the survey of that mess; this is the clean room it feeds.

THE LAYOUT — one self-contained directory per figure, fixed filenames inside:

    paper/figures/<slug>/
        figure.pdf          vector, for LaTeX
        figure.png          raster preview
        values.csv          the plotted numbers
        provenance.json     how they were made
        meta.json           where it was published FROM, when, sha256, gate result
        README.md           one paragraph: what it shows, and the caveats, in prose

A figure is one directory, so it cannot be separated from its provenance by a copy, a move or
a tarball, and "is this figure complete?" is answerable by listing one directory. Contrast the
flat-with-suffixes layout, where the same information is 6 files interleaved with 6 more from
the next figure.

COPIES, NEVER SYMLINKS. Deliberate. $SCRATCH purges after 30 idle days, symlinks do not
survive archiving (this project has already been bitten by a self-referential link), and a
symlink would let a regeneration in outputs/ silently change a figure the paper cites. The
cost is duplication; `verify` exists to detect drift instead.

THE GATE is the point. The provenance rule has existed for a while and 7 of 63 figures follow
it, because a rule enforced by discipline is not enforced. `publish` REFUSES a figure whose
sidecars are missing, unparseable or empty; nothing incomplete can get in. Warnings are
admitted but recorded in meta.json and surfaced in MANIFEST.md, so a known-imperfect figure is
visible rather than forgotten.

Note the empty-values check specifically: 38 CSVs in this repo are the correct size on disk and
100% NUL, and pandas.read_csv returns shape (0,1) on them WITHOUT raising. An existence check
passes and the figure panel comes out blank. Presence is not enough; row count is checked.

USAGE
    figures.py publish <source-stem> --slug <slug> [--title T] [--position N] [--force]
        source-stem is the path WITHOUT extension, e.g.
        outputs/plots/ps_submean_l37/nsigma_vs_lmax
    figures.py manifest          rebuild MANIFEST.md + manifest.json from the directories
    figures.py verify            re-gate everything; report drift vs source and missing files
    figures.py list              short status table

Runs under either interpreter — stdlib only, no pandas (jaxili has no pandas).
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(REPO, "paper", "figures")

# Fields a provenance sidecar should carry. Missing ones are warnings, not refusals — the
# figure is still traceable without them, just less well.
RECOMMENDED = ("figure", "generated_utc", "git_commit", "versions", "caveats")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_ok(path):
    """Structural check. %%EOF is searched in the trailer, not tested as the last 6 bytes —
    a real trailer ends 'startxref\\nNNNN\\n%%EOF\\n', so a last-6-bytes test false-negatives."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        fh.seek(max(0, os.path.getsize(path) - 2048))
        tail = fh.read()
    return head.startswith(b"%PDF") and b"%%EOF" in tail


def png_ok(path):
    with open(path, "rb") as fh:
        head = fh.read(8)
        fh.seek(max(0, os.path.getsize(path) - 32))
        tail = fh.read()
    return head == b"\x89PNG\r\n\x1a\n" and b"IEND" in tail


def csv_rows(path):
    """Data-row count. Guards the silent-empty case described in the module docstring."""
    with open(path, newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def scale_summary(prov):
    """Pull the 'which scales went in' statement out of a provenance dict.

    Generators express it differently — plot_nsigma_vs_lmax uses scales_included,
    plot_contours_three_stats uses conventions plus cuts, the Fisher script uses ps_edges
    plus hos_scales_* — so this normalises them rather than forcing every generator to be
    rewritten. Returns a list of (label, text) pairs, empty if the figure says nothing.
    """
    out = []
    for key in ("scales_included", "conventions"):
        v = prov.get(key)
        if isinstance(v, dict):
            out += [(k, str(x)) for k, x in v.items()]
        elif isinstance(v, str):
            out.append((key, v))
    cuts = prov.get("cuts")
    if isinstance(cuts, dict):
        out += [(k, str(x)) for k, x in cuts.items()]
    if prov.get("cut") and not out:
        out.append(("cut", str(prov["cut"])))
    # Fisher-style declarations
    if prov.get("ps_edges"):
        out.append(("ps_bandpower_edges", str(prov["ps_edges"])))
    for k in ("hos_scales_full", "hos_scales_baryon_safe", "regime"):
        if prov.get(k) is not None:
            out.append((k, str(prov[k])))
    if not out and prov.get("lmin") is not None:
        out.append(("lmin", str(prov["lmin"])))
    return out


def gate(stem):
    """Validate a source figure. Returns (fails, warns, info).

    fails => must not be published. warns => publish, but record it.
    """
    fails, warns, info = [], [], {}
    pdf, png = stem + ".pdf", stem + ".png"
    vals, prov = stem + "_values.csv", stem + "_provenance.json"

    if not os.path.exists(pdf):
        fails.append("no .pdf (the paper needs vector)")
    elif not pdf_ok(pdf):
        fails.append(".pdf is structurally invalid (destroyed or truncated)")

    if not os.path.exists(png):
        warns.append("no .png preview")
    elif not png_ok(png):
        warns.append(".png is structurally invalid")

    if not os.path.exists(vals):
        fails.append("no _values.csv — the plotted numbers are not recorded")
    else:
        n = csv_rows(vals)
        info["value_rows"] = n
        if n == 0:
            fails.append("_values.csv has zero data rows (100%-NUL table reads as empty, "
                         "it does not raise)")
        with open(vals, newline="") as fh:
            cols = next(csv.reader(fh), [])
        info["value_columns"] = cols
        if not any("n_seed" in c or c == "n_runs_pooled" or "n_runs" in c for c in cols):
            warns.append("_values.csv has no seed-count column; n_seeds is the column that "
                         "reveals a point averaging a different subset than it used to")

    if not os.path.exists(prov):
        fails.append("no _provenance.json")
    else:
        try:
            j = json.load(open(prov))
            info["provenance"] = j
            for f in RECOMMENDED:
                if f not in j:
                    warns.append(f"provenance is missing '{f}'")
            if j.get("git_commit") in (None, "", "unknown"):
                warns.append("provenance git_commit is unknown — the figure cannot be traced "
                             "to the code that made it")
            if "mplstyle" not in j:
                warns.append("provenance is missing 'mplstyle'")
            # A figure that does not state WHICH SCALES went into it cannot be interpreted:
            # the same statistic at lmax=460 and lmax=1020, or at scales1234 and scales234,
            # are different measurements. Standing rule — every figure declares its scales.
            if not scale_summary(j):
                warns.append("provenance does not state the SCALES included (no "
                             "'scales_included', 'conventions', 'cuts', 'ps_edges' or 'lmin') "
                             "— a figure that does not say which multipoles and wavelet "
                             "scales went in cannot be interpreted")
            info["n_caveats"] = len(j.get("caveats", []))
        except Exception as e:
            fails.append(f"_provenance.json does not parse ({type(e).__name__})")

    return fails, warns, info


def cmd_publish(args):
    stem = args.source_stem
    if stem.endswith((".pdf", ".png")):
        stem = os.path.splitext(stem)[0]
    stem = stem[:-1] if stem.endswith("_") else stem
    abs_stem = stem if os.path.isabs(stem) else os.path.join(REPO, stem)

    fails, warns, info = gate(abs_stem)
    print(f"gating {stem}")
    for f in fails:
        print(f"  FAIL  {f}")
    for w in warns:
        print(f"  warn  {w}")
    if fails:
        print("\nREFUSED — fix the source, then publish. Nothing was written.")
        print("This is the gate doing its job: an incomplete figure in paper/figures/ would "
              "be worse than an absent one, because it looks finished.")
        return 1

    dest = os.path.join(PAPER, args.slug)
    if os.path.exists(dest) and not args.force:
        print(f"\n{dest} already exists — pass --force to replace it.")
        return 1
    os.makedirs(dest, exist_ok=True)

    copied = {}
    for src_suffix, dst_name in ((".pdf", "figure.pdf"), (".png", "figure.png"),
                                 ("_values.csv", "values.csv"),
                                 ("_provenance.json", "provenance.json")):
        src = abs_stem + src_suffix
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(dest, dst_name))
        copied[dst_name] = {"from": os.path.relpath(src, REPO), "sha256": sha256(src),
                            "bytes": os.path.getsize(src)}
    # Extra sidecars some generators emit (e.g. _crossings.csv, _covariance.csv) — carry them
    # through rather than dropping information on the floor.
    for extra in ("_crossings.csv", "_covariance.csv"):
        src = abs_stem + extra
        if os.path.exists(src):
            name = extra.lstrip("_")
            shutil.copy2(src, os.path.join(dest, name))
            copied[name] = {"from": os.path.relpath(src, REPO), "sha256": sha256(src),
                            "bytes": os.path.getsize(src)}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"

    meta = {
        "slug": args.slug,
        "title": args.title or args.slug.replace("_", " "),
        "paper_position": args.position,
        "published_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "published_at_repo_commit": commit,
        "source_stem": os.path.relpath(abs_stem, REPO),
        "files": copied,
        "gate": {"passed": True, "warnings": warns},
        "value_rows": info.get("value_rows"),
        "n_caveats": info.get("n_caveats"),
    }
    json.dump(meta, open(os.path.join(dest, "meta.json"), "w"), indent=2)

    prov = info.get("provenance", {})
    with open(os.path.join(dest, "README.md"), "w") as fh:
        fh.write(f"# {meta['title']}\n\n")
        if args.note:
            fh.write(args.note.strip() + "\n\n")
        fh.write(f"- **source**: `{meta['source_stem']}`\n")
        fh.write(f"- **generator commit**: `{prov.get('git_commit','unknown')}`\n")
        fh.write(f"- **generated**: {prov.get('generated_utc','?')}\n")
        fh.write(f"- **published**: {meta['published_utc']} at repo `{commit}`\n")
        fh.write(f"- **rows in values.csv**: {info.get('value_rows','?')}\n")
        # Scales first, before caveats: it is the thing a reader needs in order to know what
        # the figure even measures.
        _sc = scale_summary(prov)
        if _sc:
            fh.write("\n## Scales included\n\n")
            for k, v in _sc:
                fh.write(f"- **{k}**: {v}\n")
        if prov.get("presentation_todo"):
            fh.write(f"\n## Presentation TODO before use\n\n{prov['presentation_todo']}\n")
        if warns:
            fh.write("\n## Known gaps\n\n" + "".join(f"- {w}\n" for w in warns))
        cav = prov.get("caveats") or []
        if cav:
            fh.write("\n## Caveats (from provenance)\n\n" + "".join(f"- {c}\n" for c in cav))
        fh.write("\n*Do not edit these files. Regenerate at the source and re-publish, so the "
                 "figure and its numbers can never disagree.*\n")

    print(f"\npublished -> paper/figures/{args.slug}/  ({len(copied)} files"
          + (f", {len(warns)} warning(s) recorded" if warns else "") + ")")
    return 0


def load_all():
    out = []
    if not os.path.isdir(PAPER):
        return out
    for slug in sorted(os.listdir(PAPER)):
        mp = os.path.join(PAPER, slug, "meta.json")
        if os.path.isfile(mp):
            try:
                out.append((slug, json.load(open(mp))))
            except Exception:
                out.append((slug, None))
    return out


def cmd_manifest(args):
    entries = load_all()
    entries.sort(key=lambda t: (t[1] or {}).get("paper_position") or 999)
    rows = []
    for slug, m in entries:
        if m is None:
            rows.append((slug, "?", "meta.json unparseable", "", "", ""))
            continue
        prov = {}
        pp = os.path.join(PAPER, slug, "provenance.json")
        if os.path.isfile(pp):
            try:
                prov = json.load(open(pp))
            except Exception:
                pass
        rows.append((slug, m.get("paper_position") or "-", m.get("title", ""),
                     prov.get("git_commit", "?"), m.get("value_rows", "?"),
                     len(m.get("gate", {}).get("warnings", []))))

    md = ["# Paper figures — manifest",
          "",
          f"Auto-generated by `scripts/paper/figures.py manifest`. **{len(rows)} figures.**",
          "",
          "Every directory here passed the provenance gate: a valid vector PDF, a non-empty",
          "`values.csv`, and a parseable `provenance.json`. Rebuild with `manifest`; re-check",
          "integrity and source drift with `verify`.",
          "",
          "| # | figure | what | generator commit | value rows | warns |",
          "|---|---|---|---|---|---|"]
    for slug, pos, title, commit, nrows, nwarn in rows:
        w = "-" if nwarn == 0 else f"**{nwarn}**"
        md.append(f"| {pos} | [`{slug}`]({slug}/) | {title} | `{commit}` | {nrows} | {w} |")
    md += ["",
           "Each directory holds `figure.pdf`, `figure.png`, `values.csv`, `provenance.json`,",
           "`meta.json` and a `README.md`. `meta.json` records the source path and a sha256 of",
           "every file at publish time, which is what makes drift detectable.",
           "",
           "**Do not edit files in place.** Regenerate at the source and re-publish, so a figure",
           "and its recorded numbers can never disagree.",
           ""]
    open(os.path.join(REPO, "paper", "figures", "MANIFEST.md"), "w").write("\n".join(md))
    json.dump([{"slug": s, "meta": m} for s, m in entries],
              open(os.path.join(REPO, "paper", "figures", "manifest.json"), "w"), indent=2)
    print(f"wrote paper/figures/MANIFEST.md and manifest.json ({len(rows)} figures)")
    return 0


def cmd_verify(args):
    entries = load_all()
    bad = 0
    print(f"verifying {len(entries)} published figures\n")
    for slug, m in entries:
        d = os.path.join(PAPER, slug)
        problems = []
        if m is None:
            print(f"  FAIL {slug}: meta.json unparseable")
            bad += 1
            continue
        for name, rec in m.get("files", {}).items():
            p = os.path.join(d, name)
            if not os.path.exists(p):
                problems.append(f"{name} missing from the published directory")
            elif sha256(p) != rec["sha256"]:
                problems.append(f"{name} was EDITED IN PLACE (sha256 differs from publish time)")
            src = os.path.join(REPO, rec["from"])
            if not os.path.exists(src):
                problems.append(f"source gone: {rec['from']} (published copy is now the only one)")
            elif sha256(src) != rec["sha256"]:
                problems.append(f"source drifted: {rec['from']} changed since publish "
                                f"-> re-publish or the paper cites a stale figure")
        # re-gate the published copy itself
        pdf = os.path.join(d, "figure.pdf")
        if os.path.exists(pdf) and not pdf_ok(pdf):
            problems.append("published figure.pdf is structurally invalid")
        v = os.path.join(d, "values.csv")
        if os.path.exists(v) and csv_rows(v) == 0:
            problems.append("published values.csv has zero data rows")
        if problems:
            bad += 1
            print(f"  FAIL {slug}")
            for p in problems:
                print(f"       - {p}")
        else:
            nw = len(m.get("gate", {}).get("warnings", []))
            print(f"  ok   {slug}" + (f"   ({nw} recorded warning(s))" if nw else ""))
    print(f"\n{len(entries) - bad} ok, {bad} with problems")
    return 1 if bad else 0


def cmd_list(args):
    for slug, m in load_all():
        pos = (m or {}).get("paper_position") or "-"
        nw = len((m or {}).get("gate", {}).get("warnings", []))
        print(f"  {str(pos):>3}  {slug:52s} rows={(m or {}).get('value_rows','?'):>5} warns={nw}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("publish", help="copy a figure + sidecars into paper/figures/<slug>/")
    p.add_argument("source_stem", help="path without extension")
    p.add_argument("--slug", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--position", type=int, default=None, help="intended figure number in the paper")
    p.add_argument("--note", default=None, help="one-paragraph description for the README")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_publish)

    for name, fn, helptext in (("manifest", cmd_manifest, "rebuild MANIFEST.md + manifest.json"),
                               ("verify", cmd_verify, "re-gate everything, detect drift"),
                               ("list", cmd_list, "short status table")):
        q = sub.add_parser(name, help=helptext)
        q.set_defaults(func=fn)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
