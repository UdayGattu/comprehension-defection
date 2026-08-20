#!/usr/bin/env python3
"""Trace every interval printed in the paper to the artefact the paper NAMES.

WHY THIS EXISTS

  Point estimates in this paper are solid. Interval endpoints are not uniformly
  traceable:

    - four intervals in section 5.5 matched no field of exp8_logodds.json, which
      carries three CI forms per cell (main, episode-count HA, drop-zero);
    - two more had no generating code at all - nothing bootstraps log-odds A on
      exp6;
    - 181 of 208 endpoints shared between artefacts differ, because
      analysis/02,15,16 use Python's Mersenne Twister and analysis/13,14 use
      numpy PCG64 (see DATA.md);
    - an artefact may hold BOTH ci_bootstrap and ci_analytic for one contrast,
      and which the paper quotes is nowhere stated.

WHY IT IS STRICT

  An earlier version searched every artefact for any two numbers that rounded to
  the printed pair. Across ~120 intervals and tens of thousands of stored values
  at four decimals, that finds coincidences: a schema term in exp2 matched an
  `opponent_spread` field from a different experiment, a different model and a
  different readout. A checker that reports coincidences manufactures confidence,
  which is worse than no checker. Two constraints remove them.

  1. DECLARED SOURCE. Appendix "Interval provenance" names one artefact per
     table and per section. An interval is resolved ONLY against that artefact.
     Reproducing somewhere else is not provenance. An interval whose location
     has no declared source is UNDECLARED, which is a finding about the paper,
     not about the number.

  2. STRUCTURAL PAIRING. The two endpoints must form an interval in the
     artefact's own structure - the two elements of one CI list, or two sibling
     keys named as a low/high pair under one parent. Two unrelated scalars that
     happen to round correctly are not a match, and values drawn from different
     elements of a value list are never a match.

  Multiple distinct declared fields matching one printed pair is AMBIGUOUS, not
  a pass: it means the paper does not say which of them it quotes.

THE ROUNDING RULE

  Rates sit on a coarse rational grid, so roughly half land exactly on the 4th
  decimal boundary. Never round the stored float: ci_bootstrap[1] for exp4 llama
  sem logit is -0.003949999999999995, which naive round() sends to -0.0039 and
  exact arithmetic to -0.0040. The paper prints -0.0040 and is correct. Values
  are reconstructed onto each plausible grid before rounding, ties away from
  zero; a match reachable only through a tie is reported MATCH_TIE.

    python analysis/17_interval_provenance.py --tex paper/src --out INTERVALS.md
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as F
from pathlib import Path

GRIDS = (20000, 19000, 40000, 38000, 32000, 30400, 10000, 1000)

PRODUCER = {
    "episode_level.json": "analysis/02_episode_level.py",
    "EXP6_FIELDS.json": "analysis/13_exp6_fields.py",
    "EXP7_FIELDS.json": "analysis/13_exp6_fields.py",
    "REVIEWER_RESPONSES.json": "analysis/14_reviewer_responses.py",
    "REVIEWER_RESPONSES_ALL.json": "analysis/14_reviewer_responses.py",
    "exp8_stability.json": "analysis/15_exp8_stability.py",
    "exp8_logodds.json": "analysis/16_exp8_logodds.py",
}

# Declared source per table label, then per section label. From the paper's own
# "Interval provenance" paragraph. `basis` constrains the field path where the
# artefact stores more than one turn-0 convention.
BY_TABLE = {
    "tab:decomp":        (r"^ep_exp4_.*\.json$",  None),
    "tab:fields":        (r"^EXP6_FIELDS\.json$", "excl_t0"),
    "tab:zerodose":      (r"^EXP6_FIELDS\.json$", "excl_t0"),
    "tab:exp8":          (r"^exp8_logodds\.json$", None),
    "tab:excluded":      (r"^exp8_logodds\.json$", None),
    "tab:contentschema": (r"^DECOMPOSITION\.json$",     None),
    "tab:stratdonor":    (r"^STRATIFIED_DONOR\.json$",  None),
}
# Mirrors the "Interval provenance" appendix. If the two disagree the CHECKER is
# wrong, not the paper: this map is transcribed from the appendix, not chosen.
ABSTRACT_RULE = (r"^ep_exp4_qwen_sem_scratchpad\.json$", None)
BY_SECTION = {
    "sec:intro":       ABSTRACT_RULE,
    "sec:theory":      ABSTRACT_RULE,
    "sec:setup":       (r"^(episode_level|ep_exp\d.*|REVIEWER_RESPONSES_ALL)\.json$", None),
    "sec:requirements":(r"^(episode_level|ep_exp\d.*|REVIEWER_RESPONSES_ALL)\.json$", None),
    "sec:audit":       (r"^(EXP6_FIELDS|EXP7_FIELDS)\.json$", "excl_t0"),
    "sec:crossconfig": (r"^(ep_exp[34]_.*|EXP[67]_FIELDS)\.json$", None),
    "sec:config":      (r"^exp8_(logodds|stability)\.json$", None),
    "sec:limits":      (r"^(episode_level|ep_exp\d.*|EXP6_FIELDS)\.json$", None),
}

CI_MACRO = re.compile(r"\\CI\{\s*([-+][0-9.]+)\s*\}\{\s*([-+][0-9.]+)\s*\}")
CI_BARE = re.compile(r"\$\[\s*([-+][0-9.]+)\s*,\s*([-+][0-9.]+)\s*\]\$")
LO_HI = [("lo", "hi"), ("ci_lo", "ci_hi"), ("A_ci_lo", "A_ci_hi"),
         ("A_lo_ci_lo", "A_lo_ci_hi"), ("A_lo_ep_ci_lo", "A_lo_ep_ci_hi"),
         ("A_lo_drop_ci_lo", "A_lo_drop_ci_hi"), ("low", "high")]
CI_LIST_KEY = re.compile(r"(ci|interval|bounds)", re.I)


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def round_exact(x, k, grid):
    fr = F(round(x * grid), grid) if grid else F(Decimal(repr(x)))
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    return d.quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP), \
        (abs(d) * 10 ** k * 2) % 2 == 1


def matches(x, printed):
    k, target = decimals(printed), Decimal(printed)
    for grid in (None, *GRIDS):
        try:
            q, tie = round_exact(x, k, grid)
        except (ValueError, OverflowError):
            continue
        if q == target:
            return True, tie
    return False, False


def interval_pairs(o, path=""):
    """Yield (field_path, lo, hi) for STRUCTURAL interval pairs only."""
    if isinstance(o, dict):
        for a, b in LO_HI:
            if isinstance(o.get(a), (int, float)) and isinstance(o.get(b), (int, float)):
                yield path or "/", float(o[a]), float(o[b])
        for k, v in o.items():
            if (isinstance(v, list) and len(v) == 2
                    and all(isinstance(t, (int, float)) for t in v)
                    and CI_LIST_KEY.search(k)):
                yield f"{path}/{k}", float(v[0]), float(v[1])
            else:
                yield from interval_pairs(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, (dict, list)):
                yield from interval_pairs(v, f"{path}[{i}]")


def load(root):
    out = {}
    for p in sorted(root.glob("*.json")) + sorted((root / "paper").glob("*.json")):
        try:
            out[p.name] = list(interval_pairs(json.loads(p.read_text())))
        except Exception:
            continue
    return out


def scan(tex_dir):
    """Yield (file, line, lo, hi, declared_artefact_re, basis, context)."""
    for f in ("abstract.tex", "body.tex", "appendices.tex"):
        p = tex_dir / f
        if not p.exists():
            continue
        sec, tbl, depth = None, None, 0
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if "\\begin{table}" in line:
                depth, tbl = depth + 1, None
            for m in re.finditer(r"\\label\{([^}]+)\}", line):
                if depth > 0:
                    tbl = m.group(1)
                elif m.group(1).startswith("sec:") or m.group(1).startswith("app:"):
                    sec = m.group(1)
            rule = (BY_TABLE.get(tbl) if depth > 0 and tbl
                    else (BY_SECTION.get(sec) if sec
                          else (ABSTRACT_RULE if str(f).endswith("abstract.tex") else None)))
            ctx = tbl if (depth > 0 and tbl) else (sec or "-")
            for rx in (CI_MACRO, CI_BARE):
                for m in rx.finditer(line):
                    yield f, n, m.group(1), m.group(2), rule, ctx
            if "\\end{table}" in line:
                depth = max(0, depth - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/src")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="INTERVALS.md")
    args = ap.parse_args()
    arte, rows = load(Path(args.root)), []

    for f, n, lo_s, hi_s, rule, ctx in scan(Path(args.tex)):
        printed = f"[{lo_s},{hi_s}]"
        if rule is None:
            rows.append((f, n, ctx, printed, "-", "-", "-", "UNDECLARED"))
            continue
        pat, basis = rule
        if pat is None:
            rows.append((f, n, ctx, printed, "-", "-", "-", "NO_ARTEFACT_YET"))
            continue
        hits, tie_any = [], False
        for name, pairs in arte.items():
            if not re.match(pat, name):
                continue
            for fld, a, b in pairs:
                if basis and basis not in fld:
                    continue
                ok_lo, t1 = matches(a, lo_s)
                ok_hi, t2 = matches(b, hi_s)
                if ok_lo and ok_hi:
                    hits.append((name, fld))
                    tie_any = tie_any or t1 or t2
        uniq = sorted(set(hits))
        if not uniq:
            rows.append((f, n, ctx, printed, "-", "-", "-", "UNTRACEABLE"))
        elif len(uniq) > 1:
            rows.append((f, n, ctx, printed, uniq[0][0],
                         PRODUCER.get(uniq[0][0], "analysis/02_episode_level.py --db <db>"),
                         f"{len(uniq)} fields match", "AMBIGUOUS"))
        else:
            name, fld = uniq[0]
            rows.append((f, n, ctx, printed, name,
                         PRODUCER.get(name, "analysis/02_episode_level.py --db <db>"),
                         fld, "MATCH_TIE" if tie_any else "MATCH"))

    tally = {}
    for r in rows:
        tally[r[7]] = tally.get(r[7], 0) + 1
    L = ["# Interval provenance\n",
         f"{len(rows)} intervals parsed from `abstract.tex`, `body.tex`, "
         "`appendices.tex`.\n",
         "Each is resolved **only** against the artefact the paper names for its "
         "table or section, and only where the two endpoints form a structural "
         "interval pair in that artefact. Exact rational arithmetic, "
         "`ROUND_HALF_UP`.\n",
         "| verdict | meaning |", "|---|---|",
         "| `MATCH` | reproduced from the declared artefact and field |",
         "| `MATCH_TIE` | reproduced, but an endpoint sits on a rounding tie where naive `round()` disagrees |",
         "| `AMBIGUOUS` | more than one field of the declared artefact reproduces it; the paper does not say which |",
         "| `UNTRACEABLE` | the declared artefact does not reproduce it |",
         "| `UNDECLARED` | the paper names no artefact for this location |",
         "| `NO_ARTEFACT_YET` | source script emits Markdown only; no machine-readable release |",
         "", "| file | line | context | printed | artefact | script | field | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| `{}` | {} | `{}` | `{}` | `{}` | `{}` | `{}` | **{}** |".format(*r))
    L.append("\n" + " · ".join(f"**{k}: {v}**" for k, v in sorted(tally.items())) + "\n")
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return 1 if tally.get("UNTRACEABLE") or tally.get("UNDECLARED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
