#!/usr/bin/env python3
"""Join EVIDENCE_cells.csv across databases into the tables the paper needs.

WHY THIS IS SEPARATE FROM 06
    06_evidence.py reports each database on its own terms, because that is what
    a provenance record should do. But every headline comparison in this study
    spans FILES:

      readout ladder    exp4_*_logit  vs  exp4_*_scratchpad  vs  exp5_*_minimal
      stack drift       exp3_llama_sem            vs  exp4_llama_sem_logit
      lexical test      exp3_*_sem  vs  exp3_*_swap  vs  exp3_*_abs

    Joining those inside 06 would have meant hard-coding which file plays which
    role, which is exactly the kind of assumption this pair of scripts exists to
    avoid. Here the roles are PARSED FROM THE FILENAMES and printed, so a
    mis-parse is visible rather than silent.

WHAT IT WILL NOT DO
    No bootstrap. Point estimates only, from the same CSV 06 produced, which
    came from SQL. Intervals live in ep_*.json and are cited by name beside each
    table so a reader can go and check.

    No interpretation. Where a comparison is confounded - the readout ladder is,
    by the exp4 instruction - the table says so in place rather than in a note
    somewhere else.

VOID CELLS ARE PROPAGATED, NOT DROPPED
    A contrast is only as good as its worse end. If either cell exceeds the
    off-task gate the row is marked VOID and the number is still printed, so an
    excluded result stays visible instead of vanishing.

    python analysis/07_cross_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# GRID-CORRECT FOUR-DECIMAL RENDERING
#
# Episode-level rates and their differences are grid-valued: with E episodes of
# T scored turns a rate is an integer multiple of 1/(E*T). Rendering with
# f"{v:.4f}" applies Python's round-half-even to a stored float, which sends a
# grid tie the wrong way half the time -- 0.021549999999999958 is exactly
# 431/20000 = 0.021550 and must print +0.0216, but f"{v:+.4f}" gives +0.0215.
#
# Reconstruct onto the grid first, then ROUND_HALF_UP (ties away from zero),
# which is the rule the manuscript declares and uses throughout.
# ---------------------------------------------------------------------------
from decimal import Decimal as _Dec, ROUND_HALF_UP as _RHU
from fractions import Fraction as _Frac

_GRIDS = (20000, 19000, 32000, 30400, 40000, 38000, 64000, 60800, 12800)


def grid4(v, signed=False):
    """Four decimals, reconstructed on the episode grid, ties away from zero."""
    if v is None:
        return "-"
    for g in _GRIDS:
        n = round(v * g)
        if abs(v * g - n) < 1e-6:
            fr = _Frac(n, g)
            d = (_Dec(fr.numerator) / _Dec(fr.denominator)).quantize(
                _Dec("0.0001"), rounding=_RHU)
            break
    else:
        d = _Dec(repr(v)).quantize(_Dec("0.0001"), rounding=_RHU)
    s = f"{d:f}"
    if signed and not s.startswith("-"):
        s = "+" + s
    return s


OFF_TASK_GATE = 0.10

RULE = "=" * 78


# ------------------------------------------------------------------ parsing

def parse(db: str) -> dict | None:
    """Recover (experiment, model, framing, readout) from a database name.

    Printed in full by --show-parse. Every downstream table is built from this,
    so if it is wrong the tables are wrong, and the only defence is looking at
    it.
    """
    m = re.match(r"^(exp\d)_([a-z]+)(?:_([a-z]+))?(?:_([a-z]+))?$", db)
    if db == "sweep":
        return {"exp": "exp1", "model": "llama", "framing": "sem",
                "readout": "logit", "prompt": "n/a"}
    if not m:
        return None
    exp, model, a, b = m.groups()

    framing, readout, prompt = "sem", "logit", "n/a"
    for tok in (a, b):
        if tok in ("sem", "abs", "swap"):
            framing = tok
        elif tok in ("logit", "scratchpad"):
            readout = tok
        elif tok == "minimal":
            readout, prompt = "scratchpad", "minimal"
        elif tok == "labelswap":
            framing = "swap"
    if exp == "exp4" and readout == "scratchpad":
        prompt = "guided"
    if exp == "exp5":
        prompt = "minimal"
    return {"exp": exp, "model": model, "framing": framing,
            "readout": readout, "prompt": prompt}


def load(path: str) -> tuple[dict, dict]:
    """cells[(db, arm, opp)] -> row ; meta[db] -> parsed roles"""
    cells, meta = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            db = r["db"]
            if db not in meta:
                p = parse(db)
                if p is None:
                    continue
                meta[db] = p
            for k, v in list(r.items()):
                if k in ("db", "arm", "opp"):
                    continue
                try:
                    r[k] = float(v) if v not in ("", "-", "None") else None
                except (TypeError, ValueError):
                    r[k] = None
            cells[(db, r["arm"], r["opp"])] = r
    return cells, meta


# ---------------------------------------------------------------- contrasts

def contrast(cells, db, a, b, opp, metric="defect_ep"):
    """Difference between two arms, with the worse end's off-task carried.

    Returns (value, off_task_max, void). Returning the value even when void is
    deliberate: an excluded number that is printed can be checked; one that is
    dropped cannot.
    """
    ra, rb = cells.get((db, a, opp)), cells.get((db, b, opp))
    if not ra or not rb:
        return None, None, False
    va, vb = ra.get(metric), rb.get(metric)
    if va is None or vb is None:
        return None, None, False
    off = max(ra.get("off_task") or 0.0, rb.get("off_task") or 0.0)
    return va - vb, off, off > OFF_TASK_GATE


def mark(v, off, void, nd=4) -> str:
    if v is None:
        return "-"
    s = f"{v:+.{nd}f}"
    return f"{s} VOID(off {off:.2f})" if void else s


def find(meta, **want) -> list[str]:
    return sorted(db for db, m in meta.items()
                  if all(m.get(k) == v for k, v in want.items()))


def ci_path(db: str) -> str:
    p = f"ep_{db}.json"
    return p if os.path.exists(p) else ""


# ------------------------------------------------------------------- tables

def table(header, rows) -> str:
    if not rows:
        return "  (no rows)\n"
    w = [max(len(str(header[i])), max(len(str(r[i])) for r in rows))
         for i in range(len(header))]
    out = ["  " + "  ".join(str(header[i]).ljust(w[i]) for i in range(len(w)))]
    out.append("  " + "  ".join("-" * x for x in w))
    for r in rows:
        out.append("  " + "  ".join(str(r[i]).ljust(w[i]) for i in range(len(w))))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="EVIDENCE_cells.csv")
    ap.add_argument("--out", default="CROSS_EXPERIMENT.md")
    ap.add_argument("--show-parse", action="store_true")
    args = ap.parse_args()

    cells, meta = load(args.csv)
    models = sorted({m["model"] for m in meta.values()})
    opps = sorted({k[2] for k in cells})
    L: list[str] = []

    def w(s=""):
        L.append(s)

    w("# Cross-experiment tables\n")
    w("Built by `analysis/07_cross_experiment.py` from `EVIDENCE_cells.csv`, "
      "which `06_evidence.py` produced from SQL. No number here was recalled "
      "or recomputed — each is a subtraction between two cells in that file.\n")
    w(f"Defect rates are **episode-level means** (`defect_ep`). A contrast is "
      f"marked `VOID` when either end exceeds the off-task gate of "
      f"{OFF_TASK_GATE}; the value is still printed so an excluded result stays "
      f"visible.\n")
    w("Bootstrap intervals are not recomputed here. The `ep_*.json` file "
      "carrying each row's interval is named beside every table.\n")

    # -- 0 -----------------------------------------------------------------
    w("\n## 0. How each database was classified\n")
    w("Every table below is built from this parse. If a row is wrong, the "
      "tables built from it are wrong — which is why it is printed first.\n")
    w("```")
    w(table(["database", "exp", "model", "framing", "readout", "CoT prompt",
             "intervals"],
            [[db, m["exp"], m["model"], m["framing"], m["readout"],
              m["prompt"], ci_path(db) or "none"]
             for db, m in sorted(meta.items())]))
    w("```")

    # -- 1 -----------------------------------------------------------------
    w("\n## 1. Readout ladder — semantic framing\n")
    w("`P(D)` per arm across three ways of eliciting the same decision from the "
      "same weights on the same game.\n")
    w("**The LOGIT vs CoT comparison is confounded within exp4**: its "
      "scratchpad instruction names the finite horizon. The `CoT minimal` "
      "column is the control — its instruction is *\"Before choosing, think "
      "step by step.\"* and names nothing. Read the two CoT columns together or "
      "not at all.\n")
    for model in models:
        lg = find(meta, model=model, framing="sem", readout="logit", exp="exp4")
        gd = find(meta, model=model, framing="sem", readout="scratchpad",
                  exp="exp4")
        mn = find(meta, model=model, framing="sem", exp="exp5")
        if not (lg or gd or mn):
            continue
        w(f"\n### {model}\n")
        rows = []
        for arm in ("1", "3b", "3"):
            for opp in opps:
                def val(dbs):
                    if not dbs:
                        return "-"
                    r = cells.get((dbs[0], arm, opp))
                    if not r or r.get("defect_ep") is None:
                        return "-"
                    v = grid4(r['defect_ep'])
                    return (f"{v} VOID" if (r.get("off_task") or 0) > OFF_TASK_GATE
                            else v)
                rows.append([f"arm {arm}", opp, val(lg), val(gd), val(mn)])
        w("```")
        w(table(["arm", "opp", "LOGIT (exp4)", "CoT guided (exp4)",
                 "CoT minimal (exp5)"], rows))
        w("```")
        srcs = [ci_path(d) for d in (lg + gd + mn) if ci_path(d)]
        if srcs:
            w("intervals: " + ", ".join(f"`{s}`" for s in srcs) + "\n")

    # -- 2 -----------------------------------------------------------------
    w("\n## 2. Contrasts across the readout ladder\n")
    w("`perturbation = P(D|3b) - P(D|1)` — does inserting any block matter?\n")
    w("`ATE_true = P(D|3) - P(D|3b)` — holding the block constant, does its "
      "content matter?\n")
    for label, defs in (("perturbation", ("3b", "1")),
                        ("ATE_true", ("3", "3b")),
                        ("ATE_naive", ("3", "1"))):
        w(f"\n### {label} = P(D|{defs[0]}) − P(D|{defs[1]})\n")
        rows = []
        for model in models:
            for opp in opps:
                cs = []
                for dbs in (find(meta, model=model, framing="sem",
                                 readout="logit", exp="exp4"),
                            find(meta, model=model, framing="sem",
                                 readout="scratchpad", exp="exp4"),
                            find(meta, model=model, framing="sem", exp="exp5")):
                    if not dbs:
                        cs.append("-")
                        continue
                    v, off, void = contrast(cells, dbs[0], defs[0], defs[1], opp)
                    cs.append(mark(v, off, void))
                rows.append([model, opp] + cs)
        w("```")
        w(table(["model", "opp", "LOGIT", "CoT guided", "CoT minimal"], rows))
        w("```")

    # -- 3 -----------------------------------------------------------------
    w("\n## 3. Stack drift — exp3 vs exp4, identical condition\n")
    w("Same model, framing, readout, arms and N-per-cell; different vLLM, "
      "torch, transformers and driver versions. Any difference is the "
      "inference stack, not the treatment.\n")
    w("Run `06_evidence.py` section 3 for the exact version strings of each.\n")
    for label, defs in (("perturbation", ("3b", "1")), ("ATE_true", ("3", "3b"))):
        rows = []
        for model in models:
            for framing in ("sem", "abs"):
                a = find(meta, exp="exp3", model=model, framing=framing)
                b = find(meta, exp="exp4", model=model, framing=framing,
                         readout="logit")
                if not (a and b):
                    continue
                for opp in opps:
                    va, oa, za = contrast(cells, a[0], *defs, opp)
                    vb, ob, zb = contrast(cells, b[0], *defs, opp)
                    d = (None if va is None or vb is None else vb - va)
                    rows.append([model, framing, opp, mark(va, oa, za),
                                 mark(vb, ob, zb),
                                 grid4(d, signed=True)])
        w(f"\n### {label}\n")
        w("```")
        w(table(["model", "framing", "opp", "exp3", "exp4 LOGIT", "drift"], rows))
        w("```")

    # -- 4 -----------------------------------------------------------------
    w("\n## 4. Lexical falsification — framing, exp3\n")
    w("Identical blocks, identical positions, identical token parity. The only "
      "difference is whether the action labels carry meaning: `sem` = "
      "Cooperate/Defect, `swap` = the same words with their meanings inverted, "
      "`abs` = X/Y.\n")
    w("Pre-specified test: if the container effect is about the word "
      "\"Cooperate\", abstract labels should shrink it. If it is unchanged "
      "under X/Y, the lexical account is wrong.\n")
    for label, defs in (("perturbation", ("3b", "1")), ("ATE_true", ("3", "3b")),
                        ("baseline P(D|1)", None)):
        w(f"\n### {label}\n")
        rows = []
        for model in models:
            for opp in opps:
                cs = []
                for framing in ("sem", "swap", "abs"):
                    dbs = find(meta, exp="exp3", model=model, framing=framing)
                    if not dbs:
                        cs.append("-")
                        continue
                    if defs is None:
                        r = cells.get((dbs[0], "1", opp))
                        if not r or r.get("defect_ep") is None:
                            cs.append("-")
                        else:
                            v = grid4(r['defect_ep'])
                            cs.append(f"{v} VOID"
                                      if (r.get("off_task") or 0) > OFF_TASK_GATE
                                      else v)
                    else:
                        v, off, void = contrast(cells, dbs[0], *defs, opp)
                        cs.append(mark(v, off, void))
                rows.append([model, opp] + cs)
        w("```")
        w(table(["model", "opp", "semantic", "swap", "abstract"], rows))
        w("```")

    # -- 5 -----------------------------------------------------------------
    w("\n## 5. Manipulation check — CPR by arm\n")
    w(f"Pre-registered gate: CPR(arm 3) >= 0.85. exp1 failed it; every run "
      f"after passed. CPR takes no partial credit — all three probes must be "
      f"correct on a probed turn.\n")
    rows = []
    for db in sorted(meta):
        for opp in opps:
            r3 = cells.get((db, "3", opp))
            r3b = cells.get((db, "3b", opp))
            r1 = cells.get((db, "1", opp))
            if not r3:
                continue
            c3 = r3.get("cpr")
            rows.append([
                db, opp,
                "-" if r1 is None or r1.get("cpr") is None else f"{r1['cpr']:.3f}",
                "-" if r3b is None or r3b.get("cpr") is None else f"{r3b['cpr']:.3f}",
                "-" if c3 is None else f"{c3:.3f}",
                "-" if c3 is None else ("PASS" if c3 >= 0.85 else "FAIL"),
            ])
    w("```")
    w(table(["database", "opp", "CPR arm 1", "CPR arm 3b", "CPR arm 3",
             "gate"], rows))
    w("```")

    # -- 6 -----------------------------------------------------------------
    w("\n## 6. Regret against solved optimal play\n")
    w("`episode_regret` is payoff lost against the optimal policy for that "
      "opponent, computed by `cdx/optimal.py` — not assumed. Collected on every "
      "run and, before this table, never reported.\n")
    w("More interpretable than a defection rate: *defects 58% of the time* is "
      "hard to weigh, *loses N points of the available total* is not.\n")
    rows = []
    for db in sorted(meta):
        for arm in ("1", "3b", "3"):
            for opp in opps:
                r = cells.get((db, arm, opp))
                if not r or r.get("ep_regret") is None:
                    continue
                rows.append([db, arm, opp, f"{r['ep_regret']:.2f}",
                             int(r.get("ep_regret_n") or 0),
                             "-" if r.get("optimal_match") is None
                             else f"{r['optimal_match']:.3f}",
                             "VOID" if (r.get("off_task") or 0) > OFF_TASK_GATE
                             else ""])
    w("```")
    w(table(["database", "arm", "opp", "mean regret", "n", "P(action=optimal)",
             ""], rows))
    w("```")

    # -- 7 -----------------------------------------------------------------
    w("\n## 7. Token parity across arms\n")
    w("The study's central methodological claim is that treatment and placebo "
      "are token-matched. This checks it per database from `scaffold_tokens`, "
      "which records the real length of every injected block.\n")
    rows = []
    for db in sorted(meta):
        seen = {}
        for arm in ("3", "3b", "3c", "3d"):
            for opp in opps:
                r = cells.get((db, arm, opp))
                if r and r.get("scaf_mean") is not None:
                    seen.setdefault(arm, []).append(
                        (r.get("scaf_min"), r.get("scaf_max")))
        if not seen:
            continue
        allv = [v for pairs in seen.values() for pr in pairs for v in pr
                if v is not None]
        rows.append([db, ",".join(sorted(seen)),
                     f"{min(allv):.0f}..{max(allv):.0f}" if allv else "-",
                     "MATCHED" if allv and min(allv) == max(allv) else "SPREAD"])
    w("```")
    w(table(["database", "arms with a block", "token range", "verdict"], rows))
    w("```")
    w("\nA `SPREAD` means the block was not a constant token length. exp1 used "
      "character padding (`Your score: 003`), which fixes character width but "
      "not token count under BPE. Later runs enforce parity on token IDs.\n")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print(f"wrote {args.out}")
    print(f"databases joined: {len(meta)}   cells: {len(cells)}")
    unparsed = {k[0] for k in cells} - set(meta)
    if unparsed:
        print(f"UNPARSED (excluded from every table): {sorted(unparsed)}")
    if args.show_parse:
        for db, m in sorted(meta.items()):
            print(f"  {db:32} {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())