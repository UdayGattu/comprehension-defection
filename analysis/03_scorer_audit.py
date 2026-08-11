#!/usr/bin/env python3
"""DIAGNOSTIC 03 — is the score failure the MODEL or the SCORER?

Two anomalies from diagnostic 01 remain unexplained, and both must be resolved
before spending GPU time, because each one changes what the re-run should be.

  Q1  Arm 3 own_score by turn:  0.079, 0.154, 0.180, 0.749
      The turn-16 value is 4x the others. Either the model genuinely gets
      better late, or something about turn 16 is scored differently.

  Q2  Turn-0 opponent_last:  arm 1 = 1.000, arm 3 = 0.216, arm 3b = 0.000
      There IS no last move at turn 0. Three arms scoring three different
      values against the same non-existent ground truth is a scorer problem,
      not a model problem.

Rather than eyeballing a dozen rows, this classifies EVERY failing answer into
a mechanism. A distribution over mechanisms is evidence; a handful of examples
is an anecdote.

MECHANISMS TESTED
  leading_zero      answer equals the first digit(s) of a zero-padded field
                    ("024" read as "0"). Directly attributable to the
                    :0{width}d format spec in scaffold.treatment_text.
  last_payoff       answer equals the most recent round's payoff (5/3/1/0)
                    rather than the cumulative total.
  per_round         answer equals score/rounds - reporting a rate.
  off_by_padding    answer equals want with leading zeros stripped/added
                    -> pure SCORER bug, the model was right.
  opponent_score    answer equals the opponent's score field instead.
  other_number      a number, none of the above.
  non_numeric       no number extracted.

    python analysis/03_scorer_audit.py --db sweep.sqlite
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

RULE = "=" * 78
JOIN = ("run_id", "episode_id", "arm", "model_id",
        "readout_mode", "opponent_policy", "turn")


def connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"database not found: {p.resolve()}")
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_answers(blob: str | None) -> dict[str, dict]:
    """probe_answers is JSON whose values are python-repr dicts. Tolerate both."""
    if not blob:
        return {}
    try:
        outer = json.loads(blob)
    except Exception:
        try:
            outer = ast.literal_eval(blob)
        except Exception:
            return {}
    if not isinstance(outer, dict):
        return {}
    parsed: dict[str, dict] = {}
    for kind, val in outer.items():
        if isinstance(val, dict):
            parsed[kind] = val
            continue
        try:
            inner = ast.literal_eval(val)
            if isinstance(inner, dict):
                parsed[kind] = inner
        except Exception:
            parsed[kind] = {"got": str(val), "want": None, "mark": None}
    return parsed


def first_number(text: str | None) -> int | None:
    if text is None:
        return None
    m = re.search(r"-?\d+", str(text))
    return int(m.group()) if m else None


def classify(got_raw: str | None, want_raw: str | None,
             turn: int, opp_score: int | None) -> str:
    got_s = "" if got_raw is None else str(got_raw).strip()
    want = first_number(want_raw)
    got = first_number(got_s)

    if got is None:
        return "non_numeric"
    if want is None:
        return "other_number"
    if got == want:
        return "off_by_padding"          # marked wrong but numerically right
    if opp_score is not None and got == opp_score:
        return "opponent_score"

    want_padded = f"{want:03d}"
    # "024" read as 0, or as 02 -> 2
    if got_s and want_padded.startswith("0"):
        for k in (1, 2):
            if got_s.lstrip("+-").startswith(want_padded[:k]) and got != want:
                if got == int(want_padded[:k]):
                    return "leading_zero"
    if got in (0, 1, 3, 5) and want > 5:
        return "last_payoff"
    if turn > 0 and want % max(turn, 1) == 0 and got == want // max(turn, 1):
        return "per_round"
    return "other_number"


def q1_turn16(conn, arm: str = "3") -> None:
    print(f"\n{RULE}\nQ1  THE TURN-16 SPIKE  (arm {arm}, own_score)\n{RULE}")
    rows = conn.execute(
        f"""SELECT t.turn, t.cpr_own_score, d.probe_answers
            FROM turns t JOIN turn_details d USING ({','.join(JOIN)})
            WHERE t.arm=? AND t.cpr_score IS NOT NULL
              AND d.probe_answers IS NOT NULL""",
        (arm,),
    ).fetchall()

    want_by_turn: dict[int, Counter] = defaultdict(Counter)
    got_by_turn: dict[int, Counter] = defaultdict(Counter)
    pass_by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        a = parse_answers(r["probe_answers"]).get("own_score")
        if not a:
            continue
        w, g = first_number(a.get("want")), first_number(a.get("got"))
        want_by_turn[r["turn"]][w] += 1
        got_by_turn[r["turn"]][g] += 1
        pass_by_turn[r["turn"]].append(r["cpr_own_score"] or 0)

    print(f"  {'turn':>5}{'n':>7}{'pass':>8}   most common WANT        most common GOT")
    print("  " + "-" * 68)
    for turn in sorted(pass_by_turn):
        marks = pass_by_turn[turn]
        rate = sum(marks) / len(marks)
        w = ", ".join(f"{k}x{v}" for k, v in want_by_turn[turn].most_common(3))
        g = ", ".join(f"{k}x{v}" for k, v in got_by_turn[turn].most_common(3))
        print(f"  {turn:>5}{len(marks):>7}{rate:>8.3f}   {w:<22}  {g}")

    print("\n  If turn 16's WANT distribution is unremarkable but its GOT")
    print("  distribution suddenly matches, the model changed. If WANT itself")
    print("  looks different (e.g. a value with no leading zero), the FORMAT")
    print("  changed and the spike is an artifact of the padding.")


def q2_turn0(conn) -> None:
    print(f"\n{RULE}\nQ2  TURN-0 opponent_last ACROSS ARMS\n{RULE}")
    print("  There is no last move at turn 0. Any disagreement between arms")
    print("  here is the scorer, not the model.\n")
    for arm in ("1", "3", "3b"):
        rows = conn.execute(
            f"""SELECT t.cpr_opponent_last, d.probe_answers
                FROM turns t JOIN turn_details d USING ({','.join(JOIN)})
                WHERE t.arm=? AND t.turn=0 AND d.probe_answers IS NOT NULL
                LIMIT 400""",
            (arm,),
        ).fetchall()
        if not rows:
            print(f"  arm {arm}: no stored answers")
            continue
        pairs = Counter()
        marks = []
        for r in rows:
            a = parse_answers(r["probe_answers"]).get("opponent_last")
            if not a:
                continue
            pairs[(str(a.get("got"))[:24], str(a.get("want"))[:24])] += 1
            marks.append(r["cpr_opponent_last"] or 0)
        rate = sum(marks) / len(marks) if marks else float("nan")
        print(f"  arm {arm}   pass {rate:.3f}   n={len(marks)}")
        for (got, want), n in pairs.most_common(4):
            print(f"      got={got!r:<26} want={want!r:<20} n={n}")
        print()


def q3_mechanisms(conn, arm: str = "3") -> None:
    print(f"\n{RULE}\nQ3  FAILURE MECHANISM DISTRIBUTION  (arm {arm}, own_score)\n{RULE}")
    rows = conn.execute(
        f"""SELECT t.turn, t.cpr_own_score, d.probe_answers
            FROM turns t JOIN turn_details d USING ({','.join(JOIN)})
            WHERE t.arm=? AND t.turn > 0 AND t.cpr_score IS NOT NULL
              AND d.probe_answers IS NOT NULL""",
        (arm,),
    ).fetchall()

    buckets = Counter()
    total = 0
    for r in rows:
        parsed = parse_answers(r["probe_answers"])
        a = parsed.get("own_score")
        if not a:
            continue
        total += 1
        if r["cpr_own_score"]:
            buckets["CORRECT"] += 1
            continue
        buckets[classify(a.get("got"), a.get("want"), r["turn"], None)] += 1

    if not total:
        print("  no rows")
        return
    print(f"  {'mechanism':<20}{'n':>8}{'share':>9}")
    print("  " + "-" * 37)
    for name, n in buckets.most_common():
        print(f"  {name:<20}{n:>8}{n/total:>9.1%}")
    print(f"  {'TOTAL':<20}{total:>8}")

    print("""
  READING THE TABLE
    off_by_padding large   -> SCORER bug. The model was right and was marked
                              wrong. Fix normalise(); no model re-run needed
                              for that portion.
    leading_zero large     -> FORMAT bug. The :0{width}d spec is the cause.
                              Patch scaffold.py and re-run arm 3.
    last_payoff large      -> the model is not summing; it reports the most
                              recent payoff. A genuine capability finding.
    other_number dominant  -> neither; investigate before spending anything.
""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sweep.sqlite")
    ap.add_argument("--arm", default="3")
    args = ap.parse_args()
    conn = connect(args.db)
    print(f"\n{RULE}\nSCORER AUDIT\n{RULE}")
    print(f"  database  {Path(args.db).resolve()}  (read-only)")
    q1_turn16(conn, args.arm)
    q2_turn0(conn)
    q3_mechanisms(conn, args.arm)
    print(f"\n{RULE}\nDECISION\n{RULE}")
    print("""
  scorer bug only          -> fix normalise(), re-score OFFLINE from the frozen
                              database. No GPU. No new data. Cheapest outcome.
  format bug               -> patch scaffold.py, re-run arm 3 only (~$0.15).
  genuine model failure    -> report as a finding; re-run only to demonstrate
                              that space-padding does not rescue it.
""")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())