#!/usr/bin/env python3
"""DIAGNOSTIC 01 — why is Arm 3 comprehension only ~0.30?

THE QUESTION
    Arm 3 prints the agent's score, the opponent's last action and the round
    count into a fixed-width [STATE] block. Answering the probe suite is
    therefore a COPY operation. Observed CPR is 0.244 (tft) / 0.307 (allc).
    The pre-registered Phase 2 gate was 0.85.

    Until this is explained, the sentence "we repaired comprehension and
    behaviour did not follow" cannot be written. A reviewer's first move is to
    say the manipulation simply did not work.

WHAT THIS SCRIPT DISTINGUISHES
    A. SCAFFOLD BUG      the [STATE] block does not carry the value the probe
                         asks for  ->  Arm 3 is void, rerun required.
    B. SCORER BUG        the model answers correctly and score_answer() rejects
                         it  ->  free fix, cheap rerun of Arm 3 only.
    C. REAL FINDING      the block is correct, the answer is wrong, the model
                         cannot read state handed to it verbatim
                         ->  this is a second result, and a better one.

    Prior evidence already argues against (A): Arms 3 and 3b are token-parity
    matched and differ only in block CONTENT, yet differ in both defection
    (0.097 vs 0.044) and CPR (0.244 vs 0.000). A block that rendered nothing
    could not produce that.

READ-ONLY. Opens the database with mode=ro so the frozen artefact cannot be
mutated by an analysis pass.

    python analysis/01_diagnose_arm3.py --db sweep.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import textwrap
from pathlib import Path
from urllib.request import pathname2url

RULE = "=" * 78
PROBE_COLS = ("cpr_own_score", "cpr_opponent_last", "cpr_rounds_played")


def ro_uri(p: Path) -> str:
    """Read-only URI for a frozen database.

    mode=ro alone is NOT enough. These databases are written in WAL journal
    mode, and opening a WAL database read-only requires SQLite to create a
    -shm shared-memory file - which mode=ro forbids. SQLite reports that as
    "unable to open database file", which points nowhere near the cause. It
    only appears to work when stale -wal/-shm files happen to sit beside the
    database, which is why sweep.sqlite opened and the exp2 files did not.

    immutable=1 asserts the file cannot change, so WAL and shared memory are
    bypassed entirely. That is exactly true of a committed artefact, and it
    doubles as a guarantee that analysis cannot mutate the data.

    pathname2url percent-encodes the path; a space in a directory name would
    otherwise produce the same opaque error.
    """
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"database not found: {p.resolve()}\n"
                         f"Did you run `gunzip -k {p.name}.gz`?")
    conn = sqlite3.connect(ro_uri(p), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# --------------------------------------------------------------------------
# 1. per-probe-kind pass rates
# --------------------------------------------------------------------------

def per_kind(conn) -> None:
    section("1  PER-PROBE PASS RATE  (probed turns only)")
    print("  A CPR of 0 with high per-kind rates means the three probes never")
    print("  succeed together. CPR takes no partial credit by design.\n")
    print(f"  {'arm':<5}{'opp':<6}{'n':>7}{'own_score':>12}{'opp_last':>11}"
          f"{'rounds':>9}{'CPR':>8}{'independent':>13}")
    print("  " + "-" * 71)
    rows = conn.execute(
        f"""SELECT arm, opponent_policy AS opp, COUNT(*) AS n,
                   AVG({PROBE_COLS[0]}) AS own,
                   AVG({PROBE_COLS[1]}) AS last,
                   AVG({PROBE_COLS[2]}) AS rounds,
                   AVG(cpr_score) AS cpr
            FROM turns
            WHERE cpr_score IS NOT NULL
            GROUP BY arm, opponent_policy
            ORDER BY arm, opponent_policy"""
    ).fetchall()
    for r in rows:
        indep = r["own"] * r["last"] * r["rounds"]
        flag = ""
        if r["cpr"] < indep - 0.02:
            flag = "  <- anti-correlated"
        print(f"  {r['arm']:<5}{r['opp']:<6}{r['n']:>7}{r['own']:>12.3f}"
              f"{r['last']:>11.3f}{r['rounds']:>9.3f}{r['cpr']:>8.3f}"
              f"{indep:>13.3f}{flag}")
    print("\n  'independent' is what CPR would be if the three probes were")
    print("  independent. CPR far below it means failures co-occur: the probes")
    print("  that pass are not the ones that pass elsewhere.")


# --------------------------------------------------------------------------
# 2. pass rate by turn — is 0.20 just "turn 0 is trivial"?
# --------------------------------------------------------------------------

def by_turn(conn) -> None:
    section("2  PASS RATE BY TURN")
    print("  At turn 0 the score is 0 and no round has been played, so those")
    print("  probes are trivially answerable WITHOUT any state tracking.")
    print("  If own_score passes only at turn 0, an overall 0.200 is exactly")
    print("  1 of 5 probed turns and represents ZERO comprehension.\n")
    for arm in ("1", "3", "3b"):
        rows = conn.execute(
            f"""SELECT turn, COUNT(*) AS n,
                       AVG({PROBE_COLS[0]}) AS own,
                       AVG({PROBE_COLS[1]}) AS last,
                       AVG({PROBE_COLS[2]}) AS rounds,
                       AVG(cpr_score) AS cpr
                FROM turns
                WHERE cpr_score IS NOT NULL AND arm = ?
                GROUP BY turn ORDER BY turn""",
            (arm,),
        ).fetchall()
        if not rows:
            continue
        print(f"  arm {arm}")
        print(f"    {'turn':>5}{'n':>8}{'own_score':>12}{'opp_last':>11}"
              f"{'rounds':>9}{'CPR':>8}")
        for r in rows:
            print(f"    {r['turn']:>5}{r['n']:>8}{r['own']:>12.3f}"
                  f"{r['last']:>11.3f}{r['rounds']:>9.3f}{r['cpr']:>8.3f}")
        nonzero = [r for r in rows if r["turn"] > 0]
        if nonzero:
            own_post = sum(r["own"] * r["n"] for r in nonzero) / sum(r["n"] for r in nonzero)
            print(f"    own_score excluding turn 0: {own_post:.3f}")
            if own_post < 0.05:
                print("    VERDICT: own_score passes at turn 0 ONLY. This arm "
                      "demonstrates no score tracking.")
        print()


# --------------------------------------------------------------------------
# 3. does the [STATE] block actually contain the answer?
# --------------------------------------------------------------------------

def prompt_check(conn, samples: int) -> None:
    section("3  DOES THE PROMPT CARRY THE STATE?  (Arm 3)")
    print("  If [STATE] is absent or the score field is missing, Arm 3 is void")
    print("  and every ATE_true in the run is meaningless.\n")
    rows = conn.execute(
        """SELECT t.turn, t.opponent_policy, d.prompt_preview
           FROM turns t JOIN turn_details d
             ON t.run_id=d.run_id AND t.episode_id=d.episode_id
            AND t.arm=d.arm AND t.turn=d.turn
            AND t.opponent_policy=d.opponent_policy
           WHERE t.arm='3' AND t.turn > 0 AND d.prompt_preview IS NOT NULL
           ORDER BY t.episode_id LIMIT ?""",
        (samples,),
    ).fetchall()
    if not rows:
        print("  NO prompt_preview rows for arm 3. Cannot verify rendering "
              "from the database; fall back to rebuilding a prompt offline.")
        return
    has_state = sum(1 for r in rows if "[STATE]" in (r["prompt_preview"] or ""))
    print(f"  sampled {len(rows)} previews; {has_state} contain '[STATE]'")
    if has_state == 0:
        print("\n  *** SCAFFOLD BUG: the state block is not in the prompt. ***")
        print("  *** Arm 3 is void. Do not report ATE_true.               ***")
    for r in rows[:3]:
        print(f"\n  --- turn {r['turn']} vs {r['opponent_policy']} ---")
        print(textwrap.indent((r["prompt_preview"] or "")[:600], "  | "))


# --------------------------------------------------------------------------
# 4. what did the model actually say?
# --------------------------------------------------------------------------

def raw_answers(conn, samples: int) -> None:
    section("4  RAW PROBE ANSWERS vs SCORES  (Arm 3, turn > 0)")
    print("  This separates a scorer bug from a comprehension failure.")
    print("  If the text is right and the score is 0, the SCORER is wrong.\n")
    rows = conn.execute(
        """SELECT t.turn, t.cpr_own_score, t.cpr_opponent_last,
                  t.cpr_rounds_played, d.probe_answers
           FROM turns t JOIN turn_details d
             ON t.run_id=d.run_id AND t.episode_id=d.episode_id
            AND t.arm=d.arm AND t.turn=d.turn
            AND t.opponent_policy=d.opponent_policy
           WHERE t.arm='3' AND t.turn > 0
             AND d.probe_answers IS NOT NULL
             AND t.cpr_own_score = 0
           ORDER BY t.episode_id LIMIT ?""",
        (samples,),
    ).fetchall()
    if not rows:
        print("  No failing own_score rows with stored answers.")
        return
    for r in rows:
        print(f"  turn {r['turn']:>2}  own={r['cpr_own_score']} "
              f"last={r['cpr_opponent_last']} rounds={r['cpr_rounds_played']}")
        try:
            payload = json.loads(r["probe_answers"])
            if isinstance(payload, dict):
                for k, v in payload.items():
                    print(f"      {k:<16} {str(v)[:110]!r}")
            else:
                print(f"      {str(payload)[:200]!r}")
        except Exception:
            print(f"      {str(r['probe_answers'])[:200]!r}")
        print()
    print("  Judge by eye: is the ANSWER correct and the SCORE zero?")
    print("    yes -> scorer bug, fix normalise()/first_segment, rerun arm 3")
    print("    no  -> the model genuinely cannot read the state it was given")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sweep.sqlite")
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    conn = connect(args.db)
    print(f"\n{RULE}\nARM 3 COMPREHENSION DIAGNOSTIC\n{RULE}")
    print(f"  database  {Path(args.db).resolve()}  (read-only)")
    meta = conn.execute(
        "SELECT probe_hash, git_commit, gpu_name FROM run_meta LIMIT 1"
    ).fetchone()
    if meta:
        print(f"  probes    {meta['probe_hash'][:32]}...")
        print(f"  code      {meta['git_commit'][:12]}   gpu {meta['gpu_name']}")

    per_kind(conn)
    by_turn(conn)
    prompt_check(conn, args.samples)
    raw_answers(conn, args.samples)

    section("WHAT TO CONCLUDE")
    print("""
  A  no '[STATE]' in prompts            -> SCAFFOLD BUG. Arm 3 void.
  B  answers correct, scores 0          -> SCORER BUG. Cheap fix + rerun.
  C  block present, answers wrong       -> REAL FINDING. Report it.

  In case C the paper gains a second result: an 8B model handed its own
  state in a fixed-width block still cannot report that state ~70% of the
  time. State that as a finding; do NOT claim comprehension was repaired.

  In every case the Arm 1 vs Arm 3b perturbation result is UNAFFECTED. The
  placebo is non-diagnostic by construction, so that contrast does not
  depend on anything this script finds.
""")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())