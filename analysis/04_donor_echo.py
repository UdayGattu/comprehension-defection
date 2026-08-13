#!/usr/bin/env python3
"""Arm 3c: did the model READ the injected block, or ignore it?

WHY THIS EXISTS
    Every behavioural result in this project rests on an unstated premise: that
    the model actually reads the [STATE] block rather than routing around it.
    exp2 inferred block-reading indirectly, from Qwen's 3c own_score dropping
    0.400 -> 0.200. exp3 added turns.donor_agent_score precisely so the question
    could be answered DIRECTLY, and then nobody asked it.

    Arm 3c renders the treatment template from ANOTHER episode's state. So when
    the model is probed for its own score, its answer falls into one of four
    buckets, and each means something different:

      CORRECT      matches the true score. The model tracked the game and
                   ignored the block, or the block happened to agree.
      DONOR_ECHO   matches the number the block DISPLAYED. Direct proof the
                   model read the block and reported it over the truth. This is
                   the mechanism behind the whole container story.
      OFF_BY_ONE   true score +/- 1. Motivated by the first rows inspected:
                   got 11 want 12, got 23 want 24. An arithmetic slip is a
                   different phenomenon from block-reading and must not be
                   silently counted as one.
      OTHER        neither. Investigate before claiming anything.

    A high DONOR_ECHO share is the cleanest possible evidence that the block is
    read. A high OFF_BY_ONE share instead says the model is summing and losing
    a round - a capability finding, not an attention finding.

TURN 0 IS EXCLUDED, AND THIS IS NOT OPTIONAL
    Every episode starts at score 0 with no last move, so no distinct donor can
    exist and donor_degenerate = 1. There the donor IS the true state, a correct
    answer is indistinguishable from an echo, and including those rows would
    manufacture echo evidence out of nothing.

WHAT THIS CANNOT SHOW
    The split is observational. Episodes are not randomised on whether the model
    echoes, so a defection difference between echoing and non-echoing episodes
    is a correlation, not a treatment effect. Report it as a descriptive
    relationship. The randomised comparison is the arm contrast, which lives in
    02_episode_level.py.

    python analysis/04_donor_echo.py --db exp3_llama_sem.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import pathname2url

RULE = "=" * 78


def ro_uri(p: Path) -> str:
    """Read-only URI that also works on WAL databases.

    mode=ro alone fails on a WAL database because SQLite wants to create the
    -shm/-wal sidecars. immutable=1 promises we are the only reader.
    """
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def classify(got: str | None, want: str | None, donor: int | None) -> str:
    if got is None or want is None:
        return "NO_ANSWER"
    g, w = got.strip(), want.strip()
    if g == w:
        return "CORRECT"
    try:
        gi, wi = int(g), int(w)
    except ValueError:
        return "OTHER"
    if donor is not None and gi == donor:
        return "DONOR_ECHO"
    if abs(gi - wi) == 1:
        return "OFF_BY_ONE"
    return "OTHER"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--arm", default="3c")
    args = ap.parse_args()

    path = Path(args.db)
    con = sqlite3.connect(ro_uri(path), uri=True)

    cols = {r[1] for r in con.execute("PRAGMA table_info(turns)")}
    if "donor_agent_score" not in cols:
        print(f"{path.name}: no donor_agent_score column - this database "
              f"predates exp3. Nothing to do.")
        return 1

    print(f"\n{RULE}\nARM {args.arm} BLOCK-READING AUDIT\n{RULE}")
    print(f"  database  {path.resolve()}  (read-only)")

    rows = con.execute(
        """
        SELECT t.opponent_policy, t.episode_id, t.turn,
               t.donor_agent_score, t.donor_degenerate, d.probe_answers
        FROM turns t
        JOIN turn_details d USING
             (run_id, episode_id, arm, model_id, readout_mode,
              opponent_policy, turn)
        WHERE t.arm = ? AND d.probe_answers IS NOT NULL
        """,
        (args.arm,),
    ).fetchall()

    if not rows:
        print(f"  no probed rows in arm {args.arm}.")
        return 1

    by_opp: dict[str, Counter] = defaultdict(Counter)
    by_turn: dict[int, Counter] = defaultdict(Counter)
    echo_eps: dict[tuple[str, int], list[bool]] = defaultdict(list)
    degenerate_skipped = 0

    for opp, ep, turn, donor, degen, raw in rows:
        # Turn 0 and any turn where no distinct donor existed: the donor IS the
        # true state, so an echo is unobservable. Excluding rather than
        # counting them is what keeps DONOR_ECHO meaningful.
        if degen:
            degenerate_skipped += 1
            continue
        try:
            probes = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        own = probes.get("own_score") or {}
        kind = classify(own.get("got"), own.get("want"), donor)
        by_opp[opp][kind] += 1
        by_turn[turn][kind] += 1
        echo_eps[(opp, ep)].append(kind == "DONOR_ECHO")

    kinds = ["CORRECT", "DONOR_ECHO", "OFF_BY_ONE", "OTHER", "NO_ANSWER"]

    print(f"\n{RULE}\n1  own_score ANSWERS BY OPPONENT\n{RULE}")
    print("  DONOR_ECHO is the number the block displayed. Any share well above")
    print("  zero is direct evidence the model read the injected block.\n")
    print(f"  {'opp':6}{'n':>7}" + "".join(f"{k:>12}" for k in kinds))
    for opp in sorted(by_opp):
        c = by_opp[opp]
        n = sum(c.values())
        print(f"  {opp:6}{n:>7}" + "".join(f"{c[k]/n:>12.3f}" for k in kinds))
    print(f"\n  rows skipped as degenerate donor (incl. every turn 0): "
          f"{degenerate_skipped}")

    print(f"\n{RULE}\n2  BY TURN\n{RULE}")
    print("  If DONOR_ECHO rises with turn, the block matters more as the true")
    print("  state drifts further from it. If it is flat, block-reading is")
    print("  unconditional.\n")
    print(f"  {'turn':>5}{'n':>8}" + "".join(f"{k:>12}" for k in kinds))
    for turn in sorted(by_turn):
        c = by_turn[turn]
        n = sum(c.values())
        print(f"  {turn:>5}{n:>8}" + "".join(f"{c[k]/n:>12.3f}" for k in kinds))

    print(f"\n{RULE}\n3  DOES ECHOING PREDICT DEFECTION?\n{RULE}")
    print("  OBSERVATIONAL. Episodes are not randomised on echoing, so this is")
    print("  a correlation. The randomised contrast is arm 3c vs arm 3.\n")

    defect = {
        (o, e): (d, n)
        for o, e, d, n in con.execute(
            "SELECT opponent_policy, episode_id, "
            "       SUM(agent_action='D'), COUNT(*) "
            "FROM turns WHERE arm=? GROUP BY opponent_policy, episode_id",
            (args.arm,),
        )
    }

    print(f"  {'opp':6}{'group':>14}{'n_ep':>7}{'P(defect)':>12}{'sd':>9}")
    for opp in sorted(by_opp):
        buckets: dict[str, list[float]] = {"echoed >=1": [], "never echoed": []}
        for (o, ep), marks in echo_eps.items():
            if o != opp or (o, ep) not in defect:
                continue
            d, n = defect[(o, ep)]
            if not n:
                continue
            buckets["echoed >=1" if any(marks) else "never echoed"].append(d / n)
        for label, vals in buckets.items():
            if not vals:
                print(f"  {opp:6}{label:>14}{0:>7}{'-':>12}{'-':>9}")
                continue
            sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"  {opp:6}{label:>14}{len(vals):>7}"
                  f"{st.mean(vals):>12.3f}{sd:>9.3f}")

    print(f"\n{RULE}\nHOW TO READ THIS\n{RULE}")
    print("""
  DONOR_ECHO high      The model reads the block and reports it over the truth.
                       This is the direct evidence exp2 could only infer, and it
                       supports every claim that the block is attended to.

  OFF_BY_ONE high      The model is summing payoffs and losing a round. A
                       capability finding about arithmetic, NOT about attention.
                       Do not report it as block-reading.

  CORRECT high         The model tracked the true state and ignored the block.
                       Then arm 3c's behavioural effect, whatever it is, is not
                       mediated by reading the false numbers - and that needs
                       saying explicitly, because it weakens the mechanism.

  OTHER dominant       Neither hypothesis. Inspect raw answers before writing
                       anything: a parser fault would look exactly like this.
""")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())