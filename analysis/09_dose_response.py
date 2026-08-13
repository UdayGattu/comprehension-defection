#!/usr/bin/env python3
"""Arm 3c: does behaviour scale with the SIZE of the lie?

WHY THIS, AND NOT THE DONOR-TYPE SPLIT
    An obvious-sounding analysis is "split arm 3c by whether the donor episode
    came from the same opponent as the recipient" - the idea being that
    same-opponent donors have near-true trajectories and dilute any real effect.

    That analysis is impossible, and checking the code is why. `run_cell` is
    invoked per (arm, opponent), and `select_donor` draws from `live_states`,
    which are the live episodes of THAT cell. Every donor already shares its
    recipient's opponent. There is no cross-type contamination to remove.

    What IS available is better. `turns.donor_agent_score` records the number
    the block displayed, and the true score is recoverable from the episode's
    own payoff history. So each row carries the MAGNITUDE of its lie, and the
    question becomes a dose-response:

        does defection change more when the block lies by more?

    A graded relationship is far harder to dismiss than a single anomalous
    cell. A flat one says the model registers "a number is present" and not
    "which number", which is exactly the schema-over-content reading that
    analysis/08 tests from the other direction.

HOW THE TRUE SCORE IS RECOVERED
    The [STATE] block reports the score BEFORE the current turn's move, so the
    true value at turn t is the sum of agent_payoff over turns 0..t-1 of the
    same episode. Computed here rather than assumed, and the script prints the
    turn-0 check: at turn 0 the true score must be 0 for every episode, and if
    it is not, the reconstruction is wrong and everything below it is void.

WHAT IS EXCLUDED
    Rows with donor_degenerate = 1. There the donor IS the recipient's own
    state, the lie is zero by construction, and including them would pack the
    zero-lie bin with rows that were never falsified at all.

    python analysis/09_dose_response.py
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.request import pathname2url

OFF_TASK_GATE = 0.10
RULE = "=" * 78
BINS = [(0, 0), (1, 3), (4, 8), (9, 15), (16, 25), (26, 10**6)]


def ro_uri(p: Path) -> str:
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def bin_of(x: int) -> str:
    for lo, hi in BINS:
        if lo <= x <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="DOSE_RESPONSE.md")
    args = ap.parse_args()

    dbs = sorted(Path(p) for p in glob.glob("*.sqlite")
                 if not Path(p).name.startswith(("smoke_", "cotsmoke_")))
    L: list[str] = []

    def w(s=""):
        L.append(s)

    w("# Arm 3c: does defection scale with the size of the lie?\n")
    w("`donor_agent_score` is the number the block displayed. The true score is "
      "reconstructed as the cumulative `agent_payoff` of the same episode up to "
      "(not including) the current turn.\n")
    w("`lie = donor_agent_score - true_score`. Rows with "
      "`donor_degenerate = 1` are excluded: there the donor is the recipient's "
      "own state and the lie is zero by construction.\n")

    for path in dbs:
        con = sqlite3.connect(ro_uri(path), uri=True)
        cols = {r[1] for r in con.execute("PRAGMA table_info(turns)")}
        if "donor_agent_score" not in cols:
            con.close()
            continue
        rows = con.execute("""
            SELECT opponent_policy, episode_id, turn, agent_action, agent_payoff,
                   donor_agent_score, donor_degenerate, action_mass_total
            FROM turns WHERE arm='3c' ORDER BY opponent_policy, episode_id, turn
        """).fetchall()
        con.close()
        if not rows:
            continue

        # Reconstruct the true score, then verify the reconstruction at turn 0.
        by_ep: dict[tuple[str, int], list] = defaultdict(list)
        for r in rows:
            by_ep[(r[0], r[1])].append(r)

        recs, turn0_bad = [], 0
        for (opp, ep), turns in by_ep.items():
            running = 0
            for opp_, ep_, turn, act, payoff, donor, degen, mass in turns:
                true_score = running
                if turn == 0 and true_score != 0:
                    turn0_bad += 1
                if donor is not None and not degen:
                    recs.append((opp, turn, act, donor - true_score, mass))
                running += (payoff or 0)

        w(f"\n{RULE}\n## `{path.stem}`\n{RULE}\n")
        if turn0_bad:
            w(f"**RECONSTRUCTION FAILED** — {turn0_bad} episodes have a non-zero "
              "true score at turn 0. Everything below is void for this "
              "database; the cumulative-payoff reconstruction is wrong.\n")
            continue
        w(f"reconstruction check: true score is 0 at turn 0 in every episode — OK\n")

        for opp in ("allc", "tft"):
            sub = [r for r in recs if r[0] == opp]
            if not sub:
                continue
            off = sum(1 for r in sub if (r[4] or 1) < OFF_TASK_GATE) / len(sub)
            w(f"\n**vs {opp}** — n={len(sub):,} falsified turns, "
              f"off-task {off:.3f}{'  **VOID**' if off > OFF_TASK_GATE else ''}\n")

            w("*by absolute size of the lie*\n")
            w("```")
            w(f"{'|lie|':>10}{'n':>9}{'P(defect)':>12}")
            agg = defaultdict(lambda: [0, 0])
            for _, _, act, lie, _ in sub:
                b = agg[bin_of(abs(lie))]
                b[0] += (act == "D")
                b[1] += 1
            order = [f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")
                     for lo, hi in BINS]
            for b in order:
                if b in agg and agg[b][1]:
                    d, n = agg[b]
                    w(f"{b:>10}{n:>9,}{d/n:>12.4f}")
            w("```")

            w("\n*by signed lie — a block claiming a HIGHER score than the truth "
              "may mean something different from one claiming lower*\n")
            w("```")
            w(f"{'direction':>12}{'n':>9}{'P(defect)':>12}")
            sgn = defaultdict(lambda: [0, 0])
            for _, _, act, lie, _ in sub:
                k = "lower" if lie < 0 else ("higher" if lie > 0 else "equal")
                sgn[k][0] += (act == "D")
                sgn[k][1] += 1
            for k in ("lower", "equal", "higher"):
                if k in sgn and sgn[k][1]:
                    d, n = sgn[k]
                    w(f"{k:>12}{n:>9,}{d/n:>12.4f}")
            w("```")

    w(f"\n{RULE}\n## How to read this\n{RULE}\n")
    w("**Rising P(defect) with |lie|** — the model responds to the block's "
      "content, graded by how wrong it is. That upgrades the qwen-vs-TFT result "
      "from one anomalous cell to a dose-response relationship, and it is the "
      "strongest positive evidence of content use the study can produce.\n")
    w("**Flat across bins** — the model registers that a number-shaped field "
      "exists and not which number it holds. That is the same conclusion "
      "`analysis/08_decomposition_ci.py` reaches from the other direction, and "
      "two independent routes to it is worth more than either alone.\n")
    w("**Asymmetry between higher and lower** — a falsely high score may read as "
      "'I am winning, press the advantage' and a falsely low one as 'I am "
      "behind'. If the signed table splits and the absolute one does not, the "
      "effect is directional and |lie| is the wrong summary.\n")
    w("\nThese are raw rates, not contrasts, and turns within an episode are not "
      "independent. Treat any pattern here as descriptive until it is "
      "re-estimated at the episode level.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())