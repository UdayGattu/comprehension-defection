#!/usr/bin/env python3
"""ANALYSIS 13 - exp6, which FIELD of the state block carries the effect.

WHAT THE RUN PRINTED IS NOT QUOTABLE, FOR FOUR REASONS
    1. TURN 0 DILUTION. Arm 3m flips the opponent's last move, but at turn 0
       there is no last move, so the block is byte-identical to arm 3. Those
       rows carry NO manipulation. They are 1/20 of every 3m cell and they pull
       every 3-3m estimate toward zero. Arm 3c has the same problem wherever
       donor_degenerate=1.

    2. TURN-LEVEL STANDARD ERRORS. The run's intervals treat 20 turns inside an
       episode as 20 independent observations. They are not. Across exp2-exp5
       the episode-level SE was 0.62x to 3.75x the turn-level one.

    3. THE CPR GATE IS BEING APPLIED TO THE WRONG ARMS. CPR scores the model
       against the TRUE state. Arms 3c, 3s and 3m show it a FALSE state, so a
       model that reads the block and believes it answers the falsified value
       and is marked wrong. Low CPR there is evidence the manipulation WORKED.
       Applying a 0.85 gate to those arms would discard exactly the cells that
       carry the result. The gate belongs on arms 1, 3b and 3 only.

    4. NO MANIPULATION-STRENGTH DENOMINATOR. Arm 3c's effect cannot be compared
       to arm 3m's without knowing how often each actually falsified anything.
       3m falsifies 100% of non-zero turns by construction; 3c falsifies only
       when the donor happened to differ, which is 0.0% vs ALLC and 37.7% vs
       TFT. Reporting the two effects side by side without that denominator
       invites the reader to conclude the fields differ in importance when the
       arms differ in dose.

WHAT THIS SCRIPT DOES
    Part 1  manipulation integrity   did each arm lie, and how often
    Part 2  gates                    CPR/off-task, gate on 1/3b/3 only
    Part 3  contrasts                episode-level bootstrap, turn 0 excluded
    Part 4  dose-response            falsification rate vs effect size
    Part 5  the echo test            arm 3s: did the model repeat the lie?

PART 5 IS THE ONE TO READ FIRST
    Arm 3s displays a score 15 points wrong and changes nothing else. If the
    model's own-score probe answer equals the DISPLAYED number rather than the
    true one, it demonstrably read the block. If 3-3s is simultaneously zero,
    then the model read a false state, repeated it back correctly, and made the
    same decision it would have made anyway.

    That is a dissociation between comprehension and use, measured directly
    rather than inferred - and it is the strongest available answer to "maybe
    the model just never read the block."

USAGE
    gunzip -kf exp6_*.sqlite.gz
    python analysis/13_exp6_fields.py
    python analysis/13_exp6_fields.py --db exp6_qwen_sem_logit.sqlite
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import pathname2url

try:
    import numpy as np
except ImportError:                                    # pragma: no cover
    sys.exit("numpy required:  pip install numpy")

RULE = "=" * 78
SEED = 20260811          # same seed as analysis/02, so intervals are comparable
N_BOOT = 10_000
CPR_GATE = 0.85
OFFTASK_GATE = 0.10

# CPR scores the model against the TRUE state, so it is only a validity check
# for the one arm that DISPLAYS the true state.
#
#   arm 1   no [STATE] block at all
#   arm 3b  a density- and token-matched block containing no real state
#   3c/3s/3m  a block that asserts something false
#
# An earlier version of this script gated arms 1 and 3b too, and flagged
# "CPR BELOW GATE" on every group in exp6 - a false alarm on arms that have
# nothing true to report. The informative quantity for 3b is not its CPR but
# the CONTRAST CPR(3) - CPR(3b), which is the run's manipulation check.
CPR_GATED_ARMS = ("3",)
NO_TRUE_STATE = ("1", "3b")              # CPR here is a floor, not a failure
LYING_ARMS = ("3c", "3s", "3m")          # CPR here is a BELIEF measure
_INT_RE = re.compile(r"-?\d+")


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def ro_uri(p: Path) -> str:
    """Read-only, WAL-independent handle.

    mode=ro alone fails on these databases: they are in WAL journal mode and
    opening WAL read-only needs to create a -shm file, which mode=ro forbids.
    immutable=1 bypasses WAL entirely - true by definition for a committed
    artefact, and a guarantee this script cannot mutate the evidence.
    """
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(
            f"not found: {path.name}\n"
            f"  The repository stores only the .gz. Run:\n"
            f"    gunzip -kf {path.name}.gz"
        )
    c = sqlite3.connect(ro_uri(path), uri=True)
    c.row_factory = sqlite3.Row
    return c


def discover(pattern: str) -> list[Path]:
    found = sorted(Path(".").glob(pattern))
    if not found:
        gz = sorted(Path(".").glob(pattern + ".gz"))
        if gz:
            raise SystemExit(
                "no decompressed databases found, but these exist:\n  "
                + "\n  ".join(p.name for p in gz)
                + "\n\nRun:  gunzip -kf exp6_*.sqlite.gz"
            )
        raise SystemExit(f"nothing matching {pattern!r} in {Path('.').resolve()}")
    return found


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def episode_rates(conn, arm: str, opponent: str, min_turn: int) -> np.ndarray:
    """Defection rate per episode, computed from turns so turn 0 can be dropped.

    analysis/02 reads episodes.defection_count, which is fixed over the whole
    episode and therefore cannot express a turn filter. Recomputing from turns
    is the only way to exclude turn 0 consistently across arms.
    """
    rows = conn.execute(
        """SELECT episode_id,
                  SUM(agent_action='D') * 1.0 / COUNT(*) AS rate
           FROM turns
           WHERE arm=? AND opponent_policy=? AND turn >= ?
           GROUP BY episode_id ORDER BY episode_id""",
        (arm, opponent, min_turn),
    ).fetchall()
    return np.array([r["rate"] for r in rows], dtype=np.float64)


def boot_diff(a: np.ndarray, b: np.ndarray, rng, n_boot: int = N_BOOT) -> dict:
    """a - b with a percentile bootstrap over EPISODES.

    Percentile rather than normal-theory because these distributions carry a
    large point mass at zero (episodes that never defect), which makes the
    normal approximation optimistic in exactly the cells where the effect is
    smallest.
    """
    if a.size == 0 or b.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "sig": False, "n_a": int(a.size), "n_b": int(b.size)}

    diff = float(a.mean() - b.mean())
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    deltas = np.sort(da - db)
    lo, hi = float(deltas[int(0.025 * n_boot)]), float(deltas[int(0.975 * n_boot)])
    return {"diff": diff, "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0),
            "n_a": int(a.size), "n_b": int(b.size)}


# ---------------------------------------------------------------------------
# part 1 - did the arms actually lie
# ---------------------------------------------------------------------------

def falsification_rates(conn) -> dict[tuple[str, str], tuple[int, int]]:
    """(arm, opponent) -> (rows where the block contradicted the truth, rows).

    The truth is the opponent's action on the PREVIOUS turn, read from the same
    episode. Self-join rather than a stored field, so this cannot agree with
    the writer by sharing its bug.
    """
    q = """
    SELECT t.arm, t.opponent_policy,
           SUM(t.displayed_opponent_last <> p.opponent_action) AS lied,
           COUNT(*) AS n
    FROM turns t
    JOIN turns p ON p.run_id=t.run_id AND p.episode_id=t.episode_id
                AND p.arm=t.arm AND p.model_id=t.model_id
                AND p.readout_mode=t.readout_mode
                AND p.opponent_policy=t.opponent_policy AND p.turn=t.turn-1
    WHERE t.displayed_opponent_last IS NOT NULL
    GROUP BY t.arm, t.opponent_policy"""
    return {(r["arm"], r["opponent_policy"]): (r["lied"], r["n"])
            for r in conn.execute(q)}


# ---------------------------------------------------------------------------
# part 2 - gates
# ---------------------------------------------------------------------------

def cell_gates(conn) -> dict[tuple[str, str], dict]:
    q = """SELECT arm, opponent_policy,
                  AVG(cpr_score)          AS cpr,
                  AVG(action_mass_total < 0.10) AS offtask,
                  AVG(agent_action='D')   AS defect,
                  COUNT(*)                AS n
           FROM turns GROUP BY arm, opponent_policy"""
    return {(r["arm"], r["opponent_policy"]):
            {"cpr": r["cpr"], "offtask": r["offtask"],
             "defect": r["defect"], "n": r["n"]}
            for r in conn.execute(q)}


# ---------------------------------------------------------------------------
# part 5 - the echo test
# ---------------------------------------------------------------------------

def _first_int(text) -> int | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text)
    m = _INT_RE.search(str(text))
    return int(m.group()) if m else None


def _own_score_answer(blob: str):
    """Pull the OWN_SCORE reply out of the stored probe payload.

    OWN_SCORE is first in PROBE_SUITE (cdx/probe.py), so position 0 for a list.
    The payload shape is whatever json.dumps(cpr_raw[pos]) produced, which is
    not pinned by a schema - hence the defensive branches rather than an index.
    """
    try:
        data = json.loads(blob)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        for key in ("own_score", "OWN_SCORE", "0"):
            if key in data:
                return data[key]
        return next(iter(data.values()), None)
    if isinstance(data, (list, tuple)) and data:
        return data[0]
    return data


def echo_test(conn, arm: str = "3s") -> dict:
    """Did the model repeat the FALSIFIED score rather than the true one?

    True score at turn t is the cumulative agent_payoff over turns < t, summed
    here rather than read from a column so it cannot inherit a writer bug.
    displayed = donor_agent_score, which gpu_run.py sets to the falsified value
    for arm 3s.
    """
    payoffs: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for r in conn.execute(
        """SELECT episode_id, opponent_policy, turn, agent_payoff
           FROM turns WHERE arm=? ORDER BY episode_id, opponent_policy, turn""",
        (arm,),
    ):
        payoffs[(r["episode_id"], r["opponent_policy"])].append(
            (r["turn"], r["agent_payoff"]))

    true_at: dict[tuple[int, str, int], int] = {}
    for key, seq in payoffs.items():
        running = 0
        for turn, pay in seq:
            true_at[(key[0], key[1], turn)] = running
            running += pay

    q = """SELECT t.episode_id, t.opponent_policy, t.turn,
                  t.donor_agent_score AS shown, d.probe_answers AS raw
           FROM turns t
           JOIN turn_details d USING
             (run_id, episode_id, arm, model_id, readout_mode,
              opponent_policy, turn)
           WHERE t.arm=? AND d.probe_answers IS NOT NULL
             AND t.donor_agent_score IS NOT NULL"""

    n = matched_shown = matched_true = matched_neither = 0
    for r in conn.execute(q, (arm,)):
        got = _first_int(_own_score_answer(r["raw"]))
        if got is None:
            continue
        truth = true_at.get((r["episode_id"], r["opponent_policy"], r["turn"]))
        if truth is None:
            continue
        n += 1
        if got == r["shown"]:
            matched_shown += 1
        elif got == truth:
            matched_true += 1
        else:
            matched_neither += 1
    return {"n": n, "shown": matched_shown, "true": matched_true,
            "neither": matched_neither}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

CONTRASTS = [
    ("content_move   3 - 3m", "3", "3m"),
    ("content_score  3 - 3s", "3", "3s"),
    ("content_donor  3 - 3c", "3", "3c"),
    ("ATE_true       3 - 3b", "3", "3b"),
    ("perturbation  3b - 1 ", "3b", "1"),
]


def analyse(path: Path, rng, n_boot: int) -> dict:
    conn = connect(path)
    group = path.stem.replace("exp6_", "")
    opponents = sorted({r[0] for r in conn.execute(
        "SELECT DISTINCT opponent_policy FROM turns")})
    arms = {r[0] for r in conn.execute("SELECT DISTINCT arm FROM turns")}

    print(f"\n{RULE}\n{group}\n{RULE}")

    # ---- part 1 ----------------------------------------------------------
    fals = falsification_rates(conn)
    print("\n  MANIPULATION INTEGRITY  (block asserted != truth)")
    print(f"    {'arm':<5}{'opp':<6}{'lied':>9}{'rows':>9}{'rate':>9}   verdict")
    integrity_ok = True
    for arm in ("3s", "3m", "3c"):
        for opp in opponents:
            if (arm, opp) not in fals:
                continue
            lied, n = fals[(arm, opp)]
            rate = lied / n if n else float("nan")
            if arm == "3m":
                ok = lied == n
                verdict = "all rows lie" if ok else "*** SOME ROWS TOLD TRUTH"
            elif arm == "3s":
                ok = lied == 0
                verdict = "move preserved" if ok else "*** 3s TOUCHED THE MOVE"
            else:
                ok = True
                verdict = "donor coincidence" if rate > 0 else "cannot falsify here"
            integrity_ok &= ok
            print(f"    {arm:<5}{opp:<6}{lied:>9,}{n:>9,}{rate:>9.4f}   {verdict}")
    if not integrity_ok:
        print("\n    INTEGRITY FAILURE - the contrasts below are not interpretable.")

    # ---- part 2 ----------------------------------------------------------
    gates = cell_gates(conn)
    print(f"\n  GATES  (CPR >= {CPR_GATE} on arm 3 only; off-task < {OFFTASK_GATE} on all)")
    print(f"    {'arm':<5}{'opp':<6}{'defect':>9}{'CPR':>8}{'off':>8}   note")
    gate_fail = False
    for arm in sorted(arms, key=lambda a: (len(a), a)):
        for opp in opponents:
            g = gates.get((arm, opp))
            if not g:
                continue
            bad = []
            if g["offtask"] > OFFTASK_GATE:
                bad.append("*** OFF-TASK")
            if arm in CPR_GATED_ARMS and g["cpr"] < CPR_GATE:
                bad.append("*** CPR BELOW GATE")
            if bad:
                gate_fail = True
                note = "; ".join(bad)
            elif arm in NO_TRUE_STATE:
                note = "no true state to report; CPR is a floor"
            elif arm in LYING_ARMS:
                note = f"belief: {1 - g['cpr']:.1%} took the block's word"
            else:
                note = "ok"
            print(f"    {arm:<5}{opp:<6}{g['defect']:>9.4f}{g['cpr']:>8.3f}"
                  f"{g['offtask']:>8.3f}   {note}")

    for opp in opponents:
        a, b = gates.get(("3", opp)), gates.get(("3b", opp))
        if a and b:
            print(f"    manipulation check  CPR(3)-CPR(3b) vs {opp:<5}"
                  f"{a['cpr']:.3f} - {b['cpr']:.3f} = {a['cpr']-b['cpr']:+.3f}")
    print("\n    Only arm 3 displays the true state, so only arm 3 can fail a CPR")
    print("    gate. Arms 1 and 3b have nothing true to report and arms 3c/3s/3m")
    print("    are shown a falsehood - low CPR there means the model BELIEVED the")
    print("    block, which is the manipulation working, not the instrument failing.")

    # ---- part 3 ----------------------------------------------------------
    print(f"\n  CONTRASTS  (episode-level, {n_boot:,} bootstrap resamples)")
    print("    'incl t0' keeps turn 0, matching the run's printed table.")
    print("    'excl t0' drops it from BOTH arms - the quotable estimate.")
    print(f"\n    {'contrast':<24}{'opp':<6}{'incl t0':>10}{'excl t0':>10}"
          f"{'95% CI (excl t0)':>22}{'':>4}")
    out: dict[str, dict] = {}
    for label, x, y in CONTRASTS:
        if x not in arms or y not in arms:
            continue
        for opp in opponents:
            incl = boot_diff(episode_rates(conn, x, opp, 0),
                             episode_rates(conn, y, opp, 0), rng, n_boot)
            excl = boot_diff(episode_rates(conn, x, opp, 1),
                             episode_rates(conn, y, opp, 1), rng, n_boot)
            star = " *" if excl["sig"] else "  "
            print(f"    {label:<24}{opp:<6}{incl['diff']:>+10.4f}"
                  f"{excl['diff']:>+10.4f}"
                  f"   [{excl['lo']:>+7.4f},{excl['hi']:>+7.4f}]{star}")
            out[f"{label.split()[0]}|{opp}"] = {"incl_t0": incl, "excl_t0": excl}
    print("\n    * = bootstrap CI excludes zero.")

    # ---- part 4 ----------------------------------------------------------
    print("\n  DOSE-RESPONSE  (how often the arm lied vs how much it moved)")
    print(f"    {'arm':<5}{'opp':<6}{'falsified':>11}{'effect':>10}"
          f"{'per lied row':>14}")
    for arm in ("3m", "3c"):
        for opp in opponents:
            if (arm, opp) not in fals:
                continue
            lied, n = fals[(arm, opp)]
            rate = lied / n if n else 0.0
            key = ("content_move" if arm == "3m" else "content_donor") + f"|{opp}"
            if key not in out:
                continue
            eff = out[key]["excl_t0"]["diff"]
            per = eff / rate if rate else float("nan")
            per_s = f"{per:>+14.4f}" if per == per else f"{'undefined':>14}"
            print(f"    {arm:<5}{opp:<6}{rate:>11.4f}{eff:>+10.4f}{per_s}")
    print("\n    'per lied row' rescales each effect by its own manipulation")
    print("    rate. If the last-move field is what matters, 3c and 3m should")
    print("    agree in this column even though their raw effects differ.")
    print("    Where the rate is 0 the arm could not falsify anything, and an")
    print("    effect near zero there is a positive control, not a null result.")

    # ---- part 5 ----------------------------------------------------------
    if "3s" in arms:
        e = echo_test(conn)
        print("\n  ECHO TEST  arm 3s - which score did the model report back?")
        if e["n"] == 0:
            print("    no scoreable probe answers stored for arm 3s.")
        else:
            print(f"    probes scored          {e['n']:,}")
            print(f"    matched DISPLAYED lie  {e['shown']:,}"
                  f"  ({e['shown']/e['n']:.1%})   <- read the block")
            print(f"    matched TRUE score     {e['true']:,}"
                  f"  ({e['true']/e['n']:.1%})   <- ignored the block")
            print(f"    matched neither        {e['neither']:,}"
                  f"  ({e['neither']/e['n']:.1%})")
            print("\n    Read alongside content_score above. A high DISPLAYED")
            print("    share with a content_score CI covering zero means the")
            print("    model read the false state, repeated it, and decided as")
            print("    if it had not - comprehension without use.")
        out["echo_3s"] = e

    conn.close()
    return {"group": group, "integrity_ok": bool(integrity_ok),
            "gate_fail": bool(gate_fail),
            "falsification": {f"{a}|{o}": {"lied": l, "n": n}
                              for (a, o), (l, n) in fals.items()},
            "gates": {f"{a}|{o}": g for (a, o), g in gates.items()},
            "contrasts": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="single database; default is every exp6_*.sqlite")
    ap.add_argument("--glob", default="exp6_*.sqlite")
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--out", default="EXP6_FIELDS.json")
    args = ap.parse_args()

    paths = [Path(args.db)] if args.db else discover(args.glob)
    rng = np.random.default_rng(SEED)

    print(f"\n{RULE}\nEXP6 - WHICH FIELD CARRIES THE EFFECT\n{RULE}")
    print(f"  databases   {len(paths)}")
    print(f"  bootstrap   {args.bootstrap:,} resamples, seed {SEED}")
    print("  unit        EPISODE")
    print("  turn 0      excluded from the quotable column, both arms")

    payload = [analyse(p, rng, args.bootstrap) for p in paths]

    print(f"\n{RULE}\nSUMMARY\n{RULE}")
    print(f"  {'group':<28}{'opp':<6}{'3-3s (score)':>16}{'3-3m (move)':>16}")
    for rec in payload:
        for opp in ("allc", "tft"):
            s = rec["contrasts"].get(f"content_score|{opp}")
            m = rec["contrasts"].get(f"content_move|{opp}")
            if not (s and m):
                continue
            print(f"  {rec['group']:<28}{opp:<6}"
                  f"{s['excl_t0']['diff']:>+13.4f}{'*' if s['excl_t0']['sig'] else ' '}  "
                  f"{m['excl_t0']['diff']:>+13.4f}{'*' if m['excl_t0']['sig'] else ' '}")
    print("\n  Prediction: the score column is null, the move column is not.")

    bad = [r["group"] for r in payload if not r["integrity_ok"]]
    if bad:
        print(f"\n  INTEGRITY FAILURES: {', '.join(bad)}")

    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  written  {Path(args.out).resolve()}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())