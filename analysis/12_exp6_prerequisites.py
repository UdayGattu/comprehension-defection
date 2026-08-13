#!/usr/bin/env python3
"""Two gates that decide whether a new opponent experiment is worth running.

Both use data already on disk. Both can kill the proposed experiment, and (B)
can kill it for a reason no choice of opponent can fix. Run before writing a
line of experiment code.

------------------------------------------------------------------------------
GATE A - IS ARM 3c ALREADY DEAD?

  Arm 3c renders arm 3's template from another episode's state, so the falsified
  score carries an error d = donor_score - true_score.

  If donors are drawn from the same distribution at the same round in the same
  cell - which is what `select_donor` does, it samples from `live_states` and
  `run_cell` is invoked per (arm, opponent) - then E[d] = 0 BY SYMMETRY. And if
  the behavioural response to d is linear and symmetric, the mean contrast

      content = P(defect | 3) - P(defect | 3c)

  is zero in expectation EVEN FOR A MODEL THAT FOLLOWS THE LIE PERFECTLY. Half
  the donors push it one way, half the other, and they cancel.

  The observed qwen-vs-TFT content effect (-0.2375, replicated) proves the
  response is NOT linear there, so cancellation is not total. But before any new
  design leans on arm 3c, three numbers are needed:

      E[d]         near zero means the mean contrast is structurally weak
      sd(d)        near zero means arm 3c is numerically almost identical to
                   arm 3 and has no power AT ALL, whatever the opponent
      P(|d|>=15)   the share of rows carrying a falsification large enough to
                   move a decision, given the measured slope of ~0.01 defection
                   rate per point of score error

  A design that needs a signed falsification can be built - split 3c by the sign
  of d - but only if sd(d) is large enough to populate both halves.

------------------------------------------------------------------------------
GATE B - DOES ANY MODEL PLAN?

  Every proposed opponent - threshold-defector, lead-guard, and any randomised
  variant - rests on the same assumption: that the model's defection rate rises
  toward the end of the game, because the horizon is finite and defecting late
  is cheap. That end-game switch point is the thing a false score is supposed to
  move.

  If P(defect | round) is FLAT in arm 3, there is no switch point, nothing for a
  falsified score to shift, and no opponent design creates one. The experiment
  would be measuring a mechanism the models do not have.

  This gate is checked in arm 3 - the true-state arm - because that is the
  condition most favourable to planning. If it is flat there it is flat
  everywhere.

  Reported per (model, framing, readout, opponent):

      slope        OLS slope of defect rate on round index. Positive means
                   defection rises toward the horizon.
      last-first   P(defect) over the final quarter minus the first quarter.
                   Robust to non-linearity in a way the slope is not.

  Cells over the off-task gate are marked; their turn profile is prose, not
  decisions.

    python analysis/12_exp6_prerequisites.py
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
import statistics as st
from collections import defaultdict
from pathlib import Path
from urllib.request import pathname2url

OFF_TASK_GATE = 0.10
BIG_LIE = 15          # |d| at which the measured slope predicts a ~15pp shift
RULE = "=" * 78


def ro_uri(p: Path) -> str:
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def cols(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def gate_a(path: Path, w):
    """Distribution of the falsification error d in arm 3c."""
    con = sqlite3.connect(ro_uri(path), uri=True)
    if "donor_agent_score" not in cols(con, "turns"):
        con.close()
        return None
    rows = con.execute("""
        SELECT opponent_policy, episode_id, turn, agent_payoff,
               donor_agent_score, donor_degenerate
        FROM turns WHERE arm='3c'
        ORDER BY opponent_policy, episode_id, turn""").fetchall()
    con.close()
    if not rows:
        return None

    by_ep = defaultdict(list)
    for r in rows:
        by_ep[(r[0], r[1])].append(r)

    ds = defaultdict(list)
    turn0_bad = 0
    for (opp, ep), turns in by_ep.items():
        running = 0
        for _, _, turn, payoff, donor, degen in turns:
            if turn == 0 and running != 0:
                turn0_bad += 1
            if donor is not None and not degen:
                ds[opp].append(donor - running)
            running += (payoff or 0)

    if turn0_bad:
        w(f"  **true-score reconstruction FAILED** on {turn0_bad} episodes — "
          "skipped\n")
        return None

    out = []
    for opp in sorted(ds):
        v = ds[opp]
        if not v:
            continue
        big = sum(1 for x in v if abs(x) >= BIG_LIE) / len(v)
        neg = sum(1 for x in v if x <= -BIG_LIE) / len(v)
        pos = sum(1 for x in v if x >= BIG_LIE) / len(v)
        out.append((opp, len(v), st.mean(v), st.pstdev(v),
                    min(v), max(v), big, neg, pos))
    return out


def gate_b(path: Path):
    """Is defection rate rising toward the horizon, in arm 3?"""
    con = sqlite3.connect(ro_uri(path), uri=True)
    tc = cols(con, "turns")
    has_mass = "action_mass_total" in tc
    rows = con.execute(f"""
        SELECT opponent_policy, turn,
               AVG(CASE WHEN agent_action='D' THEN 1.0 ELSE 0 END),
               COUNT(*),
               {"AVG(CASE WHEN action_mass_total<? THEN 1.0 ELSE 0 END)"
                if has_mass else "0"}
        FROM turns WHERE arm='3' GROUP BY opponent_policy, turn
        ORDER BY opponent_policy, turn""",
        (OFF_TASK_GATE,) if has_mass else ()).fetchall()
    con.close()
    if not rows:
        return None

    by_opp = defaultdict(list)
    for opp, turn, p, n, off in rows:
        by_opp[opp].append((turn, p, n, off))

    out = []
    for opp, series in sorted(by_opp.items()):
        if len(series) < 4:
            continue
        ts = [s[0] for s in series]
        ps = [s[1] for s in series]
        off = max(s[3] for s in series)
        mt, mp = st.mean(ts), st.mean(ps)
        den = sum((t - mt) ** 2 for t in ts)
        slope = (sum((t - mt) * (p - mp) for t, p in zip(ts, ps)) / den
                 if den else 0.0)
        q = max(len(series) // 4, 1)
        first = st.mean(ps[:q])
        last = st.mean(ps[-q:])
        out.append((opp, slope, first, last, last - first, off, ps))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="EXP6_PREREQUISITES.md")
    args = ap.parse_args()

    dbs = sorted(Path(p) for p in glob.glob("*.sqlite")
                 if not Path(p).name.startswith(("smoke_", "cotsmoke_")))
    L: list[str] = []

    def w(s=""):
        L.append(s)

    w("# Prerequisites for a new-opponent experiment\n")
    w("Two gates, both from data already on disk. Either can stop the "
      "experiment; **B can stop it for a reason no opponent design fixes**.\n")

    # ---------------- GATE A -------------------------------------------
    w(f"\n{RULE}\n## Gate A — is arm 3c already dead?\n{RULE}\n")
    w("`d = donor_score − true_score` is the falsification each arm-3c row "
      "carries. Degenerate-donor rows are excluded — there the donor *is* the "
      "recipient's state and `d = 0` by construction.\n")
    w("`select_donor` samples from `live_states`, and `run_cell` runs per "
      "(arm, opponent), so donors share the recipient's opponent and round. "
      "**If the sampling is symmetric, `E[d] = 0`** and a linear response gives "
      "a zero mean contrast even for a model that follows the lie perfectly.\n")
    w(f"At the measured slope of ≈0.01 defection per point of error, "
      f"`|d| ≥ {BIG_LIE}` is the range that moves a decision by ≈15pp.\n")
    w("```")
    w(f"{'database':26}{'opp':6}{'n':>9}{'E[d]':>9}{'sd(d)':>9}"
      f"{'min':>7}{'max':>7}{'P|d|>=15':>10}{'P d<=-15':>10}{'P d>=+15':>10}")
    any_a = False
    for path in dbs:
        res = gate_a(path, w)
        if not res:
            continue
        any_a = True
        for opp, n, m, sd, lo, hi, big, neg, pos in res:
            w(f"{path.stem:26}{opp:6}{n:>9,}{m:>9.2f}{sd:>9.2f}"
              f"{lo:>7.0f}{hi:>7.0f}{big:>10.3f}{neg:>10.3f}{pos:>10.3f}")
    w("```")
    if not any_a:
        w("\n_No database carries `donor_agent_score` — arm 3c exists only in "
          "exp2 and exp3._\n")
    w("\n**Reading it.** `sd(d)` near zero means arm 3c is numerically almost "
      "arm 3 and has no power whatever the opponent — that alone would end the "
      "design. `E[d]` near zero with large `sd(d)` means the *mean* contrast is "
      "weak but a **signed** split (3c− vs 3c+) recovers the power, provided "
      "`P(d ≤ −15)` and `P(d ≥ +15)` are both non-trivial. Compare `E[d]` "
      "against `sd(d)/sqrt(n)` before calling it zero.\n")

    # ---------------- GATE B -------------------------------------------
    w(f"\n{RULE}\n## Gate B — does any model plan toward the horizon?\n{RULE}\n")
    w("Measured in **arm 3**, the true-state arm — the condition most "
      "favourable to planning. If defection is flat in the round index here, "
      "there is no end-game switch point, nothing for a falsified score to "
      "move, and **no opponent design creates one**.\n")
    w("`slope` is OLS of P(defect) on round. `last−first` compares the final "
      "quarter of the game against the first, which survives non-linearity.\n")
    w("```")
    w(f"{'database':30}{'opp':6}{'slope':>10}{'first q':>9}{'last q':>9}"
      f"{'last-first':>12}{'off':>7}")
    for path in dbs:
        res = gate_b(path)
        if not res:
            continue
        for opp, slope, first, last, diff, off, _ in res:
            flag = "  VOID" if off > OFF_TASK_GATE else ""
            w(f"{path.stem:30}{opp:6}{slope:>+10.5f}{first:>9.3f}{last:>9.3f}"
              f"{diff:>+12.3f}{off:>7.3f}{flag}")
    w("```")
    w("\n**Reading it.** A positive slope with `last−first` well above zero "
      "means the model shortens its horizon as the game ends — the mechanism "
      "the experiment needs. Flat or negative in every valid cell means the "
      "models never engage it, and the honest response is to report that as a "
      "capability finding rather than to build a fourth opponent around a "
      "mechanism that is not there.\n")
    w("\nA caveat worth stating in the writeup: a rising profile is also "
      "consistent with defection simply being sticky — one defection provokes "
      "TFT, which provokes more. Against ALLC there is no such feedback, so "
      "**the ALLC column is the cleaner evidence of end-game planning**; if "
      "the rise appears only against TFT it is more likely retaliation "
      "spirals than horizon reasoning.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())