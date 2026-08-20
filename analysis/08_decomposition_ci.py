#!/usr/bin/env python3
"""Bootstrap intervals for the veracity/schema split and for the interaction.

WHY THIS EXISTS

  1. ATE_true is not one thing.

     ATE_true = P(D|3) - P(D|3b)
              = [P(D|3) - P(D|3c)]  +  [P(D|3c) - P(D|3b)]
                 veracity effect        schema effect

     Arm 3 and arm 3c share a template and differ only in whether the numbers
     are true. Block-reading is ~1.00 in both (analysis/04_donor_echo.py), so
     the veracity contrast holds ATTENTION constant and varies only TRUTH.
     Arm 3c and arm 3b differ in whether number-shaped fields exist at all.

     NOTE ON THE NAME. This term was called "content" in earlier revisions of
     this script and of DECOMPOSITION.md. It is "veracity" throughout the paper
     and here, because "content" already names the 3-vs-3b contrast in the
     paper's tables and the collision cost a reviewer's afternoon. The JSON
     sidecar emits `veracity`; nothing emits `content` any more.

  2. The pre-registered prediction is an INTERACTION, not two main effects.

     It says defection moves DOWN vs TFT and UP vs ALLC. That is

        interaction = ATE_true(allc) - ATE_true(tft)

     Scoring each opponent separately lets a uniformly-signed shift score half
     a win: exp5 has all six components positive, which is a main effect of
     defection, and mistral's interaction is -0.0066 - the wrong sign.

THE TURN FILTER, AND WHY THE DEFAULT CHANGED

  Until this revision episode_rates() ran over every turn, including turn 0.
  That is wrong for two of the arms this script consumes, and the paper's
  Section 4 preamble says so: turn 0 is dropped from every arm in which turn 0
  renders byte-identically to the true-state arm.

    arm 3c   at turn 0 every episode has score 0 and no last move, so the
             donor block equals the true block. The database records this as
             donor_degenerate = 1 on every turn-0 row.
    arm 3m   at turn 0 there is no last move to flip.

  Arms 1, 3, 3b and 3s are NOT degenerate at turn 0: arm 3s displays the true
  score plus 15, which at turn 0 is 15 against a true 0.

  Four filters are implemented and all four are reproducible:

    donor_matched  (DEFAULT) drop the (episode_id, turn) coordinates at which
                arm 3c's donor was degenerate - FROM EVERY ARM IN THE CONTRAST,
                not only from 3c. Two properties earn it the default. It
                removes exactly the rows on which the donor arm is not a donor
                arm, which is every turn 0 by construction plus the later turns
                where the sampled donor happened to be identical. And because
                the same coordinates come out of all three arms, the three run
                on identical turn compositions, which matters because turn
                index is correlated with defection. This is the convention the
                paper's appendix table reports and describes as "fully
                degenerate turns dropped".
    excl_t0     drop turn 0 from EVERY arm. Also symmetric, and it is the grid
                - 1/19,000 for a 1,000-episode run - the exp6-exp8 contrasts
                use. It differs from donor_matched only in the later
                coincidental degenerate rows, which are about 5% of an exp3
                cell and are enough to reverse a sign in one of them.
    degenerate  drop turn 0 only from arms in DEGENERATE_AT_TURN0. Matches the
                Section 4 sentence literally, but leaves the two sides of a
                contrast on different turn compositions. Reported for
                comparison, not quoted.
    all         no filter. The pre-revision behaviour, kept so the figures the
                released DECOMPOSITION.md printed before this change remain
                reproducible rather than merely asserted.

  A database with no donor_degenerate column (exp2) cannot express
  donor_matched, and falls back to excl_t0 with a note in the sidecar.

METHOD
  Matches analysis/02_episode_level.py: the EPISODE is the unit, because
  episodes are independently seeded and turns within an episode are not
  independent. Arms and opponents are resampled independently of each other -
  they are separate runs, not paired observations, so there is nothing to pair.

  Seed is fixed at the same value 02_episode_level.py uses, so a contrast that
  appears in both files must agree. A disagreement is a finding, not a rounding
  difference.

  p is two-sided from the bootstrap distribution: 2 x min(P(b<=0), P(b>=0)),
  floored at 1/n_boot since a bootstrap cannot resolve below its own resolution.

  Only exp2, exp3 and exp6 carry arm 3c. exp4 and exp5 dropped it, so they can
  only contribute the interaction, not the decomposition.

OUTPUTS
  DECOMPOSITION.md    human-readable, unchanged in shape
  DECOMPOSITION.json  every number in the markdown, unrounded, plus the dose
                      column the paper's appendix table prints beside them.
                      The markdown is rendered FROM this structure, so the two
                      cannot drift.

    python analysis/08_decomposition_ci.py
    python analysis/08_decomposition_ci.py --turn-filter all   # old behaviour
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import pathname2url

SEED = 20260811          # same as 02_episode_level.py
N_BOOT = 10000

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

#: Arms whose turn-0 render is byte-identical to the true-state arm's.
DEGENERATE_AT_TURN0 = frozenset({"3m", "3c"})

#: Falsification magnitude arm 3s applies, and therefore the dose threshold.
DOSE_THRESHOLD = 15

TURN_FILTERS = ("donor_matched", "excl_t0", "degenerate", "all")

LOG = logging.getLogger("decomposition")


def ro_uri(p: Path) -> str:
    """mode=ro alone fails on WAL databases; immutable=1 promises sole reader."""
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def _turn_predicate(turn_filter: str, arm: str, *, has_degen: bool = True) -> str:
    """SQL fragment restricting turns for one arm under one filter."""
    if turn_filter == "all":
        return ""
    if turn_filter == "excl_t0":
        return " AND turn > 0"
    if turn_filter == "degenerate":
        return " AND turn > 0" if arm in DEGENERATE_AT_TURN0 else ""
    if turn_filter == "donor_matched":
        # Without the column the degenerate coordinates are unknowable, and
        # every turn 0 is degenerate by construction, so excl_t0 is the closest
        # expressible filter. Recorded per group in the sidecar.
        # Applied row-wise in episode_rates(): as SQL this is a correlated
        # NOT EXISTS per outer row and SQLite takes minutes on an exp3 table.
        return "" if has_degen else " AND turn > 0"
    raise ValueError(f"unknown turn filter: {turn_filter!r}")


def _has_degen(con: sqlite3.Connection) -> bool:
    return "donor_degenerate" in {r[1] for r in con.execute("PRAGMA table_info(turns)")}


def degenerate_coords(con: sqlite3.Connection) -> set[tuple[str, int, int]]:
    """(opponent, episode, turn) at which arm 3c's donor was not distinct.

    Every turn 0 by construction, plus the later turns on which the uniform
    draw over live episodes happened to land on an identical state.
    """
    if not _has_degen(con):
        return set()
    return set(con.execute(
        "SELECT opponent_policy, episode_id, turn FROM turns "
        "WHERE arm = '3c' AND COALESCE(donor_degenerate, 0) = 1"))


def episode_rates(con: sqlite3.Connection,
                  turn_filter: str) -> dict[tuple[str, str], list[float]]:
    """Per-episode defection rate, keyed by (arm, opponent).

    One query per arm rather than one for all of them, because the `degenerate`
    filter restricts different arms to different turn ranges and a single
    GROUP BY cannot express that.
    """
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    hd = _has_degen(con)
    arms = [a for (a,) in con.execute("SELECT DISTINCT arm FROM turns")]
    drop = (degenerate_coords(con)
            if turn_filter == "donor_matched" and hd else set())

    for arm in arms:
        pred = _turn_predicate(turn_filter, arm, has_degen=hd)
        if drop:
            acc: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
            for opp, ep, turn, act in con.execute(
                    "SELECT opponent_policy, episode_id, turn, agent_action "
                    "FROM turns WHERE arm = ?" + pred, (arm,)):
                if (opp, ep, turn) in drop:
                    continue
                a = acc[(opp, ep)]
                a[0] += (act == "D")
                a[1] += 1
            for (opp, _ep), (d, n) in acc.items():
                if n:
                    out[(arm, opp)].append(d / n)
            continue
        sql = ("SELECT opponent_policy, "
               "SUM(CASE WHEN agent_action='D' THEN 1 ELSE 0 END), COUNT(*) "
               "FROM turns WHERE arm = ?" + pred +
               " GROUP BY opponent_policy, episode_id")
        for opp, d, n in con.execute(sql, (arm,)):
            if n:
                out[(arm, opp)].append(d / n)
    return out


def off_task(con: sqlite3.Connection,
             turn_filter: str) -> dict[tuple[str, str], float]:
    """Off-task fraction per (arm, opponent), on the same turns as the rates."""
    out: dict[tuple[str, str], float] = {}
    try:
        hd = _has_degen(con)
        arms = [a for (a,) in con.execute("SELECT DISTINCT arm FROM turns")]
        drop = (degenerate_coords(con)
                if turn_filter == "donor_matched" and hd else set())
        for arm in arms:
            pred = _turn_predicate(turn_filter, arm, has_degen=hd)
            if drop:
                tot: dict[str, list[int]] = defaultdict(lambda: [0, 0])
                for opp, ep, turn, mass in con.execute(
                        "SELECT opponent_policy, episode_id, turn, "
                        "action_mass_total FROM turns WHERE arm = ?" + pred,
                        (arm,)):
                    if (opp, ep, turn) in drop:
                        continue
                    t = tot[opp]
                    t[0] += (mass is not None and mass < OFF_TASK_GATE)
                    t[1] += 1
                for opp, (k, n) in tot.items():
                    out[(arm, opp)] = (k / n) if n else 0.0
                continue
            sql = ("SELECT opponent_policy, "
                   "AVG(CASE WHEN action_mass_total < ? THEN 1.0 ELSE 0 END) "
                   "FROM turns WHERE arm = ?" + pred +
                   " GROUP BY opponent_policy")
            for opp, v in con.execute(sql, (OFF_TASK_GATE, arm)):
                out[(arm, opp)] = v if v is not None else 0.0
    except sqlite3.Error as exc:                          # pragma: no cover
        LOG.warning("off-task unavailable (%s); gate not applied", exc)
        return {}
    return out


def donor_dose(con: sqlite3.Connection,
               turn_filter: str) -> dict[str, float | None]:
    """P(|donor_score - true_score| >= DOSE_THRESHOLD) on non-degenerate 3c rows.

    Returns None for an opponent whose database predates the donor_agent_score
    column (exp2), where the quantity is not recoverable rather than zero.
    """
    cols = {r[1] for r in con.execute("PRAGMA table_info(turns)")}
    if "donor_agent_score" not in cols:
        return {}
    rows = con.execute(
        "SELECT opponent_policy, episode_id, turn, agent_payoff, "
        "       donor_agent_score, donor_degenerate "
        "FROM turns WHERE arm='3c'"
        " ORDER BY opponent_policy, episode_id, turn").fetchall()
    if not rows:
        return {}
    keep_turn0 = (turn_filter == "all")

    by_ep: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for r in rows:
        by_ep[(r[0], r[1])].append(r)

    hits: dict[str, list[int]] = defaultdict(list)
    for (opp, _ep), turns in by_ep.items():
        running = 0
        for _, _, turn, payoff, donor, degen in turns:
            eligible = (keep_turn0 or turn > 0) and donor is not None and not degen
            if eligible:
                hits[opp].append(1 if abs(donor - running) >= DOSE_THRESHOLD else 0)
            running += (payoff or 0)
    return {opp: (sum(v) / len(v) if v else None) for opp, v in hits.items()}


# Bootstrapping is the whole cost of this script, and the naive form is a
# pure-Python loop doing n_boot x n randrange() calls per cell - about 40s for
# a single exp3 contrast, and roughly 80 minutes over the full corpus. numpy
# performs the identical resampling as an array gather and takes it under a
# minute. The chunking keeps the index matrix bounded: 2000 x 2000 int64 is
# 32MB, where the unchunked 10000 x 2000 would be 160MB per cell.
#
# The numpy and pure-Python paths draw from different RNG streams, so their
# resamples differ. That is expected and harmless - a bootstrap interval is not
# a function of which valid stream produced it - but it does mean numbers will
# not be bit-identical between the two paths. Both are seeded, so each is
# reproducible on its own.
try:
    import numpy as _np
except ImportError:                                     # pragma: no cover
    _np = None


def _boot_means_np(arr, n_boot: int, rng, chunk: int = 2000):
    n = len(arr)
    out = _np.empty(n_boot)
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        out[done:done + k] = arr[idx].mean(axis=1)
        done += k
    return out


def boot_diff(a: Sequence[float], b: Sequence[float], rng) -> list[float]:
    """Bootstrap distribution of mean(a) - mean(b), resampling each independently.

    Independently, not paired: the arms are separate runs, so there is no
    correspondence between episode i of arm 3 and episode i of arm 3b to
    preserve.
    """
    if _np is not None and not isinstance(rng, random.Random):
        A = _np.asarray(a, dtype=_np.float64)
        B = _np.asarray(b, dtype=_np.float64)
        return (_boot_means_np(A, N_BOOT, rng)
                - _boot_means_np(B, N_BOOT, rng)).tolist()
    na, nb = len(a), len(b)
    out = []
    for _ in range(N_BOOT):
        sa = sum(a[rng.randrange(na)] for _ in range(na)) / na
        sb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        out.append(sa - sb)
    return out


def make_rng(seed: int):
    """numpy Generator when available, stdlib Random otherwise."""
    return _np.random.default_rng(seed) if _np is not None else random.Random(seed)


def summarise(dist: Sequence[float]) -> tuple[float, float, float, float]:
    """point (mean of the distribution), lo, hi, p."""
    s = sorted(dist)
    n = len(s)
    lo, hi = s[int(0.025 * n)], s[int(0.975 * n) - 1]
    # Resamples landing exactly on zero count in BOTH tails, so below + above
    # can exceed 1 and the doubled minimum can exceed 1 with it. Floor at the
    # bootstrap's own resolution (1/n - it cannot resolve finer) and cap at 1.
    below = sum(1 for v in s if v <= 0) / n
    above = sum(1 for v in s if v >= 0) / n
    p = min(1.0, max(2 * min(below, above), 1.0 / n))
    return sum(s) / n, lo, hi, p


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "** " if p < 1e-2 else "*  " if p < 0.05 else "   "


def _interval(dist: Sequence[float], n_a: int, n_b: int) -> dict[str, Any]:
    est, lo, hi, p = summarise(dist)
    return {"diff": est, "lo": lo, "hi": hi, "p": p,
            "sig": (lo > 0) or (hi < 0), "n_a": n_a, "n_b": n_b}


def compute(dbs: Iterable[Path], turn_filter: str) -> dict[str, Any]:
    """Every number the report prints, unrounded, keyed by group and cell."""
    decomposition: dict[str, dict[str, Any]] = {}
    interaction: dict[str, dict[str, Any]] = {}

    for path in dbs:
        con = sqlite3.connect(ro_uri(path), uri=True)
        try:
            has_degen = _has_degen(con)
            rates = episode_rates(con, turn_filter)
            offs = off_task(con, turn_filter)
            dose = donor_dose(con, turn_filter)
        finally:
            con.close()
        group = path.stem

        for opp in ("allc", "tft"):
            if not all((a, opp) in rates for a in ("3", "3c", "3b")):
                continue
            gate = max(offs.get((a, opp), 0.0) for a in ("3", "3c", "3b"))
            if gate > OFF_TASK_GATE:
                LOG.info("%s|%s excluded by off-task gate (%.4f)",
                         group, opp, gate)
                continue
            rng = make_rng(SEED)
            cell: dict[str, Any] = {"off_task_max": gate,
                                    "dose": dose.get(opp),
                                    "turn_filter_applied": (
                                        turn_filter if has_degen or
                                        turn_filter != "donor_matched"
                                        else "excl_t0 (no donor_degenerate column)")}
            for label, x, y in (("veracity", "3", "3c"),
                                ("schema", "3c", "3b"),
                                ("ate_true", "3", "3b")):
                cell[label] = _interval(
                    boot_diff(rates[(x, opp)], rates[(y, opp)], rng),
                    len(rates[(x, opp)]), len(rates[(y, opp)]))
            decomposition[f"{group}|{opp}"] = cell

        if all((a, o) in rates for a in ("3", "3b") for o in ("allc", "tft")):
            # SCOPE OF THIS GATE: the arms this contrast actually consumes,
            # which for ATE_true is {3, 3b}. Arm 1 is deliberately NOT gated
            # here because no number in this table depends on it. That makes
            # VOID here a NARROWER test than the group-level exclusion in
            # EXPERIMENTS.md, which maxes over every arm including arm 1.
            # The two can therefore disagree, correctly:
            # exp4_qwen_abs_scratchpad is 0.0932 over {3,3b} and 0.2006 on
            # arm 1, so it is not VOID here and IS excluded there. Any
            # three-arm quantity - the presence effect P(D|3b) - P(D|1), and
            # hence the paper's decomposition - must use the wider rule.
            void = max(offs.get((a, o), 0.0)
                       for a in ("3", "3b")
                       for o in ("allc", "tft")) > OFF_TASK_GATE
            rng = make_rng(SEED)
            a3a, a3ba = rates[("3", "allc")], rates[("3b", "allc")]
            a3t, a3bt = rates[("3", "tft")], rates[("3b", "tft")]
            if _np is not None:
                # Four independent resamples per iteration: the two arms and
                # the two opponents are all separate runs.
                ms = [_boot_means_np(_np.asarray(v, dtype=_np.float64),
                                     N_BOOT, rng)
                      for v in (a3a, a3ba, a3t, a3bt)]
                dist = ((ms[0] - ms[1]) - (ms[2] - ms[3])).tolist()
            else:
                dist = []
                for _ in range(N_BOOT):
                    def m(v):
                        n = len(v)
                        return sum(v[rng.randrange(n)] for _ in range(n)) / n
                    dist.append((m(a3a) - m(a3ba)) - (m(a3t) - m(a3bt)))
            rec = _interval(dist, len(a3a), len(a3t))
            rec["ate_allc"] = sum(a3a) / len(a3a) - sum(a3ba) / len(a3ba)
            rec["ate_tft"] = sum(a3t) / len(a3t) - sum(a3bt) / len(a3bt)
            rec["void"] = void
            interaction[group] = rec

    return {"decomposition": decomposition, "interaction": interaction}


def aggregate(decomposition: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mean |veracity| and mean |schema|, over all cells and by opponent."""
    def stat(keys: Sequence[str]) -> dict[str, Any]:
        if not keys:
            return {"n_cells": 0}
        v = [abs(decomposition[k]["veracity"]["diff"]) for k in keys]
        s = [abs(decomposition[k]["schema"]["diff"]) for k in keys]
        mv, ms = sum(v) / len(v), sum(s) / len(s)
        return {"n_cells": len(keys), "mean_abs_veracity": mv,
                "mean_abs_schema": ms,
                "ratio": (ms / mv) if mv else None}

    allk = sorted(decomposition)
    return {"all": stat(allk),
            "allc": stat([k for k in allk if k.endswith("|allc")]),
            "tft": stat([k for k in allk if k.endswith("|tft")])}


def render(payload: dict[str, Any]) -> str:
    """DECOMPOSITION.md, built from the same structure the JSON carries."""
    dec = payload["decomposition"]
    inter = payload["interaction"]
    agg = payload["aggregate"]
    n_boot = payload["_n_boot"]
    L: list[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    w("# Veracity vs schema, and the interaction\n")
    w(f"Episode-level bootstrap, {n_boot:,} resamples, seed {payload['_seed']}. "
      "The episode is the unit; arms and opponents are resampled independently "
      "because they are separate runs.\n")
    w(f"Turn filter: `{payload['_turn_filter']}`. "
      f"{payload['_turn_filter_note']}\n")
    w("```\nveracity    = P(D|3)  - P(D|3c)   same template, TRUE vs FALSE numbers\n"
      "schema      = P(D|3c) - P(D|3b)   number-shaped fields present at all\n"
      "ATE_true    = P(D|3)  - P(D|3b)   = veracity + schema\n"
      "interaction = ATE_true(allc) - ATE_true(tft)\n```\n")
    w("Block-reading is ~1.00 in both arm 3 and arm 3c, so the veracity contrast "
      "holds attention constant and varies only truth.\n")

    w(f"\n{RULE}\n## 1. Decomposition (needs arm 3c: exp2, exp3 and exp6)\n{RULE}\n")
    w("```")
    w(f"{'database':26}{'opp':5}{'contrast':10}{'est':>9}{'95% CI':>20}{'p':>9}"
      f"{'dose':>10}  ")
    w("-" * 94)
    for key in sorted(dec):
        group, opp = key.rsplit("|", 1)
        cell = dec[key]
        dose = cell["dose"]
        dose_s = "n/r" if dose is None else f"{dose:.5f}"
        for label in ("veracity", "schema", "ate_true"):
            r = cell[label]
            shown = dose_s if label == "veracity" else ""
            w(f"{group:26}{opp:5}{label:10}{grid4(r['diff'], True):>9}"
              f"  [{grid4(r['lo'], True)},{grid4(r['hi'], True)}]{r['p']:9.4f}{shown:>10} "
              f"{stars(r['p'])}")
    w("```")

    a = agg["all"]
    if a["n_cells"]:
        w(f"\nAcross {a['n_cells']} valid cells: mean |veracity| = "
          f"**{a['mean_abs_veracity']:.4f}**, mean |schema| = "
          f"**{a['mean_abs_schema']:.4f}** ({a['ratio']:.1f}x).\n")
        t = agg["tft"]
        if t["n_cells"]:
            w(f"Restricted to the retaliator, where the donor can corrupt the "
              f"move field: mean |veracity| = **{t['mean_abs_veracity']:.4f}**, "
              f"mean |schema| = **{t['mean_abs_schema']:.4f}** "
              f"({t['ratio']:.1f}x) over {t['n_cells']} cells.\n")
        w("Neither ratio orders the two channels. Against the unconditional "
          "cooperator the dose on the move field is zero by construction, so "
          "the veracity term there is measured at partial dose against a schema "
          "term at full dose. The `dose` column is "
          f"P(|donor - true| >= {DOSE_THRESHOLD}) over non-degenerate arm-3c "
          "rows; `n/r` marks a database with no donor score column.\n")

    w(f"\n{RULE}\n## 2. The interaction — the actual pre-registered prediction\n{RULE}\n")
    w("The prediction is opponent-conditional: defection **down** vs TFT and "
      "**up** vs ALLC. That is a single quantity.\n")
    w("`interaction = ATE_true(allc) - ATE_true(tft)`, predicted **positive**.\n")
    w("Two same-signed effects are a main effect of defection, not the "
      "predicted conditional pattern, however large they are.\n")
    w("```")
    w(f"{'database':30}{'ATE(allc)':>11}{'ATE(tft)':>11}{'interaction':>13}"
      f"{'95% CI':>20}{'p':>9}  ")
    w("-" * 98)
    for group in sorted(inter):
        r = inter[group]
        w(f"{group:30}{grid4(r['ate_allc'], True):>11}{grid4(r['ate_tft'], True):>11}"
          f"{grid4(r['diff'], True):>13}  [{grid4(r['lo'], True)},{grid4(r['hi'], True)}]"
          f"{r['p']:9.4f} {stars(r['p'])}{'  VOID' if r['void'] else ''}")
    w("```")
    w("\nRead the sign, not the magnitude. A significantly **positive** "
      "interaction is the only result that supports the registration. A "
      "significantly negative one contradicts it. Zero is a null.\n")
    return "\n".join(L)


TURN_FILTER_NOTES = {
    "donor_matched": ("The (episode, turn) coordinates at which arm 3c's donor "
                      "was degenerate are dropped from every arm, so all three "
                      "run on identical turn compositions. Databases with no "
                      "donor_degenerate column fall back to excl_t0."),
    "excl_t0": ("Turn 0 is dropped from every arm, so both sides of each "
                "contrast run on the same 19 turns."),
    "degenerate": ("Turn 0 is dropped only from arms 3m and 3c, where it "
                   "renders byte-identically to arm 3. The two sides of a "
                   "contrast therefore differ in turn composition; reported "
                   "for comparison, not for quotation."),
    "all": ("No turn filter. This is the pre-revision behaviour and it "
            "includes turn-0 rows on which arm 3c is not falsifying "
            "anything."),
}


def main(argv: Sequence[str] | None = None) -> int:
    global N_BOOT

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="DECOMPOSITION.md")
    ap.add_argument("--json", dest="json_out", default="DECOMPOSITION.json",
                    help="sidecar the markdown is rendered from")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--turn-filter", choices=TURN_FILTERS,
                    default="excl_t0",
                    help="excl_t0 is the paper's declared rule and the default. "
                         "donor_matched is coordinate-level post-treatment "
                         "selection and is a sensitivity basis only; it must be "
                         "asked for explicitly.")
    ap.add_argument("--glob", default="*.sqlite")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    N_BOOT = args.boot

    dbs = sorted(Path(p) for p in glob.glob(args.glob)
                 if not Path(p).name.startswith(("smoke_", "cotsmoke_")))
    if not dbs:
        LOG.error("no databases matched %r", args.glob)
        return 1

    payload = compute(dbs, args.turn_filter)
    payload["aggregate"] = aggregate(payload["decomposition"])
    payload["_generated_by"] = "analysis/08_decomposition_ci.py"
    payload["_seed"] = SEED
    payload["_n_boot"] = N_BOOT
    payload["_turn_filter"] = args.turn_filter
    payload["_turn_filter_note"] = TURN_FILTER_NOTES[args.turn_filter]
    payload["_off_task_gate"] = OFF_TASK_GATE
    payload["_dose_threshold"] = DOSE_THRESHOLD
    payload["_databases"] = [p.stem for p in dbs]
    payload["_rule"] = ("values are unrounded; the paper displays four decimals "
                        "with ties resolved away from zero")

    Path(args.json_out).write_text(json.dumps(payload, indent=1, sort_keys=True),
                                   encoding="utf-8")
    Path(args.out).write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out} and {args.json_out} "
          f"({len(payload['decomposition'])} cells, "
          f"filter={args.turn_filter})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
