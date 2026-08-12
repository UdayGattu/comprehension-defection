#!/usr/bin/env python3
"""ANALYSIS 02 — episode-level re-estimation with correct standard errors.

WHY THE PRINTED INTERVALS WERE WRONG
    The run reported binomial intervals over 32,000 TURNS. Turns inside an
    episode are not independent: an episode that defects on turn 4 is far more
    likely to defect on turn 5. Treating them as independent understates the
    standard error.

WHY "EFFECTIVE N IS 230" IS ALSO WRONG
    The instrument gate flagged only ~230 distinct trajectories per placebo
    cell, and it is tempting to call that the sample size. It is not.

    Episodes are seeded independently:
        seed = sha256(run_id:episode_id:arm:model:readout:opponent)
    so every episode is an i.i.d. draw. Many land on the same trajectory
    because the all-cooperate path carries most of the probability mass. That
    is LOW ENTROPY, not dependence. Flip 1,600 fair coins: two distinct
    outcomes, still 1,600 observations.

    The gate remains worth reporting - it is what a deterministic-collapse bug
    would look like - but it does not shrink N.

THE CORRECT UNIT IS THE EPISODE
    y_i = defection_count_i / n_turns_i,  i = 1..1600 independent episodes.
    Comparing cell means over episodes is exactly the cluster-robust estimator
    for this two-level design, with the cluster being the episode.

    Two intervals are reported:
      * analytic  - Welch, normal approximation (n=1600 makes df irrelevant)
      * bootstrap - 10k percentile resamples, robust to the point mass at 0
                    that makes these distributions strongly non-normal

    Where the two disagree, trust the bootstrap and say so in the paper.

PRE-REGISTERED CONTRASTS (unchanged; this script re-estimates, never redefines)
    ATE_true      P(D | arm 3)  - P(D | arm 3b)     causal, parity-matched
    ATE_naive     P(D | arm 3)  - P(D | arm 1)      confounded, for contrast
    perturbation  P(D | arm 3b) - P(D | arm 1)      the placebo's own effect
    sign flip     ATE_true(tft) < 0 AND ATE_true(allc) > 0

    python analysis/02_episode_level.py --db sweep.sqlite --out episode_level.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
from pathlib import Path
from urllib.request import pathname2url

RULE = "=" * 78
BOOT_DEFAULT = 10_000
SEED = 20260811  # fixed so the bootstrap is reproducible


# --------------------------------------------------------------------------
# statistics, stdlib only (consistent with cdx/analysis.py)
# --------------------------------------------------------------------------

def phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - phi(abs(z)))


class Cell:
    """One (arm, opponent) cell as a vector of episode-level rates."""

    def __init__(self, arm: str, opponent: str, rates: list[float],
                 turn_successes: int, turn_total: int) -> None:
        self.arm = arm
        self.opponent = opponent
        self.rates = rates
        self.n = len(rates)
        self.mean = statistics.fmean(rates) if rates else 0.0
        self.sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        self.se = self.sd / math.sqrt(self.n) if self.n else float("nan")
        self.turn_successes = turn_successes
        self.turn_total = turn_total

    @property
    def naive_turn_se(self) -> float:
        """The SE the run printed: binomial over turns. Reported only to show
        how far off it is."""
        p = self.turn_successes / self.turn_total
        return math.sqrt(p * (1 - p) / self.turn_total)

    @property
    def design_effect(self) -> float:
        """How much the naive interval understated the uncertainty."""
        return self.se / self.naive_turn_se if self.naive_turn_se else float("nan")

    @property
    def zero_fraction(self) -> float:
        return sum(1 for r in self.rates if r == 0.0) / max(self.n, 1)

    def ci(self, z: float = 1.959964) -> tuple[float, float]:
        return (self.mean - z * self.se, self.mean + z * self.se)

    def label(self) -> str:
        return f"{self.arm}|{self.opponent}"


def ro_uri(p: Path) -> str:
    """Read-only URI for a frozen database.

    mode=ro alone is NOT enough: these databases are in WAL journal mode, and
    opening WAL read-only requires creating a -shm file, which mode=ro
    forbids. SQLite reports that as "unable to open database file".

    immutable=1 bypasses WAL and shared memory entirely - true by definition
    of a committed artefact, and a guarantee that analysis cannot mutate it.
    pathname2url handles spaces in the directory name.
    """
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def load_cells(db: str) -> dict[str, Cell]:
    p = Path(db)
    if not p.exists():
        raise SystemExit(f"database not found: {p.resolve()}\n"
                         f"Did you run `gunzip -k {p.name}.gz`?")
    conn = sqlite3.connect(ro_uri(p), uri=True)
    conn.row_factory = sqlite3.Row

    # agent_action stores the Action enum VALUE, which is 'C'/'D' - not
    # 'cooperate'/'defect'. Comparing against the long form silently returned
    # zero for every cell, which made naive_turn_se zero and printed the
    # design-effect column as nan. Episode-level numbers were unaffected
    # because they come from episodes.defection_count.
    turn_totals = {
        (r["arm"], r["opponent_policy"]): (r["defects"], r["n"])
        for r in conn.execute(
            """SELECT arm, opponent_policy,
                      SUM(agent_action='D') AS defects, COUNT(*) AS n
               FROM turns GROUP BY arm, opponent_policy"""
        )
    }

    cells: dict[str, Cell] = {}
    for arm, opp in turn_totals:
        rows = conn.execute(
            """SELECT defection_count, n_turns FROM episodes
               WHERE arm=? AND opponent_policy=? ORDER BY episode_id""",
            (arm, opp),
        ).fetchall()
        rates = [r["defection_count"] / r["n_turns"] for r in rows if r["n_turns"]]
        d, n = turn_totals[(arm, opp)]
        cell = Cell(arm, opp, rates, d, n)
        cells[cell.label()] = cell
    conn.close()
    return cells


def compare(a: Cell, b: Cell, boot: int, rng: random.Random) -> dict:
    """a - b, with analytic and bootstrap intervals."""
    diff = a.mean - b.mean
    se = math.sqrt(a.se ** 2 + b.se ** 2)
    z = diff / se if se else float("nan")
    p = two_sided_p(z) if se else float("nan")

    deltas = []
    ar, br, an, bn = a.rates, b.rates, a.n, b.n
    for _ in range(boot):
        deltas.append(
            statistics.fmean(rng.choices(ar, k=an))
            - statistics.fmean(rng.choices(br, k=bn))
        )
    deltas.sort()
    lo = deltas[int(0.025 * boot)]
    hi = deltas[min(int(0.975 * boot), boot - 1)]

    return {
        "diff": diff,
        "se": se,
        "z": z,
        "p": p,
        "ci_analytic": [diff - 1.959964 * se, diff + 1.959964 * se],
        "ci_bootstrap": [lo, hi],
        "significant": bool(lo > 0 or hi < 0),
    }


def fmt_p(p: float) -> str:
    if p != p:
        return "  n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sweep.sqlite")
    ap.add_argument("--out", default="episode_level.json")
    ap.add_argument("--bootstrap", type=int, default=BOOT_DEFAULT)
    args = ap.parse_args()

    rng = random.Random(SEED)
    cells = load_cells(args.db)
    opponents = sorted({c.opponent for c in cells.values()})

    print(f"\n{RULE}\nEPISODE-LEVEL RE-ESTIMATION\n{RULE}")
    print(f"  database    {Path(args.db).resolve()}  (read-only)")
    print(f"  bootstrap   {args.bootstrap:,} resamples, seed {SEED}")
    print("  unit        EPISODE (independently seeded => i.i.d.)")

    print(f"\n{RULE}\nCELL SUMMARIES\n{RULE}")
    print(f"  {'cell':<10}{'n_ep':>6}{'mean':>8}{'sd':>8}{'SE_ep':>9}"
          f"{'SE_turn':>9}{'inflate':>9}{'all-C':>8}")
    print("  " + "-" * 67)
    for key in sorted(cells):
        c = cells[key]
        print(f"  {key:<10}{c.n:>6}{c.mean:>8.4f}{c.sd:>8.4f}{c.se:>9.5f}"
              f"{c.naive_turn_se:>9.5f}{c.design_effect:>8.2f}x"
              f"{c.zero_fraction:>8.1%}")
    print("\n  SE_turn is what the run printed. 'inflate' is how much too")
    print("  narrow it was. 'all-C' is the share of episodes that never")
    print("  defected - the point mass that makes the bootstrap the honest")
    print("  interval here.")

    results: dict[str, dict] = {}

    print(f"\n{RULE}\nPRIMARY CONTRAST  ATE_true = P(D|3) - P(D|3b)\n{RULE}")
    print(f"  {'opp':<6}{'ATE':>9}{'SE':>9}{'95% bootstrap CI':>24}"
          f"{'p':>8}{'pred':>7}{'obs':>6}")
    print("  " + "-" * 69)
    signs: dict[str, float] = {}
    for opp in opponents:
        a, b = cells.get(f"3|{opp}"), cells.get(f"3b|{opp}")
        if not (a and b):
            continue
        r = compare(a, b, args.bootstrap, rng)
        results[f"ate_true|{opp}"] = r
        signs[opp] = r["diff"]
        pred = "down" if opp == "tft" else "up"
        obs = "up" if r["diff"] > 0 else "down"
        mark = "" if pred == obs else "   MISMATCH"
        print(f"  {opp:<6}{r['diff']:>+9.4f}{r['se']:>9.5f}"
              f"   [{r['ci_bootstrap'][0]:>+7.4f},{r['ci_bootstrap'][1]:>+7.4f}]"
              f"{fmt_p(r['p']):>8}{pred:>7}{obs:>6}{mark}")

    print(f"\n{RULE}\nSECONDARY CONTRASTS\n{RULE}")
    for name, (x, y) in {
        "perturbation  3b - 1": ("3b", "1"),
        "ATE_naive      3 - 1": ("3", "1"),
    }.items():
        print(f"\n  {name}")
        for opp in opponents:
            a, b = cells.get(f"{x}|{opp}"), cells.get(f"{y}|{opp}")
            if not (a and b):
                continue
            r = compare(a, b, args.bootstrap, rng)
            results[f"{x}_minus_{y}|{opp}"] = r
            print(f"    {opp:<6}{r['diff']:>+9.4f}  "
                  f"[{r['ci_bootstrap'][0]:>+7.4f},{r['ci_bootstrap'][1]:>+7.4f}]"
                  f"  p {fmt_p(r['p'])}")

    print(f"\n{RULE}\nPRE-REGISTERED SIGN-FLIP TEST\n{RULE}")
    tft, allc = signs.get("tft"), signs.get("allc")
    verdict = "INDETERMINATE"
    if tft is not None and allc is not None:
        tft_sig = results["ate_true|tft"]["significant"]
        allc_sig = results["ate_true|allc"]["significant"]
        print(f"  ATE_true(tft)   {tft:+.4f}   predicted NEGATIVE   "
              f"{'significant' if tft_sig else 'not significant'}")
        print(f"  ATE_true(allc)  {allc:+.4f}   predicted POSITIVE   "
              f"{'significant' if allc_sig else 'not significant'}")
        if tft < 0 < allc and tft_sig and allc_sig:
            verdict = "SUPPORTED"
        elif tft_sig and allc_sig:
            verdict = "REJECTED"
        else:
            verdict = "UNDERPOWERED"
        print(f"\n  VERDICT: {verdict}")
        if verdict == "REJECTED":
            print("  Both effects are significant and share a sign. The")
            print("  opponent-conditional prediction is rejected, not merely")
            print("  unsupported. Report it as a rejection.")

    print(f"\n{RULE}\nINSTRUMENT NOTE FOR THE PAPER\n{RULE}")

    # Derived from THIS database. An earlier version hardcoded exp1's numbers
    # ("~230 of 1,600") and printed them under every run regardless of N, which
    # is exactly the kind of stale boilerplate that ends up quoted in a paper.
    n_ep = max((c.n for c in cells.values()), default=0)
    worst = max(cells.values(), key=lambda c: c.zero_fraction, default=None)
    print(f"""
  N per cell: {n_ep:,} episodes, independently seeded.

  Low-entropy cells in this run (highest share of episodes that never
  defected): {worst.label() if worst else 'n/a'} at {worst.zero_fraction:.1%}.

  If the run's distinct-trajectory gate flagged those cells, report the flag
  AND report why it does not reduce N: episodes are independently seeded, so a
  repeated trajectory is a repeated DRAW, not a repeated observation. The gate
  exists to catch deterministic collapse, which looks identical in that column
  but would also show zero variance in the logit masses. Check that before
  dismissing it.

  Standard-error inflation over turn-level intervals in this run:
  {min(c.design_effect for c in cells.values()):.2f}x to \
{max(c.design_effect for c in cells.values()):.2f}x. Quote the episode-level
  intervals above, never the turn-level ones printed during the run.
""")

    payload = {
        "unit": "episode",
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": SEED,
        "sign_flip_verdict": verdict,
        "cells": {
            k: {
                "n_episodes": c.n, "mean": c.mean, "sd": c.sd, "se": c.se,
                "se_turn_naive": c.naive_turn_se,
                "se_inflation": c.design_effect,
                "all_cooperate_fraction": c.zero_fraction,
            }
            for k, c in cells.items()
        },
        "contrasts": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"  written  {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())