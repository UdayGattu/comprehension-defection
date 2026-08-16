#!/usr/bin/env python3
"""Bootstrap intervals for the content/schema split and for the interaction.

WHY THIS EXISTS

  1. ATE_true is not one thing.

     ATE_true = P(D|3) - P(D|3b)
              = [P(D|3) - P(D|3c)]  +  [P(D|3c) - P(D|3b)]
                 content effect          schema effect

     Arm 3 and arm 3c share a template and differ only in whether the numbers
     are true. Block-reading is ~1.00 in both (analysis/04_donor_echo.py), so
     the content contrast holds ATTENTION constant and varies only TRUTH.
     Arm 3c and arm 3b differ in whether number-shaped fields exist at all.

     Point estimates over 20 valid cells give mean |content| = 0.035 and mean
     |schema| = 0.147 - four times larger. If that holds up, ATE_true is mostly
     NOT an information effect, and the primary estimand of the study needs
     renaming. That is too large a claim to rest on point estimates.

  2. The pre-registered prediction is an INTERACTION, not two main effects.

     It says defection moves DOWN vs TFT and UP vs ALLC. That is

        interaction = ATE_true(allc) - ATE_true(tft) > 0

     Scoring each opponent separately lets a uniformly-signed shift score half
     a win: exp5 has all six components positive, which is a main effect of
     defection, and mistral's interaction is -0.0066 - the wrong sign.

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

  Only exp2 and exp3 carry arm 3c. exp4 and exp5 dropped it, so they can only
  contribute the interaction, not the decomposition.

    python analysis/08_decomposition_ci.py
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.request import pathname2url

SEED = 20260811          # same as 02_episode_level.py
N_BOOT = 10000
OFF_TASK_GATE = 0.10
RULE = "=" * 78


def ro_uri(p: Path) -> str:
    """mode=ro alone fails on WAL databases; immutable=1 promises sole reader."""
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def episode_rates(con) -> dict[tuple[str, str], list[float]]:
    """Per-episode defection rate, keyed by (arm, opponent)."""
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for arm, opp, d, n in con.execute("""
        SELECT arm, opponent_policy,
               SUM(CASE WHEN agent_action='D' THEN 1 ELSE 0 END), COUNT(*)
        FROM turns GROUP BY arm, opponent_policy, episode_id"""):
        if n:
            out[(arm, opp)].append(d / n)
    return out


def off_task(con) -> dict[tuple[str, str], float]:
    try:
        return {(a, o): v for a, o, v in con.execute("""
            SELECT arm, opponent_policy,
                   AVG(CASE WHEN action_mass_total < ? THEN 1.0 ELSE 0 END)
            FROM turns GROUP BY arm, opponent_policy""", (OFF_TASK_GATE,))}
    except sqlite3.Error:
        return {}


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


def boot_diff(a: list[float], b: list[float], rng) -> list[float]:
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


def summarise(dist: list[float]) -> tuple[float, float, float, float]:
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


def main() -> int:
    # Declared before any read of the name: argparse's default= below is a use,
    # and Python requires the global statement to precede it.
    global N_BOOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="DECOMPOSITION.md")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    N_BOOT = args.boot

    dbs = sorted(Path(p) for p in glob.glob("*.sqlite")
                 if not Path(p).name.startswith(("smoke_", "cotsmoke_")))
    L: list[str] = []

    def w(s=""):
        L.append(s)

    w("# Content vs schema, and the interaction\n")
    w(f"Episode-level bootstrap, {N_BOOT:,} resamples, seed {SEED}. The episode "
      "is the unit; arms and opponents are resampled independently because they "
      "are separate runs.\n")
    w("```\ncontent     = P(D|3)  - P(D|3c)   same template, TRUE vs FALSE numbers\n"
      "schema      = P(D|3c) - P(D|3b)   number-shaped fields present at all\n"
      "ATE_true    = P(D|3)  - P(D|3b)   = content + schema\n"
      "interaction = ATE_true(allc) - ATE_true(tft)\n```\n")
    w("Block-reading is ~1.00 in both arm 3 and arm 3c, so the content contrast "
      "holds attention constant and varies only truth.\n")

    # ---------------- decomposition -------------------------------------
    w(f"\n{RULE}\n## 1. Decomposition (needs arm 3c: exp2 and exp3 only)\n{RULE}\n")
    w("```")
    w(f"{'database':26}{'opp':5}{'contrast':10}{'est':>9}{'95% CI':>20}{'p':>9}  ")
    w("-" * 84)

    agg_content, agg_schema = [], []
    for path in dbs:
        con = sqlite3.connect(ro_uri(path), uri=True)
        rates, offs = episode_rates(con), off_task(con)
        for opp in ("allc", "tft"):
            need = [(a, (arm, opp)) for a, arm in
                    (("3", "3"), ("3c", "3c"), ("3b", "3b"))]
            if not all(k in rates for _, k in need):
                continue
            if max(offs.get((arm, opp), 0.0) for arm, _ in
                   (("3", 0), ("3c", 0), ("3b", 0))) > OFF_TASK_GATE:
                continue
            rng = make_rng(SEED)
            for label, x, y in (("content", "3", "3c"),
                                ("schema", "3c", "3b"),
                                ("ATE_true", "3", "3b")):
                est, lo, hi, p = summarise(
                    boot_diff(rates[(x, opp)], rates[(y, opp)], rng))
                w(f"{path.stem:26}{opp:5}{label:10}{est:+9.4f}"
                  f"  [{lo:+.4f},{hi:+.4f}]{p:9.4f} {stars(p)}")
                if label == "content":
                    agg_content.append(abs(est))
                if label == "schema":
                    agg_schema.append(abs(est))
        con.close()
    w("```")
    if agg_content:
        w(f"\nAcross {len(agg_content)} valid cells: mean |content| = "
          f"**{sum(agg_content)/len(agg_content):.4f}**, mean |schema| = "
          f"**{sum(agg_schema)/len(agg_schema):.4f}** "
          f"({(sum(agg_schema)/len(agg_schema))/(sum(agg_content)/len(agg_content)):.1f}x).\n")
        w("A cell where |schema| exceeds |content| and both CIs exclude zero is "
          "direct evidence that ATE_true is dominated by the block's SHAPE "
          "rather than by the truth of its contents.\n")

    # ---------------- interaction ---------------------------------------
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
    for path in dbs:
        con = sqlite3.connect(ro_uri(path), uri=True)
        rates, offs = episode_rates(con), off_task(con)
        ok = all((a, o) in rates for a in ("3", "3b") for o in ("allc", "tft"))
        if ok:
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
                       for a in ("3", "3b") for o in ("allc", "tft")) > OFF_TASK_GATE
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
            est, lo, hi, p = summarise(dist)
            ea = sum(a3a)/len(a3a) - sum(a3ba)/len(a3ba)
            et = sum(a3t)/len(a3t) - sum(a3bt)/len(a3bt)
            w(f"{path.stem:30}{ea:+11.4f}{et:+11.4f}{est:+13.4f}"
              f"  [{lo:+.4f},{hi:+.4f}]{p:9.4f} {stars(p)}"
              f"{'  VOID' if void else ''}")
        con.close()
    w("```")
    w("\nRead the sign, not the magnitude. A significantly **positive** "
      "interaction is the only result that supports the registration. A "
      "significantly negative one contradicts it. Zero is a null.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())