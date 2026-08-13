#!/usr/bin/env python3
"""Arm 3c, stratified: is the dose-response real, or the agent's own history?

THE CONFOUND THIS EXISTS TO REMOVE
    analysis/09_dose_response.py finds defection rising monotonically with the
    size of the lie in both semantic conditions - llama 0.086 -> 0.370, qwen
    0.043 -> 0.353. Taken at face value that is graded evidence of content use.

    It cannot be taken at face value. `lie = donor_score - true_score`, and
    true_score is the RECIPIENT'S OWN cumulative payoff. Against ALLC an agent
    that defects earns 5/round while a cooperating donor earns 3/round, so:

        agent defects early
          -> its true score runs high
          -> the lie is large and negative
          -> and it keeps defecting, because defection is autocorrelated
             within an episode.

    The large-|lie| bins are SELECTED for episodes that were already defecting.
    The unstratified gradient may be measuring persistence, not causation.

THE RANDOMISATION THAT FIXES IT
    cdx/donor.py picks uniformly at random among the live episodes whose state
    fingerprint differs from the recipient's. So CONDITIONAL ON THE RECIPIENT'S
    OWN STATE, the donor is randomly assigned. That is a real randomisation and
    it has never been used.

    Stratifying on (turn, true_score) holds the recipient's position and history
    nearly fixed - two agents at turn 8 with a true score of 24 have played
    almost the same game - and within a stratum the donor's number is random.
    The within-stratum contrast is therefore causal, where the pooled one is
    observational.

    Both are printed side by side. The gap between them IS the confound,
    measured rather than argued about.

READING IT
    stratified effect ~ naive effect   the gradient survives; content use is
                                       real and graded. Strongest positive
                                       result the study can produce.
    stratified ~ 0, naive large        the gradient was trajectory
                                       autocorrelation. Agrees with
                                       08_decomposition_ci.py, which finds
                                       content ~0 and schema large, by an
                                       independent route.

    Bootstrap is over EPISODES, not turns: turns within an episode are not
    independent, which is the whole reason this confound exists.

    python analysis/11_stratified_donor.py
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.request import pathname2url

SEED = 20260811
N_BOOT = 2000
OFF_TASK_GATE = 0.10
MIN_STRATUM = 30          # strata smaller than this carry no information
RULE = "=" * 78

try:
    import numpy as _np
except ImportError:                                       # pragma: no cover
    _np = None


def ro_uri(p: Path) -> str:
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def load(path: Path):
    """Per-turn rows of arm 3c with the recipient's true score reconstructed.

    The [STATE] block reports the score BEFORE the current move, so the true
    value at turn t is the cumulative agent_payoff over turns 0..t-1 of the same
    episode. Verified at turn 0, where it must be 0 for every episode.
    """
    con = sqlite3.connect(ro_uri(path), uri=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(turns)")}
    if "donor_agent_score" not in cols:
        con.close()
        return None, 0
    rows = con.execute("""
        SELECT opponent_policy, episode_id, turn, agent_action, agent_payoff,
               donor_agent_score, donor_degenerate, action_mass_total
        FROM turns WHERE arm='3c'
        ORDER BY opponent_policy, episode_id, turn""").fetchall()
    con.close()
    if not rows:
        return None, 0

    by_ep = defaultdict(list)
    for r in rows:
        by_ep[(r[0], r[1])].append(r)

    out, bad = [], 0
    for (opp, ep), turns in by_ep.items():
        running = 0
        for _, _, turn, act, payoff, donor, degen, mass in turns:
            if turn == 0 and running != 0:
                bad += 1
            if donor is not None and not degen:
                out.append({
                    "opp": opp, "ep": ep, "turn": turn,
                    "d": 1 if act == "D" else 0,
                    "true": running, "donor": donor,
                    "mass": mass if mass is not None else 1.0,
                })
            running += (payoff or 0)
    return out, bad


def naive_effect(rows) -> tuple[float, int, int]:
    """P(defect | donor < true) - P(defect | donor > true), pooled."""
    lo = [r["d"] for r in rows if r["donor"] < r["true"]]
    hi = [r["d"] for r in rows if r["donor"] > r["true"]]
    if not lo or not hi:
        return float("nan"), len(lo), len(hi)
    return sum(lo) / len(lo) - sum(hi) / len(hi), len(lo), len(hi)


def stratified_effect(rows) -> tuple[float, int, int]:
    """Same contrast, computed WITHIN (turn, true_score) and pooled by weight.

    Within a stratum the recipient's position and history are nearly fixed and
    the donor is randomly assigned, so this contrast is causal. Strata that do
    not contain both directions contribute nothing and are dropped - they carry
    no information about the contrast.
    """
    cells = defaultdict(lambda: ([], []))
    for r in rows:
        if r["donor"] == r["true"]:
            continue
        lo, hi = cells[(r["turn"], r["true"])]
        (lo if r["donor"] < r["true"] else hi).append(r["d"])
    num = den = 0.0
    used = 0
    for (lo, hi) in cells.values():
        if len(lo) < 2 or len(hi) < 2 or len(lo) + len(hi) < MIN_STRATUM:
            continue
        wgt = len(lo) + len(hi)
        num += wgt * (sum(lo) / len(lo) - sum(hi) / len(hi))
        den += wgt
        used += 1
    return (num / den if den else float("nan")), used, int(den)


def boot(rows, fn, n_boot=N_BOOT):
    """Bootstrap over EPISODES - turns within an episode are dependent."""
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[(r["opp"], r["ep"])].append(r)
    keys = list(by_ep)
    n = len(keys)
    if n < 2:
        return float("nan"), float("nan")
    rng = (_np.random.default_rng(SEED) if _np is not None else None)
    import random as _r
    pyr = _r.Random(SEED)
    vals = []
    for _ in range(n_boot):
        if rng is not None:
            idx = rng.integers(0, n, size=n)
        else:
            idx = [pyr.randrange(n) for _ in range(n)]
        samp = [x for i in idx for x in by_ep[keys[i]]]
        v = fn(samp)[0]
        if v == v:                      # skip NaN
            vals.append(v)
    if len(vals) < 10:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="STRATIFIED_DONOR.md")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    L: list[str] = []

    def w(s=""):
        L.append(s)

    w("# Arm 3c, stratified: does the donor's number change play?\n")
    w("`09_dose_response.py` pools all turns, so its gradient is confounded: "
      "`lie = donor_score - true_score` and `true_score` is the recipient's own "
      "cumulative payoff, so a large lie marks an agent that was *already* "
      "defecting.\n")
    w("`cdx/donor.py` selects uniformly at random among live episodes with a "
      "distinct fingerprint, so **conditional on the recipient's state the donor "
      "is randomly assigned**. Stratifying on `(turn, true_score)` holds the "
      "recipient's history nearly fixed and makes the contrast causal.\n")
    w("Both are shown. **The gap between them is the confound, measured.**\n")
    w(f"Bootstrap over episodes, {args.boot:,} resamples, seed {SEED}. Strata "
      f"with fewer than {MIN_STRATUM} rows or missing either direction are "
      "dropped — they carry no information about the contrast.\n")
    w("\n`effect = P(defect | donor < true) - P(defect | donor > true)`\n")
    w("Positive means a block understating the score drives more defection.\n")

    for path in sorted(Path(p) for p in glob.glob("*.sqlite")):
        if path.name.startswith(("smoke_", "cotsmoke_")):
            continue
        rows, bad = load(path)
        if not rows:
            continue
        w(f"\n{RULE}\n## `{path.stem}`\n{RULE}\n")
        if bad:
            w(f"**RECONSTRUCTION FAILED** — {bad} episodes have a non-zero true "
              "score at turn 0. Skipped.\n")
            continue

        w("```")
        w(f"{'opp':6}{'naive':>10}{'95% CI':>20}{'stratified':>13}"
          f"{'95% CI':>20}{'strata':>8}{'n':>9}")
        for opp in ("allc", "tft"):
            sub = [r for r in rows if r["opp"] == opp]
            if not sub:
                continue
            off = sum(1 for r in sub if r["mass"] < OFF_TASK_GATE) / len(sub)
            nv, _, _ = naive_effect(sub)
            st_, used, n = stratified_effect(sub)
            nlo, nhi = boot(sub, naive_effect, args.boot)
            slo, shi = boot(sub, stratified_effect, args.boot)
            flag = "  VOID" if off > OFF_TASK_GATE else ""
            w(f"{opp:6}{nv:+10.4f}  [{nlo:+.4f},{nhi:+.4f}]{st_:+13.4f}"
              f"  [{slo:+.4f},{shi:+.4f}]{used:>8}{n:>9,}{flag}")
        w("```")

    w(f"\n{RULE}\n## How to read it\n{RULE}\n")
    w("**Stratified effect close to the naive one** — the gradient survives "
      "randomisation. The model responds to what the block says, graded by how "
      "wrong it is. That is the strongest positive evidence of content use in "
      "the study, and it makes the replicated qwen-vs-TFT content effect "
      "(−0.2407 in exp2, −0.2375 in exp3) a dose-response rather than an "
      "anomaly.\n")
    w("**Stratified effect near zero while the naive one is large** — the "
      "gradient was the agent's own trajectory. It agrees with "
      "`08_decomposition_ci.py`, which finds content ≈ 0 and schema large, by a "
      "completely independent route. Two routes to the same conclusion is worth "
      "more than either alone.\n")
    w("\nWhichever it is, report both numbers. The naive figure is what a study "
      "without the stratification would have published, and the distance "
      "between them is the size of the mistake it would have made.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())