#!/usr/bin/env python3
"""ANALYSIS 14 - the objections a referee will raise, answered from committed data.

Nothing here needs a GPU. Every number comes out of a .sqlite that already
exists. Eight checks, in the order of how much damage they can do to the paper.

A. REVEALED-OPPONENT STRATIFICATION.  READ THIS ONE FIRST.
    The pre-registered prediction is "defection DOWN vs TFT, UP vs ALLC". It
    asks the model to condition on the opponent's type. But along a cooperative
    trajectory TFT and ALLC are OBSERVATIONALLY IDENTICAL: both emit Cooperate
    every turn until the agent defects, so no [STATE] block, no [HISTORY]
    block, and no arm of this study contains one bit that distinguishes them.
    Until the agent defects, the hypothesis asks for a discrimination the
    stimulus makes impossible, and any test over all turns is diluted by turns
    where the correct behaviour is identical under both opponents.

    So ATE_true = P(D|3) - P(D|3b) is recomputed twice:
        PRE-REVEAL    turns up to and including the agent's first defection -
                      the opponent has never had an opportunity to retaliate,
                      and the two opponents are indistinguishable
        REVEALED      every turn after it - TFT has now defected and ALLC has
                      not, so the type is in the context window as a fact
    plus the BASE RATE: the share of episodes that ever reach the revealed
    state at all. If that base rate is small the pre-registered test was
    running almost entirely on turns where it could not have succeeded.

    WHAT A REFEREE CONCLUDES
      revealed sign-flip SUPPORTED, pre-reveal null  -> the original rejection
          was a power/design artefact and the paper's central claim changes.
          Report it as such; do not bury it.
      revealed sign-flip REJECTED too                -> the rejection is robust
          to the strongest available defence of the hypothesis, which makes it
          a much stronger result than the pooled version.
      base rate near zero                            -> the study cannot speak
          to opponent-conditional play at all, and that is a limitation, not a
          finding.

    THE STRATIFICATION IS POST-TREATMENT AND THE SCRIPT SAYS SO EVERY TIME.
    "Has defected at least once" is an OUTCOME, not a covariate. Conditioning
    on it breaks randomisation between arms 3 and 3b: the revealed sub-sample
    of arm 3 is not exchangeable with the revealed sub-sample of arm 3b. The
    revealed-stratum ATE is therefore a DESCRIPTIVE contrast within a selected
    population, not a causal effect. It is still the right number to report,
    because it is the only one that answers the objection - but it is reported
    beside the selection rates that generate the bias, never alone.

B. MODEL HETEROGENEITY  (CLAIMS.md G8, currently "Untested").
    ATE_true vs ALLC is +0.074 / +0.044 / +0.029 across llama / qwen / mistral.
    Three numbers that do not overlap are three case studies; three that do are
    one effect. Bootstrap difference-of-differences, pairwise and joint.
    NOTHING POOLED MAY BE WRITTEN UNTIL THIS RUNS - which is why it runs here
    and prints a one-line verdict.

C. ARM 1 CPR - the missing denominator.
    Arm 1 has no [STATE] block, but [HISTORY] is always present, so the history
    IS the true state and every probe target is recoverable from it. Arm 3
    scores CPR 1.000 - and in arm 3 every probe target is printed verbatim in
    the block, so 1.000 may certify nothing but copying. Arm 1 is the only cell
    where the model must RECONSTRUCT. The paper's title question has no
    denominator without it. Per model, per readout, per turn index, because a
    CPR that decays with turn index is a working-memory result and a flat one
    is not.

D. THE 3c OVERSHOOT  (EXPERIMENTS.md "OPEN: the per-falsified-row overshoot").
    Rescaled by its own falsification rate, 3c exceeds 3m by 26% (llama) and
    32% (qwen). Three candidate explanations, all tested here:
      (i)   3c's falsified rows land on high-leverage turns. Tested by
            reweighting 3m's per-turn effect by 3c's per-turn falsification
            rate. If the reweighted prediction matches the observed 3c effect
            while the flat rescale does not, the overshoot is turn composition
            and there is no new mechanism.
      (ii)  a field interaction. 3c can corrupt score AND move at once; 3m
            corrupts only the move. Split 3c's falsified rows by whether the
            score was also wrong.
      (iii) POST-TREATMENT SELECTION - the explanation the open question omits.
            A 3c row is falsified precisely when the donor's asserted last move
            differs from the recipient's true one, i.e. WHEN THE RECIPIENT'S
            OWN RECENT PLAY WAS UNUSUAL. Falsified and unfalsified rows are
            therefore not exchangeable, and dividing the marginal effect by the
            marginal falsification rate is a biased estimator of the per-row
            effect. Quantified by the recipient's own prior-defection rate in
            falsified vs unfalsified rows. If those differ, the "per lied row"
            column in analysis/13 - and the overshoot computed from it - is not
            a per-row causal effect and must not be reported as one.

E. SCORE DOSE-RESPONSE - is "29x" a field statement or a dose statement?
    Arm 3s shifts the score by a FIXED +/-15. Arm 3m flips its field 100% of
    the time and categorically - a move is Cooperate or Defect, there is no
    "slightly wrong move". Comparing them and concluding "the move field
    matters more than the score field" confounds WHICH FIELD with HOW BIG THE
    LIE. Arm 3c is the only arm where the score error VARIES, so it is the only
    arm that can put a slope on the score. Defection is regressed on
    |displayed - true|, binned, with episode-clustered intervals, and the
    implied effect of a 15-point error is compared against arm 3s's measured
    effect. Agreement means the score arm is well described by a dose curve and
    "29x" is a statement about dose. Disagreement means it is not, and 3s's
    effect is a threshold or a format effect rather than a magnitude effect.

    The slope is also refitted on 3c rows where the MOVE field was NOT
    falsified. Pooled over all 3c rows the slope is contaminated by the move
    field, which is the larger effect by an order of magnitude.

F. MULTIPLICITY.
    exp6 reports ~30 bootstrap contrasts and reads significance off each one
    independently. Two sentences in CLAIMS.md depend directly on it:
      "four of six score contrasts exclude zero"  - four marginal findings in a
          family of six is exactly the pattern multiplicity control exists to
          catch;
      mistral's +0.0001 [-0.0002, +0.0003]        - a CI three ten-thousandths
          wide, from a cell whose defection rate is 0.0001. Its p-value is
          meaningless at three decimal places and it should not be counted as a
          finding whatever the correction says.
    Holm-Bonferroni (controls FWER, the right target for "which individual
    sentences may be written") and Benjamini-Hochberg (controls FDR, the right
    target for "how many of these are real") are both applied, because they
    answer different questions and the paper makes both kinds of claim.

G. NEAR-TIE / ACTION MASS.
    P(defect) here is not a probability the model emitted. It is a renormalised
    two-way projection: the mass on the Cooperate token and the mass on the
    Defect token, rescaled to sum to one. When action_mass_total is small the
    model was mostly saying something else and the renormalisation amplifies
    whatever noise remains. The off-task gate is 0.10; a decision at 0.11 is
    not off-task but it is one token away from being noise. This prints the
    distribution of action_mass_total and logit_gap per cell and the share of
    decisions in the FRAGILE band (mass < 0.25), concentrating on the cells
    carrying the largest effects. A large effect computed from a cell that is
    half fragile is a different object from the same number computed from a
    cell that is not.

H. 3b ROUND PARITY.
    Arm 3b is the placebo and must contain no decision-relevant state. It
    contains "Round parity: even/odd". That is one bit of the turn index, and
    under a KNOWN 20-round horizon the turn index is weakly endgame-relevant -
    backward induction is a statement about how close the last round is. If
    arm-3b defection moves with parity, or accelerates near the horizon in a
    way arm 1 does not, then the placebo is carrying usable state and ATE_true
    is biased toward zero. Arm 1, which has no block at all, is the control:
    an endgame slope present in BOTH arms is a property of the game, not a leak
    from the parity line.

CONVENTIONS (identical to analysis/13 so the numbers are comparable)
    read-only immutable URI, episode-level bootstrap, seed 20260811,
    10,000 resamples, percentile intervals.

TURN 0
    analysis/13 drops turn 0 from every contrast. That is right for 3m and 3c,
    which are byte-identical to arm 3 at turn 0 because there is no last move
    to flip. It is WRONG for 3s: the score is falsified at turn 0 too (0 -> 15),
    so dropping it discards real manipulation. This script therefore drops turn
    0 only from contrasts involving 3m or 3c, prints both columns anyway, and
    labels which one it quotes.

GRACEFUL DEGRADATION
    exp2-exp5 databases predate turns.displayed_opponent_last and some lack
    arms 3s/3m entirely. Every part checks for its columns and its arms, prints
    what it skipped and why, and continues. A missing column is a fact about
    the run, not a crash.

USAGE
    gunzip -kf exp6_*.sqlite.gz
    python analysis/14_reviewer_responses.py
    python analysis/14_reviewer_responses.py --db exp6_qwen_sem_logit.sqlite
    python analysis/14_reviewer_responses.py --glob 'exp*.sqlite' --bootstrap 2000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from urllib.request import pathname2url

try:
    import numpy as np
except ImportError:                                    # pragma: no cover
    sys.exit("numpy required:  pip install numpy")

RULE = "=" * 78
SUB = "-" * 78
SEED = 20260811          # same seed as analysis/02 and analysis/13
N_BOOT = 10_000

OFFTASK_GATE = 0.10      # below this the decision is not a decision (analysis/13)
FRAGILE_GATE = 0.25      # above off-task, still one token from being noise
SCORE_FALSIFICATION = 15  # cdx/scaffold.py SCORE_FALSIFICATION, arm 3s
MIN_BIN = 30             # bins smaller than this carry no information

# Arms that are byte-identical to arm 3 at turn 0, because the field they
# falsify does not exist yet. Any contrast touching one of them drops turn 0
# from BOTH sides. Arm 3s is deliberately absent: its field (the score) is
# falsified at turn 0 as well, 0 -> 15.
DEGENERATE_AT_TURN0 = {"3m", "3c"}

# The exp6 contrast family, as reported. Multiplicity in part F is applied
# across this family, every opponent, every group.
CONTRAST_FAMILY = [
    ("ATE_true       3 - 3b", "3", "3b"),
    ("content_move   3 - 3m", "3", "3m"),
    ("content_score  3 - 3s", "3", "3s"),
    ("content_donor  3 - 3c", "3", "3c"),
    ("perturbation  3b - 1 ", "3b", "1"),
]

SCORE_CONTRAST = "content_score"   # the "four of six" family in CLAIMS.md C5


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
    found = sorted(p for p in Path(".").glob(pattern)
                   if not p.name.startswith(("smoke_", "cotsmoke_")))
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


def table_columns(conn, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def identity(conn, path: Path) -> dict:
    """model / readout labels, taken from the data rather than the filename.

    A filename can be renamed; model_id and readout_mode are written on every
    row by the driver. The stem is kept as the display group because that is
    what analysis/13 prints and what the tables in EXPERIMENTS.md are keyed on.
    """
    try:
        row = conn.execute(
            "SELECT model_id, readout_mode FROM turns LIMIT 1").fetchone()
    except sqlite3.Error:
        row = None
    model_id = row["model_id"] if row else "?"
    readout = row["readout_mode"] if row else "?"
    short = str(model_id).split("/")[-1].split("-")[0].lower()
    return {"group": path.stem, "model_id": model_id, "model": short,
            "readout": readout}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def boot_deltas(a: np.ndarray, b: np.ndarray, rng, n_boot: int) -> np.ndarray:
    """Percentile bootstrap of mean(a) - mean(b), resampling EPISODES.

    Percentile rather than normal-theory because these distributions carry a
    large point mass at zero (episodes that never defect), which makes the
    normal approximation optimistic in exactly the cells where the effect is
    smallest.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros(0)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    return da - db


def boot_p(deltas: np.ndarray) -> float:
    """Two-sided achieved significance level from the bootstrap distribution.

    p = 2 * min(share of resamples <= 0, share >= 0), floored at 1/n_boot
    because a bootstrap with B resamples cannot resolve a p below that. Two
    identical arms give deltas that are all exactly 0, both shares are 1, and
    the result clips to 1.0 - which is the correct answer, not a bug.
    """
    if deltas.size == 0:
        return float("nan")
    lo = float((deltas <= 0).mean())
    hi = float((deltas >= 0).mean())
    return float(min(1.0, max(2.0 * min(lo, hi), 1.0 / deltas.size)))


def summarise(diff: float, deltas: np.ndarray, n_a: int, n_b: int) -> dict:
    if deltas.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "sig": False, "n_a": n_a, "n_b": n_b}
    s = np.sort(deltas)
    lo = float(s[int(0.025 * s.size)])
    hi = float(s[min(int(0.975 * s.size), s.size - 1)])
    return {"diff": float(diff), "lo": lo, "hi": hi, "p": boot_p(deltas),
            "sig": bool(lo > 0 or hi < 0), "n_a": int(n_a), "n_b": int(n_b)}


def boot_diff(a: np.ndarray, b: np.ndarray, rng, n_boot: int = N_BOOT) -> dict:
    d = boot_deltas(a, b, rng, n_boot)
    diff = float(a.mean() - b.mean()) if (a.size and b.size) else float("nan")
    return summarise(diff, d, a.size, b.size)


def boot_paired(d: np.ndarray, rng, n_boot: int = N_BOOT) -> dict:
    """Bootstrap of the mean of a PER-EPISODE paired difference.

    Used where both quantities come from the same episode (even turns vs odd
    turns, first five vs last five). Pairing removes between-episode variance,
    which is the dominant term here, so the unpaired version of these tests
    would be badly underpowered.
    """
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "sig": False, "n_a": 0, "n_b": 0}
    draws = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return summarise(float(d.mean()), draws, d.size, d.size)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values. Controls FWER.

    Sorted ascending, the i-th (0-based) p is multiplied by (m - i) and the
    sequence is made monotone non-decreasing. Adjusted p <= alpha is the
    decision rule, identical to the usual step-down comparison against
    alpha/(m-i) but reportable as a number.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH step-up adjusted p-values (q-values). Controls FDR.

    Sorted ascending, the i-th (1-based) p is multiplied by m/i, then the
    sequence is made monotone non-increasing from the largest downward.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        running = min(running, val)
        adj[i] = min(1.0, running)
    return adj


def signflip_verdict(allc: dict, tft: dict) -> str:
    """The pre-registered rule, verbatim from analysis/02.

    Prediction: defection DOWN vs TFT, UP vs ALLC.
    """
    a, t = allc.get("diff"), tft.get("diff")
    if a is None or t is None or a != a or t != t:
        return "INDETERMINATE"
    if t < 0 < a and allc["sig"] and tft["sig"]:
        return "SUPPORTED"
    if allc["sig"] and tft["sig"]:
        return "REJECTED"
    return "UNDERPOWERED"


def fmt(x: float, w: int = 9, prec: int = 4, sign: bool = True) -> str:
    if x is None or x != x:
        return f"{'n/a':>{w}}"
    return f"{x:>+{w}.{prec}f}" if sign else f"{x:>{w}.{prec}f}"


def fmt_p(p: float) -> str:
    if p is None or p != p:
        return "  n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def ci(r: dict) -> str:
    return f"[{fmt(r['lo'], 7)},{fmt(r['hi'], 7)}]"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_actions(conn, arm: str, opp: str) -> dict[int, list[tuple[int, int]]]:
    """episode_id -> [(turn, defected)] for one (arm, opponent) cell.

    Recomputed from turns rather than read from episodes.defection_count,
    because defection_count is fixed over the whole episode and therefore
    cannot express any turn filter - and every stratification in this file is
    a turn filter.
    """
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in conn.execute(
        """SELECT episode_id, turn, agent_action FROM turns
           WHERE arm=? AND opponent_policy=? ORDER BY episode_id, turn""",
        (arm, opp),
    ):
        out[r["episode_id"]].append((r["turn"], 1 if r["agent_action"] == "D" else 0))
    return out


def episode_rates(eps: dict[int, list[tuple[int, int]]], min_turn: int) -> np.ndarray:
    vals = []
    for seq in eps.values():
        hits = n = 0
        for t, d in seq:
            if t < min_turn:
                continue
            n += 1
            hits += d
        if n:
            vals.append(hits / n)
    return np.array(vals, dtype=np.float64)


def revealed_split(eps, min_turn: int = 0):
    """Split each episode at its FIRST defection.

    pre  = turns up to and including the first defection. The opponent has not
           yet had an opportunity to retaliate, so TFT and ALLC have emitted an
           identical sequence and are indistinguishable from the context.
    post = every turn after it. TFT has now defected and ALLC has not, so the
           opponent's type is a fact in the [HISTORY] block.

    An episode that never defects contributes all of its turns to pre and none
    to post - correctly, because its opponent was never revealed.
    """
    pre, post, reached = [], [], 0
    for seq in eps.values():
        first = next((t for t, d in seq if d), None)
        ph = pn = qh = qn = 0
        for t, d in seq:
            if t < min_turn:
                continue
            if first is None or t <= first:
                pn += 1
                ph += d
            else:
                qn += 1
                qh += d
        if pn:
            pre.append(ph / pn)
        if qn:
            post.append(qh / qn)
            reached += 1
    return (np.array(pre, dtype=np.float64), np.array(post, dtype=np.float64),
            len(eps), reached)


def load_3c_rows(conn, has_displayed: bool) -> list[dict]:
    """Arm 3c rows with the true score reconstructed and both lies labelled.

    The [STATE] block reports the score BEFORE the current move, so the true
    value at turn t is the cumulative agent_payoff over turns 0..t-1 of the
    same episode. Reconstructed here rather than read from a column, so it
    cannot agree with the writer by sharing its bug; the reconstruction is
    checked at turn 0, where it must be 0 for every episode.

    move_lied is None where displayed_opponent_last does not exist (exp2-exp5)
    - "not recorded", which is a different thing from "did not lie" and is kept
    distinguishable downstream.
    """
    cols = ("episode_id, opponent_policy, turn, agent_action, agent_payoff, "
            "opponent_action, donor_agent_score, donor_degenerate"
            + (", displayed_opponent_last" if has_displayed else ""))
    rows = conn.execute(
        f"SELECT {cols} FROM turns WHERE arm='3c' "
        f"ORDER BY opponent_policy, episode_id, turn").fetchall()

    out: list[dict] = []
    by_ep: dict[tuple[str, int], list] = defaultdict(list)
    for r in rows:
        by_ep[(r["opponent_policy"], r["episode_id"])].append(r)

    bad_turn0 = 0
    for (opp, ep), seq in by_ep.items():
        running = 0
        prev_opp = None
        ever_d = False
        prev_d = False
        for r in seq:
            if r["turn"] == 0 and running != 0:
                bad_turn0 += 1
            shown = r["displayed_opponent_last"] if has_displayed else None
            # None means "unknowable here", never "did not lie": at turn 0 there
            # is no previous opponent action for the block to contradict, and in
            # a pre-exp6 schema the column does not exist at all.
            move_lied = (bool(shown != prev_opp)
                         if (shown is not None and prev_opp is not None) else None)
            donor = r["donor_agent_score"]
            out.append({
                "opp": opp, "ep": ep, "turn": r["turn"],
                "d": 1 if r["agent_action"] == "D" else 0,
                "true_score": running,
                "donor_score": donor,
                "score_err": None if donor is None else donor - running,
                "degenerate": bool(r["donor_degenerate"]),
                "move_lied": move_lied,
                "prior_defect": ever_d,
                "prev_defect": prev_d,
            })
            running += (r["agent_payoff"] or 0)
            prev_opp = r["opponent_action"]
            prev_d = r["agent_action"] == "D"
            ever_d = ever_d or prev_d
    for r in out:
        r["bad_turn0"] = bad_turn0
    return out


def rate_by_episode(rows, key=lambda r: True) -> np.ndarray:
    """Per-episode defection rate over the subset of rows selected by key."""
    agg: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if not key(r):
            continue
        a = agg[(r["opp"], r["ep"])]
        a[0] += r["d"]
        a[1] += 1
    return np.array([h / n for h, n in agg.values() if n], dtype=np.float64)


# ---------------------------------------------------------------------------
# A - revealed-opponent stratification
# ---------------------------------------------------------------------------

def part_a(conn, arms, opponents, rng, n_boot, pools, ident) -> dict:
    print(f"\n  {SUB}\n  A. REVEALED-OPPONENT STRATIFICATION\n  {SUB}")
    if "3" not in arms or "3b" not in arms:
        print("    SKIPPED - needs arms 3 and 3b; this database has "
              f"{sorted(arms)}.")
        return {"skipped": "arms 3 and 3b required"}

    print("    Until the agent defects, TFT and ALLC have emitted an identical")
    print("    sequence, so no arm contains a bit that distinguishes them. The")
    print("    pre-registered sign-flip is untestable on those turns.")
    print(f"\n    {'opp':<6}{'stratum':<12}{'P(D|3)':>9}{'P(D|3b)':>9}"
          f"{'ATE':>10}{'95% CI':>20}{'p':>8}{'n_ep':>7}")

    out: dict[str, dict] = {}
    strat: dict[str, dict[str, dict]] = {"all": {}, "pre": {}, "revealed": {}}
    base: dict[str, dict] = {}
    # Rate vectors go into `pools`, never into the JSON payload: part B needs
    # them to bootstrap a difference-of-differences ACROSS databases, and
    # serialising 6 x 6 x 1000 floats would treble the size of the report for
    # no reader's benefit.
    rates: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for opp in opponents:
        e3 = load_actions(conn, "3", opp)
        e3b = load_actions(conn, "3b", opp)
        if not e3 or not e3b:
            print(f"    {opp:<6}SKIPPED - no episodes in one of the arms.")
            continue
        pre3, post3, n3, reach3 = revealed_split(e3)
        pre3b, post3b, n3b, reach3b = revealed_split(e3b)
        all3, all3b = episode_rates(e3, 0), episode_rates(e3b, 0)

        rows = [("all turns", all3, all3b),
                ("pre-reveal", pre3, pre3b),
                ("revealed", post3, post3b)]
        for label, a, b in rows:
            r = boot_diff(a, b, rng, n_boot)
            key = {"all turns": "all", "pre-reveal": "pre",
                   "revealed": "revealed"}[label]
            strat[key][opp] = r
            rates[(key, opp)] = (a, b)
            out[f"{key}|{opp}"] = r
            ma = float(a.mean()) if a.size else float("nan")
            mb = float(b.mean()) if b.size else float("nan")
            print(f"    {opp:<6}{label:<12}{fmt(ma, 9, 4, False)}"
                  f"{fmt(mb, 9, 4, False)}{fmt(r['diff'], 10)}"
                  f"{ci(r):>20}{fmt_p(r['p']):>8}"
                  f"{min(r['n_a'], r['n_b']):>7}")
        base[opp] = {
            "arm3_episodes": n3, "arm3_reached": reach3,
            "arm3_base_rate": reach3 / n3 if n3 else float("nan"),
            "arm3b_episodes": n3b, "arm3b_reached": reach3b,
            "arm3b_base_rate": reach3b / n3b if n3b else float("nan"),
        }

    print("\n    BASE RATE - episodes that EVER reach the revealed state")
    print(f"    {'opp':<6}{'arm 3':>18}{'arm 3b':>18}   selection gap")
    for opp, b in base.items():
        gap = b["arm3_base_rate"] - b["arm3b_base_rate"]
        print(f"    {opp:<6}"
              f"{b['arm3_reached']:>8,}/{b['arm3_episodes']:<4,}"
              f"{b['arm3_base_rate']:>6.1%}"
              f"{b['arm3b_reached']:>8,}/{b['arm3b_episodes']:<4,}"
              f"{b['arm3b_base_rate']:>6.1%}   {gap:+.3f}")

    worst = min((b["arm3_base_rate"] for b in base.values()), default=float("nan"))
    maxgap = max((abs(b["arm3_base_rate"] - b["arm3b_base_rate"])
                  for b in base.values()), default=float("nan"))

    v_all = signflip_verdict(strat["all"].get("allc", {}), strat["all"].get("tft", {}))
    v_rev = signflip_verdict(strat["revealed"].get("allc", {}),
                             strat["revealed"].get("tft", {}))
    v_pre = signflip_verdict(strat["pre"].get("allc", {}), strat["pre"].get("tft", {}))

    print("\n    SIGN-FLIP (predicted: TFT negative, ALLC positive)")
    print(f"      all turns   {v_all:<14} the pooled test, as pre-registered")
    print(f"      pre-reveal  {v_pre:<14} untestable by construction - a verdict")
    print("                                 here is noise, printed only to show")
    print("                                 the pooled number is a blend of two")
    print("                                 strata that answer different questions")
    print(f"      revealed    {v_rev:<14} the objection's own test")

    print("\n    VERDICT: ", end="")
    if v_rev == "SUPPORTED" and v_all != "SUPPORTED":
        print("THE CENTRAL CLAIM CHANGES. The hypothesis is supported once")
        print("      the test is restricted to turns where the opponent's type is")
        print("      knowable. The pooled rejection was dilution. Rewrite A1.")
    elif v_rev == "REJECTED":
        print("the rejection survives the strongest defence of the")
        print("      hypothesis. Restricted to turns where the opponent's type IS")
        print("      revealed, the sign-flip still fails. Report the stratified")
        print("      test in the paper - it forecloses this objection.")
    elif v_rev == "UNDERPOWERED":
        print("inconclusive on the revealed stratum. Only")
        print(f"      {worst:.1%} of episodes reach it in the smallest cell, so the")
        print("      study cannot resolve opponent-conditional play. That is a")
        print("      limitation to state, not a rejection to claim.")
    else:
        print("indeterminate - one of the strata is empty.")

    print("\n    CAUTION, PRINTED EVERY TIME: 'has defected at least once' is an")
    print("    OUTCOME. Conditioning on it breaks randomisation between arms 3")
    print("    and 3b, so the revealed-stratum ATE is a descriptive contrast in a")
    print("    selected population. The selection differs between arms by up to")
    print(f"    {maxgap:.3f} here; that gap is the size of the bias, and it must be")
    print("    quoted beside the estimate. This is not a causal effect.")

    pools[("A", ident["group"])] = {"strata": strat, "base": base, "rates": rates}
    return {"contrasts": out, "base_rate": base,
            "signflip": {"all": v_all, "pre": v_pre, "revealed": v_rev}}


# ---------------------------------------------------------------------------
# C - arm 1 CPR
# ---------------------------------------------------------------------------

def part_c(conn, arms, opponents, ident) -> dict:
    print(f"\n  {SUB}\n  C. ARM 1 CPR - the denominator the title question needs\n  {SUB}")
    if "1" not in arms:
        print("    SKIPPED - arm 1 absent from this database.")
        return {"skipped": "arm 1 absent"}
    probed = conn.execute(
        "SELECT COUNT(cpr_score) FROM turns WHERE arm='1'").fetchone()[0]
    if not probed:
        print("    SKIPPED - arm 1 has no non-NULL cpr_score. The probe is run")
        print("    every --probe-every turns; this run stored none for arm 1.")
        return {"skipped": "no cpr_score on arm 1"}

    per_cell = defaultdict(dict)
    for r in conn.execute(
        """SELECT arm, opponent_policy, AVG(cpr_score) c, COUNT(cpr_score) n
           FROM turns WHERE cpr_score IS NOT NULL GROUP BY arm, opponent_policy"""
    ):
        per_cell[r["arm"]][r["opponent_policy"]] = {"cpr": r["c"], "n": r["n"]}

    print(f"    model {ident['model']}   readout {ident['readout']}")
    print(f"\n    {'arm':<5}" + "".join(f"{o:>12}" for o in opponents)
          + "   what a CPR of 1.000 would mean here")
    notes = {
        "1": "reconstruction from raw history - the real denominator",
        "3": "every probe target printed verbatim; certifies COPYING",
        "3b": "no true state in the block; this is a floor",
        "3c": "belief in the donor's numbers, not comprehension",
        "3s": "scored against the truth while shown a lie; 0.000 expected",
        "3m": "scored against the truth while shown a lie; 0.000 expected",
    }
    for arm in sorted(per_cell, key=lambda a: (len(a), a)):
        cells = "".join(
            f"{per_cell[arm].get(o, {}).get('cpr', float('nan')):>12.3f}"
            if o in per_cell[arm] else f"{'-':>12}" for o in opponents)
        print(f"    {arm:<5}{cells}   {notes.get(arm, '')}")

    print("\n    ARM 1 CPR BY TURN INDEX  (pooled over opponents; arm 3 for scale)")
    by_turn = defaultdict(lambda: {})
    for r in conn.execute(
        """SELECT arm, turn, AVG(cpr_score) c, COUNT(cpr_score) n
           FROM turns WHERE cpr_score IS NOT NULL AND arm IN ('1','3')
           GROUP BY arm, turn ORDER BY turn"""
    ):
        by_turn[r["turn"]][r["arm"]] = {"cpr": r["c"], "n": r["n"]}
    print(f"    {'turn':>5}{'arm1 CPR':>11}{'n':>8}{'arm3 CPR':>11}{'n':>8}")
    turn_series = {}
    for t in sorted(by_turn):
        a1 = by_turn[t].get("1")
        a3 = by_turn[t].get("3")
        print(f"    {t:>5}"
              f"{(a1['cpr'] if a1 else float('nan')):>11.3f}"
              f"{(a1['n'] if a1 else 0):>8,}"
              f"{(a3['cpr'] if a3 else float('nan')):>11.3f}"
              f"{(a3['n'] if a3 else 0):>8,}")
        turn_series[t] = {"arm1": a1, "arm3": a3}

    arm1 = per_cell.get("1", {})
    overall = (sum(v["cpr"] * v["n"] for v in arm1.values())
               / sum(v["n"] for v in arm1.values())) if arm1 else float("nan")
    early = [v["arm1"]["cpr"] for t, v in sorted(turn_series.items())
             if v.get("arm1") and t < 5]
    late = [v["arm1"]["cpr"] for t, v in sorted(turn_series.items())
            if v.get("arm1") and t >= 15]
    drift = (sum(late) / len(late) - sum(early) / len(early)) if early and late \
        else float("nan")

    print(f"\n    VERDICT: arm-1 CPR = {overall:.3f}. Without a block, the model")
    print(f"    reconstructs the state from raw history {overall:.1%} of the time.")
    print("    THAT is the denominator of 'does comprehension cause defection?'.")
    print("    Arm 3's CPR is not a comprehension measure at all: the block prints")
    print("    every probe target verbatim, so 1.000 there is consistent with pure")
    print("    copying and cannot license a claim about understanding.")
    if drift == drift:
        print(f"    Turn drift (turns 15+ minus turns 0-4): {drift:+.3f}. ", end="")
        print("Falls with turn index -> a working-memory result." if drift < -0.05
              else ("Rises with turn index -> more history helps, not less."
                    if drift > 0.05 else "Flat -> not a working-memory result."))
    return {"per_cell": {f"{a}|{o}": v for a, d in per_cell.items()
                         for o, v in d.items()},
            "arm1_overall": overall, "arm1_turn_drift": drift,
            "by_turn": {str(t): v for t, v in turn_series.items()}}


# ---------------------------------------------------------------------------
# I - which FIELD fails without the block
# ---------------------------------------------------------------------------

_CPR_FIELDS = (
    ("cpr_own_score", "own score", "ARITHMETIC", "sum 20 payoffs from history"),
    ("cpr_opponent_last", "opponent's last move", "RECALL", "read the line above"),
    ("cpr_rounds_played", "rounds played", "COUNTING", "count the history lines"),
)


def part_i(conn, arms, opponents, ident) -> dict:
    """Decompose arm-1 CPR into its three sub-probes.

    WHY THIS EXISTS
        Part C reports arm-1 CPR at ~0.00 from turn 4 onward and calls it the
        denominator for "does comprehension cause defection?". But CPR is
        ALL THREE probes correct, and the three are not the same task:

            own score          sum 20 payoffs from the history   ARITHMETIC
            opponent's last    read the line immediately above   RECALL
            rounds played      count the history lines           COUNTING

        A conjunction is governed by its hardest term. If arm-1 CPR is zero
        purely because a 7B model cannot sum twenty numbers, then "the model
        cannot track the state" is an ARITHMETIC result wearing a
        state-tracking costume, and the paper's denominator is wrong.

    WHY IT MATTERS MORE THAN IT LOOKS
        exp6 established that the opponent's last move is the ONLY field whose
        falsification moves behaviour (2.8x-59x the score, 6 of 6 cells). So
        this decomposition tests a specific mechanistic account:

            models track the field they USE and fail the field they do not,
            and the state block repairs the field that was already irrelevant

        If arm-1 cpr_opponent_last is high while cpr_own_score is ~0, that
        account is supported and it explains the entire corpus - including why
        injecting a perfectly-read state block changes almost nothing.

        If BOTH are ~0, the account fails: the model cannot recall even the
        immediately preceding line, the block repairs something real, and
        repairing it still does not produce opponent-conditional play. That is
        the stronger version of the paper's existing claim.

        Either outcome is publishable. Only the conjunction is uninterpretable.
    """
    print(f"\n  {SUB}\n  I. WHICH FIELD FAILS - decomposing arm-1 CPR\n  {SUB}")
    if "1" not in arms:
        print("    SKIPPED - arm 1 absent.")
        return {"skipped": "arm 1 absent"}

    cols = table_columns(conn, "turns")
    present = [(c, label, kind, how) for c, label, kind, how in _CPR_FIELDS
               if c in cols]
    if not present:
        print("    SKIPPED - no per-field CPR columns in this schema.")
        return {"skipped": "no per-field columns"}

    sel = ", ".join(f"AVG({c}) AS {c}, COUNT({c}) AS n_{c}"
                    for c, _, _, _ in present)
    rows = {(r["arm"], r["turn"]): r for r in conn.execute(
        f"""SELECT arm, turn, {sel} FROM turns
            WHERE arm IN ('1','3') AND cpr_score IS NOT NULL
            GROUP BY arm, turn ORDER BY turn""")}
    if not rows:
        print("    SKIPPED - no probed rows for arms 1 or 3.")
        return {"skipped": "no probed rows"}

    turns = sorted({t for _, t in rows})
    print(f"    model {ident['model']}   readout {ident['readout']}\n")
    print("    ARM 1 - state must be reconstructed from [HISTORY]")
    print(f"    {'turn':>5}" + "".join(f"{lab:>22}" for _, lab, _, _ in present))
    print(f"    {'':>5}" + "".join(f"{k:>22}" for _, _, k, _ in present))
    series: dict = {}
    for t in turns:
        r = rows.get(("1", t))
        if r is None:
            continue
        vals = [r[c] for c, _, _, _ in present]
        print(f"    {t:>5}" + "".join(
            f"{(v if v is not None else float('nan')):>22.3f}" for v in vals))
        series[str(t)] = {c: r[c] for c, _, _, _ in present}

    def late_mean(arm: str, col: str):
        vs = [rows[(arm, t)][col] for t in turns
              if (arm, t) in rows and t > 0 and rows[(arm, t)][col] is not None]
        return sum(vs) / len(vs) if vs else float("nan")

    print("\n    NON-TRIVIAL PROBES ONLY (turn 0 excluded - at turn 0 every field")
    print("    is 0/None, so a correct answer there is not a tracking result)")
    print(f"    {'field':<22}{'task':<12}{'arm 1':>9}{'arm 3':>9}"
          f"{'repair':>9}   reading")
    out: dict = {}
    for col, label, kind, how in present:
        a1, a3 = late_mean("1", col), late_mean("3", col)
        gap = a3 - a1 if a1 == a1 and a3 == a3 else float("nan")
        if a1 != a1:
            reading = ""
        elif a1 >= 0.85:
            reading = "TRACKED without the block"
        elif a1 <= 0.15:
            reading = "FAILS without the block"
        else:
            reading = "partial"
        print(f"    {label:<22}{kind:<12}{a1:>9.3f}{a3:>9.3f}"
              f"{gap:>+9.3f}   {reading}")
        out[col] = {"arm1": a1, "arm3": a3, "repair": gap,
                    "kind": kind, "how": how}

    score = out.get("cpr_own_score", {}).get("arm1", float("nan"))
    move = out.get("cpr_opponent_last", {}).get("arm1", float("nan"))
    verdict = "INDETERMINATE"
    print()
    if score == score and move == move:
        if move >= 0.85 and score <= 0.15:
            verdict = "USED-FIELD TRACKED"
            print("    VERDICT: USED-FIELD TRACKED. Without the block the model")
            print(f"    recalls the opponent's last move ({move:.3f}) but cannot")
            print(f"    reconstruct its own score ({score:.3f}). exp6 showed the")
            print("    last move is the ONLY field whose falsification moves")
            print("    behaviour. So the block repairs the field the model was")
            print("    already failing AND already ignoring, which is why perfect")
            print("    reading buys no behavioural change. Arm-1 CPR of ~0.00 is")
            print("    then an ARITHMETIC result and must not be reported as")
            print("    'the model cannot track the state'.")
        elif move <= 0.15 and score <= 0.15:
            verdict = "BOTH FAIL"
            print("    VERDICT: BOTH FAIL. Neither the arithmetic field nor the")
            print(f"    one-line-recall field survives without the block ({move:.3f},")
            print(f"    {score:.3f}). State tracking fails in the ordinary sense, the")
            print("    block repairs something real, and repairing it still does not")
            print("    produce opponent-conditional play. This is the STRONGER")
            print("    version of the paper's claim - the conjunction is not")
            print("    carrying it, and 'comprehension is not the bottleneck' is")
            print("    earned rather than inferred.")
        elif move >= 0.85 and score >= 0.85:
            verdict = "CONJUNCTION ARTEFACT"
            print("    VERDICT: CONJUNCTION ARTEFACT. Both fields are tracked")
            print("    individually, so a near-zero all-three CPR is produced by")
            print("    the third probe or by joint failure. Arm-1 CPR must NOT be")
            print("    quoted as a state-tracking denominator; report per field.")
        else:
            print("    VERDICT: MIXED. Report per field and do not quote the")
            print("    all-three CPR as a single denominator.")
    print("\n    Whatever the verdict, report CPR per field. A conjunction over")
    print("    three tasks of unequal difficulty is governed by the hardest one,")
    print("    and exp6 proved the three fields are not behaviourally equivalent.")
    return {"verdict": verdict, "fields": out, "arm1_by_turn": series}


# ---------------------------------------------------------------------------
# D - the 3c overshoot
# ---------------------------------------------------------------------------

def part_d(conn, arms, opponents, rng, n_boot, has_displayed) -> dict:
    print(f"\n  {SUB}\n  D. THE 3c OVERSHOOT - three explanations, all testable\n  {SUB}")
    if "3c" not in arms:
        print("    SKIPPED - arm 3c absent from this database.")
        return {"skipped": "arm 3c absent"}
    cols = table_columns(conn, "turns")
    if "donor_agent_score" not in cols:
        print("    SKIPPED - turns.donor_agent_score absent (pre-exp3 schema).")
        return {"skipped": "donor_agent_score absent"}

    rows = load_3c_rows(conn, has_displayed)
    if not rows:
        print("    SKIPPED - no arm 3c rows.")
        return {"skipped": "no 3c rows"}
    if rows[0]["bad_turn0"]:
        print(f"    RECONSTRUCTION FAILED - {rows[0]['bad_turn0']} episodes have a")
        print("    non-zero true score at turn 0. Everything in D and E is void")
        print("    for this database.")
        return {"skipped": "true-score reconstruction failed"}
    print("    true-score reconstruction check: 0 at turn 0 in every episode - OK")

    result: dict = {}

    # ---- (i) turn leverage -------------------------------------------------
    print("\n    (i) ARE 3c's FALSIFIED ROWS LANDING ON HIGH-LEVERAGE TURNS?")
    if not has_displayed:
        print("        SKIPPED - turns.displayed_opponent_last absent, so which")
        print("        rows were falsified is unrecorded. exp2-exp5 predate it.")
    elif "3m" not in arms:
        print("        SKIPPED - needs arm 3m to measure per-turn leverage.")
    else:
        print("        3m falsifies every turn >= 1, so its per-turn effect")
        print("        e(t) = P(D|3,t) - P(D|3m,t) IS the per-lied-row effect at")
        print("        turn t. If the move is the only operative field, 3c's")
        print("        effect should equal sum_t rate_c(t) e(t) / T. The flat")
        print("        rescale used in analysis/13 instead uses mean(rate) x")
        print("        mean(e), and the two differ by exactly the covariance")
        print("        between where 3c lies and where lying matters.")
        print(f"\n        {'opp':<6}{'observed':>11}{'reweighted':>12}"
              f"{'flat rescale':>14}{'cov(rate,e)':>13}")
        lev = {}
        for opp in opponents:
            t3 = _turn_rates(conn, "3", opp)
            tm = _turn_rates(conn, "3m", opp)
            tc = _turn_rates(conn, "3c", opp)
            rc = _turn_falsification(conn, "3c", opp)
            turns = sorted(set(t3) & set(tm) & set(tc) & set(rc) - {0})
            if not turns:
                print(f"        {opp:<6}no shared turns >= 1.")
                continue
            e = np.array([t3[t] - tm[t] for t in turns])
            rate = np.array([rc[t] for t in turns])
            obs = float(np.mean([t3[t] - tc[t] for t in turns]))
            reweighted = float(np.mean(rate * e))
            flat = float(rate.mean() * e.mean())
            cov = reweighted - flat
            print(f"        {opp:<6}{fmt(obs, 11)}{fmt(reweighted, 12)}"
                  f"{fmt(flat, 14)}{fmt(cov, 13)}")
            lev[opp] = {"observed": obs, "reweighted": reweighted,
                        "flat_rescale": flat, "covariance": cov}
        result["turn_leverage"] = lev
        if lev:
            best = max(lev.values(), key=lambda v: abs(v["observed"]))
            near = abs(best["observed"] - best["reweighted"]) < \
                abs(best["observed"] - best["flat_rescale"])
            print("\n        VERDICT: ", end="")
            if near:
                print("turn composition explains it. The reweighted")
                print("        prediction is closer to the observed 3c effect than the")
                print("        flat rescale is, so the overshoot is WHERE 3c lies, not")
                print("        a second mechanism. The open question in EXPERIMENTS.md")
                print("        can be closed with this line.")
            else:
                print("turn composition does NOT explain it. The")
                print("        reweighted prediction is no closer than the flat one, so")
                print("        candidate (i) is eliminated and (ii) or (iii) must carry")
                print("        the overshoot.")

    # ---- (ii) field interaction -------------------------------------------
    print("\n    (ii) DO THE TWO FIELDS INTERACT WHEN BOTH ARE WRONG?")
    if not has_displayed:
        print("        SKIPPED - cannot label which rows had the move falsified.")
    else:
        print(f"        {'opp':<6}{'cell':<22}{'P(D)':>9}{'rows':>9}")
        inter = {}
        for opp in opponents:
            sub = [r for r in rows if r["opp"] == opp and not r["degenerate"]
                   and r["turn"] >= 1 and r["move_lied"] is not None
                   and r["score_err"] is not None]
            if not sub:
                print(f"        {opp:<6}no usable rows.")
                continue
            groups = {
                "move+score wrong": lambda r: r["move_lied"] and r["score_err"] != 0,
                "move only wrong": lambda r: r["move_lied"] and r["score_err"] == 0,
                "score only wrong": lambda r: (not r["move_lied"]) and r["score_err"] != 0,
                "neither wrong": lambda r: (not r["move_lied"]) and r["score_err"] == 0,
            }
            cell = {}
            for name, key in groups.items():
                sel = [r for r in sub if key(r)]
                if not sel:
                    print(f"        {opp:<6}{name:<22}{'-':>9}{0:>9}")
                    cell[name] = {"pd": float("nan"), "n": 0}
                    continue
                pd_ = sum(r["d"] for r in sel) / len(sel)
                print(f"        {opp:<6}{name:<22}{pd_:>9.4f}{len(sel):>9,}")
                cell[name] = {"pd": pd_, "n": len(sel)}
            both = cell.get("move+score wrong", {})
            only = cell.get("move only wrong", {})
            if both.get("n", 0) >= MIN_BIN and only.get("n", 0) >= MIN_BIN:
                a = rate_by_episode(sub, lambda r: r["move_lied"] and r["score_err"] != 0)
                b = rate_by_episode(sub, lambda r: r["move_lied"] and r["score_err"] == 0)
                r_ = boot_diff(a, b, rng, n_boot)
                print(f"        {opp:<6}{'interaction':<22}{fmt(r_['diff'], 9)}"
                      f"   {ci(r_)}  p {fmt_p(r_['p'])}")
                cell["interaction"] = r_
            else:
                print(f"        {opp:<6}{'interaction':<22}"
                      f"   not estimable (a cell is below {MIN_BIN} rows)")
            inter[opp] = cell
        result["field_interaction"] = inter
        print("\n        A move+score cell materially above move-only is a field")
        print("        interaction and explains the overshoot. Equal cells rule it")
        print("        out. 'Neither wrong' is the internal control: those rows are")
        print("        arm 3 wearing arm 3c's label and should match arm 3.")

    # ---- (iii) post-treatment selection ------------------------------------
    print("\n    (iii) POST-TREATMENT SELECTION - is 'per lied row' even a thing?")
    if not has_displayed:
        print("        SKIPPED - cannot label which rows had the move falsified.")
    else:
        print("        A 3c row is falsified exactly when the donor's asserted last")
        print("        move differs from the recipient's true one. Against TFT the")
        print("        opponent's last move IS the recipient's own previous action,")
        print("        so falsification SELECTS on the recipient's own recent play.")
        print("        If it does, falsified and unfalsified rows are not")
        print("        exchangeable and effect/rate is a biased per-row estimator.")
        print(f"\n        {'opp':<6}{'group':<14}{'rows':>9}{'prev turn D':>13}"
              f"{'any prior D':>13}{'P(D) now':>10}")
        sel = {}
        for opp in opponents:
            sub = [r for r in rows if r["opp"] == opp and not r["degenerate"]
                   and r["turn"] >= 1 and r["move_lied"] is not None]
            if not sub:
                print(f"        {opp:<6}no usable rows.")
                continue
            cell = {}
            for name, flag in (("falsified", True), ("unfalsified", False)):
                g = [r for r in sub if r["move_lied"] is flag]
                if not g:
                    print(f"        {opp:<6}{name:<14}{0:>9}"
                          f"{'-':>13}{'-':>13}{'-':>10}")
                    cell[name] = {"n": 0}
                    continue
                prev = sum(r["prev_defect"] for r in g) / len(g)
                prior = sum(r["prior_defect"] for r in g) / len(g)
                pd_ = sum(r["d"] for r in g) / len(g)
                print(f"        {opp:<6}{name:<14}{len(g):>9,}{prev:>13.4f}"
                      f"{prior:>13.4f}{pd_:>10.4f}")
                cell[name] = {"n": len(g), "prev_defect": prev,
                              "prior_defect": prior, "pd": pd_}
            f_, u_ = cell.get("falsified", {}), cell.get("unfalsified", {})
            if f_.get("n") and u_.get("n"):
                gap_prev = f_["prev_defect"] - u_["prev_defect"]
                gap_prior = f_["prior_defect"] - u_["prior_defect"]
                cell["gap_prev_defect"] = gap_prev
                cell["gap_prior_defect"] = gap_prior
                print(f"        {opp:<6}{'SELECTION GAP':<14}{'':>9}"
                      f"{gap_prev:>+13.4f}{gap_prior:>+13.4f}")
            sel[opp] = cell
        result["selection"] = sel
        gaps = [abs(c.get("gap_prior_defect", 0.0)) for c in sel.values()
                if "gap_prior_defect" in c]
        worst = max(gaps) if gaps else float("nan")
        print("\n        VERDICT: ", end="")
        if worst == worst and worst > 0.05:
            print("SELECTED. The recipient's own prior-defection")
            print(f"        rate differs by up to {worst:.3f} between falsified and")
            print("        unfalsified rows. Falsification is not as-if random within")
            print("        arm 3c, so analysis/13's 'per lied row' column is not a")
            print("        per-row causal effect and the 26-32% overshoot computed")
            print("        from it is not evidence of anything. Report the marginal")
            print("        3c effect and the falsification rate separately, and drop")
            print("        the ratio - or estimate it within (turn, history) strata")
            print("        as analysis/11 does for the score.")
        elif worst == worst:
            print("balanced. Prior-defection differs by only")
            print(f"        {worst:.3f} between falsified and unfalsified rows, so the")
            print("        rescale is approximately unbiased and the overshoot needs")
            print("        one of the other two explanations.")
        else:
            print("not estimable - no falsified rows in any cell.")
    return result


def _turn_rates(conn, arm: str, opp: str) -> dict[int, float]:
    return {r["turn"]: r["p"] for r in conn.execute(
        """SELECT turn, AVG(agent_action='D') p FROM turns
           WHERE arm=? AND opponent_policy=? GROUP BY turn""", (arm, opp))}


def _turn_falsification(conn, arm: str, opp: str) -> dict[int, float]:
    """Per-turn share of rows whose block contradicted the truth.

    Self-join against the previous turn's opponent_action rather than a stored
    flag, so this cannot agree with the writer by sharing its bug - the same
    construction analysis/13 uses for the aggregate rate.
    """
    q = """
    SELECT t.turn AS turn,
           AVG(t.displayed_opponent_last <> p.opponent_action) AS rate
    FROM turns t
    JOIN turns p ON p.run_id=t.run_id AND p.episode_id=t.episode_id
                AND p.arm=t.arm AND p.model_id=t.model_id
                AND p.readout_mode=t.readout_mode
                AND p.opponent_policy=t.opponent_policy AND p.turn=t.turn-1
    WHERE t.arm=? AND t.opponent_policy=? AND t.displayed_opponent_last IS NOT NULL
    GROUP BY t.turn"""
    return {r["turn"]: r["rate"] for r in conn.execute(q, (arm, opp))}


# ---------------------------------------------------------------------------
# E - score dose-response
# ---------------------------------------------------------------------------

SCORE_BINS = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 19), (20, 10 ** 6)]


def _bin_label(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+")


def _bin_of(x: int) -> str:
    for lo, hi in SCORE_BINS:
        if lo <= x <= hi:
            return _bin_label(lo, hi)
    return "?"


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    vx = x.var()
    return float(((x - x.mean()) * (y - y.mean())).mean() / vx) if vx else float("nan")


def _cluster_boot_slope(items, rng, n_boot: int) -> tuple[float, float, float]:
    """OLS slope of defect on |score error|, resampling EPISODES not rows.

    Turns inside an episode are not independent, and the score error is
    strongly autocorrelated within an episode (it is a running difference of
    two cumulative sums), so a row-level interval here would be badly too
    narrow - the same mistake analysis/02 exists to correct.

    Implemented on per-episode SUFFICIENT STATISTICS (n, sum x, sum y, sum xx,
    sum xy) rather than by rebuilding the row list on each resample. A cluster
    bootstrap that re-materialises ~19,000 rows 2,000 times is minutes of work
    for a number the closed form gives exactly, and slowness is the reason this
    kind of interval gets replaced by a wrong row-level one.
    """
    by_ep: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(items):
        by_ep[(r["opp"], r["ep"])].append(i)
    keys = list(by_ep)
    k = len(keys)
    if k < 2:
        return float("nan"), float("nan"), float("nan")

    x = np.array([abs(r["score_err"]) for r in items], dtype=np.float64)
    y = np.array([float(r["d"]) for r in items], dtype=np.float64)
    point = _ols_slope(x, y)

    stats = np.empty((k, 5), dtype=np.float64)
    for j, key in enumerate(keys):
        idx = np.array(by_ep[key])
        xs, ys = x[idx], y[idx]
        stats[j] = (xs.size, xs.sum(), ys.sum(), (xs * xs).sum(), (xs * ys).sum())

    draw = rng.integers(0, k, size=(n_boot, k))
    s = stats[draw].sum(axis=1)                 # n_boot x 5
    n, sx, sy, sxx, sxy = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = sxx / n - (sx / n) ** 2
        cov = sxy / n - (sx / n) * (sy / n)
        slopes = np.where(var > 0, cov / var, np.nan)
    slopes = np.sort(slopes[np.isfinite(slopes)])
    if slopes.size < 10:
        return point, float("nan"), float("nan")
    return (point, float(slopes[int(0.025 * slopes.size)]),
            float(slopes[min(int(0.975 * slopes.size), slopes.size - 1)]))


def part_e(conn, arms, opponents, rng, n_boot, has_displayed) -> dict:
    print(f"\n  {SUB}\n  E. SCORE DOSE-RESPONSE - is '29x' a dose or a field?\n  {SUB}")
    if "3c" not in arms or "donor_agent_score" not in table_columns(conn, "turns"):
        print("    SKIPPED - needs arm 3c and turns.donor_agent_score.")
        return {"skipped": "arm 3c / donor_agent_score absent"}
    rows = load_3c_rows(conn, has_displayed)
    rows = [r for r in rows if not r["degenerate"] and r["score_err"] is not None]
    if not rows:
        print("    SKIPPED - no non-degenerate 3c rows with a donor score.")
        return {"skipped": "no usable 3c rows"}

    print(f"    Arm 3s lies by a FIXED {SCORE_FALSIFICATION} points and arm 3m flips a")
    print("    binary field 100% of the time. The '29x' ratio between them is")
    print("    therefore a statement about DOSE as much as about FIELD. Arm 3c is")
    print("    the only arm whose score error varies, so it is the only arm that")
    print("    can put a slope on the score - a slope that predicts what a 15-point")
    print("    lie should do, testable against what arm 3s actually did.")

    fit_boot = n_boot
    out: dict = {}
    for opp in opponents:
        sub = [r for r in rows if r["opp"] == opp]
        if len(sub) < MIN_BIN:
            print(f"\n    vs {opp}: SKIPPED - {len(sub)} usable rows.")
            continue
        print(f"\n    vs {opp}  ({len(sub):,} non-degenerate rows)")
        print(f"      {'|error|':>10}{'rows':>9}{'P(defect)':>12}{'95% CI':>22}")
        bins: dict[str, list] = defaultdict(list)
        for r in sub:
            bins[_bin_of(abs(r["score_err"]))].append(r)
        binned = {}
        for lo, hi in SCORE_BINS:
            lab = _bin_label(lo, hi)
            g = bins.get(lab)
            if not g or len(g) < MIN_BIN:
                if g:
                    print(f"      {lab:>10}{len(g):>9,}{'':>12}"
                          f"{'below min bin size':>22}")
                continue
            rates = rate_by_episode(g)
            pd_ = sum(r["d"] for r in g) / len(g)
            if rates.size < 2:
                print(f"      {lab:>10}{len(g):>9,}{pd_:>12.4f}"
                      f"{'too few episodes':>22}")
                continue
            nb = min(n_boot, 4000)
            draws = rates[rng.integers(0, rates.size, size=(nb, rates.size))].mean(axis=1)
            s = np.sort(draws)
            lo_ = float(s[int(0.025 * s.size)])
            hi_ = float(s[min(int(0.975 * s.size), s.size - 1)])
            print(f"      {lab:>10}{len(g):>9,}{pd_:>12.4f}"
                  f"   [{lo_:>+7.4f},{hi_:>+7.4f}]")
            binned[lab] = {"n": len(g), "pd": pd_, "lo": lo_, "hi": hi_}

        pooled = _cluster_boot_slope(sub, rng, fit_boot)
        print(f"\n      slope, all 3c rows        {pooled[0]:+.5f} per point"
              f"   [{pooled[1]:+.5f},{pooled[2]:+.5f}]")
        clean = None
        if has_displayed:
            sub_clean = [r for r in sub if r["move_lied"] is False]
            if len(sub_clean) >= MIN_BIN:
                clean = _cluster_boot_slope(sub_clean, rng, fit_boot)
                print(f"      slope, move NOT falsified {clean[0]:+.5f} per point"
                      f"   [{clean[1]:+.5f},{clean[2]:+.5f}]"
                      f"   ({len(sub_clean):,} rows)")
            else:
                print("      slope, move NOT falsified  too few rows")
        else:
            print("      slope, move NOT falsified  SKIPPED - "
                  "displayed_opponent_last absent, so the pooled slope is")
            print("                                 contaminated by the move field.")

        use = clean if clean and clean[0] == clean[0] else pooled
        implied = SCORE_FALSIFICATION * use[0]
        implied_lo = SCORE_FALSIFICATION * use[1]
        implied_hi = SCORE_FALSIFICATION * use[2]
        print(f"      implied {SCORE_FALSIFICATION}-point effect  {implied:+.4f}"
              f"   [{implied_lo:+.4f},{implied_hi:+.4f}]")

        measured = None
        if "3s" in arms and "3" in arms:
            a = episode_rates(load_actions(conn, "3s", opp), 0)
            b = episode_rates(load_actions(conn, "3", opp), 0)
            m = boot_diff(a, b, rng, n_boot)
            measured = m
            print(f"      arm 3s measured effect    {m['diff']:+.4f}   {ci(m)}"
                  f"   (P(D|3s) - P(D|3))")
            agree = ((m["lo"] <= implied <= m["hi"])
                     or (implied_lo <= m["diff"] <= implied_hi))
            print("\n      VERDICT: ", end="")
            if agree:
                print("consistent. A 15-point lie moves defection by about")
                print("      what the 3c dose curve predicts, so the score field IS")
                print("      described by a magnitude response and the move-vs-score")
                print("      ratio is partly a statement about dose. Say 'a 15-point")
                print("      score error', never 'the score field', when quoting 29x.")
            else:
                print("inconsistent. The 3c dose curve does not predict")
                print("      arm 3s's measured effect, so the score response is not a")
                print("      simple magnitude response - it is a threshold, a format")
                print("      effect, or the 3c slope is confounded. Either way the")
                print("      dose defence of '29x' fails and the ratio must be")
                print("      reported as arm-specific.")
        else:
            print("      arm 3s absent - cannot compare to a measured 15-point effect.")
        def _slope(t):
            return None if t is None else {"slope": t[0], "lo": t[1], "hi": t[2]}

        out[opp] = {"bins": binned,
                    "slope_pooled": _slope(pooled),
                    "slope_move_clean": _slope(clean),
                    "implied_15pt": {"est": implied, "lo": implied_lo,
                                     "hi": implied_hi},
                    "arm3s_measured": measured}
    print("\n    The 3c slope inherits analysis/11's confound: |error| is large")
    print("    exactly when the recipient's own score ran unusually, which happens")
    print("    when it was already defecting. Read the slope as an upper bound.")
    return out


# ---------------------------------------------------------------------------
# G - action mass / near-tie
# ---------------------------------------------------------------------------

def part_g(conn, arms, opponents, focus: set[str]) -> dict:
    print(f"\n  {SUB}\n  G. NEAR-TIE / ACTION MASS - how solid is each P(defect)?\n  {SUB}")
    cols = table_columns(conn, "turns")
    if "action_mass_total" not in cols:
        print("    SKIPPED - turns.action_mass_total absent.")
        return {"skipped": "action_mass_total absent"}
    print("    P(defect) is the Cooperate/Defect masses renormalised to sum to 1.")
    print(f"    Below action_mass_total {OFFTASK_GATE} the run calls it off-task. Between")
    print(f"    {OFFTASK_GATE} and {FRAGILE_GATE} it is neither off-task nor solid:")
    print("    the renormalisation is dividing by a small number.")
    print(f"\n    {'arm':<5}{'opp':<6}{'n':>9}{'mass p05':>10}{'p50':>9}{'p95':>9}"
          f"{'|gap| p50':>11}{'<0.10':>8}{'<0.25':>8}")
    out = {}
    for arm in sorted(arms, key=lambda a: (len(a), a)):
        for opp in opponents:
            vals = conn.execute(
                """SELECT action_mass_total m, logit_gap g FROM turns
                   WHERE arm=? AND opponent_policy=?""", (arm, opp)).fetchall()
            if not vals:
                continue
            m = np.array([v["m"] for v in vals if v["m"] is not None], dtype=np.float64)
            g = np.array([abs(v["g"]) for v in vals if v["g"] is not None],
                         dtype=np.float64)
            if m.size == 0:
                print(f"    {arm:<5}{opp:<6}{len(vals):>9,}   all action_mass_total NULL")
                continue
            q = np.percentile(m, [5, 50, 95])
            off = float((m < OFFTASK_GATE).mean())
            frag = float((m < FRAGILE_GATE).mean())
            gm = float(np.median(g)) if g.size else float("nan")
            star = " *" if f"{arm}|{opp}" in focus else ""
            print(f"    {arm:<5}{opp:<6}{m.size:>9,}{q[0]:>10.3f}{q[1]:>9.3f}"
                  f"{q[2]:>9.3f}{gm:>11.3f}{off:>8.3f}{frag:>8.3f}{star}")
            out[f"{arm}|{opp}"] = {
                "n": int(m.size), "mass_p05": float(q[0]), "mass_p50": float(q[1]),
                "mass_p95": float(q[2]), "abs_logit_gap_p50": gm,
                "share_offtask": off, "share_fragile": frag}
    if focus:
        print("\n    * = a cell in this database's largest contrast.")
        worst = max((out[k]["share_fragile"] for k in focus if k in out),
                    default=float("nan"))
        print("    VERDICT: ", end="")
        if worst == worst and worst > 0.10:
            print(f"{worst:.1%} of decisions in the largest-effect cells sit")
            print(f"    below action_mass_total {FRAGILE_GATE}. That effect is partly a")
            print("    statement about a renormalisation of a small number. Report")
            print("    the fragile share beside the estimate, and check the effect")
            print("    survives restricting to solid decisions before quoting it.")
        elif worst == worst:
            print(f"only {worst:.1%} of decisions in the largest-effect")
            print(f"    cells are fragile (mass < {FRAGILE_GATE}). The estimates are not")
            print("    an artefact of near-ties.")
        else:
            print("no focus cells resolved.")
    return out


# ---------------------------------------------------------------------------
# H - 3b round parity and the horizon
# ---------------------------------------------------------------------------

def _parity_and_horizon(eps, rng, n_boot, horizon_window: int = 5):
    """Per-episode paired contrasts: even-odd turns, and last-k minus first-k."""
    par, hor = [], []
    for seq in eps.values():
        e = [d for t, d in seq if t % 2 == 0]
        o = [d for t, d in seq if t % 2 == 1]
        if e and o:
            par.append(sum(e) / len(e) - sum(o) / len(o))
        if not seq:
            continue
        last_t = max(t for t, _ in seq)
        first = [d for t, d in seq if t < horizon_window]
        last = [d for t, d in seq if t > last_t - horizon_window]
        if first and last:
            hor.append(sum(last) / len(last) - sum(first) / len(first))
    return (boot_paired(np.array(par, dtype=np.float64), rng, n_boot),
            boot_paired(np.array(hor, dtype=np.float64), rng, n_boot))


def _detrended_parity(eps, rng, n_boot):
    """Parity effect with any smooth turn trend differenced out locally.

    THE TRAP THIS EXISTS TO AVOID
        A raw even-minus-odd contrast is NOT a parity test. Over turns 0..19 the
        odd turns average one index later than the even ones, so any monotone
        trajectory manufactures a gap of about one turn's slope. Arm 1 and arm
        3b can show the SAME raw gap and mean opposite things when their slopes
        differ - which is exactly what happens here: in exp6_llama_sem_logit
        both show -0.031, but arm 1's slope is 0.0178/turn (gap fully explained)
        and 3b's is 0.0055/turn (gap four times larger than its trend allows).

        Differencing 3b against arm 1's RAW gap does not fix this either, since
        the two arms have different slopes by construction - the container
        effect compresses 3b's whole trajectory.

    THE ESTIMATOR
        For each episode and each odd turn 2k+1 flanked by 2k and 2k+2:

            d_k = y[2k+1] - (y[2k] + y[2k+2]) / 2

        A locally linear trend contributes exactly zero to d_k, whatever its
        slope. A parity effect c contributes c. Averaged within episode, then
        bootstrapped over episodes, so the interval respects the clustering.
    """
    # eps is {episode_id: [(turn, defected)]}, and turns may be sparse, so
    # index by turn rather than by position.
    per_ep = []
    for seq in eps.values():
        y = {t: d for t, d in seq}
        ds = [y[t] - 0.5 * (y[t - 1] + y[t + 1])
              for t in y if t % 2 == 1 and (t - 1) in y and (t + 1) in y]
        if ds:
            per_ep.append(sum(ds) / len(ds))
    if not per_ep:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "n": 0}
    a = np.asarray(per_ep, dtype=np.float64)
    # sign convention matches the raw column: even minus odd
    draws = -a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    draws.sort()
    lo, hi = float(draws[int(0.025 * n_boot)]), float(draws[int(0.975 * n_boot)])
    below = int((draws <= 0).sum()); above = int((draws >= 0).sum())
    p = min(1.0, max(2 * min(below, above) / n_boot, 1.0 / n_boot))
    return {"diff": float(-a.mean()), "lo": lo, "hi": hi, "p": p, "n": int(a.size)}


def part_h(conn, arms, opponents, rng, n_boot) -> dict:
    print(f"\n  {SUB}\n  H. 3b ROUND PARITY - is the placebo leaking state?\n  {SUB}")
    if "3b" not in arms:
        print("    SKIPPED - arm 3b absent from this database.")
        return {"skipped": "arm 3b absent"}
    print("    Arm 3b's block contains 'Round parity: even/odd'. That is one bit")
    print("    of the turn index, and under a known 20-round horizon the turn")
    print("    index is weakly endgame-relevant. If 3b defection tracks parity, or")
    print("    accelerates near the horizon in a way arm 1 does not, the placebo")
    print("    carries usable state and ATE_true is biased toward zero.")
    control = "1" if "1" in arms else None
    if control is None:
        print("    NOTE: arm 1 absent, so the endgame slope has no control. An")
        print("    endgame slope in 3b alone cannot be attributed to the parity")
        print("    line without it.")
    print("\n    'even-odd' is the RAW contrast and is confounded by the turn")
    print("    trend. 'detrended' differences each odd turn against its two")
    print("    neighbours, so a locally linear trend of ANY slope cancels. Read")
    print("    the detrended column; the raw one is shown only for continuity.")
    print(f"\n    {'arm':<5}{'opp':<6}{'even-odd':>11}{'detrended':>11}{'95% CI':>22}"
          f"{'p':>8}{'last5-first5':>14}")
    out = {}
    for arm in [a for a in ("3b", control) if a]:
        for opp in opponents:
            eps = load_actions(conn, arm, opp)
            if not eps:
                continue
            par, hor = _parity_and_horizon(eps, rng, n_boot)
            det = _detrended_parity(eps, rng, n_boot)
            print(f"    {arm:<5}{opp:<6}{fmt(par['diff'], 11)}"
                  f"{fmt(det['diff'], 11)}{ci(det):>22}"
                  f"{fmt_p(det['p']):>8}{fmt(hor['diff'], 14)}")
            out[f"{arm}|{opp}"] = {"parity": par, "horizon": hor,
                                   "parity_detrended": det}

    ctl = "arm 1 control" if control else "no control"
    print(f"\n    P(defect) BY TURN INDEX  (arm 3b; {ctl})")
    series = defaultdict(dict)
    for arm in [a for a in ("3b", control) if a]:
        for r in conn.execute(
            """SELECT turn, AVG(agent_action='D') p FROM turns
               WHERE arm=? GROUP BY turn ORDER BY turn""", (arm,)):
            series[r["turn"]][arm] = r["p"]
    print(f"    {'turn':>5}{'3b':>9}{'arm1':>9}   {'parity':>7}")
    for t in sorted(series):
        a = series[t].get("3b", float("nan"))
        b = series[t].get("1", float("nan"))
        print(f"    {t:>5}{a:>9.4f}{b:>9.4f}   {'even' if t % 2 == 0 else 'odd':>7}")

    par_sig = [k for k, v in out.items() if k.startswith("3b|") and v["parity"]["sig"]]
    hor3b = [v["horizon"] for k, v in out.items() if k.startswith("3b|")]
    hor1 = [v["horizon"] for k, v in out.items() if k.startswith("1|")]

    # A RAW GAP IS NOT A PARITY EFFECT. Test the DETRENDED coefficient, and
    # judge each arm on its own trend rather than against the other arm's raw
    # gap - the two arms have different slopes by construction, so differencing
    # raw gaps swaps one confound for another.
    det_sig = [k for k, v in out.items()
               if k.startswith("3b|") and v["parity_detrended"]["p"] == v["parity_detrended"]["p"]
               and (v["parity_detrended"]["lo"] > 0 or v["parity_detrended"]["hi"] < 0)]
    ctl_sig = [k for k, v in out.items()
               if k.startswith("1|")
               and (v["parity_detrended"]["lo"] > 0 or v["parity_detrended"]["hi"] < 0)]
    print("\n    VERDICT: ", end="")
    if det_sig:
        print(f"THE PLACEBO RESPONDS TO PARITY in {', '.join(det_sig)}.")
        print("    The effect survives local detrending, so it is not the turn")
        print("    trend. Arm 3b is therefore NOT contentless: it carries one bit")
        print("    of the turn index and the model acts on it.")
        if ctl_sig:
            print(f"    CAUTION: arm 1 also shows a detrended parity effect in")
            print(f"    {', '.join(ctl_sig)}, and arm 1 has no parity line. Something")
            print("    other than the block produces an alternation here; do not")
            print("    attribute 3b's effect to the block until that is explained.")
        else:
            print("    Arm 1, which has no parity line, shows no detrended effect -")
            print("    so the block is the source.")
        print("\n    CONSEQUENCE FOR THE PAPER: ATE_true = P(D|3) - P(D|3b) compares")
        print("    the treatment against a control that is itself partly active.")
        print("    Report the detrended parity coefficient in Methods, state that")
        print("    ATE_true is conservative, and do not describe 3b as")
        print("    'non-diagnostic' without this qualification. The pre-registered")
        print("    null is not overturned by it - a leaky placebo biases toward")
        print("    finding nothing, which is the direction already reported.")
    else:
        print("no detrended parity effect in any 3b cell. The raw")
        print("    even-odd column is the turn trend and nothing else. The")
        print("    placebo's one real bit of state does not move play - which is")
        print("    what a placebo has to demonstrate rather than assert.")

    if hor3b and hor1:
        d3 = float(np.mean([h["diff"] for h in hor3b if h["diff"] == h["diff"]]))
        d1 = float(np.mean([h["diff"] for h in hor1 if h["diff"] == h["diff"]]))
        print(f"\n    Endgame slope: 3b {d3:+.4f}, arm 1 {d1:+.4f}.")
        # Raw slopes are not comparable when the baselines differ several-fold:
        # the block compresses the whole trajectory (the container effect), so a
        # smaller absolute slope under 3b is expected even if nothing leaks.
        b3 = float(np.mean([v["late_rate"] for k, v in out.items()
                            if k.startswith("3b|") and "late_rate" in v])) \
            if any("late_rate" in v for v in out.values()) else float("nan")
        print("    Raw slopes are NOT comparable here: arm 3b's defection level is")
        print("    a fraction of arm 1's (the container effect), so its absolute")
        print("    slope is mechanically smaller. A smaller endgame slope under 3b")
        print("    is the container effect compressing the trajectory, not evidence")
        print("    that the parity line is doing endgame work. Compare on the odds")
        print("    scale before drawing any conclusion from this row.")
        del b3
    elif hor3b:
        d3 = float(np.mean([h["diff"] for h in hor3b if h["diff"] == h["diff"]]))
        print(f"\n    Endgame slope in 3b: {d3:+.4f}, uncontrolled (no arm 1).")
    return out


# ---------------------------------------------------------------------------
# per-database driver
# ---------------------------------------------------------------------------

def run_part(name: str, fn, *a, **kw):
    """Every part is isolated. A missing column or an empty arm must degrade to
    a printed line, never to a traceback that costs the other seven analyses."""
    try:
        return fn(*a, **kw)
    except Exception as exc:                            # pragma: no cover
        print(f"\n  PART {name} FAILED: {type(exc).__name__}: {exc}")
        print("  " + "\n  ".join(traceback.format_exc().strip().splitlines()[-3:]))
        print(f"  Continuing; part {name} is missing from the JSON output.")
        return {"error": f"{type(exc).__name__}: {exc}"}


def analyse(path: Path, rng, n_boot: int, pools: dict) -> dict:
    conn = connect(path)
    ident = identity(conn, path)
    opponents = sorted({r[0] for r in conn.execute(
        "SELECT DISTINCT opponent_policy FROM turns")})
    arms = {r[0] for r in conn.execute("SELECT DISTINCT arm FROM turns")}
    cols = table_columns(conn, "turns")
    has_displayed = "displayed_opponent_last" in cols

    print(f"\n{RULE}\n{ident['group']}   model {ident['model']}   "
          f"readout {ident['readout']}\n{RULE}")
    print(f"  arms {sorted(arms, key=lambda a: (len(a), a))}   "
          f"opponents {opponents}")
    if not has_displayed:
        print("  NOTE: turns.displayed_opponent_last is absent (pre-exp6 schema).")
        print("  Parts D(i), D(ii), D(iii) and E's move-clean slope are skipped;")
        print("  which rows were falsified was never recorded for this run.")

    rec: dict = {"identity": ident, "arms": sorted(arms), "opponents": opponents,
                 "has_displayed_opponent_last": has_displayed}

    rec["A_revealed"] = run_part("A", part_a, conn, arms, opponents, rng, n_boot,
                                 pools, ident)
    rec["C_arm1_cpr"] = run_part("C", part_c, conn, arms, opponents, ident)
    rec["I_field_cpr"] = run_part("I", part_i, conn, arms, opponents, ident)
    rec["D_overshoot"] = run_part("D", part_d, conn, arms, opponents, rng,
                                  n_boot, has_displayed)
    rec["E_dose"] = run_part("E", part_e, conn, arms, opponents, rng, n_boot,
                             has_displayed)

    # The contrast family feeds part F (multiplicity), which is cross-database
    # and therefore runs in main. Estimates are computed here, once.
    #
    # run_part returns {"error": ...} if the family could not be estimated, and
    # a bare `fam.values()` on that would index a string and take the whole
    # sweep down - which is precisely the failure mode run_part exists to
    # prevent. Guard, do not assume.
    fam = run_part("F-estimates", contrast_family, conn, arms, opponents,
                   rng, n_boot)
    focus: set[str] = set()
    if isinstance(fam, dict) and fam and "error" not in fam:
        best = max((v for v in fam.values()
                    if v["quotable"]["diff"] == v["quotable"]["diff"]),
                   key=lambda v: abs(v["quotable"]["diff"]), default=None)
        if best:
            focus = {f"{best['x']}|{best['opp']}", f"{best['y']}|{best['opp']}"}
        for key, v in fam.items():
            pools[("F", ident["group"], key)] = v
    rec["F_contrasts"] = fam

    rec["G_action_mass"] = run_part("G", part_g, conn, arms, opponents, focus)
    rec["H_parity"] = run_part("H", part_h, conn, arms, opponents, rng, n_boot)

    conn.close()
    return rec


def contrast_family(conn, arms, opponents, rng, n_boot) -> dict:
    """The exp6 contrast family, both turn-0 conventions, per opponent.

    Printed here so the multiplicity table in part F has visible inputs; the
    correction itself is cross-database and runs once at the end.
    """
    print(f"\n  {SUB}\n  F(inputs). CONTRAST FAMILY  (episode-level, "
          f"{n_boot:,} resamples)\n  {SUB}")
    print("    'quotable' drops turn 0 only where an arm is byte-identical to")
    print(f"    arm 3 there ({', '.join(sorted(DEGENERATE_AT_TURN0))}). Arm 3s is")
    print("    falsified at turn 0 too, so dropping it would discard real")
    print("    manipulation - analysis/13 drops it from every contrast.")
    print(f"\n    {'contrast':<24}{'opp':<6}{'incl t0':>10}{'quotable':>10}"
          f"{'95% CI':>21}{'p':>8}{'t0':>4}")
    out = {}
    cache: dict[tuple[str, str], dict] = {}

    def eps(arm, opp):
        if (arm, opp) not in cache:
            cache[(arm, opp)] = load_actions(conn, arm, opp)
        return cache[(arm, opp)]

    for label, x, y in CONTRAST_FAMILY:
        if x not in arms or y not in arms:
            continue
        min_turn = 1 if ({x, y} & DEGENERATE_AT_TURN0) else 0
        for opp in opponents:
            ex, ey = eps(x, opp), eps(y, opp)
            if not ex or not ey:
                continue
            incl = boot_diff(episode_rates(ex, 0), episode_rates(ey, 0), rng, n_boot)
            quot = incl if min_turn == 0 else boot_diff(
                episode_rates(ex, min_turn), episode_rates(ey, min_turn), rng, n_boot)
            star = " *" if quot["sig"] else "  "
            print(f"    {label:<24}{opp:<6}{fmt(incl['diff'], 10)}"
                  f"{fmt(quot['diff'], 10)}   {ci(quot)}{fmt_p(quot['p']):>8}"
                  f"{min_turn:>4}{star}")
            out[f"{label.split()[0]}|{opp}"] = {
                "label": label.split()[0], "x": x, "y": y, "opp": opp,
                "min_turn": min_turn, "incl_t0": incl, "quotable": quot}
    print("\n    * = raw bootstrap CI excludes zero, BEFORE any multiplicity")
    print("    correction. Part F decides which of these survive.")
    return out


# ---------------------------------------------------------------------------
# B - model heterogeneity  (cross-database)
# ---------------------------------------------------------------------------

def part_b(pools: dict, rng, n_boot: int) -> dict:
    print(f"\n{RULE}\nB. MODEL HETEROGENEITY  (CLAIMS.md G8, 'Untested')\n{RULE}")
    entries = {k[1]: v for k, v in pools.items() if k[0] == "A"}
    if len(entries) < 2:
        print("  SKIPPED - needs at least two databases with arms 3 and 3b.")
        print(f"  Found {len(entries)}. Run without --db to sweep the glob.")
        return {"skipped": "fewer than two comparable databases"}

    print("  CLAIMS.md G8: 'ATE_true vs ALLC is +0.074 / +0.044 / +0.029 across")
    print("  llama / qwen / mistral. Untested. A bootstrap difference-of-")
    print("  differences decides whether the paper reports a pooled effect or")
    print("  three case studies. Run before drafting Results.' This is that run.")

    out: dict = {}
    for stratum in ("all", "revealed"):
        for opp in ("allc", "tft"):
            groups = []
            for g, v in sorted(entries.items()):
                r = v["strata"].get(stratum, {}).get(opp)
                if r and r["diff"] == r["diff"]:
                    groups.append((g, r))
            if len(groups) < 2:
                continue
            print(f"\n  {SUB}\n  stratum {stratum.upper():<9} opponent {opp}\n  {SUB}")
            print(f"    {'group':<28}{'ATE_true':>11}{'95% CI':>22}{'p':>8}")
            for g, r in groups:
                print(f"    {g:<28}{fmt(r['diff'], 11)}{ci(r):>22}{fmt_p(r['p']):>8}")

            # pairwise difference-of-differences, four independent samples
            print(f"\n    {'pair':<48}{'DoD':>11}{'95% CI':>22}{'p':>8}")
            pair_out = {}
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    (ga, ra), (gb, rb) = groups[i], groups[j]
                    a = _rates_of(entries[ga], stratum, opp)
                    b = _rates_of(entries[gb], stratum, opp)
                    if a is None or b is None:
                        continue
                    d1 = boot_deltas(a[0], a[1], rng, n_boot)
                    d2 = boot_deltas(b[0], b[1], rng, n_boot)
                    if d1.size == 0 or d2.size == 0:
                        continue
                    dod = ra["diff"] - rb["diff"]
                    r_ = summarise(dod, d1 - d2, d1.size, d2.size)
                    name = f"{ga} - {gb}"
                    print(f"    {name:<48}{fmt(r_['diff'], 11)}{ci(r_):>22}"
                          f"{fmt_p(r_['p']):>8}")
                    pair_out[name] = r_

            # joint test: bootstrap the heterogeneity statistic under H0
            draws = []
            obs = []
            for g, r in groups:
                a = _rates_of(entries[g], stratum, opp)
                if a is None:
                    continue
                d = boot_deltas(a[0], a[1], rng, n_boot)
                if d.size == 0:
                    continue
                draws.append(d)
                obs.append(r["diff"])
            joint = {"q": float("nan"), "p": float("nan"), "k": len(obs)}
            if len(obs) >= 2:
                o = np.array(obs)
                q_obs = float(((o - o.mean()) ** 2).sum())
                D = np.vstack(draws)                       # k x n_boot
                C = D - np.array(obs)[:, None]             # centred at H0
                C = C - C.mean(axis=0, keepdims=True)
                q_null = (C ** 2).sum(axis=0)
                p = float((q_null >= q_obs).mean())
                joint = {"q": q_obs, "p": max(p, 1.0 / D.shape[1]), "k": len(obs)}
                print(f"\n    joint heterogeneity  Q = sum (ATE_m - mean ATE)^2 = "
                      f"{q_obs:.6f}")
                print(f"    bootstrap p (H0: all models share one ATE) = "
                      f"{fmt_p(joint['p'])}")
                spread = float(o.max() - o.min())
                print(f"    spread max-min = {spread:.4f}")
                print("\n    VERDICT: ", end="")
                if joint["p"] < 0.05:
                    print("HETEROGENEOUS. The models do not share one")
                    print("    treatment effect. THE PAPER MUST REPORT THREE CASE")
                    print("    STUDIES, NOT A POOLED EFFECT. Every sentence of the form")
                    print("    'models do X' must name the model or be rewritten.")
                else:
                    print("no detectable heterogeneity. A pooled effect is")
                    print("    defensible for this stratum and opponent - but state the")
                    print("    spread, because 'not distinguishable at n=3 models' is")
                    print("    not the same as 'the same'.")
            out[f"{stratum}|{opp}"] = {"per_group": {g: r for g, r in groups},
                                       "pairwise": pair_out, "joint": joint}
    return out


def _rates_of(entry: dict, stratum: str, opp: str):
    """Rate vectors are not kept in the JSON payload (too large); part B needs
    them, so they are carried in the pool alongside the summaries."""
    return entry.get("rates", {}).get((stratum, opp))


# ---------------------------------------------------------------------------
# F - multiplicity  (cross-database)
# ---------------------------------------------------------------------------

def part_f(pools: dict) -> dict:
    print(f"\n{RULE}\nF. MULTIPLICITY across the exp6 contrast family\n{RULE}")
    items = [(k[1], k[2], v) for k, v in pools.items() if k[0] == "F"]
    if not items:
        print("  SKIPPED - no contrasts were estimated.")
        return {"skipped": "no contrasts"}
    items.sort(key=lambda t: (t[0], t[1]))

    pvals = [v["quotable"]["p"] for _, _, v in items]
    hb = holm(pvals)
    bh = benjamini_hochberg(pvals)

    print(f"  family size {len(items)}  (every contrast x opponent x group)")
    print(f"\n  {'group':<26}{'contrast':<16}{'opp':<6}{'effect':>10}"
          f"{'raw p':>8}{'Holm':>8}{'BH':>8}  survives")
    survivors = []
    for (g, k, v), h, b in zip(items, hb, bh):
        label, opp = k.split("|")
        eff = v["quotable"]["diff"]
        tag = []
        if h < 0.05:
            tag.append("Holm")
        if b < 0.05:
            tag.append("BH")
        if tag:
            survivors.append((g, k))
        print(f"  {g:<26}{label:<16}{opp:<6}{fmt(eff, 10)}"
              f"{fmt_p(v['quotable']['p']):>8}{fmt_p(h):>8}{fmt_p(b):>8}"
              f"  {'+'.join(tag) if tag else 'neither'}")

    raw_sig = sum(1 for _, _, v in items if v["quotable"]["sig"])
    print(f"\n  raw CI excludes zero      {raw_sig} of {len(items)}")
    print(f"  survives Holm (FWER .05)  {sum(1 for h in hb if h < 0.05)}")
    print(f"  survives BH   (FDR .05)   {sum(1 for b in bh if b < 0.05)}")

    # --- the two sentences in CLAIMS.md that depend on this ------------------
    print(f"\n  {SUB}\n  FLAGGED CLAIMS\n  {SUB}")
    score = [(g, k, v) for g, k, v in items if k.startswith(SCORE_CONTRAST)]
    if not score:
        print("\n  1. \"four of six score contrasts exclude zero\" (CLAIMS.md C5)")
        print("     NOT CHECKED - no 3 - 3s contrast in this sweep. That claim is")
        print("     about exp6; run this script over the exp6 databases to test it.")
    if score:
        sp = [v["quotable"]["p"] for _, _, v in score]
        sh, sb = holm(sp), benjamini_hochberg(sp)
        raw = sum(1 for _, _, v in score if v["quotable"]["sig"])
        print("\n  1. \"four of six score contrasts exclude zero\" (CLAIMS.md C5)")
        print(f"     Family of {len(score)} score contrasts, corrected WITHIN that family:")
        print(f"     {'group':<26}{'opp':<6}{'effect':>10}{'raw p':>8}"
              f"{'Holm':>8}{'BH':>8}")
        n_h = n_b = 0
        for (g, k, v), h, b in zip(score, sh, sb):
            opp = k.split("|")[1]
            n_h += h < 0.05
            n_b += b < 0.05
            print(f"     {g:<26}{opp:<6}{fmt(v['quotable']['diff'], 10)}"
                  f"{fmt_p(v['quotable']['p']):>8}{fmt_p(h):>8}{fmt_p(b):>8}")
        print(f"\n     raw {raw} of {len(score)}   Holm {n_h} of {len(score)}   "
              f"BH {n_b} of {len(score)}")
        print("     VERDICT: ", end="")
        if n_h < raw:
            print(f"the sentence must be rewritten. {raw} contrasts")
            print(f"     exclude zero uncorrected; {n_h} survive Holm within their own")
            print("     family. Write the corrected count, or write 'uncorrected' in")
            print("     the sentence. Four marginal findings in a family of six is")
            print("     exactly the pattern multiplicity control exists to catch.")
        else:
            print(f"the sentence survives. All {raw} contrasts that")
            print("     exclude zero also survive Holm within the score family, so")
            print("     'the score effect is not a clean null' stands as written.")

    tiny = [(g, k, v) for g, k, v in items
            if abs(v["quotable"]["diff"]) < 1e-3
            and abs(v["quotable"]["hi"] - v["quotable"]["lo"]) < 1e-3]
    print("\n  2. Effects at the LOGIT floor (|effect| < 0.001, CI width < 0.001)")
    if not tiny:
        print("     none in this sweep.")
    else:
        for g, k, v in tiny:
            q = v["quotable"]
            print(f"     {g:<26}{k:<22}{fmt(q['diff'], 10)}   {ci(q)}")
        print("     VERDICT: report these as a FLOOR, never as an effect. mistral's")
        print("     +0.0001 [-0.0002,+0.0003] comes from a cell whose defection rate")
        print("     is itself ~0.0001; the interval is narrow because there is no")
        print("     variance to estimate, not because the estimate is precise. A")
        print("     significance verdict on it is a statement about arithmetic")
        print("     precision. CLAIMS.md already calls mistral_sem_logit degenerate")
        print("     rather than excluded - apply that word here too.")

    print("\n  Holm controls the probability of ANY false claim and is the right")
    print("  correction for 'which individual sentences may be written'. BH")
    print("  controls the expected share of false claims among those made and is")
    print("  the right one for 'the move field dominates in 6 of 6 cells'. The")
    print("  paper makes both kinds of claim, so both are reported.")
    return {"family_size": len(items), "raw_significant": raw_sig,
            "holm_significant": int(sum(1 for h in hb if h < 0.05)),
            "bh_significant": int(sum(1 for b in bh if b < 0.05)),
            "items": [{"group": g, "key": k, "effect": v["quotable"]["diff"],
                       "p_raw": v["quotable"]["p"], "p_holm": h, "p_bh": b}
                      for (g, k, v), h, b in zip(items, hb, bh)],
            "floor_effects": [{"group": g, "key": k} for g, k, _ in tiny]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reviewer responses computed from committed databases.")
    ap.add_argument("--db", help="single database; default is every exp6_*.sqlite")
    ap.add_argument("--glob", default="exp6_*.sqlite")
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--out", default="REVIEWER_RESPONSES.json")
    args = ap.parse_args()

    paths = [Path(args.db)] if args.db else discover(args.glob)
    rng = np.random.default_rng(SEED)

    print(f"\n{RULE}\nANALYSIS 14 - REVIEWER RESPONSES\n{RULE}")
    print(f"  databases   {len(paths)}")
    print(f"  bootstrap   {args.bootstrap:,} resamples, seed {SEED}")
    print("  unit        EPISODE (independently seeded => i.i.d.)")
    print("  A revealed-opponent stratification   E score dose-response")
    print("  B model heterogeneity                F multiplicity")
    print("  C arm-1 CPR denominator              G near-tie / action mass")
    print("  D the 3c overshoot                   H 3b round parity")

    pools: dict = {}
    payload = []
    for p in paths:
        rec = analyse(p, rng, args.bootstrap, pools)
        payload.append(rec)

    b = run_part("B", part_b, pools, rng, args.bootstrap)
    f = run_part("F", part_f, pools)

    print(f"\n{RULE}\nSUMMARY - what a referee should be told\n{RULE}")
    print(f"\n  {'group':<26}{'signflip all':>14}{'signflip revealed':>19}"
          f"{'min base rate':>15}")
    for rec in payload:
        a = rec.get("A_revealed", {})
        if "signflip" not in a:
            continue
        base = a.get("base_rate", {})
        mn = min((v["arm3_base_rate"] for v in base.values()), default=float("nan"))
        print(f"  {rec['identity']['group']:<26}{a['signflip']['all']:>14}"
              f"{a['signflip']['revealed']:>19}{mn:>15.1%}")

    print("\n  ARM-1 CPR (the denominator)")
    print(f"  {'group':<26}{'model':<10}{'readout':<12}{'arm1 CPR':>10}{'drift':>9}")
    for rec in payload:
        c = rec.get("C_arm1_cpr", {})
        if "arm1_overall" not in c:
            continue
        i = rec["identity"]
        print(f"  {i['group']:<26}{i['model']:<10}{i['readout']:<12}"
              f"{c['arm1_overall']:>10.3f}{fmt(c['arm1_turn_drift'], 9)}")

    changed = [r["identity"]["group"] for r in payload
               if r.get("A_revealed", {}).get("signflip", {}).get("revealed")
               == "SUPPORTED"
               and r.get("A_revealed", {}).get("signflip", {}).get("all")
               != "SUPPORTED"]
    if changed:
        print(f"\n  *** THE CENTRAL CLAIM CHANGES IN: {', '.join(changed)}")
        print("  The pre-registered sign-flip is supported once restricted to")
        print("  turns where the opponent's type is knowable. CLAIMS.md A1 must")
        print("  be rewritten before anything else in the paper is drafted.")
    else:
        print("\n  The revealed-opponent stratification does not rescue the")
        print("  pre-registered hypothesis in any group. That strengthens the")
        print("  rejection: it survives the best available defence.")

    out = {"seed": SEED, "bootstrap_resamples": args.bootstrap,
           "databases": [str(p) for p in paths],
           "per_database": payload,
           "B_model_heterogeneity": b,
           "F_multiplicity": f}
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  written  {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
