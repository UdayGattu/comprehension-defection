#!/usr/bin/env python3
"""Build paper/FACTS.json - the only numeric source the paper is allowed to cite.

EVERY entry carries {value, source_file, how_computed}. A number reaches this
file by exactly one of two routes:

  (a) it was READ from a machine-generated JSON artefact in the repository
      root (EXP6_FIELDS.json, EXP7_FIELDS.json, EXP7_REVIEWER.json,
      REVIEWER_RESPONSES.json, REVIEWER_RESPONSES_ALL.json, ep_*.json,
      exp*.json), with the exact JSON path recorded; or

  (b) it was COMPUTED here, by SQL or by numpy, against a committed .sqlite,
      with the query or estimator recorded in how_computed.

Nothing is typed from prose. CLAIMS.md / EXPERIMENTS.md / *.txt / *.md are
NEVER read by this script - narrative files are not numeric sources.

Run:
    python paper/extract_facts.py
    python paper/extract_facts.py --quick     # skip probe-text passes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.request import pathname2url

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "FACTS.json"

sys.path.insert(0, str(ROOT))
from cdx.probe import AnswerType, normalise  # noqa: E402  (production scorer)

SEED = 20260811          # same seed the run-time analyses used
N_BOOT = 10_000
SCORE_FALSIFICATION = 15  # cdx/scaffold.py, arm 3s; used only as a candidate
                          # offset to TEST against, never as a reported number

OPPONENTS = ("allc", "tft")

# ---------------------------------------------------------------------------
# fact table
# ---------------------------------------------------------------------------

FACTS: "OrderedDict[str, dict]" = OrderedDict()
_PROBLEMS: list[str] = []


def put(key: str, value, source_file: str, how: str) -> None:
    if key in FACTS:
        raise KeyError(f"duplicate FACTS key: {key}")
    FACTS[key] = {"value": value, "source_file": source_file,
                  "how_computed": how}


CACHE = PAPER / ".facts_cache.json"


def _signature() -> str:
    """Fingerprint of every input file, so a stale cache can never be used."""
    parts = []
    for p in sorted(list(ROOT.glob("exp*.sqlite")) + list(ROOT.glob("*.json"))):
        if p.parent != ROOT:
            continue
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


def run_stage(name: str, fn, *args, cache: bool = True):
    """Run one stage, or replay it from the cache.

    The databases total 17 GiB and a full scan does not fit inside a single
    tool-call budget on the machine this was developed on. Each stage therefore
    checkpoints the facts it produced, keyed by a fingerprint of every input
    file, so an interrupted build resumes instead of restarting. Delete
    paper/.facts_cache.json (or pass --no-cache) to force a cold rebuild.
    """
    blob = {}
    if cache and CACHE.exists():
        try:
            blob = json.loads(CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blob = {}
        if blob.get("signature") != _signature():
            blob = {}
    if cache and name in (blob.get("stages") or {}):
        st = blob["stages"][name]
        for k, v in st["facts"].items():
            FACTS[k] = v
        for m in st.get("problems", []):
            _PROBLEMS.append(m)
        print(f"\n[{name}] replayed from cache "
              f"({len(st['facts'])} facts)")
        return
    before = set(FACTS)
    before_p = len(_PROBLEMS)
    fn(*args)
    if cache:
        blob.setdefault("signature", _signature())
        blob.setdefault("stages", {})
        blob["stages"][name] = {
            "facts": {k: FACTS[k] for k in FACTS if k not in before},
            "problems": _PROBLEMS[before_p:]}
        CACHE.write_text(json.dumps(blob), encoding="utf-8")


def note(msg: str) -> None:
    _PROBLEMS.append(msg)
    print(f"    ! {msg}")


def jnum(x):
    """JSON-safe scalar."""
    if x is None:
        return None
    if isinstance(x, (bool, str)):
        return x
    f = float(x)
    if f != f:
        return None
    return f


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def ro_uri(p: Path) -> str:
    """Read-only + immutable: WAL-independent and cannot mutate the evidence."""
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def connect(p: Path) -> sqlite3.Connection:
    c = sqlite3.connect(ro_uri(p), uri=True)
    c.row_factory = sqlite3.Row
    return c


def load_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def databases() -> list[Path]:
    out = []
    for p in sorted(ROOT.glob("exp*.sqlite")):
        if p.stat().st_size == 0:
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# bootstrap helpers (episode-clustered, matching analysis/14's estimator)
# ---------------------------------------------------------------------------

def boot_mean_ci(a: np.ndarray, rng, n_boot=N_BOOT):
    """Percentile bootstrap of the mean of an episode-level statistic."""
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    draws = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    draws.sort()
    return {"mean": float(a.mean()),
            "lo": float(draws[int(0.025 * n_boot)]),
            "hi": float(draws[int(0.975 * n_boot)]),
            "n": int(a.size)}


# ===========================================================================
# 1-5. ONE PASS PER DATABASE
#
# The turns table is scanned exactly once per database and every turn-level
# quantity is derived in Python from that single pull. Doing it as five
# separate GROUP BY / self-join queries meant ten full scans of a
# multi-hundred-megabyte table per database and was hours slower for
# arithmetically identical answers.
# ===========================================================================

TURN_COLS = ["run_id", "episode_id", "arm", "model_id", "readout_mode",
             "opponent_policy", "turn", "agent_action", "opponent_action",
             "action_mass_total", "cpr_score", "cpr_own_score",
             "cpr_opponent_last", "cpr_rounds_played", "scaffold_tokens",
             "displayed_opponent_last"]

OFFTASK_GATE = 0.10      # cdx / analysis/13: below this the decision is noise


class Acc:
    """Streaming (sum, count) accumulator."""

    __slots__ = ("s", "n")

    def __init__(self):
        self.s = 0.0
        self.n = 0

    def add(self, x):
        if x is None:
            return
        self.s += float(x)
        self.n += 1

    @property
    def mean(self):
        return self.s / self.n if self.n else None


def scan_databases(dbs: list[Path]) -> dict:
    print("\n[1-5] single-pass scan: corpus, CPR by field, defection, "
          "falsification, probe baseline")
    per_group = {}
    tot_ep = tot_turn = 0
    models: dict[str, str] = {}
    parity: dict[str, int] = {}

    for i, p in enumerate(dbs, 1):
        g = p.stem
        c = connect(p)
        cols = {r[1] for r in c.execute("PRAGMA table_info(turns)")}
        sel = [x for x in TURN_COLS if x in cols]
        idx = {name: j for j, name in enumerate(sel)}

        ep_n = c.execute("SELECT COUNT(*) n FROM episodes").fetchone()["n"]
        hz = [r[0] for r in c.execute("SELECT DISTINCT horizon FROM episodes")]
        eps_per_cell = sorted({r[0] for r in c.execute(
            "SELECT COUNT(*) FROM episodes GROUP BY arm, opponent_policy")})
        m = c.execute("SELECT run_id, started_at, finished_at, model_id, "
                      "gpu_name, gpu_count, git_commit FROM run_meta").fetchone()

        # Stream in PRIMARY KEY order so the previous turn of the same episode
        # is always the previous row. That makes the turn-1 lookups O(1) memory
        # instead of holding a 400k-entry index of the whole table, which this
        # 3 GiB machine will not survive across 43 databases.
        key_cols = ["run_id", "episode_id", "arm", "model_id", "readout_mode",
                    "opponent_policy", "turn"]
        cur = c.execute(f"SELECT {', '.join(sel)} FROM turns "
                        f"ORDER BY {', '.join(key_cols)}")

        arms, opps, tokens = set(), set(), set()
        defect = defaultdict(Acc)
        offtask = defaultdict(Acc)
        cpr = defaultdict(lambda: defaultdict(Acc))       # (arm,opp) -> field
        cpr_i0 = defaultdict(Acc)                          # turn-0 inclusive
        base = defaultdict(Acc)                            # P(truth == C)
        fals = defaultdict(lambda: [0, 0])
        n_turns = 0
        prev_key = None
        prev_opp_action = None

        for r in cur:
            n_turns += 1
            arm = r[idx["arm"]]
            opp = r[idx["opponent_policy"]]
            t = r[idx["turn"]]
            arms.add(arm)
            opps.add(opp)
            if "scaffold_tokens" in idx and r[idx["scaffold_tokens"]] is not None:
                tokens.add(r[idx["scaffold_tokens"]])
            k = (arm, opp)
            # the previous row is turn t-1 of this same episode-cell iff its
            # key matches on everything but the turn index
            this_key = tuple(r[idx[c_]] for c_ in key_cols)
            pa = (prev_opp_action
                  if prev_key is not None and prev_key[:-1] == this_key[:-1]
                  and prev_key[-1] == t - 1 else None)
            prev_key, prev_opp_action = this_key, r[idx["opponent_action"]]

            defect[k].add(1.0 if r[idx["agent_action"]] == "D" else 0.0)
            if "action_mass_total" in idx and r[idx["action_mass_total"]] is not None:
                offtask[k].add(1.0 if r[idx["action_mass_total"]] < OFFTASK_GATE
                               else 0.0)

            scored = "cpr_score" in idx and r[idx["cpr_score"]] is not None
            if scored:
                cpr_i0[k].add(r[idx["cpr_score"]])
                if t > 0:
                    a = cpr[k]
                    a["cpr"].add(r[idx["cpr_score"]])
                    for f in ("cpr_own_score", "cpr_opponent_last",
                              "cpr_rounds_played"):
                        if f in idx:
                            a[f].add(r[idx[f]])
                    if pa is not None:
                        base[k].add(1.0 if pa == "C" else 0.0)

            if "displayed_opponent_last" in idx and \
                    r[idx["displayed_opponent_last"]] is not None and \
                    pa is not None:
                fals[k][1] += 1
                fals[k][0] += int(r[idx["displayed_opponent_last"]] != pa)

        c.close()

        # ---------------- corpus ----------------
        wall = None
        if m and m["started_at"] and m["finished_at"]:
            try:
                wall = (datetime.fromisoformat(m["finished_at"]) -
                        datetime.fromisoformat(m["started_at"])).total_seconds()
            except ValueError:
                wall = None
        rec = {"n_episodes": ep_n, "n_turns": n_turns,
               "n_cells": len(defect), "arms": sorted(arms),
               "opponents": sorted(opps), "episodes_per_cell": eps_per_cell,
               "horizon": hz[0] if len(hz) == 1 else sorted(hz),
               "model_id": m["model_id"] if m else None,
               "gpu_name": m["gpu_name"] if m else None,
               "gpu_count": m["gpu_count"] if m else None,
               "git_commit": m["git_commit"] if m else None,
               "wall_clock_s": wall,
               "parity_target_tokens": (sorted(tokens)[0] if len(tokens) == 1
                                        else sorted(tokens) or None)}
        per_group[g] = rec
        tot_ep += ep_n
        tot_turn += n_turns
        if m and m["model_id"]:
            models[g] = m["model_id"]
            if len(tokens) == 1:
                parity[m["model_id"]] = sorted(tokens)[0]
        put(f"corpus.{g}", rec, p.name,
            "single scan of turns plus COUNT/DISTINCT over episodes and "
            "run_meta: episode count, turn count, distinct arms, distinct "
            "opponents, distinct scaffold_tokens, finished_at - started_at")

        if len(eps_per_cell) == 1 and isinstance(rec["horizon"], int):
            expect = (len(arms) * len(opps) * eps_per_cell[0] * rec["horizon"])
            if expect != n_turns:
                note(f"{g}: turn count {n_turns} != arms*opps*N*horizon "
                     f"= {expect}")
        else:
            note(f"{g}: ragged cell sizes {eps_per_cell} or horizons "
                 f"{rec['horizon']}; the turn-count identity does not apply")

        # ---------------- CPR by field ----------------
        for k, a in sorted(cpr.items()):
            arm, opp = k
            put(f"cpr.{g}.{arm}|{opp}", {
                "cpr_all_or_nothing": a["cpr"].mean,
                "own_score": a["cpr_own_score"].mean,
                "opponent_last": a["cpr_opponent_last"].mean,
                "rounds_played": a["cpr_rounds_played"].mean,
                "n_probe_turns": a["cpr"].n,
                "cpr_incl_turn0": cpr_i0[k].mean,
                "n_probe_turns_incl_turn0": cpr_i0[k].n}, p.name,
                "mean of turns.cpr_score / cpr_own_score / cpr_opponent_last / "
                "cpr_rounds_played over rows with cpr_score NOT NULL AND "
                "turn > 0, grouped by (arm, opponent_policy). CPR is "
                "all-or-nothing over the three probes. cpr_incl_turn0 repeats "
                "it without the turn>0 filter, so the turn-0 dilution is "
                "visible rather than assumed.")

        # ---------------- constant-answer baseline ----------------
        for k, a in sorted(base.items()):
            arm, opp = k
            if not a.n:
                continue
            pC = a.mean
            put(f"probe_baseline.{g}.{arm}|{opp}", {
                "p_true_is_cooperate": pC,
                "constant_answer_accuracy": max(pC, 1.0 - pC),
                "n_probe_turns": a.n}, p.name,
                "for every probe turn t>0, the true opponent_last is the "
                "opponent's action at t-1, looked up in the same episode. "
                "p_true_is_cooperate is its mean; a model that ignores the "
                "prompt and always answers the majority label would score "
                "max(pC, 1-pC) on this binary probe.")

        # ---------------- defection and opponent spread ----------------
        by_arm = defaultdict(dict)
        for k, a in sorted(defect.items()):
            arm, opp = k
            put(f"defect.{g}.{arm}|{opp}", {
                "defect_rate": a.mean, "n_turns": a.n,
                "offtask_rate": offtask[k].mean}, p.name,
                "mean of (agent_action == 'D') over all turns in the cell; "
                "offtask_rate is the mean of (action_mass_total < 0.10)")
            by_arm[arm][opp] = a.mean
        for arm, d in sorted(by_arm.items()):
            if "allc" in d and "tft" in d:
                put(f"opponent_spread.{g}.{arm}", {
                    "allc": d["allc"], "tft": d["tft"],
                    "spread": d["tft"] - d["allc"],
                    "abs_spread": abs(d["tft"] - d["allc"])}, p.name,
                    "defection rate vs TFT minus defection rate vs ALLC within "
                    "one arm: how much the policy depends on who it is playing")

        # ---------------- falsification ----------------
        for k, (lied, n) in sorted(fals.items()):
            arm, opp = k
            if not n:
                continue
            put(f"falsification.{g}.{arm}|{opp}", {
                "lied": lied, "n": n, "rate": lied / n}, p.name,
                "over rows where displayed_opponent_last IS NOT NULL, the "
                "fraction where it differs from the opponent's action on the "
                "PREVIOUS turn of the same episode. The truth is recovered "
                "from the turn record itself, never from the writer's own "
                "notion of what it displayed.")

        if i % 8 == 0 or i == len(dbs):
            print(f"    {i}/{len(dbs)} databases scanned "
                  f"({tot_turn:,} turns so far)")

    put("corpus.total.n_databases", len(dbs), "exp*.sqlite",
        "count of non-empty exp*.sqlite in the repository root")
    put("corpus.total.n_episodes", tot_ep, "exp*.sqlite",
        "sum of COUNT(*) FROM episodes over all databases")
    put("corpus.total.n_turns", tot_turn, "exp*.sqlite",
        "sum of the per-database turn counts")
    put("corpus.models", sorted(set(models.values())), "exp*.sqlite",
        "DISTINCT run_meta.model_id over all databases")
    put("corpus.parity_targets", parity, "exp*.sqlite",
        "DISTINCT turns.scaffold_tokens per model_id - the enforced [STATE] "
        "block token count the padder must hit exactly")
    total_wall = sum(v["wall_clock_s"] or 0 for v in per_group.values())
    put("corpus.total.wall_clock_s", total_wall, "exp*.sqlite (run_meta)",
        "sum over databases of run_meta.finished_at - run_meta.started_at")
    print(f"    {len(dbs)} databases, {tot_ep:,} episodes, {tot_turn:,} turns, "
          f"{total_wall/3600:.2f} h wall clock")
    print(f"    parity targets: {parity}")
    return per_group


# ===========================================================================
# 6. THE ECHO TEST  (arm 3s: did the model repeat the displayed lie?)
# ===========================================================================

def echo_test(dbs: list[Path]) -> None:
    print("\n[6] echo test: arm 3s own_score answers vs displayed vs true")
    for p in dbs:
        g = p.stem
        c = connect(p)
        arms = {r[0] for r in c.execute("SELECT DISTINCT arm FROM turns")}
        if "3s" not in arms:
            c.close()
            continue
        rows = c.execute(
            "SELECT opponent_policy opp, turn, probe_answers FROM turn_details "
            "WHERE arm='3s' AND probe_answers IS NOT NULL").fetchall()
        c.close()
        agg = defaultdict(lambda: Counter())
        for opp, turn, raw in rows:
            try:
                pr = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            o = pr.get("own_score") or {}
            got, want = o.get("got"), o.get("want")
            if got is None or want is None:
                continue
            gi = normalise(str(got), AnswerType.NUMBER)
            wi = normalise(str(want), AnswerType.NUMBER)
            try:
                gi_n, wi_n = int(gi), int(wi)
            except (TypeError, ValueError):
                agg[opp]["unparsed"] += 1
                agg[opp]["n"] += 1
                continue
            agg[opp]["n"] += 1
            shown_hi = wi_n + SCORE_FALSIFICATION
            shown_lo = wi_n - SCORE_FALSIFICATION
            if gi_n == wi_n:
                agg[opp]["matched_true"] += 1
            elif gi_n in (shown_hi, shown_lo):
                agg[opp]["matched_displayed"] += 1
            else:
                agg[opp]["neither"] += 1
        tot = Counter()
        for opp, a in agg.items():
            tot.update(a)
            put(f"echo.{g}.3s|{opp}", {
                "n_probes_scored": a["n"],
                "matched_displayed": a["matched_displayed"],
                "matched_true": a["matched_true"],
                "neither": a["neither"],
                "frac_matched_displayed": a["matched_displayed"] / a["n"] if a["n"] else None,
                "frac_matched_true": a["matched_true"] / a["n"] if a["n"] else None,
            }, p.name,
                "parse turn_details.probe_answers for arm 3s through "
                "cdx.probe.normalise(NUMBER); classify the answer as "
                "matched_true (== own_score.want), matched_displayed "
                "(== want +/- 15, the arm-3s score offset from cdx/scaffold.py), "
                "or neither")
        if tot["n"]:
            put(f"echo.{g}.3s|pooled", {
                "n_probes_scored": tot["n"],
                "matched_displayed": tot["matched_displayed"],
                "matched_true": tot["matched_true"],
                "neither": tot["neither"],
                "frac_matched_displayed": tot["matched_displayed"] / tot["n"],
                "frac_matched_true": tot["matched_true"] / tot["n"],
            }, p.name, "as echo.<group>.3s|<opp>, pooled over both opponents")
            print(f"    {g:<32} n={tot['n']:>6}  displayed="
                  f"{tot['matched_displayed']:>6}  true={tot['matched_true']:>6}"
                  f"  neither={tot['neither']:>6}")


# ===========================================================================
# 7. DONOR ECHO  (arm 3c: did the answer equal the DONOR's number?)
# ===========================================================================

def donor_echo(dbs: list[Path]) -> None:
    print("\n[7] donor echo: arm 3c own_score answers vs donor_agent_score")
    for p in dbs:
        g = p.stem
        c = connect(p)
        cols = {r[1] for r in c.execute("PRAGMA table_info(turns)")}
        arms = {r[0] for r in c.execute("SELECT DISTINCT arm FROM turns")}
        if "3c" not in arms or "donor_agent_score" not in cols:
            c.close()
            continue
        # Two light passes joined in Python. A single SQL join drags
        # turn_details' large text columns through the planner and costs far
        # more memory than this machine has.
        donors = {(r[0], r[1], r[2]): r[3] for r in c.execute(
            "SELECT opponent_policy, episode_id, turn, donor_agent_score "
            "FROM turns WHERE arm='3c' AND donor_agent_score IS NOT NULL "
            "AND COALESCE(donor_degenerate,0)=0")}
        rows = []
        for opp, ep, turn, raw in c.execute(
                "SELECT opponent_policy, episode_id, turn, probe_answers "
                "FROM turn_details WHERE arm='3c' AND probe_answers IS NOT NULL"):
            d = donors.get((opp, ep, turn))
            if d is not None:
                rows.append((opp, d, raw))
        c.close()
        del donors
        agg = defaultdict(Counter)
        for opp, donor, raw in rows:
            try:
                pr = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            o = pr.get("own_score") or {}
            got, want = o.get("got"), o.get("want")
            if got is None or want is None:
                continue
            try:
                gi = int(normalise(str(got), AnswerType.NUMBER))
                wi = int(normalise(str(want), AnswerType.NUMBER))
            except (TypeError, ValueError):
                agg[opp]["unparsed"] += 1
                agg[opp]["n"] += 1
                continue
            di = int(donor)
            agg[opp]["n"] += 1
            if di == wi:
                agg[opp]["donor_equals_truth"] += 1
                # indistinguishable; excluded from the echo denominator below
                continue
            agg[opp]["n_informative"] += 1
            if gi == di:
                agg[opp]["matched_donor"] += 1
            elif gi == wi:
                agg[opp]["matched_true"] += 1
            elif abs(gi - wi) == 1:
                agg[opp]["off_by_one"] += 1
            else:
                agg[opp]["other"] += 1
        for opp, a in agg.items():
            ni = a["n_informative"]
            put(f"donor_echo.{g}.3c|{opp}", {
                "n_probes_scored": a["n"],
                "n_informative": ni,
                "matched_donor": a["matched_donor"],
                "matched_true": a["matched_true"],
                "off_by_one": a["off_by_one"],
                "other": a["other"],
                "donor_equals_truth": a["donor_equals_truth"],
                "frac_matched_donor": a["matched_donor"] / ni if ni else None,
                "frac_matched_true": a["matched_true"] / ni if ni else None,
            }, p.name,
                "join turns(arm 3c, donor_degenerate=0) to turn_details on the "
                "full key; parse own_score through cdx.probe.normalise(NUMBER); "
                "classify against turns.donor_agent_score (the number the block "
                "DISPLAYED) vs own_score.want (the truth). Rows where the donor "
                "happened to equal the truth are excluded from the denominator "
                "because an echo is not identifiable there.")
            print(f"    {g:<32} {opp:<5} informative={ni:>6}  "
                  f"donor={a['matched_donor']:>6}  true={a['matched_true']:>6}")


# ===========================================================================
# 8. SWAP RESCORE  (computed here, through the production normaliser)
# ===========================================================================

SWAP = {"cooperate": "Defect", "defect": "Cooperate"}
_TYPE = {"own_score": AnswerType.NUMBER,
         "opponent_last": AnswerType.ACTION,
         "rounds_played": AnswerType.NUMBER}


def _mark(got, want, kind) -> int:
    t = _TYPE[kind]
    return int(normalise(str(got), t) == normalise(str(want), t))


def _swapped(ans):
    if ans is None:
        return None
    key = normalise(str(ans), AnswerType.ACTION)
    return SWAP.get(key, key)


def swap_rescore() -> None:
    print("\n[8] label-swap probe rescore (5 swap groups)")
    paths = sorted(set(list(ROOT.glob("exp*_swap*.sqlite")) +
                       list(ROOT.glob("exp*_labelswap*.sqlite"))))
    paths = [p for p in paths if p.stat().st_size > 0]
    for p in paths:
        g = p.stem
        c = connect(p)
        td = {r[1] for r in c.execute("PRAGMA table_info(turn_details)")}
        if "probe_answers" not in td:
            c.close()
            note(f"{g}: no probe_answers column; swap rescore not possible")
            continue
        rows = c.execute(
            "SELECT arm, opponent_policy, turn, probe_answers FROM turn_details "
            "WHERE probe_answers IS NOT NULL").fetchall()
        c.close()
        if not rows:
            note(f"{g}: probe_answers all NULL; swap rescore not possible")
            continue

        cont: Counter = Counter()
        orig = defaultdict(lambda: [0, 0])
        fixed = defaultdict(lambda: [0, 0])
        for arm, opp, turn, raw in rows:
            try:
                pr = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            o = pr.get("opponent_last") or {}
            if o.get("got") is not None and o.get("want") is not None:
                cont[(str(o["want"]).strip(), str(o["got"]).strip())] += 1
            mo, mf = [], []
            for kind in ("own_score", "opponent_last", "rounds_played"):
                d = pr.get(kind) or {}
                got, want = d.get("got"), d.get("want")
                if got is None or want is None:
                    mo.append(0)
                    mf.append(0)
                    continue
                mo.append(_mark(got, want, kind))
                g2 = _swapped(got) if kind == "opponent_last" else got
                mf.append(_mark(g2, want, kind))
            k = (arm, opp)
            orig[k][0] += int(all(mo))
            orig[k][1] += 1
            fixed[k][0] += int(all(mf))
            fixed[k][1] += 1

        diag = sum(v for (a, b), v in cont.items() if a.lower() == b.lower())
        inv = sum(v for (a, b), v in cont.items()
                  if a.lower() in SWAP and b.lower() == SWAP[a.lower()].lower())
        tot = sum(cont.values())
        put(f"swap_contingency.{g}", {
            "n": tot, "agrees": diag, "exactly_inverted": inv,
            "frac_agrees": diag / tot if tot else None,
            "frac_inverted": inv / tot if tot else None,
            "clean_inversion": bool(tot and inv / tot > 0.5)}, p.name,
            "raw want x got contingency of the opponent_last probe from "
            "turn_details.probe_answers, NO rescoring applied. The rescore "
            "below is only justified if the off-diagonal dominates.")
        for k in sorted(orig):
            arm, opp = k
            n = orig[k][1]
            put(f"swap_rescore.{g}.{arm}|{opp}", {
                "n": n,
                "cpr_as_run": orig[k][0] / n if n else None,
                "cpr_rescored": fixed[k][0] / n if n else None,
                "gate_0.85_as_run": bool(n and orig[k][0] / n >= 0.85),
                "gate_0.85_rescored": bool(n and fixed[k][0] / n >= 0.85),
            }, p.name,
                "all-or-nothing CPR over the three probes, recomputed from "
                "turn_details.probe_answers through cdx.probe.normalise. "
                "'rescored' inverts ONLY the opponent_last answer "
                "(Cooperate<->Defect) before comparison, because the swap "
                "condition inverts the action words while the run-time scorer "
                "compared against unswapped truth.")
        arm3 = [FACTS[f"swap_rescore.{g}.3|{o}"]["value"]["cpr_rescored"]
                for o in OPPONENTS
                if f"swap_rescore.{g}.3|{o}" in FACTS]
        print(f"    {g:<26} inverted={inv/tot if tot else 0:.3f}  "
              f"arm3 rescored={['%.3f' % x for x in arm3]}")


# ===========================================================================
# 9. CONTRASTS, PARITY, HETEROGENEITY, MULTIPLICITY  (read from JSON artefacts)
# ===========================================================================

REV_SOURCES = [("EXP7_REVIEWER.json", "EXP7_REVIEWER.json"),
               ("REVIEWER_RESPONSES_ALL.json", "REVIEWER_RESPONSES_ALL.json"),
               ("REVIEWER_RESPONSES.json", "REVIEWER_RESPONSES.json")]


def reviewer_facts() -> None:
    print("\n[9] contrasts / parity / heterogeneity / multiplicity (JSON artefacts)")
    seen_contrast, seen_parity = set(), set()
    for fname, label in REV_SOURCES:
        d = load_json(fname)
        put(f"meta.{label}.bootstrap", {
            "seed": d["seed"], "resamples": d["bootstrap_resamples"],
            "n_databases": len(d["databases"])}, label,
            "top-level 'seed' / 'bootstrap_resamples' / len('databases')")
        for rec in d["per_database"]:
            g = rec["identity"]["group"]
            put(f"identity.{g}", rec["identity"], label,
                "per_database[i].identity") if f"identity.{g}" not in FACTS else None

            fc = rec.get("F_contrasts") or {}
            if isinstance(fc, dict) and "skipped" not in fc:
                for key, v in fc.items():
                    fk = f"contrast.{g}.{key}"
                    if fk in seen_contrast:
                        continue
                    seen_contrast.add(fk)
                    put(fk, {
                        "label": v.get("label"), "x": v.get("x"),
                        "y": v.get("y"), "opponent": v.get("opp"),
                        "min_turn": v.get("min_turn"),
                        "incl_t0": {kk: jnum(vv) if kk != "sig" else bool(vv)
                                    for kk, vv in v["incl_t0"].items()},
                        "quotable": {kk: jnum(vv) if kk != "sig" else bool(vv)
                                     for kk, vv in v["quotable"].items()},
                    }, label,
                        f"per_database[group={g}].F_contrasts['{key}']: "
                        "episode-clustered percentile bootstrap of the "
                        "difference in per-episode defection rate between arms "
                        "x and y; 'quotable' drops turn 0 where the arm is "
                        "byte-identical to arm 3 at turn 0 (3m, 3c)")

            hp = rec.get("H_parity") or {}
            if isinstance(hp, dict) and "skipped" not in hp:
                for key, v in hp.items():
                    pk = f"parity.{g}.{key}"
                    if pk in seen_parity:
                        continue
                    seen_parity.add(pk)
                    det = v["parity_detrended"]
                    put(pk, {
                        "detrended_diff": jnum(det["diff"]),
                        "lo": jnum(det["lo"]), "hi": jnum(det["hi"]),
                        "p": jnum(det["p"]), "n_episodes": det["n"],
                        "raw_even_minus_odd": jnum(v["parity"]["diff"]),
                        "raw_lo": jnum(v["parity"]["lo"]),
                        "raw_hi": jnum(v["parity"]["hi"]),
                        "horizon_last5_minus_first5": jnum(v["horizon"]["diff"]),
                    }, label,
                        f"per_database[group={g}].H_parity['{key}']. The "
                        "detrended coefficient differences each odd turn "
                        "against its two neighbours, y[2k+1]-(y[2k]+y[2k+2])/2, "
                        "averaged within episode then bootstrapped over "
                        "episodes; sign convention even minus odd. A locally "
                        "linear turn trend of any slope contributes zero.")

        # heterogeneity. REVIEWER_RESPONSES.json is a single-database rerun of
        # one exp7 group, so it has no across-model family of its own; its
        # per-database contrasts are already captured above.
        if len(d["databases"]) < 2:
            continue
        fam = "exp7" if label.startswith("EXP7") else "exp2-exp6"
        for stratum, v in (d.get("B_model_heterogeneity") or {}).items():
            if not isinstance(v, dict):
                continue
            pg = v.get("per_group") or {}
            diffs = [x["diff"] for x in pg.values()]
            if not diffs:
                continue
            jt = v.get("joint") or {}
            put(f"heterogeneity.{fam}.{stratum}", {
                "joint_Q": jnum(jt.get("q")), "p": jnum(jt.get("p")),
                "k_groups": jt.get("k"),
                "spread": max(diffs) - min(diffs),
                "min": min(diffs), "max": max(diffs),
                "argmin": min(pg, key=lambda g: pg[g]["diff"]),
                "argmax": max(pg, key=lambda g: pg[g]["diff"]),
                "per_group": {g: jnum(x["diff"]) for g, x in pg.items()},
            }, label,
                f"B_model_heterogeneity['{stratum}']: joint_Q and p read from "
                ".joint (Q is the bootstrap dispersion statistic across the k "
                "group effects, p its bootstrap tail probability); 'spread' "
                "computed HERE as max(per_group.diff) - min(per_group.diff)")

        mult = d.get("F_multiplicity") or {}
        if mult:
            put(f"multiplicity.{fam}", {
                "family_size": mult.get("family_size"),
                "raw_significant": mult.get("raw_significant"),
                "holm_significant": mult.get("holm_significant"),
                "bh_significant": mult.get("bh_significant"),
                "n_floor_effects": len(mult.get("floor_effects") or []),
            }, label,
                "F_multiplicity: family_size / raw_significant / "
                "holm_significant / bh_significant, computed at run time over "
                "the quotable p-values of the whole contrast family "
                "(5 contrasts x opponents x groups)")


# ===========================================================================
# 10. EXP6/EXP7 FIELDS: gates + falsification as published, echo as published
# ===========================================================================

def fields_facts() -> None:
    print("\n[10] EXP6_FIELDS.json / EXP7_FIELDS.json gates and integrity")
    for fname in ("EXP6_FIELDS.json", "EXP7_FIELDS.json"):
        for rec in load_json(fname):
            g = rec["group"]
            put(f"integrity.{g}", {
                "integrity_ok": bool(rec["integrity_ok"]),
                "gate_fail": bool(rec["gate_fail"])}, fname,
                "top-level 'integrity_ok' / 'gate_fail' for the group")
            for key, v in (rec.get("gates") or {}).items():
                put(f"gate.{g}.{key}", {
                    "cpr": jnum(v["cpr"]), "offtask": jnum(v["offtask"]),
                    "defect": jnum(v["defect"]), "n_turns": v["n"]},
                    fname, f"gates['{key}'] as published by analysis/13")
            ec = (rec.get("contrasts") or {}).get("echo_3s")
            if ec:
                put(f"echo_published.{g}", {
                    "n": ec["n"], "matched_displayed": ec["shown"],
                    "matched_true": ec["true"], "neither": ec["neither"]},
                    fname, "contrasts['echo_3s'] as published by analysis/13; "
                    "kept alongside the independently recomputed echo.* facts "
                    "so the two can be compared")
            for key, v in (rec.get("falsification") or {}).items():
                put(f"falsification_published.{g}.{key}", {
                    "lied": v["lied"], "n": v["n"],
                    "rate": v["lied"] / v["n"] if v["n"] else None},
                    fname, f"falsification['{key}'] as published by analysis/13")
            for key, v in (rec.get("contrasts") or {}).items():
                if key == "echo_3s":
                    continue
                put(f"contrast_fields.{g}.{key}", {
                    "incl_t0": {k: (bool(x) if k == "sig" else jnum(x))
                                for k, x in v["incl_t0"].items()},
                    "quotable": {k: (bool(x) if k == "sig" else jnum(x))
                                 for k, x in v["excl_t0"].items()},
                }, fname,
                    f"contrasts['{key}'] from analysis/13: episode-level "
                    "bootstrap, 'excl_t0' is the turn-0-excluded (quotable) "
                    "column, relabelled 'quotable' here for consistency")


# ===========================================================================
# 11. EPISODE-LEVEL CELL SUMMARIES (ep_*.json) and SIGN-FLIP VERDICTS
# ===========================================================================

def episode_facts() -> None:
    print("\n[11] episode-level bootstrap summaries (ep_*.json)")
    n = 0
    for p in sorted(ROOT.glob("ep_exp*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        g = p.stem[3:]
        put(f"signflip.{g}", d.get("sign_flip_verdict"), p.name,
            "top-level 'sign_flip_verdict'")
        for cell, v in (d.get("cells") or {}).items():
            put(f"episode_cell.{g}.{cell}", {
                "n_episodes": v["n_episodes"], "mean": jnum(v["mean"]),
                "sd": jnum(v["sd"]), "se_episode": jnum(v["se"]),
                "se_turn_naive": jnum(v["se_turn_naive"]),
                "se_inflation": jnum(v["se_inflation"])}, p.name,
                f"cells['{cell}']: per-episode defection rate, its sd, the "
                "episode-clustered SE, the turn-naive SE, and their ratio")
            n += 1
    print(f"    {n} cells over {len(list(ROOT.glob('ep_exp*.json')))} files")


# ===========================================================================
# 12. THE ARM LADDER STIMULUS (real rendered prompts, for figure 2)
# ===========================================================================

LADDER_ARMS = ("1", "3b", "3", "3c", "3s", "3m")


def arm_ladder() -> None:
    print("\n[12] arm ladder stimulus: real rendered [STATE] blocks")
    # exp6_llama_sem_logit is the only family of databases carrying all six
    # arms in one run, so the six blocks below are byte-comparable.
    db = ROOT / "exp6_llama_sem_logit.sqlite"
    if not db.exists():
        note("exp6_llama_sem_logit.sqlite absent; arm ladder cannot be built")
        return
    c = connect(db)
    tok = c.execute("SELECT DISTINCT scaffold_tokens FROM turns "
                    "WHERE scaffold_tokens IS NOT NULL").fetchall()
    model = c.execute("SELECT model_id FROM run_meta").fetchone()["model_id"]
    blocks = {}
    for arm in LADDER_ARMS:
        r = c.execute(
            "SELECT episode_id, turn, prompt_full FROM turn_details "
            "WHERE arm=? AND opponent_policy='tft' AND turn=5 "
            "AND prompt_full IS NOT NULL ORDER BY episode_id LIMIT 1",
            (arm,)).fetchone()
        if not r:
            note(f"arm ladder: no stored prompt_full for arm {arm}")
            continue
        pf = r["prompt_full"]
        if "[STATE]" in pf:
            body = pf.split("[STATE]", 1)[1].split("[HISTORY]", 1)[0]
            block = "[STATE]\n" + body.strip("\n")
        else:
            block = ""   # arm 1 renders no block at all
        blocks[arm] = {"episode_id": r["episode_id"], "turn": r["turn"],
                       "state_block": block,
                       "has_state_block": bool(block)}
    c.close()
    put("arm_ladder.source", {
        "database": db.name, "model_id": model, "opponent": "tft", "turn": 5,
        "parity_target_tokens": tok[0][0] if len(tok) == 1 else None},
        db.name,
        "turn_details.prompt_full, first stored episode of each arm at turn 5 "
        "vs TFT; text is sliced verbatim between the [STATE] and [HISTORY] "
        "markers and never retyped")
    for arm, v in blocks.items():
        put(f"arm_ladder.{arm}", v, db.name,
            f"turn_details.prompt_full WHERE arm='{arm}' AND "
            "opponent_policy='tft' AND turn=5, first stored episode; text "
            "between '[STATE]' and '[HISTORY]' taken verbatim")
    print(f"    {len(blocks)} arms rendered from {db.name} "
          f"(parity target {tok[0][0] if len(tok)==1 else tok} tokens)")


# ===========================================================================
# 13. DERIVED: opponent-pooled CPR and probe baseline
#
# Pooling is a weighted mean of facts already in the table, not a new pass over
# the data, so the provenance chain is unbroken: source_file still names the
# database the counts came from.
# ===========================================================================

def derived() -> None:
    print("\n[13] derived: opponent-pooled CPR and constant-answer baseline")
    groups = defaultdict(list)
    for k in list(FACTS):
        if k.startswith("cpr.") and k.count(".") == 2 and "|" in k:
            _, g, cell = k.split(".", 2)
            arm, opp = cell.split("|")
            if opp in OPPONENTS:
                groups[("cpr", g, arm)].append(k)
        if k.startswith("probe_baseline.") and "|" in k:
            _, g, cell = k.split(".", 2)
            arm, opp = cell.split("|")
            if opp in OPPONENTS:
                groups[("probe_baseline", g, arm)].append(k)
    for (kind, g, arm), keys in sorted(groups.items()):
        src = FACTS[keys[0]]["source_file"]
        if kind == "cpr":
            n = sum(FACTS[k]["value"]["n_probe_turns"] for k in keys)
            if not n:
                continue
            out = {"n_probe_turns": n}
            for f in ("cpr_all_or_nothing", "own_score", "opponent_last",
                      "rounds_played"):
                vals = [(FACTS[k]["value"][f], FACTS[k]["value"]["n_probe_turns"])
                        for k in keys if FACTS[k]["value"][f] is not None]
                out[f] = (sum(x * w for x, w in vals) / sum(w for _, w in vals)
                          if vals else None)
            put(f"cpr.{g}.{arm}|pooled", out, src,
                "turn-count-weighted mean of the per-opponent cpr.* facts for "
                "this arm, i.e. the same quantity computed over both opponent "
                "cells at once")
        else:
            n = sum(FACTS[k]["value"]["n_probe_turns"] for k in keys)
            if not n:
                continue
            pC = sum(FACTS[k]["value"]["p_true_is_cooperate"] *
                     FACTS[k]["value"]["n_probe_turns"] for k in keys) / n
            put(f"probe_baseline.{g}.{arm}|pooled", {
                "p_true_is_cooperate": pC,
                "constant_answer_accuracy": max(pC, 1.0 - pC),
                "n_probe_turns": n}, src,
                "turn-count-weighted mean of the per-opponent "
                "p_true_is_cooperate, then max(pC, 1-pC): the score a model "
                "that always answers the same label would get on the binary "
                "opponent_last probe over both opponent cells")


# ===========================================================================
# consistency assertions
# ===========================================================================

def consistency() -> None:
    print("\n[13] internal consistency")
    ok = 0

    def check(name, cond, detail=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"    OK   {name}")
        else:
            note(f"FAILED {name} {detail}")

    # a. arm 3 clears the pre-registered 0.85 comprehension gate wherever the
    #    block tells the truth and the action labels are not swapped. It is NOT
    #    exactly 1.000 everywhere - three mistral cells sit just below - so the
    #    fact table records the observed minimum rather than asserting a
    #    round number the data does not support.
    arm3 = {k: v["value"]["cpr_all_or_nothing"] for k, v in FACTS.items()
            if k.startswith("cpr.") and k.endswith(("3|allc", "3|tft"))
            and "swap" not in k}
    lo_k = min(arm3, key=arm3.get)
    put("cpr.arm3_nonswap.min", {
        "min_cpr": arm3[lo_k], "cell": lo_k, "n_cells": len(arm3),
        "n_cells_at_1.000": sum(1 for x in arm3.values() if x >= 1.0 - 1e-12)},
        "exp*.sqlite",
        "minimum over every non-swap arm-3 cell of the turn-0-excluded "
        "all-or-nothing CPR computed in stage 1-5")
    check("arm-3 CPR clears the pre-registered 0.85 gate in every non-swap cell",
          arm3[lo_k] >= 0.85, f"min {arm3[lo_k]:.4f} at {lo_k}")

    # b. swap arm-3 CPR rescores to >= 0.85 in every swap group
    bad = [k for k, v in FACTS.items()
           if k.startswith("swap_rescore.") and ".3|" in k
           and (v["value"]["cpr_rescored"] or 0) < 0.85]
    check("swap arm-3 CPR rescores above the 0.85 gate", not bad, str(bad))

    # c. echo test: recomputed vs published
    diffs = []
    for k, v in FACTS.items():
        if not k.startswith("echo_published."):
            continue
        g = k.split(".", 1)[1]
        mine = FACTS.get(f"echo.{g}.3s|pooled")
        if not mine:
            continue
        a, b = v["value"], mine["value"]
        if a["n"] != b["n_probes_scored"] or \
           a["matched_displayed"] != b["matched_displayed"] or \
           a["matched_true"] != b["matched_true"]:
            diffs.append((g, a, b))
    check("recomputed echo test reproduces the published echo_3s counts",
          not diffs, str(diffs[:2]))

    # d. falsification: recomputed vs published
    diffs = []
    for k, v in FACTS.items():
        if not k.startswith("falsification_published."):
            continue
        rest = k.split(".", 1)[1]
        mine = FACTS.get("falsification." + rest)
        if not mine:
            continue
        if (v["value"]["lied"], v["value"]["n"]) != \
           (mine["value"]["lied"], mine["value"]["n"]):
            diffs.append((rest, v["value"], mine["value"]))
    check("recomputed falsification reproduces the published counts",
          not diffs, str(diffs[:3]))

    # e. arm 3m falsifies 100% of turns > 0 by construction
    bad = [k for k, v in FACTS.items()
           if k.startswith("falsification.") and ".3m|" in k
           and abs(v["value"]["rate"] - 1.0) > 1e-9]
    check("arm 3m falsification rate == 1.000 on every turn > 0", not bad,
          str(bad[:5]))

    # f. turn counts: arms x opponents x N x horizon
    bad = []
    for k, v in FACTS.items():
        if not k.startswith("corpus.") or not isinstance(v["value"], dict):
            continue
        c = v["value"]
        if len(c.get("episodes_per_cell") or []) != 1:
            continue
        if not isinstance(c.get("horizon"), int):
            continue
        exp = (len(c["arms"]) * len(c["opponents"]) *
               c["episodes_per_cell"][0] * c["horizon"])
        if exp != c["n_turns"]:
            bad.append((k, exp, c["n_turns"]))
    check("n_turns == arms x opponents x N x horizon in every database",
          not bad, str(bad[:5]))

    # g. episodes = cells x N
    bad = []
    for k, v in FACTS.items():
        if not k.startswith("corpus.") or not isinstance(v["value"], dict):
            continue
        c = v["value"]
        if len(c.get("episodes_per_cell") or []) != 1:
            continue
        if c["n_cells"] * c["episodes_per_cell"][0] != c["n_episodes"]:
            bad.append(k)
    check("n_episodes == n_cells x episodes_per_cell", not bad, str(bad[:5]))

    # h. every parity target is one of the three calibrated values
    tg = FACTS["corpus.parity_targets"]["value"]
    check("parity targets are model-specific and stable",
          len(set(tg.values())) == len(tg) and all(isinstance(x, int)
                                                   for x in tg.values()),
          str(tg))

    # i. contrast CIs bracket their point estimate
    bad = []
    for k, v in FACTS.items():
        if not k.startswith("contrast."):
            continue
        q = v["value"]["quotable"]
        if q["diff"] is None or q["lo"] is None:
            continue
        if not (q["lo"] - 1e-9 <= q["diff"] <= q["hi"] + 1e-9):
            bad.append((k, q))
    check("every quotable CI brackets its point estimate", not bad,
          str(bad[:3]))

    # j. multiplicity survivor counts are monotone: holm <= bh <= raw
    bad = []
    for k, v in FACTS.items():
        if not k.startswith("multiplicity."):
            continue
        m = v["value"]
        if not (m["holm_significant"] <= m["bh_significant"] <=
                m["raw_significant"] <= m["family_size"]):
            bad.append((k, m))
    check("multiplicity: holm <= BH <= raw <= family size", not bad, str(bad))

    print(f"    {ok}/10 consistency checks passed")


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="force a cold rebuild, ignoring paper/.facts_cache.json")
    args = ap.parse_args()
    use_cache = not args.no_cache

    PAPER.mkdir(exist_ok=True)
    dbs = databases()
    print(f"FACTS build: {len(dbs)} databases under {ROOT}")

    run_stage("1-5_scan", scan_databases, dbs, cache=use_cache)
    run_stage("6_echo", echo_test, dbs, cache=use_cache)
    run_stage("7_donor_echo", donor_echo, dbs, cache=use_cache)
    run_stage("8_swap_rescore", swap_rescore, cache=use_cache)
    run_stage("9_reviewer", reviewer_facts, cache=use_cache)
    run_stage("10_fields", fields_facts, cache=use_cache)
    run_stage("11_episode", episode_facts, cache=use_cache)
    run_stage("12_arm_ladder", arm_ladder, cache=use_cache)
    derived()
    consistency()

    payload = {
        "_generated_by": "paper/extract_facts.py",
        "_rule": ("every value is either read from a machine-generated JSON "
                  "artefact or computed here from a committed .sqlite; no "
                  "value is transcribed from prose"),
        "_bootstrap_seed": SEED,
        "_bootstrap_resamples": N_BOOT,
        "_problems": _PROBLEMS,
        "facts": FACTS,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"\nwrote {OUT}  ({len(FACTS)} facts, "
          f"{OUT.stat().st_size/1024:.0f} KiB)")
    pref = Counter(k.split(".")[0] for k in FACTS)
    print("\nfacts by namespace:")
    for k, v in sorted(pref.items()):
        print(f"    {k:<26}{v:>6}")
    if _PROBLEMS:
        print(f"\n{len(_PROBLEMS)} problem(s) recorded in FACTS._problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
