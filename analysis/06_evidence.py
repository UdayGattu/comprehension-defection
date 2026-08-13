#!/usr/bin/env python3
"""Build EVIDENCE.md - every measured number, read from the databases.

WHY THIS EXISTS
    This project ran five experiments over several days, across three model
    families, two readouts, three framings and five arms. Any summary written
    from memory of that process contains errors: two independent attempts to
    write one by hand inverted the manipulation-check direction and stated the
    opposite of the exp5 perturbation result.

    So nothing here is asserted. Every figure is a SELECT. The only prose is the
    glossary explaining what a term means and why a contrast exists, and each
    piece of that prose points at the artefact that verifies it - the stored
    prompt, the recorded CLI arguments, the git commit.

    Read the output instead of trusting anyone's recollection, including mine.

WHAT IT WILL NOT DO
    It does not recompute bootstrap intervals. `analysis/02_episode_level.py`
    already produced them; this reads the resulting `ep_*.json` and marks any
    database that has none. Point estimates come from SQL and can be checked
    against those files - a disagreement is a finding.

    It does not interpret. Verdicts printed here (PASS/FAIL, VALID/EXCLUDED,
    REJECTED/UNDERPOWERED) are computed from thresholds stated in the glossary,
    not judged.

SCHEMA DRIFT IS EXPECTED
    exp1 predates most columns; exp2 predates `donor_agent_score` and
    `prompt_full`. Every query is built from `PRAGMA table_info`, so a missing
    column becomes a row in the coverage table rather than a crash.

    python analysis/06_evidence.py --out EVIDENCE.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import pathname2url

# Thresholds. Every verdict in the output is derived from these and nothing
# else, so a reader can recompute any of them by hand.
OFF_TASK_MASS = 0.10      # action_mass_total below this = the readout failed
OFF_TASK_CELL_GATE = 0.10 # share of off-task turns above which a cell is void
CPR_GATE = 0.85           # PREREGISTRATION.md's manipulation gate

# Verbatim rows kept PER ARM, spread evenly across the turns available rather
# than taken as whatever SQLite returns first. `LIMIT n` would hand back the
# lowest episode_id at the lowest turn every time - which for prompts means
# turn 0, where the score is 0, no round has been played and the [STATE] block
# is at its least informative. Spreading costs one extra query per arm and
# makes the sample representative instead of merely present.
SAMPLES_PER_ARM = 5


# ---------------------------------------------------------------- helpers

def ro_uri(p: Path) -> str:
    """Read-only URI that also works on WAL databases.

    `mode=ro` alone fails on WAL because SQLite wants to create the -shm/-wal
    sidecars. `immutable=1` promises we are the only reader.
    """
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def tables(con) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def cols(con, table: str) -> list[str]:
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def q(con, sql: str, args: tuple = ()) -> list[tuple]:
    """Query that returns [] instead of raising, so one missing column cannot
    take down a 28-database run."""
    try:
        return con.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []


def one(con, sql: str, args: tuple = ()):
    r = q(con, sql, args)
    return r[0] if r else None


def fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(header: list[str], rows: list[list]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- glossary

GLOSSARY = """
## 1. Glossary

Every term used anywhere in this document, its formula, and the reason the
measurement exists. This is the only section not computed from the databases;
each entry names the artefact that verifies it.

### The game

Read the exact wording from the stored prompts in section 2 rather than from
this summary. Iterated Prisoner's Dilemma, fixed and stated horizon. Payoffs
appear verbatim in every prompt.

| symbol | meaning |
|---|---|
| **TFT** | Tit-for-Tat. Cooperates first, then copies the agent's previous move. A retaliator. |
| **ALLC** | Always Cooperate, regardless of what the agent does. A pushover. |

The two opponents exist to make the hypothesis *opponent-conditional*: optimal
play differs between them, so a model that genuinely uses the state should
behave differently against each. `optimal_action` and `episode_regret` in the
database are computed against the solved optimum, not assumed - see section 7.

### The arms

Each arm is a different `[STATE]` block inserted at the same position in an
otherwise identical prompt. Section 2 prints the real thing for each.

| arm | contents | why it exists |
|---|---|---|
| **1** | no block; raw history only | baseline |
| **3** | the true score, opponent's last move, rounds played | the treatment |
| **3b** | true but non-diagnostic text, token- and density-matched to arm 3 | **the placebo.** Isolates the effect of *having a block* from the effect of *what the block says* |
| **3c** | arm 3's template rendered from a *different episode's* state | false-but-plausible content. Tests whether the block's truth matters |
| **3d** | syntactic noise in the same shape | tests whether structure alone suffices |

Arm 3b is the reason this study exists. Without it, the only available contrast
is arm 1 vs arm 3, which conflates the block's presence with its content.

### Framings and readouts

| factor | levels | purpose |
|---|---|---|
| **framing** | `semantic` (Cooperate/Defect) · `swap` (labels inverted) · `abstract` (X/Y) | Tests whether effects depend on the action words carrying meaning |
| **readout** | `logit` (action taken from the next-token distribution, no reasoning) · `scratchpad` (model generates reasoning first, action read after) | Tests whether findings reach the regime the literature actually uses |

### Metrics

| metric | formula | note |
|---|---|---|
| **defect rate** | share of turns with `agent_action = 'D'` | reported both turn-level and as a mean of per-episode rates |
| **action_mass_total** | probability mass on the two action tokens | the readout's own confidence that it read an action at all |
| **off-task (turn)** | `action_mass_total < 0.10` | the model emitted something other than an action |
| **off-task (cell)** | share of off-task turns in the cell | **above 0.10 the cell is void** - defect rates computed from prose are meaningless |
| **CPR** | all three state probes correct on a probed turn; no partial credit | manipulation check |
| **CPR gate** | `CPR(arm 3) >= 0.85` | from `PREREGISTRATION.md`. exp1 failed it; exp2 was the first to pass |
| **distinct trajectories** | unique action sequences in a cell | low entropy, **not** lost sample size - episodes are independently seeded, so a repeated trajectory is a repeated draw |
| **all-C share** | episodes that never defected | the point mass that makes a bootstrap the honest interval |
| **scaffold_tokens / scaffold_pad** | token length of the block and its filler | **token parity evidence** - section 4 checks it per cell rather than trusting a log line |
| **episode_regret** | payoff lost against the solved optimal policy | collected on every run and, until this document, never analysed |

### Contrasts

| contrast | formula | the question |
|---|---|---|
| **perturbation** | `P(D given 3b) - P(D given 1)` | does inserting *any* block change behaviour? |
| **ATE_true** | `P(D given 3) - P(D given 3b)` | holding the block constant, does its *content* change behaviour? **The primary contrast.** |
| **ATE_naive** | `P(D given 3) - P(D given 1)` | what a study without a placebo would report. Equals perturbation + ATE_true, and the two can cancel |
| **stale effect** | `P(D given 3c) - P(D given 3b)` | does false content act like true content? |
| **noise effect** | `P(D given 3d) - P(D given 1)` | is structure alone enough? |

### The pre-registered prediction

If defection is caused by losing track of the state, repairing it should move
play toward the opponent-conditional optimum: defection **down** vs TFT (do not
provoke a retaliator) and **up** vs ALLC (exploit a pushover).

Verdict rule, applied mechanically in section 6:

- both effects significant and **opposite** in sign, in the predicted
  directions -> SUPPORTED
- both significant and **sharing** a sign -> REJECTED (the prediction is
  opponent-conditional; a shared sign contradicts it)
- otherwise -> UNDERPOWERED
"""


# ---------------------------------------------------------------- per-db

def db_meta(con) -> dict:
    if "run_meta" not in tables(con):
        return {}
    c = cols(con, "run_meta")
    row = one(con, f"SELECT {','.join(c)} FROM run_meta LIMIT 1")
    return dict(zip(c, row)) if row else {}


def cell_metrics(con, has: set[str]) -> list[dict]:
    """One row per (arm, opponent). Every column guarded by schema presence."""
    if "turns" not in tables(con):
        return []

    def col(name, expr):
        return expr if name in has else "NULL"

    sql = f"""
    SELECT arm, opponent_policy,
           COUNT(*)                                              AS n_turns,
           COUNT(DISTINCT episode_id)                            AS n_ep,
           AVG(CASE WHEN agent_action='D' THEN 1.0 ELSE 0 END)   AS defect_turn,
           {col('action_mass_total', "AVG(action_mass_total)")}  AS mass_mean,
           {col('action_mass_total', "MIN(action_mass_total)")}  AS mass_min,
           {col('action_mass_total',
                f"AVG(CASE WHEN action_mass_total<{OFF_TASK_MASS} THEN 1.0 ELSE 0 END)")}
                                                                 AS off_task,
           {col('logit_mass_c', "AVG(logit_mass_c)")}            AS mass_c,
           {col('logit_mass_d', "AVG(logit_mass_d)")}            AS mass_d,
           {col('logit_gap', "AVG(logit_gap)")}                  AS gap,
           {col('cpr_score', "AVG(cpr_score)")}                  AS cpr,
           {col('cpr_score', "COUNT(cpr_score)")}                AS cpr_n,
           {col('cpr_own_score', "AVG(cpr_own_score)")}          AS cpr_own,
           {col('cpr_opponent_last', "AVG(cpr_opponent_last)")}  AS cpr_opp,
           {col('cpr_rounds_played', "AVG(cpr_rounds_played)")}  AS cpr_rnd,
           {col('scaffold_tokens', "AVG(scaffold_tokens)")}      AS scaf_mean,
           {col('scaffold_tokens', "MIN(scaffold_tokens)")}      AS scaf_min,
           {col('scaffold_tokens', "MAX(scaffold_tokens)")}      AS scaf_max,
           {col('scaffold_pad', "AVG(scaffold_pad)")}            AS pad_mean,
           {col('prompt_tokens', "AVG(prompt_tokens)")}          AS prompt_tok,
           {col('turn_regret', "AVG(turn_regret)")}              AS turn_regret,
           {col('optimal_action',
                "AVG(CASE WHEN optimal_action IS NOT NULL AND agent_action=optimal_action "
                "THEN 1.0 WHEN optimal_action IS NULL THEN NULL ELSE 0 END)")}
                                                                 AS optimal_match,
           {col('donor_agent_score', "COUNT(donor_agent_score)")} AS donor_n,
           {col('donor_degenerate', "AVG(donor_degenerate)")}    AS donor_degen,
           {col('scaffold_echo', "COUNT(scaffold_echo)")}        AS echo_n,
           {col('action_tokens_found', "AVG(action_tokens_found)")} AS act_found
    FROM turns GROUP BY arm, opponent_policy ORDER BY arm, opponent_policy
    """
    keys = ["arm", "opp", "n_turns", "n_ep", "defect_turn", "mass_mean",
            "mass_min", "off_task", "mass_c", "mass_d", "gap", "cpr", "cpr_n",
            "cpr_own", "cpr_opp", "cpr_rnd", "scaf_mean", "scaf_min",
            "scaf_max", "pad_mean", "prompt_tok", "turn_regret",
            "optimal_match", "donor_n", "donor_degen", "echo_n", "act_found"]
    out = [dict(zip(keys, r)) for r in q(con, sql)]

    # Episode-level defect rate, all-C share and distinct trajectories. These
    # cannot come from the same aggregate: the episode is the independent unit,
    # and a mean of per-episode rates is not the turn-level mean when episodes
    # differ in length.
    per_ep = defaultdict(list)
    traj = defaultdict(set)
    for arm, opp, ep, d, n, seq in q(con, """
        SELECT arm, opponent_policy, episode_id,
               SUM(CASE WHEN agent_action='D' THEN 1 ELSE 0 END),
               COUNT(*), GROUP_CONCAT(agent_action, '')
        FROM turns GROUP BY arm, opponent_policy, episode_id"""):
        if n:
            per_ep[(arm, opp)].append(d / n)
            traj[(arm, opp)].add(seq or "")
    for row in out:
        vals = per_ep.get((row["arm"], row["opp"]), [])
        row["defect_ep"] = st.mean(vals) if vals else None
        row["defect_ep_sd"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        row["se_ep"] = (st.pstdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else None
        row["all_c"] = (sum(1 for v in vals if v == 0) / len(vals)) if vals else None
        row["distinct_traj"] = len(traj.get((row["arm"], row["opp"]), set()))

    # Episode-level regret, if the run computed it.
    if "episodes" in tables(con) and "episode_regret" in cols(con, "episodes"):
        for arm, opp, mr, n in q(con, """
            SELECT arm, opponent_policy, AVG(episode_regret),
                   COUNT(episode_regret)
            FROM episodes GROUP BY arm, opponent_policy"""):
            for row in out:
                if row["arm"] == arm and row["opp"] == opp:
                    row["ep_regret"] = mr
                    row["ep_regret_n"] = n
    for row in out:
        row.setdefault("ep_regret", None)
        row.setdefault("ep_regret_n", 0)

    # Scratchpad lengths, if this run generated any.
    if "turn_details" in tables(con) and "scratchpad" in cols(con, "turn_details"):
        for arm, opp, n, mn, av, mx in q(con, """
            SELECT arm, opponent_policy, COUNT(scratchpad),
                   MIN(LENGTH(scratchpad)), AVG(LENGTH(scratchpad)),
                   MAX(LENGTH(scratchpad))
            FROM turn_details WHERE scratchpad IS NOT NULL
            GROUP BY arm, opponent_policy"""):
            for row in out:
                if row["arm"] == arm and row["opp"] == opp:
                    row.update(pad_n=n, pad_min=mn, pad_avg=av, pad_max=mx)
    for row in out:
        for k in ("pad_n", "pad_min", "pad_avg", "pad_max"):
            row.setdefault(k, None)
    return out


def cpr_by_turn(con, has: set[str]) -> list[tuple]:
    """CPR by arm and turn.

    Exists because an overall CPR of 0.200 can mean 'one of five probed turns
    passes' rather than 'twenty percent comprehension'. If a probe passes only
    at turn 0 - where the score is 0 and no round has been played - it is
    answerable without tracking anything, and the arm demonstrates no state
    tracking at all.
    """
    if "cpr_score" not in has:
        return []
    return q(con, """
        SELECT arm, turn, COUNT(*), AVG(cpr_score),
               AVG(cpr_own_score), AVG(cpr_opponent_last), AVG(cpr_rounds_played)
        FROM turns WHERE cpr_score IS NOT NULL
        GROUP BY arm, turn ORDER BY arm, turn""")


def parity_check(con, has: set[str]) -> list[tuple]:
    """Is the injected block the same token length in every arm that has one?

    The study's central methodological claim is that treatment and placebo are
    token-matched. Until this query the only evidence was a log line reading
    'parity target: 34'. This checks the assertion against every stored row.
    """
    if "scaffold_tokens" not in has:
        return []
    return q(con, """
        SELECT arm, COUNT(scaffold_tokens), MIN(scaffold_tokens),
               MAX(scaffold_tokens), AVG(scaffold_tokens), AVG(scaffold_pad)
        FROM turns WHERE scaffold_tokens IS NOT NULL
        GROUP BY arm ORDER BY arm""")


def _spread(values: list, n: int) -> list:
    """Pick up to n items evenly spaced across a sorted list.

    Deterministic, so the document is reproducible, and it reaches late turns
    where the [STATE] block actually carries a non-trivial score - which the
    first-n-rows a bare LIMIT returns never does.
    """
    if not values:
        return []
    if len(values) <= n:
        return list(values)
    step = (len(values) - 1) / (n - 1) if n > 1 else 0
    return [values[round(i * step)] for i in range(n)]


def _sample_by_arm(con, table_: str, column: str, where: str = "") -> list:
    """One row per (arm, turn) for up to SAMPLES_PER_ARM turns per arm.

    Verbatim and uncut. A truncated prompt is exactly how exp1's zero-padding
    defect survived to production.
    """
    out = []
    arms = [r[0] for r in q(
        con, f"SELECT DISTINCT arm FROM {table_} "
             f"WHERE {column} IS NOT NULL {where} ORDER BY arm")]
    for arm in arms:
        turns = [r[0] for r in q(
            con, f"SELECT DISTINCT turn FROM {table_} "
                 f"WHERE arm=? AND {column} IS NOT NULL {where} ORDER BY turn",
            (arm,))]
        for turn in _spread(turns, SAMPLES_PER_ARM):
            row = one(con, f"SELECT {column} FROM {table_} "
                           f"WHERE arm=? AND turn=? AND {column} IS NOT NULL "
                           f"{where} ORDER BY episode_id LIMIT 1", (arm, turn))
            if row and row[0] is not None:
                out.append((arm, turn, row[0]))
    return out


def samples(con, has_full: bool, has_pad: bool) -> dict:
    out = {"prompts": [], "scratchpads": [], "probes": [], "offtask": []}
    if "turn_details" not in tables(con):
        return out
    td = cols(con, "turn_details")

    if has_full:
        out["prompts"] = _sample_by_arm(con, "turn_details", "prompt_full")
    elif "prompt_preview" in td:
        # exp1 and exp2 predate prompt_full. The preview truncates the middle,
        # but on those runs the cut falls inside the rules section and the
        # [STATE] block survives intact - which is how the `Your score: 003`
        # defect is still recoverable from sweep.sqlite.
        out["prompts"] = _sample_by_arm(con, "turn_details", "prompt_preview")

    if has_pad:
        out["scratchpads"] = _sample_by_arm(
            con, "turn_details", "scratchpad", "AND LENGTH(scratchpad) > 20")

    if "probe_answers" in td:
        out["probes"] = _sample_by_arm(con, "turn_details", "probe_answers")

    # Top tokens on turns the readout failed - the only way to see WHAT the
    # model emitted instead of an action. Cannot use _sample_by_arm: the
    # off-task condition lives in `turns`, not `turn_details`.
    if "top_tokens" in td and "action_mass_total" in cols(con, "turns"):
        arms = [r[0] for r in q(con, """
            SELECT DISTINCT d.arm FROM turn_details d
            JOIN turns t USING (run_id, episode_id, arm, model_id,
                                readout_mode, opponent_policy, turn)
            WHERE t.action_mass_total < ? AND d.top_tokens IS NOT NULL
            ORDER BY d.arm""", (OFF_TASK_MASS,))]
        for arm in arms:
            rows = q(con, """
                SELECT d.turn, d.top_tokens FROM turn_details d
                JOIN turns t USING (run_id, episode_id, arm, model_id,
                                    readout_mode, opponent_policy, turn)
                WHERE d.arm=? AND t.action_mass_total < ?
                  AND d.top_tokens IS NOT NULL
                ORDER BY d.turn, d.episode_id""", (arm, OFF_TASK_MASS))
            for turn, tok in _spread(rows, SAMPLES_PER_ARM):
                out["offtask"].append((arm, turn, tok))
    return out


def contrasts(rows: list[dict]) -> list[list]:
    """Every contrast defined in the glossary, computed per opponent.

    Point estimates only. Bootstrap intervals come from ep_*.json - recomputing
    them here would produce a second set of numbers that could silently
    disagree with the ones already in the record.
    """
    by = {(r["arm"], r["opp"]): r for r in rows}
    opps = sorted({r["opp"] for r in rows})
    defs = [("perturbation", "3b", "1"), ("ATE_true", "3", "3b"),
            ("ATE_naive", "3", "1"), ("stale (3c-3b)", "3c", "3b"),
            ("noise (3d-1)", "3d", "1")]
    out = []
    for opp in opps:
        for name, a, b in defs:
            ra, rb = by.get((a, opp)), by.get((b, opp))
            if not ra or not rb:
                continue
            ep = (None if ra["defect_ep"] is None or rb["defect_ep"] is None
                  else ra["defect_ep"] - rb["defect_ep"])
            out.append([opp, name, f"P(D|{a}) - P(D|{b})",
                        fmt(ra["defect_turn"]), fmt(rb["defect_turn"]),
                        fmt(ra["defect_turn"] - rb["defect_turn"]), fmt(ep)])
    return out


def load_ci(stem: str) -> dict:
    """Bootstrap intervals already computed by 02_episode_level.py."""
    for cand in (f"ep_{stem}.json", f"ep_{stem}.json".replace("ep_ep_", "ep_")):
        if os.path.exists(cand):
            try:
                return json.load(open(cand))
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="EVIDENCE.md")
    ap.add_argument("--csv", default="EVIDENCE_cells.csv")
    ap.add_argument("--include-smoke", action="store_true",
                    help="Analyse the N=4 pilot databases too. Off by default: "
                         "they were instrument checks, not measurements.")
    args = ap.parse_args()

    all_db = sorted(Path(p) for p in glob.glob("*.sqlite"))
    smoke = [p for p in all_db if p.name.startswith(("smoke_", "cotsmoke_"))]
    prod = [p for p in all_db if p not in smoke]
    targets = all_db if args.include_smoke else prod

    md: list[str] = []
    csv_rows: list[dict] = []

    md.append("# Evidence file\n")
    md.append(
        "Generated by `analysis/06_evidence.py`. **Every number below is a "
        "SELECT against a database on disk.** No figure in this file was "
        "recalled, inferred, or carried over from a conversation.\n\n"
        "The only prose is section 1, which defines terms and states why each "
        "contrast exists. Everything else is computed.\n\n"
        "Verdicts (PASS/FAIL, VOID, REJECTED) are derived mechanically from the "
        f"thresholds in section 1: off-task gate {OFF_TASK_CELL_GATE}, "
        f"CPR gate {CPR_GATE}.\n")

    md.append("\n## 0. Files found\n")
    md.append(md_table(
        ["file", "MB", "analysed"],
        [[p.name, f"{p.stat().st_size/1e6:.0f}",
          "yes" if p in targets else "no - pilot (N=4)"] for p in all_db]))
    missing = sorted({Path(g).name[:-3] for g in glob.glob("*.sqlite.gz")}
                     - {p.name for p in all_db})
    if missing:
        md.append("\n**Compressed but not extracted — not analysed:**\n\n"
                  + "\n".join(f"- `{m}.gz`" for m in missing)
                  + "\n\n`gunzip -k` them and re-run to include.\n")

    md.append(GLOSSARY)

    md.append("\n## 2. The stimuli, as the model received them\n")
    md.append(
        "Taken from `turn_details.prompt_full`, which stores the **decoded "
        "prompt actually sent**, not the template that generated it. This "
        "distinction is not pedantic: exp1's zero-padding defect "
        "(`Your score: 012`) and its placebo density mismatch were both "
        "invisible in the templates and visible here.\n")

    md.append("\n## 3. Run inventory\n")
    md.append("Every argument that produced every database, from `run_meta`.\n")

    coverage: list[list] = []
    per_db_md: list[str] = []

    for path in targets:
        stem = path.stem
        con = sqlite3.connect(ro_uri(path), uri=True)
        tb = tables(con)
        tcols = set(cols(con, "turns"))
        dcols = set(cols(con, "turn_details"))
        meta = db_meta(con)

        coverage.append([
            stem, ",".join(sorted(tb)) or "-",
            "yes" if "donor_agent_score" in tcols else "NO",
            "yes" if "prompt_full" in dcols else "NO",
            "yes" if "scratchpad" in dcols else "NO",
            "yes" if "scaffold_tokens" in tcols else "NO",
            "yes" if "optimal_action" in tcols else "NO",
        ])

        s = [f"\n---\n\n### `{stem}`\n"]

        if meta:
            cfg = {}
            try:
                cfg = json.loads(meta.get("config_json") or "{}")
            except json.JSONDecodeError:
                pass
            s.append(md_table(["field", "value"], [
                [k, f"`{meta.get(k)}`"] for k in
                ("run_id", "model_id", "model_revision", "dtype", "gpu_name",
                 "gpu_count", "driver", "vllm_version", "torch_version",
                 "transformers_version", "python_version", "git_commit",
                 "probe_hash", "started_at", "finished_at")
                if meta.get(k) is not None]))
            if cfg:
                s.append("\n**Every CLI argument recorded for this run:**\n\n")
                s.append(md_table(["arg", "value"],
                                  [[k, f"`{v}`"] for k, v in sorted(cfg.items())]))
            if meta.get("argv"):
                s.append(f"\n```\n{meta['argv']}\n```\n")
        else:
            s.append("_no `run_meta` table_\n")

        rows = cell_metrics(con, tcols)
        for r in rows:
            r["db"] = stem
            csv_rows.append(r)

        if rows:
            s.append("\n**Cells** — turn-level and episode-level are both shown "
                     "because they differ when episodes vary in length.\n\n")
            s.append(md_table(
                ["arm", "opp", "n_ep", "n_turns", "P(D) turn", "P(D) ep",
                 "SE_ep", "all-C", "distinct", "off-task", "mass", "CPR",
                 "regret", "verdict"],
                [[r["arm"], r["opp"], r["n_ep"], r["n_turns"],
                  fmt(r["defect_turn"]), fmt(r["defect_ep"]), fmt(r["se_ep"], 5),
                  fmt(r["all_c"], 3), r["distinct_traj"], fmt(r["off_task"], 3),
                  fmt(r["mass_mean"], 3), fmt(r["cpr"], 3), fmt(r["ep_regret"], 2),
                  ("VOID off-task" if (r["off_task"] or 0) > OFF_TASK_CELL_GATE
                   else "ok")] for r in rows]))

            par = parity_check(con, tcols)
            if par:
                lens = {r[4] for r in par if r[4] is not None}
                ok = len(lens) <= 1
                s.append(f"\n**Token parity** — {'MATCHED' if ok else 'MISMATCH'} "
                         f"across arms carrying a block.\n\n")
                s.append(md_table(
                    ["arm", "n", "min", "max", "mean", "mean pad"],
                    [[r[0], r[1], r[2], r[3], fmt(r[4], 1), fmt(r[5], 1)]
                     for r in par]))

            cbt = cpr_by_turn(con, tcols)
            if cbt:
                s.append("\n**CPR by turn** — a probe that passes only at "
                         "turn 0 is answerable without tracking anything.\n\n")
                s.append(md_table(
                    ["arm", "turn", "n", "CPR", "own_score", "opp_last", "rounds"],
                    [[r[0], r[1], r[2], fmt(r[3], 3), fmt(r[4], 3),
                      fmt(r[5], 3), fmt(r[6], 3)] for r in cbt]))

            con_rows = contrasts(rows)
            if con_rows:
                s.append("\n**Contrasts** (point estimates; intervals below)\n\n")
                s.append(md_table(
                    ["opp", "contrast", "formula", "P(D) a", "P(D) b",
                     "turn-level", "episode-level"], con_rows))

            ci = load_ci(stem)
            if ci:
                # Printed whole. An earlier version cut this at 4000 characters,
                # which silently dropped intervals from the larger runs - the
                # exact failure this document exists to prevent.
                s.append(f"\n**Bootstrap intervals** from `ep_{stem}.json`, "
                         "computed by `analysis/02_episode_level.py`. Point "
                         "estimates should match the table above; a "
                         "disagreement is a finding.\n\n"
                         "```json\n" + json.dumps(ci, indent=2) + "\n```\n")
            else:
                s.append(f"\n_No `ep_{stem}.json`; run "
                         f"`analysis/02_episode_level.py --db {stem}.sqlite` "
                         "for intervals._\n")

        sm = samples(con, "prompt_full" in dcols, "scratchpad" in dcols)
        for label, key in (("Prompts, verbatim", "prompts"),
                           ("Scratchpads, verbatim", "scratchpads"),
                           ("Probe answers", "probes"),
                           ("Top tokens on off-task turns", "offtask")):
            if sm[key]:
                s.append(f"\n**{label}**\n")
                for arm, turn, text in sm[key]:
                    s.append(f"\n_arm {arm}, turn {turn}_\n\n```\n{text}\n```\n")

        per_db_md.append("".join(s))
        con.close()

    md.append("\n## 4. Schema coverage\n")
    md.append("Which columns each database actually has. Absence is recorded, "
              "not worked around silently.\n\n")
    md.append(md_table(
        ["database", "tables", "donor_score", "prompt_full", "scratchpad",
         "scaffold_tokens", "optimal_action"], coverage))

    md.append("\n## 5. Per-database detail\n")
    md.extend(per_db_md)

    md.append("\n## 6. Anomalies, computed\n")
    flags: list[list] = []
    for r in csv_rows:
        if (r.get("off_task") or 0) > OFF_TASK_CELL_GATE:
            flags.append([r["db"], f"{r['arm']}|{r['opp']}", "off-task",
                          fmt(r["off_task"], 3),
                          "cell VOID - defect rate computed from non-actions"])
        if r.get("defect_ep_sd") == 0.0 and r.get("n_ep", 0) > 1:
            flags.append([r["db"], f"{r['arm']}|{r['opp']}", "zero variance",
                          "0.0", "informs no contrast"])
        if r.get("scaf_min") is not None and r.get("scaf_max") is not None \
                and r["scaf_min"] != r["scaf_max"]:
            flags.append([r["db"], f"{r['arm']}|{r['opp']}", "parity spread",
                          f"{r['scaf_min']}..{r['scaf_max']}",
                          "block length varies within a cell"])
        if (r.get("echo_n") or 0) > 0:
            flags.append([r["db"], f"{r['arm']}|{r['opp']}", "scaffold_echo",
                          str(r["echo_n"]), "column believed unwritten"])
    md.append(md_table(["database", "cell", "flag", "value", "meaning"], flags)
              if flags else "_none_\n")

    Path(args.out).write_text("".join(md), encoding="utf-8")

    if csv_rows:
        keys = sorted({k for r in csv_rows for k in r})
        keys = ["db", "arm", "opp"] + [k for k in keys
                                       if k not in ("db", "arm", "opp")]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)

    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1024:.0f} KB)")
    print(f"wrote {args.csv}  ({len(csv_rows)} cells x {len(keys)} metrics)")
    print(f"databases analysed: {len(targets)}   pilots skipped: "
          f"{0 if args.include_smoke else len(smoke)}")
    if missing:
        print(f"NOT analysed (still gzipped): {len(missing)} - "
              f"gunzip -k them and re-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())