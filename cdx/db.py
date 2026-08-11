"""SQLite persistence with crash-safe resume.

Design decisions and why:

  WAL journaling      survives process kill mid-write. Requires LOCAL storage:
                      WAL uses shared memory and is unsupported on network
                      filesystems, so a Drive/FUSE mount would make this worse
                      rather than better. This is a large part of why the spec
                      forbids Colab.

  commit per EPISODE  not per turn. ~55k commits instead of ~1.1M. A crash costs
                      at most one episode, and hash-derived seeding means the
                      replacement episode is byte-identical to the lost one.

  wide data split out top_tokens JSON lives in a side table so the main table
                      stays narrow and fast to scan during analysis.

  reproducibility set seed, model revision, temperature, config fingerprint and
                      prompt hash on every row. The paper's differentiator is
                      that prior work used closed models at provider-default
                      temperatures; omitting these fields would forfeit it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .config import Action, Arm, Framing, HorizonMode, OpponentPolicy, ReadoutMode
from .seeding import EpisodeKey

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    run_id            TEXT    NOT NULL,
    episode_id        INTEGER NOT NULL,
    arm               TEXT    NOT NULL,
    model_id          TEXT    NOT NULL,
    model_revision    TEXT    NOT NULL,
    readout_mode      TEXT    NOT NULL,
    opponent_policy   TEXT    NOT NULL,
    framing           TEXT    NOT NULL,
    horizon_mode      TEXT    NOT NULL,
    horizon           INTEGER NOT NULL,
    temperature       REAL    NOT NULL,
    seed              INTEGER NOT NULL,
    config_fingerprint TEXT   NOT NULL,
    prompt_hash       TEXT    NOT NULL,
    n_turns           INTEGER NOT NULL,
    agent_score       INTEGER NOT NULL,
    opponent_score    INTEGER NOT NULL,
    defection_count   INTEGER NOT NULL,
    episode_regret    INTEGER,
    completed_at      TEXT    NOT NULL,
    PRIMARY KEY (run_id, episode_id, arm, model_id, readout_mode, opponent_policy)
);

CREATE TABLE IF NOT EXISTS turns (
    run_id            TEXT    NOT NULL,
    episode_id        INTEGER NOT NULL,
    arm               TEXT    NOT NULL,
    model_id          TEXT    NOT NULL,
    readout_mode      TEXT    NOT NULL,
    opponent_policy   TEXT    NOT NULL,
    turn              INTEGER NOT NULL,
    agent_action      TEXT    NOT NULL,
    opponent_action   TEXT    NOT NULL,
    agent_payoff      INTEGER NOT NULL,
    optimal_action    TEXT,
    turn_regret       INTEGER,
    logit_mass_c      REAL,
    logit_mass_d      REAL,
    action_mass_total REAL,
    logit_gap         REAL,
    scaffold_tokens   INTEGER,
    scaffold_pad      INTEGER,
    cpr_score         INTEGER,
    cpr_method        TEXT,
    scaffold_echo     INTEGER,
    cpr_own_score     INTEGER,
    cpr_opponent_last INTEGER,
    cpr_rounds_played INTEGER,
    turn_regret_calc  INTEGER,
    action_tokens_found INTEGER,
    prompt_tokens     INTEGER,

    -- Arm 3c only. The score the DONOR block actually displayed, and whether a
    -- distinct donor existed at all.
    --
    -- Without donor_agent_score there is no way to test the scaffold-echo
    -- question offline: if a probe answer equals the donor's number rather
    -- than the true one, the model demonstrably READ the block. That is the
    -- difference between "did not read it" and "read it and could not use
    -- it", and run exp2 could only infer it indirectly.
    --
    -- donor_degenerate marks turns where no distinct donor existed (every
    -- episode at turn 0 has score 0 and no last move). On those turns Arm 3c
    -- is identical to Arm 3 and contributes nothing; the aggregate rate hides
    -- WHICH turns, so it is recorded per row.
    donor_agent_score INTEGER,
    donor_degenerate  INTEGER,
    PRIMARY KEY (run_id, episode_id, arm, model_id, readout_mode, opponent_policy, turn)
);

-- Wide/rare payloads kept out of the hot table.
CREATE TABLE IF NOT EXISTS turn_details (
    run_id          TEXT NOT NULL,
    episode_id      INTEGER NOT NULL,
    arm             TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    readout_mode    TEXT NOT NULL,
    opponent_policy TEXT NOT NULL,
    turn            INTEGER NOT NULL,
    top_tokens      TEXT,
    scratchpad      TEXT,
    probe_answers   TEXT,   -- raw model text per probe; without this a CPR of 0
                            -- cannot be diagnosed after the fact
    prompt_preview  TEXT,   -- first+last 300 chars of the assembled prompt

    -- The COMPLETE decoded prompt, stored only for the first few episodes of
    -- each cell (see full_prompt_episodes). prompt_preview truncates the
    -- middle, which is exactly where the [STATE] block sits once the history
    -- grows - so on later turns the preview cannot show whether the block
    -- rendered at all. Storing every prompt would add gigabytes; storing a
    -- bounded sample costs almost nothing and answers the question.
    prompt_full     TEXT,
    PRIMARY KEY (run_id, episode_id, arm, model_id, readout_mode, opponent_policy, turn)
);

-- Environment provenance. Without this a result cannot be reproduced on
-- different hardware, and "we release seeded trajectories" is not checkable.
CREATE TABLE IF NOT EXISTS run_meta (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT,
    finished_at     TEXT,
    model_id        TEXT,
    model_revision  TEXT,
    dtype           TEXT,
    gpu_name        TEXT,
    gpu_count       INTEGER,
    driver          TEXT,
    vllm_version    TEXT,
    torch_version   TEXT,
    transformers_version TEXT,
    python_version  TEXT,
    git_commit      TEXT,
    probe_hash      TEXT,
    config_json     TEXT,
    argv            TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_analysis
    ON turns (run_id, model_id, arm, opponent_policy, readout_mode);
"""


@dataclass(frozen=True)
class EpisodeRecord:
    key: EpisodeKey
    model_revision: str
    framing: Framing
    horizon_mode: HorizonMode
    horizon: int
    temperature: float
    config_fingerprint: str
    prompt_hash: str
    n_turns: int
    agent_score: int
    opponent_score: int
    defection_count: int
    episode_regret: int | None


@dataclass(frozen=True)
class TurnRecord:
    turn: int
    agent_action: Action
    opponent_action: Action
    agent_payoff: int
    optimal_action: Action | None
    turn_regret: int | None
    logit_mass_c: float | None
    logit_mass_d: float | None
    action_mass_total: float | None
    logit_gap: float | None
    scaffold_tokens: int | None
    scaffold_pad: int | None
    cpr_score: int | None = None
    cpr_method: str | None = None
    scaffold_echo: int | None = None
    cpr_own_score: int | None = None
    cpr_opponent_last: int | None = None
    cpr_rounds_played: int | None = None
    turn_regret_calc: int | None = None
    action_tokens_found: int | None = None
    prompt_tokens: int | None = None
    donor_agent_score: int | None = None
    donor_degenerate: int | None = None
    top_tokens: str | None = None
    scratchpad: str | None = None
    probe_answers: str | None = None
    prompt_preview: str | None = None
    prompt_full: str | None = None


# Explicit column lists. The inserts below used positional VALUES(?,?,...) with
# a hand-counted number of placeholders; adding a column then required editing
# the count in a second place, and getting it wrong shifts every value one
# position to the left without raising - which would write plausible numbers
# into the wrong fields. Naming the columns makes that class of bug impossible.
_TURN_COLUMNS = (
    "run_id", "episode_id", "arm", "model_id", "readout_mode", "opponent_policy",
    "turn", "agent_action", "opponent_action", "agent_payoff", "optimal_action",
    "turn_regret", "logit_mass_c", "logit_mass_d", "action_mass_total",
    "logit_gap", "scaffold_tokens", "scaffold_pad", "cpr_score", "cpr_method",
    "scaffold_echo", "cpr_own_score", "cpr_opponent_last", "cpr_rounds_played",
    "turn_regret_calc", "action_tokens_found", "prompt_tokens",
    "donor_agent_score", "donor_degenerate",
)

_DETAIL_COLUMNS = (
    "run_id", "episode_id", "arm", "model_id", "readout_mode", "opponent_policy",
    "turn", "top_tokens", "scratchpad", "probe_answers", "prompt_preview",
    "prompt_full",
)

_EPISODE_COLUMNS = (
    "run_id", "episode_id", "arm", "model_id", "model_revision", "readout_mode",
    "opponent_policy", "framing", "horizon_mode", "horizon", "temperature",
    "seed", "config_fingerprint", "prompt_hash", "n_turns", "agent_score",
    "opponent_score", "defection_count", "episode_regret", "completed_at",
)


def _insert(table: str, columns: tuple[str, ...]) -> str:
    return (f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) "
            f"VALUES ({','.join('?' * len(columns))})")


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._configure()
        self._conn.executescript(SCHEMA)

    def _configure(self) -> None:
        # WAL requires local storage. Fail loudly rather than silently degrade.
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            raise RuntimeError(
                f"Could not enable WAL journaling (got {mode!r}). This usually means "
                f"the database is on a network filesystem, where WAL is unsupported "
                f"and crash-safety is not guaranteed. Move it to local storage."
            )
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def episode_transaction(self) -> Iterator[sqlite3.Connection]:
        """One transaction per episode. Either the whole episode lands or none
        of it does, so resume never sees a half-written trajectory."""
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def completed_keys(self, run_id: str) -> set[tuple]:
        """Coordinates already durably written. Resume skips exactly these."""
        rows = self._conn.execute(
            "SELECT episode_id, arm, model_id, readout_mode, opponent_policy "
            "FROM episodes WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return set(rows)

    def is_complete(self, key: EpisodeKey) -> bool:
        return (
            key.episode_id,
            key.arm.value,
            key.model_id,
            key.readout_mode.value,
            key.opponent.value,
        ) in self.completed_keys(key.run_id)

    def write_episode(
        self, record: EpisodeRecord, turns: Sequence[TurnRecord], completed_at: str
    ) -> None:
        k = record.key
        coords = (
            k.run_id,
            k.episode_id,
            k.arm.value,
            k.model_id,
            k.readout_mode.value,
            k.opponent.value,
        )
        with self.episode_transaction() as conn:
            conn.execute(
                _insert("episodes", _EPISODE_COLUMNS),
                (
                    k.run_id,
                    k.episode_id,
                    k.arm.value,
                    k.model_id,
                    record.model_revision,
                    k.readout_mode.value,
                    k.opponent.value,
                    record.framing.value,
                    record.horizon_mode.value,
                    record.horizon,
                    record.temperature,
                    k.seed(),
                    record.config_fingerprint,
                    record.prompt_hash,
                    record.n_turns,
                    record.agent_score,
                    record.opponent_score,
                    record.defection_count,
                    record.episode_regret,
                    completed_at,
                ),
            )
            for t in turns:
                conn.execute(
                    _insert("turns", _TURN_COLUMNS),
                    coords
                    + (
                        t.turn,
                        t.agent_action.value,
                        t.opponent_action.value,
                        t.agent_payoff,
                        t.optimal_action.value if t.optimal_action else None,
                        t.turn_regret,
                        t.logit_mass_c,
                        t.logit_mass_d,
                        t.action_mass_total,
                        t.logit_gap,
                        t.scaffold_tokens,
                        t.scaffold_pad,
                        t.cpr_score,
                        t.cpr_method,
                        t.scaffold_echo,
                        t.cpr_own_score,
                        t.cpr_opponent_last,
                        t.cpr_rounds_played,
                        t.turn_regret_calc,
                        t.action_tokens_found,
                        t.prompt_tokens,
                        t.donor_agent_score,
                        t.donor_degenerate,
                    ),
                )
                if any((t.top_tokens, t.scratchpad, t.probe_answers,
                        t.prompt_preview, t.prompt_full)):
                    conn.execute(
                        _insert("turn_details", _DETAIL_COLUMNS),
                        coords + (t.turn, t.top_tokens, t.scratchpad,
                                  t.probe_answers, t.prompt_preview,
                                  t.prompt_full),
                    )

    def write_run_meta(self, run_id: str, **fields) -> None:
        """Record environment provenance. Called once at run start and updated at
        the end. Cheap, and the difference between a reproducible release and a
        pile of numbers."""
        cols = ["run_id"] + list(fields)
        placeholders = ",".join("?" * len(cols))
        with self.episode_transaction() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO run_meta ({','.join(cols)}) "
                f"VALUES ({placeholders})",
                (run_id, *fields.values()),
            )

    def close(self) -> None:
        self._conn.close()


def encode_top_tokens(pairs: Sequence[tuple[str, float]]) -> str:
    return json.dumps([[t, round(p, 6)] for t, p in pairs], separators=(",", ":"))