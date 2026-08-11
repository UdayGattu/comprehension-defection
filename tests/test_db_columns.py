"""Every field lands in the column it belongs to.

THE RISK THIS COVERS
    The inserts used positional VALUES(?,?,...) with a hand-counted number of
    placeholders. Adding a column meant editing the count in a second place,
    and getting it wrong shifts every subsequent value one position left
    WITHOUT raising - SQLite happily writes an int into the next int column.
    The result is a database full of plausible numbers in the wrong fields,
    discoverable only by noticing that an analysis makes no sense.

    The inserts now name their columns, which makes that impossible. This file
    proves it stays that way: every field is given a DISTINCT value, and every
    one is read back and checked. A shift by one position fails here rather
    than three hours into a paid run.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import Action, Arm, Framing, HorizonMode, OpponentPolicy, ReadoutMode
from cdx.db import (
    _DETAIL_COLUMNS,
    _EPISODE_COLUMNS,
    _TURN_COLUMNS,
    EpisodeRecord,
    Store,
    TurnRecord,
)
from cdx.seeding import EpisodeKey

# Distinct values throughout: if any two matched, a one-position shift could
# pass unnoticed.
TURN = TurnRecord(
    turn=7,
    agent_action=Action.DEFECT,
    opponent_action=Action.COOPERATE,
    agent_payoff=5,
    optimal_action=Action.COOPERATE,
    turn_regret=11,
    logit_mass_c=0.11,
    logit_mass_d=0.22,
    action_mass_total=0.33,
    logit_gap=0.44,
    scaffold_tokens=34,
    scaffold_pad=3,
    cpr_score=1,
    cpr_method="replay",
    scaffold_echo=0,
    cpr_own_score=1,
    cpr_opponent_last=0,
    cpr_rounds_played=1,
    turn_regret_calc=13,
    action_tokens_found=6,
    prompt_tokens=512,
    donor_agent_score=27,
    donor_degenerate=1,
    top_tokens='[["Coop",0.9]]',
    scratchpad="thinking",
    probe_answers='{"own_score": "27"}',
    prompt_preview="preview text",
    prompt_full="the complete prompt",
)

KEY = EpisodeKey("testrun", 3, Arm.PLACEBO_STALE, "m/x",
                 ReadoutMode.LOGIT, OpponentPolicy.TFT)

EPISODE = EpisodeRecord(
    key=KEY,
    model_revision="rev1",
    framing=Framing.SEMANTIC,
    horizon_mode=HorizonMode.KNOWN,
    horizon=20,
    temperature=0.7,
    config_fingerprint="fp",
    prompt_hash="ph",
    n_turns=17,
    agent_score=41,
    opponent_score=43,
    defection_count=9,
    episode_regret=21,
)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    s.write_episode(EPISODE, [TURN], "2026-01-01T00:00:00")
    yield s
    s.close()


def row(store, table):
    cur = store._conn.execute(f"SELECT * FROM {table}")
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone()))


# --- column alignment -------------------------------------------------------


def test_turn_fields_land_in_the_right_columns(store):
    r = row(store, "turns")
    assert r["turn"] == 7
    assert r["agent_action"] == "D"
    assert r["opponent_action"] == "C"
    assert r["agent_payoff"] == 5
    assert r["optimal_action"] == "C"
    assert r["turn_regret"] == 11
    assert r["logit_mass_c"] == pytest.approx(0.11)
    assert r["logit_mass_d"] == pytest.approx(0.22)
    assert r["action_mass_total"] == pytest.approx(0.33)
    assert r["logit_gap"] == pytest.approx(0.44)
    assert r["scaffold_tokens"] == 34
    assert r["scaffold_pad"] == 3
    assert r["cpr_score"] == 1
    assert r["cpr_method"] == "replay"
    assert r["scaffold_echo"] == 0
    assert r["cpr_own_score"] == 1
    assert r["cpr_opponent_last"] == 0
    assert r["cpr_rounds_played"] == 1
    assert r["turn_regret_calc"] == 13
    assert r["action_tokens_found"] == 6
    assert r["prompt_tokens"] == 512
    assert r["donor_agent_score"] == 27
    assert r["donor_degenerate"] == 1


def test_detail_fields_land_in_the_right_columns(store):
    r = row(store, "turn_details")
    assert r["top_tokens"] == '[["Coop",0.9]]'
    assert r["scratchpad"] == "thinking"
    assert r["probe_answers"] == '{"own_score": "27"}'
    assert r["prompt_preview"] == "preview text"
    assert r["prompt_full"] == "the complete prompt"


def test_episode_fields_land_in_the_right_columns(store):
    r = row(store, "episodes")
    assert r["episode_id"] == 3
    assert r["arm"] == "3c"
    assert r["opponent_policy"] == "tft"
    assert r["framing"] == "semantic"
    assert r["horizon"] == 20
    assert r["n_turns"] == 17
    assert r["agent_score"] == 41
    assert r["opponent_score"] == 43
    assert r["defection_count"] == 9
    assert r["episode_regret"] == 21


# --- the column lists must match the schema ---------------------------------


@pytest.mark.parametrize("table,columns", [
    ("turns", _TURN_COLUMNS),
    ("turn_details", _DETAIL_COLUMNS),
    ("episodes", _EPISODE_COLUMNS),
])
def test_column_lists_match_the_schema(tmp_path, table, columns):
    """A column added to SCHEMA but not to the insert list would be silently
    left NULL forever."""
    s = Store(tmp_path / f"{table}.sqlite")
    actual = [r[1] for r in s._conn.execute(f"PRAGMA table_info({table})")]
    s.close()
    assert list(columns) == actual, (
        f"{table}: insert list and schema disagree.\n"
        f"  insert: {list(columns)}\n"
        f"  schema: {actual}"
    )


def test_new_columns_are_actually_present(tmp_path):
    """Guards the three fields added for exp3."""
    s = Store(tmp_path / "n.sqlite")
    turns = {r[1] for r in s._conn.execute("PRAGMA table_info(turns)")}
    details = {r[1] for r in s._conn.execute("PRAGMA table_info(turn_details)")}
    s.close()
    assert {"donor_agent_score", "donor_degenerate"} <= turns
    assert "prompt_full" in details


def test_nulls_round_trip(tmp_path):
    """Most turns have no donor and no full prompt. Those must store as NULL,
    not as 0 - a 0 donor score is a real, meaningful value."""
    s = Store(tmp_path / "z.sqlite")
    bare = TurnRecord(
        turn=0, agent_action=Action.COOPERATE, opponent_action=Action.COOPERATE,
        agent_payoff=3, optimal_action=None, turn_regret=None,
        logit_mass_c=None, logit_mass_d=None, action_mass_total=None,
        logit_gap=None, scaffold_tokens=None, scaffold_pad=None,
    )
    s.write_episode(EPISODE, [bare], "2026-01-01T00:00:00")
    r = row(s, "turns")
    s.close()
    assert r["donor_agent_score"] is None
    assert r["donor_degenerate"] is None