"""Arm 3c donor selection.

Donor selection decides what NUMBERS the model is shown in the stale-state arm.
Get it wrong and the arm silently becomes a copy of Arm 3 (donor identical to
the truth) or of nothing at all (donor is the recipient itself), while still
producing plausible-looking output. There is no downstream check that would
catch either, so it is checked here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import Action
from cdx.donor import DonorStats, select_donor, state_fingerprint


class S:
    """Only the fields the treatment template renders."""

    def __init__(self, agent: int, opponent: int, turn: int, last=None) -> None:
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self._last = last

    def last_opponent_action(self):
        return self._last


def rng(seed: int = 1) -> random.Random:
    return random.Random(seed)


# --- correctness -----------------------------------------------------------


def test_donor_is_never_the_recipient():
    states = [S(i * 3, i * 3, 4) for i in range(20)]
    for i in range(len(states)):
        donor, degenerate = select_donor(states, i, rng(i))
        assert not degenerate
        assert donor is not states[i]


def test_donor_differs_from_the_true_state():
    """The entire point of the arm. An identical donor is no manipulation."""
    states = [S(i * 3, i * 3, 4) for i in range(20)]
    for i in range(len(states)):
        donor, _ = select_donor(states, i, rng(i))
        assert state_fingerprint(donor) != state_fingerprint(states[i])


def test_identical_states_are_reported_degenerate_not_silently_accepted():
    """Turn 0: every episode has score 0 and no last move."""
    states = [S(0, 0, 0, None) for _ in range(50)]
    donor, degenerate = select_donor(states, 7, rng())
    assert degenerate
    assert state_fingerprint(donor) == state_fingerprint(states[7])


def test_a_single_distinct_state_is_found_despite_sampling():
    """Random sampling could miss a lone candidate; the exhaustive fallback
    must not. Without it the arm would be reported degenerate when it is not."""
    states = [S(0, 0, 4, None) for _ in range(200)]
    states[137] = S(15, 9, 4, Action.DEFECT)
    donor, degenerate = select_donor(states, 0, rng(), max_draws=2)
    assert not degenerate
    assert donor is states[137]


def test_single_live_episode_degrades_instead_of_crashing():
    """Late in a stochastic run a cell can have one episode left. Returning
    None here would make build_pair raise and kill the cell on rented
    hardware. Degenerate is a measurement, not an error."""
    only = S(3, 3, 1)
    donor, degenerate = select_donor([only], 0, rng())
    assert donor is only
    assert degenerate


def test_empty_pool_is_a_programming_error():
    with pytest.raises(ValueError):
        select_donor([], 0, rng())


def test_returned_donor_is_never_none():
    """build_pair(donor=None) raises for Arm 3c, so no code path may produce
    None. Covers the pools that could plausibly occur mid-run."""
    for pool in ([S(1, 1, 1)],
                 [S(1, 1, 1), S(1, 1, 1)],
                 [S(1, 1, 1), S(2, 2, 1)],
                 [S(0, 0, 0, None)] * 40):
        for i in range(len(pool)):
            donor, _ = select_donor(pool, i, rng(i))
            assert donor is not None


# --- determinism -----------------------------------------------------------


def test_selection_is_reproducible_for_a_given_seed():
    """Re-running an episode must reproduce its donors, or the released
    trajectories are not replayable."""
    states = [S(i, i, 4) for i in range(30)]
    a = [select_donor(states, i, rng(99))[0] for i in range(30)]
    b = [select_donor(states, i, rng(99))[0] for i in range(30)]
    assert [id(x) for x in a] == [id(x) for x in b]


def test_fingerprint_covers_every_rendered_field():
    """A field the template renders but the fingerprint ignores would let an
    'identical' donor actually differ, or the reverse."""
    base = S(3, 3, 1, Action.COOPERATE)
    for other in (
        S(4, 3, 1, Action.COOPERATE),
        S(3, 4, 1, Action.COOPERATE),
        S(3, 3, 2, Action.COOPERATE),
        S(3, 3, 1, Action.DEFECT),
        S(3, 3, 1, None),
    ):
        assert state_fingerprint(base) != state_fingerprint(other)


# --- accounting ------------------------------------------------------------


def test_stats_namespace_cannot_clobber_the_manipulation_check():
    """The cell result already has 'by_turn' holding CPR per turn. An
    unnamespaced merge would overwrite it with donor data and look plausible."""
    stats = DonorStats()
    stats.record(0, True)
    assert "by_turn" not in stats.summary()
    assert "donor_by_turn" in stats.summary()


def test_degeneracy_rate_is_accurate():
    stats = DonorStats()
    for turn, degenerate in [(0, True), (0, True), (1, False), (1, True), (2, False)]:
        stats.record(turn, degenerate)
    assert stats.rate == pytest.approx(3 / 5)
    # String keys: these round-trip through JSON for cell-level resume, and
    # JSON would stringify them anyway.
    assert stats.summary()["donor_by_turn"] == {"0": 1.0, "1": 0.5, "2": 0.0}