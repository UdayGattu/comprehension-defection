"""Arms 3s and 3m must falsify EXACTLY ONE field each.

WHY THIS FILE EXISTS
    Arm 3c replaces the whole [STATE] block with another episode's, so a
    behavioural difference cannot be attributed to any single field. 3s and 3m
    split that: one changes only "Your score", the other only "Opponent's last
    move". The entire value of the pair depends on that being literally true in
    the rendered prompt - if 3m also perturbs the score by a digit, the contrast
    is confounded and the experiment answers nothing.

    So these tests compare the rendered blocks LINE BY LINE against arm 3 and
    assert that exactly one line differs, and which one. Not that the code
    "looks right" - that the string the model receives differs in one place.

    This is the same class of check that would have caught exp1's zero-padded
    score ("Your score: 003") and its 44%-density placebo, both of which passed
    code review and were only found by inspecting rendered output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import Action, Arm, Framing, ScaffoldConfig
from cdx.scaffold import (
    SCORE_FALSIFICATION,
    ScaffoldBuilder,
    _MAX_REACHABLE_SCORE,
    falsified_view,
    move_was_falsified,
)


class FakeState:
    """Minimal stand-in exposing the four attributes the templates touch."""

    def __init__(self, agent=24, opponent=18, turn=8, last=Action.COOPERATE):
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self.turns = ()
        self._last = last

    def last_opponent_action(self):
        return self._last


class CharTokenizer:
    """One token per character. Makes token counts inspectable by hand, and is
    a harsher parity test than a BPE tokenizer because every character shows."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def builder():
    return ScaffoldBuilder(CharTokenizer(), ScaffoldConfig())


def lines(text):
    return [ln for ln in text.split("\n") if ln]


def differing_lines(a, b):
    la, lb = lines(a), lines(b)
    assert len(la) == len(lb), "field count changed - template was altered"
    return [(x, y) for x, y in zip(la, lb) if x != y]


# --- the core guarantee ----------------------------------------------------


def test_3s_changes_only_the_score_line(builder):
    s = FakeState()
    truth = builder.treatment_text(s, Framing.SEMANTIC)
    lie = builder.score_falsified_text(s, Framing.SEMANTIC, SCORE_FALSIFICATION)
    diff = differing_lines(truth, lie)
    assert len(diff) == 1, f"3s changed {len(diff)} lines, expected 1: {diff}"
    assert diff[0][0].startswith("Your score:")


def test_3m_changes_only_the_last_move_line(builder):
    s = FakeState()
    truth = builder.treatment_text(s, Framing.SEMANTIC)
    lie = builder.move_falsified_text(s, Framing.SEMANTIC)
    diff = differing_lines(truth, lie)
    assert len(diff) == 1, f"3m changed {len(diff)} lines, expected 1: {diff}"
    assert diff[0][0].startswith("Opponent's last move:")


def test_3m_flips_the_action_rather_than_blanking_it(builder):
    for true_move, expected in ((Action.COOPERATE, "Defect"),
                                (Action.DEFECT, "Cooperate")):
        s = FakeState(last=true_move)
        lie = builder.move_falsified_text(s, Framing.SEMANTIC)
        line = [ln for ln in lines(lie) if ln.startswith("Opponent's last move:")][0]
        assert line.endswith(expected), line


def test_3s_moves_the_score_by_the_stated_amount(builder):
    s = FakeState(agent=24)
    v = falsified_view(s, score_offset=SCORE_FALSIFICATION)
    assert v.agent_score == 24 + SCORE_FALSIFICATION


# --- what must NOT change --------------------------------------------------


def test_falsification_never_mutates_the_real_state():
    """The engine must keep advancing on the truth. If the view aliased the
    state, the falsification would leak into the next turn's history and the
    game itself would diverge between arms."""
    s = FakeState(agent=24, last=Action.COOPERATE)
    falsified_view(s, score_offset=SCORE_FALSIFICATION, flip_move=True)
    assert s.agent_score == 24
    assert s.last_opponent_action() is Action.COOPERATE


def test_3s_leaves_the_move_alone_and_3m_leaves_the_score_alone(builder):
    s = FakeState()
    for text, must_match in (
        (builder.score_falsified_text(s, Framing.SEMANTIC, SCORE_FALSIFICATION),
         "Opponent's last move:"),
        (builder.move_falsified_text(s, Framing.SEMANTIC), "Your score:"),
    ):
        truth_line = [ln for ln in lines(builder.treatment_text(s, Framing.SEMANTIC))
                      if ln.startswith(must_match)][0]
        lie_line = [ln for ln in lines(text) if ln.startswith(must_match)][0]
        assert truth_line == lie_line


# --- the degenerate case ---------------------------------------------------


def test_turn_zero_cannot_be_falsified_and_says_so(builder):
    """No last move exists at turn 0, so 3m renders identically to arm 3.

    Those rows carry no falsification and must be excluded from the contrast,
    exactly as `donor_degenerate` rows are in arm 3c. Including them dilutes
    the effect with unfalsified data.
    """
    s = FakeState(turn=0, last=None)
    assert move_was_falsified(s) is False
    assert (builder.move_falsified_text(s, Framing.SEMANTIC)
            == builder.treatment_text(s, Framing.SEMANTIC))
    assert move_was_falsified(FakeState(last=Action.COOPERATE)) is True


# --- score clamping --------------------------------------------------------


def test_score_stays_inside_the_reachable_range():
    """A negative or impossible score would be a giveaway that the block is
    fabricated, confounding 'false state' with 'obviously false state'. It would
    also risk rendering outside the calibration range and aborting the run."""
    low = falsified_view(FakeState(agent=3), score_offset=-SCORE_FALSIFICATION)
    assert 0 <= low.agent_score <= _MAX_REACHABLE_SCORE
    assert low.agent_score != 3, "a score of 3 can be falsified upward instead"

    high = falsified_view(FakeState(agent=98), score_offset=SCORE_FALSIFICATION)
    assert 0 <= high.agent_score <= _MAX_REACHABLE_SCORE
    assert high.agent_score != 98


# --- token parity, the claim the whole study rests on ----------------------


@pytest.mark.parametrize("arm", [Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE])
@pytest.mark.parametrize("framing", [Framing.SEMANTIC, Framing.ABSTRACT])
def test_new_arms_hold_token_parity(builder, arm, framing):
    """"Cooperate" and "Defect" are different lengths, and so are "24" and "39".
    Parity is restored by padding in build_pair; this asserts it actually is."""
    for state in (FakeState(agent=0, turn=0, last=None),
                  FakeState(agent=24, turn=8, last=Action.COOPERATE),
                  FakeState(agent=97, turn=19, last=Action.DEFECT)):
        t, o = builder.build_pair(arm, state, framing)
        assert len(t.token_ids) == len(o.token_ids) == builder.block_tokens


@pytest.mark.parametrize("arm", [Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE])
def test_new_arms_match_every_other_arm_in_length(builder, arm):
    """Parity within an arm is not enough - 3s must also match 3, 3b, 3c and 3d,
    or a cross-arm contrast compares prompts of different sizes."""
    s = FakeState()
    donor = FakeState(agent=41, opponent=33, turn=8, last=Action.DEFECT)
    lengths = set()
    for a in (Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC,
              Arm.PLACEBO_STALE, arm):
        _, blk = builder.build_pair(
            a, s, Framing.SEMANTIC, donor=donor if a is Arm.PLACEBO_STALE else None)
        lengths.add(len(blk.token_ids))
    assert len(lengths) == 1, f"block lengths differ across arms: {lengths}"


# --- enum wiring -----------------------------------------------------------


def test_new_arms_are_registered_as_block_injecting():
    assert Arm.PLACEBO_SCORE.injects_block
    assert Arm.PLACEBO_MOVE.injects_block


def test_falsifying_arms_are_exactly_the_three_that_lie():
    """3b and 3d inject a block but assert nothing false. Only falsifying arms
    need a displayed-value column, so this set drives what gets logged."""
    liars = {a for a in Arm if a.falsifies_field}
    assert liars == {Arm.PLACEBO_STALE, Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE}


def test_3s_rejects_a_zero_offset(builder):
    """A zero offset would silently produce an exact copy of arm 3 and the cell
    would look like a null result rather than a misconfiguration."""
    with pytest.raises(ValueError, match="non-zero"):
        builder.build_pair(Arm.PLACEBO_SCORE, FakeState(), Framing.SEMANTIC,
                           score_offset=0)