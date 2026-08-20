"""Guards the score-field rendering and the ID-level parity mechanism.

WHY THIS FILE EXISTS
    The treatment block rendered scores zero-padded ("012", "048") to give the
    block a constant token count. Run `sweep` showed Llama-3.1-8B reading the
    leading zero rather than the value: shown "Your score: 024" it answered "0".
    49.7% of treatment score-probe failures are attributable to that one format
    spec.

    The obvious fix - space-padding - does not work either. Character padding
    cannot control token count, because BPE decides how a padded string splits
    and that varies by value and by tokenizer. scripts/tokenizer_check.py
    reports lengths=[5, 6] for " {v:>3d}" on Llama-3.1.

    So: render naturally, and enforce constant size on TOKEN IDS. These tests
    guard both halves of that.

WHAT A STUB TOKENIZER CAN AND CANNOT PROVE
    CharTokenizer maps one character to one token, so it CANNOT tell you whether
    a real BPE tokenizer produces a constant length. It is used here to test the
    PADDING LOGIC. The parity TARGET must be derived against the real tokenizer
    with scripts/calibrate_block.py before any paid run. Do not treat a green
    test suite as evidence that parity holds on the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.backends import CharTokenizer
from cdx.config import Action, Arm, Framing, ScaffoldConfig
from cdx.scaffold import ScaffoldBuilder, TokenParityError

MAX_SCORE = 120
MAX_TURN = 20


class StubState:
    """Minimal GameState surface the scaffold templates touch. Deliberately not
    the real GameState, so this file fails on scaffold regressions only."""

    def __init__(self, agent: int, opponent: int, turn: int,
                 last: Action | None) -> None:
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self._last = last
        self.turns = ()

    def last_opponent_action(self) -> Action | None:
        return self._last


def states():
    for turn in range(0, MAX_TURN + 1):
        for score in (0, 1, 3, 9, 10, 12, 48, 99, 100, MAX_SCORE):
            for last in (None, Action.COOPERATE, Action.DEFECT):
                yield StubState(score, max(score - 3, 0), turn, last)


@pytest.fixture
def builder():
    """Target auto-derived from the tokeniser at construction."""
    return ScaffoldBuilder(CharTokenizer(), ScaffoldConfig())


# Alias kept so the rendering tests below read naturally.
legacy = builder


# --- rendering -------------------------------------------------------------


def test_score_field_is_not_zero_padded(legacy):
    """The regression this whole file exists for."""
    for score in (0, 3, 12, 48):
        text = legacy.treatment_text(
            StubState(score, score, 1, Action.COOPERATE), Framing.SEMANTIC
        )
        line = next(l for l in text.splitlines() if l.startswith("Your score:"))
        value = line.split(":", 1)[1].strip()
        assert value == str(score), (
            f"score {score} rendered as {value!r}. A leading zero makes the "
            f"model report the zero instead of the number."
        )


def test_every_numeric_field_parses_back(legacy):
    for state in states():
        text = legacy.treatment_text(state, Framing.SEMANTIC)
        fields = dict(
            (k.strip(), v.strip())
            for k, v in (l.split(":", 1) for l in text.splitlines() if ":" in l)
        )
        assert int(fields["Your score"]) == state.agent_score
        assert int(fields["Opponent score"]) == state.opponent_score
        assert int(fields["Rounds played"]) == state.turn_index


def test_no_character_padding_anywhere_in_the_block(legacy):
    """Trailing spaces are the signature of the retired ljust() approach."""
    for state in states():
        for line in legacy.treatment_text(state, Framing.SEMANTIC).splitlines():
            assert line == line.rstrip(), f"trailing padding in {line!r}"


def test_block_structure_is_unchanged(legacy):
    """Field names and line count are part of the pre-registered template.
    Only the padding may change."""
    lines = legacy.treatment_text(
        StubState(12, 12, 4, Action.COOPERATE), Framing.SEMANTIC
    ).splitlines()
    assert lines[0] == "[STATE]"
    assert lines[1].startswith("Your score:")
    assert lines[2].startswith("Opponent score:")
    assert lines[3].startswith("Opponent's last move:")
    assert lines[4].startswith("Rounds played:")
    assert len(lines) == 5


# --- parity ----------------------------------------------------------------


@pytest.mark.parametrize(
    "arm", [Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC]
)
def test_target_holds_for_every_state(builder, arm):
    for state in states():
        t, o = builder.build_pair(arm, state, Framing.SEMANTIC)
        assert t.n_tokens == o.n_tokens == builder.block_tokens, (
            f"{arm.value} at score={state.agent_score} turn={state.turn_index}: "
            f"{t.n_tokens} vs {o.n_tokens}, target {builder.block_tokens}"
        )


def test_target_is_derived_from_the_tokenizer_not_the_config():
    """The bug this replaced: a constant in config was correct for Llama-3.1
    (34) and wrong for a character-level tokeniser (84). Any hardcoded value
    silently breaks the multi-model design."""
    auto = ScaffoldBuilder(CharTokenizer(), ScaffoldConfig())
    assert auto.block_tokens > 34, (
        "a character tokeniser needs far more tokens than a BPE one; if this "
        "fails the target is not being derived from the tokeniser"
    )


def test_treatment_is_padded_too(builder):
    """The change from the old design. Previously only the placebo was padded,
    so the treatment's natural length WAS the target and varied by state."""
    t, _ = builder.build_pair(
        Arm.PLACEBO_NONDIAGNOSTIC,
        StubState(3, 3, 1, Action.COOPERATE),
        Framing.SEMANTIC,
    )
    assert t.pad_tokens_added > 0
    assert t.n_tokens == builder.block_tokens


def test_stale_donor_never_exceeds_target(builder):
    """Arm 3c renders the treatment template from a DIFFERENT episode. A
    constant target is what makes that safe."""
    donor = StubState(MAX_SCORE, MAX_SCORE, MAX_TURN, Action.DEFECT)
    for state in states():
        t, o = builder.build_pair(
            Arm.PLACEBO_STALE, state, Framing.SEMANTIC, donor=donor
        )
        assert t.n_tokens == o.n_tokens == builder.block_tokens


def test_donor_longer_than_its_recipient_is_handled(builder):
    """The case that broke when rendering became natural: a donor five rounds
    ahead renders longer than a turn-0 recipient's own treatment block, so the
    target cannot be the treatment's length."""
    t, o = builder.build_pair(
        Arm.PLACEBO_STALE,
        StubState(0, 0, 0, None),
        Framing.SEMANTIC,
        donor=StubState(MAX_SCORE, MAX_SCORE, MAX_TURN, Action.DEFECT),
    )
    assert t.n_tokens == o.n_tokens
    assert t.pad_tokens_added > o.pad_tokens_added, "treatment is shorter here"


def test_a_too_small_configured_target_is_rejected_at_construction():
    """A silent parity violation invalidates the causal estimate. Catch it when
    the builder is created, not thousands of episodes into a paid run."""
    with pytest.raises(TokenParityError):
        ScaffoldBuilder(CharTokenizer(), ScaffoldConfig(treatment_block_tokens=5))


def test_an_adequate_configured_target_is_honoured():
    auto = ScaffoldBuilder(CharTokenizer(), ScaffoldConfig()).block_tokens
    pinned = ScaffoldBuilder(
        CharTokenizer(), ScaffoldConfig(treatment_block_tokens=auto + 10)
    )
    assert pinned.block_tokens == auto + 10


def test_placebos_are_mostly_content_not_filler(builder):
    """Token parity is necessary but not sufficient.

    exp1's 3b was two content lines padded with 17 filler tokens into a
    32-token block, 47% content, against a treatment carrying no filler at all.
    Counts matched; the stimuli did not. Arm 3d was worse, which would have made
    the 3b-vs-3d contrast nearly vacuous. (From exp2 the parity target is 34
    tokens and the treatment carries 2 filler tokens, 94%.)
    """
    state = StubState(12, 12, 4, Action.COOPERATE)
    for arm in (Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC):
        _, block = builder.build_pair(arm, state, Framing.SEMANTIC)
        content = 1.0 - block.pad_tokens_added / block.n_tokens
        assert content >= 0.75, (
            f"{arm.value} is only {content:.0%} content; it is mostly padding "
            f"and is not a comparable stimulus to the treatment"
        )