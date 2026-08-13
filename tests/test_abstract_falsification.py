"""Under X/Y the falsification must keep its INFORMATION and lose its WORDS.

WHY THIS FILE EXISTS
    exp6 ran under semantic labels only, and arm 3m injects the literal token
    "Defect" into the context. This project's own exp3 measured labels
    dominating everything else it tested: Llama's baseline defection goes
    0.28-0.31 under Cooperate/Defect to 0.71-0.74 under X/Y, and Qwen's
    container effect REVERSES SIGN between framings (CLAIMS.md D2). So the exp6
    headline - "the model conditions on the last-move field" - is currently
    indistinguishable from "the string 'Defect' raises P(Defect)".

    exp7's abstract arm is the falsification test. It only works if the abstract
    rendering has two properties, and neither is obvious from reading the code:

      1. THE INFORMATION SURVIVES. "Opponent's last move: Y" must be exactly as
         decision-relevant against TFT as "Opponent's last move: Defect" - same
         field, same position, same one-line change, a bijective relabelling of
         the same proposition.

      2. THE LEXICAL FORCE IS GONE. If the word "Defect" appears ANYWHERE in the
         abstract prompt - the rules, the block, the history, the instruction -
         the priming account is not excluded and the cell answers nothing.

    Property 2 is the fragile one. It is a property of the whole assembled
    prompt, not of the block, and it is one careless template edit away from
    being false. It is asserted here on the rendered string.

RELATIONSHIP TO tests/test_field_falsification.py
    That file proves 3s and 3m change exactly one line under SEMANTIC labels.
    This one proves the same thing holds under ABSTRACT labels, and adds the
    lexical-strip property that makes the abstract cell worth running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import (
    Action,
    Arm,
    Framing,
    GameConfig,
    ReadoutMode,
    ScaffoldConfig,
)
from cdx.game import TitForTat, replay
from cdx.runner import instruction_for
from cdx.scaffold import (
    SCORE_FALSIFICATION,
    PromptAssembler,
    ScaffoldBuilder,
    falsified_view,
    render_action,
)

SEMANTIC_TOKENS = ("Cooperate", "Defect")
GAME = GameConfig()
BLOCK_ARMS = [Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC,
              Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE]


class CharTokenizer:
    """One token per character - the same double the existing parity tests use,
    so a failure here is about rendering and never about a merge."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeState:
    """The four attributes the block templates touch."""

    def __init__(self, agent=24, opponent=18, turn=8, last=Action.COOPERATE):
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self.turns = ()
        self._last = last

    def last_opponent_action(self):
        return self._last


@pytest.fixture
def tok():
    return CharTokenizer()


@pytest.fixture
def builder(tok):
    return ScaffoldBuilder(tok, ScaffoldConfig())


@pytest.fixture
def assembler(tok):
    return PromptAssembler(tok, ScaffoldConfig())


def lines(text):
    return [ln for ln in text.split("\n") if ln]


def differing_lines(a, b):
    la, lb = lines(a), lines(b)
    assert len(la) == len(lb), "field count changed - template was altered"
    return [(x, y) for x, y in zip(la, lb) if x != y]


# --- (a) exactly one line differs, under X/Y just as under words -----------


@pytest.mark.parametrize("last", [Action.COOPERATE, Action.DEFECT])
def test_3m_changes_only_the_last_move_line_under_abstract(builder, last):
    s = FakeState(last=last)
    diff = differing_lines(builder.treatment_text(s, Framing.ABSTRACT),
                           builder.move_falsified_text(s, Framing.ABSTRACT))
    assert len(diff) == 1, f"3m changed {len(diff)} lines, expected 1: {diff}"
    assert diff[0][0].startswith("Opponent's last move:")


def test_3s_changes_only_the_score_line_under_abstract(builder):
    s = FakeState()
    diff = differing_lines(
        builder.treatment_text(s, Framing.ABSTRACT),
        builder.score_falsified_text(s, Framing.ABSTRACT, SCORE_FALSIFICATION))
    assert len(diff) == 1, f"3s changed {len(diff)} lines, expected 1: {diff}"
    assert diff[0][0].startswith("Your score:")


def test_the_falsified_move_line_differs_by_exactly_one_character(builder):
    """Under X/Y the entire manipulation is one character. That is the tightest
    possible form of "same information, no lexical force": nothing else in the
    prompt can be carrying the effect, including line length, which changes
    under semantic labels ("Cooperate" is 9 characters, "Defect" is 6)."""
    s = FakeState(last=Action.COOPERATE)
    truth, lie = differing_lines(
        builder.treatment_text(s, Framing.ABSTRACT),
        builder.move_falsified_text(s, Framing.ABSTRACT))[0]
    assert len(truth) == len(lie)
    assert sum(a != b for a, b in zip(truth, lie)) == 1


@pytest.mark.parametrize("true_move,expected", [(Action.COOPERATE, "Y"),
                                                (Action.DEFECT, "X")])
def test_3m_flips_between_the_abstract_labels(builder, true_move, expected):
    line = [ln for ln in lines(builder.move_falsified_text(
        FakeState(last=true_move), Framing.ABSTRACT))
        if ln.startswith("Opponent's last move:")][0]
    assert line.endswith(expected), line


def test_the_flip_is_on_the_action_not_on_the_label(builder):
    """`falsified_view` inverts the ACTION and rendering happens afterwards, so
    the same manipulation is expressed in whatever vocabulary the framing uses.
    If the flip were implemented on the rendered string instead, the abstract
    and semantic arms would be different manipulations and the comparison exp7
    is built on would be invalid."""
    s = FakeState(last=Action.COOPERATE)
    view = falsified_view(s, flip_move=True)
    assert view.last_opponent_action() is Action.DEFECT
    for framing in (Framing.SEMANTIC, Framing.ABSTRACT):
        # The falsified block is the TRUE template rendered over a flipped view,
        # in both vocabularies. One manipulation, two renderings.
        assert (builder.move_falsified_text(s, framing)
                == builder.treatment_text(view, framing))


# --- (2) the lexical force must be gone from the WHOLE prompt --------------


@pytest.mark.parametrize("arm", BLOCK_ARMS)
@pytest.mark.parametrize("n_rounds", [0, 1, 9])
def test_no_semantic_action_word_survives_anywhere_under_abstract(
    assembler, builder, tok, arm, n_rounds
):
    """The property that makes the abstract cell a falsification test rather
    than a second semantic cell. Checked on the ASSEMBLED prompt - rules,
    block, history and instruction - because any one of them could leak the
    word the priming account is about."""
    state = replay(GAME, TitForTat(), [Action.COOPERATE] * n_rounds)
    _, block = builder.build_pair(arm, state, Framing.ABSTRACT)
    for readout in (ReadoutMode.LOGIT, ReadoutMode.SCRATCHPAD):
        text = tok.decode(assembler.assemble(
            game_config=GAME, state=state, framing=Framing.ABSTRACT,
            block=block,
            instruction_suffix=instruction_for(Framing.ABSTRACT, readout)))
        for word in SEMANTIC_TOKENS:
            assert word not in text, (
                f"{word!r} leaked into the abstract prompt via arm {arm.value} "
                f"under {readout.value}; the priming confound is not controlled")


def test_the_semantic_arm_really_does_inject_the_word(builder):
    """The confound being controlled, stated as a test so it cannot be
    described in the paper as smaller than it is: under semantic labels, arm 3m
    puts the literal string "Defect" into a context where the true last move was
    Cooperate."""
    s = FakeState(last=Action.COOPERATE)
    assert "Defect" not in builder.treatment_text(s, Framing.SEMANTIC)
    assert "Defect" in builder.move_falsified_text(s, Framing.SEMANTIC)


def test_abstract_labels_are_a_bijection_of_the_semantic_ones():
    """"Same information, different words" requires the relabelling to be
    one-to-one. If both actions rendered to the same abstract token the field
    would carry no information and the cell would be a null by construction."""
    abstract = {a: render_action(a, Framing.ABSTRACT) for a in Action}
    semantic = {a: render_action(a, Framing.SEMANTIC) for a in Action}
    assert len(set(abstract.values())) == len(Action) == len(set(semantic.values()))
    assert set(abstract.values()).isdisjoint(set(semantic.values()))


# --- parity, which must not move between framings -------------------------


@pytest.mark.parametrize("arm", BLOCK_ARMS)
def test_abstract_blocks_hit_the_same_parity_target(builder, arm):
    """`_derive_block_tokens` sweeps BOTH framings on purpose: a target that
    changed with framing would make the abstract and semantic runs differ in
    prompt length, adding a confound to the factor that is supposed to vary only
    the labels."""
    for state in (FakeState(agent=0, turn=0, last=None),
                  FakeState(agent=24, turn=8, last=Action.COOPERATE),
                  FakeState(agent=97, turn=19, last=Action.DEFECT)):
        t, o = builder.build_pair(arm, state, Framing.ABSTRACT)
        assert len(t.token_ids) == len(o.token_ids) == builder.block_tokens


def test_abstract_and_semantic_blocks_are_the_same_width(builder):
    """Between-framing comparability, at the level the prompt is assembled in.
    "Cooperate" and "X" are different strings; padding to a framing-invariant
    target is what makes the two conditions comparable at all."""
    s = FakeState()
    widths = set()
    for framing in (Framing.SEMANTIC, Framing.ABSTRACT):
        for arm in BLOCK_ARMS:
            widths.add(len(builder.build_pair(arm, s, framing)[1].token_ids))
    assert len(widths) == 1, f"block width varies by framing or arm: {widths}"


# --- the exp7 abstract cell must still be a falsification ------------------


def test_turn_zero_is_still_degenerate_under_abstract(builder):
    """At turn 0 there is no last move to flip, so 3m is byte-identical to arm 3
    and those rows carry no manipulation - under X/Y exactly as under words.
    They are excluded from the contrast via donor_degenerate."""
    s = FakeState(turn=0, last=None)
    assert (builder.move_falsified_text(s, Framing.ABSTRACT)
            == builder.treatment_text(s, Framing.ABSTRACT))
    assert "none" in builder.treatment_text(s, Framing.ABSTRACT)
