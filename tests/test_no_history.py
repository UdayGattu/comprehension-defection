"""The no-history condition must remove [HISTORY] and NOTHING else.

WHY THIS FILE EXISTS
    Three reviewers made the same observation: `PromptAssembler.assemble` has
    always rendered [HISTORY] with every round in it, one section below the
    injected block. So arms 3c/3s/3m were never false-state manipulations - they
    were CONTRADICTION manipulations, with the truth sitting in the same context
    window. The rival account ("models discount a locally contradicted claim,
    and discount it more when the contradiction is cheap to verify") predicts
    every feature of the exp6 result, including the score/move asymmetry.

    exp7 removes the refutation. That makes the flag a load-bearing part of a
    causal claim, and it has exactly two ways to be wrong:

      1. IT REMOVES TOO MUCH. If the block, the rules, the instruction or the
         block's POSITION move as well, the no-history cells are not comparable
         to exp6's and the contrast measures four things at once. Position is
         the worst of these - lost-in-the-middle effects produce >30% swings
         from placement alone.

      2. IT LEAKS WHEN OFF. `assemble` is on the path of every prompt in every
         experiment. exp1-exp6 are on the record with committed databases and
         published numbers; if the default rendering changed by one token, none
         of them reproduce from the repository that claims to produce them.

    So the tests below do not check that the flag "works". They pin the exact
    token sequence: with the flag off, against a frozen re-implementation of the
    pre-flag assembler; with it on, against that same sequence minus one
    contiguous run of tokens.

WHAT A STUB TOKENIZER CAN AND CANNOT PROVE
    CharTokenizer is one token per character, so it cannot tell you what a BPE
    vocabulary does at the section seams. It CAN prove that the assembler emits
    the same sections in the same order, which is the property under test - the
    block's token IDs are inserted raw and never re-encoded.
"""

from __future__ import annotations

import inspect
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
from cdx.game import AlwaysCooperate, TitForTat, replay
from cdx.runner import instruction_for
from cdx.scaffold import (
    HISTORY_HEADER,
    PromptAssembler,
    ScaffoldBuilder,
)

FRAMINGS = [Framing.SEMANTIC, Framing.ABSTRACT]
BLOCK_ARMS = [Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC,
              Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE]


class CharTokenizer:
    """One token per character. Every character shows, so an off-by-one in the
    section order is visible rather than absorbed by a merge."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tok():
    return CharTokenizer()


@pytest.fixture
def builder(tok):
    return ScaffoldBuilder(tok, ScaffoldConfig())


@pytest.fixture
def assembler(tok):
    return PromptAssembler(tok, ScaffoldConfig())


GAME = GameConfig()


def state_after(n_rounds: int, opponent=None):
    """A real GameState, not a stub. The history section renders `state.turns`,
    so a stub with `turns = ()` would make every test below vacuous."""
    return replay(GAME, opponent or TitForTat(), [Action.COOPERATE] * n_rounds)


def block_for(builder, arm, state, framing):
    donor = state_after(3) if arm is Arm.PLACEBO_STALE else None
    return builder.build_pair(arm, state, framing, donor=donor)[1]


def suffix(framing):
    return instruction_for(framing, ReadoutMode.LOGIT)


def frozen_assemble(assembler, tok, state, framing, block, instruction):
    """The pre-flag implementation, reproduced verbatim.

    This is the reference exp1-exp6 were run against. It is duplicated here on
    purpose: a test that called the real `assemble` with `include_history=True`
    would pass no matter what the real one did to the default path.
    """
    sections = [
        tok.encode(assembler._rules(GAME, framing)),
        tok.encode(assembler._history_section(state, framing)),
        tok.encode(instruction),
    ]
    if block is not None:
        sections.insert(1, list(block.token_ids))
    return [tid for section in sections for tid in section]


# --- (c) exp1-exp6 must render byte-identically with the flag off -----------


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS + [Arm.PLACEBO_STALE])
@pytest.mark.parametrize("n_rounds", [0, 1, 7, 19])
def test_default_rendering_matches_the_pre_flag_assembler(
    assembler, builder, tok, framing, arm, n_rounds
):
    """The invariant the whole flag is subordinate to.

    Every prompt in exp1-exp6 came out of the three-section path below. If this
    fails, 300,000 committed episodes stop being reproducible from HEAD.
    """
    state = state_after(n_rounds)
    block = block_for(builder, arm, state, framing)
    got = assembler.assemble(
        game_config=GAME, state=state, framing=framing, block=block,
        instruction_suffix=suffix(framing))
    want = frozen_assemble(assembler, tok, state, framing, block, suffix(framing))
    assert got == want


@pytest.mark.parametrize("framing", FRAMINGS)
def test_default_rendering_matches_for_the_no_block_arm(assembler, tok, framing):
    """Arm 1 has no block, so it exercises a different branch of the insert."""
    state = state_after(5)
    got = assembler.assemble(
        game_config=GAME, state=state, framing=framing, block=None,
        instruction_suffix=suffix(framing))
    assert got == frozen_assemble(assembler, tok, state, framing, None,
                                  suffix(framing))


def test_the_flag_defaults_to_history_present(assembler):
    """A default of False would silently rewrite every historical experiment,
    and every call site that does not pass the argument - including
    `cdx.runner.Runner`, which is what the laptop pipeline uses."""
    param = inspect.signature(PromptAssembler.assemble).parameters["include_history"]
    assert param.default is True
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_runner_path_still_renders_history(assembler, builder):
    """`cdx.runner.Runner.run_episode` calls `assemble` without the argument.
    Pinned because a future refactor that threads the flag through the Runner
    must not change what the default does."""
    state = state_after(4)
    ids = assembler.assemble(
        game_config=GAME, state=state, framing=Framing.SEMANTIC,
        block=block_for(builder, Arm.TREATMENT, state, Framing.SEMANTIC),
        instruction_suffix=suffix(Framing.SEMANTIC))
    assert HISTORY_HEADER in CharTokenizer().decode(ids)


# --- (b) the flag removes the history section and nothing else --------------


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS)
@pytest.mark.parametrize("n_rounds", [0, 1, 7, 19])
def test_no_history_removes_exactly_one_contiguous_section(
    assembler, builder, tok, framing, arm, n_rounds
):
    """The strongest available statement: the no-history prompt IS the with-
    history prompt with one contiguous run of tokens deleted, and that run is
    exactly the history section. Not "similar", not "shorter" - identical
    either side of the cut."""
    state = state_after(n_rounds)
    block = block_for(builder, arm, state, framing)
    kw = dict(game_config=GAME, state=state, framing=framing, block=block,
              instruction_suffix=suffix(framing))

    with_hist = assembler.assemble(**kw, include_history=True)
    without = assembler.assemble(**kw, include_history=False)
    section = tok.encode(assembler._history_section(state, framing))

    # Where the history sits: after the rules and the block, before the
    # instruction. Computed from the parts rather than searched for, so a
    # section that moved would fail here rather than be found somewhere else.
    start = len(tok.encode(assembler._rules(GAME, framing))) + len(block.token_ids)
    end = start + len(section)

    assert with_hist[start:end] == section, "history is not where it should be"
    assert with_hist[:start] + with_hist[end:] == without, (
        "the no-history prompt is not the with-history prompt minus exactly "
        "the history section")
    assert len(without) == len(with_hist) - len(section)


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS)
def test_the_block_is_byte_identical_and_in_the_same_place(
    assembler, builder, tok, framing, arm
):
    """The block is the treatment. If removing the history shifted it by one
    token, or re-encoded it, the exp7 cells would not be comparable to exp6's
    and no contrast in the run would be interpretable."""
    state = state_after(9)
    block = block_for(builder, arm, state, framing)
    ids = list(block.token_ids)
    rules = tok.encode(assembler._rules(GAME, framing))
    kw = dict(game_config=GAME, state=state, framing=framing, block=block,
              instruction_suffix=suffix(framing))

    for include in (True, False):
        prompt = assembler.assemble(**kw, include_history=include)
        start = len(rules)
        assert prompt[start:start + len(ids)] == ids, (
            f"block moved or changed with include_history={include}")
        assert prompt[:start] == rules, "the rules section moved"


@pytest.mark.parametrize("framing", FRAMINGS)
def test_history_header_is_present_iff_requested(assembler, builder, tok, framing):
    state = state_after(6)
    block = block_for(builder, Arm.TREATMENT, state, framing)
    kw = dict(game_config=GAME, state=state, framing=framing, block=block,
              instruction_suffix=suffix(framing))
    assert HISTORY_HEADER in tok.decode(
        assembler.assemble(**kw, include_history=True))
    assert HISTORY_HEADER not in tok.decode(
        assembler.assemble(**kw, include_history=False))


@pytest.mark.parametrize("framing", FRAMINGS)
def test_no_round_lines_survive_the_removal(assembler, builder, tok, framing):
    """The header is not the only thing that could leak. `render_history` emits
    one "Round n: you=... opponent=..." line per turn, and those lines are the
    refutation the condition exists to remove."""
    state = state_after(12)
    block = block_for(builder, Arm.TREATMENT, state, framing)
    text = tok.decode(assembler.assemble(
        game_config=GAME, state=state, framing=framing, block=block,
        instruction_suffix=suffix(framing), include_history=False))
    assert "Round 1:" not in text
    assert "opponent=" not in text


# --- the property the driver's gate reads ----------------------------------


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS + [None])
def test_prompt_width_is_turn_invariant_without_history(
    assembler, builder, tok, framing, arm
):
    """`scripts/exp7_confounds.sh` audits the whole run from
    `turns.prompt_tokens`: one distinct width per cell means the history was
    absent. That gate is only valid if history is the ONLY section that grows
    with the turn index. Asserted here so the gate cannot quietly become a
    tautology - or a false alarm - if another section starts tracking the turn.
    """
    widths = set()
    for n in (0, 1, 5, 13, 19):
        state = state_after(n)
        block = None if arm is None else block_for(builder, arm, state, framing)
        widths.add(len(assembler.assemble(
            game_config=GAME, state=state, framing=framing, block=block,
            instruction_suffix=suffix(framing), include_history=False)))
    assert len(widths) == 1, f"prompt width still moves with the turn: {widths}"


@pytest.mark.parametrize("framing", FRAMINGS)
def test_prompt_width_grows_with_history(assembler, builder, tok, framing):
    """The other half of the gate: with history on, width MUST grow, or the
    'constant width' test above proves nothing."""
    widths = [
        len(assembler.assemble(
            game_config=GAME, state=state, framing=framing,
            block=block_for(builder, Arm.TREATMENT, state, framing),
            instruction_suffix=suffix(framing), include_history=True))
        for state in (state_after(0), state_after(5), state_after(19))
    ]
    assert widths[0] < widths[1] < widths[2]


# --- position, the confound that would be invisible ------------------------


def test_block_after_history_is_refused_when_history_is_gone(builder, tok):
    """With `insertion_index=2` the block sits AFTER the history. Remove the
    history and index 2 is still in range - it would place the block after the
    instruction instead, a different position with no error. Refuse."""
    config = ScaffoldConfig(insertion_index=2)
    assembler = PromptAssembler(tok, config)
    state = state_after(4)
    block = ScaffoldBuilder(tok, config).build_pair(
        Arm.TREATMENT, state, Framing.SEMANTIC)[1]
    kw = dict(game_config=GAME, state=state, framing=Framing.SEMANTIC,
              block=block, instruction_suffix=suffix(Framing.SEMANTIC))

    assembler.assemble(**kw, include_history=True)          # legal
    with pytest.raises(ValueError, match="after"):
        assembler.assemble(**kw, include_history=False)


# --- the bonus conditions the removal creates ------------------------------


def test_arm_1_without_history_contains_no_state_at_all(assembler, tok):
    """With history gone, arm 1 is a genuine state-DEPRIVATION condition rather
    than "the same state, minus a summary". That is what makes arm-1 CPR
    interpretable as a floor, and what the driver's deprivation gate reads."""
    state = replay(GAME, AlwaysCooperate(), [Action.DEFECT] * 6)
    assert state.agent_score == 30, "payoffs changed; pick a non-colliding score"
    text = tok.decode(assembler.assemble(
        game_config=GAME, state=state, framing=Framing.SEMANTIC, block=None,
        instruction_suffix=suffix(Framing.SEMANTIC), include_history=False))
    assert HISTORY_HEADER not in text
    assert "[STATE]" not in text
    assert "Round 1:" not in text
    # 30 is not one of the numbers the rules section prints (5/3/1/0/20), so its
    # absence means the score is genuinely unavailable rather than coincidental.
    assert str(state.agent_score) not in text


def test_arm_3_without_history_still_carries_every_field(assembler, builder, tok):
    """The other side of it: arm 3 must still state the score, the last move and
    the round count, or the deprivation contrast has no treatment."""
    state = state_after(7)
    text = tok.decode(assembler.assemble(
        game_config=GAME, state=state, framing=Framing.SEMANTIC,
        block=block_for(builder, Arm.TREATMENT, state, Framing.SEMANTIC),
        instruction_suffix=suffix(Framing.SEMANTIC), include_history=False))
    assert f"Your score: {state.agent_score}" in text
    assert "Opponent's last move: Cooperate" in text
    assert f"Rounds played: {state.turn_index}" in text
