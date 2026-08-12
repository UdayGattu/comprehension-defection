"""The instruction line must depend on how the action is read.

THE BUG THIS EXISTS TO PREVENT
    The LOGIT instruction ends "Respond with exactly one word: Cooperate or
    Defect." Under SCRATCHPAD the model is asked to generate 128 tokens of
    reasoning from that same prompt - so it emits "Cooperate" and stops. The
    first exp4 smoke run produced exactly that: every scratchpad was the bare
    answer, zero longer than 20 characters, and the CoT condition silently
    became an expensive re-read of a decision already made.

    Nothing about that fails loudly. Defection rates, CPR and off-task all look
    normal, because the readout still works - it just is not measuring
    chain-of-thought. Only the scratchpad-length gate caught it.

WHAT MUST HOLD
    The instruction varies with READOUT, and within a readout it is IDENTICAL
    ACROSS ARMS. The second property is what keeps the treatment/placebo
    contrast clean; the first is what makes the CoT condition real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import Framing, ReadoutMode
from cdx.runner import (
    DEFAULT_SCRATCHPAD_PROMPT,
    SCRATCHPAD_PROMPTS,
    _INSTRUCTION,
    _INSTRUCTION_SCRATCHPAD,
    _INSTRUCTION_SCRATCHPAD_MINIMAL,
    instruction_for,
)


FRAMINGS = [Framing.SEMANTIC, Framing.ABSTRACT]
VARIANTS = sorted(SCRATCHPAD_PROMPTS)


# --- the regression --------------------------------------------------------


@pytest.mark.parametrize("framing", FRAMINGS)
def test_scratchpad_asks_for_reasoning(framing):
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD).lower()
    assert "step by step" in text


@pytest.mark.parametrize("framing", FRAMINGS)
def test_scratchpad_specifies_no_output_format(framing):
    """The regression this file exists for.

    A format clause is what suppressed the reasoning: Qwen2.5 obeyed "exactly
    one word" and emitted 9 characters on every turn, while Llama-3.1 produced
    ~520. The action is read from a logit position, so the instruction has no
    need to describe an output format at all.
    """
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD).lower()
    for banned in ("exactly one", "one word", "one character", "respond with"):
        assert banned not in text, f"format clause {banned!r} suppresses reasoning"


@pytest.mark.parametrize("framing", FRAMINGS)
def test_scratchpad_instruction_names_no_action(framing):
    """Naming Cooperate/Defect/X/Y here would plant a lexical cue - the very
    thing exp3 showed drives behaviour - and it would differ by framing."""
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD)
    for word in ("Cooperate", "Defect", " X ", " Y "):
        assert word not in text


def test_scratchpad_opener_is_neutral():
    """The prefill conditions every scratchpad, so it must not favour an
    action or vary by condition."""
    from cdx.backends_vllm import SCRATCHPAD_OPENER
    assert SCRATCHPAD_OPENER.strip()
    for word in ("cooperate", "defect", "x", "y"):
        assert word not in SCRATCHPAD_OPENER.lower().split()


def test_scratchpad_cue_demands_one_word_but_names_none():
    """The format demand lives in the CUE, not the instruction.

    In the instruction it suppressed reasoning (Qwen answered and stopped). A
    bare cue without it left Qwen at ~100% off-task: verbose markdown reasoning
    truncated mid-structure, and the model continued the list instead of
    answering. Llama was unaffected either way - so a weak cue works for one
    model and silently fails for another.

    It must still name no action, or it becomes a lexical cue in the abstract
    condition.
    """
    from cdx.backends_vllm import SCRATCHPAD_CUE
    assert "one word" in SCRATCHPAD_CUE.lower()
    for word in ("Cooperate", "Defect", " X", " Y"):
        assert word not in SCRATCHPAD_CUE


@pytest.mark.parametrize("framing", FRAMINGS)
def test_logit_still_demands_one_token(framing):
    """LOGIT reads a single position and must keep its original instruction, or
    exp4's LOGIT arm stops being comparable to exp3."""
    assert instruction_for(framing, ReadoutMode.LOGIT) == _INSTRUCTION[framing]
    assert "exactly one" in _INSTRUCTION[framing].lower()


@pytest.mark.parametrize("framing", FRAMINGS)
def test_the_two_readouts_differ(framing):
    assert (instruction_for(framing, ReadoutMode.LOGIT)
            != instruction_for(framing, ReadoutMode.SCRATCHPAD))


# --- what must NOT vary ----------------------------------------------------


@pytest.mark.parametrize("readout", list(ReadoutMode))
def test_instruction_is_independent_of_arm(readout):
    """instruction_for takes no arm argument, so the suffix cannot differ
    between treatment and placebo. Pinned because a future signature change
    could quietly introduce an arm-dependent instruction, which would confound
    every contrast in the study.

    Previously this asserted an exact parameter list, which made it fire on the
    addition of scratchpad_prompt - a parameter that is fine. The invariant was
    never "these exact parameters"; it is "nothing arm-shaped".
    """
    import inspect
    params = list(inspect.signature(instruction_for).parameters)
    assert params[:2] == ["framing", "readout"]
    for banned in ("arm", "state", "turn", "history", "opponent", "score"):
        assert not any(banned in p for p in params), f"{banned!r} in {params}"


def test_logit_action_labels_match_the_framing():
    """LOGIT names the options, and must name the RIGHT pair - the wrong pair
    produced a 100% off-task rate once already.

    SCRATCHPAD deliberately names none; see
    test_scratchpad_instruction_names_no_action.
    """
    abstract = instruction_for(Framing.ABSTRACT, ReadoutMode.LOGIT)
    semantic = instruction_for(Framing.SEMANTIC, ReadoutMode.LOGIT)
    assert "X or Y" in abstract
    assert "Cooperate or Defect" in semantic
    assert "Cooperate" not in abstract
    assert "X or Y" not in semantic


def test_scratchpad_instruction_carries_no_lexical_cue():
    """The two framings differ only in "opponent" vs "other player" - neither
    names an action. So the instruction cannot itself be the lexical difference
    between the semantic and abstract conditions, which is the difference the
    experiment is trying to measure."""
    a = instruction_for(Framing.ABSTRACT, ReadoutMode.SCRATCHPAD)
    s = instruction_for(Framing.SEMANTIC, ReadoutMode.SCRATCHPAD)
    assert "step by step" in a and "step by step" in s
    for text in (a, s):
        for word in ("Cooperate", "Defect", " X ", " Y "):
            assert word not in text


def test_every_framing_is_covered():
    """A missing key would raise mid-run, after a model load."""
    for framing in Framing:
        for readout in ReadoutMode:
            assert instruction_for(framing, readout)
    assert set(_INSTRUCTION) == set(_INSTRUCTION_SCRATCHPAD) == set(Framing)


# --- the MINIMAL variant, and the horizon confound it measures --------------
#
# exp4's GUIDED prompt names "how many rounds remain". Finite horizons induce
# backward induction, which is the textbook argument for defecting from round
# one - so exp4's 6x jump in defection under SCRATCHPAD cannot be attributed to
# reasoning. MINIMAL names nothing, which is what makes the comparison possible.


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("framing", FRAMINGS)
def test_every_variant_asks_for_reasoning(framing, variant):
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD, variant).lower()
    assert "step by step" in text


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("framing", FRAMINGS)
def test_no_variant_specifies_an_output_format(framing, variant):
    """The property that made Qwen emit 9 characters per turn must hold for
    every variant, not only the one that happened to be tested first."""
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD, variant).lower()
    for banned in ("exactly one", "one word", "one character", "respond with",
                   "format your"):
        assert banned not in text, f"{variant}: format clause {banned!r}"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("framing", FRAMINGS)
def test_no_variant_names_an_action(framing, variant):
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD, variant)
    for word in ("Cooperate", "Defect", " X ", " Y "):
        assert word not in text, f"{variant}: lexical cue {word!r}"


@pytest.mark.parametrize("framing", FRAMINGS)
def test_minimal_names_no_decision_relevant_content(framing):
    """THE POINT OF THE VARIANT.

    GUIDED points the model at the state, the opponent and the horizon. Each is
    a rival explanation for exp4's cross-readout result: horizon salience
    invites backward induction, and "reason about the current state" replaces
    passive lexical priming with active attention. MINIMAL must name none of
    them or it measures nothing new.
    """
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD, "minimal").lower()
    for banned in ("round", "remain", "opponent", "other player", "state",
                   "score", "history", "behaviour", "behavior", "turn"):
        assert banned not in text, f"minimal leaks {banned!r}: {text!r}"


def test_minimal_is_identical_across_framings():
    """Because it names nothing, MINIMAL can be byte-identical across framings -
    unlike GUIDED, which must say "opponent" or "other player". Under MINIMAL
    the two framings therefore differ ONLY in the scaffold, so the instruction
    contributes no part of the semantic/abstract gap."""
    assert (_INSTRUCTION_SCRATCHPAD_MINIMAL[Framing.ABSTRACT]
            == _INSTRUCTION_SCRATCHPAD_MINIMAL[Framing.SEMANTIC])


def test_guided_still_names_the_horizon():
    """Pinned deliberately. GUIDED is exp4's prompt and must not be silently
    'fixed' - exp4's databases were produced by this exact wording, and the
    minimal-vs-guided contrast is only interpretable if guided stays put."""
    text = instruction_for(Framing.SEMANTIC, ReadoutMode.SCRATCHPAD, "guided")
    assert "how many rounds remain" in text


def test_variants_actually_differ():
    for framing in FRAMINGS:
        assert (instruction_for(framing, ReadoutMode.SCRATCHPAD, "minimal")
                != instruction_for(framing, ReadoutMode.SCRATCHPAD, "guided"))


def test_default_is_guided_so_exp4_reproduces():
    """A later commit must reproduce exp4 without extra flags, or the run's
    recorded git_commit stops being sufficient provenance."""
    assert DEFAULT_SCRATCHPAD_PROMPT == "guided"
    for framing in FRAMINGS:
        assert (instruction_for(framing, ReadoutMode.SCRATCHPAD)
                == _INSTRUCTION_SCRATCHPAD[framing])


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_is_ignored_under_logit(variant):
    """LOGIT generates no reasoning, so the variant must not reach the prompt.
    If it did, exp4's LOGIT cells would stop being comparable to exp3's."""
    for framing in FRAMINGS:
        assert (instruction_for(framing, ReadoutMode.LOGIT, variant)
                == _INSTRUCTION[framing])


def test_unknown_variant_fails_loudly():
    """Must raise at prompt-assembly time, not silently fall back to guided -
    a typo'd flag would otherwise produce a database labelled 'minimal' that
    contains guided prompts."""
    with pytest.raises(ValueError, match="unknown scratchpad prompt"):
        instruction_for(Framing.SEMANTIC, ReadoutMode.SCRATCHPAD, "mimimal")


def test_every_variant_covers_every_framing():
    """A missing key raises after a model load - ten minutes into a paid run."""
    for name, table in SCRATCHPAD_PROMPTS.items():
        assert set(table) == set(Framing), name