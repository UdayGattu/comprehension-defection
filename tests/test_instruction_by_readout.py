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
from cdx.runner import _INSTRUCTION, _INSTRUCTION_SCRATCHPAD, instruction_for


FRAMINGS = [Framing.SEMANTIC, Framing.ABSTRACT]


# --- the regression --------------------------------------------------------


@pytest.mark.parametrize("framing", FRAMINGS)
def test_scratchpad_does_not_demand_a_single_token(framing):
    """"exactly one word" is precisely what suppressed the reasoning."""
    text = instruction_for(framing, ReadoutMode.SCRATCHPAD).lower()
    assert "think step by step" in text
    assert not text.startswith("\nrespond with exactly one")


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
    every contrast in the study."""
    import inspect
    params = list(inspect.signature(instruction_for).parameters)
    assert params == ["framing", "readout"]


@pytest.mark.parametrize("readout", list(ReadoutMode))
def test_action_labels_match_the_framing(readout):
    """Abstract must name X/Y, semantic must name Cooperate/Defect. Naming the
    wrong pair produced a 100% off-task rate once already."""
    abstract = instruction_for(Framing.ABSTRACT, readout)
    semantic = instruction_for(Framing.SEMANTIC, readout)
    assert "X or Y" in abstract
    assert "Cooperate or Defect" in semantic
    assert "Cooperate" not in abstract
    assert "X or Y" not in semantic


def test_every_framing_is_covered():
    """A missing key would raise mid-run, after a model load."""
    for framing in Framing:
        for readout in ReadoutMode:
            assert instruction_for(framing, readout)
    assert set(_INSTRUCTION) == set(_INSTRUCTION_SCRATCHPAD) == set(Framing)