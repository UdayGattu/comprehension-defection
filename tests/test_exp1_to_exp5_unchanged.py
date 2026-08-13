"""Adding arms 3s and 3m must not change what exp1-exp5 would render.

WHY THIS FILE EXISTS
    exp1 through exp5 are on the record. Their databases are committed, their
    numbers are in EXPERIMENTS.md, and `run_meta.git_commit` points at code that
    now contains two extra arms. Anyone re-running exp3 from HEAD must get the
    same prompts, byte for byte, or the published numbers stop being
    reproducible from the repository that claims to produce them.

    The danger is not the new arms themselves - it is the shared machinery they
    touch. One mechanism in particular:

        PARITY TARGET CONTAMINATION
        `block_tokens` is derived as the longest block any reachable state
        produces. Every arm is padded UP to it. If a new arm renders longer than
        the old maximum and gets added to that derivation, the target rises, and
        EVERY block in EVERY arm of EVERY experiment gets wider - including the
        five already run. Prompt lengths shift, and the study's own exp3->exp4
        measurement says a change of that kind moves a causal estimate by up to
        0.04, against effects as small as 0.017.

    So the invariant is not "the new arms have parity". It is "the new arms fit
    inside the target the old arms already established, and the derivation never
    learned about them."

WHAT THIS DOES NOT COVER
    It cannot detect a change in the real tokenisers' targets (34 Llama, 39
    Qwen, 45 Mistral) because those need the real tokenisers. It pins the
    mechanism that decides them instead, which is what a change would have to go
    through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import Action, Arm, Framing, ScaffoldConfig
from cdx.scaffold import ScaffoldBuilder, _calibration_states


class Char:
    """One token per character - the harshest possible parity test."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class St:
    def __init__(self, a, o, t, last):
        self.agent_score, self.opponent_score = a, o
        self.turn_index, self.turns, self._last = t, (), last

    def last_opponent_action(self):
        return self._last


ORIGINAL_ARMS = (Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC,
                 Arm.PLACEBO_STALE, Arm.PLACEBO_SYNTACTIC)
NEW_ARMS = (Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE)

STATES = [St(0, 0, 0, None),
          St(24, 24, 8, Action.COOPERATE),
          St(48, 33, 14, Action.DEFECT),
          St(97, 61, 19, Action.DEFECT)]


@pytest.fixture
def builder():
    return ScaffoldBuilder(Char(), ScaffoldConfig())


# --- the invariant that matters --------------------------------------------


def test_parity_target_is_derived_from_the_original_blocks_only(builder):
    """The derivation must consider exactly the three block types that existed
    when exp1-exp5 ran. If a fourth is ever added to it, every historical prompt
    width changes and nothing in the repo reproduces."""
    longest = 0
    for state in _calibration_states():
        for framing in (Framing.SEMANTIC, Framing.ABSTRACT):
            for text in (builder.treatment_text(state, framing),
                         builder.nondiagnostic_text(state, framing),
                         builder.syntactic_text(state, framing)):
                longest = max(longest, len(builder.tokenizer.encode(text)))
    assert builder.block_tokens == longest + 2, (
        "block_tokens no longer equals the pre-exp6 derivation. Every prompt in "
        "every historical experiment just changed width."
    )


def test_new_arms_fit_inside_the_existing_target(builder):
    """3s and 3m must never be the longest block. If either were, the honest fix
    would be raising the target - which would invalidate exp1-exp5 - so they are
    designed to render no wider than arm 3 plus a digit."""
    for state in STATES:
        for framing in (Framing.SEMANTIC, Framing.ABSTRACT):
            for text in (builder.score_falsified_text(state, framing, 15),
                         builder.move_falsified_text(state, framing)):
                assert len(builder.tokenizer.encode(text)) <= builder.block_tokens


# --- the original arms must render identically ------------------------------


@pytest.mark.parametrize("arm", ORIGINAL_ARMS)
@pytest.mark.parametrize("framing", [Framing.SEMANTIC, Framing.ABSTRACT])
def test_original_arms_still_produce_target_length_blocks(builder, arm, framing):
    donor = St(41, 33, 8, Action.DEFECT)
    for state in STATES:
        _, blk = builder.build_pair(
            arm, state, framing,
            donor=donor if arm is Arm.PLACEBO_STALE else None)
        assert len(blk.token_ids) == builder.block_tokens


def test_build_pair_ignores_score_offset_for_every_original_arm(builder):
    """`score_offset` was added to build_pair with a default. For arms that
    predate it the value must be inert - otherwise re-running exp3 from HEAD
    would silently produce different blocks."""
    donor = St(41, 33, 8, Action.DEFECT)
    for arm in ORIGINAL_ARMS:
        kw = {"donor": donor} if arm is Arm.PLACEBO_STALE else {}
        a = builder.build_pair(arm, STATES[1], Framing.SEMANTIC, **kw)
        b = builder.build_pair(arm, STATES[1], Framing.SEMANTIC,
                               score_offset=999, **kw)
        assert a[1].token_ids == b[1].token_ids
        assert a[0].token_ids == b[0].token_ids


def test_treatment_block_is_byte_identical_across_arms(builder):
    """build_pair returns (treatment, arm_block) and the treatment half must not
    depend on which arm is being built - it is the parity reference."""
    donor = St(41, 33, 8, Action.DEFECT)
    ref = builder.build_pair(Arm.TREATMENT, STATES[1], Framing.SEMANTIC)[0]
    for arm in ORIGINAL_ARMS + NEW_ARMS:
        kw = {"donor": donor} if arm is Arm.PLACEBO_STALE else {}
        assert builder.build_pair(
            arm, STATES[1], Framing.SEMANTIC, **kw)[0].token_ids == ref.token_ids


# --- enum and identifier stability -----------------------------------------


def test_existing_arm_values_are_unchanged():
    """These strings are the primary key in every committed database. A change
    here orphans 300,000 episodes."""
    assert Arm.BASELINE.value == "1"
    assert Arm.TREATMENT.value == "3"
    assert Arm.PLACEBO_NONDIAGNOSTIC.value == "3b"
    assert Arm.PLACEBO_STALE.value == "3c"
    assert Arm.PLACEBO_SYNTACTIC.value == "3d"


def test_injects_block_is_unchanged_for_original_arms():
    for arm in ORIGINAL_ARMS:
        assert arm.injects_block
    assert not Arm.BASELINE.injects_block


def test_config_fingerprint_does_not_see_the_arm_set():
    """`ExperimentConfig.fingerprint()` is stored on every episode row. Arm is
    not one of its fields, so extending the enum cannot change a historical
    fingerprint - pinned here because a future refactor might add it."""
    from dataclasses import fields
    from cdx.config import ExperimentConfig
    names = {f.name for f in fields(ExperimentConfig)}
    assert "arm" not in names and "arms" not in names