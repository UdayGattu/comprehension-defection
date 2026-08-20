"""exp8's second prompt family must not touch the first one.

WHY THIS FILE EXISTS
    Two referees independently made the same objection: a placebo-controlled
    ablation method that has only ever been run on ONE [STATE] template, ONE
    field order and ONE insertion position is a single observation, not an
    instrument. exp8 crosses a second template, a permuted field order and a
    second insertion position against the existing arms.

    That is a change to the machinery every committed experiment ran through,
    and it has exactly three ways to be wrong.

      1. PARITY TARGET CONTAMINATION.  `block_tokens` is the length every arm
         is padded UP to, derived as the longest block any reachable state
         produces. The reworded template's labels are longer, so its blocks are
         longer. If the derivation ever learned about more than one template,
         the target would rise, EVERY block in EVERY arm of exp1-exp7 would get
         wider, and the study's own exp3->exp4 measurement says a prompt-width
         change of that kind moves a causal estimate by up to 0.04 against
         effects as small as 0.017. The invariant is not "the new template has
         parity" - it is "the new template has its OWN target and the original's
         is untouched".

      2. FINGERPRINT CONTAMINATION.  `ExperimentConfig.fingerprint()` hashes
         `ScaffoldConfig` and the result sits on every episode row of every
         committed database. A `state_template` field there would change the
         fingerprint of ~300,000 historical rows. It is a call argument for the
         same reason the scratchpad variant and `include_history` are
         (EXPERIMENTS.md, known defect 2).

      3. A MANIPULATION THAT DID NOT APPLY.  A permutation tuple that is never
         threaded through renders the canonical order under a permuted name. A
         template that silently falls back to the default produces a clean
         replication of exp6 mislabelled as a generalisation test. Both look
         like real results.

WHAT A STUB TOKENIZER CAN AND CANNOT PROVE
    CharTokenizer is one token per character, so the absolute targets below
    (119 / 149) are properties of THIS stub, not of Llama-3.1 (34), Qwen2.5 (39)
    or Mistral (45). What it proves is the MECHANISM that decides them: that the
    original number does not move when a longer template is registered, and that
    a change would have to go through the assertion below.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import (
    Action,
    Arm,
    ExperimentConfig,
    Framing,
    GameConfig,
    ReadoutMode,
    ScaffoldConfig,
)
from cdx.game import TitForTat, replay
from cdx.runner import instruction_for
from cdx.scaffold import (
    TEMPLATE_DENSITY_TOLERANCE,
    DEFAULT_STATE_TEMPLATE,
    FIELD_KEYS,
    HISTORY_HEADER,
    SCORE_FALSIFICATION,
    STATE_HEADER,
    STATE_TEMPLATES,
    PromptAssembler,
    ScaffoldBuilder,
    StateTemplate,
    _calibration_states,
    resolve_state_template,
)

GAME = GameConfig()
FRAMINGS = [Framing.SEMANTIC, Framing.ABSTRACT]
TEMPLATES = sorted(STATE_TEMPLATES)
BLOCK_ARMS = (Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC,
              Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE)

# The value the CharTokenizer derives for the template exp1-exp7 ran on. Pinned
# as a LITERAL, not recomputed: a test that recomputes the target from the same
# code it is testing passes no matter what that code does to it.
ORIGINAL_CHAR_TARGET = 119
REWORDED_CHAR_TARGET = 149


class Char:
    """One token per character - the harshest possible parity test, and the
    same double tests/test_no_history.py and tests/test_field_falsification.py
    already use."""

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


STATES = [St(0, 0, 0, None),
          St(24, 18, 8, Action.COOPERATE),
          St(48, 33, 14, Action.DEFECT),
          St(97, 61, 19, Action.DEFECT)]


@pytest.fixture
def tok():
    return Char()


def builder_for(name, tok=None):
    return ScaffoldBuilder(tok or Char(), ScaffoldConfig(), state_template=name)


def content_lines(text):
    return [ln for ln in text.split("\n") if ln]


def state_after(n, opponent=None):
    return replay(GAME, opponent or TitForTat(), [Action.COOPERATE] * n)


def suffix(framing):
    return instruction_for(framing, ReadoutMode.LOGIT)


def block_for(builder, arm, state, framing):
    donor = St(41, 33, 8, Action.DEFECT) if arm is Arm.PLACEBO_STALE else None
    return builder.build_pair(arm, state, framing, donor=donor)[1]


# =========================================================================
# 1. exp1-exp7 render byte-identically
# =========================================================================


def test_the_default_is_the_template_exp1_to_exp7_ran_on():
    assert DEFAULT_STATE_TEMPLATE == "original"
    assert resolve_state_template(None).name == "original"


def test_builder_defaults_to_the_original_template():
    """Every call site in the repository that omits the argument - including
    `cdx.runner.Runner`, which is what the laptop pipeline uses - must keep
    rendering what it has always rendered."""
    import inspect
    param = inspect.signature(ScaffoldBuilder.__init__).parameters["state_template"]
    assert param.default is None
    assert ScaffoldBuilder(Char(), ScaffoldConfig()).template_name == "original"


@pytest.mark.parametrize("framing,last,expected", [
    (Framing.SEMANTIC, Action.COOPERATE,
     "[STATE]\n"
     "Your score: 24\n"
     "Opponent score: 18\n"
     "Opponent's last move: Cooperate\n"
     "Rounds played: 8\n"),
    (Framing.ABSTRACT, Action.DEFECT,
     "[STATE]\n"
     "Your score: 24\n"
     "Opponent score: 18\n"
     "Opponent's last move: Y\n"
     "Rounds played: 8\n"),
    (Framing.SEMANTIC, None,
     "[STATE]\n"
     "Your score: 24\n"
     "Opponent score: 18\n"
     "Opponent's last move: none\n"
     "Rounds played: 8\n"),
])
def test_original_treatment_block_is_byte_for_byte_what_it_always_was(
    framing, last, expected
):
    """Against literal strings. The template machinery renders through a
    registry now; if the registry entry drifts by one character, every prompt in
    seven experiments drifts with it and no recomputed expectation would notice.
    """
    assert builder_for("original").treatment_text(
        St(24, 18, 8, last), framing) == expected


def test_original_placebo_blocks_are_byte_for_byte_what_they_always_were():
    b = builder_for("original")
    assert b.nondiagnostic_text(St(24, 18, 8, Action.COOPERATE), Framing.SEMANTIC) == (
        "[STATE]\n"
        "Round parity: even\n"
        "Interaction type: repeated\n"
        "Payoff scale: integer\n"
        "Move space: binary\n"
        "Record status: logged\n")
    assert b.nondiagnostic_text(St(24, 18, 7, Action.COOPERATE), Framing.SEMANTIC) == (
        "[STATE]\n"
        "Round parity: odd\n"
        "Interaction type: repeated\n"
        "Payoff scale: integer\n"
        "Move space: binary\n"
        "Record status: logged\n")
    assert b.syntactic_text(STATES[1], Framing.SEMANTIC) == (
        "[STATE]\n" + "<node attr />\n" * 6)


def test_registering_a_longer_template_does_not_move_the_original_target():
    """THE invariant. The reworded template's blocks are longer; if the
    derivation were a max over the registry, the original's target would rise
    and every historical prompt would get wider."""
    assert builder_for("original").block_tokens == ORIGINAL_CHAR_TARGET
    assert builder_for("reworded").block_tokens == REWORDED_CHAR_TARGET
    assert builder_for("reworded").block_tokens > ORIGINAL_CHAR_TARGET
    # construct the long one FIRST, in case anything caches at class level
    _ = builder_for("reworded_permuted")
    assert builder_for("original").block_tokens == ORIGINAL_CHAR_TARGET


def test_target_is_derived_from_this_template_and_only_this_template():
    """The generalisation of tests/test_exp1_to_exp5_unchanged.py's derivation
    check to every registered template."""
    for name in TEMPLATES:
        b = builder_for(name)
        longest = 0
        for state in _calibration_states():
            for framing in FRAMINGS:
                for text in (b.treatment_text(state, framing),
                             b.nondiagnostic_text(state, framing),
                             b.syntactic_text(state, framing)):
                    longest = max(longest, len(b.tokenizer.encode(text)))
        assert b.block_tokens == longest + 2, name


def test_permuting_the_order_does_not_change_the_target():
    """An order permutation reorders lines; it does not add or remove any. A
    target that moved would mean the renderer is doing something other than
    reordering."""
    assert builder_for("original_permuted").block_tokens == ORIGINAL_CHAR_TARGET
    assert builder_for("reworded_permuted").block_tokens == REWORDED_CHAR_TARGET


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS)
@pytest.mark.parametrize("n_rounds", [0, 1, 7, 19])
def test_full_prompt_is_unchanged_under_the_default(tok, framing, arm, n_rounds):
    """End to end: the assembled prompt with everything left at its default is
    byte-identical to a frozen re-implementation of the pre-template
    assembler."""
    b = ScaffoldBuilder(tok, ScaffoldConfig())
    a = PromptAssembler(tok, ScaffoldConfig())
    state = state_after(n_rounds)
    block = block_for(b, arm, state, framing)
    got = a.assemble(game_config=GAME, state=state, framing=framing,
                     block=block, instruction_suffix=suffix(framing))
    want = (tok.encode(a._rules(GAME, framing))
            + list(block.token_ids)
            + tok.encode(a._history_section(state, framing))
            + tok.encode(suffix(framing)))
    assert got == want


# =========================================================================
# 2. the template is NOT a config field
# =========================================================================


def test_scaffold_config_has_no_template_field():
    names = {f.name for f in fields(ScaffoldConfig)}
    assert "state_template" not in names
    assert "template" not in names
    assert not any("template" in n for n in names)


def test_config_fingerprint_is_unmoved_by_the_template():
    """Stored on every episode row of every committed database. Pinned as a
    literal so that adding ANY field to ScaffoldConfig fails here loudly rather
    than orphaning 300,000 rows quietly."""
    assert ExperimentConfig(run_id="x").fingerprint() == (
        "9f1d9a042e6c1a3a6ed87fa2deefcd6319897e179be3e2e54fa6b1e5402226f8")


def test_two_templates_share_one_fingerprint():
    cfg = ScaffoldConfig()
    a = ScaffoldBuilder(Char(), cfg, state_template="original")
    b = ScaffoldBuilder(Char(), cfg, state_template="reworded")
    assert a.config is b.config
    assert a.block_tokens != b.block_tokens          # they DO differ
    assert ExperimentConfig(run_id="r", scaffold=cfg).fingerprint() == (
        ExperimentConfig(run_id="r", scaffold=cfg).fingerprint())


def test_insertion_index_does_move_the_fingerprint_and_that_is_correct():
    """The contrast that makes the design legible: position is a config field
    and moves the hash, safely, because every historical row carries 1."""
    one = ExperimentConfig(run_id="r", scaffold=ScaffoldConfig()).fingerprint()
    two = ExperimentConfig(
        run_id="r", scaffold=ScaffoldConfig(insertion_index=2)).fingerprint()
    assert one != two
    assert ScaffoldConfig().insertion_index == 1


# =========================================================================
# 3. the second template holds parity WITHIN ITSELF
# =========================================================================


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS)
def test_every_arm_hits_its_own_template_target(name, framing, arm):
    b = builder_for(name)
    for state in STATES:
        t, o = b.build_pair(arm, state, framing)
        assert len(t.token_ids) == len(o.token_ids) == b.block_tokens


@pytest.mark.parametrize("name", TEMPLATES)
def test_all_arms_share_one_length_within_a_template(name):
    """Parity within an arm is not enough: a cross-arm contrast at different
    block widths is a width contrast."""
    b = builder_for(name)
    donor = St(41, 33, 8, Action.DEFECT)
    lengths = set()
    for arm in BLOCK_ARMS + (Arm.PLACEBO_STALE,):
        _, blk = b.build_pair(arm, STATES[1], Framing.SEMANTIC,
                              donor=donor if arm is Arm.PLACEBO_STALE else None)
        lengths.add(len(blk.token_ids))
    assert len(lengths) == 1, f"{name}: block lengths differ across arms {lengths}"


@pytest.mark.parametrize("name", TEMPLATES)
def test_no_reachable_state_overflows_its_template_target(name):
    """Padding can only lengthen. A state that renders longer than the target
    aborts the run mid-flight, which is the failure `_CAL_MARGIN` exists to
    keep off the GPU."""
    b = builder_for(name)
    for state in _calibration_states():
        for framing in FRAMINGS:
            for text in (b.treatment_text(state, framing),
                         b.nondiagnostic_text(state, framing),
                         b.syntactic_text(state, framing),
                         b.score_falsified_text(state, framing, SCORE_FALSIFICATION),
                         b.move_falsified_text(state, framing)):
                assert len(b.tokenizer.encode(text)) <= b.block_tokens, (name, text)


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_block_is_at_least_70_percent_content(name):
    """A FLOOR, not a match. The name used to claim a match; it never checked
    one, and the distinction matters.

    Token parity alone is not comparability: exp1's 3b was 15 content tokens
    padded with 17 filler into a 32-token block, 47% content, against a
    treatment block of 32 content tokens and no filler at all. The contrast
    therefore confounded
    'decision-relevant content' with 'dense text vs whitespace'. This test
    bounds that failure from below only. Measured spread WITHIN a template is
    22-28 points (non-diagnostic ~98%, treatment ~71-77%) and both templates
    carry it equally, so a within-template match assertion would fail on
    `original` - i.e. on the condition exp1-exp7 actually ran. The property
    exp8 needs is cross-template, and it is asserted separately below."""
    b = builder_for(name)
    state = St(120, 120, 25, Action.COOPERATE)
    for text in (b.treatment_text(state, Framing.SEMANTIC),
                 b.nondiagnostic_text(state, Framing.SEMANTIC),
                 b.syntactic_text(state, Framing.SEMANTIC)):
        density = len(b.tokenizer.encode(text)) / b.block_tokens
        assert density >= 0.70, f"{name}: block is {1 - density:.0%} filler"


# =========================================================================
# 4. the second template says the same things, differently
# =========================================================================


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_template_has_the_same_four_fields(name):
    t = STATE_TEMPLATES[name]
    assert sorted(t.field_order) == sorted(FIELD_KEYS)
    assert len(t.labels) == 4 == len(set(t.labels))


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("framing", FRAMINGS)
def test_every_template_renders_the_same_four_values(name, framing):
    """Same information, different words. The rendered VALUES must be identical
    across templates or the template factor is confounded with a content
    factor."""
    state = STATES[2]
    ref = builder_for("original").treatment_text(state, framing)
    got = builder_for(name).treatment_text(state, framing)
    ref_vals = sorted(ln.split(": ", 1)[1] for ln in content_lines(ref)[1:])
    got_vals = sorted(ln.split(": ", 1)[1] for ln in content_lines(got)[1:])
    assert ref_vals == got_vals


def test_the_reworded_template_shares_no_field_label_with_the_original():
    """A template that reused a label would leave the lexical account of the
    exp6 result untested for that field."""
    a = set(STATE_TEMPLATES["original"].labels)
    b = set(STATE_TEMPLATES["reworded"].labels)
    assert a & b == set()


def _words(labels):
    """Lowercased word set of a label family, with possessive/punctuation
    stripped so "Opponent's" and "Opponent" do not count as different words."""
    return {w.strip(":'s").lower() for lab in labels for w in lab.split()}


def test_the_two_label_families_share_exactly_two_words():
    """PREREGISTRATION_EXP8.md §3 states the overlap is exactly {your, round}.

    It originally claimed NO shared content word, which was false. The claim is
    now measured rather than asserted, and pinned here so the pre-registration
    and the code cannot drift apart again."""
    shared = _words(STATE_TEMPLATES["original"].labels) & \
             _words(STATE_TEMPLATES["reworded"].labels)
    assert shared == {"your", "round"}, shared


@pytest.mark.parametrize("name", [t for t in TEMPLATES if t != "original"])
@pytest.mark.parametrize("framing", FRAMINGS)
def test_templates_have_matched_density_profiles(name, framing):
    """The property exp8's T factor actually depends on.

    If `reworded` carried a different filler fraction than `original`, the
    template factor would be confounded with a padding factor and `A` would be
    measuring whitespace. Compared per block type, at MATCHED game states, so
    the turn-to-turn variation in treatment density (~6 points) cancels.

    CharTokenizer only. A BPE vocabulary can split these labels with different
    efficiency, which no CPU test can see - scripts/gpu_run.py STEP 2 runs the
    same comparison against the real tokeniser before any GPU time is spent."""
    ref, this = builder_for("original"), builder_for(name)
    for state in STATES:
        for block in ("treatment_text", "nondiagnostic_text", "syntactic_text"):
            d_ref = len(ref.tokenizer.encode(
                getattr(ref, block)(state, framing))) / ref.block_tokens
            d_this = len(this.tokenizer.encode(
                getattr(this, block)(state, framing))) / this.block_tokens
            assert abs(d_this - d_ref) <= TEMPLATE_DENSITY_TOLERANCE, (
                f"{name} {block} at turn {state.turn_index}: "
                f"{d_this:.1%} vs original {d_ref:.1%} "
                f"(tolerance {TEMPLATE_DENSITY_TOLERANCE:.0%})")


def test_the_permutation_moves_both_falsifiable_fields():
    """Score first->last, last move third->first. Anything less and the ORDER
    factor cannot separate 'the last-move field dominates' from 'the third line
    dominates'."""
    canon = STATE_TEMPLATES["original"].field_order
    perm = STATE_TEMPLATES["original_permuted"].field_order
    assert canon.index("agent_score") == 0 and perm.index("agent_score") == 3
    assert canon.index("last_move") == 2 and perm.index("last_move") == 0
    assert canon != perm


@pytest.mark.parametrize("name", TEMPLATES)
def test_rendered_line_order_follows_the_declared_order(name):
    t = STATE_TEMPLATES[name]
    rendered = content_lines(
        builder_for(name).treatment_text(STATES[1], Framing.SEMANTIC))[1:]
    assert [ln.split(":", 1)[0] for ln in rendered] == list(t.labels)


def test_a_template_with_a_broken_field_set_is_refused():
    for bad in (("agent_score", "agent_score", "last_move", "rounds"),
                ("agent_score", "opponent_score", "last_move")):
        with pytest.raises(ValueError, match="permutation"):
            StateTemplate("bad", "a", "b", "c", "d", bad, ("x",), ("y",))


def test_a_template_with_a_colon_in_a_label_is_refused():
    with pytest.raises(ValueError, match="':'"):
        StateTemplate("bad", "a: b", "b", "c", "d", FIELD_KEYS, ("x",), ("y",))


def test_an_unknown_template_name_is_refused():
    with pytest.raises(ValueError, match="unknown state template"):
        ScaffoldBuilder(Char(), ScaffoldConfig(), state_template="nope")


# =========================================================================
# 5. falsification changes EXACTLY ONE LINE under BOTH templates
# =========================================================================


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("state", STATES[1:])
def test_3s_changes_exactly_one_line_under_every_template(name, framing, state):
    b = builder_for(name)
    truth = content_lines(b.treatment_text(state, framing))
    lie = content_lines(b.score_falsified_text(state, framing, SCORE_FALSIFICATION))
    assert len(truth) == len(lie), f"{name}: field count changed"
    diff = [(x, y) for x, y in zip(truth, lie) if x != y]
    assert len(diff) == 1, f"{name}: 3s changed {len(diff)} lines: {diff}"
    assert diff[0][0].startswith(
        STATE_TEMPLATES[name].agent_score_label + ":"), diff


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("state", STATES[1:])
def test_3m_changes_exactly_one_line_under_every_template(name, framing, state):
    b = builder_for(name)
    truth = content_lines(b.treatment_text(state, framing))
    lie = content_lines(b.move_falsified_text(state, framing))
    assert len(truth) == len(lie), f"{name}: field count changed"
    diff = [(x, y) for x, y in zip(truth, lie) if x != y]
    assert len(diff) == 1, f"{name}: 3m changed {len(diff)} lines: {diff}"
    assert diff[0][0].startswith(
        STATE_TEMPLATES[name].last_move_label + ":"), diff


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_falsified_line_moves_with_the_permutation(name):
    """The point of the ORDER factor: under the permuted order the falsified
    last-move line is the FIRST content line, not the third. If the diff stayed
    at index 2 the permutation never reached the renderer."""
    b = builder_for(name)
    state = STATES[1]
    truth = content_lines(b.treatment_text(state, Framing.SEMANTIC))
    lie = content_lines(b.move_falsified_text(state, Framing.SEMANTIC))
    idx = [i for i, (x, y) in enumerate(zip(truth, lie)) if x != y]
    expected = STATE_TEMPLATES[name].field_order.index("last_move") + 1  # +1 header
    assert idx == [expected], (name, idx, expected)


@pytest.mark.parametrize("name", TEMPLATES)
def test_turn_zero_still_cannot_be_falsified_under_any_template(name):
    b = builder_for(name)
    s = St(0, 0, 0, None)
    assert b.move_falsified_text(s, Framing.SEMANTIC) == b.treatment_text(
        s, Framing.SEMANTIC)


# =========================================================================
# 6. position 2 puts the block where it is supposed to go
# =========================================================================


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("arm", BLOCK_ARMS)
@pytest.mark.parametrize("name", TEMPLATES)
def test_position_2_places_the_block_after_the_history(tok, framing, arm, name):
    """The exact slice, computed from the section lengths rather than searched
    for, so a block that drifted fails here rather than being found somewhere
    else."""
    cfg = ScaffoldConfig(insertion_index=2)
    b = ScaffoldBuilder(tok, cfg, state_template=name)
    a = PromptAssembler(tok, cfg)
    state = state_after(9)
    block = block_for(b, arm, state, framing)

    ids = a.assemble(game_config=GAME, state=state, framing=framing,
                     block=block, instruction_suffix=suffix(framing))
    rules = tok.encode(a._rules(GAME, framing))
    hist = tok.encode(a._history_section(state, framing))
    start = len(rules) + len(hist)

    assert ids[:len(rules)] == rules
    assert ids[len(rules):start] == hist
    assert ids[start:start + len(block.token_ids)] == list(block.token_ids)
    assert ids[start + len(block.token_ids):] == tok.encode(suffix(framing))
    assert a.block_section_index(include_history=True) == 2


@pytest.mark.parametrize("name", TEMPLATES)
def test_position_1_and_2_differ_only_by_moving_the_block(tok, name):
    """The strongest available statement about the POSITION factor: the two
    prompts are the same multiset of sections in a different order. Not
    'similar', not 'the same length' - the same three parts."""
    state = state_after(11)
    parts = {}
    for idx in (1, 2):
        cfg = ScaffoldConfig(insertion_index=idx)
        b = ScaffoldBuilder(tok, cfg, state_template=name)
        a = PromptAssembler(tok, cfg)
        block = block_for(b, Arm.TREATMENT, state, Framing.SEMANTIC)
        parts[idx] = a.assemble(game_config=GAME, state=state,
                                framing=Framing.SEMANTIC, block=block,
                                instruction_suffix=suffix(Framing.SEMANTIC))
        parts[f"block{idx}"] = list(block.token_ids)

    assert parts["block1"] == parts["block2"]
    assert len(parts[1]) == len(parts[2])
    assert parts[1] != parts[2]
    one, two = tok.decode(parts[1]), tok.decode(parts[2])
    assert one.index(STATE_HEADER) < one.index(HISTORY_HEADER)
    assert two.index(STATE_HEADER) > two.index(HISTORY_HEADER)


def test_position_2_is_refused_without_history(tok):
    """Already pinned in tests/test_no_history.py for the original template;
    repeated here across the whole family, because exp8 is the run that will
    actually pass insertion_index=2 on a driver command line."""
    for name in TEMPLATES:
        cfg = ScaffoldConfig(insertion_index=2)
        b = ScaffoldBuilder(tok, cfg, state_template=name)
        a = PromptAssembler(tok, cfg)
        state = state_after(4)
        kw = dict(game_config=GAME, state=state, framing=Framing.SEMANTIC,
                  block=block_for(b, Arm.TREATMENT, state, Framing.SEMANTIC),
                  instruction_suffix=suffix(Framing.SEMANTIC))
        a.assemble(**kw, include_history=True)
        with pytest.raises(ValueError, match="after"):
            a.assemble(**kw, include_history=False)
        with pytest.raises(ValueError, match="after"):
            a.block_section_index(include_history=False)


def test_a_block_after_the_instruction_is_refused(tok):
    """`insertion_index == len(sections)` used to be in range. It appends the
    block AFTER the instruction suffix - the last thing the model reads and the
    thing that tells it what to emit. That is not a third position, it is a
    different task."""
    state = state_after(4)
    for idx, include in ((3, True), (2, False)):
        cfg = ScaffoldConfig(insertion_index=idx)
        b = ScaffoldBuilder(tok, cfg)
        a = PromptAssembler(tok, cfg)
        with pytest.raises(ValueError):
            a.assemble(game_config=GAME, state=state, framing=Framing.SEMANTIC,
                       block=block_for(b, Arm.TREATMENT, state, Framing.SEMANTIC),
                       instruction_suffix=suffix(Framing.SEMANTIC),
                       include_history=include)


def test_position_does_not_change_the_block_itself(tok):
    """Position must move the block and nothing else - not its width, not its
    padding, not its bytes. Otherwise the POSITION factor is confounded with a
    parity change."""
    state = state_after(6)
    ref = None
    for idx in (0, 1, 2):
        cfg = ScaffoldConfig(insertion_index=idx)
        blk = block_for(ScaffoldBuilder(tok, cfg), Arm.PLACEBO_MOVE, state,
                        Framing.SEMANTIC)
        ref = ref or blk.token_ids
        assert blk.token_ids == ref
        assert blk.pad_tokens_added == ScaffoldBuilder(
            tok, cfg).build_pair(Arm.PLACEBO_MOVE, state,
                                 Framing.SEMANTIC)[1].pad_tokens_added
