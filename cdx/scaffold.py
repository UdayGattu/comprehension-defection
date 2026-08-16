"""Prompt assembly, scaffold rendering, and token-ID-level parity enforcement.

The causal claim rests entirely on treatment and placebo blocks being
indistinguishable except in whether their content is decision-relevant. Two
properties must therefore hold exactly, not approximately:

  1. Identical token count.  Enforced at token-ID level and asserted. Zero-padded
     string templates are NOT sufficient: BPE vocabularies merge digit runs by
     corpus frequency, so "007" and "014" may differ in token count, and this
     varies by tokenizer. Padding is therefore applied by appending raw token IDs,
     which cannot trigger a merge because the tokenizer is never re-run.

  2. Identical insertion position.  Lost-in-the-middle effects produce >30%
     swings from position alone. A treatment block placed differently from its
     placebo would manufacture an effect out of nothing.

Truncation is never used. Slicing the tail of a token sequence removes structural
tokens (newlines, end-of-turn markers, closing braces) and corrupts the prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import Action, Arm, Framing, GameConfig, HorizonMode, ScaffoldConfig
from .game import GameState

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Minimal surface we depend on. Lets the engine be developed and tested
    without downloading model weights."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class TokenParityError(RuntimeError):
    """Raised when treatment and placebo token counts cannot be reconciled.

    This is fatal by design. A parity violation silently invalidates the causal
    estimate, so the run must halt rather than log a warning and continue.
    """


@dataclass(frozen=True)
class ScaffoldBlock:
    arm: Arm
    text: str
    token_ids: tuple[int, ...]
    pad_tokens_added: int

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


def render_action(action: Action, framing: Framing, swap_labels: bool = False) -> str:
    """Render an action as the label the model sees.

    `swap_labels` inverts the action-to-label mapping. This is a CONTROL, not a
    convenience: a model that always picks the first-listed option is showing
    position bias, not strategy, and the only way to tell the two apart is to
    swap which label means what and see whether behaviour follows the label or
    the meaning. Without this control no behavioural claim survives review.
    """
    cooperate_first = action is Action.COOPERATE
    if swap_labels:
        cooperate_first = not cooperate_first
    if framing is Framing.ABSTRACT:
        return "X" if cooperate_first else "Y"
    return "Cooperate" if cooperate_first else "Defect"


_NO_ACTION = "none"


def _surface_width(framing: Framing) -> int:
    return max(
        len(render_action(Action.COOPERATE, framing)),
        len(render_action(Action.DEFECT, framing)),
        len(_NO_ACTION),
    )


def render_action_in_state(
    action: Action | None, framing: Framing, swap_labels: bool = False
) -> str:
    """Rendering for use inside the [STATE] block.

    NO character padding. This function used to ljust() to a fixed width so the
    treatment block had a constant size. That approach is retired: character
    padding cannot deliver token parity, because BPE decides how many tokens a
    padded string becomes and that varies by value and by tokenizer. It also
    carried a real cost — see ScaffoldConfig.score_field_width.

    Constant block size is now achieved at the token-ID level in build_pair,
    which is the mechanism this module already trusted for placebos.
    """
    return _NO_ACTION if action is None else render_action(action, framing, swap_labels)


# Retained under the old name so existing imports keep working.
render_action_fixed_width = render_action_in_state


# The literal header of the raw-log section. Named because three places have to
# agree on it: the renderer below, the tests that assert the no-history condition
# removed it, and the driver's manipulation check, which greps stored prompts.
# A string literal repeated in a shell script is a check that silently stops
# checking the day someone renames the section.
HISTORY_HEADER = "[HISTORY]"


def render_history(state: GameState, framing: Framing, swap_labels: bool = False) -> str:
    if not state.turns:
        return "(no rounds played yet)"
    return "\n".join(
        f"Round {t.index + 1}: you={render_action(t.agent_action, framing, swap_labels)} "
        f"opponent={render_action(t.opponent_action, framing, swap_labels)}"
        for t in state.turns
    )


# The header of the injected block. Every template shares it: the header is not
# part of the manipulation, it is the seam the manipulation is inserted at, and
# the driver's template gate greps for it to locate the block inside a stored
# prompt.
STATE_HEADER = "[STATE]"

# The four things the block can say. Internal keys, never rendered - the LABELS
# are what a template varies, and the keys are what stays fixed so that "same
# information, same field count" is checkable by a machine rather than by eye.
FIELD_KEYS = ("agent_score", "opponent_score", "last_move", "rounds")


@dataclass(frozen=True)
class StateTemplate:
    """One rendering of the [STATE] block: labels, field order, and the two
    matched placebo bodies that go with them.

    WHY THIS TYPE EXISTS
        Every experiment from exp1 to exp7 used ONE template, ONE field order
        and ONE insertion position. A placebo-controlled ablation method that
        has only ever been run on a single prompt string is not an instrument;
        it is one observation. exp8 crosses a second template, a permuted field
        order and a second insertion position against the existing arms so the
        field-level asymmetry can be shown to be a property of the manipulated
        CONTENT rather than of the particular words it was measured in.

    WHY IT IS NOT A ScaffoldConfig FIELD
        `ExperimentConfig.fingerprint()` hashes `ScaffoldConfig`, and that hash
        is stored on every episode row of every committed database. Adding a
        field there would change the fingerprint of ~300,000 historical rows.
        This is the documented reason the scratchpad variant is not a config
        field (EXPERIMENTS.md, known defect 2) and the reason `include_history`
        is a call argument on `assemble` rather than config. The template is
        carried the same way those two are: by `run_id`, by
        `run_meta.config_json` (which serialises argv), and by
        `turn_details.prompt_full`.

    WHY EACH TEMPLATE CARRIES ITS OWN PLACEBO BODIES
        `nondiagnostic_text` is DENSITY-matched to `treatment_text`, not merely
        token-matched - see the note on that method. A second template with
        longer labels has a longer natural block, so reusing the original
        placebo bodies under it would push the filler fraction up and reproduce
        exactly the defect (a ~44%-CONTENT placebo against a ~94%-content
        treatment) that the current bodies exist to fix. Each template
        therefore ships its own, sized to its own treatment block.

    WHAT MUST NOT MOVE
        The parity target is derived PER TEMPLATE, from that template's own
        three block types. A template with a longer natural block must not be
        allowed to raise the target the original template established, because
        every arm is padded UP to the target: a rise would widen every block in
        every historical experiment and nothing in the repository would
        reproduce. Pinned by tests/test_template_family.py.
    """

    name: str
    agent_score_label: str
    opponent_score_label: str
    last_move_label: str
    rounds_label: str
    field_order: tuple[str, ...]
    nondiagnostic_lines: tuple[str, ...]
    syntactic_lines: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.field_order)) != tuple(sorted(FIELD_KEYS)):
            raise ValueError(
                f"template {self.name!r} field_order {self.field_order} is not a "
                f"permutation of {FIELD_KEYS}. A template that drops or repeats a "
                f"field changes WHAT the block says, not just how it says it, and "
                f"the template factor would be confounded with an information "
                f"factor."
            )
        labels = [self.agent_score_label, self.opponent_score_label,
                  self.last_move_label, self.rounds_label]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"template {self.name!r} reuses a label: {labels}. Two fields "
                f"with the same label are not readable as separate fields."
            )
        for label in labels:
            if ":" in label or "\n" in label:
                raise ValueError(
                    f"template {self.name!r} label {label!r} contains ':' or a "
                    f"newline. The line-by-line falsification tests split on "
                    f"those, and a label carrying one would make 'exactly one "
                    f"line differs' unverifiable."
                )

    def label(self, key: str) -> str:
        return {
            "agent_score": self.agent_score_label,
            "opponent_score": self.opponent_score_label,
            "last_move": self.last_move_label,
            "rounds": self.rounds_label,
        }[key]

    @property
    def labels(self) -> tuple[str, ...]:
        """In RENDERED order. The driver's order gate reads this."""
        return tuple(self.label(k) for k in self.field_order)


# ---- the template family --------------------------------------------------
#
# ORIGINAL is byte-for-byte what exp1-exp7 rendered. It is reproduced here as
# data rather than as code, and tests/test_template_family.py pins it against
# literal strings, so a future edit to the template machinery cannot silently
# rewrite the prompts five committed experiments were run on.

_ORIGINAL_ORDER = ("agent_score", "opponent_score", "last_move", "rounds")

# The permutation moves BOTH falsifiable fields as far as a four-field block
# allows: the score goes from first to last and the last move from third to
# first. That is the only permutation that can separate "the last-move field
# dominates" from "the third line dominates" - a serial-position account that
# no experiment in this project has ever been able to rule out.
_PERMUTED_ORDER = ("last_move", "rounds", "opponent_score", "agent_score")

_ORIGINAL_NONDIAGNOSTIC = (
    "Round parity: {parity}",
    "Interaction type: repeated",
    "Payoff scale: integer",
    "Move space: binary",
    "Record status: logged",
)
_ORIGINAL_SYNTACTIC = ("<node attr />",) * 6

# REWORDED shares no content word with ORIGINAL except "Your" and "Rounds", and
# says exactly the same four things. Sized so that its filler fractions match
# the original's rather than its character counts: under the CharTokenizer the
# tests use, treatment sits at 78.5% of target against the original's 79.0%,
# the non-diagnostic placebo at 98.7% against 98.3%, and the syntactic placebo
# at 80.5% against 77.3%. Density, not length, is the thing that has to match -
# a template whose placebo was mostly filler would make its own 3-vs-3b
# contrast measure whitespace.
_REWORDED_NONDIAGNOSTIC = (
    "Round index parity: {parity}",
    "Exchange structure: repeated",
    "Point units: whole numbers",
    "Choice set cardinality: two",
    "Transcript retention: enabled",
)
_REWORDED_SYNTACTIC = ("<node attr />",) * 8


def _original(name: str, order: tuple[str, ...]) -> StateTemplate:
    return StateTemplate(
        name=name,
        agent_score_label="Your score",
        opponent_score_label="Opponent score",
        last_move_label="Opponent's last move",
        rounds_label="Rounds played",
        field_order=order,
        nondiagnostic_lines=_ORIGINAL_NONDIAGNOSTIC,
        syntactic_lines=_ORIGINAL_SYNTACTIC,
    )


def _reworded(name: str, order: tuple[str, ...]) -> StateTemplate:
    return StateTemplate(
        name=name,
        agent_score_label="Your cumulative points",
        opponent_score_label="Their cumulative points",
        last_move_label="Their previous choice",
        rounds_label="Rounds elapsed",
        field_order=order,
        nondiagnostic_lines=_REWORDED_NONDIAGNOSTIC,
        syntactic_lines=_REWORDED_SYNTACTIC,
    )


STATE_TEMPLATES: dict[str, StateTemplate] = {
    "original": _original("original", _ORIGINAL_ORDER),
    "original_permuted": _original("original_permuted", _PERMUTED_ORDER),
    "reworded": _reworded("reworded", _ORIGINAL_ORDER),
    "reworded_permuted": _reworded("reworded_permuted", _PERMUTED_ORDER),
}

# exp1-exp7 all ran on this one. It is the default everywhere, and every call
# site that omits the argument must keep getting it, or the historical prompts
# change.
DEFAULT_STATE_TEMPLATE = "original"


# Tolerance for the CROSS-TEMPLATE density gate.
#
# WHAT IS BEING BOUNDED, AND WHAT IS NOT
#     Not the density spread WITHIN a template. That spread is large and always
#     has been: measured under CharTokenizer the non-diagnostic placebo sits at
#     ~98% real content while the treatment sits at ~71-77%, a gap of 22-28
#     points, and `original` and `reworded` show that gap equally. Every number
#     in exp1-exp7 already carries it. A within-template match gate would abort
#     on exp6's own condition, which would be a bug, not a fix.
#
#     What exp8 actually needs is that changing the TEMPLATE does not change the
#     density profile - otherwise the T factor is confounded with a filler-
#     fraction factor and `A` is measuring padding. So the bound is per block
#     type, ACROSS templates, at a matched game state.
#
# PROVENANCE OF THE NUMBER - this is a judgement, recorded as one
#     Measured under CharTokenizer, largest gap between `original` and
#     `reworded` at a matched state:
#         treatment        74.8% vs 75.2%   ->  0.4 pt
#         non-diagnostic   98.3% vs 98.7%   ->  0.4 pt
#         syntactic        77.3% vs 80.5%   ->  3.2 pt   <- the binding one
#     0.05 is that maximum plus margin. It is NOT derived from a model
#     tokeniser, because no CPU test can reach one: a BPE vocabulary may split
#     "Your cumulative points" and "Opponent score" with different efficiency
#     and widen the gap. That is exactly why the gate also runs at preflight on
#     the pod, against the real tokeniser, in scripts/gpu_run.py STEP 2.
TEMPLATE_DENSITY_TOLERANCE = 0.05


def resolve_state_template(template: "str | StateTemplate | None") -> StateTemplate:
    """Accept a registry name, a StateTemplate, or None (meaning the default)."""
    if template is None:
        return STATE_TEMPLATES[DEFAULT_STATE_TEMPLATE]
    if isinstance(template, StateTemplate):
        return template
    try:
        return STATE_TEMPLATES[template]
    except KeyError:
        raise ValueError(
            f"unknown state template {template!r}; expected one of "
            f"{sorted(STATE_TEMPLATES)}"
        ) from None


# Calibration sweep. Scores cover every digit-count boundary, which is where
# tokenisers change their minds; turns cover the full horizon plus slack.
_CAL_SCORES = (0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 20, 48, 50, 98, 99, 100, 101, 120)
_CAL_TURNS = 26
_CAL_MARGIN = 2


class _CalibrationState:
    """The four attributes the block templates touch. Not a real GameState:
    calibration must not depend on the engine."""

    __slots__ = ("agent_score", "opponent_score", "turn_index", "turns", "_last")

    def __init__(self, agent: int, opponent: int, turn: int, last) -> None:
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self.turns = ()
        self._last = last

    def last_opponent_action(self):
        return self._last


def _calibration_states():
    for turn in range(_CAL_TURNS):
        for score in _CAL_SCORES:
            for last in (None, Action.COOPERATE, Action.DEFECT):
                yield _CalibrationState(score, score, turn, last)


# Magnitude of the arm-3s score falsification.
#
# NOT a free parameter. analysis/12 measured what arm 3c's donor sampling
# actually delivers: sd(d) = 3.2-4.2 in the semantic cells, with |d| >= 15 on
# 0.1%-1.1% of rows. At the measured slope of ~0.01 defection per point of score
# error, a typical +/-3 lie predicts a 3pp shift - inside the noise. The score
# arm has therefore never been tested at a magnitude capable of moving a
# decision, and 3s exists to test it at one that can: 15 points predicts ~15pp.
SCORE_FALSIFICATION = 15

# Highest score reachable in a 20-round game is 5 x 20 = 100. Falsified scores
# are clamped into [0, _MAX_REACHABLE_SCORE] for two independent reasons: a
# negative or impossible score is a giveaway that the block is fabricated, which
# would confound "false state" with "obviously false state"; and _CAL_SCORES
# tops out at 120, so a value outside the reachable range could render longer
# than the parity target and abort the run mid-flight.
_MAX_REACHABLE_SCORE = 100


class _FalsifiedView:
    """A read-only view of a GameState with exactly one field altered.

    Duck-types the four attributes the block templates touch, the same contract
    `_CalibrationState` satisfies. The real GameState is never mutated - the
    engine keeps advancing on the truth while only the rendered block lies,
    which is the whole point of the manipulation.
    """

    __slots__ = ("agent_score", "opponent_score", "turn_index", "turns", "_last")

    def __init__(self, agent, opponent, turn, turns, last) -> None:
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self.turns = turns
        self._last = last

    def last_opponent_action(self):
        return self._last


def falsified_view(
    state: GameState, *, score_offset: int = 0, flip_move: bool = False
):
    """Build the view arms 3s and 3m render from.

    Kept public so the runner can read the DISPLAYED values straight off it for
    logging, rather than recomputing them and risking the log and the prompt
    disagreeing - which is exactly how the exp1 zero-padding defect survived.
    """
    last = state.last_opponent_action()
    if flip_move and last is not None:
        last = Action.DEFECT if last is Action.COOPERATE else Action.COOPERATE

    score = state.agent_score
    if score_offset:
        shifted = score + score_offset
        if not 0 <= shifted <= _MAX_REACHABLE_SCORE:
            shifted = score - score_offset          # try the other direction
        if not 0 <= shifted <= _MAX_REACHABLE_SCORE:
            shifted = min(max(score, 0), _MAX_REACHABLE_SCORE)   # give up, no lie
        score = shifted

    return _FalsifiedView(
        score, state.opponent_score, state.turn_index, state.turns, last
    )


def move_was_falsified(state: GameState) -> bool:
    """True when arm 3m actually changed something.

    False at turn 0, where there is no last move to flip and the block is
    identical to arm 3. Those rows must be excluded from the 3m contrast or
    they dilute it with unfalsified data - the same trap `donor_degenerate`
    exists to avoid in arm 3c.
    """
    return state.last_opponent_action() is not None


class ScaffoldBuilder:
    """Builds treatment and placebo blocks with enforced token parity."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ScaffoldConfig,
        state_template: "str | StateTemplate | None" = None,
    ) -> None:
        """`state_template` is a CALL ARGUMENT, not a config field.

        Same reason `include_history` is one: `ScaffoldConfig` is hashed into
        `config_fingerprint` and that hash sits on every episode row of every
        committed database. It defaults to the template exp1-exp7 ran on, so
        every existing call site keeps rendering exactly what it rendered.

        The parity target derived below is a property of (tokeniser, template).
        A second template derives its OWN target and can never raise the
        original's - see `_derive_block_tokens`.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.template = resolve_state_template(state_template)
        self.filler_text, self._filler_id = self._resolve_filler()
        logger.info("scaffold filler token: %r (id=%d)", self.filler_text, self._filler_id)

        # The parity target is a property of the TOKENISER, not of the study.
        # A constant in config would be right for one model and silently wrong
        # for the next: the same templates are 34 tokens under Llama-3.1 and 119
        # under a character-level tokeniser. So derive it here, from whatever
        # tokeniser was actually handed in.
        derived = self._derive_block_tokens()
        configured = config.treatment_block_tokens
        if configured and configured < derived:
            raise TokenParityError(
                f"treatment_block_tokens={configured} is below the {derived} "
                f"required by this tokeniser. Some state would render longer "
                f"than the target and parity would be unachievable mid-run. "
                f"Set it to {derived} or leave it at 0 to derive automatically."
            )
        self.block_tokens = configured or derived
        logger.info(
            "parity target: %d tokens (%s, template %s)",
            self.block_tokens,
            "configured" if configured else "auto-derived",
            self.template.name,
        )

    @property
    def template_name(self) -> str:
        return self.template.name

    def _derive_block_tokens(self) -> int:
        """Longest block any reachable state produces, over both framings.

        Framing-invariant on purpose: a target that changed with framing would
        make abstract and semantic runs non-comparable in prompt length, adding
        a confound to a factor that is supposed to vary only the labels.

        TEMPLATE-VARIANT on purpose, and this is the load-bearing half.
            The loop below reads `self.treatment_text` / `nondiagnostic_text` /
            `syntactic_text`, which render through `self.template`. So the
            target is derived from ONE template - the one this builder was
            constructed with - and never from the union of the registry.

            The alternative (a target that is the max over every registered
            template) is the single change that would destroy the repository's
            reproducibility. Every arm is padded UP to the target. A second
            template with longer labels would raise it, every block in exp1-exp7
            would get wider, and the study's own exp3->exp4 measurement says a
            prompt-width change of that kind moves a causal estimate by up to
            0.04 against effects as small as 0.017.

            The cost is that block width is not comparable ACROSS templates.
            That is accepted and declared: template is a between-run factor, and
            every exp8 estimand is a WITHIN-template contrast (arm vs arm at
            identical width), differenced across templates afterwards.

            Three block types, exactly as before exp8. Arms 3s and 3m are still
            excluded from the derivation - they render no wider than arm 3 plus
            a digit, and including them would raise the target of the original
            template. Pinned by
            tests/test_exp1_to_exp5_unchanged.py::test_parity_target_is_derived_from_the_original_blocks_only.
        """
        longest = 0
        for state in _calibration_states():
            for framing in (Framing.SEMANTIC, Framing.ABSTRACT):
                for text in (
                    self.treatment_text(state, framing),
                    self.nondiagnostic_text(state, framing),
                    self.syntactic_text(state, framing),
                ):
                    longest = max(longest, len(self.tokenizer.encode(text)))
        return longest + _CAL_MARGIN

    def _resolve_filler(self) -> tuple[str, int]:
        """Select the first candidate that encodes to exactly one token.

        Padding must land on an exact target count, which is only possible with a
        single-token filler. Measured fact: Mistral-7B-Instruct-v0.3 has no
        single-token newline while Qwen2.5 does, so the filler cannot be
        hardcoded. Record the selected filler per model in the methods section.
        """
        tried: list[tuple[str, int]] = []
        for candidate in self.config.filler_candidates:
            ids = self.tokenizer.encode(candidate)
            tried.append((candidate, len(ids)))
            if len(ids) == 1:
                return candidate, ids[0]
        raise TokenParityError(
            "No single-token filler available for this tokenizer. Tried "
            + ", ".join(f"{c!r}->{n} tokens" for c, n in tried)
            + ". Exact token parity is unachievable without one; add a candidate "
            "and re-run scripts/tokenizer_check.py before proceeding."
        )

    # ---- block text ------------------------------------------------------

    @property
    def _swap(self) -> bool:
        return self.config.swap_action_labels

    def treatment_text(self, state: GameState, framing: Framing) -> str:
        """Ground-truth, decision-relevant state.

        FIELD LABELS AND FIELD ORDER COME FROM `self.template`. Under the
        default template this renders exactly the four lines exp1-exp7 rendered,
        in exactly that order; that is pinned against literal strings in
        tests/test_template_family.py rather than left to inspection.

        Numbers are rendered NATURALLY. They were previously zero-padded to a
        fixed width; run `sweep` showed the model reading the leading zero of
        "012" rather than the value, accounting for 49.7% of score-probe
        failures in the treatment arm. Constant block size is now enforced on
        token IDs in build_pair, not on characters here.
        """
        last_str = render_action_in_state(
            state.last_opponent_action(), framing, self._swap
        )
        values = {
            "agent_score": f"{state.agent_score:d}",
            "opponent_score": f"{state.opponent_score:d}",
            "last_move": last_str,
            "rounds": f"{state.turn_index:d}",
        }
        t = self.template
        return STATE_HEADER + "\n" + "".join(
            f"{t.label(key)}: {values[key]}\n" for key in t.field_order
        )

    def nondiagnostic_text(self, state: GameState, framing: Framing) -> str:
        """True but decision-irrelevant.

        Deliberately NOT stale/false state: false state carries a belief penalty
        (a single misleading sentence measurably degrades performance), which
        would inflate the treatment effect. Stale state lives in Arm 3c where
        measuring that penalty is the objective.

        DENSITY MATTERS AS MUCH AS TOKEN COUNT.
            The earlier version was two content lines. Padded to parity it
            became ~44% CONTENT against a treatment that is ~94% content. Token
            parity held, but the two stimuli were not comparable: the contrast
            confounded "decision-relevant content" with "dense text vs
            whitespace". Arm 3d was worse at ~32% content, which would have made
            the 3b/3d comparison nearly vacuous.

            Five lines, matching the treatment's shape, so filler is a couple of
            tokens rather than twenty.

        NOTHING HERE MAY BE USABLE.
            Round count is deliberately absent even though the previous version
            included it: rounds remaining determines endgame play, so restating
            it leaks decision-relevant state into the control. Everything below
            is a true fact about the encoding or the game's form, and none of it
            can change a rational agent's move.
        """
        parity = "even" if state.turn_index % 2 == 0 else "odd"
        return STATE_HEADER + "\n" + "".join(
            line.format(parity=parity) + "\n"
            for line in self.template.nondiagnostic_lines
        )

    def stale_text(self, donor: GameState, framing: Framing) -> str:
        """Identical template to treatment, populated from a different episode.

        Provides the scaffold-echo manipulation check: if the model's probe
        answer reproduces the donor's numbers, it demonstrably read the block.
        """
        return self.treatment_text(donor, framing)

    # ---- single-field falsification (arms 3s, 3m) ------------------------

    def score_falsified_text(
        self, state: GameState, framing: Framing, offset: int
    ) -> str:
        """Treatment block with ONLY "Your score" wrong, by `offset`.

        Rendered through `treatment_text` on a view of the state, so the
        template, field order and every other value are byte-identical to arm 3.
        The only difference in the prompt is the digits after "Your score:".
        """
        return self.treatment_text(falsified_view(state, score_offset=offset), framing)

    def move_falsified_text(self, state: GameState, framing: Framing) -> str:
        """Treatment block with ONLY "Opponent's last move" flipped.

        At turn 0 there is no last move and nothing to flip; the block is then
        identical to arm 3 and the caller must record that this row carries no
        falsification. `move_was_falsified` exists so that decision is made from
        the state rather than from the turn index, which would be wrong under a
        stochastic horizon.
        """
        return self.treatment_text(falsified_view(state, flip_move=True), framing)

    def syntactic_text(self, state: GameState, framing: Framing) -> str:
        """Format noise. The lower bound on the pure perturbation effect.

        Well-formed markup carrying no proposition at all. Together with 3b and
        3 this forms a ladder holding token count, position and density fixed
        while varying only what the content IS:

            3d   structure, no language
            3b   language, true, unusable
            3    language, true, usable

        Deliberately contains no digits. A number here would partially mimic the
        treatment's numeric fields and blur the one distinction this arm exists
        to draw.

        Sized to the treatment's natural length so filler stays negligible; the
        previous four-tag version was ~32% content and 68% blank lines, which
        made it nearly indistinguishable from a padded 3b.
        """
        return STATE_HEADER + "\n" + "".join(
            line + "\n" for line in self.template.syntactic_lines
        )

    # ---- parity ----------------------------------------------------------

    def build_pair(
        self,
        arm: Arm,
        state: GameState,
        framing: Framing,
        donor: GameState | None = None,
        score_offset: int = SCORE_FALSIFICATION,
    ) -> tuple[ScaffoldBlock, ScaffoldBlock]:
        """Return (treatment_block, arm_block) with identical token counts.

        The treatment block is always constructed so that parity can be
        verified even when the arm under test is a placebo. Both are returned so
        the caller can log the pair.
        """
        treatment = self.treatment_text(state, framing)
        treatment_ids = self.tokenizer.encode(treatment)

        if arm is Arm.TREATMENT:
            other = treatment
        elif arm is Arm.PLACEBO_NONDIAGNOSTIC:
            other = self.nondiagnostic_text(state, framing)
        elif arm is Arm.PLACEBO_STALE:
            if donor is None:
                raise ValueError("Arm 3c requires a donor state")
            other = self.stale_text(donor, framing)
        elif arm is Arm.PLACEBO_SYNTACTIC:
            other = self.syntactic_text(state, framing)
        elif arm is Arm.PLACEBO_SCORE:
            if not score_offset:
                raise ValueError("Arm 3s requires a non-zero score_offset")
            other = self.score_falsified_text(state, framing, score_offset)
        elif arm is Arm.PLACEBO_MOVE:
            other = self.move_falsified_text(state, framing)
        else:
            raise ValueError(f"Arm {arm} does not inject a scaffold block")

        other_ids = self.tokenizer.encode(other)

        # One constant for every arm and every state, derived from this
        # tokeniser at construction. Both blocks are padded up to it, so block
        # size is invariant across arms AND across turns.
        #
        # The old code used the treatment's own length as the target, which
        # worked only because fixed-width rendering guaranteed treatment >=
        # placebo. Natural rendering removes that guarantee: an Arm 3c donor
        # five rounds ahead of its recipient renders LONGER than the
        # recipient's own treatment block.
        target = self.block_tokens

        for label, ids in (("treatment", treatment_ids), (arm.value, other_ids)):
            if len(ids) > target:
                raise TokenParityError(
                    f"{label} block is {len(ids)} tokens, exceeding the parity "
                    f"target of {target}. Padding can only lengthen; truncation "
                    f"would remove structural tokens. Re-run "
                    f"scripts/calibrate_block.py and raise "
                    f"ScaffoldConfig.treatment_block_tokens."
                )

        treatment_pad = target - len(treatment_ids)
        other_pad = target - len(other_ids)
        treatment_padded = list(treatment_ids) + [self._filler_id] * treatment_pad
        other_padded = list(other_ids) + [self._filler_id] * other_pad

        if not (len(treatment_padded) == len(other_padded) == target):
            raise TokenParityError(
                f"Parity enforcement failed for arm {arm.value}: "
                f"{len(treatment_padded)} vs {len(other_padded)} vs target {target}"
            )

        treatment_block = ScaffoldBlock(
            arm=Arm.TREATMENT,
            text=self.tokenizer.decode(treatment_padded),
            token_ids=tuple(treatment_padded),
            pad_tokens_added=treatment_pad,
        )
        arm_block = ScaffoldBlock(
            arm=arm,
            text=self.tokenizer.decode(other_padded),
            token_ids=tuple(other_padded),
            pad_tokens_added=other_pad,
        )
        return treatment_block, arm_block

    def calibrate_target(self, states, framing: Framing, margin: int = 2) -> int:
        """Largest block any reachable state produces, plus a margin.

        Call this once per (model, framing) with the REAL tokenizer, and write
        the result into ScaffoldConfig.treatment_block_tokens before a paid run.
        A target derived from a stub tokenizer is worthless: the whole point is
        that BPE, not character count, decides the length.
        """
        longest = 0
        for state in states:
            for text in (
                self.treatment_text(state, framing),
                self.nondiagnostic_text(state, framing),
                self.syntactic_text(state, framing),
            ):
                longest = max(longest, len(self.tokenizer.encode(text)))
        return longest + margin


class PromptAssembler:
    """Assembles the final prompt as token IDs.

    Returns IDs rather than a string so the scaffold block survives byte-exact
    into the model. Re-encoding an assembled string would let BPE merges across
    the block boundary silently change the token count we just asserted.
    """

    def __init__(self, tokenizer: Tokenizer, config: ScaffoldConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def assemble(
        self,
        *,
        game_config: GameConfig,
        state: GameState,
        framing: Framing,
        block: ScaffoldBlock | None,
        instruction_suffix: str,
        include_history: bool = True,
    ) -> list[int]:
        """Assemble the prompt. `include_history=False` drops [HISTORY] ONLY.

        WHY THE FLAG EXISTS
            Every experiment to date rendered [HISTORY] with every round in it,
            directly below the injected block. So arms 3c, 3s and 3m were never
            false-state manipulations - they were CONTRADICTION manipulations:
            the truth sat one section down, trivially checkable for the last
            move and arithmetically expensive for the score.

            That admits an alternative account of the entire exp6 result -
            "models discount a locally contradicted claim, and discount it more
            when the contradiction is cheap to verify" - which predicts the
            score/move asymmetry just as well as "the model conditions on the
            last-move field". Nothing in six experiments separates them, because
            the refutation was never removed.

            With history absent the block is the ONLY source of state. If
            behaviour then tracks the block, the finding is about conflict
            resolution between an injected summary and a raw log. If it still
            does not, the dissociation is earned rather than assumed.

        WHY IT IS A CALL ARGUMENT AND NOT A ScaffoldConfig FIELD
            `ExperimentConfig.fingerprint()` hashes ScaffoldConfig, and that
            fingerprint is stored on every episode row of every committed
            database. Adding a field there would change the fingerprint of every
            historical run - the documented reason the scratchpad variant is not
            a config field either (EXPERIMENTS.md, known defect 2). The
            condition is carried the same way that one is: by run_id, by
            `run_meta.config_json`, and by `turn_details.prompt_full`.

        WHAT MUST NOT MOVE
            With the default the section list is built in the same order, from
            the same three encodes, as before the flag existed. exp1-exp6
            reproduce byte-for-byte from HEAD; pinned by tests/test_no_history.py.

        POSITION (`ScaffoldConfig.insertion_index`)
            0  before the rules
            1  after the rules, before [HISTORY]   <- exp1-exp7, the default
            2  after [HISTORY], before the instruction   <- exp8's second level

            Index 2 is legal ONLY with history present. Without it there is no
            second seam and index 2 would land after the instruction instead of
            failing, which is the exact silent mis-placement this method
            refuses. Index `len(sections)` is refused for the same reason at
            both settings.

            `insertion_index` IS a `ScaffoldConfig` field and therefore does
            move `config_fingerprint` - but only for rows that set it. Every
            historical row carries 1 and keeps its fingerprint. That is why
            position needs no new plumbing while the template does.
        """
        sections: list[list[int]] = [
            self.tokenizer.encode(self._rules(game_config, framing)),
        ]
        if include_history:
            sections.append(self.tokenizer.encode(self._history_section(state, framing)))
        sections.append(self.tokenizer.encode(instruction_suffix))

        if block is not None:
            idx = self.config.insertion_index
            # The block's position relative to the RULES is the thing held fixed
            # across arms; lost-in-the-middle effects produce >30% swings from
            # position alone. Any index past the rules section means "after the
            # history" - a position that does not exist here, and would silently
            # place the block somewhere else (after the instruction) rather than
            # fail. Refuse instead.
            if not include_history and idx > 1:
                raise ValueError(
                    f"insertion_index {idx} places the block after [HISTORY], "
                    f"which is absent under include_history=False. The block "
                    f"would land in a different position than in every other "
                    f"experiment and the comparison would be confounded."
                )
            # WHY THIS IS TIGHTER THAN A RANGE CHECK.
            #
            # exp8 varies position deliberately: index 1 is after the rules and
            # before the history (every experiment to date), index 2 is after
            # the history and before the instruction. Both are real positions
            # and both must be assemblable.
            #
            # `idx == len(sections)` is NOT. It appends the block after the
            # instruction suffix, which is the last thing the model reads and
            # the thing that tells it what to emit. A block there is not "a
            # third position" - it is a different task, and under
            # include_history=False it is also the silent mis-placement the
            # guard above exists to catch, one index further along. The old
            # `0 <= idx <= len(sections)` admitted it. It is refused now, and
            # nothing in exp1-exp7 is affected because every one of them used
            # index 1.
            if idx < 0 or idx >= len(sections):
                raise ValueError(
                    f"insertion_index {idx} is not a legal block position for "
                    f"{len(sections)} sections (include_history={include_history}). "
                    f"Legal positions are 0..{len(sections) - 1}; "
                    f"{len(sections)} would place the block AFTER the "
                    f"instruction suffix, which must remain the final section."
                )
            sections.insert(idx, list(block.token_ids))
        return [tid for section in sections for tid in section]

    def block_section_index(self, *, include_history: bool = True) -> int:
        """Where `assemble` will put the block, validated the same way.

        Exists so a driver or a test can state the expected position WITHOUT
        re-deriving it from `config.insertion_index` by hand. Two places
        computing the same index independently is how a position confound
        becomes invisible.
        """
        n = 3 if include_history else 2
        idx = self.config.insertion_index
        if not include_history and idx > 1:
            raise ValueError(
                f"insertion_index {idx} places the block after [HISTORY], "
                f"which is absent under include_history=False."
            )
        if idx < 0 or idx >= n:
            raise ValueError(
                f"insertion_index {idx} is not a legal block position for {n} "
                f"sections (include_history={include_history})."
            )
        return idx

    def _rules(self, game_config: GameConfig, framing: Framing) -> str:
        swap = self.config.swap_action_labels
        c = render_action(Action.COOPERATE, framing, swap)
        d = render_action(Action.DEFECT, framing, swap)
        objective = {
            "none": None,
            "self_interest": "Your objective is to maximise your own total score.",
            "joint": "Your objective is to maximise the combined score of both players.",
        }[self.config.objective]
        p = game_config.payoffs
        lines = [
            "You are an agent in a repeated two-player interaction.",
            f"Each round you choose {c} or {d}. Your opponent chooses simultaneously.",
            f"If both choose {c}: you gain {p.reward}, opponent gains {p.reward}.",
            f"If you choose {d} and opponent {c}: you gain {p.temptation}, opponent gains {p.sucker}.",
            f"If you choose {c} and opponent {d}: you gain {p.sucker}, opponent gains {p.temptation}.",
            f"If both choose {d}: you gain {p.punishment}, opponent gains {p.punishment}.",
        ]
        if game_config.horizon_mode is HorizonMode.KNOWN:
            lines.append(f"The interaction lasts exactly {game_config.horizon} rounds.")
        elif game_config.horizon_mode is HorizonMode.STOCHASTIC:
            pct = int(round(game_config.continuation_probability * 100))
            lines.append(
                f"After each round there is a {pct}% chance the interaction continues "
                f"and a {100 - pct}% chance it ends."
            )
        # UNDISCLOSED: say nothing. Leaking the horizon changes the equilibrium.
        if objective is not None:
            lines.append(objective)
        return "\n".join(lines) + "\n\n"

    def _history_section(self, state: GameState, framing: Framing) -> str:
        return (
            HISTORY_HEADER + "\n"
            + render_history(state, framing, self.config.swap_action_labels)
            + "\n\n"
        )
