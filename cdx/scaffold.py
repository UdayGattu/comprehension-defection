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


def render_history(state: GameState, framing: Framing, swap_labels: bool = False) -> str:
    if not state.turns:
        return "(no rounds played yet)"
    return "\n".join(
        f"Round {t.index + 1}: you={render_action(t.agent_action, framing, swap_labels)} "
        f"opponent={render_action(t.opponent_action, framing, swap_labels)}"
        for t in state.turns
    )


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


class ScaffoldBuilder:
    """Builds treatment and placebo blocks with enforced token parity."""

    def __init__(self, tokenizer: Tokenizer, config: ScaffoldConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.filler_text, self._filler_id = self._resolve_filler()
        logger.info("scaffold filler token: %r (id=%d)", self.filler_text, self._filler_id)

        # The parity target is a property of the TOKENISER, not of the study.
        # A constant in config would be right for one model and silently wrong
        # for the next: the same templates are 34 tokens under Llama-3.1 and 84
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
            "parity target: %d tokens (%s)",
            self.block_tokens,
            "configured" if configured else "auto-derived",
        )

    def _derive_block_tokens(self) -> int:
        """Longest block any reachable state produces, over both framings.

        Framing-invariant on purpose: a target that changed with framing would
        make abstract and semantic runs non-comparable in prompt length, adding
        a confound to a factor that is supposed to vary only the labels.
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

        Numbers are rendered NATURALLY. They were previously zero-padded to a
        fixed width; run `sweep` showed the model reading the leading zero of
        "012" rather than the value, accounting for 49.7% of score-probe
        failures in the treatment arm. Constant block size is now enforced on
        token IDs in build_pair, not on characters here.
        """
        last_str = render_action_in_state(
            state.last_opponent_action(), framing, self._swap
        )
        return (
            "[STATE]\n"
            f"Your score: {state.agent_score:d}\n"
            f"Opponent score: {state.opponent_score:d}\n"
            f"Opponent's last move: {last_str}\n"
            f"Rounds played: {state.turn_index:d}\n"
        )

    def nondiagnostic_text(self, state: GameState, framing: Framing) -> str:
        """True but decision-irrelevant.

        Deliberately NOT stale/false state: false state carries a belief penalty
        (a single misleading sentence measurably degrades performance), which
        would inflate the treatment effect. Stale state lives in Arm 3c where
        measuring that penalty is the objective.

        DENSITY MATTERS AS MUCH AS TOKEN COUNT.
            The earlier version was two content lines. Padded to parity it
            became ~44% blank lines against a treatment that is ~94% text. Token
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
        return (
            "[STATE]\n"
            f"Round parity: {'even' if state.turn_index % 2 == 0 else 'odd'}\n"
            "Interaction type: repeated\n"
            "Payoff scale: integer\n"
            "Move space: binary\n"
            "Record status: logged\n"
        )

    def stale_text(self, donor: GameState, framing: Framing) -> str:
        """Identical template to treatment, populated from a different episode.

        Provides the scaffold-echo manipulation check: if the model's probe
        answer reproduces the donor's numbers, it demonstrably read the block.
        """
        return self.treatment_text(donor, framing)

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
        return (
            "[STATE]\n"
            "<node attr />\n"
            "<node attr />\n"
            "<node attr />\n"
            "<node attr />\n"
            "<node attr />\n"
            "<node attr />\n"
        )

    # ---- parity ----------------------------------------------------------

    def build_pair(
        self,
        arm: Arm,
        state: GameState,
        framing: Framing,
        donor: GameState | None = None,
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
    ) -> list[int]:
        sections: list[list[int]] = [
            self.tokenizer.encode(self._rules(game_config, framing)),
            self.tokenizer.encode(self._history_section(state, framing)),
            self.tokenizer.encode(instruction_suffix),
        ]
        if block is not None:
            idx = self.config.insertion_index
            if not 0 <= idx <= len(sections):
                raise ValueError(
                    f"insertion_index {idx} out of range for {len(sections)} sections"
                )
            sections.insert(idx, list(block.token_ids))
        return [tid for section in sections for tid in section]

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
            "[HISTORY]\n"
            + render_history(state, framing, self.config.swap_action_labels)
            + "\n\n"
        )