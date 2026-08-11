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


def render_action_fixed_width(
    action: Action | None, framing: Framing, swap_labels: bool = False
) -> str:
    """Fixed-width rendering for use inside the [STATE] block only.

    Without this, the treatment block's length varies by turn ("none" vs
    "Cooperate" vs "Defect"), and a stale-state donor can render LONGER than the
    treatment it must match — making parity unachievable by padding, since
    truncation is forbidden.

    Fixed width makes the treatment block a constant size, so every placebo is
    guaranteed to be shorter or equal and can always be padded up.
    """
    text = _NO_ACTION if action is None else render_action(action, framing, swap_labels)
    return text.ljust(_surface_width(framing))


def render_history(state: GameState, framing: Framing, swap_labels: bool = False) -> str:
    if not state.turns:
        return "(no rounds played yet)"
    return "\n".join(
        f"Round {t.index + 1}: you={render_action(t.agent_action, framing, swap_labels)} "
        f"opponent={render_action(t.opponent_action, framing, swap_labels)}"
        for t in state.turns
    )


class ScaffoldBuilder:
    """Builds treatment and placebo blocks with enforced token parity."""

    def __init__(self, tokenizer: Tokenizer, config: ScaffoldConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.filler_text, self._filler_id = self._resolve_filler()
        logger.info("scaffold filler token: %r (id=%d)", self.filler_text, self._filler_id)

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

        Every field is fixed-width so the block has a constant token count
        regardless of game state. This guarantees placebos are never longer than
        the treatment they must match.
        """
        width = self.config.score_field_width
        last_str = render_action_fixed_width(
            state.last_opponent_action(), framing, self._swap
        )
        return (
            "[STATE]\n"
            f"Your score: {state.agent_score:0{width}d}\n"
            f"Opponent score: {state.opponent_score:0{width}d}\n"
            f"Opponent's last move: {last_str}\n"
            f"Rounds played: {state.turn_index:0{width}d}\n"
        )

    def nondiagnostic_text(self, state: GameState, framing: Framing) -> str:
        """True but decision-irrelevant.

        Deliberately NOT stale/false state: false state carries a belief penalty
        (a single misleading sentence measurably degrades performance), which
        would inflate the treatment effect. Stale state lives in Arm 3c where
        measuring that penalty is the objective.
        """
        width = self.config.score_field_width
        return (
            "[STATE]\n"
            f"Rounds elapsed: {state.turn_index:0{width}d}\n"
            f"Round parity: {'even' if state.turn_index % 2 == 0 else 'odd '}\n"
        )

    def stale_text(self, donor: GameState, framing: Framing) -> str:
        """Identical template to treatment, populated from a different episode.

        Provides the scaffold-echo manipulation check: if the model's probe
        answer reproduces the donor's numbers, it demonstrably read the block.
        """
        return self.treatment_text(donor, framing)

    def syntactic_text(self, state: GameState, framing: Framing) -> str:
        """Format noise. Lower bound on the pure perturbation effect."""
        return "[STATE]\n<meta />\n<meta />\n<meta />\n<meta />\n"

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
        target = len(treatment_ids)

        if len(other_ids) > target:
            raise TokenParityError(
                f"Arm {arm.value} block is {len(other_ids)} tokens vs treatment's "
                f"{target}. Padding can only lengthen; truncation would remove "
                f"structural tokens. Shorten the placebo template or draw a "
                f"different donor."
            )

        pad = target - len(other_ids)
        padded_ids = list(other_ids) + [self._filler_id] * pad

        if len(padded_ids) != target:
            raise TokenParityError(
                f"Parity enforcement failed for arm {arm.value}: "
                f"{len(padded_ids)} != {target}"
            )

        treatment_block = ScaffoldBlock(
            arm=Arm.TREATMENT,
            text=treatment,
            token_ids=tuple(treatment_ids),
            pad_tokens_added=0,
        )
        arm_block = ScaffoldBlock(
            arm=arm,
            text=self.tokenizer.decode(padded_ids),
            token_ids=tuple(padded_ids),
            pad_tokens_added=pad,
        )
        return treatment_block, arm_block


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
