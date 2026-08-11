"""Inference backends and the decision record they produce.

Two readout modes, both first-class experimental conditions:

  LOGIT      one forward pass, action taken from argmax over the action tokens,
             zero tokens generated. Cheap and deterministic under argmax, and it
             yields the probability mass on valid actions for free.

  SCRATCHPAD the model generates a reasoning block, then emits an action. This
             matches the regime the prior literature ran in. Omitting it invites
             the objection that we crippled the models by forbidding a scratchpad.

`action_mass_total` is recorded on every decision. If the model wanted to emit
"Based on the history..." rather than an action token, the mass on valid actions
will be near zero, and renormalising two near-zero logits measures noise. That is
logged and reported as an off-task rate, not silently discarded: it is direct
evidence about whether the model was playing the game at all.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .config import Action, Framing, ModelConfig, ReadoutMode

logger = logging.getLogger(__name__)

# Surface forms are FRAMING-DEPENDENT. This was originally hardcoded to the
# abstract X/Y forms, which meant a semantic-framing run measured probability
# mass on tokens the prompt never asked for — producing a 100% off-task rate and
# actions decided entirely by the tie-break. Caught by the local pilot.
#
# "X", " X" and "X\n" are distinct token IDs in every BPE vocabulary we target;
# missing one silently under-measures the action mass.
ACTION_SURFACE_FORMS_BY_FRAMING: dict[Framing, dict[Action, tuple[str, ...]]] = {
    Framing.ABSTRACT: {
        Action.COOPERATE: ("X", " X", "X\n"),
        Action.DEFECT: ("Y", " Y", "Y\n"),
    },
    Framing.SEMANTIC: {
        Action.COOPERATE: ("Cooperate", " Cooperate", "cooperate", " cooperate"),
        Action.DEFECT: ("Defect", " Defect", "defect", " defect"),
    },
}

# Retained for the abstract default; prefer the framing-aware mapping above.
ACTION_SURFACE_FORMS = ACTION_SURFACE_FORMS_BY_FRAMING[Framing.ABSTRACT]


@dataclass(frozen=True)
class Decision:
    action: Action
    logit_mass_cooperate: float
    logit_mass_defect: float
    action_mass_total: float
    logit_gap: float
    top_tokens: tuple[tuple[str, float], ...] = ()
    scratchpad: str | None = None

    @property
    def is_off_task(self) -> bool:
        return self.action_mass_total < 0.1

    @property
    def is_near_tie(self) -> bool:
        """Decisions inside the floating-point noise band are the only ones at
        risk from vLLM's non-deterministic batching. Reporting the share of
        these is how we characterise reproducibility instead of claiming a
        bit-identity we cannot deliver."""
        return self.logit_gap < 1e-4


class LLMBackend(Protocol):
    model_config: ModelConfig

    def decide(
        self,
        prompt_token_ids: Sequence[int],
        readout_mode: ReadoutMode,
        seed: int,
        framing: Framing = Framing.ABSTRACT,
    ) -> Decision: ...

    def probe(self, prompt_token_ids: Sequence[int], seed: int) -> str: ...


@dataclass
class CharTokenizer:
    """Byte-level tokenizer for development and tests.

    Deliberately trivial and dependency-free so the entire engine can be built
    and tested on a laptop with no model weights. It is NOT representative of
    BPE digit-merging behaviour - that is what scripts/tokenizer_check.py is for.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


@dataclass
class DummyBackend:
    """Deterministic stand-in used for smoke tests and the resume test.

    Simulates the phenomenon under study: a defect-leaning default that becomes
    responsive to the opponent only when explicit state is present in the
    prompt. This lets the full pipeline - including the sign-flip analysis - be
    exercised end to end before any GPU is rented. It proves the plumbing, never
    the hypothesis.
    """

    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(model_id="dummy/deterministic")
    )
    defect_bias: float = 0.9

    def decide(
        self,
        prompt_token_ids: Sequence[int],
        readout_mode: ReadoutMode,
        seed: int,
        framing: Framing = Framing.ABSTRACT,
    ) -> Decision:
        import random

        rng = random.Random(seed ^ (len(prompt_token_ids) * 2654435761))
        has_state_block = self._contains_state_marker(prompt_token_ids)
        bias = 0.5 if has_state_block else self.defect_bias

        p_defect = min(max(rng.gauss(bias, 0.05), 0.01), 0.99)
        p_coop = 1.0 - p_defect
        total = 0.85 if not has_state_block else 0.95

        action = Action.DEFECT if p_defect >= p_coop else Action.COOPERATE
        return Decision(
            action=action,
            logit_mass_cooperate=p_coop * total,
            logit_mass_defect=p_defect * total,
            action_mass_total=total,
            logit_gap=abs(math.log(p_defect) - math.log(p_coop)),
            top_tokens=(("X", p_coop * total), ("Y", p_defect * total)),
            scratchpad="(dummy scratchpad)" if readout_mode is ReadoutMode.SCRATCHPAD else None,
        )

    def probe(self, prompt_token_ids: Sequence[int], seed: int) -> str:
        import random

        return random.Random(seed).choice(["0", "1", "2", "3"])

    @staticmethod
    def _contains_state_marker(prompt_token_ids: Sequence[int]) -> bool:
        marker = list("[STATE]".encode("utf-8"))
        n = len(marker)
        return any(
            list(prompt_token_ids[i : i + n]) == marker
            for i in range(max(0, len(prompt_token_ids) - n + 1))
        )


class VLLMBackend:
    """Production backend. Requires a GPU; not importable-and-usable on a Mac.

    Deliberately left as a thin, explicit stub rather than a speculative
    implementation: it must be written against the measured behaviour of the
    installed vLLM version, not against assumptions. Fill this in during the
    one-hour paid calibration session described in the spec, after measuring
    real generation throughput.

    Implementation notes for that session:
      - pass prompt_token_ids, never a re-encoded string, or the asserted token
        parity is lost at the boundary
      - request logprobs for the full action surface-form set and sum them
      - for LOGIT mode use max_tokens=0/1 and take argmax; do not sample
      - record the pre-renormalisation mass as action_mass_total
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        raise NotImplementedError(
            "VLLMBackend is intentionally unimplemented. Build it during the paid "
            "calibration session against the installed vLLM version. Use "
            "DummyBackend for all Mac-side development."
        )
