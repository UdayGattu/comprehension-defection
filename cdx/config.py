"""Typed configuration. No magic numbers anywhere else in the codebase.

Every experimental parameter that could plausibly be varied lives here and is
serialised into the database alongside every row, so any result can be traced
back to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class Action(str, Enum):
    """Internal action representation. Rendering to X/Y or Cooperate/Defect is
    a *framing* concern handled in scaffold.py, never here."""

    COOPERATE = "C"
    DEFECT = "D"


class Arm(str, Enum):
    BASELINE = "1"
    INLINE_PROBE = "2"
    TREATMENT = "3"
    PLACEBO_NONDIAGNOSTIC = "3b"
    PLACEBO_STALE = "3c"
    PLACEBO_SYNTACTIC = "3d"

    @property
    def injects_block(self) -> bool:
        """Whether this arm inserts a scaffold block into the prompt."""
        return self in {
            Arm.TREATMENT,
            Arm.PLACEBO_NONDIAGNOSTIC,
            Arm.PLACEBO_STALE,
            Arm.PLACEBO_SYNTACTIC,
        }


class ReadoutMode(str, Enum):
    """Readout is an experimental FACTOR, not an implementation detail.

    LOGIT denies the model a scratchpad, which is cheap and deterministic under
    argmax but is not the regime the prior literature ran in. SCRATCHPAD matches
    prior work. Running only LOGIT invites the (correct) objection that we
    crippled the models.
    """

    LOGIT = "logit"
    SCRATCHPAD = "scratchpad"


class HorizonMode(str, Enum):
    KNOWN = "known"
    UNDISCLOSED = "undisclosed"
    STOCHASTIC = "stochastic"


class Framing(str, Enum):
    ABSTRACT = "abstract"      # X / Y
    SEMANTIC = "semantic"      # Cooperate / Defect


class OpponentPolicy(str, Enum):
    TFT = "tft"
    ALLD = "alld"
    ALLC = "allc"
    GRIM = "grim"
    QTABLE = "qtable"
    LLM = "llm"

    @property
    def is_scripted(self) -> bool:
        return self is not OpponentPolicy.LLM


@dataclass(frozen=True)
class Payoffs:
    """Prisoner's Dilemma payoff matrix from the acting agent's perspective.

    Validity requires T > R > P > S (it is a dilemma) and 2R > T + S (mutual
    cooperation beats alternating exploitation). Both are asserted at
    construction; a malformed matrix silently invalidates every downstream
    game-theoretic claim, so this must fail loudly.
    """

    temptation: int = 5   # T: I defect, they cooperate
    reward: int = 3       # R: both cooperate
    punishment: int = 1   # P: both defect
    sucker: int = 0       # S: I cooperate, they defect

    def __post_init__(self) -> None:
        t, r, p, s = self.temptation, self.reward, self.punishment, self.sucker
        if not (t > r > p > s):
            raise ValueError(f"Not a prisoner's dilemma: need T>R>P>S, got {t}>{r}>{p}>{s}")
        if 2 * r <= t + s:
            raise ValueError(f"Alternating exploitation dominates cooperation: 2R={2*r} <= T+S={t+s}")

    def payoff(self, mine: Action, theirs: Action) -> int:
        if mine is Action.COOPERATE:
            return self.reward if theirs is Action.COOPERATE else self.sucker
        return self.temptation if theirs is Action.COOPERATE else self.punishment

    @property
    def tft_stability_threshold(self) -> float:
        """Minimum continuation probability for cooperation to be sustainable
        against a retaliating opponent: gamma >= (T-R)/(T-P).

        For T=5,R=3,P=1 this is 0.5, so the configured gamma=0.9 is comfortably
        above threshold. Phase 2 is ill-posed below it.
        """
        return (self.temptation - self.reward) / (self.temptation - self.punishment)


@dataclass(frozen=True)
class GameConfig:
    payoffs: Payoffs = field(default_factory=Payoffs)
    horizon: int = 20
    horizon_mode: HorizonMode = HorizonMode.KNOWN
    continuation_probability: float = 0.9   # only used when horizon_mode is STOCHASTIC

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError(f"horizon must be >= 2, got {self.horizon}")
        if not 0.0 < self.continuation_probability < 1.0:
            raise ValueError(
                f"continuation_probability must be in (0,1), got {self.continuation_probability}"
            )
        if self.horizon_mode is HorizonMode.STOCHASTIC:
            threshold = self.payoffs.tft_stability_threshold
            if self.continuation_probability < threshold:
                raise ValueError(
                    f"gamma={self.continuation_probability} is below the TFT-stability "
                    f"threshold {threshold:.3f} for these payoffs; cooperation is not "
                    f"sustainable and Phase 2 would be ill-posed"
                )


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str = "main"
    temperature: float = 0.7
    max_scratchpad_tokens: int = 192
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class ScaffoldConfig:
    """Placement and padding policy for the injected block.

    insertion_index is the position of the block within the assembled prompt
    sections and MUST be identical across arms: lost-in-the-middle effects
    produce >30% swings from position alone, which would masquerade as a
    treatment effect.
    """

    insertion_index: int = 1
    max_donor_draws: int = 64
    score_field_width: int = 3

    # Inverts which label means which action. A model that always picks the
    # first-listed option is showing position bias, not strategy; swapping the
    # mapping is the only way to distinguish the two. Must be logged.
    swap_action_labels: bool = False

    # Whether the prompt states an objective at all.
    #
    # The original prompt described the game and never said what the agent was
    # trying to achieve. Qwen2.5-7B responded by cooperating 20/20 with
    # P(defect) = 0.000 even against an unconditional cooperator, where
    # defecting every round is worth 100 against 60. An instruction-tuned model
    # given no goal defaults to agreeable behaviour.
    #
    # The literature this work corrects reports 80-92% defection, so their
    # prompts state an objective. Omitting one does not reproduce the
    # phenomenon we set out to measure. This is therefore a FACTOR: NONE is the
    # honest baseline, SELF_INTEREST matches prior work.
    objective: str = "self_interest"   # none | self_interest | joint

    # Padding filler, tried in order; the first that encodes to exactly ONE token
    # for the active tokenizer is used.
    #
    # This is a list rather than a single value because of a measured fact:
    # Mistral-7B-Instruct-v0.3 has NO single-token newline, while Qwen2.5 does.
    # A hardcoded "\n" default therefore halts on Mistral at startup. The
    # selected filler is logged and must be recorded per model in the methods
    # section — padding differs by tokenizer, and that has to be disclosed.
    filler_candidates: tuple[str, ...] = ("\n", " ", ".", "-")


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    game: GameConfig = field(default_factory=GameConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    probe_text_hash: str = ""    # must be set before the P1b probe pass; see spec S9
    off_task_mass_threshold: float = 0.1

    def fingerprint(self) -> str:
        """Stable hash of the full configuration, stored on every row."""
        return stable_hash(json.dumps(_jsonable(asdict(self)), sort_keys=True))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def stable_hash(text: str) -> str:
    """Deterministic across processes and Python versions.

    Python's builtin hash() is salted per-process, which would silently break
    seed reproducibility across runs. Never use it for seeding.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
