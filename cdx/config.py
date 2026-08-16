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

    # WHY 3s AND 3m EXIST
    #
    # Arm 3c replaces the WHOLE block with another episode's, so a behavioural
    # difference cannot be attributed to any particular field. Two facts
    # measured after exp3-exp5 say that ambiguity was hiding the mechanism:
    #
    #   1. analysis/12 found the score falsification arm 3c actually delivers
    #      is tiny. In exp3_qwen_sem vs TFT, sd(d) = 3.30 and only 0.1% of rows
    #      carry |d| >= 15. At the measured ~0.01 defection per point that
    #      predicts a 3pp shift. The OBSERVED effect in that cell is 24pp,
    #      replicated in exp2. The score cannot be what moved it.
    #
    #   2. The content effect is larger against TFT than ALLC in 8 of 9 cells,
    #      and enormously so in the two qwen semantic cells. Under ALLC the
    #      opponent's last move is ALWAYS Cooperate, so a donor drawn from the
    #      same cell shows Cooperate too and that field cannot be falsified.
    #      Under TFT it mirrors the agent and is falsified constantly.
    #
    # Together those point at "Opponent's last move" rather than the score -
    # which matters, because the last move IS decision-relevant against TFT.
    # The standing objection to this study is that cumulative score is a sunk
    # variable, so falsifying it should change nothing. That objection does not
    # apply to the last move.
    #
    # 3s and 3m falsify exactly one field each, DELIBERATELY rather than by
    # donor sampling, so the contrast is identified per field:
    #
    #   3s   Your score offset by +/-15, everything else true
    #   3m   Opponent's last move flipped, everything else true
    #
    # Arm 3m against ALLC is a condition the existing corpus cannot produce:
    # the block asserts a betrayal while the [HISTORY] section directly below
    # lists an unbroken run of cooperation. The lie and its refutation sit in
    # the same context window.
    PLACEBO_SCORE = "3s"
    PLACEBO_MOVE = "3m"

    @property
    def injects_block(self) -> bool:
        """Whether this arm inserts a scaffold block into the prompt."""
        return self in {
            Arm.TREATMENT,
            Arm.PLACEBO_NONDIAGNOSTIC,
            Arm.PLACEBO_STALE,
            Arm.PLACEBO_SYNTACTIC,
            Arm.PLACEBO_SCORE,
            Arm.PLACEBO_MOVE,
        }

    @property
    def falsifies_field(self) -> bool:
        """Arms that show a value contradicting the true game state.

        Distinct from `injects_block`: 3b and 3d inject a block that is true (or
        contentless), while these assert something false. Only these arms need a
        displayed-value column recorded so the falsification is auditable.
        """
        return self in {Arm.PLACEBO_STALE, Arm.PLACEBO_SCORE, Arm.PLACEBO_MOVE}


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
    """Opponent policies. Only TFT and ALLC have ever been run.

    NOT IMPLEMENTED, and named here so nobody infers a capability from the
    enum:

      QTABLE  IMPLEMENTED AND WORKING, but never run. An earlier version of
              this comment claimed the Q update was dead and the learner was a
              coin flip; that was wrong. `Game.step` calls `observe()` through a
              duck-typed local binding (game.py:261), which a grep for
              `.observe(` cannot see. Measured through the real engine: the
              table fills in 500/500 episodes and the opponent cooperates on
              11.3% of turns against an always-defecting agent, against 49.9%
              with the update disabled. Real limitations: the table resets every
              episode (fresh instance per EpisodeKey), the state is the agent's
              last action alone, and runner.py:287/:310 and optimal.py:73
              EXCLUDE it from regret and optimal-sequence computation.
      LLM     `build_opponent` raises on it: "LLM opponents are Phase 2 and are
              constructed by the runner". No runner constructs one.

    ALLD and GRIM are implemented in _SCRIPTED and correct, but no experiment
    has run them either.
    """

    TFT = "tft"
    ALLD = "alld"
    ALLC = "allc"
    GRIM = "grim"
    QTABLE = "qtable"      # NOT IMPLEMENTED - a coin flip, see above
    LLM = "llm"            # NOT IMPLEMENTED - build_opponent raises

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

    IDENTICAL ACROSS ARMS, NOT ACROSS RUNS. exp8 varies it BETWEEN runs, which
    is the point: a >30% swing from position alone is a statement about this
    instrument that the instrument has never actually measured, because no
    driver in exp1-exp7 ever set this to anything but 1. Within any single run
    it still holds for every arm, so every contrast stays position-matched.

    This one IS a config field and does move `config_fingerprint` - safely,
    because every historical row carries the default 1 and only new rows carry
    2. Contrast with the state TEMPLATE, which is deliberately NOT a field here
    (see cdx.scaffold.StateTemplate): adding it would rewrite the fingerprint
    of every row ever written.

    Legal values, enforced in PromptAssembler.assemble:
        0  before the rules
        1  after the rules, before [HISTORY]        (exp1-exp7)
        2  after [HISTORY], before the instruction  (exp8; requires history)
    """

    insertion_index: int = 1
    max_donor_draws: int = 64

    # RETIRED as a padding mechanism. Kept only so old configs still load.
    #
    # Numeric fields were rendered zero-padded ("012") to give the treatment
    # block a constant token count. Measured consequence: Llama-3.1-8B read the
    # leading zero instead of the number. 49.7% of ALL treatment score probes
    # (6,366 of 12,800; 70.1% of the failures alone) -- score-probe
    # failures in run `sweep` are attributable to this single format spec.
    # Space-padding does not fix it - " 12" and "100" tokenise to different
    # counts under byte-level BPE, verified by scripts/tokenizer_check.py.
    #
    # Character padding cannot deliver token parity. ID-level padding can, and
    # already exists. See treatment_block_tokens.
    score_field_width: int = 3

    # Token count that EVERY injected block is padded to, for every arm and
    # every game state. Padding is applied by appending the single-token filler
    # to raw token IDs, which cannot trigger a BPE merge because the tokenizer
    # is never re-run over the result.
    #
    # LEAVE THIS AT 0. 0 means AUTO: ScaffoldBuilder sweeps every reachable
    # state at construction and derives the value from the tokeniser it was
    # handed.
    #
    # The correct value is a property of the TOKENISER, not of the study. The
    # same templates are 34 tokens under Llama-3.1 and 119 under a
    # character-level tokeniser. A constant here is therefore right for one
    # model and silently wrong for the next, which would break every
    # multi-model comparison in the design.
    #
    # A non-zero value is honoured only if it is >= the derived minimum;
    # otherwise construction raises. Use it to pin a value across a
    # re-analysis, never to guess one. The value in force is logged and written
    # to run_meta.
    treatment_block_tokens: int = 0

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