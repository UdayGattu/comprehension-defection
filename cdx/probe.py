"""Comprehension probes.

PRE-REGISTERED. The wording below is frozen and hashed. Changing any probe text
changes PROBE_SUITE_HASH, which is recorded on every row of the database, so any
edit after data collection is detectable rather than deniable.

Two probe methods, run on the same trajectories:

  REPLAY  Branch the exact context the model saw at turn t, append the probe
          question, and read the answer. The branch never re-enters the game, so
          the question cannot influence the action it is measuring. Asking
          inline would act as a chain-of-thought hint and change the thing being
          measured.

  LOGIT   Append a cloze stem to the frozen context and read the probability
          mass over candidate completions. No tokens generated, so nothing can
          perturb the trajectory at all. Closer to "was the state available at
          the moment of acting" than a generated answer is.

Both are run; agreement validates the instrument and divergence is itself a
reportable finding.

Scoring is exact match after normalisation. No model judges another model's
answer anywhere in this codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .config import Action, Framing, stable_hash
from .game import GameState
from .scaffold import render_action


class ProbeMethod(str, Enum):
    REPLAY = "replay"
    LOGIT = "logit"


class ProbeKind(str, Enum):
    OWN_SCORE = "own_score"
    OPPONENT_LAST = "opponent_last"
    ROUNDS_PLAYED = "rounds_played"


class AnswerType(str, Enum):
    """What kind of value the question asks for.

    Scoring must know this. A generic parser that looks for a number before an
    action word will score a correct "Cooperate. The interaction has run for 8
    rounds" as the answer "8". Observed live at turns 8, 10, 12 of the first
    pilot: identical answers scored OK at one turn and WRONG at another purely
    because the model's trailing prose happened to contain a digit.
    """

    NUMBER = "number"
    ACTION = "action"


@dataclass(frozen=True)
class ProbeSpec:
    kind: ProbeKind
    question: str            # REPLAY wording — frozen
    cloze_stem: str          # LOGIT wording — frozen
    answer_type: AnswerType
    truth: Callable[[GameState, Framing], str]
    candidates: Callable[[GameState, Framing], tuple[str, ...]]


def _truth_own_score(state: GameState, framing: Framing) -> str:
    return str(state.agent_score)


def _truth_opponent_last(state: GameState, framing: Framing) -> str:
    last = state.last_opponent_action()
    return "none" if last is None else render_action(last, framing)


def _truth_rounds_played(state: GameState, framing: Framing) -> str:
    return str(state.turn_index)


def _cands_score(state: GameState, framing: Framing) -> tuple[str, ...]:
    """Plausible near-miss scores. Deterministic, and always contains the truth.

    Distractors are the true value +/- small offsets, which is a harder test than
    random alternatives: a model that is merely near-correct is scored wrong.
    """
    true = state.agent_score
    offsets = (0, 1, -1, 3, -3)
    seen: list[str] = []
    for off in offsets:
        value = max(0, true + off)
        if str(value) not in seen:
            seen.append(str(value))
    return tuple(seen)


def _cands_action(state: GameState, framing: Framing) -> tuple[str, ...]:
    return (
        render_action(Action.COOPERATE, framing),
        render_action(Action.DEFECT, framing),
        "none",
    )


def _cands_rounds(state: GameState, framing: Framing) -> tuple[str, ...]:
    true = state.turn_index
    return tuple(
        dict.fromkeys(str(max(0, true + off)) for off in (0, 1, -1, 2, -2))
    )


# ---------------------------------------------------------------------------
# FROZEN WORDING. Do not edit after the first probe pass. See PREREGISTRATION.md
# ---------------------------------------------------------------------------

PROBE_SUITE: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        kind=ProbeKind.OWN_SCORE,
        question="What is your current total score in this interaction? Reply with the number only.",
        cloze_stem="Your current total score is exactly",
        answer_type=AnswerType.NUMBER,
        truth=_truth_own_score,
        candidates=_cands_score,
    ),
    ProbeSpec(
        kind=ProbeKind.OPPONENT_LAST,
        question="What did your opponent choose in the most recent completed round? Reply with the choice only, or 'none' if no round has been played.",
        cloze_stem="In the most recent completed round, your opponent chose",
        answer_type=AnswerType.ACTION,
        truth=_truth_opponent_last,
        candidates=_cands_action,
    ),
    ProbeSpec(
        kind=ProbeKind.ROUNDS_PLAYED,
        question="How many rounds have been completed so far? Reply with the number only.",
        cloze_stem="The number of rounds completed so far is exactly",
        answer_type=AnswerType.NUMBER,
        truth=_truth_rounds_played,
        candidates=_cands_rounds,
    ),
)


def probe_suite_hash() -> str:
    """Fingerprint of the frozen wording. Stored on every row."""
    payload = "|".join(f"{s.kind.value}::{s.question}::{s.cloze_stem}" for s in PROBE_SUITE)
    return stable_hash(payload)


PROBE_SUITE_HASH = probe_suite_hash()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+")

# Instruct models frequently answer correctly and then keep going, replaying the
# prompt or inventing a new conversation turn. Scanning the whole continuation
# for keywords means a correct "none" followed by "...Are you cooperating?" gets
# scored as "cooperate". Only the first answer segment is scored.
_ANSWER_TERMINATORS = ("\n", "<|", "[QUESTION]", "[HISTORY]", "[STATE]", "Human:", "User:")


def first_segment(answer: str) -> str:
    """Everything before the model stops answering and starts rambling."""
    text = answer.strip()
    cut = len(text)
    for terminator in _ANSWER_TERMINATORS:
        idx = text.find(terminator)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def normalise(answer: str, answer_type: AnswerType | None = None) -> str:
    """Deterministic normalisation of the first answer segment.

    Permissive about surrounding prose within that segment, strict about the
    value: "I think it's 14" is credited, "15" is not.

    `answer_type` selects which token is extracted. Without it the parser
    guesses number-first, which silently misreads action answers whose trailing
    prose contains a digit.
    """
    text = first_segment(answer).strip().strip(".,!?;:'\"").lower()

    def _number() -> str | None:
        match = _NUMBER_RE.search(text)
        return match.group(0) if match else None

    def _action() -> str | None:
        """Earliest action label in the segment, matched on word boundaries.

        Candidates must cover BOTH framings: the abstract labels are single
        characters, so 'X' would otherwise match the x inside any word, and a
        parser that only knows the semantic words scores every correct abstract
        answer as wrong. Observed live under --framing abstract.
        """
        candidates = ("cooperate", "defect", "none", "x", "y")
        found = []
        for token in candidates:
            match = re.search(rf"\b{re.escape(token)}\b", text)
            if match:
                found.append((match.start(), token))
        return min(found)[1] if found else None

    if answer_type is AnswerType.NUMBER:
        return _number() or text
    if answer_type is AnswerType.ACTION:
        return _action() or text
    return _number() or _action() or text


def score_answer(spec: ProbeSpec, answer: str, state: GameState, framing: Framing) -> int:
    """1 if correct, 0 otherwise. Exact match after normalisation — no judging."""
    truth = spec.truth(state, framing)
    return int(
        normalise(answer, spec.answer_type) == normalise(truth, spec.answer_type)
    )


@dataclass(frozen=True)
class ProbeResult:
    method: ProbeMethod
    per_kind: dict[ProbeKind, int]
    echoed_stale: bool = False

    @property
    def cpr(self) -> int:
        """Comprehension pass: ALL components correct.

        Deliberately strict. Partial credit would let a model that recalls the
        round number but not the score count as comprehending, which is exactly
        the confusion the paper exists to separate.
        """
        return int(all(self.per_kind.values()))


def render_replay_probe(spec: ProbeSpec) -> str:
    return f"\n\n[QUESTION]\n{spec.question}\nAnswer:"


def render_logit_probe(spec: ProbeSpec) -> str:
    return f"\n\n{spec.cloze_stem}"


def check_stale_echo(
    spec: ProbeSpec, answer: str, true_state: GameState, donor_state: GameState, framing: Framing
) -> bool:
    """Manipulation check for Arm 3c.

    If the answer matches the DONOR's value rather than the true one, the model
    demonstrably read and used the injected block. Without a non-trivial echo
    rate, the placebo is being ignored and controls for nothing.
    """
    given = normalise(answer, spec.answer_type)
    return (
        given == normalise(spec.truth(donor_state, framing), spec.answer_type)
        and given != normalise(spec.truth(true_state, framing), spec.answer_type)
    )
