"""Deterministic Iterated Prisoner's Dilemma engine and scripted opponents.

The engine is the single source of truth for game state. No language model ever
scores, adjudicates or terminates a game. Everything here is pure Python and is
held to a bit-identical reproducibility standard, unlike the inference layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .config import Action, GameConfig, HorizonMode, OpponentPolicy, Payoffs
from .seeding import EpisodeKey, purpose_rng

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    index: int
    agent_action: Action
    opponent_action: Action
    agent_payoff: int
    opponent_payoff: int


@dataclass
class GameState:
    """Authoritative game state. The prompt is a *rendering* of this; the two
    must never diverge, which is why nothing else may mutate it."""

    config: GameConfig
    turns: list[Turn] = field(default_factory=list)
    terminated: bool = False

    @property
    def turn_index(self) -> int:
        return len(self.turns)

    @property
    def agent_score(self) -> int:
        return sum(t.agent_payoff for t in self.turns)

    @property
    def opponent_score(self) -> int:
        return sum(t.opponent_payoff for t in self.turns)

    @property
    def agent_history(self) -> list[Action]:
        return [t.agent_action for t in self.turns]

    @property
    def opponent_history(self) -> list[Action]:
        return [t.opponent_action for t in self.turns]

    @property
    def agent_defected_ever(self) -> bool:
        return any(t.agent_action is Action.DEFECT for t in self.turns)

    def last_opponent_action(self) -> Action | None:
        return self.turns[-1].opponent_action if self.turns else None

    def turns_remaining(self) -> int | None:
        """None when the horizon is not disclosed to the agent.

        Callers rendering prompts must respect this: leaking the horizon in an
        undisclosed condition changes the equilibrium and silently invalidates
        the cell.
        """
        if self.config.horizon_mode is HorizonMode.UNDISCLOSED:
            return None
        return max(0, self.config.horizon - self.turn_index)


class Opponent(Protocol):
    """Scripted opponents are pure functions of observable history.

    They must not consult randomness unless explicitly seeded, and must not see
    the agent's current action before choosing their own.
    """

    policy: OpponentPolicy

    def move(self, state: GameState) -> Action: ...


@dataclass
class TitForTat:
    policy: OpponentPolicy = OpponentPolicy.TFT

    def move(self, state: GameState) -> Action:
        if not state.turns:
            return Action.COOPERATE
        return state.turns[-1].agent_action


@dataclass
class AlwaysDefect:
    policy: OpponentPolicy = OpponentPolicy.ALLD

    def move(self, state: GameState) -> Action:
        return Action.DEFECT


@dataclass
class AlwaysCooperate:
    policy: OpponentPolicy = OpponentPolicy.ALLC

    def move(self, state: GameState) -> Action:
        return Action.COOPERATE


@dataclass
class GrimTrigger:
    policy: OpponentPolicy = OpponentPolicy.GRIM

    def move(self, state: GameState) -> Action:
        return Action.DEFECT if state.agent_defected_ever else Action.COOPERATE


@dataclass
class TabularQLearner:
    """Deterministic-given-seed tabular Q-learner. WORKS, but NEVER RUN.

    CORRECTION, 2026-08-15. An earlier version of this docstring claimed the Q
    update was dead code and that this class "behaves as a fair coin", citing
    50.4% cooperation against an always-defecting agent. THAT WAS WRONG, and it
    was wrong because of how it was tested: the learner was driven by hand and
    `observe()` was never called, so of course the table stayed empty. A grep
    for `\.observe(` also finds nothing, because `Game.step` invokes it through
    a duck-typed local binding:

        observe = getattr(self.opponent, "observe", None)      # game.py:261
        if callable(observe):
            observe(prev_state, opponent_action, turn.opponent_payoff, self.state)

    Measured through the real engine: over 500 episodes x 20 turns against an
    agent that defects every turn, the table is NON-EMPTY in 500/500 episodes
    and the opponent cooperates on 11.3% of turns. Disabling `observe`
    reproduces 49.9%, which is where the bogus 50.4% came from.

    WHAT IS ACTUALLY TRUE, and it is a limitation rather than a defect:
      * `build_opponent` constructs a fresh instance per EpisodeKey, so the
        table resets every episode. Learning is WITHIN-episode only: 19
        transitions over a state space of the agent's last action alone
        (3 states x 2 actions).
      * No experiment has ever run it. Every driver passes `--opponents tft
        allc`.
      * runner.py:287, runner.py:310 and optimal.py:73 EXCLUDE it from the
        regret and optimal-sequence computations, so a run would silently lose
        two derived measures.

    Intended as an ablation - if an 8B model cannot outperform a small Q-table
    in a mixed-motive game, the language model is contributing nothing.
    """

    seed: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    policy: OpponentPolicy = OpponentPolicy.QTABLE
    _q: dict[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import random as _random

        self._rng = _random.Random(self.seed)

    def _state_key(self, state: GameState) -> str:
        last = state.turns[-1].agent_action.value if state.turns else "-"
        return last

    def move(self, state: GameState) -> Action:
        s = self._state_key(state)
        if self._rng.random() < self.epsilon:
            return self._rng.choice([Action.COOPERATE, Action.DEFECT])
        qc = self._q.get((s, Action.COOPERATE.value), 0.0)
        qd = self._q.get((s, Action.DEFECT.value), 0.0)
        if qc == qd:
            return self._rng.choice([Action.COOPERATE, Action.DEFECT])
        return Action.COOPERATE if qc > qd else Action.DEFECT

    def observe(self, prev: GameState, action: Action, reward: int, nxt: GameState) -> None:
        s, s2 = self._state_key(prev), self._state_key(nxt)
        best_next = max(
            self._q.get((s2, Action.COOPERATE.value), 0.0),
            self._q.get((s2, Action.DEFECT.value), 0.0),
        )
        key = (s, action.value)
        old = self._q.get(key, 0.0)
        self._q[key] = old + self.alpha * (reward + self.gamma * best_next - old)


_SCRIPTED: dict[OpponentPolicy, type] = {
    OpponentPolicy.TFT: TitForTat,
    OpponentPolicy.ALLD: AlwaysDefect,
    OpponentPolicy.ALLC: AlwaysCooperate,
    OpponentPolicy.GRIM: GrimTrigger,
}


def build_opponent(policy: OpponentPolicy, key: EpisodeKey) -> Opponent:
    if policy is OpponentPolicy.QTABLE:
        return TabularQLearner(seed=purpose_rng(key, "qtable").getrandbits(63))
    if policy is OpponentPolicy.LLM:
        raise ValueError("LLM opponents are Phase 2 and are constructed by the runner")
    try:
        return _SCRIPTED[policy]()
    except KeyError as exc:
        raise ValueError(f"Unknown scripted opponent policy: {policy}") from exc


class Game:
    """Drives one episode. Terminates by horizon or by stochastic continuation."""

    def __init__(self, config: GameConfig, opponent: Opponent, key: EpisodeKey) -> None:
        self.config = config
        self.opponent = opponent
        self.key = key
        self.state = GameState(config=config)
        self._termination_rng = purpose_rng(key, "stochastic_termination")

    def should_continue(self) -> bool:
        """PURE QUERY. Must be safe to call any number of times per turn.

        It used to draw from the termination RNG on every call under STOCHASTIC
        mode, which made the answer depend on how often it was asked. Two
        callers ask per turn - step() at the end of a turn, and the batched
        runner when it rebuilds its live set at the start of the next - so
        every turn consumed TWO draws. The realised continuation probability
        was gamma^2 = 0.81 rather than the configured 0.90, cutting expected
        episode length from ~10 turns to ~5.3.

        Termination is now decided exactly once per turn, in step(), and
        recorded on the state. This reads that decision.
        """
        if self.state.terminated:
            return False
        return self.state.turn_index < self.config.horizon

    def _decide_termination(self) -> None:
        """Called ONCE per turn, from step(). The single point at which the
        stochastic continuation draw is consumed."""
        # Hard cap first, so an episode cannot run unboundedly and the draw is
        # not wasted on a turn that was ending regardless.
        if self.state.turn_index >= self.config.horizon:
            self.state.terminated = True
            return
        if self.config.horizon_mode is HorizonMode.STOCHASTIC:
            if self._termination_rng.random() >= self.config.continuation_probability:
                self.state.terminated = True

    def step(self, agent_action: Action) -> Turn:
        """Advance one turn. The opponent chooses without seeing the agent's
        current move, preserving simultaneity."""
        if self.state.terminated:
            raise RuntimeError("step() called on a terminated game")

        opponent_action = self.opponent.move(self.state)
        payoffs: Payoffs = self.config.payoffs
        turn = Turn(
            index=self.state.turn_index,
            agent_action=agent_action,
            opponent_action=opponent_action,
            agent_payoff=payoffs.payoff(agent_action, opponent_action),
            opponent_payoff=payoffs.payoff(opponent_action, agent_action),
        )
        prev_state = GameState(config=self.config, turns=list(self.state.turns))
        self.state.turns.append(turn)

        observe = getattr(self.opponent, "observe", None)
        if callable(observe):
            observe(prev_state, opponent_action, turn.opponent_payoff, self.state)

        self._decide_termination()
        return turn


def replay(config: GameConfig, opponent: Opponent, actions: Sequence[Action]) -> GameState:
    """Replay a fixed action sequence. Used by tests and by the optimal-play
    verifier; never by the live runner."""
    state = GameState(config=config)
    for action in actions:
        opponent_action = opponent.move(state)
        state.turns.append(
            Turn(
                index=state.turn_index,
                agent_action=action,
                opponent_action=opponent_action,
                agent_payoff=config.payoffs.payoff(action, opponent_action),
                opponent_payoff=config.payoffs.payoff(opponent_action, action),
            )
        )
    return state