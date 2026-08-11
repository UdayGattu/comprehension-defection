"""Optimal play against scripted opponents, and regret.

Regret is the primary outcome measure. Defection rate saturates and is
opponent-dependent in sign; regret is continuous, ceiling-free, and correctly
treats Always-Defect as optimal against a pushover and terrible against a
retaliator.

The sign-flip prediction is derived here rather than asserted:

    against TFT/GRIM   optimal play is near-cooperative  -> comprehension should REDUCE defection
    against ALLC/ALLD  optimal play is full defection    -> comprehension should INCREASE defection

No prompt-formatting artifact can produce opposite signs conditioned on an
opponent the model was never told about. That is the point.

Note this is a single-agent finite-horizon MDP, not a two-player game: the
opponent is a fixed automaton. Backward induction to mutual defection applies
only when BOTH players are strategic, which is Phase 2, not Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .config import Action, GameConfig, OpponentPolicy, Payoffs

_ACTIONS = (Action.COOPERATE, Action.DEFECT)


def _scripted_move(
    policy: OpponentPolicy,
    turn: int,
    agent_last: Action | None,
    agent_defected_ever: bool,
) -> Action:
    """Pure reimplementation of the scripted policies for use inside the DP.

    Deliberately duplicated from game.py rather than shared: tests assert the
    two agree, which catches drift in either. A single shared implementation
    would make that check vacuous.
    """
    if policy is OpponentPolicy.TFT:
        return Action.COOPERATE if turn == 0 else (agent_last or Action.COOPERATE)
    if policy is OpponentPolicy.ALLC:
        return Action.COOPERATE
    if policy is OpponentPolicy.ALLD:
        return Action.DEFECT
    if policy is OpponentPolicy.GRIM:
        return Action.DEFECT if agent_defected_ever else Action.COOPERATE
    raise ValueError(f"No closed-form optimal play defined for {policy}")


@dataclass(frozen=True)
class OptimalPlay:
    policy: OpponentPolicy
    value: int
    sequence: tuple[Action, ...]

    @property
    def defection_rate(self) -> float:
        return sum(a is Action.DEFECT for a in self.sequence) / len(self.sequence)


def solve(policy: OpponentPolicy, config: GameConfig) -> OptimalPlay:
    """Exact backward-induction solution over the finite horizon.

    Raises for stochastic horizons and for the Q-learner, both of which have no
    well-defined deterministic optimum. Callers must not silently substitute an
    approximation - a wrong optimum corrupts every regret figure downstream.
    """
    if policy is OpponentPolicy.QTABLE:
        raise ValueError("Q-learner is adaptive; optimal play is not well defined")
    if policy is OpponentPolicy.LLM:
        raise ValueError("LLM opponents have no closed-form optimum")

    payoffs: Payoffs = config.payoffs
    horizon = config.horizon

    @lru_cache(maxsize=None)
    def best(turn: int, agent_last: Action | None, defected_ever: bool) -> tuple[int, tuple[Action, ...]]:
        if turn == horizon:
            return 0, ()
        candidates = []
        for action in _ACTIONS:
            opp = _scripted_move(policy, turn, agent_last, defected_ever)
            immediate = payoffs.payoff(action, opp)
            future_value, future_seq = best(
                turn + 1, action, defected_ever or action is Action.DEFECT
            )
            candidates.append((immediate + future_value, (action,) + future_seq))
        # max() on tuples breaks ties by comparing sequences; sort explicitly on
        # value only, then prefer cooperation, so ties are resolved
        # deterministically rather than by Action enum ordering.
        best_value = max(v for v, _ in candidates)
        for value, seq in candidates:
            if value == best_value:
                return value, seq
        raise AssertionError("unreachable")

    value, sequence = best(0, None, False)
    return OptimalPlay(policy=policy, value=value, sequence=sequence)


def realised_value(
    policy: OpponentPolicy, config: GameConfig, actions: Sequence[Action]
) -> int:
    total = 0
    agent_last: Action | None = None
    defected_ever = False
    for turn, action in enumerate(actions):
        opp = _scripted_move(policy, turn, agent_last, defected_ever)
        total += config.payoffs.payoff(action, opp)
        defected_ever = defected_ever or action is Action.DEFECT
        agent_last = action
    return total


def episode_regret(
    policy: OpponentPolicy, config: GameConfig, actions: Sequence[Action]
) -> int:
    """Total regret over the episode: optimal value minus realised value.

    Always non-negative. Zero means the agent played an optimal line, whatever
    that line happens to be for this opponent.
    """
    regret = solve(policy, config).value - realised_value(policy, config, actions)
    if regret < 0:
        raise AssertionError(
            f"Negative regret for {policy}: realised value exceeded the computed "
            f"optimum, which means solve() is wrong. Refusing to continue."
        )
    return regret


def predicted_defection_direction(policy: OpponentPolicy, config: GameConfig) -> str:
    """Sign-flip prediction for the treatment arm against this opponent.

    Returns 'down' if optimal play defects less than the observed baseline
    tendency, 'up' otherwise. Used to pre-register the direction per cell so the
    analysis cannot be reverse-engineered after seeing the data.
    """
    optimal = solve(policy, config)
    return "down" if optimal.defection_rate < 0.5 else "up"


def summary_table(config: GameConfig) -> list[dict[str, object]]:
    """Regression-tested table. Values for T=5,R=3,P=1,S=0 horizon 20:

        TFT  optimal 62, ALLD 24, regret 38
        GRIM optimal 62, ALLD 24, regret 38
        ALLC optimal 100, ALLD 100, regret 0
        ALLD optimal 20, ALLD 20, regret 0
    """
    rows: list[dict[str, object]] = []
    always_defect = [Action.DEFECT] * config.horizon
    always_cooperate = [Action.COOPERATE] * config.horizon
    for policy in (OpponentPolicy.TFT, OpponentPolicy.GRIM, OpponentPolicy.ALLC, OpponentPolicy.ALLD):
        opt = solve(policy, config)
        rows.append(
            {
                "opponent": policy.value,
                "optimal": opt.value,
                "alld": realised_value(policy, config, always_defect),
                "allc": realised_value(policy, config, always_cooperate),
                "regret_of_alld": episode_regret(policy, config, always_defect),
                "optimal_defection_rate": round(opt.defection_rate, 3),
                "predicted_direction": predicted_defection_direction(policy, config),
            }
        )
    return rows
